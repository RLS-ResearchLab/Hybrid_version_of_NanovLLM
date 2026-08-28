import os
import sys
import types
import pickle
import torch
import torch._dynamo
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

if __package__ in (None, "") and "nanovllm" not in sys.modules:
    package_root = os.path.dirname(os.path.dirname(__file__))
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [package_root]
    nanovllm_pkg.__file__ = os.path.join(package_root, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.models.qwen3_5 import Qwen35ForCausalLM
from nanovllm.engine.state_manager import StateManager
from nanovllm.models.qwen3_5 import Qwen35LinearAttention
from nanovllm.layers.linear import local_num_kv_heads


ARCH_DISPATCH = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen35ForCausalLM": Qwen35ForCausalLM,
    "Qwen35MoEForCausalLM": Qwen35ForCausalLM,
    "Qwen3_5MoeForConditionalGeneration": Qwen35ForCausalLM,
    "Qwen3_5MoeForCausalLM": Qwen35ForCausalLM,
}


def _resolve_dtype(dtype) -> torch.dtype:
    """hf_config.dtype comes straight off config.json and may be a plain
    string (e.g. "bfloat16") rather than an actual torch.dtype -- real VLM
    checkpoints serialize it that way. Resolve generically via getattr(torch,
    name) so any dtype string transformers emits works, not just bfloat16."""
    if isinstance(dtype, torch.dtype):
        return dtype
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unrecognized torch dtype string: {dtype!r}")
    return resolved


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], ack_event: Event | list[Event] | None = None):
        self.config = config
        hf_config = config.hf_config
        hf_config.dtype = _resolve_dtype(hf_config.dtype)
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.ack_event = ack_event
        self.state_manager = None
        torch._dynamo.config.disable = self.enforce_eager
        # Found 2026-08-27: PyTorch's default recompile_limit (8) exactly
        # matches self.graph_bs's bucket count ([1,2,4,8,16,32,48,64], 8
        # buckets) -- but layers/layernorm.py's Qwen35RMSNorm.rms_forward
        # (@torch.compile) is called on q_norm AND k_norm, which have
        # DIFFERENT shapes at the same batch size under GQA (num_heads !=
        # num_kv_heads), so capturing all 8 decode buckets needs 16 distinct
        # compiled shape-variants of the same function against an 8-slot
        # budget. Once exhausted, torch._dynamo silently falls back to slow
        # uncompiled eager execution for whichever buckets get captured
        # last -- confirmed via a real "hit config.recompile_limit (8)"
        # warning during capture, not a guess. Since CUDA graph capture
        # freezes whatever actually ran at capture time, an eager fallback
        # here would be permanent for the life of the server, on every
        # layer, every decode step. Set generously above the ~16+ shapes
        # this model's own capture loop needs (decode buckets x q/k norm
        # variants, plus prefill's own varying shapes, plus whatever margin
        # other @torch.compile call sites in this codebase need) rather than
        # tuned to the exact minimum -- cheap safety margin, no downside
        # besides a marginally larger compile cache.
        #
        # 2026-08-28 (GPU window #2): on torch 2.13+cu130 the single
        # `recompile_limit = 64` assignment was a NO-OP -- every server log
        # still showed "torch._dynamo hit config.recompile_limit (8)". torch
        # 2.13 also carries a separate `cache_size_limit` (the pre-2.5 name,
        # now de-aliased), and the warning path checks a limit that plain
        # assignment to one of the two names does not raise. Belt-and-
        # suspenders: set both names + the accumulated cap, all guarded so a
        # future rename can't crash __init__. The [DYNAMO] print is a
        # deliberate startup breadcrumb -- grep it against the recompile
        # warning to confirm the limit actually took on whatever torch ships.
        for _n in ("recompile_limit", "cache_size_limit"):
            try:
                setattr(torch._dynamo.config, _n, 64)
            except Exception:
                pass
        try:
            torch._dynamo.config.accumulated_recompile_limit = max(
                512, getattr(torch._dynamo.config, "accumulated_recompile_limit", 512))
        except Exception:
            pass
        print("[DYNAMO] recompile_limit=%s cache_size_limit=%s accumulated=%s" % (
            getattr(torch._dynamo.config, "recompile_limit", "?"),
            getattr(torch._dynamo.config, "cache_size_limit", "?"),
            getattr(torch._dynamo.config, "accumulated_recompile_limit", "?")), flush=True)

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        arch_list = getattr(hf_config, "architectures", None)
        if arch_list is None:
            arch, model_cls = "Qwen3ForCausalLM", Qwen3ForCausalLM
        else:
            arch = arch_list[0]
            if arch not in ARCH_DISPATCH:
                raise ValueError(
                    f"Unrecognized architectures[0]={arch!r} in this checkpoint's config -- "
                    f"ARCH_DISPATCH only knows {sorted(ARCH_DISPATCH)}. Add {arch!r} to "
                    f"ARCH_DISPATCH (pointing at Qwen3ForCausalLM or Qwen35ForCausalLM, "
                    f"whichever this checkpoint actually is) rather than letting this fall "
                    f"through silently -- see the comment above for the incident this guards against."
                )
            model_cls = ARCH_DISPATCH[arch]
        self._is_hybrid_model = (model_cls is Qwen35ForCausalLM)

        self.model = model_cls(hf_config)
        load_model(self.model, config.model, strict=True)
        if getattr(config, "use_moe_w8a8", False):
            # moe_int8_integration.py lives under tests/, not on any production
            # sys.path -- the models/qwen3_5.py import above happens to already put
            # tests/ on sys.path as a side effect (see that file's own comment), which
            # is why this worked in every real invocation today, but that's an
            # implicit dependency on import order, not a guarantee. Guard explicitly.
            _tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
            if _tests_dir not in sys.path:
                sys.path.insert(0, _tests_dir)
            from moe_int8_integration import apply_moe_int8_quantization
            n_quantized = apply_moe_int8_quantization(
                self.model, config.moe_w8a8_weight_group_size
            )
            torch.cuda.empty_cache()
            print(f"[MOE W8A8] quantized {n_quantized} Experts module(s) to "
                  f"INT8, group_size={config.moe_w8a8_weight_group_size}")
        if getattr(config, "use_moe_w8a8_hopper", False):
            # True W8A8 Hopper path -- separate scheme from use_moe_w8a8 above,
            # additive (distinct gate_up_proj_fp8/... buffer names), see
            # config.py's own comment on the two-flag split. UNVALIDATED --
            # do not set this True before Phase 0 (moe_w8a8.cu compile +
            # isolated smoke test) has run on real Hopper hardware; see
            # w8a8_activation_quant_scoping_memo.md.
            _tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
            if _tests_dir not in sys.path:
                sys.path.insert(0, _tests_dir)
            from moe_w8a8_hopper_integration import apply_moe_w8a8_hopper_quantization
            n_quantized_fp8 = apply_moe_w8a8_hopper_quantization(
                self.model, config.moe_w8a8_hopper_weight_group_size
            )
            torch.cuda.empty_cache()
            print(f"[MOE W8A8 HOPPER] quantized {n_quantized_fp8} Experts module(s) to "
                  f"FP8, group_size={config.moe_w8a8_hopper_weight_group_size}")
        if getattr(config, "use_lm_head_int8", False):
            # lm_head INT8 -- CAPACITY win, NOT a confirmed throughput win.
            # See tests/lm_head_int8_integration.py's module docstring before
            # assuming this speeds anything up.
            _tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
            if _tests_dir not in sys.path:
                sys.path.insert(0, _tests_dir)
            from lm_head_int8_integration import apply_lm_head_int8_quantization
            n_quantized_lm_head = apply_lm_head_int8_quantization(
                self.model, config.lm_head_int8_group_size
            )
            torch.cuda.empty_cache()
            print(f"[LM_HEAD INT8] quantized {n_quantized_lm_head} lm_head module(s) to "
                  f"INT8, group_size={config.lm_head_int8_group_size}")
        if self._is_hybrid_model:
            num_linear_layers = sum(1 for t in self.model.model.layer_types if t == "linear_attention")
            la0 = next(m for m in self.model.modules() if isinstance(m, Qwen35LinearAttention))
            self.state_manager = StateManager(
                max_num_seqs=config.max_num_seqs,
                num_linear_layers=num_linear_layers,
                lvh=la0.lvh, lhd=la0.lhd, qkv_dim=la0.qkv_dim,
                conv_kernel_size=la0.ck,
                dtype=hf_config.dtype,
                check_slot_zeroed=getattr(config, "debug_check_state_slot_zeroed", False),
            )
            torch.cuda.synchronize()
            print(f"[MEM DEBUG] after StateManager construction: "
                  f"current={torch.cuda.memory_stats()['allocated_bytes.all.current']}, "
                  f"state_manager.memory_bytes()={self.state_manager.memory_bytes()}")

            assert self.state_manager is not None, (
                "StateManager must be constructed before warmup_model() runs, "
                "so the peak-memory probe used by allocate_kv_cache() reflects "
                "real state-cache usage, not just KV-cache usage."
            )
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # A pre-existing "nanovllm" segment at this point can only be
                # a stale leftover from a previous run that crashed/was killed
                # before exit() could unlink() it (this engine only ever runs
                # one instance under this fixed name at a time) -- never a
                # live peer of THIS process group, which is still mid-startup.
                # Clear it proactively so a crashed prior run doesn't force a
                # full reload+warmup just to fail again at this exact line.
                try:
                    stale = SharedMemory(name="nanovllm")
                    stale.close()
                    stale.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        # Only after the payload is fully copied out into local Python
        # objects (pickle.loads above) is it safe for rank0 to overwrite
        # the shared buffer -- signal that now, not after this call's
        # method actually finishes executing (NCCL collectives inside the
        # method already provide their own cross-rank synchronization;
        # this ack is purely about the raw shared-memory bytes).
        self.ack_event.set()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # Wait for every other rank to have finished reading the PREVIOUS
        # message before overwriting the buffer with this one. Without this
        # handshake, rank0 can race ahead (nothing throttles it for cheap,
        # non-collective calls like allocate_state_slot) and overwrite the
        # shared buffer while a rank>0 process is still mid-read, corrupting
        # its pickle stream (observed as `pickle data was truncated`) or
        # desyncing the two ranks onto different method calls entirely,
        # which then deadlocks forever the moment either side hits a
        # collective op the other isn't also calling.
        for ack in self.ack_event:
            ack.wait()
            ack.clear()
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for i,seq in enumerate(seqs):
            seq.num_scheduled_tokens = seq_len
            if self.state_manager is not None:
                seq.state_slot = i
        if self.state_manager is not None:
            assert num_seqs > 0, "warmup_model built zero sequences — state-cache path not exercised"
        self.run(seqs, True)
        torch.cuda.empty_cache()
        if self.state_manager is not None:
            assert len(self.state_manager.free_slot_ids) == self.state_manager.max_num_seqs, (
                f"warmup left {self.state_manager.max_num_seqs - len(self.state_manager.free_slot_ids)} "
                f"slot(s) marked used after warmup — state_slot bookkeeping bug"
            )

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        print(f"[MEM DEBUG] allocate_kv_cache entry: current={current}, peak={peak}")

        # Calculate StateManager footprint
        state_bytes = self.state_manager.memory_bytes() if self.state_manager is not None else 0

        # local_num_kv_heads (nanovllm.layers.linear) is the single source of
        # truth for this computation, shared with Qwen35FullAttention's own
        # per-rank kv-head sizing (models/qwen3_5.py) -- computing it
        # independently here used to be `num_key_value_heads // world_size`,
        # which silently returns 0 (a zero-sized KV-cache dimension, no
        # assert to catch it) whenever world_size > num_key_value_heads. The
        # dense model path (models/qwen3.py) has its own separate, unmodified
        # `assert total_num_kv_heads % tp_size == 0` in Qwen3Attention.__init__
        # -- which runs before this method, during self.model = model_cls(...)
        # -- so an impossible dense-model tp_size still fails loudly there,
        # before this formula is ever reached; only the hybrid model's
        # previously-unreachable-without-a-crash replication case changes here.
        num_kv_heads = local_num_kv_heads(hf_config.num_key_value_heads, self.world_size)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        if self._is_hybrid_model:
            num_kv_layers = sum(1 for t in self.model.model.layer_types if t == "full_attention")
        else:
            num_kv_layers = hf_config.num_hidden_layers
        old_layers = hf_config.num_hidden_layers

        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current ) // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        print(f"[KV DEBUG] num_hidden_layers={old_layers}  num_kv_layers(full-attn only)={num_kv_layers}")
        print(f"[KV DEBUG] block_bytes (per block, per layer count of {num_kv_layers})={block_bytes}")
        print(f"[KV DEBUG] state_bytes subtracted={state_bytes}")
        print(f"[KV DEBUG] num_kvcache_blocks={config.num_kvcache_blocks}")
        print(f"[KV DEBUG] kv_cache tensor shape={tuple(self.kv_cache.shape)}")
        print(f"[KV DEBUG] total kv_cache bytes = {self.kv_cache.numel() * self.kv_cache.element_size()}")

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_kv_layers, f"expected {num_kv_layers} attention modules with k/v cache, wired {layer_id}"

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_slot_ids = None
        if self.state_manager is not None:
            state_slot_ids = torch.tensor(
                [seq.state_slot for seq in seqs], dtype=torch.int64, pin_memory=True
                ).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,state_slot_ids)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        cu_seqlens_q = None
        state_slot_ids = None
        if self.state_manager is not None:
            n = len(seqs)
            cu_seqlens_q = torch.arange(0, n + 1, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            state_slot_ids = torch.tensor(
                [seq.state_slot for seq in seqs], dtype=torch.int64, pin_memory=True
                ).cuda(non_blocking=True)
        set_context(False, cu_seqlens_q=cu_seqlens_q, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, state_slot_ids=state_slot_ids)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        # Computed on the host, from data already sitting in plain Python,
        # BEFORE the GPU tensor below even exists -- passed through to
        # Sampler.forward() so its hot-path decision doesn't need to
        # re-derive the same fact via a GPU->host sync
        # (`bool((temperatures == 0).all())`) every decode step. See
        # Sampler.forward()'s docstring for why that sync was a real,
        # measured-variance cost, not just theoretical overhead.
        all_greedy = all(t == 0 for t in temperatures)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, all_greedy

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures, all_greedy = self.prepare_sample(seqs) if self.rank == 0 else (None, None)
 
        if self.state_manager is not None:
            context = get_context()
            use_graph = (not is_prefill and not self.enforce_eager
                         and hasattr(self, "graphs") and input_ids.size(0) <= 512)
            if use_graph:
                bs = input_ids.size(0)
                graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
                gv = self.graph_vars
                gv["input_ids"][:bs] = input_ids
                gv["positions"][:bs] = positions
                gv["slot_mapping"].fill_(-1)
                gv["slot_mapping"][:bs] = context.slot_mapping
                gv["context_lens"].zero_()
                gv["context_lens"][:bs] = context.context_lens
                # Reset before the partial overwrite, matching slot_mapping/
                # state_slot_ids above -- without this, padding rows
                # (bs:max_bs) keep stale block ids from whatever larger-bs
                # call last wrote this static buffer, potentially pointing
                # at blocks freed and reassigned to an unrelated sequence
                # since. context_lens=0 for these rows should already make
                # flash_attn_with_kvcache skip reading block_table for them
                # entirely (same paired zero/-1 gating slot_mapping/
                # state_slot_ids rely on) -- this is defense in depth, not
                # a behavior change, in case that assumption is ever wrong.
                gv["block_tables"].fill_(-1)
                gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                # Padding rows (this graph's captured size may exceed bs)
                # must resolve to the reserved scratch slot, not leftover
                # real slot ids from a previous, larger-bs call -- see
                # capture_cudagraph()'s _step() for why this matters.
                gv["state_slot_ids"].fill_(self.state_manager.scratch_slot_id)
                gv["state_slot_ids"][:bs] = context.state_slot_ids
                # cu_seqlens_q for decode is always arange(0, bs+1) — already
                # correct as a static buffer, no per-call refresh needed.

                # nvtx ranges added 2026-08-28 (CPU window, no GPU access) --
                # purely diagnostic, torch.cuda.nvtx push/pop cost is
                # negligible whether or not a profiler is attached. Exists
                # to settle, on the next nsys capture, whether the
                # unidentified ~2ms/step elementwise op (SESSION_HANDOFF_
                # 2026-08-28.md) lives in the eager lm_head GEMM or the
                # eager sampler call below -- both run OUTSIDE the captured
                # graph today (capture_cudagraph()'s _step() only covers
                # self.model.model(...), the 40-layer backbone), so both are
                # real candidates for host-dispatch-induced GPU idle bubbles
                # that a graph replay's kernels don't suffer from.
                torch.cuda.nvtx.range_push("decode_graph_replay")
                graph.replay()
                torch.cuda.nvtx.range_pop()

                hidden = gv["outputs"][:bs]
                # State write-back happens INSIDE the captured graph now
                # (see capture_cudagraph()'s _step()) -- no eager
                # index_copy_ needed here anymore.
                torch.cuda.nvtx.range_push("decode_lm_head_eager")
                logits = self.model.compute_logits(hidden)
                torch.cuda.nvtx.range_pop()
            else:
                num_layers = len(self.model.model.layers)
                linear_idx = self.model.model.linear_layer_indices
                states, conv_states = self.state_manager.get_all(context.state_slot_ids, num_layers, linear_idx)
                logits, new_states, new_conv_states = self.model(
                    input_ids, positions, context.cu_seqlens_q, states, conv_states
                )
                self.state_manager.set_all(context.state_slot_ids, new_states, new_conv_states, linear_idx)
                logits = self.model.compute_logits(logits)
        else:
            logits = self.run_model(input_ids, positions, is_prefill)
 
        # Not "decode_"-prefixed like the two ranges above -- this call site
        # also fires for prefill (is_prefill=True takes the eager branch
        # above, not the graph-replay one), so the label stays accurate for
        # both.
        torch.cuda.nvtx.range_push("sampler_eager")
        token_ids = self.sampler(logits, temperatures, all_greedy=all_greedy).tolist() if self.rank == 0 else None
        torch.cuda.nvtx.range_pop()
        if is_prefill and self.rank == 0 and self.config.debug_print_prefill_samples:
            argmax_ids = logits.argmax(dim=-1).tolist()
            for seq, argmax_id, token_id in zip(seqs, argmax_ids, token_ids):
                print(f"[PREFILL-SAMPLE DEBUG] seq_id={seq.seq_id} argmax={argmax_id} sampled={token_id} "
                      f"num_scheduled_tokens={seq.num_scheduled_tokens} num_cached_tokens={seq.num_cached_tokens} "
                      f"num_tokens={seq.num_tokens}")
        reset_context()
        return token_ids

    def allocate_state_slot(self, seq: Sequence) -> None:
        """Called via .call(...) from Scheduler.schedule() (engine/scheduler.py)
        for every real request, not just diagnostics -- StateManager.allocate()
        must run on EVERY rank's own StateManager instance, not just rank0's --
        each rank at tensor_parallel_size>1 is a separate process with its own
        independent state/conv_state buffers (each ran its own warmup_model(),
        which leaves genuine non-zero residue in whatever slot it used, never
        re-zeroed afterward -- see reference_check_phase6.py's phase_engine()
        for the full writeup). Calling state_manager.allocate(seq) directly on
        only one rank's ModelRunner object -- which is what Scheduler used to
        do, calling straight into its own local state_manager reference with
        no cross-rank dispatch at all -- only zeros THAT rank's copy, leaving
        every other rank reading whatever the slot's previous tenant (often
        still-warm warmup residue) left behind, with no crash or NaN, just
        silently wrong recurrent state from the very first forward pass.
        Dispatching through .call(...) (pickles+sends seq to every other rank
        via the same shared-memory mechanism run()/get_prefill_logits() use)
        runs allocate() on all of them, so every rank's slot for this seq
        starts genuinely zeroed."""
        if self.state_manager is not None:
            self.state_manager.allocate(seq)

    def free_state_slot(self, seq: Sequence) -> None:
        """Called via .call(...) from Scheduler.postprocess()/preempt() --
        the free-side counterpart of allocate_state_slot(), same reason:
        StateManager.free() must zero every rank's own copy of this seq's
        slot, not just rank0's, so the NEXT sequence to receive this slot
        (via allocate_state_slot(), possibly on a different rank first)
        doesn't inherit this one's leftover recurrent/conv state anywhere."""
        if self.state_manager is not None:
            self.state_manager.free(seq)

    @torch.inference_mode()
    def get_prefill_logits(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """Diagnostic-only, additive -- does not change run()'s behavior at
        all. Mirrors run()'s prefill (non-graph) branch exactly, but returns
        raw logits at each seq's final position (already sliced by
        ParallelLMHead.forward()'s context.is_prefill branch, same as the
        logits run() would have sampled from) instead of sampled token ids.
        Never calls self.sampler. Returns None on non-rank-0, matching
        ParallelLMHead's gather-onto-rank-0-only behavior."""
        input_ids, positions = self.prepare_prefill(seqs)
        if self.state_manager is not None:
            context = get_context()
            num_layers = len(self.model.model.layers)
            linear_idx = self.model.model.linear_layer_indices
            states, conv_states = self.state_manager.get_all(context.state_slot_ids, num_layers, linear_idx)
            logits, new_states, new_conv_states = self.model(
                input_ids, positions, context.cu_seqlens_q, states, conv_states
            )
            self.state_manager.set_all(context.state_slot_ids, new_states, new_conv_states, linear_idx)
            logits = self.model.compute_logits(logits)
        else:
            logits = self.run_model(input_ids, positions, True)
        reset_context()
        return logits.cpu() if logits is not None else None

    @torch.inference_mode()
    def sample_from_logits(self, logits: torch.Tensor, temperatures: torch.Tensor) -> list[int] | None:
        """Diagnostic-only, additive -- does not change run()'s behavior at
        all. Calls self.sampler directly on an ALREADY-COMPUTED logits
        tensor (e.g. captured via get_prefill_logits), the exact same call
        run() makes (`self.sampler(logits, temperatures)`), to isolate
        whether the (@torch.compile-decorated) sampler itself introduces
        any batch-shape-dependent behavior for a FIXED row's greedy
        argmax, independent of whether the upstream logits computation
        itself is shape-dependent (already checked separately via
        get_prefill_logits). Only meaningful on rank0 -- other ranks never
        run the sampler (ParallelLMHead gathers logits onto rank0 only) --
        returns None elsewhere, matching every other rank0-only diagnostic
        in this file."""
        if self.rank != 0:
            return None
        return self.sampler(logits.cuda(), temperatures.cuda()).tolist()

    @torch.inference_mode()
    def get_prefill_layer_states(self, seqs: list[Sequence]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor] | None:
        """Diagnostic-only, additive -- does not change run()'s or
        get_prefill_logits()'s behavior at all. Mirrors get_prefill_logits's
        prefill path exactly, but captures the residual-stream hidden state
        after the embedding layer and after each decoder layer, for
        layer-by-layer comparison against HF's `output_hidden_states=True`
        tuple (index 0 = embedding output, index i+1 = output of layer i).
        "hidden_states + residual" is the same fused-add value
        Qwen35RMSNorm computes internally for the next layer's input, i.e.
        the actual residual-stream value at that point, not the
        pre-add/pre-norm intermediate qwen3_5.py's layer forward returns
        split.

        A per-decoder-layer checkpoint conflates two very different
        sublayers (self_attn/linear_attn, then a full MoE FFN pass) --
        inlines Qwen35DecoderLayer.forward()'s steps here (instead of
        calling layer(...) as one opaque call) to ALSO capture each layer's
        raw attention/linear-attn sublayer output on its own, before
        post_attention_layernorm/mlp run, so a HF-side forward hook on the
        same submodule can isolate "attention sublayer wrong" from "MoE FFN
        wrong" instead of only seeing their combined effect.

        Also captures each layer's post-input_layernorm, pre-attention
        tensor (post_ln) -- the exact input fed into self_attn/linear_attn,
        so it can be compared 1:1 against a HF-side hook at the same point.
        Isolates "input_layernorm/RMSNorm already wrong" from "attention
        itself wrong", since the embedding checkpoint above is captured
        BEFORE input_layernorm and can't distinguish those on its own.

        Also computes final-norm + lm_head logits the same way
        get_prefill_logits does, as one more checkpoint past the last
        decoder layer. Returns (layer_states, attn_only_states, post_ln,
        logits), or None on non-rank-0.
        """
        input_ids, positions = self.prepare_prefill(seqs)
        context = get_context()
        model = self.model.model    # Qwen35Model
        num_layers = len(model.layers)
        states = [None] * num_layers
        conv_states = [None] * num_layers
        if self.state_manager is not None:
            states, conv_states = self.state_manager.get_all(
                context.state_slot_ids, num_layers, model.linear_layer_indices
            )
        hidden_states = model.embed_tokens(input_ids)
        residual = None
        captured = [hidden_states.float().cpu()]
        attn_only = []
        # post-input_layernorm, pre-attention -- the exact tensor fed INTO
        # self_attn/linear_attn, so it lines up 1:1 with HF's mixer_io[i]["input"]
        # (captured via a forward hook on the SAME point in HF's own
        # DecoderLayer.forward). Isolates "input_layernorm/RMSNorm is
        # already wrong before attention runs" from "attention itself is
        # wrong" -- the "embedding" checkpoint above is BEFORE
        # input_layernorm, so it can't answer that on its own.
        post_ln = []
        for i, layer in enumerate(model.layers):
            if residual is None:
                hs_in, res_in = layer.input_layernorm(hidden_states), hidden_states
            else:
                hs_in, res_in = layer.input_layernorm(hidden_states, residual)
            post_ln.append(hs_in.float().cpu())
            if layer.is_full:
                attn_out = layer.self_attn(positions, hs_in)
            else:
                attn_out, ns, nc = layer.linear_attn(
                    hs_in, context.cu_seqlens_q, states=states[i], conv_states=conv_states[i]
                )
            attn_only.append(attn_out.float().cpu())
            hs_mid, residual = layer.post_attention_layernorm(attn_out, res_in)
            hidden_states = layer.mlp(hs_mid, context.cu_seqlens_q)
            captured.append((hidden_states + residual).float().cpu())
        final_hidden, _ = model.norm(hidden_states, residual)
        logits = self.model.compute_logits(final_hidden)
        reset_context()
        if self.rank != 0:
            return None
        return captured, attn_only, post_ln, logits.float().cpu()

    @torch.inference_mode()
    def capture_cudagraph(self):
        # DECODE ONLY -- never called for prefill (see run_model()/run() above:
        # graph replay is only used when `not is_prefill`). Load-bearing, not
        # incidental: the fused-GDR kernel path (models/qwen3_5.py's
        # Qwen35LinearAttention._fla_chunk_gated_delta_rule) is NOT capturable
        # inside torch.cuda.graph() (reproduced as
        # cudaErrorStreamCaptureInvalidated) -- this only stays safe because
        # (a) that class's `use_fused` gate guarantees decode never takes the
        # fused branch, and (b) this method never captures prefill, the only
        # shape the fused branch is ever reached for. Do not extend graph
        # capture to prefill without re-checking that constraint first.
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
 
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
 
        graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
 
        if self._is_hybrid_model:
            # Decode: every sequence contributes exactly 1 token, so
            # cu_seqlens is always arange(0, bs+1) — same construction
            # prepare_decode() already uses at eager time.
            cu_seqlens_q = torch.arange(0, max_bs + 1, dtype=torch.int32)
            # Default-filled with the reserved scratch slot, not zeros --
            # rows beyond the real batch size (this graph's capture size
            # may exceed it) must never resolve to a real, in-use slot.
            # run() re-fills this the same way before every replay; see
            # StateManager.scratch_slot_id's docstring for why.
            state_slot_ids = torch.full((max_bs,), self.state_manager.scratch_slot_id, dtype=torch.int64)
            graph_vars["cu_seqlens_q"] = cu_seqlens_q
            graph_vars["state_slot_ids"] = state_slot_ids
 
            num_layers = len(self.model.model.layers)
            linear_idx = self.model.model.linear_layer_indices
            sm = self.state_manager

            # Sanity check on the "fixed address" assumption this design
            # depends on — verify it, don't just assert it from reasoning.
            state_ptr_before = sm.states.data_ptr()
            conv_ptr_before = sm.conv_states.data_ptr()
 
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
 
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
 
            if self._is_hybrid_model:
                cu_seqlens_bs = graph_vars["cu_seqlens_q"][:bs + 1]
                slot_ids_bs = graph_vars["state_slot_ids"][:bs]
                set_context(False, cu_seqlens_q=cu_seqlens_bs,
                            slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                            block_tables=block_tables[:bs], state_slot_ids=slot_ids_bs)
 
                def _step():
                    states, conv_states = sm.get_all(slot_ids_bs, num_layers, linear_idx)
                    hidden, new_states, new_conv_states = self.model.model(
                        input_ids[:bs], positions[:bs], cu_seqlens_bs, states, conv_states
                    )
                    outputs[:bs] = hidden
                    for compact_idx, full_idx in enumerate(linear_idx):
                        # Write the new recurrent/conv state directly back
                        # into StateManager's real buffers, INSIDE the
                        # captured graph -- eliminates the 64 index_copy_
                        # calls/decode-step (32 linear layers x 2) that
                        # previously ran in eager Python after every
                        # replay (measured ~8.6% of total GPU time in a
                        # graph-mode nsys profile, the largest per-step
                        # cost found after the fused MoE kernel work).
                        # Safe against padding rows (this graph's bs may
                        # exceed the real request count) ONLY because
                        # run() fills state_slot_ids' padding region with
                        # StateManager.scratch_slot_id before every
                        # replay -- without that, a padding row's stale
                        # slot id could alias and silently corrupt a
                        # different, currently-in-use sequence's state.
                        sm.states[compact_idx].index_copy_(
                            0, slot_ids_bs, new_states[full_idx].to(sm.states.dtype))
                        sm.conv_states[compact_idx].index_copy_(
                            0, slot_ids_bs, new_conv_states[full_idx].to(sm.conv_states.dtype))
 
                _step()    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    _step()    # capture
            else:
                set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                            block_tables=block_tables[:bs])
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
 
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
 
        if self._is_hybrid_model:
            assert sm.states.data_ptr() == state_ptr_before, (
                "StateManager.states moved address during graph capture — "
                "the 'no static copy needed' assumption is FALSE, this design "
                "is unsafe as written and needs a static input buffer instead."
            )
            assert sm.conv_states.data_ptr() == conv_ptr_before, (
                "StateManager.conv_states moved address during graph capture — same issue."
            )
 
        self.graph_vars = graph_vars