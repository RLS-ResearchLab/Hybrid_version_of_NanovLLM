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


ARCH_DISPATCH = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen35ForCausalLM": Qwen35ForCausalLM,
    "Qwen35MoEForCausalLM": Qwen35ForCausalLM,
    "Qwen3_5MoeForConditionalGeneration": Qwen35ForCausalLM,
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

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        hf_config.dtype = _resolve_dtype(hf_config.dtype)
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.state_manager = None

        # enforce_eager elsewhere in this file only gates CUDA graph
        # capture/replay -- it does nothing about the @torch.compile
        # decorators in layers/{layernorm,sampler,activation,rotary_embedding}.py,
        # which are applied at import time independent of any engine config.
        # Disabling dynamo globally is the only way enforce_eager=True actually
        # guarantees zero compilation anywhere in the forward path. Set
        # unconditionally (not just when True) so a later ModelRunner built
        # with enforce_eager=False in the same process re-enables compilation
        # instead of staying silently stuck off.
        torch._dynamo.config.disable = self.enforce_eager

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        arch = getattr(hf_config, "architectures", ["Qwen3ForCausalLM"])[0]
        model_cls = ARCH_DISPATCH.get(arch, Qwen3ForCausalLM)
        self._is_hybrid_model = (model_cls is Qwen35ForCausalLM)

        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        if self._is_hybrid_model:
            num_linear_layers = sum(1 for t in self.model.model.layer_types if t == "linear_attention")
            la0 = next(m for m in self.model.modules() if isinstance(m, Qwen35LinearAttention))
            self.state_manager = StateManager(
                max_num_seqs=config.max_num_seqs,
                num_linear_layers=num_linear_layers,
                lvh=la0.lvh, lhd=la0.lhd, qkv_dim=la0.qkv_dim,
                conv_kernel_size=la0.ck,
                dtype=hf_config.dtype,
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
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
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

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
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
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

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
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
 
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
                gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                gv["state_slot_ids"][:bs] = context.state_slot_ids
                # cu_seqlens_q for decode is always arange(0, bs+1) — already
                # correct as a static buffer, no per-call refresh needed.
 
                graph.replay()
 
                hidden = gv["outputs"][:bs]
                num_layers = len(self.model.model.layers)
                linear_idx = self.model.model.linear_layer_indices
                new_states = [None] * num_layers
                new_conv_states = [None] * num_layers
                for compact_idx, full_idx in enumerate(linear_idx):
                    new_states[full_idx] = gv["state_out_bufs"][compact_idx][:bs]
                    new_conv_states[full_idx] = gv["conv_out_bufs"][compact_idx][:bs]
                self.state_manager.set_all(context.state_slot_ids, new_states, new_conv_states, linear_idx)
                logits = self.model.compute_logits(hidden)
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
 
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

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
    def get_prefill_layer_states(self, seqs: list[Sequence]) -> tuple[list[torch.Tensor], torch.Tensor] | None:
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
        split. Also computes final-norm + lm_head logits the same way
        get_prefill_logits does, as one more checkpoint past the last
        decoder layer. Returns (layer_states, logits), or None on non-rank-0.
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
        for i, layer in enumerate(model.layers):
            hidden_states, residual, ns, nc = layer(
                positions, hidden_states, residual, context.cu_seqlens_q,
                state=states[i], conv_state=conv_states[i],
            )
            captured.append((hidden_states + residual).float().cpu())
        final_hidden, _ = model.norm(hidden_states, residual)
        logits = self.model.compute_logits(final_hidden)
        reset_context()
        if self.rank != 0:
            return None
        return captured, logits.float().cpu()

    @torch.inference_mode()
    def capture_cudagraph(self):
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
            state_slot_ids = torch.zeros(max_bs, dtype=torch.int64)
            graph_vars["cu_seqlens_q"] = cu_seqlens_q
            graph_vars["state_slot_ids"] = state_slot_ids
 
            num_layers = len(self.model.model.layers)
            linear_idx = self.model.model.linear_layer_indices
            sm = self.state_manager
 
            # Static output buffers for the recurrent state — one per
            # linear-attention layer, sized like StateManager's own
            # per-layer slices. These live outside StateManager (StateManager
            # itself gets written to AFTER replay, via set_all, same as
            # compute_logits happens after replay).
            state_out_bufs = [
                torch.zeros(max_bs, sm.states.shape[2], sm.states.shape[3], sm.states.shape[4],
                            dtype=torch.float32)
                for _ in range(sm.num_linear_layers)
            ]
            conv_out_bufs = [
                torch.zeros(max_bs, sm.conv_states.shape[2], sm.conv_states.shape[3],
                            dtype=sm.conv_states.dtype)
                for _ in range(sm.num_linear_layers)
            ]
            graph_vars["state_out_bufs"] = state_out_bufs
            graph_vars["conv_out_bufs"] = conv_out_bufs
 
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
                        state_out_bufs[compact_idx][:bs] = new_states[full_idx]
                        conv_out_bufs[compact_idx][:bs] = new_conv_states[full_idx]
 
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