"""One-off diagnostic -- does the bf16 Experts.gate_up_proj/down_proj
survive quantization as a lingering reference (leak hypothesis), and what
are the REAL decode-path gather-buffer sizes (gate_up_proj vs down_proj)?

Monkeypatches three points, all applied BEFORE engine construction so they
wrap rank0's in-process ModelRunner (LLMEngine builds rank0 directly in this
process; rank>0 is spawned via mp.get_context("spawn"), a fresh interpreter
that does not see these patches -- fine here since both ranks hold
symmetric shards under this TP scheme, so rank0 alone answers the question):

  1. moe_int8_integration.apply_moe_int8_quantization -- prints GPU memory
     + a scan of large (>1GB) CUDA tensors immediately AFTER quantization
     completes. If bf16-shaped (2*MI/H or H/MI) expert tensors show up
     here, the original bf16 weights survived the `del` -- a real leak.

  2. engine.model_runner.ModelRunner.capture_cudagraph -- prints the same
     two things immediately BEFORE graph capture starts (i.e. after
     warmup_model()/allocate_kv_cache() too). This is the "right before
     capture_cudagraph()" snapshot the OOM hypothesis is actually about.
     No-ops under --enforce-eager (never called).

  3. models.qwen3_5.Qwen35MoE._forward_gathered_ep -- prints the REAL
     gate_up_proj/down_proj gather sizes (numel * element_size) the first
     time decode's EP-gathered path runs, int8 or bf16 branch, whichever
     is actually hit.

Usage:
    # Safe baseline -- no graph capture, so no OOM risk from this diagnostic
    # itself. Matches the "clean" enforce_eager=True configuration the
    # 19.06GB figure was reportedly measured under.
    python tests/diag_moe_int8_memory.py --concurrency 16 --tp 2 --moe-w8a8 --enforce-eager

    # The actual OOM scenario -- graph capture enabled. If it OOMs, the
    # prints from patches 1 and 2 already happened (flush=True) before the
    # crash, so the leak-vs-no-leak question is answered either way.
    python tests/diag_moe_int8_memory.py --concurrency 16 --tp 2 --moe-w8a8
"""
import argparse
import gc
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import bench_throughput as bt  # noqa: E402


def _dump_cuda_memory(label: str) -> None:
    import torch
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    free, total = torch.cuda.mem_get_info()
    print(f"\n=== [MEM DUMP] {label} ===", flush=True)
    print(f"  allocated: {allocated:.2f} GB   reserved: {reserved:.2f} GB   "
          f"free: {free/2**30:.2f} GB / total: {total/2**30:.2f} GB", flush=True)
    big = [
        (tuple(o.shape), o.dtype, o.numel() * o.element_size() / 2**30)
        for o in gc.get_objects()
        if torch.is_tensor(o) and o.is_cuda and o.numel() * o.element_size() > 2**30 * 0.05
    ]
    big.sort(key=lambda x: -x[2])
    print(f"  {len(big)} CUDA tensor(s) > 50MB:", flush=True)
    for shape, dtype, gb in big[:25]:
        print(f"    {gb:7.3f} GB  {dtype}  {shape}", flush=True)
    print("=== [/MEM DUMP] ===\n", flush=True)


def _patch_quantization():
    import moe_int8_integration as m
    orig = m.apply_moe_int8_quantization

    def wrapped(model, group_size):
        n = orig(model, group_size)
        _dump_cuda_memory(f"immediately after apply_moe_int8_quantization "
                           f"(quantized {n} module(s), group_size={group_size})")
        return n

    m.apply_moe_int8_quantization = wrapped


def _patch_capture_cudagraph():
    from nanovllm.engine.model_runner import ModelRunner
    orig = ModelRunner.capture_cudagraph

    def wrapped(self):
        _dump_cuda_memory("immediately BEFORE capture_cudagraph() "
                           "(after warmup_model()/allocate_kv_cache())")
        return orig(self)

    ModelRunner.capture_cudagraph = wrapped


def _patch_gather_shapes():
    from nanovllm.models.qwen3_5 import Qwen35MoE
    orig = Qwen35MoE._forward_gathered_ep
    state = {"printed": False}

    def wrapped(self, x):
        result = orig(self, x)
        if not state["printed"]:
            state["printed"] = True
            if hasattr(self.experts, "gate_up_proj_int8"):
                gu = self.experts.gate_up_proj_int8
                dp = self.experts.down_proj_int8
                kind = "int8"
            else:
                gu = self.experts.gate_up_proj
                dp = self.experts.down_proj
                kind = str(gu.dtype)
            N, TK = x.shape[0], self.top_k
            gu_gather_bytes = N * TK * gu.shape[1] * gu.shape[2] * gu.element_size()
            dp_gather_bytes = N * TK * dp.shape[1] * dp.shape[2] * dp.element_size()
            print(f"\n=== [GATHER SIZE] _forward_gathered_ep, N={N} TK={TK}, "
                  f"expert weight dtype={kind} ===", flush=True)
            print(f"  gate_up_proj[local_slots] gather: shape=({N},{TK},{gu.shape[1]},{gu.shape[2]}) "
                  f"-> {gu_gather_bytes/2**20:.2f} MiB", flush=True)
            print(f"  down_proj[local_slots]    gather: shape=({N},{TK},{dp.shape[1]},{dp.shape[2]}) "
                  f"-> {dp_gather_bytes/2**20:.2f} MiB", flush=True)
            print("=== [/GATHER SIZE] ===\n", flush=True)
        return result

    Qwen35MoE._forward_gathered_ep = wrapped


class _Args:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(ROOT), "qwen35_checkpoint"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    ap.add_argument("--moe-w8a8", action="store_true", default=False)
    ap.add_argument("--moe-w8a8-group-size", type=int, default=128)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    cli = ap.parse_args()

    _patch_quantization()
    _patch_capture_cudagraph()
    _patch_gather_shapes()

    args = _Args()
    args.model = cli.checkpoint
    args.max_num_batched_tokens = 4096
    args.concurrency = [cli.concurrency]
    args.max_model_len = cli.max_model_len
    args.gpu_memory_utilization = cli.gpu_memory_utilization
    args.tensor_parallel_size = cli.tp
    args.enforce_eager = cli.enforce_eager
    args.use_fused_gdr_kernel = False
    args.use_moe_w8a8 = cli.moe_w8a8
    args.moe_w8a8_weight_group_size = cli.moe_w8a8_group_size
    args.fake_config_loader = False

    print(f"enforce_eager={args.enforce_eager}  use_moe_w8a8={args.use_moe_w8a8}  "
          f"group_size={args.moe_w8a8_weight_group_size}  concurrency={cli.concurrency}  "
          f"tp={args.tensor_parallel_size}  gpu_memory_utilization={args.gpu_memory_utilization}",
          flush=True)

    engine = bt.build_engine(args)
    print("[OK] engine constructed and warmed up without OOM", flush=True)

    result = bt.run_trial(engine, cli.concurrency, cli.prompt_len, cli.output_len, seed=1234)
    print(f"[OK] decode trial completed: tok/s={result['tok_s']:.1f} wall_s={result['wall_s']:.2f}",
          flush=True)

    engine.exit()


if __name__ == "__main__":
    main()
