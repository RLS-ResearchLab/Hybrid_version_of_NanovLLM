# tests/test_qwen35_bisect.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config
from test_qwen35_batching import build_model_and_state, run_single, run_packed
import torch

def run_bisect(force_all_linear: bool, force_all_full: bool):
    config = make_small_config()
    if force_all_linear:
        config.full_attention_interval = 10**9   # never true -> all linear
    if force_all_full:
        config.layers_block_type = ["full_attention"] * config.num_hidden_layers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sm = build_model_and_state(config, device, max_num_seqs=8)

    torch.manual_seed(99)
    seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11, 5, 9]]
    baseline = [run_single(model, sm, s, device) for s in seqs]
    packed = run_packed(model, sm, seqs, device)

    for i, (b, p) in enumerate(zip(baseline, packed)):
        cos = torch.nn.functional.cosine_similarity(
            b.float().reshape(1, -1), p.float().reshape(1, -1)).item()
        print(f"  seq {i} (len={len(seqs[i])}): cosine={cos:.6f}")

if __name__ == "__main__":
    init_dist()
    print("=== ALL LINEAR ATTENTION (no flash-attn path) ===")
    run_bisect(force_all_linear=True, force_all_full=False)
    print("\n=== ALL FULL ATTENTION (no GDR path) ===")
    run_bisect(force_all_linear=False, force_all_full=True)