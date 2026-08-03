"""Does Sampler's greedy (temperature==0) argmax for one fixed row change
depending on how many OTHER rows are stacked alongside it in the same
call? Standalone, no checkpoint needed, seconds to run -- built to test a
specific hypothesis from the GSM8K decode-contamination investigation:
gsm8k_prefill_scale_check.py already confirmed raw prefill LOGITS match
(cosine>=0.99 + argmax) between concurrency=1 and concurrency=8 for real
GSM8K-scale prompts, but tests/gsm8k_decode_contamination_check.py (which
goes through the REAL generate()/run()/self.sampler() path, not raw
logits) still found the actual SAMPLED first token diverge. Since
layers/sampler.py's Sampler.forward is @torch.compile-decorated and
unconditionally computes a second, RNG-consuming path (Gumbel-max
`sampled_tokens`) for every row even when `temperature==0` picks
`greedy_tokens` via torch.where -- a compiled kernel COULD specialize
differently per batch shape. This test isolates exactly that: same fixed
target row's logits, same RNG seed before each call, only the batch size
(how many OTHER rows are packed alongside it) varies. If greedy output for
the fixed row changes with batch size, that's a real, checkpoint-
independent bug in the sampler itself -- not decode-EP, not prefill
packing, not KV-cache/state.

Must run on the SAME kind of machine production runs on (Linux + CUDA, a
working C++ compiler for torch.compile's Inductor backend) -- this can't
be verified on a Windows/no-compiler dev box, which is why this is handed
off rather than run directly.

Usage:
    python tests/test_sampler_batch_invariance.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from layers.sampler import Sampler  # noqa: E402

VOCAB = 248320  # matches this checkpoint's real vocab size (see gsm8k_prefill_scale_check.py's
                 # "logits shape (248320,)" output) -- shape matters here, so use the real size
BATCH_SIZES = [1, 2, 3, 8, 16]
N_TRIALS = 5  # different fixed rows, to not rely on one row being a fluke


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    sampler = Sampler().to(device)

    all_pass = True
    for trial in range(N_TRIALS):
        torch.manual_seed(1000 + trial)
        target_row = torch.randn(VOCAB, device=device)
        eager_argmax = target_row.argmax().item()

        results = {}
        for bs in BATCH_SIZES:
            torch.manual_seed(999)  # identical RNG state before every call, across batch sizes
            logits = torch.randn(bs, VOCAB, device=device)
            logits[0] = target_row  # row 0 is always this trial's SAME fixed logits
            temperatures = torch.zeros(bs, device=device)  # all greedy
            out = sampler(logits, temperatures)
            results[bs] = out[0].item()

        consistent = len(set(results.values())) == 1 and list(results.values())[0] == eager_argmax
        status = "PASS" if consistent else "FAIL"
        print(f"[{status}] trial {trial}: eager_argmax={eager_argmax}  "
              f"sampler_row0_by_batch_size={results}")
        if not consistent:
            all_pass = False

    print()
    if all_pass:
        print(f"RESULT: PASS -- greedy argmax for a fixed row is IDENTICAL across all batch sizes "
              f"{BATCH_SIZES} and matches eager .argmax(), over {N_TRIALS} trials. Sampler is not the "
              f"cause of the decode-contamination-check divergence.")
    else:
        print(f"RESULT: FAIL -- greedy argmax for a fixed row CHANGED depending on batch size in at "
              f"least one trial. This is a real, checkpoint-independent bug in Sampler.forward "
              f"(layers/sampler.py) -- likely explains gsm8k_decode_contamination_check.py's divergence, "
              f"unrelated to decode-EP/prefill/KV-cache/state.")


if __name__ == "__main__":
    main()
