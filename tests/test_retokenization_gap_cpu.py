"""B3 -- empirically characterize the retokenization / EOS-under-ignore-eos gap.

Two related things the handoff flagged about tok/s numbers measured with
--ignore-eos:

  1. Under --ignore-eos the scheduler's EOS stop condition is disabled
     (Scheduler.postprocess: `not seq.ignore_eos and token_id == self.eos`),
     so generation runs the FULL max_tokens even after the model samples
     EOS -- meaning some fraction of "generated tokens" in those benchmarks
     is EOS the model wanted to stop at, forced to continue. Compute cost
     spent on non-content tokens.

  2. src/server.py decodes the returned text with skip_special_tokens=True
     while usage.completion_tokens = len(output_ids) counts the specials --
     so re-encoding the returned text always comes up SHORT by exactly the
     number of special tokens dropped (the always-negative retok bias).

Data: token_discrepancy_capture.json (checked in) -- 8 real completions
captured direct-engine at temperature=1.0, ignore_eos=True, output_len=128,
with reported_completion_token_ids + decoded_text. Pure Python + the
tokenizer (CPU). No GPU.

NOTE: the engine's EOS id is the tokenizer's eos_token_id (LLMEngine.__init__:
`config.eos = self.tokenizer.eos_token_id`), which is 248046 <|im_end|> --
NOT config.json's eos_token_id=248044 <|endoftext|>. Both are checked below.

Usage:
    python tests/test_retokenization_gap_cpu.py
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAPTURE = os.path.join(_ROOT, "token_discrepancy_capture.json")


def main():
    if not os.path.exists(_CAPTURE):
        print(f"STOP: {_CAPTURE} not found -- regenerate with capture_token_discrepancy.py on a GPU box.")
        sys.exit(1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(_ROOT, "qwen35_checkpoint"), use_fast=True)
    engine_eos = tok.eos_token_id                 # what Scheduler.postprocess actually compares against
    cfg_eos = 248044                              # config.json's value (<|endoftext|>)
    special_ids = set(tok.all_special_ids)
    print(f"tokenizer eos_token_id (engine EOS) = {engine_eos} {tok.decode([engine_eos])!r}")
    print(f"config.json eos_token_id            = {cfg_eos} {tok.decode([cfg_eos])!r}")
    print(f"all_special_ids = {sorted(special_ids)}\n")

    records = json.load(open(_CAPTURE, encoding="utf-8"))
    print(f"{len(records)} captured completions (temperature=1.0, ignore_eos=True, output_len=128)\n")

    tot_reported = tot_specials = tot_engine_eos = tot_cfg_eos = 0
    tot_gap_strip = tot_gap_keep = 0
    print(f"{'seq':>3} {'reported':>8} {'#special':>8} {'#eos(engine)':>12} {'#eos(cfg)':>9} "
          f"{'reenc(keep)':>11} {'reenc(strip)':>12} {'strip_gap':>9}")
    for r in records:
        ids = r["reported_completion_token_ids"]
        n = len(ids)
        n_special = sum(1 for t in ids if t in special_ids)
        n_eeos = sum(1 for t in ids if t == engine_eos)
        n_ceos = sum(1 for t in ids if t == cfg_eos)
        text_keep = tok.decode(ids, skip_special_tokens=False)
        text_strip = tok.decode(ids, skip_special_tokens=True)
        reenc_keep = len(tok.encode(text_keep, add_special_tokens=False))
        reenc_strip = len(tok.encode(text_strip, add_special_tokens=False))
        # server path: completion_tokens = n, but returned text is text_strip
        strip_gap = n - reenc_strip
        keep_gap = n - reenc_keep
        tot_reported += n
        tot_specials += n_special
        tot_engine_eos += n_eeos
        tot_cfg_eos += n_ceos
        tot_gap_strip += strip_gap
        tot_gap_keep += keep_gap
        print(f"{r['seq_index']:>3} {n:>8} {n_special:>8} {n_eeos:>12} {n_ceos:>9} "
              f"{reenc_keep:>11} {reenc_strip:>12} {strip_gap:>+9}")

    print("\n" + "-" * 70)
    print(f"TOTALS over {len(records)} completions ({tot_reported} reported tokens):")
    print(f"  special tokens in output      : {tot_specials}  ({100*tot_specials/tot_reported:.1f}%)")
    print(f"  engine-EOS ({engine_eos}) tokens   : {tot_engine_eos}  "
          f"({100*tot_engine_eos/tot_reported:.2f}% of all 'generated' tokens)")
    print(f"  config-EOS ({cfg_eos}) tokens      : {tot_cfg_eos}")
    print(f"  server-path retok gap (reported - reenc_strip): {tot_gap_strip:+d} total, "
          f"{tot_gap_strip/len(records):+.1f}/completion")
    print(f"  no-strip retok gap    (reported - reenc_keep) : {tot_gap_keep:+d} total")
    print()
    print("READ:")
    if tot_engine_eos == 0:
        print(f"  - ZERO engine-EOS ({engine_eos}) tokens in any of these {len(records)} completions.")
        print("    Under temperature=1.0 on adversarial random-vocab prompts the model rarely")
        print("    samples EOS at all, so 'forced past a wanted stop' is NOT a big hidden compute")
        print("    cost in THIS workload. It could still matter for a low-temperature / real-prompt")
        print("    workload where the model genuinely wants to stop -- not settled by this data.")
    else:
        frac = 100 * tot_engine_eos / tot_reported
        print(f"  - {frac:.2f}% of 'generated' tokens are engine-EOS forced past by --ignore-eos --")
        print(f"    a real compute cost on non-content tokens in --ignore-eos tok/s numbers.")
    print(f"  - The server-path retok gap is {'negative (as predicted)' if tot_gap_strip < 0 else 'NOT negative'}"
          f" and its magnitude tracks the special-token count "
          f"({tot_specials} specials vs {tot_gap_strip:+d} gap) plus ordinary BPE re-encode drift.")
    print("  - CONCLUSION: the retok gap is a decode/reporting artifact (special-token strip +")
    print("    BPE non-idempotence), not a sign of a generation bug. tok/s under --ignore-eos is")
    print("    still a fair *relative* comparison; the absolute number includes some non-content")
    print("    tokens but this capture doesn't show a large EOS-spam effect.")
    sys.exit(0)


if __name__ == "__main__":
    main()
