# tests/make_fake_tokenizer.py
"""Attaches a real HF tokenizer (gpt2) to the fake Qwen3.5 config dir so
LLMEngine.__init__'s unconditional AutoTokenizer.from_pretrained(config.model)
call doesn't throw on missing tokenizer files.

This measures nothing architectural about Qwen3.5 -- gpt2's vocabulary has no
relationship to the model's tokenization, it's only here so the engine can
construct. The one real check: gpt2's vocab must stay under the fake config's
vocab_size (248320), or an encoded prompt could hand the model an id its
embedding table can't index -- an out-of-bounds lookup, not a formality.

Usage:
    python tests/make_fake_hf_config.py   # if not already run
    python tests/make_fake_tokenizer.py
"""
import os

from transformers import AutoTokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
VOCAB_SIZE = 248320  # must match tests/make_fake_hf_config.py's config["vocab_size"]


def main():
    assert os.path.isdir(OUT_DIR), "run tests/make_fake_hf_config.py first"

    print("Downloading/saving gpt2 tokenizer...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.save_pretrained(OUT_DIR)
    print(f"Saved tokenizer files to {OUT_DIR}")

    print("Reloading via AutoTokenizer.from_pretrained(fake_dir), matching LLMEngine's call...")
    reloaded = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)
    print(f"  OK: {type(reloaded).__name__}, vocab_size={reloaded.vocab_size}")

    test_strings = [
        "The quick brown fox jumps over the lazy dog.",
        "Qwen3.5 hybrid GDR linear attention smoke test 12345 !@#$%",
        "",
    ]
    max_id_seen = -1
    for s in test_strings:
        ids = reloaded.encode(s)
        this_max = max(ids) if ids else None
        if ids:
            max_id_seen = max(max_id_seen, max(ids))
        print(f"  encode({s!r}) -> {len(ids)} ids, max_id={this_max}")

    print(f"Overall max token id seen: {max_id_seen} (model vocab_size={VOCAB_SIZE})")
    assert max_id_seen < VOCAB_SIZE, (
        f"encoded token id {max_id_seen} >= vocab_size {VOCAB_SIZE} -- "
        f"this would be an out-of-bounds embedding lookup"
    )
    print("[PASS] no encoded id reaches or exceeds the model's vocab_size")


if __name__ == "__main__":
    main()
