# tests/add_fake_chat_template.py
"""Attaches a minimal chat template to the fake fixture's tokenizer.

tests/make_fake_tokenizer.py attaches plain gpt2 -- by design, its own
docstring says it's "only here so the engine can construct," not a
chat-capable tokenizer. gpt2 ships no `chat_template`, so
`tokenizer.apply_chat_template(...)` (used by src/server.py's
/v1/chat/completions route, copied verbatim from the basic engine) raises
`ValueError: Cannot use chat template functions because
tokenizer.chat_template is not set...` for every request.

This only affects the small fixture. The real Qwen3.5-35B-A3B checkpoint's
own tokenizer ships a real chat template, so this script and its output
are dev/smoke-test-only -- not something src/server.py itself needs to
work around.

IMPORTANT -- edits tokenizer_config.json directly, does NOT round-trip
through `AutoTokenizer.from_pretrained(OUT_DIR)`. This directory's
config.json has `model_type: "qwen3_5_moe"`, which is now a REAL
registered type in this transformers version (it wasn't when these
fixtures were first written -- see src/server.py's fake-config-loader
shim docstring for the AutoConfig side of the same drift). Loading the
tokenizer back from OUT_DIR mis-resolves it to the real `Qwen3_5Tokenizer`
class instead of the gpt2 files actually on disk (silently: no error, just
vocab_size=1 and every encode() call returning []). save_pretrained() on
that mis-resolved instance then overwrites tokenizer_config.json with
`tokenizer_class: "Qwen3_5Tokenizer"` and GPT2-incompatible fields
(add_prefix_space: null) -- corrupting the fixture. (This actually
happened once; tests/fake_qwen35_small was regenerated from a fresh
`AutoTokenizer.from_pretrained("gpt2")` load to fix it.) Editing the JSON
in place sidesteps the whole mis-resolution path.

Usage:
    python tests/make_fake_hf_config.py    # if not already run
    python tests/make_fake_tokenizer.py    # if not already run
    python tests/add_fake_chat_template.py
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
CONFIG_PATH = os.path.join(OUT_DIR, "tokenizer_config.json")

# Minimal, readable role: content transcript -- not meant to resemble any
# real model's template, just enough structure for apply_chat_template to
# succeed and produce a decodable prompt string.
_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def main():
    assert os.path.isfile(CONFIG_PATH), "run tests/make_fake_tokenizer.py first"

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Guard against exactly the corruption described above: if a prior run
    # (or the buggy round-trip this script used to do) already clobbered
    # tokenizer_class, fail loudly instead of silently patching a broken
    # fixture -- regenerate via `AutoTokenizer.from_pretrained("gpt2")` +
    # `save_pretrained(OUT_DIR)` instead of re-running this script over it.
    assert config.get("tokenizer_class") in ("GPT2Tokenizer", "GPT2TokenizerFast"), (
        f"tokenizer_config.json has tokenizer_class={config.get('tokenizer_class')!r}, "
        f"expected GPT2Tokenizer(Fast) -- fixture looks corrupted, regenerate it "
        f"(see this script's docstring) rather than patching over it."
    )

    config["chat_template"] = _TEMPLATE
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"chat_template written to {CONFIG_PATH}")

    # Verify via a completely fresh process-level import path, same as
    # LLMEngine.__init__'s own AutoTokenizer.from_pretrained(config.model)
    # call -- this is the one read of the file we actually want, done last,
    # after all writes are already on disk.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)
    assert type(tok).__name__ in ("GPT2Tokenizer", "GPT2TokenizerFast"), (
        f"reload resolved to {type(tok).__name__}, not GPT2Tokenizer(Fast) -- "
        f"the model_type mis-resolution bug is back somehow"
    )
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Say hello in one sentence."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) > 0, "encode() returned no tokens -- tokenizer is still broken"
    print(f"Verified OK: rendered={text!r} -> {len(ids)} tokens")


if __name__ == "__main__":
    main()
