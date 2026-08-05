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

Usage:
    python tests/make_fake_hf_config.py    # if not already run
    python tests/make_fake_tokenizer.py    # if not already run
    python tests/add_fake_chat_template.py
"""
import os

from transformers import AutoTokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")

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
    assert os.path.isdir(OUT_DIR), "run tests/make_fake_tokenizer.py first"

    tok = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)
    tok.chat_template = _TEMPLATE
    tok.save_pretrained(OUT_DIR)

    reloaded = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)
    text = reloaded.apply_chat_template(
        [{"role": "user", "content": "Say hello in one sentence."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    print(f"chat_template attached to {OUT_DIR}")
    print(f"Rendered sample:\n{text!r}")


if __name__ == "__main__":
    main()
