"""Ground-truth introspection: what does the REAL HF model class actually
look like for the linear-attention (GDR/GatedDeltaNet) layer -- module
names, parameter shapes, and forward()/__init__() source -- independent of
any assumption this repo's own code or src/model_small_qwen3.5.py makes.

The engine/layerwise diagnostic still shows a catastrophic break exactly at
layer 0 (linear_attention) even after fixing in_proj_qkv's Q/K/V-vs-tp_rank
sharding, which means either that fix has its own bug, or the real
checkpoint's on-disk layout/module structure doesn't match what
src/model_small_qwen3.5.py assumes (that file is this repo's own
hand-written reference, not necessarily identical to whatever the installed
`transformers` version actually implements for this checkpoint's
architecture -- especially since phase_hf's own output already warned:
"The fast path is not available... flash-linear-attention... causal-conv1d",
implying the HF side's linear-attention mixer follows the fla-org library's
conventions, which may differ from this repo's 4-separate-projections
(qkv/z/a/b) design).

Cheap: builds the HF model class on the meta device (real module tree and
class hierarchy, zero actual tensor storage / no multi-GB weight load), so
this doesn't need to fight the same-GPU-memory constraint the other phase6
scripts have. No GPU needed at all.

Usage: python tests/inspect_hf_linear_attn.py
"""
import inspect
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")


def main():
    config = AutoConfig.from_pretrained(CKPT_DIR, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print(f"top-level model class: {type(model).__module__}.{type(model).__name__}")

    # Find the actual decoder-layer container, whatever it's nested under
    # (language_model wrapper, etc.) -- don't assume a path.
    layers = None
    layers_owner_path = None
    for name, module in model.named_modules():
        if name.endswith("layers") and hasattr(module, "__len__") and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") or "Layer" in type(first).__name__ or "Block" in type(first).__name__:
                layers = module
                layers_owner_path = name
                break
    if layers is None:
        print("Could not auto-locate the decoder layer list -- dumping top 3 levels of named_modules instead:")
        for name, module in model.named_modules():
            if name.count(".") <= 2:
                print(f"  {name}: {type(module).__name__}")
        return

    print(f"decoder layers found at: model.{layers_owner_path}  (count={len(layers)})")

    layer0 = layers[0]
    print(f"\nlayer 0 class: {type(layer0).__module__}.{type(layer0).__name__}")
    print("layer 0 direct children:")
    for name, child in layer0.named_children():
        print(f"  .{name}: {type(child).__module__}.{type(child).__name__}")

    # Find the linear-attention / GDR / GatedDeltaNet mixer submodule specifically.
    mixer = None
    mixer_name = None
    for name, child in layer0.named_children():
        cls_name = type(child).__name__.lower()
        if any(tag in cls_name for tag in ("delta", "linear", "gdr", "mixer", "gated")):
            mixer = child
            mixer_name = name
            break
    if mixer is None:
        print("\nCould not auto-identify the linear-attention mixer by class-name heuristic. "
              "Inspect the children list above manually.")
        return

    print(f"\nlinear-attention mixer: layer0.{mixer_name} -- {type(mixer).__module__}.{type(mixer).__name__}")
    print("\nparameters (name: shape):")
    for pname, p in mixer.named_parameters():
        print(f"  {pname}: {tuple(p.shape)}")

    print("\n=== __init__ source ===")
    try:
        print(inspect.getsource(type(mixer).__init__))
    except (OSError, TypeError) as e:
        print(f"(unavailable: {e})")

    print("\n=== forward source ===")
    try:
        print(inspect.getsource(type(mixer).forward))
    except (OSError, TypeError) as e:
        print(f"(unavailable: {e})")

    print(f"\nsource file: {inspect.getfile(type(mixer))}")


if __name__ == "__main__":
    main()
