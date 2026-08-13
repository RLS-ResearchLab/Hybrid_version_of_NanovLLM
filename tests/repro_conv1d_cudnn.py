"""Minimal repro for the CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH crash seen
in tests/test_qwen35_fused_gdr.py's decode-shaped sub-case
(packed_multi_segment seg_lens=(1,1,1,1)) -- isolates the exact depthwise
conv1d shape (groups==channels==2048, spatial length 1) from everything
else (fla, the GDR scan, cu_seqlens, etc.) to confirm whether this is a
pure cuDNN backend-selection/library-version issue on this machine, or
something else.

Usage:
    python tests/repro_conv1d_cudnn.py
"""
import torch
import torch.nn as nn


def main():
    if not torch.cuda.is_available():
        print("[SKIP] no CUDA GPU available")
        return

    conv = nn.Conv1d(2048, 2048, 4, groups=2048, padding=3).cuda().to(torch.bfloat16)
    x = torch.randn(1, 2048, 1, device="cuda", dtype=torch.bfloat16)
    y = conv(x)
    print(f"OK - no crash, output shape {tuple(y.shape)}")


if __name__ == "__main__":
    main()
