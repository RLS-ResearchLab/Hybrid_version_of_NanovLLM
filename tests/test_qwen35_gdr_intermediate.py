# tests/test_qwen35_gdr_intermediate.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config
from test_qwen35_batching import build_model_and_state
from nanovllm.models.qwen3_5 import Qwen35LinearAttention
import torch

def get_first_linear_layer(model):
    return next(m for m in model.modules() if isinstance(m, Qwen35LinearAttention))

def run_and_capture(la, hidden_states, cu_seqlens, states, conv_states, tag):
    """Monkey-patch-free capture: replicate forward() up to post-conv, per segment."""
    import torch.nn.functional as F
    lkh, lvh, lhd, ck = la.lkh, la.lvh, la.lhd, la.ck
    z = la.in_proj_z(hidden_states)
    a = la.in_proj_a(hidden_states)
    b = la.in_proj_b(hidden_states)
    qkv = la.in_proj_qkv(hidden_states)
    num_segments = cu_seqlens.numel() - 1
    for i in range(num_segments):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i+1])
        T_i = end - start
        seg_qkv = qkv[start:end].unsqueeze(0)
        seg_a = a[start:end].unsqueeze(0)
        seg_b = b[start:end].unsqueeze(0)
        seg_conv_state = conv_states[i:i+1] if conv_states is not None else None
        qkv_t = seg_qkv.transpose(1, 2)
        if seg_conv_state is not None:
            qkv_t_cat = torch.cat([seg_conv_state, qkv_t], dim=2)
        else:
            qkv_t_cat = qkv_t
        qkv_conv = la.conv1d(qkv_t_cat)
        offset = seg_conv_state.shape[2] if seg_conv_state is not None else 0
        qkv_conv = qkv_conv[:, :, offset:offset+T_i].transpose(1, 2)
        qkv_conv = F.silu(qkv_conv)
        print(f"  [{tag}] seg{i} T={T_i} conv_out sum={qkv_conv.float().sum().item():.6f} "
              f"g_input(a) sum={seg_a.float().sum().item():.6f}")

def main():
    init_dist()
    config = make_small_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sm = build_model_and_state(config, device, max_num_seqs=8)
    la = get_first_linear_layer(model)

    torch.manual_seed(99)
    seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11]]

    from nanovllm.utils.context import set_context, reset_context

    # single seq0 alone
    ids0 = torch.tensor(seqs[0], device=device)
    hidden0 = model.model.embed_tokens(ids0)
    cu0 = torch.tensor([0, len(seqs[0])], dtype=torch.int32, device=device)
    conv0 = torch.zeros(1, la.qkv_dim, la.ck-1, device=device, dtype=torch.bfloat16)
    run_and_capture(la, hidden0, cu0, None, conv0, "single-seq0")

    # packed seq0+seq1
    ids_packed = torch.tensor(sum(seqs, []), device=device)
    hidden_packed = model.model.embed_tokens(ids_packed)
    cu_packed = torch.tensor([0, len(seqs[0]), len(seqs[0])+len(seqs[1])], dtype=torch.int32, device=device)
    conv_packed = torch.zeros(2, la.qkv_dim, la.ck-1, device=device, dtype=torch.bfloat16)
    run_and_capture(la, hidden_packed, cu_packed, None, conv_packed, "packed-seq0")

if __name__ == "__main__":
    main()