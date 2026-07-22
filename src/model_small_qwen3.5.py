"""
Qwen3.5-MoE-Small: scaled-down architecture for fast experimentation.
Same structure as model.py — GDR linear attention, GQA full attention,
MoE FFN with shared expert — but with much smaller dims, ~290M parameters.
Initializes with random weights; no weight loading.

Small config vs full:
  H     512   (was 2048)
  L     8     (was 40)  — 2 groups of (3 GDR + 1 GQA)
  NQ    8     (was 16)  — Q heads in full attention
  NKV   1     (was 2)   — KV heads in full attention (GQA)
  DH    128   (was 256) — head dim in full attention
  LKH   8     (was 16)  — K/Q heads in GDR
  LVH   16    (was 32)  — V/SSM heads in GDR
  LHD   64    (was 128) — head dim in GDR
  NE    32    (was 256) — number of MoE experts
  TK    4     (was 8)   — top-K routing
  MI    256   (was 512) — expert intermediate size
  SI    256   (was 512) — shared expert intermediate size
  vocab 248320 (same)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Small model constants ──────────────────────────────────────────────────────
H     = 512
L     = 8
FAI   = 4          # full_attention_interval (every 4th layer = GQA)
NQ    = 8          # Q heads (full attention)
NKV   = 1          # KV heads (full attention)
DH    = 128        # head dim (full attention)
LKH   = 8          # K/Q heads (GDR)
LVH   = 16         # V/SSM heads (GDR)
LHD   = 64         # head dim (GDR)
CK    = 4          # conv1d kernel size
THETA = 10_000_000.0
PROT  = 0.25       # partial rotary factor → rot_dim = int(DH*PROT) = 32
NE    = 32         # MoE experts
TK    = 4          # top-K
MI    = 256        # expert intermediate size
SI    = 256        # shared expert intermediate size
EPS   = 1e-6
VOCAB = 248320

def is_full(i): return (i + 1) % FAI == 0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        dt = x.dtype
        x  = x.float()
        x  = x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * (1.0 + self.weight.float())).to(dt)


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, gate):
        dt   = x.dtype
        x    = x.float()
        x    = x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x    = (self.weight.to(dt) * x.to(dt)).float()
        return (x * F.silu(gate.float())).to(dt)


def build_rope(T, device):
    rot = int(DH * PROT)
    inv = 1.0 / THETA ** (torch.arange(0, rot, 2, device=device).float() / rot)
    t   = torch.arange(T, device=device).float()
    f   = torch.outer(t, inv)
    return f.cos(), f.sin()


def rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


def apply_rope(q, k, cos, sin):
    rot = cos.shape[-1] * 2
    qr, qp = q[..., :rot], q[..., rot:]
    kr, kp = k[..., :rot], k[..., rot:]
    T = q.shape[2]
    c = torch.cat([cos[:T], cos[:T]], -1).to(q)[None, None]
    s = torch.cat([sin[:T], sin[:T]], -1).to(q)[None, None]
    return (torch.cat([qr*c + rot_half(qr)*s, qp], -1),
            torch.cat([kr*c + rot_half(kr)*s, kp], -1))


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class FullAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, 2 * NQ * DH, bias=False)
        self.k_proj = nn.Linear(H, NKV * DH, bias=False)
        self.v_proj = nn.Linear(H, NKV * DH, bias=False)
        self.o_proj = nn.Linear(NQ * DH, H, bias=False)
        self.q_norm = RMSNorm(DH)
        self.k_norm = RMSNorm(DH)

    def forward(self, x, cos, sin, mask=None, kv=None):
        B, T, _ = x.shape
        q_full = self.q_proj(x).view(B, T, NQ, DH * 2)
        q, gate = q_full.chunk(2, dim=-1)
        gate = gate.reshape(B, T, -1)

        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(B, T, NKV, DH).transpose(1, 2))
        v = self.v_proj(x).view(B, T, NKV, DH).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if kv is not None:
            k = torch.cat([kv[0], k], 2)
            v = torch.cat([kv[1], v], 2)
        new_kv = (k, v)

        g = NQ // NKV
        k = k.repeat_interleave(g, 1)
        v = v.repeat_interleave(g, 1)

        scale = DH ** -0.5
        a = (q @ k.transpose(-2, -1)) * scale
        if mask is not None: a = a + mask
        a = F.softmax(a.float(), -1).to(q.dtype)
        o = (a @ v).transpose(1, 2).reshape(B, T, NQ * DH)
        o = o * torch.sigmoid(gate.to(o.dtype))
        return self.o_proj(o), new_kv


class LinearAttn(nn.Module):
    QKV = (LKH + LKH + LVH) * LHD  # (8+8+16)×64 = 2048

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(H, self.QKV, bias=False)
        self.in_proj_z   = nn.Linear(H, LVH * LHD, bias=False)
        self.in_proj_a   = nn.Linear(H, LVH, bias=False)
        self.in_proj_b   = nn.Linear(H, LVH, bias=False)
        self.conv1d      = nn.Conv1d(self.QKV, self.QKV, CK, groups=self.QKV, padding=CK-1, bias=False)
        self.A_log       = nn.Parameter(torch.zeros(LVH))
        self.dt_bias     = nn.Parameter(torch.zeros(LVH))
        self.norm        = RMSNormGated(LHD)
        self.out_proj    = nn.Linear(LVH * LHD, H, bias=False)

    def forward(self, x, state=None, conv_state=None):
        B, T, _ = x.shape
        QKV = self.QKV

        z   = self.in_proj_z(x)
        a   = self.in_proj_a(x)
        b   = self.in_proj_b(x)
        qkv = self.in_proj_qkv(x)

        qkv_t = qkv.transpose(1, 2)
        if conv_state is not None and T == 1:
            combined = torch.cat([conv_state, qkv_t], dim=2)
            new_conv = combined[:, :, -(CK-1):].detach()  #Keep only the last kernel_size-1 elements of the sequence as convolution history, remove them from
                                                          #the gradient graph, and store them as the new convolution cache for the next autoregressive step
            qkv_conv = F.conv1d(combined, self.conv1d.weight, self.conv1d.bias,
                                padding=0, groups=QKV).transpose(1, 2)
        else:
            if conv_state is not None:
                qkv_t = torch.cat([conv_state, qkv_t], dim=2)
            qkv_conv = self.conv1d(qkv_t)
            new_conv  = qkv_t[:, :, -(CK-1):].detach()
            offset = conv_state.shape[2] if conv_state is not None else 0
            qkv_conv = qkv_conv[:, :, offset:offset+T].transpose(1, 2)
        qkv_conv = F.silu(qkv_conv)

        q = qkv_conv[:, :, :LKH*LHD].view(B, T, LKH, LHD)
        k = qkv_conv[:, :, LKH*LHD:2*LKH*LHD].view(B, T, LKH, LHD)
        v = qkv_conv[:, :, 2*LKH*LHD:].view(B, T, LVH, LHD)

        g    = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        beta = b.sigmoid()

        q = q.repeat_interleave(2, dim=2)
        k = k.repeat_interleave(2, dim=2)

        scale = LHD ** -0.5
        q = l2norm(q.float()) * scale
        k = l2norm(k.float())

        # GDR scan in float32 (matches model.py / HF torch_recurrent_gated_delta_rule)
        dt = x.dtype
        g    = g.float()
        beta = beta.float()
        v    = v.float()

        if state is None:
            S = torch.zeros(B, LVH, LHD, LHD, device=x.device, dtype=torch.float32)
        else:
            S = state.float()

        ys = []
        for t in range(T):
            g_t    = g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, t, :].unsqueeze(-1)
            k_t    = k[:, t, :, :]
            v_t    = v[:, t, :, :]
            q_t    = q[:, t, :, :]
            S = S * g_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta  = (v_t - kv_mem) * beta_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            ys.append((S * q_t.unsqueeze(-1)).sum(dim=-2))

        new_state = S.detach()
        y = torch.stack(ys, dim=1).to(dt)   # cast back to model dtype before gated norm
        y = y.reshape(B * T * LVH, LHD)
        z_flat = z.reshape(B * T * LVH, LHD)
        y = self.norm(y, z_flat)
        y = y.reshape(B, T, LVH * LHD)
        return self.out_proj(y), new_state, new_conv


class SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, SI, bias=False)
        self.up_proj   = nn.Linear(H, SI, bias=False)
        self.down_proj = nn.Linear(SI, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(NE, 2*MI, H))
        self.down_proj    = nn.Parameter(torch.empty(NE, H, MI))


class MoEFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts            = Experts()
        self.gate               = nn.Linear(H, NE, bias=False)
        self.shared_expert      = SharedExpert()
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        xf = x.reshape(-1, H)
        N  = xf.shape[0]

        w, idx = torch.topk(self.gate(xf), TK, -1)
        w = F.softmax(w, -1).to(x.dtype)

        flat_idx = idx.reshape(-1)
        flat_w   = w.reshape(-1)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)

        sort_order    = torch.argsort(flat_idx, stable=True)
        sorted_idx    = flat_idx[sort_order]
        sorted_tokens = token_rep[sort_order]
        sorted_weights = flat_w[sort_order]

        expert_counts  = torch.bincount(sorted_idx, minlength=NE)
        expert_offsets = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long),
                                    expert_counts.cumsum(0)[:-1]])

        sorted_out = torch.zeros(N * TK, H, device=x.device, dtype=x.dtype)
        for e in range(NE):
            cnt = expert_counts[e].item()
            if cnt == 0:
                continue
            start = expert_offsets[e].item()
            xt = sorted_tokens[start:start+cnt]
            gw, uw = self.experts.gate_up_proj[e].chunk(2, 0)
            h = F.silu(xt @ gw.t()) * (xt @ uw.t())
            h = h @ self.experts.down_proj[e].t()
            sorted_out[start:start+cnt] = sorted_weights[start:start+cnt].unsqueeze(-1) * h

        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1)
        sg  = torch.sigmoid(self.shared_expert_gate(xf))
        return (out + sg * self.shared_expert(xf)).view(B, T, H)


class Layer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i    = i
        self.full = is_full(i)
        self.input_layernorm          = RMSNorm(H)
        self.post_attention_layernorm = RMSNorm(H)
        if self.full:
            self.self_attn = FullAttn()
        else:
            self.linear_attn = LinearAttn()
        self.mlp = MoEFFN()

    def forward(self, x, cos=None, sin=None, mask=None, kv=None, state=None, conv=None):
        r = x
        h = self.input_layernorm(x)
        if self.full:
            a, new_kv    = self.self_attn(h, cos, sin, mask, kv)
            new_s, new_c = None, None
        else:
            a, new_s, new_c = self.linear_attn(h, state, conv)
            new_kv = None
        x = r + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv, new_s, new_c


class Qwen35MoESmall(nn.Module):
    """
    Qwen3.5-MoE-Small: ~290M parameters, same architecture pattern as the 35B model.
    8 layers (2× groups of 3 GDR + 1 GQA). Initializes with random weights.
    """
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, H)
        self.layers  = nn.ModuleList([Layer(i) for i in range(L)])
        self.norm    = RMSNorm(H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        self._rope_T = 0
        self.register_buffer('_cos', torch.zeros(1, 1), persistent=False)
        self.register_buffer('_sin', torch.zeros(1, 1), persistent=False)

    def _ensure_rope(self, T, dev):
        if T > self._rope_T:
            c, s = build_rope(T + 64, dev)
            self._cos, self._sin = c, s
            self._rope_T = T + 64

    @torch.no_grad()
    def forward(self, ids, kvs=None, states=None, convs=None):
        B, T  = ids.shape
        dev   = ids.device
        x     = self.embed_tokens(ids)

        if kvs    is None: kvs    = [None] * L
        if states is None: states = [None] * L
        if convs  is None: convs  = [None] * L

        past = kvs[3][0].shape[2] if kvs[3] is not None else 0
        Ttot = T + past
        self._ensure_rope(Ttot, dev)

        mask = torch.full((T, Ttot), float('-inf'), device=dev, dtype=x.dtype)
        for i in range(T): mask[i, :past+i+1] = 0.0
        mask = mask[None, None]

        nkvs, nss, ncs = [], [], []
        for i, layer in enumerate(self.layers):
            x, nk, ns, nc = layer(x, cos=self._cos, sin=self._sin,
                                  mask=mask, kv=kvs[i],
                                  state=states[i], conv=convs[i])
            nkvs.append(nk); nss.append(ns); ncs.append(nc)

        return self.lm_head(self.norm(x)), nkvs, nss, ncs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = Qwen35MoESmall().to(torch.bfloat16).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.1f}M")
    print(f"  H={H}, L={L}, NQ={NQ}, NKV={NKV}, DH={DH}")
    print(f"  GDR: LKH={LKH}, LVH={LVH}, LHD={LHD}")
    print(f"  MoE: NE={NE}, TK={TK}, MI={MI}")

    ids = torch.randint(0, 1000, (1, 16), device=device)

    with torch.no_grad():
        logits, kvs, states, convs = model(ids)

    assert logits.shape == (1, 16, VOCAB), f"Unexpected shape: {logits.shape}"
    print(f"Forward pass OK — logits shape: {logits.shape}")
    print(f"  Top-1 predicted next token: {logits[0, -1].argmax().item()}")

    # Second pass with KV/state cache (decode step)
    ids2 = torch.randint(0, 1000, (1, 1), device=device)
    with torch.no_grad():
        logits2, _, _, _ = model(ids2, kvs=kvs, states=states, convs=convs)
    assert logits2.shape == (1, 1, VOCAB)
    print(f"Decode step OK — logits shape: {logits2.shape}")
    print("Qwen3.5-MoE-Small: all checks passed.")