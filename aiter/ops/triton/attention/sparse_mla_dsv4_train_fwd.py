"""
DeepSeek DSv4 sparse-MLA TRAINING forward — public entry (returns O and sink-inclusive LSE).

Companion to the inference `sparse_attn_prefill` (sparse_attention_dsv4.py). The training path takes
DENSE `topk_indices[T, TOPK]` and emits `lse[T, H]` (needed by the backward: P_j = exp(S_j - LSE)).

Backend dispatch (mirrors sparse_attn_prefill):
  * gfx950 + Triton >= 3.6  -> gluon `mla_gluon(has_pe=False, return_lse=True)`  (fastest on CDNA4)
  * otherwise (incl. gfx942) -> the dense Triton kernel in
    `_triton_kernels/attention/sparse_mla_dsv4_train_fwd.py`

V4 contract: q/kv are `[.., 512]` with RoPE already applied in-place caller-side; K == V == kv;
scale defaults to 1/sqrt(512); attn_sink folds into the softmax denominator only. Returns O
pre-un-rotation (the caller un-rotates the trailing 64) and the sink-inclusive natural-log LSE.
"""
import torch
import triton
from packaging.version import Version

from aiter.ops.triton._triton_kernels.attention.sparse_mla_dsv4_train_fwd import (
    sparse_mla_fwd_v4_triton,
)
from aiter.ops.triton.utils._triton import arch_info

# Gluon (CDNA4) path — opt-in, gated on Triton >= 3.6 + arch=gfx950 (same gate as sparse_attn_prefill).
_TRITON_GE_36 = Version(triton.__version__) >= Version("3.6.0")
_gluon_mla = None
if _TRITON_GE_36 and arch_info.get_arch() == "gfx950":
    try:
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon as _gluon_mla
    except Exception:
        _gluon_mla = None


def sparse_mla_fwd_v4_train(q, kv, topk_indices, attn_sink=None, scale=None, backend="auto"):
    """DSv4 sparse-MLA training forward.

    Args:
        q:            [T, H, 512] bf16 (RoPE in-place).
        kv:           [T, 512] or [T, 1, 512] bf16 (K == V).
        topk_indices: [T, TOPK] int32, dense; -1 marks invalid slots.
        attn_sink:    [H] fp32, optional per-head sink logit (denominator only).
        scale:        float, default 1/sqrt(512).
        backend:      "auto" (gluon on gfx950 else triton), "gluon", or "triton".

    Returns:
        o:   [T, H, 512] q.dtype (pre-un-rotation).
        lse: [T, H] fp32, sink-inclusive.
    """
    T, H, D = q.shape
    assert D == 512, f"DSv4 training fwd expects head_dim 512, got {D}"
    TOPK = topk_indices.shape[1]
    if scale is None:
        scale = D ** -0.5

    use_gluon = _gluon_mla is not None if backend == "auto" else (backend == "gluon")
    if use_gluon:
        assert _gluon_mla is not None, "gluon backend unavailable (needs Triton>=3.6 + gfx950)"
        kvc = (kv[:, 0, :] if kv.dim() == 3 else kv).contiguous()
        o = torch.empty(T, H, D, dtype=q.dtype, device=q.device)
        ki = topk_indices.reshape(-1).to(torch.int32).contiguous()
        kp = torch.arange(0, (T + 1) * TOPK, TOPK, dtype=torch.int32, device=q.device)
        o, lse = _gluon_mla(
            q.contiguous(), kvc, kvc, o, ki, kp, float(scale),
            has_pe=False, min_kv_seq_len=TOPK, return_lse=True, attn_sink=attn_sink,
        )
        return o, lse

    return sparse_mla_fwd_v4_triton(q, kv, topk_indices, attn_sink=attn_sink,
                                    kv_lora_rank=D, scale=scale)
