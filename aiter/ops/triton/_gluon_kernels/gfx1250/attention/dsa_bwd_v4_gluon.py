"""
End-to-end gluon backward for DeepSeek V4 sparse MLA — gfx1250 / MI450 (M4).

Mirror of the gfx950 `dsa_bwd_v4_gluon.py`, using the gfx1250 gluon kernels:
Triton preprocess (Delta) + gluon dQ (gfx1250) + gluon dKV-intermediate (gfx1250) +
Triton gather (CSR inverted-topk) + torch d_sink (no atomics).

STATUS (2026-06-22): dQ kernel VALIDATED (median_rel 2.5e-3). dKV-interm + this
end-to-end dispatch are CODE-COMPLETE but **NOT yet validated** — the gfx1250 A0 node
went unreachable during dkv-interm validation. Validate before wiring into
`sparse_mla_bwd_v4(backend="gluon")` / committing to the PR.

Tilings: dQ BLOCK_H=64/TILE_K=32 (WMMA K), dKV BLOCK_H=32/TILE_K=64.
"""
import torch
import triton

from .dsa_bwd_dq import _sparse_mla_bwd_dq_gl_kernel_gfx1250
from .dsa_bwd_dkv_interm import _sparse_mla_bwd_dkv_interm_gl_kernel_gfx1250

from aiter.ops.triton._triton_kernels.attention._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess
from aiter.ops.triton._triton_kernels.attention._dsa_bwd_gather import (
    _build_inverted_topk_slice, _bwd_dkv_gather_acc,
)


def sparse_mla_bwd_v4_gluon_gfx1250(q, kv, o, do, topk_indices, lse, attn_sink=None,
                                    kv_lora_rank=512, scale=None):
    assert q.is_contiguous() and kv.is_contiguous() and o.is_contiguous()
    assert do.is_contiguous() and topk_indices.is_contiguous() and lse.is_contiguous()
    total, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # preprocess: Delta = rowsum(O*dO) (Triton)
    delta = torch.empty(total, num_heads, dtype=torch.float32, device=q.device)
    BH_PRE = triton.next_power_of_2(min(64, num_heads))
    _sparse_mla_bwd_preprocess[(total, triton.cdiv(num_heads, BH_PRE))](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=BH_PRE)

    R_CHUNK = min(256, topk)
    BH_DQ, TK_DQ = 64, 32           # gfx1250: TILE_K=32 (WMMA K-dim)
    BH_DKV, TK_DKV = 32, 64
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(total, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    padlen = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    topk_p = torch.cat([topk_indices, torch.full((total, padlen - topk), -1, dtype=torch.int32, device=q.device)], 1).contiguous() if padlen != topk else topk_indices
    all_csr = [_build_inverted_topk_slice(topk_p[:, rs:rs + R_CHUNK], rs, R_CHUNK) for rs in range(0, topk, R_CHUNK)]

    for ci, r_start in enumerate(range(0, topk, R_CHUNK)):
        _sparse_mla_bwd_dq_gl_kernel_gfx1250[(total, num_hg_dq)](
            q, kv, do, topk_p, lse, delta,
            dq, chunk_dS, chunk_P,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_p.stride(0),
            chunk_dS.stride(0), chunk_dS.stride(1),
            scale, num_heads, r_start,
            R_CHUNK=R_CHUNK, BLOCK_H=BH_DQ, TILE_K=TK_DQ,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            IS_FIRST_CHUNK=(r_start == 0), num_warps=4, waves_per_eu=1)

        _sparse_mla_bwd_dkv_interm_gl_kernel_gfx1250[(total,)](
            q, do, chunk_dS, chunk_P, interm,
            q.stride(0), q.stride(1), do.stride(0), do.stride(1),
            chunk_dS.stride(0), chunk_dS.stride(1),
            interm.stride(0), interm.stride(1),
            num_heads,
            R_CHUNK=R_CHUNK, TILE_K=TK_DKV, BLOCK_H=BH_DKV,
            NUM_HG=num_hg_dkv, D_V=kv_lora_rank, D_ROPE=rope_rank, num_warps=4)

        inv_ptr, inv_data = all_csr[ci]
        _bwd_dkv_gather_acc[(total,)](
            interm, inv_ptr, inv_data, dkv_acc,
            interm.stride(1), dkv_acc.stride(0),
            D_V=kv_lora_rank, D_ROPE=rope_rank, num_warps=4)

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)
    return dq, dkv_acc.to(kv.dtype).unsqueeze(1), d_sink
