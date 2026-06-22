"""
End-to-end gluon backward for DeepSeek V4 sparse MLA (M4).

Wires: Triton preprocess (Delta) + gluon dQ + gluon dKV-intermediate +
Triton gather (CSR inverted-topk) + torch d_sink reduction.

Drop-in for `sparse_mla_bwd_v4` (chunked_gather). Differences vs the Triton path:
  - dQ kernel = gluon `_sparse_mla_bwd_dq_gl_kernel` (BH=64, TK=16)
  - dKV-interm kernel = gluon `_sparse_mla_bwd_dkv_interm_gl_kernel` (BH=32, TK=64),
    takes UNtransposed q/do -> the `q.transpose(1,2).contiguous()` copies are GONE.
  - d_sink computed in torch (no in-kernel atomics).
  - gather + preprocess unchanged (Triton).
"""
import torch
import triton

from .dsa_bwd_dq import _sparse_mla_bwd_dq_gl_kernel
from .dsa_bwd_dkv_interm import _sparse_mla_bwd_dkv_interm_gl_kernel

from aiter.ops.triton._triton_kernels.attention._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess
from aiter.ops.triton._triton_kernels.attention._dsa_bwd_gather import (
    _build_inverted_topk_slice, _bwd_dkv_gather_acc,
)


def sparse_mla_bwd_v4_gluon(q, kv, o, do, topk_indices, lse, attn_sink=None,
                            kv_lora_rank=512, scale=None):
    assert q.is_contiguous() and kv.is_contiguous() and o.is_contiguous()
    assert do.is_contiguous() and topk_indices.is_contiguous() and lse.is_contiguous()

    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # ---- preprocess: Delta = rowsum(O*dO) (Triton, unchanged) ----
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)
    BLOCK_H_PRE = triton.next_power_of_2(min(64, num_heads))
    _sparse_mla_bwd_preprocess[(total_tokens, triton.cdiv(num_heads, BLOCK_H_PRE))](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=BLOCK_H_PRE,
    )

    # ---- config ----
    R_CHUNK = min(256, topk)
    BH_DQ, TK_DQ = 64, 16
    BH_DKV, TK_DKV = 32, 64
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(total_tokens, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    # ---- pad topk to R_CHUNK multiple ----
    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = torch.full((total_tokens, topk_padded_len - topk), -1, dtype=torch.int32, device=q.device)
        topk_padded = torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_padded = topk_indices

    all_csr = [_build_inverted_topk_slice(topk_padded[:, rs:rs + R_CHUNK], rs, R_CHUNK)
               for rs in range(0, topk, R_CHUNK)]

    for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
        is_first = (r_start == 0)

        # gluon dQ (writes dq RMW, chunk_dS, chunk_P)
        _sparse_mla_bwd_dq_gl_kernel[(total_tokens, num_hg_dq)](
            q, kv, do, topk_padded, lse, delta,
            dq, chunk_dS, chunk_P,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_padded.stride(0),
            chunk_dS.stride(0), chunk_dS.stride(1),
            scale, num_heads, r_start,
            R_CHUNK=R_CHUNK, BLOCK_H=BH_DQ, TILE_K=TK_DQ,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            IS_FIRST_CHUNK=is_first, num_warps=4,
            waves_per_eu=1,  # sweep: 1 is best (+3-7%); >=2 spills catastrophically
        )

        # gluon dKV-intermediate (untransposed q/do -> no external transpose)
        _sparse_mla_bwd_dkv_interm_gl_kernel[(total_tokens,)](
            q, do, chunk_dS, chunk_P, interm,
            q.stride(0), q.stride(1), do.stride(0), do.stride(1),
            chunk_dS.stride(0), chunk_dS.stride(1),
            interm.stride(0), interm.stride(1),
            num_heads,
            R_CHUNK=R_CHUNK, TILE_K=TK_DKV, BLOCK_H=BH_DKV,
            NUM_HG=num_hg_dkv, D_V=kv_lora_rank, D_ROPE=rope_rank, num_warps=4,
        )

        # Triton gather (CSR inverted-topk reduce interm -> dkv_acc)
        inv_ptr, inv_data = all_csr[chunk_idx]
        _bwd_dkv_gather_acc[(total_tokens,)](
            interm, inv_ptr, inv_data, dkv_acc,
            interm.stride(1), dkv_acc.stride(0),
            D_V=kv_lora_rank, D_ROPE=rope_rank, num_warps=4,
        )

    # ---- d_sink in torch: -sum_t exp(sink - lse) * delta ----
    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv_out = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv_out, d_sink
