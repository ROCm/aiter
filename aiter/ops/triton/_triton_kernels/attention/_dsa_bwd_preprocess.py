import torch
import triton
import triton.language as tl


# =====================================================================
# Backward — preprocess kernel (Delta computation)
# =====================================================================
@triton.jit
def _sparse_mla_bwd_preprocess(
    O_ptr,          # [total_tokens, num_heads, D_V]
    dO_ptr,         # [total_tokens, num_heads, D_V]
    Delta_ptr,      # [total_tokens, num_heads]
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    num_heads: tl.int32,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Delta[t, h] = sum_d(O[t, h, d] * dO[t, h, d])

    Grid: (total_tokens, cdiv(num_heads, BLOCK_H))
    """
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)

    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_d = tl.arange(0, D_V)

    base = token_idx * stride_o_t

    O = tl.load(
        O_ptr + base + offs_h[:, None] * stride_o_h + offs_d[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    dO = tl.load(
        dO_ptr + base + offs_h[:, None] * stride_o_h + offs_d[None, :],
        mask=mask_h[:, None], other=0.0,
    )

    delta = tl.sum(O.to(tl.float32) * dO.to(tl.float32), axis=1)

    tl.store(
        Delta_ptr + token_idx * num_heads + offs_h,
        delta, mask=mask_h,
    )
