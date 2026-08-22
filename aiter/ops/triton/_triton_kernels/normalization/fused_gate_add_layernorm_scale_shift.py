# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Fused adaLN-gate block op for DiT-style transformers (e.g. SD3, SD3.5).
# One launch computes, per row:
#     h   = x + gate * attn        (gated residual, written back)
#     out = LayerNorm(h) * (1 + scale) + shift   (adaLN modulation)
# gate/scale/shift are per-modulation-group (M, D) broadcast across the rows
# of each group (a group = all tokens sharing one timestep embedding).
import triton
import triton.language as tl


@triton.jit
def _fused_gate_add_layernorm_scale_shift_kernel(
    x_ptr,
    attn_ptr,
    gate_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    out_res_ptr,
    eps,
    M,
    N,
    ROWS_PER_GROUP: tl.constexpr,
    x_stride_m,
    attn_stride_m,
    mod_stride_m,
    out_stride_m,
    out_res_stride_m,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    tl.assume(pid_m >= 0)
    gid = pid_m // ROWS_PER_GROUP  # modulation group index for this row

    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < N

    x = tl.load(
        x_ptr + pid_m * x_stride_m + n_offs, mask=mask, other=0.0, cache_modifier=".cg"
    ).to(tl.float32)
    a = tl.load(
        attn_ptr + pid_m * attn_stride_m + n_offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    g = tl.load(
        gate_ptr + gid * mod_stride_m + n_offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    h = x + g * a

    # LayerNorm (no learned affine) over the row, fp32 accumulation
    hm = tl.where(mask, h, 0.0)
    mean = tl.sum(hm, axis=-1) / N
    hc = tl.where(mask, h - mean, 0.0)
    var = tl.sum(hc * hc, axis=-1) / N
    hn = hc * tl.rsqrt(var + eps)

    sc = tl.load(
        scale_ptr + gid * mod_stride_m + n_offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    sh = tl.load(
        shift_ptr + gid * mod_stride_m + n_offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    tl.store(
        out_res_ptr + pid_m * out_res_stride_m + n_offs,
        h.to(out_res_ptr.dtype.element_ty),
        mask=mask,
    )
    out = (hn * (1.0 + sc) + sh).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + pid_m * out_stride_m + n_offs, out, mask=mask)
