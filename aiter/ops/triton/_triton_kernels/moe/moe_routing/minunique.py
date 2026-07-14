import triton
import triton.language as tl


@triton.jit
def _keepk_sort0(
    Vin,
    Iin,
    stride_in,  # [M, KP1] candidate weights / expert ids (int16)
    Pop,  # [n_expts_tot] int32 popularity over the (k+1) selection
    Vout,
    Iout,
    stride_out,  # [M, K] compacted kept weights / expert ids
    Hist,  # [n_expts_tot] int32 post-drop histogram (PRE-ZEROED) — atomic
    Part,
    stride_pm,
    stride_pn,  # [NUM_BLOCKS, n_expts_tot] cross-block prefix (PRE-ZEROED) — atomic
    n_rows,
    n_expts_tot,
    HIST_BLOCK_M: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    KP1: tl.constexpr,
    K: tl.constexpr,
    KP1_PAD: tl.constexpr,
    APPLY_SOFTMAX: tl.constexpr,
    APPLY_RENORM: tl.constexpr = False,
    ROUTED_SCALING: tl.constexpr = 1.0,
):
    tl.static_assert(
        not (APPLY_SOFTMAX and APPLY_RENORM),
        "APPLY_SOFTMAX and APPLY_RENORM are mutually exclusive",
    )
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    blk = pid // HIST_BLOCK_M
    cols = tl.arange(0, KP1_PAD)
    m = cols < KP1
    idx = tl.load(Iin + pid * stride_in + cols, mask=m, other=0).to(tl.int32)
    val = tl.load(Vin + pid * stride_in + cols, mask=m, other=0.0).to(tl.float32)
    safe_idx = tl.where(m, idx, 0)
    cp = tl.load(Pop + safe_idx, mask=m, other=0).to(tl.float32)

    # drop = argmin over (popularity, then logit, then column); tl.max-only so it
    # cannot mis-codegen under HIP graph capture, NaN/inf-sanitized + clamped.
    negpop = tl.where(m, -cp, -3.0e38)
    maxnp = tl.max(negpop, axis=0)
    is_mp = negpop == maxnp
    finite = (val == val) & (val < 3.0e38) & (val > -3.0e38)
    valc = tl.where(finite, val, -3.0e38)
    negval = tl.where(is_mp, -valc, -3.0e38)
    maxnv = tl.max(negval, axis=0)
    is_drop = is_mp & (negval == maxnv)
    colf = cols.to(tl.float32)
    dropf = tl.max(tl.where(is_drop, -colf, -3.0e38), axis=0)
    drop = (-dropf).to(tl.int32)
    drop = tl.where((drop >= 0) & (drop < KP1), drop, KP1 - 1)
    keep = m & (cols != drop)

    keep_i = keep.to(tl.int32)
    outpos = tl.cumsum(keep_i, axis=0) - keep_i
    if APPLY_SOFTMAX:
        neg = tl.where(keep, val, -3.0e38)
        vmax = tl.max(neg, axis=0)
        ex = tl.where(keep, tl.exp(val - vmax), 0.0)
        w = ex / tl.sum(ex, axis=0)
    elif APPLY_RENORM:
        w_f = tl.where(keep, val, 0.0)
        s = tl.sum(w_f, axis=0)
        w = w_f / (s + 1e-20) * ROUTED_SCALING
    elif ROUTED_SCALING != 1.0:
        w = val * ROUTED_SCALING
    else:
        w = val
    tl.store(Iout + pid * stride_out + outpos, idx.to(tl.int16), mask=keep)
    tl.store(Vout + pid * stride_out + outpos, w.to(Vout.dtype.element_ty), mask=keep)

    # post-drop histogram (atomic)
    tl.atomic_add(Hist + safe_idx, 1, mask=keep, sem="relaxed")
    # cross-block prefix: each token carries +1 to every LATER block's offsets
    # (this is exactly _sum_bitmatrix_rows' carry, done per-token with atomics).
    for bp in tl.static_range(NUM_BLOCKS):
        tl.atomic_add(
            Part + bp * stride_pm + safe_idx * stride_pn,
            1,
            mask=keep & (bp > blk),
            sem="relaxed",
        )
