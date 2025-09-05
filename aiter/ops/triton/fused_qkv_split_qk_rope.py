import torch
import triton
import triton.language as tl
from aiter.ops.triton.rope import _get_gptj_rotated_x, _get_neox_rotated_x


@triton.jit
def _fused_qkv_split_qk_rope_kernel(
    qkv_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    T,
    stride_qkv_t,
    stride_qkv_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_h,
    stride_kv_d,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    tl.assume(stride_qkv_t > 0)
    tl.assume(stride_qkv_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_q_t > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_t > 0)
    tl.assume(stride_kv_h > 0)
    tl.assume(stride_kv_d > 0)

    pid_t = tl.program_id(0)
    hq = tl.program_id(1)

    tl.assume(pid_t >= 0)
    tl.assume(hq >= 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    offs_nope_ratio = 1
    if HAVE_NOPE:
        offs_nope_ratio = 2

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        qk_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        qk_rotated_mask = (d_offs % 2 == 0)[None, :]

    H_OFFS_SIZE = hq * BLOCK_D
    d_offs += nope_offs
    q_in_offs = (
        t_offs[:, None] * stride_qkv_t
        + (H_OFFS_SIZE * offs_nope_ratio + d_offs)[None, :] * stride_qkv_d
    )
    q = tl.load(qkv_ptr + q_in_offs, mask=x_mask)

    if IS_NEOX:
        q_rotated = _get_neox_rotated_x(
            q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        q_rotated = _get_gptj_rotated_x(
            q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    q_out_offs = (
        t_offs[:, None] * stride_q_t + d_offs[None, :] * stride_q_d + hq * stride_q_h
    )
    q = q * cos + q_rotated * sin
    q = q.to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + q_out_offs, q, mask=x_mask)

    if HAVE_NOPE:
        if NOPE_FIRST:
            q = tl.load(qkv_ptr + q_in_offs - BLOCK_D * stride_qkv_d, mask=x_mask)
            tl.store(q_ptr + q_out_offs - BLOCK_D * stride_q_d, q, mask=x_mask)
        else:
            q = tl.load(qkv_ptr + q_in_offs + BLOCK_D * stride_qkv_d, mask=x_mask)
            tl.store(q_ptr + q_out_offs + BLOCK_D * stride_q_d, q, mask=x_mask)

    if hq < KVH:
        Q_SIZE = QH * BLOCK_D
        KV_SIZE = KVH * BLOCK_D
        k_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + H_OFFS_SIZE) * offs_nope_ratio + d_offs)[None, :]
            * stride_qkv_d
        )
        v_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_SIZE + H_OFFS_SIZE) * offs_nope_ratio + d_offs)[None, :]
            * stride_qkv_d
        )
        k = tl.load(qkv_ptr + k_in_offs, mask=x_mask)
        v = tl.load(qkv_ptr + v_in_offs, mask=x_mask)

        if IS_NEOX:
            k_rotated = _get_neox_rotated_x(
                k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
        else:
            k_rotated = _get_gptj_rotated_x(
                k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )

        kv_out_offs = (
            t_offs[:, None] * stride_kv_t
            + d_offs[None, :] * stride_kv_d
            + hq * stride_kv_h
        )
        k = k * cos + k_rotated * sin
        k = k.to(k_ptr.dtype.element_ty)
        tl.store(k_ptr + kv_out_offs, k, mask=x_mask)
        v = v.to(v_ptr.dtype.element_ty)
        tl.store(v_ptr + kv_out_offs, v, mask=x_mask)

        if HAVE_NOPE:
            if NOPE_FIRST:
                k = tl.load(qkv_ptr + k_in_offs - BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(k_ptr + kv_out_offs - BLOCK_D * stride_kv_d, k, mask=x_mask)
                v = tl.load(qkv_ptr + v_in_offs - BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(v_ptr + kv_out_offs - BLOCK_D * stride_kv_d, v, mask=x_mask)
            else:
                k = tl.load(qkv_ptr + k_in_offs + BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(k_ptr + kv_out_offs + BLOCK_D * stride_kv_d, k, mask=x_mask)
                v = tl.load(qkv_ptr + v_in_offs + BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(v_ptr + kv_out_offs + BLOCK_D * stride_kv_d, v, mask=x_mask)


def fused_qkv_split_qk_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    nope_first: bool = False,
):
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    assert qh >= kvh and qh % kvh == 0, "qh must be mutiple of kvh"

    q = torch.empty((qkv.shape[0], qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)

    if cos.shape[-1] == head_dim // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == head_dim // 4:
        have_nope = True
    else:
        have_nope = False

    assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim // ((2 if have_nope else 1)) == triton.next_power_of_2(
        head_dim // ((2 if have_nope else 1))
    ), "head_dim should be power of 2"

    if have_nope:
        BLOCK_D = head_dim // 2
        BLOCK_D_HALF = head_dim // 4
    else:
        BLOCK_D = head_dim
        BLOCK_D_HALF = head_dim // 2

    BLOCK_T = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (triton.cdiv(T, BLOCK_T), qh, 1)

    _fused_qkv_split_qk_rope_kernel[grid](
        qkv,
        cos,
        sin,
        positions,
        offsets,
        q,
        k,
        v,
        T,
        *qkv.stride(),
        cos.stride(0),
        cos.stride(-1),
        *positions.stride(),
        *q.stride(),
        *k.stride(),
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        QH=qh,
        KVH=kvh,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return q, k, v
