import functools
import torch
import triton
import aiter
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    map_dims,
)
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    sage_quant_v_kernel,
    sage_quant_kernel,
    _rot_k_only_kernel,
    _rot_q_kernel,
    _rotate_quantize_q_kernel,
    _rotate_quantize_k_kernel,
    _compute_delta_s_kernel,
)

from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp


def fused_sage_quant_mxfp4(
    q,
    k,
    v,
    BLOCK_M,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    q_smoothing=False,
    layout="bshd",
):

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")

    # padded_head_dim = max(16, 1 << (head_dim - 1).bit_length())
    sm_scale = head_dim**-0.5

    q_fp4, q_scale, k_fp4, k_scale, delta_s = smooth_rotate_downcast_qk(
        q,
        k,
        BLOCK_SIZE_M=BLOCK_M,
        hadamard_rotation=hadamard_rotation,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(FP8_TYPE).max
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    BLOCK_K = 1024
    K_NUM_BLKS = (kv_len + BLOCK_K - 1) // BLOCK_K

    # V tensor per channel quantization
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)
    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_d_v,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLOCK_K,
        num_stages=5,
        num_warps=8,
    )

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


def sage_quant_mxfp4(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    USE_RNE=False,
    R=None,
    BLOCK_R=32,
):
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    # Apply K tensor smoothing following SageAttention approach
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)

    # padded_head_dim = max(16, 1 << (head_dim - 1).bit_length())

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_d_v,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    downcast_func = downcast_to_mxfp

    q_fp4, q_scale = downcast_func(q, torch.uint8, axis=-1)
    k_fp4, k_scale = downcast_func(k, torch.uint8, axis=-1)

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


def sage_quant(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    layout="bshd",
    smooth_k=True,
):
    """
    Quantize Q and K tensors to INT8 with per-block scaling.

    Args:
        q: Query tensor
        k: Key tensor
        km: Optional pre-computed K smoothing factors (if None and smooth_k=True, will be computed)
        BLKQ: Block size for Q quantization
        BLKK: Block size for K quantization
        sm_scale: Softmax scale factor (defaults to head_dim^-0.5)
        layout: Either "bshd" or "bhsd"
        smooth_k: Whether to apply SageAttention-style smoothing to K tensor (default: True)

    Returns:
        q_int8: Quantized Q tensor
        q_scale: Per-block scales for Q
        k_int8: Quantized K tensor
        k_scale: Per-block scales for K
        k_smooth: K smoothing factors applied (or None if smooth_k=False)
    """
    q_int8 = torch.empty_like(q, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty_like(k, dtype=torch.int8, device=k.device)
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    Q_NUM_BLKS = (qo_len + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    # Apply K tensor smoothing following SageAttention approach
    if smooth_k:
        k = k - k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    q_scale = torch.empty((b, h_qo, Q_NUM_BLKS), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, K_NUM_BLKS), device=q.device, dtype=torch.float32)

    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q_task_count = b * h_qo * Q_NUM_BLKS
    k_task_count = b * h_kv * K_NUM_BLKS
    v_task_count = b * h_kv * K_NUM_BLKS

    grid = (q_task_count + k_task_count + v_task_count,)

    # call sage_quant_kernel
    sage_quant_kernel[grid](
        q,
        q_int8,
        q_scale,
        k,
        k_int8,
        k_scale,
        v,
        v_fp8,
        v_scale,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        (sm_scale * 1.4426950408889634),
        q_task_count,
        k_task_count,
        b,
        h_qo,
        h_kv,
        Q_NUM_BLKS,
        K_NUM_BLKS,
        qo_len,
        kv_len,
        triton.next_power_of_2(kv_len),
        FP8_MAX=FP8_MAX,
        INT8_MAX=torch.iinfo(q_int8.dtype).max,
        D=head_dim,
        BLK_Q=BLKQ,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    return q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale


def rotation_smooth_qk(
    q,
    k,
    BLOCK_SIZE_M,
    R=None,
    BLOCK_R=32,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
):

    if R is None:  # Generate Hadamard Matrix R if not given
        assert (
            BLOCK_R is not None
        ), "if not passing R (hadamard matrix), BLOCK_R (size of the hadamard matrix) must be provided."
        R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (BLOCK_R**0.5)
    else:
        BLOCK_R = R.shape[-1]

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_rot = torch.empty_like(q)
    K_rot = torch.empty_like(k)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_qob, stride_qom, stride_qoh, stride_qod = map_dims(Q_rot.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)
    stride_kob, stride_kon, stride_koh, stride_kod = map_dims(K_rot.stride(), bshd)
    # rotate q and optionally smooth
    grid_q = (b * h_q, Q_NUM_BLKS, d // BLOCK_R)
    _rot_q_kernel[grid_q](
        q,
        Q_rot,
        q_mean,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qob,
        stride_qoh,
        stride_qom,
        stride_qod,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        R.stride(0),
        R.stride(1),
        h_q,
        s_q,
        d,
        q_smoothing=q_smoothing,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=BLOCK_R,
    )

    # rotate k
    grid_k = (b * h_k, K_NUM_BLKS, d // BLOCK_R)
    _rot_k_only_kernel[grid_k](
        k,
        K_rot,
        R,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_kob,
        stride_koh,
        stride_kon,
        stride_kod,
        R.stride(0),
        R.stride(1),
        h_k,
        s_k,
        d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=BLOCK_R,
    )

    # smooth k
    K_rot = K_rot - K_rot.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    if q_smoothing:
        # compute delta s that needs to be added due to q smoothing
        # Q x K = Q x H x H.T x K
        # = ((Q x H - q_mean + q_mean) x H.T x K
        # = Q_rot x K_rot + q_mean x K_rot
        # = Q_rot x K_rot + delta_s
        grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            K_rot,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_q,
            h_k,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_rot, K_rot, delta_s


def smooth_rotate_downcast_qk(
    q,
    k,
    BLOCK_SIZE_M,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
):
    if hadamard_rotation:
        if R is None:
            assert (
                BLOCK_R is not None
            ), "if using hadamard rotation, BLOCK_R (size of the hadamard matrix) must be provided."
            R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (BLOCK_R**0.5)
        else:
            BLOCK_R = R.shape[-1]

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)

    Q_q = torch.empty((*q.shape[:-1], d // 2), dtype=torch.uint8, device=q.device)
    Q_descale = torch.empty(
        (*q.shape[:-1], d // 32), dtype=torch.uint8, device=q.device
    )
    K_q = torch.empty((*k.shape[:-1], d // 2), dtype=torch.uint8, device=k.device)
    K_descale = torch.empty(
        (*k.shape[:-1], d // 32), dtype=torch.uint8, device=k.device
    )

    stride_qqb, stride_qqm, stride_qqh, stride_qqd = map_dims(Q_q.stride(), bshd)
    stride_kqb, stride_kqn, stride_kqh, stride_kqd = map_dims(K_q.stride(), bshd)

    stride_qsb, stride_qsm, stride_qsh, stride_qsd = map_dims(Q_descale.stride(), bshd)
    stride_ksb, stride_ksn, stride_ksh, stride_ksd = map_dims(K_descale.stride(), bshd)

    grid_q = (b * h_q * Q_NUM_BLKS,)
    _rotate_quantize_q_kernel[grid_q](
        q,
        Q_q,
        Q_descale,
        q_mean,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qqb,
        stride_qqm,
        stride_qqh,
        stride_qqd,
        stride_qsb,
        stride_qsm,
        stride_qsh,
        stride_qsd,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        b,
        h_q,
        s_q,
        d,
        q_smoothing=q_smoothing,
        hadamard_rotation=hadamard_rotation,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    grid_k = (b * h_k * K_NUM_BLKS,)
    _rotate_quantize_k_kernel[grid_k](
        q,
        Q_q,
        Q_descale,
        q_mean,
        k,
        K_q,
        K_descale,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qqb,
        stride_qqm,
        stride_qqh,
        stride_qqd,
        stride_qsb,
        stride_qsm,
        stride_qsh,
        stride_qsd,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_kqb,
        stride_kqn,
        stride_kqh,
        stride_kqd,
        stride_ksb,
        stride_ksn,
        stride_ksh,
        stride_ksd,
        b,
        h_q,
        h_k,
        s_q,
        s_k,
        d,
        q_smoothing=q_smoothing,
        hadamard_rotation=hadamard_rotation,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    if q_smoothing:
        # 3. Compute Smoothing Delta S
        # Grid: Each Q-block x Each K-block
        grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            k,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_k,
            h_q,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_q, Q_descale, K_q, K_descale, delta_s


@functools.lru_cache(maxsize=16)
def create_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    """
    Returns a Hadamard matrix of size block_size x block_size. Remember to normalize with sqrt(block_size) for it to be orthogonal.
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half

    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # remember to divide by sqrt(block_size) to get orthogonal matrix
    return H


def create_random_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    # 1. Generate the deterministic Hadamard matrix (H)
    H = create_hadamard_matrix(block_size, device=device, dtype=dtype) / (
        block_size**0.5
    )
    # 2. Create the random diagonal matrix D (represented as a vector for efficiency)
    # This generates random +1 or -1 for each column
    random_signs = (
        torch.randint(0, 2, (block_size,), device=device, dtype=torch.int) * 2 - 1
    )
    # 3. Apply the random signs (H @ D)
    # Multiplying by a diagonal matrix on the right is equivalent to scaling columns
    H_tilde = H * random_signs
    return H_tilde
