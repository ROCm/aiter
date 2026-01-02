import triton
import triton.language as tl
import torch
import math
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def print_irs_to_files(compiled_kernel, prefix):
    for key in compiled_kernel.asm.keys():
        with open(f"{prefix}_{key}.txt", "w") as fptr:
            print(compiled_kernel.asm[key], file=fptr)


@gluon.jit
def fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    # strides
    stride_q_s: gl.int32,
    stride_q_h: gl.constexpr,
    stride_q_d: gl.constexpr,
    stride_kv_s: gl.int32,
    stride_kv_d: gl.constexpr,
    stride_w_s: gl.int32,
    stride_w_h: gl.constexpr,
    stride_logits_s: gl.int64,
    stride_logits_k: gl.int64,
    # block sizes
    BLOCK_KV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PRESHUFFLE_KV: gl.constexpr,
):
    row_id = gl.program_id(axis=0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = gl.num_programs(0) - row_id - 1
    # row_id = remap_xcd(row_id, gl.num_programs(0), 8)
    NUM_THREADS: gl.constexpr = 64

    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    layout_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    layout_scales: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[NUM_WARPS],
        order=[0],
    )
    # TODO (cagri): instr_shape should be [32, 32, 16] for newer triton versions
    # the kernel needs to be updated to support this based on the version of triton
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[
            32,
            32,
        ],  # V_MFMA_F32_32x32x16_FP8_FP8 instruction
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    # Read and clamp start/end indices for this row
    start_ind = gl.load(cu_start_ptr + row_id)
    end_ind = gl.load(cu_end_ptr + row_id)
    start_ind = gl.maximum(start_ind, 0)
    end_ind = gl.minimum(end_ind, seq_len_kv)

    # Prepare KV block load offsets
    if PRESHUFFLE_KV:
        # TODO (cagri): double check if this is correct
        kv_block_offsets = (
            gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, dot_b_layout)) % 16
            + gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, dot_b_layout))
            // 16
            * 256
        )[:, None] + (
            gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, dot_b_layout)) % 16 * 16
            + gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, dot_b_layout))
            // 16
            * 16
            * HEAD_SIZE
        )[
            None, :
        ]
        kv_inds_ld = (gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, dot_b_layout)))[
            None, :
        ] + start_ind
    else:
        kv_block_offsets = (
            gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, layout_kv)) * stride_kv_d
        )[:, None] + (
            (gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, layout_kv)) + start_ind)
            * stride_kv_s
        )[
            None, :
        ]
        kv_inds_ld = (gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, layout_kv)))[
            None, :
        ] + start_ind

    kv_block = gl.amd.cdna3.buffer_load(
        ptr=KV_ptr,
        mask=kv_inds_ld < end_ind,
        offsets=kv_block_offsets,
    )
    mfma_k = gl.convert_layout(kv_block, dot_b_layout)
    kv_block_offsets += BLOCK_KV * stride_kv_s
    kv_inds_ld += BLOCK_KV

    kv_scale_offsets = (
        gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout)) + start_ind
    )[None, :]

    # load Q[NUM_HEADS, HEAD_SIZE]
    q_block = gl.amd.cdna3.buffer_load(
        ptr=Q_ptr,
        offsets=row_id * stride_q_s
        + (gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, layout_q)) * stride_q_h)[
            :, None
        ]
        + (gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q)) * stride_q_d)[
            None, :
        ],
        cache=".cg",
    )

    w_block = gl.amd.cdna3.buffer_load(
        ptr=weights_ptr,
        offsets=row_id * stride_w_s
        + (gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout)) * stride_w_h)[
            :, None
        ],
        cache=".cg",
    )

    zero = gl.zeros((NUM_HEADS, BLOCK_KV), dtype=tl.float32, layout=mfma_layout)
    mfma_q = gl.convert_layout(q_block, dot_a_layout)

    logits_offsets = row_id * stride_logits_s + (
        (gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout)) + start_ind)
        * stride_logits_k
    )
    kv_inds = (
        gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
    ) + start_ind

    # ------ MASKING FOR PRE-LOOP STORE ------
    # Loop over full KV tiles
    tile_start = start_ind
    tile_end = end_ind
    num_full_tiles = (tile_end - tile_start) // BLOCK_KV

    for _ in tl.range(0, num_full_tiles - 1):
        kv_block_nxt = gl.amd.cdna3.buffer_load(
            ptr=KV_ptr,
            # mask=kv_inds_ld < end_ind,
            offsets=kv_block_offsets,
        )

        kv_scales = gl.amd.cdna3.buffer_load(
            ptr=kv_scales_ptr,
            # mask=kv_inds[None, :] < end_ind,
            offsets=kv_scale_offsets,
        )
        scores = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        scores = scores * kv_scales
        scores = gl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = gl.sum(scores, axis=0)
        # mask = (kv_inds < end_ind)
        gl.store(
            logits_ptr + logits_offsets,
            scores,
            # mask=mask,
        )

        kv_scale_offsets += BLOCK_KV
        kv_block_offsets += BLOCK_KV * stride_kv_s
        logits_offsets += BLOCK_KV * stride_logits_k
        kv_inds += BLOCK_KV
        kv_inds_ld += BLOCK_KV

        mfma_k = gl.convert_layout(kv_block_nxt, dot_b_layout)

    # 2 more mfma, 1 from loop peeling, 1 from the staging
    kv_block_nxt = gl.amd.cdna3.buffer_load(
        ptr=KV_ptr,
        mask=kv_inds_ld < end_ind,
        offsets=kv_block_offsets,
    )
    kv_scales = gl.amd.cdna3.buffer_load(
        ptr=kv_scales_ptr,
        offsets=kv_scale_offsets,
        mask=kv_inds[None, :] < end_ind,
    )
    scores = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
    scores = scores * kv_scales
    scores = gl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = gl.sum(scores, axis=0)
    mask = kv_inds < end_ind
    gl.store(
        logits_ptr + logits_offsets,
        scores,
        mask=mask,
    )

    kv_scale_offsets += BLOCK_KV
    logits_offsets += BLOCK_KV * stride_logits_k
    kv_inds += BLOCK_KV
    kv_inds_ld += BLOCK_KV

    kv_scales = gl.amd.cdna3.buffer_load(
        ptr=kv_scales_ptr,
        offsets=kv_scale_offsets,
        mask=kv_inds[None, :] < end_ind,
    )

    mfma_k = gl.convert_layout(kv_block_nxt, dot_b_layout)
    scores = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
    scores = scores * kv_scales
    scores = gl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = gl.sum(scores, axis=0)
    mask = kv_inds < end_ind
    gl.store(
        logits_ptr + logits_offsets,
        scores,
        mask=mask,
    )


def fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends, preshuffle_kv=False):
    """
    Q:           [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:          [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:   [seq_len_kv], dtype float32
    weights:     [seq_len, NUM_HEADS], dtype float32
    cu_starts:   [seq_len], dtype int32
    cu_ends:     [seq_len], dtype int32

    Returns:
    logits:      [seq_len, seq_len_kv], dtype float32 (must be initialized to -inf, because of causal masking)
    """
    # TODO (cagri): double check what value to put for causally masked logits, 0 or -inf?
    # TODO (cagri): Tune/optimize
    BLOCK_KV = 128
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    # TODO (cagri): Currently assuming num_heads and head_size is power of 2.
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    # Initialize with -inf because of causal masking
    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=Q.device,
    )

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    matrix_instr_nonkdim = 32
    if seq_len <= 1024:
        matrix_instr_nonkdim = 16

    k = fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=Q,
        KV_ptr=KV,
        kv_scales_ptr=kv_scales,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=BLOCK_KV,
        NUM_WARPS=4,
        num_warps=4,
        num_stages=1,
        waves_per_eu=4,
        PRESHUFFLE_KV=preshuffle_kv,
        # matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    if getattr(fp8_mqa_logits, "print", True):
        setattr(fp8_mqa_logits, "print", False)
        print_irs_to_files(k, "fp8_mqa_logits_single_q_3_5_gluon")
    return logits
