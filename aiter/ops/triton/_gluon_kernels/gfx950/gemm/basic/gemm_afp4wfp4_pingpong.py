"""gfx950 Gluon MXFP4 GEMM with an 8-wave ping-pong schedule ("compute_bound" kernel_type).

Eight waves compute the four 128x128 quadrants of a 256x256 output tile. Each quadrant region
has an MFMA stage and a memory stage, and the two wave groups run one stage apart, so the MFMAs
of one group overlap the LDS reads and the direct-to-LDS copies of the other group.
"""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import (
    AMDMFMALayout,
    cdna4,
    warp_pipeline_stage,
)
from triton.experimental.gluon.language.amd.cdna4 import async_copy as acp

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gemm_afp4wfp4_pingpong_repr = make_kernel_repr(
    "_gemm_afp4wfp4_pingpong_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "NUM_WARPS",
    ],
)


@gluon.constexpr_function
def _d2l_layout(padded, num_warps, vec_bytes, warp_size=64):
    """Direct-to-LDS copy layout for a padded LDS destination, as in Triton's
    CoalesceAsyncCopy: one 16-B run per lane, one contiguous LDS run per wave."""
    bases = [list(b) for b in padded.offset_bases]
    rank = len(padded.shape)
    nv = int(math.log2(vec_bytes))
    nl = int(math.log2(warp_size))
    nw = int(math.log2(num_warps))
    reg = bases[:nv]
    lane = bases[nv : nv + nl]
    warp = bases[nv + nl : nv + nl + nw]
    warp = warp + [[0] * rank for _ in range(nw - len(warp))]
    reg = reg + bases[nv + nl + nw :]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=list(padded.shape),
    )


@gluon.constexpr_function
def _mma_layout(num_warps):
    return AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[2, num_warps // 2],
    )


@gluon.constexpr_function
def _dot_layout(operand_index, num_warps):
    return gl.DotOperandLayout(
        operand_index=operand_index, parent=_mma_layout(num_warps), k_width=16
    )


@gluon.constexpr_function
def _half_smem_layout(operand_index, num_warps):
    """Padded LDS layout of one A half [128, 128 B] or one B half [128 B, 128]."""
    shape = [128, 128]
    return cdna4.compute_efficient_padded_shared_layout(
        _dot_layout(operand_index, num_warps), shape, gl.uint8, is_k_contig=True
    )


@gluon.constexpr_function
def _half_dma_layout(operand_index, num_warps):
    return _d2l_layout(
        _half_smem_layout(operand_index, num_warps), num_warps, vec_bytes=16
    )


@gluon.constexpr_function
def _scale_layout(operand_index, num_warps, rows):
    return cdna4.get_mfma_scale_layout(
        _dot_layout(operand_index, num_warps), [rows, 8]  # 8 scales per 256-K step
    )


@gluon.jit
def _split_halves(s, NUM_WARPS: gl.constexpr, OPERAND: gl.constexpr):
    """[256, 8] scales -> rows 0..127 and 128..255. In the [256, 8] scale layout
    the top/bottom half is one register bit, so this is a register selection."""
    HALF_LAYOUT: gl.constexpr = _scale_layout(OPERAND, NUM_WARPS, 128)
    top, bot = gl.split(gl.permute(s.reshape([2, 128, 8]), [1, 2, 0]))
    return gl.convert_layout(top, HALF_LAYOUT), gl.convert_layout(bot, HALF_LAYOUT)


@gluon.jit
def _copy(smem, src, off, mask, MASK_K: gl.constexpr):
    if MASK_K:
        acp.buffer_load_to_shared(smem, src, off, mask=mask)
    else:
        acp.buffer_load_to_shared(smem, src, off)
    acp.commit_group()


@gluon.jit
def _issue_tile(
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    smem_as,
    smem_bs,
    stage,
    a_ptr,
    b_ptr,
    a_scales_ptr,
    b_scales_ptr,
    off_a_top,
    off_a_bot,
    off_b_left,
    off_b_right,
    off_as,
    off_bs,
    k_a,
    k_b,
    g_s,
    kt,
    K,
    stride_ask,
    stride_bsk,
    MASK_K: gl.constexpr,
):
    """Copy K tile kt (six tensors) into LDS stage `stage`, one commit group
    each. Offsets are loop invariant; the K advance is a scalar base add."""
    BKB: gl.constexpr = 128
    NG: gl.constexpr = 8
    k_left = K - kt * BKB
    g_left = (2 * K) // 32 - kt * NG
    m_a = k_a[None, :] < k_left
    m_b = k_b[:, None] < k_left
    m_s = g_s[None, :] < g_left
    _copy(
        smem_as.index(stage), a_scales_ptr + kt * (NG * stride_ask), off_as, m_s, MASK_K
    )
    _copy(
        smem_bs.index(stage), b_scales_ptr + kt * (NG * stride_bsk), off_bs, m_s, MASK_K
    )
    _copy(smem_a_top.index(stage), a_ptr + kt * BKB, off_a_top, m_a, MASK_K)
    _copy(smem_a_bot.index(stage), a_ptr + kt * BKB, off_a_bot, m_a, MASK_K)
    _copy(smem_b_left.index(stage), b_ptr + kt * BKB, off_b_left, m_b, MASK_K)
    _copy(smem_b_right.index(stage), b_ptr + kt * BKB, off_b_right, m_b, MASK_K)


@gluon.jit
def _ld_a(smem, stage, NUM_WARPS: gl.constexpr):
    return smem.index(stage).load(_dot_layout(0, NUM_WARPS))


@gluon.jit
def _ld_b(smem, stage, NUM_WARPS: gl.constexpr):
    return smem.index(stage).load(_dot_layout(1, NUM_WARPS))


@gluon.jit
def _ld_scales(smem, stage, OPERAND: gl.constexpr, NUM_WARPS: gl.constexpr):
    s = smem.index(stage).load(_scale_layout(OPERAND, NUM_WARPS, 256))
    return _split_halves(s, NUM_WARPS, OPERAND)


@gluon.jit
def _refill(smem, stage, src, off, mask, MASK_K: gl.constexpr):
    if MASK_K:
        acp.buffer_load_to_shared(smem.index(stage), src, off, mask=mask)
    else:
        acp.buffer_load_to_shared(smem.index(stage), src, off)


@gluon.jit
def _issue_stage(
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    smem_as,
    smem_bs,
    stage,
    a_ptr,
    b_ptr,
    a_scales_ptr,
    b_scales_ptr,
    off_a_top,
    off_a_bot,
    off_b_left,
    off_b_right,
    off_as,
    off_bs,
    k_a,
    k_b,
    g_s,
    kt,
    K,
    stride_ask,
    stride_bsk,
    MASK_K: gl.constexpr,
):
    """Copy K step kt into stage `stage` as the loop's four commit groups,
    in consumption order: {B_left + B scales}, {A_top + A scales}, {A_bot},
    {B_right}."""
    BKB: gl.constexpr = 128
    NG: gl.constexpr = 8
    k_left = K - kt * BKB
    g_left = (2 * K) // 32 - kt * NG
    m_a = k_a[None, :] < k_left
    m_b = k_b[:, None] < k_left
    m_s = g_s[None, :] < g_left
    _refill(smem_b_left, stage, b_ptr + kt * BKB, off_b_left, m_b, MASK_K)
    _refill(smem_bs, stage, b_scales_ptr + kt * (NG * stride_bsk), off_bs, m_s, MASK_K)
    acp.commit_group()
    _refill(smem_a_top, stage, a_ptr + kt * BKB, off_a_top, m_a, MASK_K)
    _refill(smem_as, stage, a_scales_ptr + kt * (NG * stride_ask), off_as, m_s, MASK_K)
    acp.commit_group()
    _refill(smem_a_bot, stage, a_ptr + kt * BKB, off_a_bot, m_a, MASK_K)
    acp.commit_group()
    _refill(smem_b_right, stage, b_ptr + kt * BKB, off_b_right, m_b, MASK_K)
    acp.commit_group()


@gluon.jit
def _read_tile_plain(
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    smem_as,
    smem_bs,
    stage,
    NUM_WARPS: gl.constexpr,
):
    a_s = smem_as.index(stage).load(_scale_layout(0, NUM_WARPS, 256))
    b_s = smem_bs.index(stage).load(_scale_layout(1, NUM_WARPS, 256))
    a_top = _ld_a(smem_a_top, stage, NUM_WARPS)
    a_bot = _ld_a(smem_a_bot, stage, NUM_WARPS)
    b_left = _ld_b(smem_b_left, stage, NUM_WARPS)
    b_right = _ld_b(smem_b_right, stage, NUM_WARPS)
    return a_top, a_bot, b_left, b_right, a_s, b_s


@gluon.jit
def _fix_k_tail(a_top, a_bot, a_s, b_s, kt, K, NUM_WARPS: gl.constexpr):
    """Last K tile when K % 256 != 0: zero A beyond K and give both scales the
    finite value 2^0 there, so the invalid K range adds exactly 0 (masked
    copies already zero-fill; this does not rely on it)."""
    BKB: gl.constexpr = 128
    NG: gl.constexpr = 8
    k_left = K - kt * BKB
    g_left = (2 * K) // 32 - kt * NG
    k_a = gl.arange(0, BKB, layout=gl.SliceLayout(0, _dot_layout(0, NUM_WARPS)))
    a_top = gl.where(k_a[None, :] < k_left, a_top, 0)
    a_bot = gl.where(k_a[None, :] < k_left, a_bot, 0)
    g_a = gl.arange(0, NG, layout=gl.SliceLayout(0, _scale_layout(0, NUM_WARPS, 256)))
    a_s = gl.where(g_a[None, :] < g_left, a_s, 127)
    g_b = gl.arange(0, NG, layout=gl.SliceLayout(0, _scale_layout(1, NUM_WARPS, 256)))
    b_s = gl.where(g_b[None, :] < g_left, b_s, 127)
    return a_top, a_bot, a_s, b_s


@gluon.jit
def _mma_quadrants(
    a_top,
    a_bot,
    b_left,
    b_right,
    a_s,
    b_s,
    acc_tl,
    acc_bl,
    acc_tr,
    acc_br,
    NUM_WARPS: gl.constexpr,
):
    as_top, as_bot = _split_halves(a_s, NUM_WARPS, 0)
    bs_left, bs_right = _split_halves(b_s, NUM_WARPS, 1)
    acc_tl = cdna4.mfma_scaled(a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl)
    acc_bl = cdna4.mfma_scaled(a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl)
    acc_tr = cdna4.mfma_scaled(a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr)
    acc_br = cdna4.mfma_scaled(a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _store_quadrant(
    acc, c_ptr, row0, col0, M, N, stride_cm, stride_cn, MMA: gl.constexpr
):
    c = acc.to(c_ptr.type.element_ty)
    offs_cm = row0 + gl.arange(0, 128, layout=gl.SliceLayout(1, MMA))
    offs_cn = col0 + gl.arange(0, 128, layout=gl.SliceLayout(0, MMA))
    c_offs = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    cdna4.buffer_store(c, c_ptr, c_offs, mask=c_mask)


@gluon.jit(repr=_gemm_afp4wfp4_pingpong_repr)
def _gemm_afp4wfp4_pingpong_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """C = A x B for MXFP4 A (M, K) and B (K, N) with E8M0 scales.

    K is the packed size in bytes (x.shape[1]). B is the (K, N) view of w.
    """
    gl.static_assert(BLOCK_SIZE_M == 256)
    gl.static_assert(BLOCK_SIZE_N == 256)
    gl.static_assert(BLOCK_SIZE_K == 256)
    gl.static_assert(NUM_WARPS == 8)
    gl.static_assert(stride_ak == 1)
    gl.static_assert(stride_bk == 1)
    gl.static_assert(stride_asm == 1)
    gl.static_assert(stride_bsn == 1)
    gl.assume(stride_am > 0)
    gl.assume(stride_bn > 0)
    gl.assume(stride_cm > 0)
    gl.assume(stride_ask > 0)
    gl.assume(stride_bsk > 0)

    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    HM: gl.constexpr = BLOCK_SIZE_M // 2
    HN: gl.constexpr = BLOCK_SIZE_N // 2
    BKB: gl.constexpr = BLOCK_SIZE_K // 2
    NG: gl.constexpr = BLOCK_SIZE_K // 32
    MMA: gl.constexpr = _mma_layout(NUM_WARPS)
    SM_A: gl.constexpr = _half_smem_layout(0, NUM_WARPS)
    SM_B: gl.constexpr = _half_smem_layout(1, NUM_WARPS)
    LD_A: gl.constexpr = _half_dma_layout(0, NUM_WARPS)
    LD_B: gl.constexpr = _half_dma_layout(1, NUM_WARPS)
    LD_S: gl.constexpr = gl.BlockedLayout([4, 1], [64, 1], [1, NUM_WARPS], [0, 1])
    SM_S: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)
    pid = remap_xcd(gl.program_id(axis=0), num_pid_m * num_pid_n, NUM_XCDS=NUM_XCDS)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    # Loop-invariant DMA offsets; rows past M (columns past N) wrap to valid
    # rows and their results are masked at the store.
    r_a = gl.arange(0, HM, layout=gl.SliceLayout(1, LD_A))
    k_a = gl.arange(0, BKB, layout=gl.SliceLayout(0, LD_A))
    off_a_top = ((pid_m * BM + r_a) % M)[:, None] * stride_am + k_a[None, :]
    off_a_bot = ((pid_m * BM + HM + r_a) % M)[:, None] * stride_am + k_a[None, :]
    k_b = gl.arange(0, BKB, layout=gl.SliceLayout(1, LD_B))
    c_b = gl.arange(0, HN, layout=gl.SliceLayout(0, LD_B))
    off_b_left = k_b[:, None] + ((pid_n * BN + c_b) % N)[None, :] * stride_bn
    off_b_right = k_b[:, None] + ((pid_n * BN + HN + c_b) % N)[None, :] * stride_bn
    rows_as = (pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, LD_S))) % M
    rows_bs = (pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, LD_S))) % N
    g_s = gl.arange(0, NG, layout=gl.SliceLayout(0, LD_S))
    off_as = rows_as[:, None] + g_s[None, :] * stride_ask
    off_bs = rows_bs[:, None] + g_s[None, :] * stride_bsk

    smem_a_top = gl.allocate_shared_memory(gl.uint8, [2, HM, BKB], SM_A)
    smem_a_bot = gl.allocate_shared_memory(gl.uint8, [2, HM, BKB], SM_A)
    smem_b_left = gl.allocate_shared_memory(gl.uint8, [2, BKB, HN], SM_B)
    smem_b_right = gl.allocate_shared_memory(gl.uint8, [2, BKB, HN], SM_B)
    smem_as = gl.allocate_shared_memory(gl.uint8, [2, BM, NG], SM_S)
    smem_bs = gl.allocate_shared_memory(gl.uint8, [2, BN, NG], SM_S)

    num_full = (
        K // BKB
    )  # full K steps; a partial last step is handled after the pipeline
    k0 = num_full % 2  # peel one full K step when their number is odd
    acc_tl = gl.zeros([HM, HN], gl.float32, MMA)
    acc_bl = gl.zeros([HM, HN], gl.float32, MMA)
    acc_tr = gl.zeros([HM, HN], gl.float32, MMA)
    acc_br = gl.zeros([HM, HN], gl.float32, MMA)

    if k0 == 1:
        # Peeled K step 0, lockstep.
        _issue_tile(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            0,
            a_ptr,
            b_ptr,
            a_scales_ptr,
            b_scales_ptr,
            off_a_top,
            off_a_bot,
            off_b_left,
            off_b_right,
            off_as,
            off_bs,
            k_a,
            k_b,
            g_s,
            0,
            K,
            stride_ask,
            stride_bsk,
            False,
        )
        acp.wait_group(0)
        gl.barrier()
        a_top, a_bot, b_left, b_right, a_s, b_s = _read_tile_plain(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            0,
            NUM_WARPS,
        )
        acc_tl, acc_bl, acc_tr, acc_br = _mma_quadrants(
            a_top,
            a_bot,
            b_left,
            b_right,
            a_s,
            b_s,
            acc_tl,
            acc_bl,
            acc_tr,
            acc_br,
            NUM_WARPS,
        )
        gl.barrier()  # WAR: stage 0 is refilled by the prologue below

    if num_full - k0 >= 2:
        # Prologue: K steps k0 (stage 0) and k0+1 (stage 1), 8 commit groups.
        _issue_stage(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            0,
            a_ptr,
            b_ptr,
            a_scales_ptr,
            b_scales_ptr,
            off_a_top,
            off_a_bot,
            off_b_left,
            off_b_right,
            off_as,
            off_bs,
            k_a,
            k_b,
            g_s,
            k0,
            K,
            stride_ask,
            stride_bsk,
            False,
        )
        _issue_stage(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            1,
            a_ptr,
            b_ptr,
            a_scales_ptr,
            b_scales_ptr,
            off_a_top,
            off_a_bot,
            off_b_left,
            off_b_right,
            off_as,
            off_bs,
            k_a,
            k_b,
            g_s,
            k0 + 1,
            K,
            stride_ask,
            stride_bsk,
            False,
        )
        acp.wait_group(6)  # {B_left0 + Bs0}, {A_top0 + As0} are in LDS
        gl.barrier()
        b_left = _ld_b(smem_b_left, 0, NUM_WARPS)
        bs_left, bs_right = _ld_scales(smem_bs, 0, 1, NUM_WARPS)
        a_top = _ld_a(smem_a_top, 0, NUM_WARPS)
        as_top, as_bot = _ld_scales(smem_as, 0, 0, NUM_WARPS)

        kr = k0 + 2  # K step refilled into stage 0 (stage 1 gets kr + 1)
        for _ in range((num_full - k0 - 2) // 2):
            # ---------------- K step kr - 2: stage 0 ----------------
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_tl = cdna4.mfma_scaled(
                    a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl
                )
            with warp_pipeline_stage("mem", priority=1):
                a_bot = _ld_a(smem_a_bot, 0, NUM_WARPS)
                _refill(smem_b_left, 0, b_ptr + kr * BKB, off_b_left, None, False)
                _refill(
                    smem_bs,
                    0,
                    b_scales_ptr + kr * (NG * stride_bsk),
                    off_bs,
                    None,
                    False,
                )
                acp.commit_group()
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_bl = cdna4.mfma_scaled(
                    a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl
                )
            with warp_pipeline_stage("mem", priority=1):
                b_right = _ld_b(smem_b_right, 0, NUM_WARPS)
                _refill(smem_a_top, 0, a_ptr + kr * BKB, off_a_top, None, False)
                _refill(
                    smem_as,
                    0,
                    a_scales_ptr + kr * (NG * stride_ask),
                    off_as,
                    None,
                    False,
                )
                acp.commit_group()
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_tr = cdna4.mfma_scaled(
                    a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr
                )
            with warp_pipeline_stage("mem", priority=1):
                b_left = _ld_b(smem_b_left, 1, NUM_WARPS)
                bs_left_n, bs_right_n = _ld_scales(smem_bs, 1, 1, NUM_WARPS)
                _refill(smem_a_bot, 0, a_ptr + kr * BKB, off_a_bot, None, False)
                acp.commit_group()
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_br = cdna4.mfma_scaled(
                    a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br
                )
            with warp_pipeline_stage("mem", priority=1):
                a_top = _ld_a(smem_a_top, 1, NUM_WARPS)
                as_top, as_bot = _ld_scales(smem_as, 1, 0, NUM_WARPS)
                _refill(smem_b_right, 0, b_ptr + kr * BKB, off_b_right, None, False)
                acp.commit_group()
            # ---------------- K step kr - 1: stage 1 ----------------
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_tl = cdna4.mfma_scaled(
                    a_top, as_top, "e2m1", b_left, bs_left_n, "e2m1", acc_tl
                )
            with warp_pipeline_stage("mem", priority=1):
                a_bot = _ld_a(smem_a_bot, 1, NUM_WARPS)
                _refill(smem_b_left, 1, b_ptr + (kr + 1) * BKB, off_b_left, None, False)
                _refill(
                    smem_bs,
                    1,
                    b_scales_ptr + (kr + 1) * (NG * stride_bsk),
                    off_bs,
                    None,
                    False,
                )
                acp.commit_group()
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_bl = cdna4.mfma_scaled(
                    a_bot, as_bot, "e2m1", b_left, bs_left_n, "e2m1", acc_bl
                )
            with warp_pipeline_stage("mem", priority=1):
                b_right = _ld_b(smem_b_right, 1, NUM_WARPS)
                _refill(smem_a_top, 1, a_ptr + (kr + 1) * BKB, off_a_top, None, False)
                _refill(
                    smem_as,
                    1,
                    a_scales_ptr + (kr + 1) * (NG * stride_ask),
                    off_as,
                    None,
                    False,
                )
                acp.commit_group()
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_tr = cdna4.mfma_scaled(
                    a_top, as_top, "e2m1", b_right, bs_right_n, "e2m1", acc_tr
                )
            with warp_pipeline_stage("mem", priority=1):
                b_left = _ld_b(smem_b_left, 0, NUM_WARPS)
                bs_left, bs_right = _ld_scales(smem_bs, 0, 1, NUM_WARPS)
                _refill(smem_a_bot, 1, a_ptr + (kr + 1) * BKB, off_a_bot, None, False)
                acp.commit_group()
            acp.wait_group(5)
            with warp_pipeline_stage("mfma", priority=0):
                acc_br = cdna4.mfma_scaled(
                    a_bot, as_bot, "e2m1", b_right, bs_right_n, "e2m1", acc_br
                )
            with warp_pipeline_stage("mem", priority=1):
                a_top = _ld_a(smem_a_top, 0, NUM_WARPS)
                as_top, as_bot = _ld_scales(smem_as, 0, 0, NUM_WARPS)
                _refill(
                    smem_b_right, 1, b_ptr + (kr + 1) * BKB, off_b_right, None, False
                )
                acp.commit_group()
                kr += 2

        # Drain: K steps num_full-2 (stage 0) and num_full-1 (stage 1), lockstep.
        acp.wait_group(0)
        gl.barrier()
        a_bot = _ld_a(smem_a_bot, 0, NUM_WARPS)
        b_right = _ld_b(smem_b_right, 0, NUM_WARPS)
        acc_tl = cdna4.mfma_scaled(
            a_top, as_top, "e2m1", b_left, bs_left, "e2m1", acc_tl
        )
        acc_bl = cdna4.mfma_scaled(
            a_bot, as_bot, "e2m1", b_left, bs_left, "e2m1", acc_bl
        )
        acc_tr = cdna4.mfma_scaled(
            a_top, as_top, "e2m1", b_right, bs_right, "e2m1", acc_tr
        )
        acc_br = cdna4.mfma_scaled(
            a_bot, as_bot, "e2m1", b_right, bs_right, "e2m1", acc_br
        )
        a_top, a_bot, b_left, b_right, a_s, b_s = _read_tile_plain(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            1,
            NUM_WARPS,
        )
        acc_tl, acc_bl, acc_tr, acc_br = _mma_quadrants(
            a_top,
            a_bot,
            b_left,
            b_right,
            a_s,
            b_s,
            acc_tl,
            acc_bl,
            acc_tr,
            acc_br,
            NUM_WARPS,
        )

    if not EVEN_K:
        # Partial last K step (index num_full), lockstep: masked copies (masked
        # bytes zero-fill), then the register fix-up.
        gl.barrier()  # WAR: stage 0 may still be read by the drain above
        _issue_tile(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            0,
            a_ptr,
            b_ptr,
            a_scales_ptr,
            b_scales_ptr,
            off_a_top,
            off_a_bot,
            off_b_left,
            off_b_right,
            off_as,
            off_bs,
            k_a,
            k_b,
            g_s,
            num_full,
            K,
            stride_ask,
            stride_bsk,
            True,
        )
        acp.wait_group(0)
        gl.barrier()
        a_top, a_bot, b_left, b_right, a_s, b_s = _read_tile_plain(
            smem_a_top,
            smem_a_bot,
            smem_b_left,
            smem_b_right,
            smem_as,
            smem_bs,
            0,
            NUM_WARPS,
        )
        a_top, a_bot, a_s, b_s = _fix_k_tail(
            a_top, a_bot, a_s, b_s, num_full, K, NUM_WARPS
        )
        acc_tl, acc_bl, acc_tr, acc_br = _mma_quadrants(
            a_top,
            a_bot,
            b_left,
            b_right,
            a_s,
            b_s,
            acc_tl,
            acc_bl,
            acc_tr,
            acc_br,
            NUM_WARPS,
        )

    row0 = pid_m * BM
    col0 = pid_n * BN
    _store_quadrant(acc_tl, c_ptr, row0, col0, M, N, stride_cm, stride_cn, MMA)
    _store_quadrant(acc_bl, c_ptr, row0 + HM, col0, M, N, stride_cm, stride_cn, MMA)
    _store_quadrant(acc_tr, c_ptr, row0, col0 + HN, M, N, stride_cm, stride_cn, MMA)
    _store_quadrant(
        acc_br, c_ptr, row0 + HM, col0 + HN, M, N, stride_cm, stride_cn, MMA
    )
