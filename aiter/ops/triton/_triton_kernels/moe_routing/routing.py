import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe_routing.expt_data import (
    _expt_data_compute_stage1,
    _expt_data_compute_stage2,
    _expt_data_compute_stage2_fused,
)
from aiter.ops.triton._triton_kernels.moe_routing.bitmatrix import (
    _sum_bitmatrix_rows_fused,
)


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xFFFF0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _routing_compute_indx(
    pid_m,
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    PartialOffs,
    stride_pm,
    stride_pn,
    TokensStart,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
):

    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)

    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    if EVEN_M:
        expert = tl.load(ExptIndx + offs).to(tl.uint32)
    else:
        expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xFFFF)

    if EVEN_M:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn)
        gates += tl.load(TokensStart + expert)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates)
        tl.store(GatherIndx + gates, offs)
        tl.store(GateScal + gates, gate_scal)
    else:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
        gates += tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _routing_compute_indx_fused(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    TokensStart,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
):

    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)

    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = local_offs
    if EVEN_M:
        expert = tl.load(ExptIndx + offs).to(tl.uint32)
    else:
        expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = kv_pairs & 0xFFFF

    if EVEN_M:
        gate_scal = tl.load(ExptScal + offs)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(TokensStart + expert)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates)
        tl.store(GatherIndx + gates, offs)
        tl.store(GateScal + gates, gate_scal)
    else:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _combined_routing(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    PartialOffs,
    stride_pm,
    stride_pn,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    ExpertHist,
    n_expts_tot,
    TokenStart,
    TileStart,
    blocks1a,
    MDTileInfo,
    max_num_tiles,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
):

    pid = tl.program_id(0)

    _expt_data_compute_stage1(
        pid,
        ExpertHist,
        n_expts_tot,
        TokenStart,
        TileStart,
        MDTileInfo,
        max_num_tiles,
        n_gates,
        tile_dim_log2,
        BLOCK_A,
        EQUAL_A,
    )

    if pid < blocks1a:
        _expt_data_compute_stage2(pid, ExpertHist, TileStart, MDTileInfo, tile_dim_log2)
    else:
        pid -= blocks1a
        _routing_compute_indx(
            pid,
            GatherIndx,
            ScatterIndx,
            GateScal,
            ExptScal,
            ExptIndx,
            PartialOffs,
            stride_pm,
            stride_pn,
            TokenStart,
            n_gates,
            BLOCK_M,
            EVEN_M,
            N_EXPTS_ACT,
        )


@triton.jit
def _combined_routing_fused(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    Bitmatrix,
    shape_bm,
    stride_bm,
    stride_bn,
    N_BLKS_BITMATRIX: tl.constexpr,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_TOT: tl.constexpr,
    ExpertHist,
    TokenStart,
    TileStart,
    blocks1a,
    MDTileInfo,
    max_num_tiles,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
):

    pid = tl.program_id(0)

    _sum_bitmatrix_rows_fused(
        Bitmatrix,
        shape_bm,
        stride_bm,
        stride_bn,
        ExpertHist,
        N_BLKS_BITMATRIX,
        BLOCK_M,
        EVEN_M,
    )

    if pid != 0 and pid < blocks1a:
        n_tokens = tl.load(ExpertHist + pid)
        if n_tokens == 0:
            return

    _expt_data_compute_stage1(
        pid,
        ExpertHist,
        N_EXPTS_TOT,
        TokenStart,
        TileStart,
        MDTileInfo,
        max_num_tiles,
        n_gates,
        tile_dim_log2,
        BLOCK_A,
        EQUAL_A,
    )

    if pid < blocks1a:
        _expt_data_compute_stage2_fused(pid, ExpertHist, TileStart, MDTileInfo)
    else:
        _routing_compute_indx_fused(
            GatherIndx,
            ScatterIndx,
            GateScal,
            ExptScal,
            ExptIndx,
            TokenStart,
            n_gates,
            BLOCK_M,
            EVEN_M,
            N_EXPTS_ACT,
        )
