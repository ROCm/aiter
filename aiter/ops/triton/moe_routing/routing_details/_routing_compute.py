import triton
import triton.language as tl

from ._expt_data import _expt_data_compute


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _routing_compute_indx(pid_m, GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm,
                          stride_pn, TokensStart, n_tokens, ExpertHist, hist_size, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):

    loop_iterations = (hist_size + BLOCK_N - 1) // BLOCK_N
    x = tl.zeros([BLOCK_N], ExpertHist.dtype.element_ty)
    for i in range(loop_iterations):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hist_size
        hist2 = tl.load(ExpertHist + offs_n, mask=mask_n)
        tok_starts = tl.cumsum(hist2, 0) - hist2 + x
        x += tl.sum(hist2, 0)
        tl.store(TokensStart + offs_n, tok_starts, mask=mask_n)
        offs_n += BLOCK_N

    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    n_gates = n_tokens * N_EXPTS_ACT

    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)

    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xffff)
    mask = expert != 0xffff
    gate_scal = tl.load(ExptScal + offs, mask=mask)

    # compute run lengths in expert-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
    gates += tl.load(TokensStart + expert, mask=mask)
    gates += exclusive_run_lengths

    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _combined_routing(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm, stride_pn,
                        TokensStart, n_tokens, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
                        ExpertHist, hist_size,
                        n_expts_tot, MDStarts, tile_starts_stridem,
                        blocks1a, MDTileInfo, tile_info_stridem, max_num_tiles, first_tile_dim_log2, SIZES: tl.constexpr, BLOCK_A: tl.constexpr,
                        BLOCK_N: tl.constexpr):

    pid = tl.program_id(0)

    if pid < blocks1a:
        _expt_data_compute(ExpertHist, n_expts_tot, MDStarts, tile_starts_stridem, MDTileInfo, tile_info_stridem, max_num_tiles, first_tile_dim_log2,
                          SIZES, BLOCK_A)
    else:
        pid -= blocks1a
        _routing_compute_indx(pid, GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm,
                              stride_pn, TokensStart, n_tokens, ExpertHist, hist_size, BLOCK_N, BLOCK_M, N_EXPTS_ACT)
