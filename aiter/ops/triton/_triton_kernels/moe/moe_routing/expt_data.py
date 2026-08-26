import triton
import triton.language as tl


@triton.jit
def _cdiv_pow2(n, log2_k):
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _expt_data_compute_stage1(
    pid,
    Hist,
    n_expts_tot: tl.constexpr,
    TokenStart,
    TileStart,
    MDTileInfo,
    max_num_tiles,
    n_gates,
    tile_dim_log2: tl.constexpr,
    BLOCK: tl.constexpr,
    EQUAL_BLOCK: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK)
    if EQUAL_BLOCK:
        hist_token = tl.load(Hist + offs_n)
    else:
        mask_n = offs_n < n_expts_tot
        hist_token = tl.load(Hist + offs_n, mask=mask_n, other=0)
    hist_tile = _cdiv_pow2(hist_token, tile_dim_log2)
    token_starts = tl.cumsum(hist_token, 0) - hist_token
    tile_starts = tl.cumsum(hist_tile, 0) - hist_tile

    if pid < n_expts_tot:
        expt_id = tl.zeros([1], tl.int32) + pid
        tile_start = tl.gather(tile_starts, expt_id, 0)
        token_start = tl.gather(token_starts, expt_id, 0)
        tl.store(TileStart + expt_id, tile_start)
        tl.store(TokenStart + expt_id, token_start)
    else:
        tile_start = tl.zeros([1], tl.int32)
        if EQUAL_BLOCK:
            tl.store(TokenStart + offs_n, token_starts)
        else:
            tl.store(TokenStart + offs_n, token_starts, mask=mask_n)

    if pid == 0:
        tl.store(TokenStart + n_expts_tot, n_gates)
        tile_off_last = tl.sum(hist_tile, 0)
        tl.store(TileStart + n_expts_tot, tile_off_last)

        MEMSET_BLOCK: tl.constexpr = 16
        for block_off in range(tile_off_last, max_num_tiles, MEMSET_BLOCK):
            block_offs = block_off + tl.arange(0, MEMSET_BLOCK)
            tl.store(
                MDTileInfo + block_offs, 0xFFFFFFFF, mask=block_offs < max_num_tiles
            )

    return tile_start


@triton.jit
def _expt_data_compute_stage2(
    pid, Hist, tile_start, TileInfo, tile_dim_log2: tl.constexpr
):

    expt_id = pid

    n_tokens = tl.load(Hist + expt_id)
    if n_tokens == 0:
        return
    BLOCK: tl.constexpr = 8
    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    TileInfo += tile_start

    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    block_offs = tl.arange(0, BLOCK)
    for i in range(0, n_blocks, BLOCK):
        data = (block_offs << 16) + expt_id
        tl.store(TileInfo + block_offs, data, mask=block_offs < n_blocks)
        block_offs += BLOCK


@triton.jit
def _expt_data_compute_stage2_fused(expt_id, Hist, tile_start, TileInfo):
    n_tokens = tl.load(Hist + expt_id)
    if n_tokens == 0:
        return
    TileInfo += tile_start
    tl.store(TileInfo, expt_id)
