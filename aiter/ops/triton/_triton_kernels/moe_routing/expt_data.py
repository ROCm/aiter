import triton
import triton.language as tl


@triton.jit
def _cdiv_pow2(n, log2_k):
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _expt_data_compute_stage1(
    pid,
    Hist,
    n_expts_tot,
    TokenStart,
    TileStart,
    MDTileInfo,
    max_num_tiles,
    n_gates,
    tile_dim_log2: tl.constexpr,
    BLOCK: tl.constexpr,
    EQUAL_BLOCK: tl.constexpr,
):
    if EQUAL_BLOCK:
        offs_n = tl.arange(0, BLOCK)
        hist_token = tl.load(Hist + offs_n)
        hist_tile = _cdiv_pow2(hist_token, tile_dim_log2)
        token_starts = tl.cumsum(hist_token, 0) - hist_token
        tile_starts = tl.cumsum(hist_tile, 0) - hist_tile
        tl.store(TokenStart + offs_n, token_starts)
        tl.store(TileStart + offs_n, tile_starts)
    else:
        token_acc = tl.zeros([BLOCK], dtype=TokenStart.dtype.element_ty)
        tile_acc = tl.zeros([BLOCK], dtype=TileStart.dtype.element_ty)
        offs_n = tl.arange(0, BLOCK)
        for i in range(0, n_expts_tot, BLOCK):
            mask_n = offs_n < n_expts_tot
            hist_token = tl.load(Hist + offs_n, mask=mask_n, other=0)
            hist_tile = _cdiv_pow2(hist_token, tile_dim_log2)
            token_starts = tl.cumsum(hist_token, 0) - hist_token + token_acc
            tile_starts = tl.cumsum(hist_tile, 0) - hist_tile + tile_acc
            token_acc += tl.sum(hist_token, 0)
            tile_acc += tl.sum(hist_tile, 0)
            tl.store(TokenStart + offs_n, token_starts)
            tl.store(TileStart + offs_n, tile_starts)
            offs_n += BLOCK

    if pid == 0:
        tl.store(TokenStart + n_expts_tot, n_gates)

        hist_tok_last = tl.load(Hist + n_expts_tot - 1)
        hist_tile_last = _cdiv_pow2(hist_tok_last, tile_dim_log2)
        tile_off_last = tl.load(TileStart + n_expts_tot - 1) + hist_tile_last
        tl.store(TileStart + n_expts_tot, tile_off_last)

        MEMSET_BLOCK: tl.constexpr = 16
        for block_off in range(tile_off_last, max_num_tiles, MEMSET_BLOCK):
            block_offs = block_off + tl.arange(0, MEMSET_BLOCK)
            tl.store(
                MDTileInfo + block_offs, 0xFFFFFFFF, mask=block_offs < max_num_tiles
            )


@triton.jit
def _expt_data_compute_stage2(
    pid, Hist, TileStart, TileInfo, tile_dim_log2: tl.constexpr
):

    expt_id = pid

    n_tokens = tl.load(Hist + expt_id)
    if n_tokens == 0:
        return
    BLOCK: tl.constexpr = 8
    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    TileInfo += tl.load(TileStart + expt_id)

    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    block_offs = tl.arange(0, BLOCK)
    for i in range(0, n_blocks, BLOCK):
        data = (block_offs << 16) + expt_id
        tl.store(TileInfo + block_offs, data, mask=block_offs < n_blocks)
        block_offs += BLOCK


@triton.jit
def _expt_data_compute_stage2_fused(expt_id, Hist, TileStart, TileInfo):
    n_tokens = tl.load(Hist + expt_id)
    if n_tokens == 0:
        return
    TileInfo += tl.load(TileStart + expt_id)
    tl.store(TileInfo, expt_id)
