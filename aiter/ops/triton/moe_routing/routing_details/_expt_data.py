import triton
import triton.language as tl


@triton.jit
def _cdiv_pow2(n, log2_k):
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _expt_data_compute(Hist, n_expts_tot, MDStarts, tile_starts_stridem, MDTileInfo, tile_info_stridem, max_num_tiles, first_tile_dim_log2,
                      SIZES: tl.constexpr, BLOCK: tl.constexpr):

    pid = tl.program_id(0)
    
    
    if pid == 0:
        x_tile = tl.zeros([BLOCK], dtype=MDStarts.dtype.element_ty)
        Tile_ptrs = MDStarts + tl.arange(0, BLOCK)

        for i in range(0, n_expts_tot + 1, BLOCK):
            offs_n = tl.arange(0, BLOCK) + i
            mask_n0 = offs_n < n_expts_tot
            hist_tok = tl.load(Hist + offs_n, mask=mask_n0, other=0)

            tile_starts = tl.cumsum(hist_tok, 0) + x_tile
            x_tile += tl.sum(hist_tok, 0).to(MDStarts.dtype.element_ty)
            tl.store(Tile_ptrs, tile_starts - hist_tok)
            Tile_ptrs += BLOCK

    else:

        expt_id = (pid-1) // SIZES
        buff_id = (pid-1) % SIZES

        Tile_ptrs = MDStarts + (buff_id + 1) * tile_starts_stridem
        x_tile = tl.zeros([BLOCK], dtype=MDStarts.dtype.element_ty)
        Tile_ptrs = Tile_ptrs + tl.arange(0, BLOCK)
        tile_dim_log2 = first_tile_dim_log2 + buff_id

        for i in range(0, n_expts_tot + 1, BLOCK):
            offs_n = tl.arange(0, BLOCK) + i
            mask_n0 = offs_n < n_expts_tot
            hist_tok = tl.load(Hist + offs_n, mask=mask_n0, other=0)
            hist_tile = _cdiv_pow2(hist_tok, tile_dim_log2)

            tile_starts = tl.cumsum(hist_tile, 0) + x_tile
            x_tile += tl.sum(hist_tile, 0).to(MDStarts.dtype.element_ty)
            tl.store(Tile_ptrs, tile_starts - hist_tile)
            Tile_ptrs += BLOCK

        MDStarts += (buff_id + 1) * tile_starts_stridem
        MDTileInfo += buff_id * tile_info_stridem
        
        n_tokens = tl.load(Hist + expt_id)
        tile_dim_log2 = first_tile_dim_log2 + buff_id
        n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
        tile_off = tl.load(MDStarts + expt_id)

        for block_off in range(0, n_blocks, BLOCK):
            block_offs = block_off + tl.arange(0, BLOCK)
            data = (block_offs << 16) + expt_id
            tl.store(MDTileInfo + tile_off + block_offs, data, mask=block_offs < n_blocks)

        if expt_id == n_expts_tot - 1:
            tile_off = tl.load(MDStarts + n_expts_tot)
            for block_off in range(tile_off, max_num_tiles, BLOCK):
                block_offs = block_off + tl.arange(0, BLOCK)
                tl.store(MDTileInfo + block_offs, 0xffffffff, mask=block_offs < max_num_tiles)
