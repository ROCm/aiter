import torch
import triton
import triton.language as tl


# =====================================================================
# Backward method="privatized" — dKV kernel with copy-based privatization
# =====================================================================
@triton.jit
def _bwd_dkv_privatized(
    Q_T_ptr,            # [T, D_QK, H] bf16
    dO_T_ptr,           # [T, D_V,  H] bf16
    dS_ptr,             # [T, H, TOPK] bf16
    P_ptr,              # [T, H, TOPK] bf16
    TopK_ptr,           # [T, TOPK] int32
    dKV_copies_ptr,     # [NUM_COPIES, T, D_QK] fp32
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_copies: tl.int64,   # T * D_QK — gap between consecutive copies
    stride_dkv_t: tl.int64,   # D_QK
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    NUM_COPIES: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    Privatized dKV scatter kernel.

    Grid: (total_tokens,) -- one program per query token.
    Routes atomic_add to copy (token_idx % NUM_COPIES), so each copy
    receives only TOPK/NUM_COPIES writers per address instead of TOPK.
    A separate reduction kernel sums the copies afterward.
    """
    token_idx = tl.program_id(0)
    copy_idx = token_idx % NUM_COPIES

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    copy_base = copy_idx * stride_copies

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs,
                           mask=tile_offs < TOPK, other=-1)
        valid = (tile_offs < TOPK) & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        dKV_lora = tl.zeros([D_V,    TILE_K], dtype=tl.float32)
        dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            qt_base = token_idx * stride_qt_t
            Q_lora_T = tl.load(
                Q_T_ptr + qt_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            Q_rope_T = tl.load(
                Q_T_ptr + qt_base + (D_V + offs_r[:, None]) * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            dot_base = token_idx * stride_dot_t
            dO_T = tl.load(
                dO_T_ptr + dot_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            ds_base = token_idx * stride_ds_t
            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK), other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK), other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T,     P_val.to(dO_T.dtype)).to(tl.float32)
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        # Scatter into this token's private copy — NUM_COPIES times fewer writers
        dkv_lora_ptrs = (dKV_copies_ptr + copy_base
                         + safe_pos[None, :] * stride_dkv_t + offs_v[:, None])
        tl.atomic_add(dkv_lora_ptrs, dKV_lora, mask=valid[None, :], sem="relaxed")

        dkv_rope_ptrs = (dKV_copies_ptr + copy_base
                         + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None]))
        tl.atomic_add(dkv_rope_ptrs, dKV_rope, mask=valid[None, :], sem="relaxed")


# =====================================================================
# Backward method="xcd_privatized" — dKV kernel with true XCD-local routing
# =====================================================================
@triton.jit
def _bwd_dkv_xcd_local(
    Q_T_ptr,            # [T, D_QK, H] bf16
    dO_T_ptr,           # [T, D_V,  H] bf16
    dS_ptr,             # [T, H, TOPK] bf16
    P_ptr,              # [T, H, TOPK] bf16
    TopK_ptr,           # [T, TOPK] int32
    dKV_copies_ptr,     # [NUM_XCD, T, D_QK] fp32
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_copies: tl.int64,   # T * D_QK
    stride_dkv_t: tl.int64,   # D_QK
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    NUM_XCD: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    XCD-local privatized dKV scatter kernel.

    Reads the AMDGCN HW_ID hardware register at runtime to obtain the true
    XCD (shader engine) index for each CTA.  gfx942 bit layout:
      bits [17:14] = SE_ID = XCD index (0-7 on MI300X).
    All writers to copy k originate from XCD k — atomic adds stay L2-local,
    eliminating cross-XCD coherence write-backs.
    """
    token_idx = tl.program_id(0)
    hw_id = tl.inline_asm_elementwise(
        "s_getreg_b32 s4, hwreg(4, 0, 32)\n"
        "v_mov_b32 $0, s4",
        "=v,~{s4}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    copy_idx = (hw_id >> 14) & 0x7  # SE_ID = XCD index, range 0..NUM_XCD-1

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    copy_base = copy_idx * stride_copies

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs,
                           mask=tile_offs < TOPK, other=-1)
        valid = (tile_offs < TOPK) & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        dKV_lora = tl.zeros([D_V,    TILE_K], dtype=tl.float32)
        dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            qt_base = token_idx * stride_qt_t
            Q_lora_T = tl.load(
                Q_T_ptr + qt_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            Q_rope_T = tl.load(
                Q_T_ptr + qt_base + (D_V + offs_r[:, None]) * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            dot_base = token_idx * stride_dot_t
            dO_T = tl.load(
                dO_T_ptr + dot_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )
            ds_base = token_idx * stride_ds_t
            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK), other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK), other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T,     P_val.to(dO_T.dtype)).to(tl.float32)
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        # Scatter into XCD-local copy — all writers to copy k are on XCD k
        dkv_lora_ptrs = (dKV_copies_ptr + copy_base
                         + safe_pos[None, :] * stride_dkv_t + offs_v[:, None])
        tl.atomic_add(dkv_lora_ptrs, dKV_lora, mask=valid[None, :], sem="relaxed")

        dkv_rope_ptrs = (dKV_copies_ptr + copy_base
                         + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None]))
        tl.atomic_add(dkv_rope_ptrs, dKV_rope, mask=valid[None, :], sem="relaxed")


# Experiment: non-atomic scatter — same traffic pattern as _bwd_dkv_hg_fused
# but uses tl.store instead of tl.atomic_add.  Results are INCORRECT (races),
# but WRITE_SIZE from rocprof tells us how much of the atomic write traffic is
# due to the atomic mechanism itself vs. the data volume.
@triton.jit
def _bwd_dkv_nonatomic_scatter(
    Q_T_ptr, dO_T_ptr, dS_ptr, P_ptr, TopK_ptr, dKV_ptr,
    stride_qt_t: tl.int64, stride_dot_t: tl.int64,
    stride_ds_t: tl.int64, stride_ds_h: tl.int64,
    stride_topk_t: tl.int64, stride_dkv_t: tl.int64,
    num_heads: tl.int32,
    TOPK: tl.constexpr, TILE_K: tl.constexpr, BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr, D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    """
    Diagnostic-only: identical compute to _bwd_dkv_hg_fused but writes with
    tl.store instead of tl.atomic_add.  Results are wrong due to races.
    Use only for rocprof WRITE_SIZE measurement to isolate atomic overhead.
    """
    token_idx = tl.program_id(0)
    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    offs_v    = tl.arange(0, D_V)
    offs_r    = tl.arange(0, D_ROPE)

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs  = tile_start + offs_tile
        topk_pos   = tl.load(TopK_ptr + topk_base + tile_offs,
                             mask=tile_offs < TOPK, other=-1)
        valid   = (tile_offs < TOPK) & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        dKV_lora = tl.zeros([D_V,    TILE_K], dtype=tl.float32)
        dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads
            qt_base  = token_idx * stride_qt_t
            dot_base = token_idx * stride_dot_t
            ds_base  = token_idx * stride_ds_t

            Q_lora_T = tl.load(
                Q_T_ptr + qt_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0)
            Q_rope_T = tl.load(
                Q_T_ptr + qt_base + (D_V + offs_r[:, None]) * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0)
            dO_T = tl.load(
                dO_T_ptr + dot_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0)
            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK), other=0.0)
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & (tile_offs[None, :] < TOPK), other=0.0)

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T,     P_val.to(dO_T.dtype)).to(tl.float32)
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        # Non-atomic store — WRONG results, measures write traffic only
        tl.store(dKV_ptr + safe_pos[None, :] * stride_dkv_t + offs_v[:, None],
                 dKV_lora, mask=valid[None, :])
        tl.store(dKV_ptr + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None]),
                 dKV_rope, mask=valid[None, :])


@triton.jit
def _bwd_dkv_reduce_copies(
    dKV_copies_ptr,     # [NUM_COPIES * T * D_QK] fp32, flattened
    dKV_ptr,            # [T * D_QK] fp32, flattened
    stride_copies: tl.int64,   # T * D_QK
    total_elems: tl.int32,
    NUM_COPIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum NUM_COPIES privatized dKV buffers element-wise into dKV."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(NUM_COPIES):
        acc += tl.load(dKV_copies_ptr + k * stride_copies + offs,
                       mask=mask, other=0.0)
    tl.store(dKV_ptr + offs, acc, mask=mask)
