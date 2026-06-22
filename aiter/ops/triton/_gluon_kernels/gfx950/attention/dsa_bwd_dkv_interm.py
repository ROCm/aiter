"""
Gluon dKV-intermediate backward kernel for DeepSeek V4 sparse MLA (gfx950 / MI355X).

M3 port of the Triton `_bwd_compute_dkv_intermediate`. Key gluon delta vs Triton:
the Triton path materializes a transposed Q/dO in HBM (`q.transpose(1,2).contiguous()`);
this kernel loads Q/dO UNtransposed and transposes in-LDS via `ds_read_*_tr`, removing
the external transpose copy.

Per program: 1 query token. Grid: (total_tokens,).
Per rank-tile (loop NUM_TILES = R_CHUNK / TILE_K), summed over head groups:
    dKV_lora[D_V, TILE_K]   = sum_hg ( Q_lora_T @ dS + dO_T @ P )    # contract over heads
    dKV_rope[D_ROPE, TILE_K] = sum_hg ( Q_rope_T @ dS )
    store interm[token, rank, :D_QK]

Q_lora_T/dO_T/Q_rope_T ([D, BLOCK_H]) are the opIdx-0 *transposed* operands -> staged in
LDS, read transposed with ds_read_tr. dS/P ([BLOCK_H, TILE_K]) are opIdx-1 natural-layout
-> register load + convert. M1 config: BLOCK_H=64, TILE_K=64, single-buffered.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _sparse_mla_bwd_dkv_interm_gl_kernel(
    Q_ptr,          # [T, H, D_QK] bf16 (UNtransposed)
    dO_ptr,         # [T, H, D_V]  bf16 (UNtransposed)
    dS_ptr,         # [T, H, R_CHUNK] bf16
    P_ptr,          # [T, H, R_CHUNK] bf16
    Interm_ptr,     # [T, R_CHUNK, D_QK] bf16
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_ds_t: tl.int64, stride_ds_h: tl.int64,
    stride_interm_t: tl.int64, stride_interm_r: tl.int64,
    num_heads: tl.int32,
    R_CHUNK: gl.constexpr, TILE_K: gl.constexpr, BLOCK_H: gl.constexpr,
    NUM_HG: gl.constexpr, D_V: gl.constexpr, D_ROPE: gl.constexpr,
):
    # ===================== constexpr layouts =====================
    # MMA output is [D_V, TILE_K] (and [D_ROPE, TILE_K]); contraction over BLOCK_H heads.
    mfma: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[4, 1],
    )

    # ---- Blocked layouts for HBM loads ----
    # Q/dO [BLOCK_H, D_V] : load coalesced then stage to LDS for transpose-read.
    _q_tpw_k: gl.constexpr = min(64, D_V // 8)
    _q_tpw_m: gl.constexpr = 64 // _q_tpw_k
    blk_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[_q_tpw_m, _q_tpw_k],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blk_qrope: gl.constexpr = gl.BlockedLayout(   # [BLOCK_H, D_ROPE]
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    # dS / P [BLOCK_H, TILE_K] : opIdx-1, register load + convert (no transpose).
    blk_ds: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0],
    )

    # ---- Shared layouts (Q/dO/Q_rope staged for transpose read) ----
    sh_q: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [BLOCK_H, D_V], [1, 0])
    sh_do: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [BLOCK_H, D_V], [1, 0])
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])

    # ---- Dot operand layouts ----
    dot_qT_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_doT_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_qropeT_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_ds_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    dot_p_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)

    token_idx = gl.program_id(axis=0)
    NUM_TILES: gl.constexpr = R_CHUNK // TILE_K

    # ---- LDS for Q/dO/Q_rope (single-buffered) ----
    smem_q = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_V], layout=sh_q)
    smem_do = gl.allocate_shared_memory(dO_ptr.dtype.element_ty, [BLOCK_H, D_V], layout=sh_do)
    smem_qrope = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_ROPE], layout=sh_qrope)

    q_base = token_idx.to(tl.int64) * stride_q_t
    do_base = token_idx.to(tl.int64) * stride_do_t
    ds_base = token_idx.to(tl.int64) * stride_ds_t
    interm_base = token_idx.to(tl.int64) * stride_interm_t

    # store offsets (mfma layout): dKV[d, col] -> interm[token, t*TILE_K+col, d]
    offs_d_st = gl.arange(0, D_V, layout=gl.SliceLayout(1, mfma))
    offs_col_st = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma))
    offs_dr_st = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, mfma))

    for t in range(NUM_TILES):
        dKV_lora = gl.zeros([D_V, TILE_K], dtype=gl.float32, layout=mfma)
        dKV_rope = gl.zeros([D_ROPE, TILE_K], dtype=gl.float32, layout=mfma)

        for hg in range(NUM_HG):
            hg_off = hg * BLOCK_H

            # ---- stage Q/dO/Q_rope (this head group) into LDS ----
            offs_h_q = hg_off + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_q))
            offs_v_q = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_q))
            mask_h_q = offs_h_q < num_heads
            q_offs = q_base + offs_h_q[:, None].to(tl.int64) * stride_q_h + offs_v_q[None, :].to(tl.int64)
            do_offs = do_base + offs_h_q[:, None].to(tl.int64) * stride_do_h + offs_v_q[None, :].to(tl.int64)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_q, ptr=Q_ptr, offsets=q_offs.to(tl.int32), mask=mask_h_q[:, None])
            gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_do, ptr=dO_ptr, offsets=do_offs.to(tl.int32), mask=mask_h_q[:, None])

            offs_h_qr = hg_off + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
            offs_r_qr = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
            mask_h_qr = offs_h_qr < num_heads
            qr_offs = q_base + offs_h_qr[:, None].to(tl.int64) * stride_q_h + (D_V + offs_r_qr[None, :]).to(tl.int64)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_qrope, ptr=Q_ptr, offsets=qr_offs.to(tl.int32), mask=mask_h_qr[:, None])
            gl.amd.cdna4.async_copy.commit_group()

            # ---- load dS / P (this tile, this head group) -> dot operands ----
            offs_h_ds = hg_off + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_ds))
            offs_col_ds = t * TILE_K + gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_ds))
            mask_h_ds = offs_h_ds < num_heads
            dsp_offs = ds_base + offs_h_ds[:, None].to(tl.int64) * stride_ds_h + offs_col_ds[None, :].to(tl.int64)
            dS_blk = gl.amd.cdna4.buffer_load(ptr=dS_ptr, offsets=dsp_offs.to(tl.int32), mask=mask_h_ds[:, None], other=0.0)
            P_blk = gl.amd.cdna4.buffer_load(ptr=P_ptr, offsets=dsp_offs.to(tl.int32), mask=mask_h_ds[:, None], other=0.0)
            dS_dot = gl.convert_layout(dS_blk, dot_ds_b)
            P_dot = gl.convert_layout(P_blk, dot_p_b)

            # ---- wait + transpose-read Q/dO/Q_rope ----
            gl.amd.cdna4.async_copy.wait_group(0)
            Q_T = smem_q.permute([1, 0]).load(dot_qT_a)          # [D_V, BLOCK_H]
            dO_T = smem_do.permute([1, 0]).load(dot_doT_a)       # [D_V, BLOCK_H]
            Q_rope_T = smem_qrope.permute([1, 0]).load(dot_qropeT_a)  # [D_ROPE, BLOCK_H]

            dKV_lora = gl.amd.cdna4.mfma(Q_T, dS_dot, dKV_lora)
            dKV_lora = gl.amd.cdna4.mfma(dO_T, P_dot, dKV_lora)
            dKV_rope = gl.amd.cdna4.mfma(Q_rope_T, dS_dot, dKV_rope)

        # ---- store interm[token, t*TILE_K : +TILE_K, :] (direct from mfma layout) ----
        col = t * TILE_K + offs_col_st
        interm_lora_offs = interm_base + col[None, :].to(tl.int64) * stride_interm_r + offs_d_st[:, None].to(tl.int64)
        gl.amd.cdna4.buffer_store(stored_value=dKV_lora.to(Interm_ptr.dtype.element_ty), ptr=Interm_ptr, offsets=interm_lora_offs.to(tl.int32))
        interm_rope_offs = interm_base + col[None, :].to(tl.int64) * stride_interm_r + (D_V + offs_dr_st[:, None]).to(tl.int64)
        gl.amd.cdna4.buffer_store(stored_value=dKV_rope.to(Interm_ptr.dtype.element_ty), ptr=Interm_ptr, offsets=interm_rope_offs.to(tl.int32))


def sparse_mla_bwd_dkv_interm_gl(q, do, chunk_dS, chunk_P, R_CHUNK,
                                 kv_lora_rank=512, BLOCK_H=32, TILE_K=64):
    """
    Gluon dKV-intermediate for one chunk. Takes UNtransposed q/do (transposes in-kernel).

    Returns interm [T, R_CHUNK, D_QK] bf16.
    """
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    assert R_CHUNK % TILE_K == 0
    num_hg = triton.cdiv(num_heads, BLOCK_H)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    _sparse_mla_bwd_dkv_interm_gl_kernel[(total_tokens,)](
        q, do, chunk_dS, chunk_P, interm,
        q.stride(0), q.stride(1),
        do.stride(0), do.stride(1),
        chunk_dS.stride(0), chunk_dS.stride(1),
        interm.stride(0), interm.stride(1),
        num_heads,
        R_CHUNK=R_CHUNK, TILE_K=TILE_K, BLOCK_H=BLOCK_H,
        NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
        num_warps=4,
    )
    return interm
