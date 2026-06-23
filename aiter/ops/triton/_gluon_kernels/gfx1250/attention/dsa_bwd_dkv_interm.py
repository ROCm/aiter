"""
Gluon dKV-intermediate backward kernel for DeepSeek V4 sparse MLA — gfx1250 / MI450.

Port of the gfx950 gluon `_sparse_mla_bwd_dkv_interm_gl_kernel` (which ports the Triton
`_bwd_compute_dkv_intermediate`). Loads Q/dO UNtransposed and transposes in-LDS, removing
the external `q.transpose(1,2).contiguous()` copy. Intrinsic swaps (validated in the
gfx1250 dQ port): MFMA->WMMA (AMDWMMALayout v3, instr [16,16,32], k_width 8), wave32
blocked layouts, async_copy.global_to_shared (pointer-tensor form), permute().load
transpose, gfx1250 buffer_load/store.

Per program: 1 query token. Grid: (total_tokens,).
Per rank-tile (NUM_TILES = R_CHUNK / TILE_K), summed over head groups:
    dKV_lora[D_V, TILE_K]    = sum_hg ( Q_lora_T @ dS + dO_T @ P )   # contract over heads
    dKV_rope[D_ROPE, TILE_K] = sum_hg ( Q_rope_T @ dS )

Contraction dim = BLOCK_H heads -> BLOCK_H=32 fills the WMMA K-dim (32). TILE_K=64 (=N,
multiple of 16). M = D_V/D_ROPE. 4 warps along M (warp_bases=[(1,0),(2,0)]).
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _sparse_mla_bwd_dkv_interm_gl_kernel_gfx1250(
    Q_ptr, dO_ptr, dS_ptr, P_ptr, Interm_ptr,
    stride_q_t: gl.constexpr, stride_q_h: gl.constexpr,
    stride_do_t: gl.constexpr, stride_do_h: gl.constexpr,
    stride_ds_t: gl.constexpr, stride_ds_h: gl.constexpr,
    stride_interm_t: gl.constexpr, stride_interm_r: gl.constexpr,
    num_heads,
    R_CHUNK: gl.constexpr, TILE_K: gl.constexpr, BLOCK_H: gl.constexpr,
    NUM_HG: gl.constexpr, D_V: gl.constexpr, D_ROPE: gl.constexpr,
):
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[(1, 0), (2, 0)], reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    # ---- blocked layouts (wave32: threads_per_warp product = 32) ----
    blk_q: gl.constexpr = gl.BlockedLayout(        # [BLOCK_H, D_V]  D_V contiguous (dim1)
        size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    blk_qrope: gl.constexpr = gl.BlockedLayout(    # [BLOCK_H, D_ROPE]
        size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0])
    blk_ds: gl.constexpr = gl.BlockedLayout(       # [BLOCK_H, TILE_K]  opIdx1 (reg load)
        size_per_thread=[1, 4], threads_per_warp=[8, 4], warps_per_cta=[4, 1], order=[1, 0])

    sh_q: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [BLOCK_H, D_V], [1, 0])
    sh_do: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [BLOCK_H, D_V], [1, 0])
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])

    dot_qT_a:     gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_doT_a:    gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_qropeT_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_ds_b:     gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    dot_p_b:      gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)

    token_idx = gl.program_id(axis=0)
    NUM_TILES: gl.constexpr = R_CHUNK // TILE_K

    smem_q = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_V], layout=sh_q)
    smem_do = gl.allocate_shared_memory(dO_ptr.dtype.element_ty, [BLOCK_H, D_V], layout=sh_do)
    smem_qrope = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_ROPE], layout=sh_qrope)

    q_base = token_idx * stride_q_t
    do_base = token_idx * stride_do_t
    ds_base = token_idx * stride_ds_t
    interm_base = token_idx * stride_interm_t

    offs_d_st = gl.arange(0, D_V, layout=gl.SliceLayout(1, wmma))
    offs_col_st = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, wmma))
    offs_dr_st = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, wmma))

    for t in range(NUM_TILES):
        dKV_lora = gl.zeros([D_V, TILE_K], dtype=gl.float32, layout=wmma)
        dKV_rope = gl.zeros([D_ROPE, TILE_K], dtype=gl.float32, layout=wmma)

        for hg in range(NUM_HG):
            hg_off = hg * BLOCK_H

            offs_h_q = hg_off + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_q))
            offs_v_q = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_q))
            mask_h_q = offs_h_q < num_heads
            q_offs = q_base + offs_h_q[:, None] * stride_q_h + offs_v_q[None, :]
            do_offs = do_base + offs_h_q[:, None] * stride_do_h + offs_v_q[None, :]
            gl.amd.gfx1250.async_copy.global_to_shared(smem_q, Q_ptr + q_offs, mask=mask_h_q[:, None])
            gl.amd.gfx1250.async_copy.global_to_shared(smem_do, dO_ptr + do_offs, mask=mask_h_q[:, None])

            offs_h_qr = hg_off + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
            offs_r_qr = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
            mask_h_qr = offs_h_qr < num_heads
            qr_offs = q_base + offs_h_qr[:, None] * stride_q_h + (D_V + offs_r_qr[None, :])
            gl.amd.gfx1250.async_copy.global_to_shared(smem_qrope, Q_ptr + qr_offs, mask=mask_h_qr[:, None])
            gl.amd.gfx1250.async_copy.commit_group()

            offs_h_ds = hg_off + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_ds))
            offs_col_ds = t * TILE_K + gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_ds))
            mask_h_ds = offs_h_ds < num_heads
            dsp_offs = ds_base + offs_h_ds[:, None] * stride_ds_h + offs_col_ds[None, :]
            dS_blk = gl.amd.gfx1250.buffer_load(dS_ptr, dsp_offs, mask=mask_h_ds[:, None], other=0.0)
            P_blk = gl.amd.gfx1250.buffer_load(P_ptr, dsp_offs, mask=mask_h_ds[:, None], other=0.0)
            dS_dot = gl.convert_layout(dS_blk, dot_ds_b)
            P_dot = gl.convert_layout(P_blk, dot_p_b)

            gl.amd.gfx1250.async_copy.wait_group(0)
            Q_T = smem_q.permute([1, 0]).load(dot_qT_a)              # [D_V, BLOCK_H]
            dO_T = smem_do.permute([1, 0]).load(dot_doT_a)
            Q_rope_T = smem_qrope.permute([1, 0]).load(dot_qropeT_a)  # [D_ROPE, BLOCK_H]

            dKV_lora = gl.amd.gfx1250.wmma(Q_T, dS_dot, dKV_lora)
            dKV_lora = gl.amd.gfx1250.wmma(dO_T, P_dot, dKV_lora)
            dKV_rope = gl.amd.gfx1250.wmma(Q_rope_T, dS_dot, dKV_rope)

        col = t * TILE_K + offs_col_st
        interm_lora_offs = interm_base + col[None, :] * stride_interm_r + offs_d_st[:, None]
        gl.amd.gfx1250.buffer_store(dKV_lora.to(Interm_ptr.dtype.element_ty), Interm_ptr, interm_lora_offs)
        interm_rope_offs = interm_base + col[None, :] * stride_interm_r + (D_V + offs_dr_st[:, None])
        gl.amd.gfx1250.buffer_store(dKV_rope.to(Interm_ptr.dtype.element_ty), Interm_ptr, interm_rope_offs)


def sparse_mla_bwd_dkv_interm_gl_gfx1250(q, do, chunk_dS, chunk_P, R_CHUNK,
                                         kv_lora_rank=512, BLOCK_H=32, TILE_K=64):
    """gfx1250 gluon dKV-intermediate for one chunk. UNtransposed q/do. Returns interm."""
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    assert R_CHUNK % TILE_K == 0
    num_hg = triton.cdiv(num_heads, BLOCK_H)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)
    _sparse_mla_bwd_dkv_interm_gl_kernel_gfx1250[(total_tokens,)](
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
