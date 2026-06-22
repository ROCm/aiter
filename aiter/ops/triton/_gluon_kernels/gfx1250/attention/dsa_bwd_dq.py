"""
Gluon dQ backward kernel for DeepSeek V4 sparse MLA — **gfx1250 / MI450** port.

Structural port of the gfx950 `dsa_bwd_dq.py` (which itself ports the Triton
`_bwd_chunk_dq_store_ds_v4`). Algorithm + decomposition identical; only the
hardware-control layer is swapped CDNA4→gfx1250:
  * MFMA (cdna4.AMDMFMALayout v4, instr 16x16x16) -> WMMA (AMDWMMALayout v3, 16x16x32)
  * cdna4.mfma                          -> gfx1250.wmma
  * cdna4.buffer_load / buffer_store    -> gfx1250.buffer_load / buffer_store
  * cdna4.async_copy.buffer_load_to_shared(dest, ptr, offsets, mask)
        -> gfx1250.async_copy.global_to_shared(smem, ptr_tensor, mask)   (pointer tensor!)
  * ds_read_tr transpose read           -> smem.permute([1,0]).load(dot)  (same gluon API)
  * BlockedLayout thread splits: wave64 (prod 64) -> wave32 (prod 32)
  * TILE_K: 16 (MFMA K) -> 32 (WMMA bf16 K-dim) so the dQ-acc MMA fills the K dim
  * k_width=8 for BOTH mmas (bf16); gfx950 used 8/4

Validated machinery (wmma_probe.py): WMMA layouts, wave32 blocked loads, async
gather (Route A, needs vectorizable/contiguous gather dim), permute().load
transpose, both wmma calls — all confirmed vs CPU ref.

Per program: 1 token x BLOCK_H heads x one rank chunk [R_START, R_START+R_CHUNK).
dQ is read-modify-written across chunks (RMW folded at store time in blocked layout).
d_sink is NOT done here (torch reduction in the launcher) -> no atomics.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _sparse_mla_bwd_dq_gl_kernel_gfx1250(
    Q_ptr, KV_ptr, dO_ptr, TopK_ptr, LSE_ptr, Delta_ptr,
    dQ_ptr, dS_ptr, P_ptr,
    stride_q_t: gl.constexpr, stride_q_h: gl.constexpr,
    stride_kv_t: gl.constexpr,
    stride_do_t: gl.constexpr, stride_do_h: gl.constexpr,
    stride_dq_t: gl.constexpr, stride_dq_h: gl.constexpr,
    stride_topk_t: gl.constexpr,
    stride_ds_t: gl.constexpr, stride_ds_h: gl.constexpr,
    scale,
    num_heads,
    R_START,
    R_CHUNK: gl.constexpr, BLOCK_H: gl.constexpr, TILE_K: gl.constexpr,
    D_V: gl.constexpr, D_ROPE: gl.constexpr,
    IS_FIRST_CHUNK: gl.constexpr,
):
    # ===================== WMMA layouts (4 warps along M=heads) =====================
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[(1, 0), (2, 0)], reg_bases=[],
        instr_shape=[16, 16, 32],
    )

    # ---- blocked layouts for global loads (wave32: threads_per_warp product = 32) ----
    blk_qlora: gl.constexpr = gl.BlockedLayout(        # [BLOCK_H, D_V]  Q_lora, dO, dQ_lora
        size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    blk_qrope: gl.constexpr = gl.BlockedLayout(        # [BLOCK_H, D_ROPE]
        size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0])
    blk_klora: gl.constexpr = gl.BlockedLayout(        # [D_V, TILE_K]  (D_V contiguous)
        size_per_thread=[8, 1], threads_per_warp=[8, 4], warps_per_cta=[1, 4], order=[0, 1])
    blk_krope: gl.constexpr = gl.BlockedLayout(        # [D_ROPE, TILE_K]
        size_per_thread=[2, 1], threads_per_warp=[16, 2], warps_per_cta=[1, 4], order=[0, 1])

    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [D_V, TILE_K], [0, 1])
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1])

    # ---- dot operands (k_width=8 bf16, both mmas) ----
    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_do_a:    gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    dot_ds_a:      gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_klora_v_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    dot_krope_v_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)

    # ===================== program ids =====================
    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H
    NUM_TILES: gl.constexpr = R_CHUNK // TILE_K

    # ===================== Q / dO offsets (blocked) =====================
    offs_h_qlora = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads
    q_base = token_idx * stride_q_t
    q_offs_lora = q_base + offs_h_qlora[:, None] * stride_q_h + offs_v_qlora[None, :]

    offs_h_qrope = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = offs_h_qrope < num_heads
    q_offs_rope = q_base + offs_h_qrope[:, None] * stride_q_h + (D_V + offs_r_qrope[None, :])

    offs_h_do = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_do = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_do = offs_h_do < num_heads
    do_base = token_idx * stride_do_t
    do_offs = do_base + offs_h_do[:, None] * stride_do_h + offs_v_do[None, :]

    # ===================== load Q_lora, Q_rope, dO -> VGPR -> dot operand =====================
    q_lora_blk = gl.amd.gfx1250.buffer_load(Q_ptr, q_offs_lora, mask=mask_h_qlora[:, None], other=0.0)
    q_rope_blk = gl.amd.gfx1250.buffer_load(Q_ptr, q_offs_rope, mask=mask_h_qrope[:, None], other=0.0)
    do_blk = gl.amd.gfx1250.buffer_load(dO_ptr, do_offs, mask=mask_h_do[:, None], other=0.0)
    Q_lora_dot = gl.convert_layout(q_lora_blk, dot_qlora_a)
    Q_rope_dot = gl.convert_layout(q_rope_blk, dot_qrope_a)
    dO_dot = gl.convert_layout(do_blk, dot_do_a)

    # ===================== topk / KV offsets =====================
    topk_base = token_idx * stride_topk_t + R_START
    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, wmma))
    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    # ===================== shared mem (double-buffered) =====================
    smem_krope = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_ROPE, TILE_K], layout=sh_krope)
    smem_klora = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_V, TILE_K], layout=sh_klora)

    # ===================== dQ accumulators (zero-init; RMW folded at store) =====================
    dQ_lora = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=wmma)
    dQ_rope = gl.zeros([BLOCK_H, D_ROPE], dtype=gl.float32, layout=wmma)

    # ===================== lse / delta in wmma row-slice layout =====================
    offs_h_s = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
    mask_h_s = offs_h_s < num_heads
    lse = gl.amd.gfx1250.buffer_load(LSE_ptr, token_idx * num_heads + offs_h_s, mask=mask_h_s, other=0.0)
    delta = gl.amd.gfx1250.buffer_load(Delta_ptr, token_idx * num_heads + offs_h_s, mask=mask_h_s, other=0.0)

    # ===================== prologue: K tile 0 -> buffer 0 =====================
    topk_pos_klora = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + offs_tile_klora, mask=offs_tile_klora < R_CHUNK, other=-1)
    topk_pos_krope = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + offs_tile_krope, mask=offs_tile_krope < R_CHUNK, other=-1)
    topk_pos_mma = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + offs_tile_mma, mask=offs_tile_mma < R_CHUNK, other=-1)
    valid_klora = topk_pos_klora != -1
    valid_krope = topk_pos_krope != -1
    valid_mma = topk_pos_mma != -1
    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    klora_offs = safe_klora[None, :] * stride_kv_t + offs_v_klora[:, None]
    gl.amd.gfx1250.async_copy.global_to_shared(smem_klora.index(0), KV_ptr + klora_offs, mask=valid_klora[None, :])
    krope_offs = safe_krope[None, :] * stride_kv_t + (D_V + offs_r_krope[:, None])
    gl.amd.gfx1250.async_copy.global_to_shared(smem_krope.index(0), KV_ptr + krope_offs, mask=valid_krope[None, :])
    gl.amd.gfx1250.async_copy.commit_group()

    # dS / P store offsets in the wmma layout (store directly from compute layout)
    ds_base = token_idx * stride_ds_t + hg_idx * BLOCK_H * stride_ds_h
    offs_h_dsp = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
    offs_tile_dsp = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, wmma))
    mask_h_dsp = (hg_offset + offs_h_dsp) < num_heads

    # ===================== main loop: prefetch t+1, compute t =====================
    cur_buf = 0
    for t in range(NUM_TILES - 1):
        next_offs_klora = (t + 1) * TILE_K + offs_tile_klora
        next_offs_krope = (t + 1) * TILE_K + offs_tile_krope
        next_offs_mma = (t + 1) * TILE_K + offs_tile_mma
        topk_pos_klora_next = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + next_offs_klora, mask=next_offs_klora < R_CHUNK, other=-1)
        topk_pos_krope_next = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + next_offs_krope, mask=next_offs_krope < R_CHUNK, other=-1)
        topk_pos_mma_next = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + next_offs_mma, mask=next_offs_mma < R_CHUNK, other=-1)
        valid_klora_next = (next_offs_klora < R_CHUNK) & (topk_pos_klora_next != -1)
        valid_krope_next = (next_offs_krope < R_CHUNK) & (topk_pos_krope_next != -1)
        valid_mma_next = (next_offs_mma < R_CHUNK) & (topk_pos_mma_next != -1)
        safe_klora_next = gl.where(valid_klora_next, topk_pos_klora_next, 0)
        safe_krope_next = gl.where(valid_krope_next, topk_pos_krope_next, 0)

        next_buf = 1 - cur_buf
        klora_offs_next = safe_klora_next[None, :] * stride_kv_t + offs_v_klora[:, None]
        gl.amd.gfx1250.async_copy.global_to_shared(smem_klora.index(next_buf), KV_ptr + klora_offs_next, mask=valid_klora_next[None, :])
        krope_offs_next = safe_krope_next[None, :] * stride_kv_t + (D_V + offs_r_krope[:, None])
        gl.amd.gfx1250.async_copy.global_to_shared(smem_krope.index(next_buf), KV_ptr + krope_offs_next, mask=valid_krope_next[None, :])
        gl.amd.gfx1250.async_copy.commit_group()
        gl.amd.gfx1250.async_copy.wait_group(1)

        klora_smem_cur = smem_klora.index(cur_buf)
        K_lora_T_dot = klora_smem_cur.load(dot_klora_b)                 # [D_V, TILE_K]
        K_lora_v_dot = klora_smem_cur.permute([1, 0]).load(dot_klora_v_b)  # [TILE_K, D_V]
        krope_smem_cur = smem_krope.index(cur_buf)
        K_rope_T_dot = krope_smem_cur.load(dot_krope_b)
        K_rope_v_dot = krope_smem_cur.permute([1, 0]).load(dot_krope_v_b)

        S = gl.amd.gfx1250.wmma(Q_lora_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma))
        S = gl.amd.gfx1250.wmma(Q_rope_dot, K_rope_T_dot, S)
        S = S * scale
        offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
        valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
        S = gl.where(valid_mask, S, float("-inf"))

        P = gl.exp(S - lse[:, None])
        P = gl.where(valid_mask, P, 0.0)
        dP = gl.amd.gfx1250.wmma(dO_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma))
        dS = P * (dP - delta[:, None]) * scale
        dS = gl.where(valid_mask, dS, 0.0)

        dS_bf = dS.to(KV_ptr.dtype.element_ty)
        dS_dot = gl.convert_layout(dS_bf, dot_ds_a)
        dQ_lora = gl.amd.gfx1250.wmma(dS_dot, K_lora_v_dot, dQ_lora)
        dQ_rope = gl.amd.gfx1250.wmma(dS_dot, K_rope_v_dot, dQ_rope)

        col = t * TILE_K + offs_tile_dsp
        dsp_offs = ds_base + offs_h_dsp[:, None] * stride_ds_h + col[None, :]
        gl.amd.gfx1250.buffer_store(dS_bf, dS_ptr, dsp_offs, mask=mask_h_dsp[:, None])
        gl.amd.gfx1250.buffer_store(P.to(KV_ptr.dtype.element_ty), P_ptr, dsp_offs, mask=mask_h_dsp[:, None])

        cur_buf = next_buf
        valid_mma = valid_mma_next

    # ===================== epilogue: last tile =====================
    gl.amd.gfx1250.async_copy.wait_group(0)
    t = NUM_TILES - 1
    klora_smem_cur = smem_klora.index(cur_buf)
    K_lora_T_dot = klora_smem_cur.load(dot_klora_b)
    K_lora_v_dot = klora_smem_cur.permute([1, 0]).load(dot_klora_v_b)
    krope_smem_cur = smem_krope.index(cur_buf)
    K_rope_T_dot = krope_smem_cur.load(dot_krope_b)
    K_rope_v_dot = krope_smem_cur.permute([1, 0]).load(dot_krope_v_b)

    S = gl.amd.gfx1250.wmma(Q_lora_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma))
    S = gl.amd.gfx1250.wmma(Q_rope_dot, K_rope_T_dot, S)
    S = S * scale
    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
    valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
    S = gl.where(valid_mask, S, float("-inf"))

    P = gl.exp(S - lse[:, None])
    P = gl.where(valid_mask, P, 0.0)
    dP = gl.amd.gfx1250.wmma(dO_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma))
    dS = P * (dP - delta[:, None]) * scale
    dS = gl.where(valid_mask, dS, 0.0)

    dS_bf = dS.to(KV_ptr.dtype.element_ty)
    dS_dot = gl.convert_layout(dS_bf, dot_ds_a)
    dQ_lora = gl.amd.gfx1250.wmma(dS_dot, K_lora_v_dot, dQ_lora)
    dQ_rope = gl.amd.gfx1250.wmma(dS_dot, K_rope_v_dot, dQ_rope)

    col = t * TILE_K + offs_tile_dsp
    dsp_offs = ds_base + offs_h_dsp[:, None] * stride_ds_h + col[None, :]
    gl.amd.gfx1250.buffer_store(dS_bf, dS_ptr, dsp_offs, mask=mask_h_dsp[:, None])
    gl.amd.gfx1250.buffer_store(P.to(KV_ptr.dtype.element_ty), P_ptr, dsp_offs, mask=mask_h_dsp[:, None])

    # ===================== store dQ (lora + rope) with cross-chunk RMW fold =====================
    dq_base = token_idx * stride_dq_t
    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    dq_offs_lora = dq_base + offs_h_o[:, None] * stride_dq_h + offs_v_o[None, :]
    dq_lora_blk = gl.convert_layout(dQ_lora.to(dQ_ptr.dtype.element_ty), blk_qlora)
    if not IS_FIRST_CHUNK:
        prev_lora = gl.amd.gfx1250.buffer_load(dQ_ptr, dq_offs_lora, mask=mask_h_o[:, None], other=0.0)
        dq_lora_blk = (dq_lora_blk.to(gl.float32) + prev_lora.to(gl.float32)).to(dQ_ptr.dtype.element_ty)
    gl.amd.gfx1250.buffer_store(dq_lora_blk, dQ_ptr, dq_offs_lora, mask=mask_h_o[:, None])

    offs_h_or = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_or = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_or = offs_h_or < num_heads
    dq_offs_rope = dq_base + offs_h_or[:, None] * stride_dq_h + (D_V + offs_r_or[None, :])
    dq_rope_blk = gl.convert_layout(dQ_rope.to(dQ_ptr.dtype.element_ty), blk_qrope)
    if not IS_FIRST_CHUNK:
        prev_rope = gl.amd.gfx1250.buffer_load(dQ_ptr, dq_offs_rope, mask=mask_h_or[:, None], other=0.0)
        dq_rope_blk = (dq_rope_blk.to(gl.float32) + prev_rope.to(gl.float32)).to(dQ_ptr.dtype.element_ty)
    gl.amd.gfx1250.buffer_store(dq_rope_blk, dQ_ptr, dq_offs_rope, mask=mask_h_or[:, None])


def sparse_mla_bwd_dq_gl_gfx1250(q, kv, do, topk_indices_padded, lse, delta,
                                 R_CHUNK, topk, kv_lora_rank=512, scale=None,
                                 BLOCK_H=64, TILE_K=32):
    """gfx1250 gluon dQ pass. Returns dq, last-chunk dS/P. d_sink in the caller."""
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    assert R_CHUNK % TILE_K == 0, "TILE_K must divide R_CHUNK"

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    num_hg = triton.cdiv(num_heads, BLOCK_H)
    grid = (total_tokens, num_hg)

    for r_start in range(0, topk, R_CHUNK):
        _sparse_mla_bwd_dq_gl_kernel_gfx1250[grid](
            q, kv, do, topk_indices_padded, lse, delta,
            dq, chunk_dS, chunk_P,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq.stride(0), dq.stride(1),
            topk_indices_padded.stride(0),
            chunk_dS.stride(0), chunk_dS.stride(1),
            scale, num_heads, r_start,
            R_CHUNK=R_CHUNK, BLOCK_H=BLOCK_H, TILE_K=TILE_K,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            IS_FIRST_CHUNK=(r_start == 0),
            num_warps=4, waves_per_eu=1,
        )
    return dq, chunk_dS, chunk_P
