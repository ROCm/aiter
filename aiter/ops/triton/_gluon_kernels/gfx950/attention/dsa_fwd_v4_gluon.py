"""
Gluon forward for DeepSeek V4 sparse MLA (gfx950 / CDNA4), with attention sink.

Based on Leon's (leonling-ll) V3.2 gluon forward from `leonling-ll/aiter` branch
`liyang/dsa` -- which adapted our V3.2 Triton forward and added the gfx950 hardware
control (MFMA4 layouts, padded/swizzled shared, double-buffered K, async DMA pipeline,
ds_read_tr transpose, explicit dot-operand layouts); see also his DSA PR ROCm/aiter#3456.
This file adds the V4 attention-sink epilogue (sink-inclusive LSE) so it matches
`sparse_mla_fwd_v4`, and is exposed via `sparse_mla_fwd_v4(..., backend="gluon")`.

Pipeline (Leon's):
  Prologue:  Q_lora + Q_rope -> shared via async DMA (group A).
             K_lora tile 0 + K_rope tile 0 -> shared via async DMA (group B, double-buffered).
             wait_group(1) -> Q in shared; load Q dot operands once.
  Loop tile t (0..N-2): prefetch K[t+1]; wait_group(1); read K_lora/V_lora(permute)/K_rope;
             S = Q_lora @ K_lora_T + Q_rope @ K_rope_T; online softmax; acc += P @ V_lora.
  Epilogue:  wait_group(0) -> last tile; fold sink into the denominator (V4); write O, LSE.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# =====================================================================
# Utility
# =====================================================================
def _get_lds_limit():
    """Return the per-CU LDS limit in bytes for the current GPU.

    gfx942 (MI300X): 64 KB = 65536 bytes
    gfx950 (MI355X): 160 KB = 163840 bytes
    """
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(prop, "gcnArchName", "")
        if "gfx950" in gcn_arch:
            return 163840
    return 65536


_LDS_LIMIT = _get_lds_limit()


# =====================================================================
# Forward — autotune configs and pruning
# =====================================================================
def _fwd_prune_configs(configs, named_args, **kwargs):
    """Prune autotune configs that would exceed per-CU LDS."""
    D_V = kwargs.get("D_V", named_args.get("D_V"))
    D_ROPE = kwargs.get("D_ROPE", named_args.get("D_ROPE"))
    pruned = []
    for config in configs:
        bh = config.kwargs["BLOCK_H"]
        tk = config.kwargs["TILE_K"]
        ns = config.num_stages
        kv_lds = (D_V + D_ROPE) * tk * 2 * ns
        if kv_lds <= _LDS_LIMIT:
            pruned.append(config)
    if not pruned:
        pruned.append(configs[0])
    return pruned


def _get_fwd_autotune_configs():
    configs = [
        triton.Config(
            {"BLOCK_H": BLOCK_H, "TILE_K": TILE_K,
             "waves_per_eu": WPE},
            num_warps=nw,
        )
        for BLOCK_H in [16, 32, 64]
        for TILE_K in [16, 32, 64, 128]
        for WPE in [0, 1, 2]
        for nw in [4]   # num_warps must be 4 to align with kernel implementation
    ]
    # configs = [triton.Config({"BLOCK_H": 64, "TILE_K": 32, "waves_per_eu": 0}, num_warps=4),]
    return configs


@triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["num_heads", "TOPK", "D_V", "D_ROPE"],
    prune_configs_by={"early_config_prune": _fwd_prune_configs},
)
@gluon.jit
def _sparse_mla_fwd_gl_kernel(
    Q_ptr,          # [total_tokens, num_heads, D_QK] bf16
    KV_ptr,         # [total_tokens, 1, D_QK]         bf16
    TopK_ptr,       # [total_tokens, TOPK]            int32
    Sink_ptr,       # [num_heads]                     fp32; ignored if HAS_SINK == False
    O_ptr,          # [total_tokens, num_heads, D_V]  bf16
    LSE_ptr,        # [total_tokens, num_heads]       fp32 (sink-inclusive if HAS_SINK)
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    TOPK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D_V: gl.constexpr,
    D_ROPE: gl.constexpr,
    HAS_SINK: gl.constexpr,
):
    # ---------- constexpr layouts ----------
    mfma_s: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        transposed=True, warps_per_cta=[4, 1],
    )
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        transposed=True, warps_per_cta=[4, 1],
    )

    # Blocked layouts for global loads.
    _qlora_tpw_k: gl.constexpr = min(64, D_V // 8)
    _qlora_tpw_m: gl.constexpr = 64 // _qlora_tpw_k
    blk_qlora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[_qlora_tpw_m, _qlora_tpw_k],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )

    _klora_tpw_m: gl.constexpr = min(64, D_V // 8)
    _klora_tpw_n: gl.constexpr = 64 // _klora_tpw_m
    blk_klora: gl.constexpr = gl.BlockedLayout(   # [D_V, TILE_K]
        size_per_thread=[8, 1],
        threads_per_warp=[_klora_tpw_m, _klora_tpw_n],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_krope: gl.constexpr = gl.BlockedLayout(   # [D_ROPE, TILE_K] = [64, 16]
        size_per_thread=[2, 1], threads_per_warp=[32, 2],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_topk: gl.constexpr = gl.BlockedLayout(    # [TILE_K] int32
        size_per_thread=[1], threads_per_warp=[64],
        warps_per_cta=[4], order=[0],
    )
    blk_lse: gl.constexpr = gl.BlockedLayout(     # [BLOCK_H] fp32
        size_per_thread=[1], threads_per_warp=[64],
        warps_per_cta=[4], order=[0],
    )

    # Shared layouts.
    sh_qlora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BLOCK_H, D_V], [1, 0],
    )
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[1, 0],
    )
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [D_V, TILE_K], [0, 1],
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[0, 1],
    )

    # Dot operand layouts
    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_p_a: gl.constexpr     = gl.DotOperandLayout(operand_index=0, parent=mfma_acc, k_width=4)
    dot_v_b: gl.constexpr     = gl.DotOperandLayout(operand_index=1, parent=mfma_acc, k_width=4)

    # ---------- program ids ----------
    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H

    # ---------- offsets for Q ----------
    # Q_lora  [BLOCK_H, D_V]
    offs_h_qlora = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads

    q_base = token_idx.to(tl.int64) * stride_q_t
    q_offs_lora = (
        q_base
        + offs_h_qlora[:, None].to(tl.int64) * stride_q_h
        + offs_v_qlora[None, :].to(tl.int64)
    )
    q_mask_lora = mask_h_qlora[:, None]

    # Q_rope  [BLOCK_H, D_ROPE]
    offs_h_qrope = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = offs_h_qrope < num_heads

    q_offs_rope = (
        q_base
        + offs_h_qrope[:, None].to(tl.int64) * stride_q_h
        + (D_V + offs_r_qrope[None, :]).to(tl.int64)
    )
    q_mask_rope = mask_h_qrope[:, None]

    smem_qlora = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_V],  layout=sh_qlora)
    smem_qrope = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_ROPE], layout=sh_qrope)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qlora,
        ptr=Q_ptr,
        offsets=q_offs_lora.to(tl.int32),
        mask=q_mask_lora,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qrope,
        ptr=Q_ptr,
        offsets=q_offs_rope.to(tl.int32),
        mask=q_mask_rope,
    )
    gl.amd.cdna4.async_copy.commit_group()

    # ---------- topk and KV offsets ----------
    NUM_TILES: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx.to(tl.int64) * stride_topk_t

    # offs_tile in three layouts (sliced from each of the three loaders)
    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma_s))
    offs_tile_topk = gl.arange(0, TILE_K, layout=blk_topk)

    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    # (removed dead `topk_pos_reg` prologue load — was never consumed)

    # ---------- shared mem allocations for the K loop ----------
    smem_krope = gl.allocate_shared_memory(
        KV_ptr.dtype.element_ty, [2, D_ROPE, TILE_K], layout=sh_krope,
    )
    smem_klora = gl.allocate_shared_memory(
        KV_ptr.dtype.element_ty, [2, D_V, TILE_K], layout=sh_klora,
    )

    # ---------- accumulators ----------
    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32,
                  layout=gl.SliceLayout(1, mfma_s))
    l_i = gl.full([BLOCK_H], 0.0, dtype=gl.float32,
                  layout=gl.SliceLayout(1, mfma_s))
    acc = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=mfma_acc)

    # ---------- tile-0 prefetch (prologue) ----------
    # Load K_lora and K_rope for tile 0.
    topk_pos_klora = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_klora,
        mask=offs_tile_klora < TOPK, other=-1,
    )
    topk_pos_krope = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_krope,
        mask=offs_tile_krope < TOPK, other=-1,
    )
    topk_pos_mma = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_mma,
        mask=offs_tile_mma < TOPK, other=-1,
    )

    # Deep-prefetch tile-1 topk ONCE in the neutral blk_topk layout (DEDUP). Carried as a
    # single register set; converted to the klora/krope/mma layouts at point of use, to
    # minimize carried register pressure (the 3-layout carry caused an acc-rescale codegen
    # regression -- see att_fwd_gluon_mi350/RESULTS.md).
    p1_off_topk = TILE_K + offs_tile_topk
    tkraw = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + p1_off_topk,
        mask=p1_off_topk < TOPK, other=-1,
    )

    valid_klora = (topk_pos_klora != -1)  # tile_start=0 -> offs_tile<TOPK already true
    valid_krope = (topk_pos_krope != -1)
    valid_mma = (topk_pos_mma != -1)

    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    # K_lora async DMA into smem_klora[0]
    klora_offs = (
        safe_klora[None, :].to(tl.int64) * stride_kv_t
        + offs_v_klora[:, None].to(tl.int64)
    )
    klora_smem0 = smem_klora.index(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=klora_smem0,
        ptr=KV_ptr,
        offsets=klora_offs.to(tl.int32),
        mask=valid_klora[None, :],
    )

    # K_rope async DMA into smem_krope[0] — same group as K_lora.
    krope_offs = (
        safe_krope[None, :].to(tl.int64) * stride_kv_t
        + (D_V + offs_r_krope[:, None]).to(tl.int64)
    )
    krope_smem0 = smem_krope.index(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=krope_smem0,
        ptr=KV_ptr,
        offsets=krope_offs.to(tl.int32),
        mask=valid_krope[None, :],
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(1)
    Q_lora_dot = smem_qlora.load(dot_qlora_a)
    Q_rope_dot = smem_qrope.load(dot_qrope_a)

    # ---------- main loop (3-stage): gather tile t+1 from ALREADY-loaded topk,
    #            deep-prefetch topk for tile t+2, compute tile t ----------
    # Carried across iterations: tkraw_{klora,krope,mma} = raw topk for the tile to be
    # GATHERED this iter (loaded the previous iter / prologue); valid_mma = mask for the
    # tile being COMPUTED this iter. This lifts the topk load off the KV-gather critical
    # path so the async_copy is never gated by a fresh index load.
    cur_buf = 0
    for t in range(NUM_TILES - 1):
        next_buf = 1 - cur_buf

        # validity / safe-pos for tile t+1: convert the single carried tkraw (blk_topk) to
        # each consumer layout HERE (transient, not carried) -> low carried register pressure.
        tk_klora = gl.convert_layout(tkraw, gl.SliceLayout(0, blk_klora))
        tk_krope = gl.convert_layout(tkraw, gl.SliceLayout(0, blk_krope))
        tk_mma = gl.convert_layout(tkraw, gl.SliceLayout(0, mfma_s))
        n_off_klora = (t + 1) * TILE_K + offs_tile_klora
        n_off_krope = (t + 1) * TILE_K + offs_tile_krope
        n_off_mma = (t + 1) * TILE_K + offs_tile_mma
        valid_klora_next = (n_off_klora < TOPK) & (tk_klora != -1)
        valid_krope_next = (n_off_krope < TOPK) & (tk_krope != -1)
        valid_mma_next = (n_off_mma < TOPK) & (tk_mma != -1)
        safe_klora_next = gl.where(valid_klora_next, tk_klora, 0)
        safe_krope_next = gl.where(valid_krope_next, tk_krope, 0)

        # issue KV[t+1] async gather into next buffer (UNGATED by any topk load)
        klora_offs_next = (
            safe_klora_next[None, :].to(tl.int64) * stride_kv_t
            + offs_v_klora[:, None].to(tl.int64)
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_klora.index(next_buf),
            ptr=KV_ptr,
            offsets=klora_offs_next.to(tl.int32),
            mask=valid_klora_next[None, :],
        )
        krope_offs_next = (
            safe_krope_next[None, :].to(tl.int64) * stride_kv_t
            + (D_V + offs_r_krope[:, None]).to(tl.int64)
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_krope.index(next_buf),
            ptr=KV_ptr,
            offsets=krope_offs_next.to(tl.int32),
            mask=valid_krope_next[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()

        # deep-prefetch raw topk for tile t+2 (consumed NEXT iter; its latency hides
        # behind this iter's wait_group + compute). buffer_load uses vmcnt, independent
        # of the async_copy group counter that wait_group(1) tracks below.
        f_off_topk = (t + 2) * TILE_K + offs_tile_topk
        tkraw_n = gl.amd.cdna4.buffer_load(
            ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + f_off_topk,
            mask=f_off_topk < TOPK, other=-1,
        )

        # wait for the *current* tile t's KV (older async group; one outstanding = t+1)
        gl.amd.cdna4.async_copy.wait_group(1)

        # ---------- compute current tile t ----------
        # K_lora in two views: dot-operand and transposed (V_lora). permute = view, no move.
        klora_smem_cur = smem_klora.index(cur_buf)
        K_lora_T_dot = klora_smem_cur.load(dot_klora_b)              # opIdx=1 of mfma_s
        V_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)    # opIdx=1 of mfma_acc
        krope_smem_cur = smem_krope.index(cur_buf)
        K_rope_T_dot = krope_smem_cur.load(dot_krope_b)

        S_zero = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s)
        S = gl.amd.cdna4.mfma(Q_lora_dot, K_lora_T_dot, S_zero)
        S = gl.amd.cdna4.mfma(Q_rope_dot, K_rope_T_dot, S)
        S = S * scale

        offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        mask_h_mma = offs_h_mma < num_heads
        valid_mask = valid_mma[None, :] & mask_h_mma[:, None]
        S = gl.where(valid_mask, S, float("-inf"))

        m_j = gl.max(S, axis=1)
        m_new = gl.maximum(m_i, m_j)
        m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = gl.exp(m_i - m_new)
        P = gl.exp(S - m_new[:, None])
        l_new = alpha * l_i + gl.sum(P, axis=1)

        alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, mfma_acc))
        acc = acc * alpha_acc[:, None]

        P_bf = P.to(Q_ptr.dtype.element_ty)
        P_dot = gl.convert_layout(P_bf, dot_p_a)
        acc = gl.amd.cdna4.mfma(P_dot, V_lora_dot, acc)

        m_i = m_new
        l_i = l_new

        # ---------- promote ----------
        cur_buf = next_buf
        valid_mma = valid_mma_next       # tile t+1 -> the tile computed next iter
        tkraw = tkraw_n                  # tile t+2 -> the tile gathered next iter (single, blk_topk)

    # ---------- epilogue: process the last tile (NUM_TILES-1) ----------
    gl.amd.cdna4.async_copy.wait_group(0)

    klora_smem_cur = smem_klora.index(cur_buf)
    K_lora_T_dot = klora_smem_cur.load(dot_klora_b)
    V_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)
    krope_smem_cur = smem_krope.index(cur_buf)
    K_rope_T_dot = krope_smem_cur.load(dot_krope_b)

    S_zero = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s)
    S = gl.amd.cdna4.mfma(Q_lora_dot, K_lora_T_dot, S_zero)
    S = gl.amd.cdna4.mfma(Q_rope_dot, K_rope_T_dot, S)
    S = S * scale

    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
    mask_h_mma = offs_h_mma < num_heads
    valid_mask = valid_mma[None, :] & mask_h_mma[:, None]
    S = gl.where(valid_mask, S, float("-inf"))

    m_j = gl.max(S, axis=1)
    m_new = gl.maximum(m_i, m_j)
    m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
    alpha = gl.exp(m_i - m_new)
    P = gl.exp(S - m_new[:, None])
    l_new = alpha * l_i + gl.sum(P, axis=1)

    alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, mfma_acc))
    acc = acc * alpha_acc[:, None]

    P_bf = P.to(Q_ptr.dtype.element_ty)
    P_dot = gl.convert_layout(P_bf, dot_p_a)
    acc = gl.amd.cdna4.mfma(P_dot, V_lora_dot, acc)

    m_i = m_new
    l_i = l_new

    # ---------- epilogue: fold sink into the denominator (V4 delta) ----------
    if HAS_SINK:
        offs_h_sink = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        sink = gl.amd.cdna4.buffer_load(
            ptr=Sink_ptr, offsets=offs_h_sink.to(tl.int32),
            mask=offs_h_sink < num_heads, other=float("-inf"),
        )
        m_final = gl.maximum(m_i, sink)
        alpha_fix = gl.exp(m_i - m_final)
        l_total = l_i * alpha_fix + gl.exp(sink - m_final)
        alpha_fix_acc = gl.convert_layout(alpha_fix, gl.SliceLayout(1, mfma_acc))
        acc = acc * alpha_fix_acc[:, None]
        l_total_acc = gl.convert_layout(l_total, gl.SliceLayout(1, mfma_acc))
        acc = acc / l_total_acc[:, None]
        lse = m_final + gl.log(l_total)
    else:
        l_i_acc = gl.convert_layout(l_i, gl.SliceLayout(1, mfma_acc))
        acc = acc / l_i_acc[:, None]
        lse = m_i + gl.log(l_i)

    # Output O[token_idx, h, v]
    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    o_base = token_idx.to(tl.int64) * stride_o_t
    o_offs = (
        o_base
        + offs_h_o[:, None].to(tl.int64) * stride_o_h
        + offs_v_o[None, :].to(tl.int64)
    )
    acc_bf = acc.to(O_ptr.dtype.element_ty)
    acc_bf_blk = gl.convert_layout(acc_bf, blk_qlora)
    gl.amd.cdna4.buffer_store(
        stored_value=acc_bf_blk,
        ptr=O_ptr,
        offsets=o_offs.to(tl.int32),
        mask=mask_h_o[:, None],
    )

    # LSE[token_idx, h]
    offs_h_lse = hg_offset + gl.arange(0, BLOCK_H, layout=blk_lse)
    mask_h_lse = offs_h_lse < num_heads
    lse_base = token_idx * num_heads
    lse_offs = lse_base + offs_h_lse
    lse_blk = gl.convert_layout(lse, blk_lse)
    gl.amd.cdna4.buffer_store(
        stored_value=lse_blk,
        ptr=LSE_ptr,
        offsets=lse_offs.to(tl.int32),
        mask=mask_h_lse,
    )


# =====================================================================
# Launcher
# =====================================================================
def sparse_mla_fwd_v4_gluon(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """
    DeepSeek V4 sparse MLA forward (Gluon, gfx950 / CDNA4), with attention sink.

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
        kv:            [total_tokens, 1, d_qk] bfloat16 (or [total_tokens, d_qk])
        topk_indices:  [total_tokens, topk] int32 (SWA + sparse, -1 marks invalid)
        attn_sink:     [num_heads] fp32, optional per-head learnable sink logit.
                       When None, behaves like the V3.2 forward.
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] same dtype as q
        lse: [total_tokens, num_heads] float32 (sink-inclusive when attn_sink is given)
    """
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert topk_indices.is_contiguous()

    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)

    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.shape[0] == total_tokens and kv.shape[-1] == d_qk

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink_ptr = attn_sink
    else:
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)  # guarded by HAS_SINK

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # Grid is autotune-aware: BLOCK_H comes from the chosen config.
    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))

    _sparse_mla_fwd_gl_kernel[grid](
        Q_ptr=q, KV_ptr=kv, TopK_ptr=topk_indices, Sink_ptr=sink_ptr,
        O_ptr=o, LSE_ptr=lse,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1),
        stride_kv_t=kv.stride(0),
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        stride_topk_t=topk_indices.stride(0),
        scale=scale, num_heads=num_heads,
        TOPK=topk,
        D_V=kv_lora_rank, D_ROPE=rope_rank,
        HAS_SINK=has_sink,
    )

    return o, lse
