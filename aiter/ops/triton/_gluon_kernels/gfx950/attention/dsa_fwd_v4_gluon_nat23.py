"""
Gluon forward for DeepSeek V3.2/V4 sparse MLA (gfx950 / CDNA4), with attention sink.

=== NATURAL-ORDER + K-PREFETCH variant (the proposed pipeline) ===

This is the kernel for the schedule in docs/NATURAL_ORDER_SCHEDULE_PROPOSAL.md. Differences
vs the dedup baseline (dsa_fwd_v4_dedup.py) -- which it is forked from:

  - NATURAL compute order: QK[t] -> softmax[t] -> PV[t] (no lag). softmax is left EXPOSED
    (it's only ~7%); we instead hide the two memory feeds.
  - QK reads K[t] from REGISTERS (K_lora_cur / K_rope_cur), prefetched during the previous
    iter's PV -- so the QK MFMA never stalls on an LDS read (no inline K-feed-wait).
  - V[t] -> VGPR is issued BEFORE QK[t]  =>  V_rd || QK.
  - K[t+1] -> VGPR is issued AFTER softmax, before PV[t]  =>  K_rd || PV. The result is
    carried in registers and becomes K_cur for QK[t+1]. One live copy of K (read after QK[t]
    has freed K[t]) => no VGPR doubling.
  - gather tile t+1 issued at the loop top (|| QK..softmax), drained by wait_group(0) just
    before the K[t+1] read. 2 LDS buffers (tile t for V_rd, tile t+1 for gather + K_rd).
  - topk dedup (one tkraw, convert at use) + far-prefetch (t+2) carried over from dedup.

Tight dependency (the thing to measure -- risk #2 in the spec): the gather of tile t+1 has
only QK+softmax to land before the K[t+1] read in PV[t]. If it doesn't land, the wait_group(0)
re-exposes a feed-wait (just moved from QK to PV). ATT the lgkmcnt/vmcnt to confirm.
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
    D_V = kwargs.get("D_V", named_args.get("D_V"))
    D_ROPE = kwargs.get("D_ROPE", named_args.get("D_ROPE"))
    pruned = []
    for config in configs:
        tk = config.kwargs["TILE_K"]
        kv_lds = (D_V + D_ROPE) * tk * 2 * 2  # 2 buffers
        if kv_lds <= _LDS_LIMIT:
            pruned.append(config)
    if not pruned:
        pruned.append(configs[0])
    return pruned


def _get_fwd_autotune_configs():
    configs = [
        triton.Config(
            {"BLOCK_H": BLOCK_H, "TILE_K": TILE_K, "waves_per_eu": WPE},
            num_warps=nw,
        )
        for BLOCK_H in [16, 32, 64]
        for TILE_K in [16, 32, 64, 128]
        for WPE in [0, 1, 2]
        for nw in [4]
    ]
    return configs


@triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["num_heads", "TOPK", "D_V", "D_ROPE"],
    prune_configs_by={"early_config_prune": _fwd_prune_configs},
)
@gluon.jit
def _sparse_mla_fwd_nat23_kernel(
    Q_ptr, KV_ptr, TopK_ptr, Sink_ptr, O_ptr, LSE_ptr,
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
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1],
    )
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1],
    )

    _qlora_tpw_k: gl.constexpr = min(64, D_V // 8)
    _qlora_tpw_m: gl.constexpr = 64 // _qlora_tpw_k
    blk_qlora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[_qlora_tpw_m, _qlora_tpw_k],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    _klora_tpw_m: gl.constexpr = min(64, D_V // 8)
    _klora_tpw_n: gl.constexpr = 64 // _klora_tpw_m
    blk_klora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[_klora_tpw_m, _klora_tpw_n],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_krope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[8, 8],   # 8 D_ROPE/lane -> b128 (was [2,1]=b32)
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_topk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0],
    )
    blk_topk2d: gl.constexpr = gl.BlockedLayout(  # [1,64] full-warp tile for HBM->LDS async
        size_per_thread=[1, 1], threads_per_warp=[1, 64], warps_per_cta=[1, 4], order=[1, 0],
    )
    blk_lse: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0],
    )

    sh_qlora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BLOCK_H, D_V], [1, 0],
    )
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [D_V, TILE_K], [0, 1],
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1])
    sh_topk: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [1, 2 * TILE_K], [1, 0])

    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_p_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_acc, k_width=4)
    dot_v_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_acc, k_width=4)

    # ---------- program ids ----------
    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H

    # ---------- offsets for Q ----------
    offs_h_qlora = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads

    q_base = token_idx.to(tl.int64) * stride_q_t
    q_offs_lora = (
        q_base + offs_h_qlora[:, None].to(tl.int64) * stride_q_h
        + offs_v_qlora[None, :].to(tl.int64)
    )
    q_mask_lora = mask_h_qlora[:, None]

    offs_h_qrope = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = offs_h_qrope < num_heads
    q_offs_rope = (
        q_base + offs_h_qrope[:, None].to(tl.int64) * stride_q_h
        + (D_V + offs_r_qrope[None, :]).to(tl.int64)
    )
    q_mask_rope = mask_h_qrope[:, None]

    # nat23: Q goes HBM->VGPR directly (NO Q LDS) -> frees 72KB for a 3rd K buffer

    # ---------- topk and KV offsets ----------
    NUM_TILES: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx.to(tl.int64) * stride_topk_t
    # int32 KV-gather addressing (nat3): the KV offset (token_idx * D_QK + d) fits int32 as long as
    # total_tokens * D_QK < 2^31 (~3.7M positions for D_QK=576). Computing the offset in int32 instead
    # of int64 halves the gather-address VALU (the ~1160-cyc elephant). Host must guard for huge KV.
    stride_kv_t_i32: tl.int32 = stride_kv_t.to(tl.int32)

    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma_s))
    offs_tile_topk2 = gl.arange(0, 2 * TILE_K, layout=gl.SliceLayout(0, blk_topk2d))  # load 64, use 32
    offs_tile_topk = gl.arange(0, TILE_K, layout=blk_topk)

    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    smem_klora = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_V, TILE_K], layout=sh_klora)
    smem_topk = gl.allocate_shared_memory(TopK_ptr.dtype.element_ty, [2, 1, 2 * TILE_K], layout=sh_topk)

    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, mfma_s))
    m_frame = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, mfma_s))
    l_i = gl.full([BLOCK_H], 0.0, dtype=gl.float32, layout=gl.SliceLayout(1, mfma_s))
    acc = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=mfma_acc)

    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
    mask_h_mma = offs_h_mma < num_heads

    # ---------- prologue: gather tile 0 into buf0, read K[0] into REGISTERS ----------
    topk_pos_klora = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_klora,
        mask=offs_tile_klora < TOPK, other=-1)
    topk_pos_krope = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_krope,
        mask=offs_tile_krope < TOPK, other=-1)
    topk_pos_mma = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_mma,
        mask=offs_tile_mma < TOPK, other=-1)

    # far-prefetch tile-1 topk DIRECTLY in the 3 gather layouts (no convert_layout LDS round-trips)
    _o1 = TILE_K
    tk_klora = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + (_o1 + offs_tile_klora), mask=(_o1 + offs_tile_klora) < TOPK, other=-1)
    tk_krope = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + (_o1 + offs_tile_krope), mask=(_o1 + offs_tile_krope) < TOPK, other=-1)
    tk_mma = gl.amd.cdna4.buffer_load(ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + (_o1 + offs_tile_mma), mask=(_o1 + offs_tile_mma) < TOPK, other=-1)

    valid_klora = (topk_pos_klora != -1)
    valid_krope = (topk_pos_krope != -1)
    valid_mma = (topk_pos_mma != -1)
    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    klora_offs = safe_klora[None, :] * stride_kv_t_i32 + offs_v_klora[:, None]
    p2_off_topk = 2 * TILE_K + offs_tile_topk2   # topk[2] HBM->LDS
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_topk.index(0), ptr=TopK_ptr,
        offsets=(topk_base.to(tl.int32) + p2_off_topk)[None, :], mask=(p2_off_topk < TOPK)[None, :])
    gl.amd.cdna4.async_copy.commit_group()   # group: topk[2]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_klora.index(0), ptr=KV_ptr, offsets=klora_offs, mask=valid_klora[None, :])
    gl.amd.cdna4.async_copy.commit_group()   # group: KV[0]   -> in flight [topk[2], KV[0]]

    # nat23: Q HBM->VGPR DIRECTLY in dot-operand layout (offsets built in dot layout) -> NO convert scratch
    _oh_l = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, dot_qlora_a))
    _ov_l = gl.arange(0, D_V, layout=gl.SliceLayout(0, dot_qlora_a))
    _qoff_l = q_base.to(tl.int32) + _oh_l[:, None] * stride_q_h.to(tl.int32) + _ov_l[None, :]
    Q_lora_dot = gl.amd.cdna4.buffer_load(ptr=Q_ptr, offsets=_qoff_l, mask=(_oh_l < num_heads)[:, None], other=0.0)

    # nat23: no prologue K read; K/V streamed just-in-time via load_shared_relaxed in the loop

    cur_buf = 0

    # ---------- main loop: compute tile t (K from regs), gather + K-prefetch tile t+1 ----------
    for t in range(NUM_TILES - 1):
        next_buf = 1 - cur_buf

        # (1) topk[t+3] HBM->LDS deep-prefetch; commit  (topk runs 2+ ahead)
        p3_off_topk = (t + 3) * TILE_K + offs_tile_topk2
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_topk.index(next_buf), ptr=TopK_ptr,
            offsets=(topk_base.to(tl.int32) + p3_off_topk)[None, :], mask=(p3_off_topk < TOPK)[None, :])
        gl.amd.cdna4.async_copy.commit_group()

        # (2) keep topk[t+3] in flight, drain KV[t] + topk[t+2]  (NEVER wait_group(0))
        gl.amd.cdna4.async_copy.wait_group(1)
        _tkbase = smem_topk.index(cur_buf).reshape([2 * TILE_K]).slice(0, TILE_K, 0)
        tk_klora_n = _tkbase.load(gl.SliceLayout(0, blk_klora))
        tk_mma_n = _tkbase.load(gl.SliceLayout(0, mfma_s))

        # (3) QK[t]: K[t] streamed from cur (gathered a full iter ago -> landed)
        K_cur = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_klora.index(cur_buf), dot_klora_b)
        S = gl.amd.cdna4.mfma(Q_lora_dot, K_cur, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s))
        S = S * (scale * 1.4426950408889634)
        S = gl.where(valid_mma[None, :] & mask_h_mma[:, None], S, float("-inf"))

        # (4) issue KV[t+1] gather AFTER QK (hides under softmax+PV), from carried tk_klora
        valid_klora_next = (((t + 1) * TILE_K + offs_tile_klora) < TOPK) & (tk_klora != -1)
        valid_mma_next = (((t + 1) * TILE_K + offs_tile_mma) < TOPK) & (tk_mma != -1)
        safe_klora_next = gl.where(valid_klora_next, tk_klora, 0)
        klora_offs_next = safe_klora_next[None, :] * stride_kv_t_i32 + offs_v_klora[:, None]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_klora.index(next_buf), ptr=KV_ptr, offsets=klora_offs_next, mask=valid_klora_next[None, :])
        gl.amd.cdna4.async_copy.commit_group()

        # (5) softmax[t] (exp2, deferred) -- overlaps the KV[t+1] gather
        m_j = gl.max(S, axis=1)
        m_new = gl.maximum(m_i, m_j)
        m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
        if gl.max(m_new - m_frame) > 11.541560327111707:
            beta = gl.exp2(m_frame - m_new)
            beta_acc = gl.convert_layout(beta, gl.SliceLayout(1, mfma_acc))
            acc = acc * beta_acc[:, None]
            l_i = l_i * beta
            m_frame = m_new
        m_i = m_new
        P = gl.exp2(S - m_frame[:, None])
        l_i = l_i + gl.sum(P, axis=1)
        P_dot = gl.convert_layout(P.to(Q_ptr.dtype.element_ty), dot_p_a)

        # (6) V[t] streamed + PV[t]
        V_cur = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_klora.index(cur_buf).permute([1, 0]), dot_v_b)
        acc = gl.amd.cdna4.mfma(P_dot, V_cur, acc)

        # promote
        valid_mma = valid_mma_next
        tk_klora = tk_klora_n
        tk_mma = tk_mma_n
        cur_buf = next_buf

    # ---------- epilogue: last tile N-1 (JIT streamed reads) ----------
    gl.amd.cdna4.async_copy.wait_group(0)
    K_cur = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_klora.index(cur_buf), dot_klora_b)
    S = gl.amd.cdna4.mfma(Q_lora_dot, K_cur, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s))
    S = S * (scale * 1.4426950408889634)
    S = gl.where(valid_mma[None, :] & mask_h_mma[:, None], S, float("-inf"))
    m_j = gl.max(S, axis=1)
    m_new = gl.maximum(m_i, m_j)
    m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
    if gl.max(m_new - m_frame) > 11.541560327111707:
        beta = gl.exp2(m_frame - m_new)
        beta_acc = gl.convert_layout(beta, gl.SliceLayout(1, mfma_acc))
        acc = acc * beta_acc[:, None]
        l_i = l_i * beta
        m_frame = m_new
    m_i = m_new
    P = gl.exp2(S - m_frame[:, None])
    l_i = l_i + gl.sum(P, axis=1)
    P_dot = gl.convert_layout(P.to(Q_ptr.dtype.element_ty), dot_p_a)
    V_cur = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_klora.index(cur_buf).permute([1, 0]), dot_v_b)
    acc = gl.amd.cdna4.mfma(P_dot, V_cur, acc)

    # ---------- epilogue: fold sink into the denominator (V4 delta) ----------
    if HAS_SINK:
        offs_h_sink = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        sink = gl.amd.cdna4.buffer_load(
            ptr=Sink_ptr, offsets=offs_h_sink.to(tl.int32), mask=offs_h_sink < num_heads, other=float("-inf"))
        sink = sink * 1.4426950408889634
        # deferred: acc/l are in the m_frame frame, so fold the sink against m_frame (not m_i)
        m_final = gl.maximum(m_frame, sink)
        alpha_fix = gl.exp2(m_frame - m_final)
        l_total = l_i * alpha_fix + gl.exp2(sink - m_final)
        alpha_fix_acc = gl.convert_layout(alpha_fix, gl.SliceLayout(1, mfma_acc))
        acc = acc * alpha_fix_acc[:, None]
        l_total_acc = gl.convert_layout(l_total, gl.SliceLayout(1, mfma_acc))
        acc = acc / l_total_acc[:, None]
        lse = m_final * 0.6931471805599453 + gl.log(l_total)
    else:
        l_i_acc = gl.convert_layout(l_i, gl.SliceLayout(1, mfma_acc))
        acc = acc / l_i_acc[:, None]
        lse = m_frame * 0.6931471805599453 + gl.log(l_i)

    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    o_base = token_idx.to(tl.int64) * stride_o_t
    o_offs = o_base + offs_h_o[:, None].to(tl.int64) * stride_o_h + offs_v_o[None, :].to(tl.int64)
    acc_bf = acc.to(O_ptr.dtype.element_ty)
    acc_bf_blk = gl.convert_layout(acc_bf, blk_qlora)
    gl.amd.cdna4.buffer_store(stored_value=acc_bf_blk, ptr=O_ptr, offsets=o_offs.to(tl.int32), mask=mask_h_o[:, None])

    offs_h_lse = hg_offset + gl.arange(0, BLOCK_H, layout=blk_lse)
    mask_h_lse = offs_h_lse < num_heads
    lse_base = token_idx * num_heads
    lse_offs = lse_base + offs_h_lse
    lse_blk = gl.convert_layout(lse, blk_lse)
    gl.amd.cdna4.buffer_store(stored_value=lse_blk, ptr=LSE_ptr, offsets=lse_offs.to(tl.int32), mask=mask_h_lse)


# =====================================================================
# Launcher
# =====================================================================
def sparse_mla_fwd_v4_gluon_nat23(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """Natural-order + K-prefetch variant. Same signature/semantics as the dedup launcher."""
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
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))

    _sparse_mla_fwd_nat23_kernel[grid](
        Q_ptr=q, KV_ptr=kv, TopK_ptr=topk_indices, Sink_ptr=sink_ptr,
        O_ptr=o, LSE_ptr=lse,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1),
        stride_kv_t=kv.stride(0),
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        stride_topk_t=topk_indices.stride(0),
        scale=scale, num_heads=num_heads,
        TOPK=topk, D_V=kv_lora_rank, D_ROPE=rope_rank, HAS_SINK=has_sink,
    )
    return o, lse
