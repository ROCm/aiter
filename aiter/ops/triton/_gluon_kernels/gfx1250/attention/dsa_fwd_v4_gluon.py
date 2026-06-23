"""
Gluon forward for DeepSeek V4 sparse MLA — gfx1250 / MI450, with attention sink.

Port of the gfx950 gluon forward (`dsa_fwd_v4_gluon.py`, Leon's V3.2 gluon fwd + our V4
sink epilogue) to gfx1250. Intrinsic swaps (all validated on the gfx1250 dQ/dkv ports):
MFMA->WMMA (AMDWMMALayout v3, instr [16,16,32], k_width 8), wave32 blocked layouts,
async_copy.global_to_shared (pointer-tensor form), permute().load K-as-V transpose,
gfx1250 buffer_load/store. WMMA constraint: TILE_K>=32 (it is the K-dim of the P@V matmul).

Exposed via `sparse_mla_fwd_v4(..., backend="gluon")` (routed by gcnArchName).
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def _get_fwd_autotune_configs():
    # gfx1250: TILE_K in {32,64,128} (>=32 = WMMA K-dim for P@V); num_warps=4.
    import os
    if os.environ.get("DSA_FAST") == "1":   # pin one config for fast/safe correctness runs
        return [triton.Config({"BLOCK_H": 64, "TILE_K": 64, "waves_per_eu": 0}, num_warps=4)]
    return [
        triton.Config({"BLOCK_H": BLOCK_H, "TILE_K": TILE_K, "waves_per_eu": WPE}, num_warps=4)
        for BLOCK_H in [16, 32, 64]
        for TILE_K in [32, 64, 128]
        for WPE in [0, 1, 2]
    ]


def _fwd_prune_configs(configs, named_args, **kwargs):
    # gfx1250 has large LDS — the K double-buffer at TILE_K=64 needs ~147 KB and runs
    # fine. The old 65536 limit pruned every TILE_K>=32 config and fell back to a tiny
    # BLOCK_H=16 config (3x slower). Use 163840 so TILE_K in {32,64} survive.
    D_V = kwargs.get("D_V", named_args.get("D_V"))
    D_ROPE = kwargs.get("D_ROPE", named_args.get("D_ROPE"))
    pruned = [c for c in configs if (D_V + D_ROPE) * c.kwargs["TILE_K"] * 2 * 2 <= 163840]
    return pruned or [configs[0]]


@triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["num_heads", "TOPK", "D_V", "D_ROPE"],
    prune_configs_by={"early_config_prune": _fwd_prune_configs},
)
@gluon.jit
def _sparse_mla_fwd_gl_kernel_gfx1250(
    Q_ptr, KV_ptr, TopK_ptr, Sink_ptr, O_ptr, LSE_ptr,
    stride_q_t: gl.constexpr, stride_q_h: gl.constexpr,
    stride_kv_t: gl.constexpr,
    stride_o_t: gl.constexpr, stride_o_h: gl.constexpr,
    stride_topk_t: gl.constexpr,
    scale, num_heads,
    TOPK: gl.constexpr, BLOCK_H: gl.constexpr, TILE_K: gl.constexpr,
    D_V: gl.constexpr, D_ROPE: gl.constexpr, HAS_SINK: gl.constexpr,
):
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[(1, 0), (2, 0)], reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    # blocked layouts (wave32: threads_per_warp product = 32)
    blk_qlora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[4, 1], order=[1, 0])
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0])
    blk_klora: gl.constexpr = gl.BlockedLayout(   # [D_V, TILE_K]  D_V contiguous
        size_per_thread=[8, 1], threads_per_warp=[8, 4], warps_per_cta=[1, 4], order=[0, 1])
    blk_krope: gl.constexpr = gl.BlockedLayout(   # [D_ROPE, TILE_K]
        size_per_thread=[2, 1], threads_per_warp=[16, 2], warps_per_cta=[1, 4], order=[0, 1])
    blk_topk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    blk_lse: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])

    sh_qlora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [BLOCK_H, D_V], [1, 0])
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[512, 16]], [D_V, TILE_K], [0, 1])
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1])

    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    dot_p_a:     gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_v_b:     gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)

    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H

    offs_h_qlora = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads
    q_base = token_idx * stride_q_t
    q_offs_lora = q_base + offs_h_qlora[:, None] * stride_q_h + offs_v_qlora[None, :]

    offs_h_qrope = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = offs_h_qrope < num_heads
    q_offs_rope = q_base + offs_h_qrope[:, None] * stride_q_h + (D_V + offs_r_qrope[None, :])

    smem_qlora = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_V], layout=sh_qlora)
    smem_qrope = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_ROPE], layout=sh_qrope)
    gl.amd.gfx1250.async_copy.global_to_shared(smem_qlora, Q_ptr + q_offs_lora, mask=mask_h_qlora[:, None])
    gl.amd.gfx1250.async_copy.global_to_shared(smem_qrope, Q_ptr + q_offs_rope, mask=mask_h_qrope[:, None])
    gl.amd.gfx1250.async_copy.commit_group()

    NUM_TILES: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, wmma))
    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    smem_krope = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_ROPE, TILE_K], layout=sh_krope)
    smem_klora = gl.allocate_shared_memory(KV_ptr.dtype.element_ty, [2, D_V, TILE_K], layout=sh_klora)

    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, wmma))
    l_i = gl.full([BLOCK_H], 0.0, dtype=gl.float32, layout=gl.SliceLayout(1, wmma))
    acc = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=wmma)

    topk_pos_klora = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + offs_tile_klora, mask=offs_tile_klora < TOPK, other=-1)
    topk_pos_krope = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + offs_tile_krope, mask=offs_tile_krope < TOPK, other=-1)
    topk_pos_mma = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + offs_tile_mma, mask=offs_tile_mma < TOPK, other=-1)
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

    gl.amd.gfx1250.async_copy.wait_group(1)
    Q_lora_dot = smem_qlora.load(dot_qlora_a)
    Q_rope_dot = smem_qrope.load(dot_qrope_a)

    cur_buf = 0
    for t in range(NUM_TILES - 1):
        next_offs_klora = (t + 1) * TILE_K + offs_tile_klora
        next_offs_krope = (t + 1) * TILE_K + offs_tile_krope
        next_offs_mma = (t + 1) * TILE_K + offs_tile_mma
        topk_pos_klora_next = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + next_offs_klora, mask=next_offs_klora < TOPK, other=-1)
        topk_pos_krope_next = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + next_offs_krope, mask=next_offs_krope < TOPK, other=-1)
        topk_pos_mma_next = gl.amd.gfx1250.buffer_load(TopK_ptr, topk_base + next_offs_mma, mask=next_offs_mma < TOPK, other=-1)
        valid_klora_next = (next_offs_klora < TOPK) & (topk_pos_klora_next != -1)
        valid_krope_next = (next_offs_krope < TOPK) & (topk_pos_krope_next != -1)
        valid_mma_next = (next_offs_mma < TOPK) & (topk_pos_mma_next != -1)
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
        K_lora_T_dot = klora_smem_cur.load(dot_klora_b)
        V_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)
        krope_smem_cur = smem_krope.index(cur_buf)
        K_rope_T_dot = krope_smem_cur.load(dot_krope_b)

        S = gl.amd.gfx1250.wmma(Q_lora_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma))
        S = gl.amd.gfx1250.wmma(Q_rope_dot, K_rope_T_dot, S)
        S = S * scale
        offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
        valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
        S = gl.where(valid_mask, S, float("-inf"))

        m_j = gl.max(S, axis=1)
        m_new = gl.maximum(m_i, m_j)
        m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = gl.exp(m_i - m_new)
        P = gl.exp(S - m_new[:, None])
        l_new = alpha * l_i + gl.sum(P, axis=1)

        alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, wmma))
        acc = acc * alpha_acc[:, None]
        P_dot = gl.convert_layout(P.to(Q_ptr.dtype.element_ty), dot_p_a)
        acc = gl.amd.gfx1250.wmma(P_dot, V_lora_dot, acc)

        m_i = m_new
        l_i = l_new
        cur_buf = next_buf
        valid_mma = valid_mma_next

    gl.amd.gfx1250.async_copy.wait_group(0)
    klora_smem_cur = smem_klora.index(cur_buf)
    K_lora_T_dot = klora_smem_cur.load(dot_klora_b)
    V_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)
    krope_smem_cur = smem_krope.index(cur_buf)
    K_rope_T_dot = krope_smem_cur.load(dot_krope_b)

    S = gl.amd.gfx1250.wmma(Q_lora_dot, K_lora_T_dot, gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=wmma))
    S = gl.amd.gfx1250.wmma(Q_rope_dot, K_rope_T_dot, S)
    S = S * scale
    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
    valid_mask = valid_mma[None, :] & (offs_h_mma < num_heads)[:, None]
    S = gl.where(valid_mask, S, float("-inf"))

    m_j = gl.max(S, axis=1)
    m_new = gl.maximum(m_i, m_j)
    m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
    alpha = gl.exp(m_i - m_new)
    P = gl.exp(S - m_new[:, None])
    l_new = alpha * l_i + gl.sum(P, axis=1)
    alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, wmma))
    acc = acc * alpha_acc[:, None]
    P_dot = gl.convert_layout(P.to(Q_ptr.dtype.element_ty), dot_p_a)
    acc = gl.amd.gfx1250.wmma(P_dot, V_lora_dot, acc)
    m_i = m_new
    l_i = l_new

    if HAS_SINK:
        offs_h_sink = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, wmma))
        sink = gl.amd.gfx1250.buffer_load(Sink_ptr, offs_h_sink, mask=offs_h_sink < num_heads, other=float("-inf"))
        m_final = gl.maximum(m_i, sink)
        alpha_fix = gl.exp(m_i - m_final)
        l_total = l_i * alpha_fix + gl.exp(sink - m_final)
        acc = acc * gl.convert_layout(alpha_fix, gl.SliceLayout(1, wmma))[:, None]
        acc = acc / gl.convert_layout(l_total, gl.SliceLayout(1, wmma))[:, None]
        lse = m_final + gl.log(l_total)
    else:
        acc = acc / gl.convert_layout(l_i, gl.SliceLayout(1, wmma))[:, None]
        lse = m_i + gl.log(l_i)

    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    o_offs = token_idx * stride_o_t + offs_h_o[:, None] * stride_o_h + offs_v_o[None, :]
    gl.amd.gfx1250.buffer_store(gl.convert_layout(acc.to(O_ptr.dtype.element_ty), blk_qlora), O_ptr, o_offs, mask=mask_h_o[:, None])

    offs_h_lse = hg_offset + gl.arange(0, BLOCK_H, layout=blk_lse)
    lse_offs = token_idx * num_heads + offs_h_lse
    gl.amd.gfx1250.buffer_store(gl.convert_layout(lse, blk_lse), LSE_ptr, lse_offs, mask=offs_h_lse < num_heads)


def sparse_mla_fwd_v4_gluon_gfx1250(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek V4 sparse MLA forward (gluon, gfx1250). Returns (o, lse)."""
    assert q.is_contiguous() and kv.is_contiguous() and topk_indices.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)
        sink_ptr = attn_sink
    else:
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)
    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)
    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))
    _sparse_mla_fwd_gl_kernel_gfx1250[grid](
        Q_ptr=q, KV_ptr=kv, TopK_ptr=topk_indices, Sink_ptr=sink_ptr, O_ptr=o, LSE_ptr=lse,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1), stride_kv_t=kv.stride(0),
        stride_o_t=o.stride(0), stride_o_h=o.stride(1), stride_topk_t=topk_indices.stride(0),
        scale=scale, num_heads=num_heads, TOPK=topk,
        D_V=kv_lora_rank, D_ROPE=rope_rank, HAS_SINK=has_sink,
    )
    return o, lse
