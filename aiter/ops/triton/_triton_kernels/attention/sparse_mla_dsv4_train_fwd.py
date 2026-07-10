"""
DeepSeek DSv4 sparse-MLA TRAINING forward (pure Triton, dense top-k, emits LSE).

Distinct from the inference-prefill kernel in `sparse_attention_dsv4.py` (ragged `kv_indptr`, no LSE):
this is the training path — dense `topk[T, TOPK]` indices and a sink-inclusive LSE output (needed by
the backward, `P_j = exp(S_j - LSE)`). V4 form: K = V = kv (dense 512, rope in-place caller-side),
scale 1/√512, attn_sink → softmax denominator only. On gfx950 the faster gluon `mla_gluon` is used
instead (see the public wrapper); this Triton kernel is the gfx942 (and any-non-gluon) training path.

Per program: 1 query token × BLOCK_H heads × all TOPK tokens. Grid (T, cdiv(H, BLOCK_H)).
  S   = scale * Q[t,h] · KV[topk[t,j]]        over D=512 (K=V; V = trans(K), single load)
  m   = max_j S_j (running); online flash softmax
  L   = Σ_j exp(S_j - m) [+ exp(sink - m)]    (sink → denominator only, no V term)
  O   = Σ_j exp(S_j - LSE) · KV[topk[t,j]]
  LSE = m + log(L)                            (sink-inclusive)
"""
import torch
import triton
import triton.language as tl


def _get_fwd_configs():
    # Autotune space matched to PR3833's (BLOCK_K up to 64, num_warps 4/8, matrix_instr_nonkdim 16/32) —
    # these MFMA-shape + warp-count knobs are the levers on gfx942. num_stages≤3 (ns=4 fails to compile
    # on triton 3.4.0 release AMD backend). Self-contained (no aiter import).
    return [
        triton.Config({"BLOCK_H": bh, "TILE_K": tk, "waves_per_eu": 0, "matrix_instr_nonkdim": nk},
                      num_warps=nw, num_stages=ns)
        for bh in (32, 64)
        for tk in (16, 32, 64)
        for nk in (16, 32)
        for nw in (4, 8)
        for ns in (1, 2, 3)
    ]


@triton.autotune(configs=_get_fwd_configs(), key=["num_heads", "TOPK", "D_V"])
@triton.jit
def _sparse_mla_fwd_v4_kernel(
    Q_ptr,          # [T, H, D]            bf16
    KV_ptr,         # [T, D]               bf16 (K == V)
    TopK_ptr,       # [T, TOPK]            int32
    Sink_ptr,       # [H]                  fp32 (ignored if HAS_SINK == False)
    O_ptr,          # [T, H, D]            bf16
    LSE_ptr,        # [T, H]               fp32 (sink-inclusive)
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)

    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)

    # ---- Load Q once into registers ----
    q_base = token_idx * stride_q_t
    Q_lora = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
        mask=mask_h[:, None], other=0.0,
    )

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)

    topk_pos = tl.load(TopK_ptr + topk_base + offs_tile, mask=offs_tile < TOPK, other=-1)
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < TOPK
        valid = valid & (topk_pos != -1)

        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(TopK_ptr + topk_base + next_offs, mask=next_offs < TOPK, other=-1)

        safe_pos = tl.where(valid, topk_pos, 0)

        # K_lora [D_V, TILE_K] doubles as V_lora via register transpose (MQA / K==V trick).
        K_lora = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
            mask=valid[None, :], other=0.0,
        )
        S = tl.dot(Q_lora, K_lora)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))

        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_j)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_new = alpha * l_i + tl.sum(P, axis=1)

        acc = acc * alpha[:, None]
        V_lora = tl.trans(K_lora)
        acc += tl.dot(P.to(V_lora.dtype), V_lora)

        m_i = m_new
        l_i = l_new

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    # ---- Epilogue: fold sink into denominator (no V term) ----
    if HAS_SINK:
        sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=float("-inf"))
        m_final = tl.maximum(m_i, sink)
        alpha_fix = tl.exp(m_i - m_final)
        l_total = l_i * alpha_fix + tl.exp(sink - m_final)
        acc = acc * alpha_fix[:, None]
        lse = m_final + tl.log(l_total)
        acc = acc / l_total[:, None]
    else:
        lse = m_i + tl.log(l_i)
        acc = acc / l_i[:, None]

    o_base = token_idx * stride_o_t
    tl.store(
        O_ptr + o_base + offs_h[:, None] * stride_o_h + offs_v[None, :],
        acc.to(Q_lora.dtype), mask=mask_h[:, None],
    )
    tl.store(LSE_ptr + token_idx * num_heads + offs_h, lse, mask=mask_h)


def sparse_mla_fwd_v4_triton(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA forward (pure Triton / MFMA, PR2922-derived, rope-stripped).

    q:[T,H,512] bf16, kv:[T,512] bf16 (K=V, rope in-place caller-side) or [T,1,512],
    topk_indices:[T,TOPK] int32 (-1 = invalid), attn_sink:[H] fp32 or None.
    Returns (o[T,H,512] q.dtype, lse[T,H] fp32, sink-inclusive). scale default 1/√512.
    """
    assert q.is_contiguous() and topk_indices.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    D_V = kv_lora_rank
    assert d_qk == D_V, f"V4 dense form: d_qk({d_qk}) must equal kv_lora_rank({D_V}); no appended rope"
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)
    if kv.dim() == 3:
        kv = kv[:, 0, :]
    kv = kv.contiguous()
    assert kv.shape[0] >= total_tokens and kv.shape[-1] == D_V

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink_ptr = attn_sink
    else:
        sink_ptr = torch.empty(1, dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, D_V, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    grid = lambda META: (total_tokens, triton.cdiv(num_heads, META["BLOCK_H"]))
    _sparse_mla_fwd_v4_kernel[grid](
        q, kv, topk_indices, sink_ptr, o, lse,
        q.stride(0), q.stride(1), kv.stride(0),
        o.stride(0), o.stride(1), topk_indices.stride(0),
        scale, num_heads,
        TOPK=topk, D_V=D_V, HAS_SINK=has_sink,
    )
    return o, lse
