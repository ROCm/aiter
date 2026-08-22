"""BLOCK_Q fp8 MQA-logits Gluon kernel for gfx950 (CDNA4 / MI350X-MI355X).

Specialization of the DSA / lightning-indexer ``fp8_mqa_logits`` for the
short-query x long-KV prefill shapes used by GLM-5.x / DeepSeek-V3.2 sparse
attention under context-parallel (CP) ragged ranges.

Design (vs the CUDA/DeepGEMM sm100 kernel and the TileLang kernel)
-----------------------------------------------------------------
DeepGEMM (B200) and TileLang both parametrize by head count: ``BLOCK_Q =
128 / num_heads`` so that ``UMMA_N = BLOCK_Q * num_heads == 128`` fills one
Blackwell tcgen05 MMA N-tile. Their accumulator lives in Tensor Memory, so
packing many query rows into that fixed-128 tile is essentially free.

On gfx950 the MMA is a 32x32x64 MFMA whose accumulator lives in *registers*,
so the invariant is different: the tile to fill is the MFMA **M dimension = 32**,
not a 128-wide N. One query row's ``num_heads`` occupies the M dimension:

  * ``num_heads == 32``  -> one row already fills M=32 (our validated case).
  * ``num_heads  < 32``  -> a row under-fills M; a future variant can pack
    ``32 // num_heads`` rows into one MFMA (segmented head-reduce) - not yet
    implemented / validated here.

Crucially, and unlike TMEM, you must **not** keep packing past one M-tile: extra
rows cost registers and occupancy. So we separate two knobs that DeepGEMM/TileLang
conflate into a single ``BLOCK_Q``:

  * fill knob  = rows packed into one MFMA (fixed by the M=32 invariant), and
  * share knob = query rows that share one double-buffered KV tile (this file
    shares across **2** rows), an occupancy-tuned parameter alongside
    ``BLOCK_KV`` / ``waves_per_eu``.

This kernel implements the share=2 variant (each row still runs its own
``_mqa_dot`` at M=num_heads, so the MFMA layout and head reduction are identical
to the validated single-row path). It is general ragged (reads real
``cu_start`` / ``cu_end``), handles an odd tail row, and only writes the valid
``[start, end)`` span (masked stores; assumes ``clean_logits=False``).

Supported / validated shapes are gated by ``_BLOCKQ_CONFIGS`` (per (num_heads,
head_dim) tuned occupancy config); anything not in the table should fall back to
the stock row-per-program kernel in the caller.

Measured on MI355X (gfx950), exact GLM-5.1 indexer prefill shape
(Q=6156, K=49243, num_heads=32, head_dim=128, CP ke=1+row*8): 0.82 ms vs 1.14 ms
for the stock kernel (~28%% faster), max_abs ~3e-5 vs the reference.
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .fp8_mqa_logits import (
    MQAAsyncKVLoader,
    _load_kv_scales_block,
    _mqa_dot,
    _weighted_sum_fma_fold,
    _store_logits_block,
)

# gfx950 invariant: fill exactly one MFMA M-tile with heads.
MFMA_M_TILE = 32

# Query rows sharing one double-buffered KV tile (occupancy knob, not a fill knob).
_KV_SHARE = 2

# Per-(num_heads, head_dim) occupancy configs, tuned + validated on gfx950.
# Add an entry only after measuring correctness and a win vs the stock kernel for
# that shape. Absent shapes fall back to the stock row-per-program kernel.
#
# Measured on MI355X (Q=6156, K=49243, CP ke=1+row*8), median vs stock:
#   (32,128) 0.83 vs 1.11 (1.35x)   (16,128) 0.76 vs 1.10 (1.45x)
#   ( 8,128) 0.75 vs 1.11 (1.48x)   (32, 64) 0.64 vs 0.79 (1.23x)
#   (16, 64) 0.59 vs 0.78 (1.33x)
# num_heads == 64 is intentionally excluded: it spans 2 MFMA M-tiles (M=64),
# doubling the register accumulator, so it loses to the stock kernel (0.92x) -
# falls back. num_heads < 32 under-fills the M=32 MFMA tile but still wins on KV
# reuse; a future packed-M variant (pack 32/num_heads rows per MFMA) could recover
# the wasted M lanes.
_MQA_BLOCKQ_CFG = dict(BLOCK_KV=64, NUM_WARPS=2, NUM_BUFFERS=2, NUM_CHAINS=0, waves_per_eu=4)
_BLOCKQ_CONFIGS = {
    (8, 128): _MQA_BLOCKQ_CFG,
    (16, 64): _MQA_BLOCKQ_CFG,
    (16, 128): _MQA_BLOCKQ_CFG,
    (32, 64): _MQA_BLOCKQ_CFG,
    (32, 128): _MQA_BLOCKQ_CFG,  # GLM-5.1 / DeepSeek-V3.2 indexer prefill shape
}


def get_blockq_config(num_heads: int, head_dim: int):
    """Return the tuned BLOCK_Q config for this shape, or None if unsupported
    (caller should fall back to the stock kernel)."""
    return _BLOCKQ_CONFIGS.get((num_heads, head_dim))


@gluon.jit
def _bq2_body(
    mfma_q,
    w,
    mfma_k,
    kv_scales,
    NUM_HEADS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    mfma_layout: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
):
    s = _mqa_dot(mfma_q, mfma_k, NUM_HEADS, BLOCK_KV, mfma_layout)
    s = gl.maximum(s, 0)
    s = _weighted_sum_fma_fold(s, w, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS) * kv_scales
    return s


@gluon.jit
def _blockq2_fp8_mqa_logits_kernel(
    Q_ptr,
    KV_ptr,
    kv_scales_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    seq_len: gl.int32,
    seq_len_kv: gl.int32,
    stride_q_s: gl.int32,
    stride_q_h: gl.constexpr,
    stride_q_d: gl.constexpr,
    stride_kv_s: gl.int32,
    stride_kv_d: gl.constexpr,
    stride_w_s: gl.int32,
    stride_w_h: gl.constexpr,
    stride_logits_s: gl.int32,
    stride_logits_k: gl.int32,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    USE_PADDED_SHARED_LAYOUT: gl.constexpr,
):
    block_id = gl.num_programs(0) - gl.program_id(axis=0) - 1
    row0 = block_id * BLOCK_Q
    row1 = row0 + 1
    has_row1 = row1 < seq_len
    row1_eff = gl.minimum(row1, seq_len - 1)

    WARP_SIZE: gl.constexpr = 64
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64], transposed=False, warps_per_cta=[1, NUM_WARPS]
    )
    K_WIDTH: gl.constexpr = 16
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=K_WIDTH)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=K_WIDTH)

    Q_INNER: gl.constexpr = HEAD_SIZE // 16
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[WARP_SIZE // Q_INNER, Q_INNER],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    # General ragged: read the real cu_start / cu_end for both rows.
    s0 = gl.maximum(gl.load(cu_start_ptr + row0), 0)
    e0 = gl.minimum(gl.load(cu_end_ptr + row0), seq_len_kv)
    s1 = gl.maximum(gl.load(cu_start_ptr + row1_eff), 0)
    e1 = gl.minimum(gl.load(cu_end_ptr + row1_eff), seq_len_kv)
    common_start = gl.minimum(s0, s1)
    common_start = (common_start // BLOCK_KV) * BLOCK_KV
    common_end = gl.maximum(e0, e1)

    kv_loader = MQAAsyncKVLoader.initialize(
        KV_ptr, seq_len_kv, stride_kv_s, stride_kv_d, BLOCK_KV, HEAD_SIZE,
        NUM_WARPS, WARP_SIZE, NUM_BUFFERS, USE_PADDED_SHARED_LAYOUT,
    )

    offs_h_q = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, layout_q))[:, None]
    offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q))[None, :]
    q0 = gl.amd.cdna4.buffer_load(ptr=Q_ptr, offsets=row0 * stride_q_s + offs_h_q * stride_q_h + offs_d_q * stride_q_d, cache='.cg')
    q1 = gl.amd.cdna4.buffer_load(ptr=Q_ptr, offsets=row1_eff * stride_q_s + offs_h_q * stride_q_h + offs_d_q * stride_q_d, cache='.cg')
    mfma_q0 = gl.convert_layout(q0, dot_a_layout)
    mfma_q1 = gl.convert_layout(q1, dot_a_layout)

    offs_h_w = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))[:, None]
    w0 = gl.amd.cdna4.buffer_load(ptr=weights_ptr, offsets=row0 * stride_w_s + offs_h_w * stride_w_h, cache='.cg')
    w1 = gl.amd.cdna4.buffer_load(ptr=weights_ptr, offsets=row1_eff * stride_w_s + offs_h_w * stride_w_h, cache='.cg')

    store_arange = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
    store_offsets = store_arange * stride_logits_k
    logits0 = logits_ptr + row0 * stride_logits_s
    logits1 = logits_ptr + row1_eff * stride_logits_s

    kv_loader.load_to_shared(common_start, buffer_id=0, USE_BUFFER_LOAD=USE_BUFFER_LOAD, masked=True)
    kv_loader.load_to_shared(common_start + BLOCK_KV, buffer_id=1, USE_BUFFER_LOAD=USE_BUFFER_LOAD, masked=True)
    kv_pos = common_start
    num_full_tiles = (common_end - common_start) // BLOCK_KV
    buf_cur: gl.int32 = 0

    for i in tl.range(0, num_full_tiles - 2):
        kv_scales = _load_kv_scales_block(kv_scales_ptr, kv_pos, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD, common_end)
        mfma_k = kv_loader.load_from_shared(wait_count=1, target_layout=dot_b_layout, buffer_id=buf_cur)
        kv_loader.load_to_shared(common_start + (i + 2) * BLOCK_KV, buffer_id=buf_cur, USE_BUFFER_LOAD=USE_BUFFER_LOAD)
        r0 = _bq2_body(mfma_q0, w0, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
        r1 = _bq2_body(mfma_q1, w1, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
        m0 = (store_arange >= (s0 - kv_pos)) & (store_arange < (e0 - kv_pos))
        m1 = (store_arange >= (s1 - kv_pos)) & (store_arange < (e1 - kv_pos))
        _store_logits_block(logits0 + kv_pos * stride_logits_k, store_offsets, r0, USE_BUFFER_STORE, mask=m0)
        if has_row1:
            _store_logits_block(logits1 + kv_pos * stride_logits_k, store_offsets, r1, USE_BUFFER_STORE, mask=m1)
        kv_pos += BLOCK_KV
        buf_cur = 1 - buf_cur

    if num_full_tiles > 1:
        kv_scales = _load_kv_scales_block(kv_scales_ptr, kv_pos, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD, common_end)
        mfma_k = kv_loader.load_from_shared(wait_count=1, target_layout=dot_b_layout, buffer_id=buf_cur)
        kv_loader.load_to_shared(common_start + num_full_tiles * BLOCK_KV, buffer_id=buf_cur, USE_BUFFER_LOAD=USE_BUFFER_LOAD, masked=True)
        r0 = _bq2_body(mfma_q0, w0, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
        r1 = _bq2_body(mfma_q1, w1, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
        m0 = (store_arange >= (s0 - kv_pos)) & (store_arange < (e0 - kv_pos))
        m1 = (store_arange >= (s1 - kv_pos)) & (store_arange < (e1 - kv_pos))
        _store_logits_block(logits0 + kv_pos * stride_logits_k, store_offsets, r0, USE_BUFFER_STORE, mask=m0)
        if has_row1:
            _store_logits_block(logits1 + kv_pos * stride_logits_k, store_offsets, r1, USE_BUFFER_STORE, mask=m1)
        kv_pos += BLOCK_KV
        buf_cur = 1 - buf_cur

    kv_scales = _load_kv_scales_block(kv_scales_ptr, kv_pos, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD, common_end, masked=True)
    mfma_k = kv_loader.load_from_shared(wait_count=1, target_layout=dot_b_layout, buffer_id=buf_cur)
    r0 = _bq2_body(mfma_q0, w0, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
    r1 = _bq2_body(mfma_q1, w1, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
    m0 = (store_arange >= (s0 - kv_pos)) & (store_arange < (e0 - kv_pos))
    m1 = (store_arange >= (s1 - kv_pos)) & (store_arange < (e1 - kv_pos))
    _store_logits_block(logits0 + kv_pos * stride_logits_k, store_offsets, r0, USE_BUFFER_STORE, mask=m0)
    if has_row1:
        _store_logits_block(logits1 + kv_pos * stride_logits_k, store_offsets, r1, USE_BUFFER_STORE, mask=m1)
    kv_pos += BLOCK_KV
    buf_cur = 1 - buf_cur

    kv_scales = _load_kv_scales_block(kv_scales_ptr, kv_pos, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD, common_end, masked=True)
    mfma_k = kv_loader.load_from_shared(wait_count=0, target_layout=dot_b_layout, buffer_id=buf_cur)
    r0 = _bq2_body(mfma_q0, w0, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
    r1 = _bq2_body(mfma_q1, w1, mfma_k, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS)
    m0 = (store_arange >= (s0 - kv_pos)) & (store_arange < (e0 - kv_pos))
    m1 = (store_arange >= (s1 - kv_pos)) & (store_arange < (e1 - kv_pos))
    _store_logits_block(logits0 + kv_pos * stride_logits_k, store_offsets, r0, USE_BUFFER_STORE, mask=m0)
    if has_row1:
        _store_logits_block(logits1 + kv_pos * stride_logits_k, store_offsets, r1, USE_BUFFER_STORE, mask=m1)


def _launch(Q, KV, kv_scales, weights, cu_starts, cu_ends, logits, *,
            BLOCK_KV, NUM_WARPS, NUM_BUFFERS=2, NUM_CHAINS=0, waves_per_eu):
    """Launch the share=2 kernel with an explicit occupancy config."""
    QLEN, H, D = Q.shape
    KLEN = KV.shape[0]
    BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
    use_buffer_load = KV.numel() * KV.element_size() < BUFFER_LIMIT_BYTES
    use_buffer_store = logits.numel() * logits.element_size() < BUFFER_LIMIT_BYTES
    grid = (triton.cdiv(QLEN, _KV_SHARE),)
    _blockq2_fp8_mqa_logits_kernel[grid](
        Q, KV, kv_scales, weights, cu_starts, cu_ends, logits, QLEN, KLEN,
        Q.stride(0), Q.stride(1), Q.stride(2), KV.stride(0), KV.stride(1),
        weights.stride(0), weights.stride(1), logits.stride(0), logits.stride(1),
        H, D, _KV_SHARE, BLOCK_KV, NUM_WARPS, NUM_BUFFERS, NUM_CHAINS,
        use_buffer_load, use_buffer_store, False,
        num_warps=NUM_WARPS, waves_per_eu=waves_per_eu,
    )
    return logits


def fp8_mqa_logits_blockq(Q, KV, kv_scales, weights, cu_starts, cu_ends, logits):
    """Config-gated BLOCK_Q launcher. Writes into the pre-allocated ``logits`` in
    place (masked to the valid [start, end) span per row; assumes clean_logits=False).

    Raises ValueError if (num_heads, head_dim) has no validated config - the caller
    is expected to check ``get_blockq_config(...)`` first and fall back.

    Shapes match ``fp8_mqa_logits``: Q [seq_len, num_heads, head_dim] fp8,
    KV [seq_len_kv, head_dim] fp8, kv_scales [seq_len_kv] f32,
    weights [seq_len, num_heads] f32, cu_starts/cu_ends [seq_len] i32,
    logits [seq_len, seq_len_kv] f32.
    """
    _, H, D = Q.shape
    cfg = get_blockq_config(H, D)
    if cfg is None:
        raise ValueError(f"no BLOCK_Q config for (num_heads={H}, head_dim={D})")
    return _launch(Q, KV, kv_scales, weights, cu_starts, cu_ends, logits, **cfg)
