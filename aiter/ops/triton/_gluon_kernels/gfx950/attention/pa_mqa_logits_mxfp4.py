# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged MXFP4 MQA-logits kernel for gfx950, plus its scheduler kernel.

logits[b*next_n + n, t] = sum_h relu(q[b, n, h, :] . kv[t, :]) * w[b*next_n + n, h]
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate

# gl.constexpr, not plain ints: a @gluon.jit body can only read globals that
# are already constexpr.
SCALE_GROUP = gl.constexpr(32)  # OCP MX block size; the scaled MFMA takes no other
WARP_SIZE = gl.constexpr(64)
K_WIDTH = gl.constexpr(16)  # byte containers per lane along K
_NAN_ALL = gl.constexpr(PropagateNan.ALL)


@gluon.constexpr_function
def _axis_layout(layout, keep, rank):
    # Peel from the highest axis inward so the indices still to remove stay valid.
    out = layout
    for d in range(rank - 1, -1, -1):
        if d != keep:
            out = gl.SliceLayout(d, out)
    return out


@gluon.constexpr_function
def _scale_group_layout(scale_layout, n_per_tile, s_lo, s_hi, n_hi):
    """The MFMA scale layout re-indexed onto [n_hi, s_lo, n_per_tile, s_hi].

    The stored order groups one MFMA tile and puts the scale axis innermost, so
    that axis is the run a load widens over. Each [BLOCK_KV, NUM_SCALES] basis
    splits the same way: token into group and row-in-group, scale into run
    element and lane. Group is a register basis rather than a warp one because
    the preshuffled path runs one warp.
    """
    assert not scale_layout.warp_bases, "preshuffled scales expect one warp"
    return gl.DistributedLinearLayout(
        reg_bases=[
            [b[0] // n_per_tile, 0, 0, b[1] // s_lo] for b in scale_layout.reg_bases
        ],
        lane_bases=[
            [0, b[1] % s_lo, b[0] % n_per_tile, 0] for b in scale_layout.lane_bases
        ],
        warp_bases=[],
        block_bases=[],
        shape=[n_hi, s_lo, n_per_tile, s_hi],
    )


@gluon.constexpr_function
def _offset_bases_to_blocked(bases, contiguity, num_warps, warp_size, shape):
    # Mirror Triton's CoalesceAsyncCopy partition of a shared layout's bases:
    # lg2(C) to reg, lg2(WS) to lane, lg2(NW) to warp, leftovers back to reg.
    # Keeping the blocked layout in step with the shared one is what lets the
    # async copy fold into one global-to-LDS instruction with no staging VGPRs.
    rank = len(shape)
    lg2_c = contiguity.bit_length() - 1
    lg2_nw = num_warps.bit_length() - 1
    lg2_ws = warp_size.bit_length() - 1
    i = lg2_c
    reg = bases[:lg2_c]
    lane = bases[i : i + lg2_ws]
    i += lg2_ws
    warp = bases[i : i + lg2_nw]
    i += lg2_nw
    warp = warp + [[0] * rank] * (lg2_nw - len(warp))
    return gl.DistributedLinearLayout(
        reg_bases=reg + bases[i:],
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=shape,
    )


@gluon.constexpr_function
def _staging_layouts(head_bytes, block_kv, num_warps, warp_size):
    # (blocked, shared) for a token-major [head_bytes, block_kv] tile. The XOR
    # term on the token axis is what keeps the ds_reads conflict free; the
    # padding pair covers the rest.
    c = 16  # 128-bit vector of 8-bit containers
    lg2_hs = head_bytes.bit_length() - 1
    lg2_ts = block_kv.bit_length() - 1
    hs_lane = lg2_hs - (c.bit_length() - 1)
    bases = [[1 << i, 0] for i in range(lg2_hs)] + [
        [0, 1 << ((i + hs_lane) % lg2_ts)] for i in range(lg2_ts)
    ]
    shared = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 16]],
        offset_bases=bases,
        cga_layout=[],
        shape=[head_bytes, block_kv],
    )
    return (
        _offset_bases_to_blocked(
            bases, c, num_warps, warp_size, [head_bytes, block_kv]
        ),
        shared,
    )


@gluon.jit
def _shuffled_offsets(
    offs_d,
    offs_n,
    D_EXTENT: gl.constexpr,
    N_PER_TILE: gl.constexpr,
    D_PER_TILE: gl.constexpr,
):
    # Byte offset of (row, container) inside one preshuffled page.
    TILE: gl.constexpr = N_PER_TILE * D_PER_TILE
    GROUP: gl.constexpr = N_PER_TILE * D_EXTENT
    return (
        (offs_d % D_PER_TILE)
        + (offs_d // D_PER_TILE) * TILE
        + (offs_n % N_PER_TILE) * D_PER_TILE
        + (offs_n // N_PER_TILE) * GROUP
    )


@gluon.jit
def _shuffled_scale_offsets(
    offs_n, offs_s, NUM_SCALES: gl.constexpr, N_PER_TILE: gl.constexpr
):
    # The e8m0 twin, used for Q. Grouping N_PER_TILE rows puts each register
    # element's 64 lanes on one contiguous 64-byte run.
    return (
        (offs_n % N_PER_TILE)
        + offs_s * N_PER_TILE
        + (offs_n // N_PER_TILE) * (N_PER_TILE * NUM_SCALES)
    )


@gluon.jit
def _token_major_offsets(offs_d, offs_n, HEAD_BYTES: gl.constexpr):
    return offs_n * HEAD_BYTES + offs_d


@gluon.jit
def _token_major_scale_offsets(offs_n, offs_s, NUM_SCALES: gl.constexpr):
    return offs_n * NUM_SCALES + offs_s


@gluon.constexpr_function
def _fold_plan(linear_layout, num_heads, block_kv, num_chains):
    # Split the head bits into the ones a lane holds in registers, which fold
    # with an FMA, and the ones living across lanes, which cross once.
    assert num_chains >= 1 and (num_chains & (num_chains - 1)) == 0
    head_bits = num_heads.bit_length() - 1
    chain_bits = num_chains.bit_length() - 1
    reg_bases = [tuple(b) for b in linear_layout.reg_bases]
    summed, folded = [], []
    for bit in range(head_bits):
        stride = 1 << (head_bits - 1 - bit)
        (folded if (stride, 0) in reg_bases else summed).append(bit)
    assert chain_bits <= len(folded), f"num_chains={num_chains} needs more folded bits"
    depth = len(folded) - chain_bits
    return (
        tuple([2] * head_bits + [block_kv]),
        tuple(summed + [head_bits] + folded[:chain_bits] + folded[chain_bits:]),
        tuple([1 << len(summed), block_kv, num_chains] + [2] * depth),
        depth,
        1 << depth,
    )


# ends arrives as a row slice, so its alignment moves; it is read once per row
@triton.jit(do_not_specialize_on_alignment=["ends_ptr"])
def _prepare_candidates_kernel(
    ids_ptr,
    ends_ptr,
    bt_ptr,
    pos_ptr,
    cu_ptr,
    voff_ptr,
    soff_ptr,
    stride_ids,
    stride_bt,
    K: tl.constexpr,
    C: tl.constexpr,
    PAGE: tl.constexpr,
    NPT: tl.constexpr,
    HEAD_BYTES: tl.constexpr,
    NUM_SCALES: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    KVS_STRIDE: tl.constexpr,
    KW: tl.constexpr,
    SU: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
    OFF64: tl.constexpr,
):
    """Ranked block ids -> the walk's candidate list, one row per program.

    The rank emits by score and the walk needs by position, so the sort is
    here; everything after it must stay in step with the tests' block_offsets.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, K)
    ids = tl.load(ids_ptr + row * stride_ids + cols)
    end = tl.load(ends_ptr + row)
    nblocks = (end + C - 1) // C

    ok = (ids >= 0) & (ids < nblocks)
    n_valid = tl.sum(ok.to(tl.int32), axis=0)
    max_id = tl.max(tl.where(ok, ids, -1), axis=0)
    key = tl.sort(tl.where(ok, ids, 0x7FFFFFFF))

    # Padding repeats the last legal block, so a short row still walks blocks
    # it owns and row_ends drops the slots.
    last = (nblocks - 1) * C
    pos = tl.where(key == 0x7FFFFFFF, last, key * C)
    tl.store(pos_ptr + row * K + cols, pos)
    tail = tl.minimum(C, end - max_id * C)
    tl.store(cu_ptr + row, tl.where(n_valid > 0, (n_valid - 1) * C + tail, 0))

    page = pos // PAGE
    t0 = pos % PAGE
    pid = tl.load(bt_ptr + row * stride_bt + page).to(tl.int64)
    bn = (t0 % NPT) * KW + (t0 // NPT) * (NPT * HEAD_BYTES)
    bs = (t0 % NPT) * TOK_STRIDE + (t0 // NPT) * (NPT * NUM_SCALES)
    v = (pid * KV_STRIDE + bn) // KW
    sc = (pid * KVS_STRIDE + bs) // SU
    # i32 while it reaches; a pool past 4 GiB of scale bytes needs the width
    tl.store(voff_ptr + row * K + cols, v if OFF64 else v.to(tl.int32))
    tl.store(soff_ptr + row * K + cols, sc if OFF64 else sc.to(tl.int32))


@gluon.jit
def _max_nan(a, b):
    return gl.maximum(a, b, propagate_nan=_NAN_ALL)


@gluon.jit
def _relu(x, RELU_ADD: gl.constexpr):
    # RELU_ADD spells relu as (x + |x|)/2: one v_add_f32 with an abs source
    # modifier where max(x, 0) is a v_maximum3_f32, same instruction count and
    # registers, just a faster (VALU overlap). Needs to be paired with the
    # scalar FMA fold to stop the v_add getting packed: v_pk_add_f32 has no abs
    # modifier.
    if RELU_ADD:
        return x + gl.abs(x)
    return gl.maximum(x, 0, propagate_nan=_NAN_ALL)


@gluon.jit
def _fma_unpacked(a, b, c):
    # a * b + c as one v_fma_f32 the SLP vectorizer cannot pair. Packed FP32
    # cannot co-issue in an MFMA shadow on gfx950 and needs even-aligned pairs
    return gl.inline_asm_elementwise(
        "v_fma_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        [a, b, c],
        dtype=gl.float32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _split_leaf(x, IDX: gl.constexpr, DEPTH: gl.constexpr):
    for bit in gl.static_range(0, DEPTH):
        lo, hi = x.split()
        x = lo if (IDX // (2**bit)) % 2 == 0 else hi
    return x


@gluon.jit
def _fold_heads(
    s,
    w_col,
    NUM_HEADS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    mfma_layout: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
    FOLD_ASM: gl.constexpr,
    acc=None,
    HAS_ACC: gl.constexpr = False,
):
    # sum_h s[h, k] * w[h] over the head bits that live in registers. Returns
    # the pre-cross-lane accumulator, so a caller folding several head chunks
    # pays the lane crossing once.
    ll: gl.constexpr = gl.to_linear_layout(mfma_layout, [NUM_HEADS, BLOCK_KV])
    plan: gl.constexpr = _fold_plan(ll, NUM_HEADS, BLOCK_KV, NUM_CHAINS)
    shape: gl.constexpr = plan[0]
    order: gl.constexpr = plan[1]
    folded: gl.constexpr = plan[2]
    depth: gl.constexpr = plan[3]
    leaves: gl.constexpr = plan[4]

    w = w_col.broadcast_to([NUM_HEADS, BLOCK_KV])
    s = s.reshape(shape).permute(order).reshape(folded)
    w = w.reshape(shape).permute(order).reshape(folded)
    if not HAS_ACC:
        acc = _split_leaf(s, 0, depth) * _split_leaf(w, 0, depth)
    for i in gl.static_range(0, leaves):
        if HAS_ACC or i > 0:
            a = _split_leaf(s, i, depth)
            b = _split_leaf(w, i, depth)
            acc = _fma_unpacked(a, b, acc) if FOLD_ASM else gl.fma(a, b, acc)
    return acc


@gluon.jit
def _fold_tail(acc, mfma_layout: gl.constexpr):
    return gl.convert_layout(
        gl.sum(gl.sum(acc, axis=2), axis=0), gl.SliceLayout(0, mfma_layout)
    )


@aggregate
@strip_annotate
class Config:
    NUM_HEADS: gl.constexpr
    HEAD_BYTES: gl.constexpr
    NUM_SCALES: gl.constexpr
    PAGE_SIZE: gl.constexpr
    BLOCK_KV: gl.constexpr
    BLOCK_M: gl.constexpr
    KV_PAGE_STRIDE: gl.constexpr
    KVS_PAGE_STRIDE: gl.constexpr
    NUM_WARPS: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    DEPTH: gl.constexpr
    UNROLL: gl.constexpr
    PAGE_PIPE: gl.constexpr
    M_CHUNK: gl.constexpr
    A_ROWS: gl.constexpr
    N_CHUNK: gl.constexpr
    NUM_CHAINS: gl.constexpr
    FOLD_ASM: gl.constexpr
    RELU_ADD: gl.constexpr
    PRESHUFFLE: gl.constexpr
    SCALE_MODE: gl.constexpr
    USE_BUFFER_LOAD: gl.constexpr
    RELAXED_STORE: gl.constexpr
    VARLEN: gl.constexpr
    GATHER: gl.constexpr
    GATHER_BLOCK: gl.constexpr
    GATHER_PIPE: gl.constexpr
    BSCORE: gl.constexpr
    STORE_LOGITS: gl.constexpr
    BSCORE_BLOCK: gl.constexpr
    BLOCKS_PER_TILE: gl.constexpr
    N_PER_TILE: gl.constexpr
    D_PER_TILE: gl.constexpr
    Q_CACHE: gl.constexpr
    KV_CACHE: gl.constexpr
    mfma_layout: gl.constexpr
    dot_a: gl.constexpr
    dot_b: gl.constexpr
    q_scale_layout: gl.constexpr
    PAGE_MASK: gl.constexpr
    PAGE_SHIFT: gl.constexpr
    kv_scale_layout: gl.constexpr
    blocked_kv: gl.constexpr
    shared_kv: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        NUM_HEADS,
        HEAD_SIZE,
        PAGE_SIZE,
        BLOCK_KV,
        BLOCK_M,
        KV_PAGE_STRIDE,
        KVS_PAGE_STRIDE,
        NUM_WARPS,
        NUM_BUFFERS,
        DEPTH,
        UNROLL,
        PAGE_PIPE,
        M_CHUNK,
        NUM_CHAINS,
        FOLD_ASM,
        RELU_ADD,
        PRESHUFFLE,
        SCALE_MODE,
        USE_BUFFER_LOAD,
        RELAXED_STORE,
        VARLEN,
        MFMA_NONK_DIM,
        HAS_KV_SPLIT,
        KV_REREAD,
        GATHER,
        GATHER_BLOCK,
        GATHER_PIPE,
        BSCORE,
        BSCORE_BLOCK,
    ):
        self.NUM_HEADS = gl.constexpr(NUM_HEADS)
        self.HEAD_BYTES = gl.constexpr(HEAD_SIZE // 2)
        self.NUM_SCALES = gl.constexpr(HEAD_SIZE // SCALE_GROUP.value)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        assert PAGE_SIZE & (PAGE_SIZE - 1) == 0, "page_size must be a power of two"
        # Split the position with a mask and a shift. `%` and `//` are signed,
        # and a signed divide by a power of two carries a sign correction the
        # value can never need, cheaper SALU
        self.PAGE_MASK = gl.constexpr(PAGE_SIZE - 1)
        self.PAGE_SHIFT = gl.constexpr((PAGE_SIZE - 1).bit_length())
        self.BLOCK_KV = gl.constexpr(BLOCK_KV)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.KV_PAGE_STRIDE = gl.constexpr(KV_PAGE_STRIDE)
        self.KVS_PAGE_STRIDE = gl.constexpr(KVS_PAGE_STRIDE)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.DEPTH = gl.constexpr(DEPTH)
        self.UNROLL = gl.constexpr(UNROLL)
        self.PAGE_PIPE = gl.constexpr(PAGE_PIPE)
        self.M_CHUNK = gl.constexpr(M_CHUNK)
        self.NUM_CHAINS = gl.constexpr(NUM_CHAINS)
        self.FOLD_ASM = gl.constexpr(FOLD_ASM)
        self.RELU_ADD = gl.constexpr(RELU_ADD)
        self.PRESHUFFLE = gl.constexpr(PRESHUFFLE)
        # e8m0 byte order. 1 reads a whole MFMA tile as one wide run, 0 costs a
        # buffer_load_ubyte per group. Both are gatherable at any granularity up
        # to N_PER_TILE, so the choice is the dense reader's, not the gather's.
        self.SCALE_MODE = gl.constexpr(SCALE_MODE)
        self.USE_BUFFER_LOAD = gl.constexpr(USE_BUFFER_LOAD)
        self.RELAXED_STORE = gl.constexpr(RELAXED_STORE)
        self.VARLEN = gl.constexpr(VARLEN)
        # Walk a candidate list in GATHER_BLOCK-token units
        # instead of BLOCK_KV contiguous positions. GATHER_PIPE reads that list
        # an iteration early, the way PAGE_PIPE reads the block table.
        self.GATHER = gl.constexpr(GATHER)
        self.GATHER_BLOCK = gl.constexpr(GATHER_BLOCK)
        self.GATHER_PIPE = gl.constexpr(GATHER_PIPE)
        # A per-row max over every BSCORE_BLOCK columns, so the indexer's block
        # scores cost a reduce in the walk rather than a second pass:
        #   0  off
        #   1  maxima beside the logits
        #   2  maxima instead of them, so no [rows, ctx] fp32 tensor exists
        # 2 writes the same values as 1; it only drops the logits store.
        self.BSCORE = gl.constexpr(BSCORE)
        self.STORE_LOGITS = gl.constexpr(BSCORE != 2)
        self.BSCORE_BLOCK = gl.constexpr(BSCORE_BLOCK)
        self.BLOCKS_PER_TILE = gl.constexpr(BLOCK_KV // BSCORE_BLOCK)
        self.N_PER_TILE = gl.constexpr(MFMA_NONK_DIM)
        self.D_PER_TILE = gl.constexpr(K_WIDTH.value)
        # A split re-reads Q, so keep it in L1. The KV stream takes .cg when one
        # row block reads a sequence's pages and every byte is touched once;
        # splits do not count, they read disjoint ranges.
        self.Q_CACHE = gl.constexpr("" if HAS_KV_SPLIT else ".cg")
        self.KV_CACHE = gl.constexpr("" if KV_REREAD else ".cg")

        mfma = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 128] if MFMA_NONK_DIM == 16 else [32, 32, 64],
            transposed=False,
            warps_per_cta=[1, NUM_WARPS],
        )
        self.mfma_layout = gl.constexpr(mfma)
        self.dot_a = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=K_WIDTH.value)
        )
        self.dot_b = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=K_WIDTH.value)
        )

        A_ROWS = M_CHUNK if M_CHUNK > 0 else NUM_HEADS
        self.A_ROWS = gl.constexpr(A_ROWS)
        self.N_CHUNK = gl.constexpr(NUM_HEADS // A_ROWS)
        self.q_scale_layout = gl.constexpr(
            gl.amd.cdna4.get_mfma_scale_layout(
                self.dot_a.value, [A_ROWS, HEAD_SIZE // SCALE_GROUP.value]
            )
        )
        self.kv_scale_layout = gl.constexpr(
            gl.amd.cdna4.get_mfma_scale_layout(
                self.dot_b.value, [BLOCK_KV, HEAD_SIZE // SCALE_GROUP.value]
            )
        )

        # Only the unshuffled path stages through LDS
        blocked, shared = _staging_layouts(
            HEAD_SIZE // 2, BLOCK_KV, NUM_WARPS, WARP_SIZE.value
        )
        self.blocked_kv = gl.constexpr(blocked)
        self.shared_kv = gl.constexpr(shared)


@gluon.jit
def _load_q_chunk(
    cfg, q_ptr, qs_ptr, w_ptr, HEAD_OFFSET: gl.constexpr, ROWS: gl.constexpr
):
    # The query row rides in the pointer, not the offsets, so the i32 buffer
    # offset spans one row rather than the whole tensor.
    hs = gl.arange(0, ROWS, layout=gl.SliceLayout(1, cfg.q_scale_layout))[:, None]
    ss = gl.arange(0, cfg.NUM_SCALES, layout=gl.SliceLayout(0, cfg.q_scale_layout))[
        None, :
    ]
    # Q is token major and converted here
    lay: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[WARP_SIZE // (cfg.HEAD_BYTES // 16), cfg.HEAD_BYTES // 16],
        warps_per_cta=[cfg.NUM_WARPS, 1],
        order=[1, 0],
    )
    offs_h = gl.arange(0, ROWS, layout=gl.SliceLayout(1, lay)) + HEAD_OFFSET
    offs_d = gl.arange(0, cfg.HEAD_BYTES, layout=gl.SliceLayout(0, lay))
    q = gl.convert_layout(
        gl.amd.cdna4.buffer_load(
            ptr=q_ptr,
            offsets=(offs_h * cfg.HEAD_BYTES)[:, None] + offs_d[None, :],
            cache=cfg.Q_CACHE,
        ),
        cfg.dot_a,
    )
    qs = gl.amd.cdna4.buffer_load(
        ptr=qs_ptr,
        offsets=((hs + HEAD_OFFSET) * cfg.NUM_SCALES) + ss,
        cache=cfg.Q_CACHE,
    )

    hw = gl.arange(0, ROWS, layout=gl.SliceLayout(1, cfg.mfma_layout))
    w = gl.amd.cdna4.buffer_load(
        ptr=w_ptr, offsets=(hw + HEAD_OFFSET)[:, None], cache=cfg.Q_CACHE
    )
    if cfg.RELU_ADD:
        # The /2 of (x + |x|)/2. Once per row block, outside the KV walk.
        w = w * 0.5
    return q, qs, w


@gluon.jit
def _load_q_row(cfg, q_ptr, qs_ptr, w_ptr):
    # if/else rather than an early return: Gluon traces the fall-through of a
    # constexpr if even when it cannot be reached.
    if cfg.M_CHUNK > 0:
        q, qs, w = (), (), ()
        for c in gl.static_range(0, cfg.N_CHUNK):
            a, b, d = _load_q_chunk(
                cfg, q_ptr, qs_ptr, w_ptr, c * cfg.M_CHUNK, cfg.M_CHUNK
            )
            q, qs, w = q + (a,), qs + (b,), w + (d,)
        return q, qs, w
    else:
        return _load_q_chunk(cfg, q_ptr, qs_ptr, w_ptr, 0, cfg.NUM_HEADS)


@aggregate
@strip_annotate
class KVState:
    """The cache, the block table and the offsets inside a page.

    Address is page_id * PAGE_STRIDE + (tile_pos % PAGE_SIZE) * PER_TOKEN
    """

    cfg: Config
    KV_ptr: gl.tensor
    kv_scales_ptr: gl.tensor
    blk_ptr: gl.tensor
    last_page_row: gl.tensor
    val_offsets: gl.tensor
    scale_offsets: gl.tensor
    gv_ptr: gl.tensor
    gs_ptr: gl.tensor
    last_blk: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        KV_ptr,
        kv_scales_ptr,
        blk_ptr,
        last_page_row,
        val_offsets,
        scale_offsets,
        gv_ptr,
        gs_ptr,
        last_blk,
    ):
        self.cfg = cfg
        self.KV_ptr = KV_ptr
        self.kv_scales_ptr = kv_scales_ptr
        self.blk_ptr = blk_ptr
        self.last_page_row = last_page_row
        self.val_offsets = val_offsets
        self.scale_offsets = scale_offsets
        self.gv_ptr = gv_ptr
        self.gs_ptr = gs_ptr
        self.last_blk = last_blk

    @gluon.jit
    def initialize(
        cfg,
        KV_ptr,
        kv_scales_ptr,
        blk_ptr,
        last_page_row,
        gv_ptr,
        gs_ptr,
        last_blk,
        val_layout: gl.constexpr,
    ):
        offs_d = gl.arange(0, cfg.HEAD_BYTES, layout=gl.SliceLayout(1, val_layout))[
            :, None
        ]
        offs_n = gl.arange(0, cfg.BLOCK_KV, layout=gl.SliceLayout(0, val_layout))[
            None, :
        ]
        # Do the loop invariant calculation early
        if cfg.GATHER:
            offs_n = offs_n % cfg.GATHER_BLOCK
        if cfg.PRESHUFFLE:
            val = _shuffled_offsets(
                offs_d, offs_n, cfg.HEAD_BYTES, cfg.N_PER_TILE, cfg.D_PER_TILE
            )
        else:
            val = _token_major_offsets(offs_d, offs_n, cfg.HEAD_BYTES)

        sn = gl.arange(0, cfg.BLOCK_KV, layout=gl.SliceLayout(1, cfg.kv_scale_layout))[
            :, None
        ]
        ss = gl.arange(
            0, cfg.NUM_SCALES, layout=gl.SliceLayout(0, cfg.kv_scale_layout)
        )[None, :]
        if cfg.GATHER:
            sn = sn % cfg.GATHER_BLOCK
        if cfg.PRESHUFFLE:
            sc = _shuffled_scale_offsets(sn, ss, cfg.NUM_SCALES, cfg.N_PER_TILE)
        else:
            sc = _token_major_scale_offsets(sn, ss, cfg.NUM_SCALES)
        return KVState(
            cfg,
            KV_ptr,
            kv_scales_ptr,
            blk_ptr,
            last_page_row,
            val,
            sc,
            gv_ptr,
            gs_ptr,
            last_blk,
        )

    @gluon.jit
    def gather_base(self, tile_pos, base_ptr, layout: gl.constexpr, AXIS: gl.constexpr):
        """This tile's per-column base offset.

        One int32 per candidate block, already holding the page address plus
        the block's own offset inside its page, so the walk carries no
        block -> position -> block table -> page chain: one load and one
        broadcast add.

        `tile_pos` is a slot index here, not a KV position, the walk is over
        the compact candidate list and the store columns are compact too.
        Clamped rather than masked, store mask handles the rest
        """
        cols = gl.arange(0, self.cfg.BLOCK_KV, layout=gl.SliceLayout(AXIS, layout))
        # Split so the tile term stays scalar
        blk = gl.minimum(
            tile_pos // self.cfg.GATHER_BLOCK + cols // self.cfg.GATHER_BLOCK,
            self.last_blk,
        )
        # Buffer, not pointer: the row base is scalar and folds into the
        # descriptor, so the per-column term stays i32
        return gl.amd.cdna4.buffer_load(ptr=base_ptr, offsets=blk)

    @gluon.jit
    def gather_scale_base(self, tile_pos):
        """The scale stream's per-column base in the wide load's index space.

        The preshuffled scale tile is read as [N_HI, S_LO, N_PER_TILE, S_HI]
        rather than [BLOCK_KV, NUM_SCALES] (see _load_scales_wide), so its
        candidate index has to be built on that shape: the token is
        b0 * N_PER_TILE + b2, and the block it belongs to is that over
        GATHER_BLOCK. Broadcast over the two scale axes
        """
        cfg: gl.constexpr = self.cfg
        S_LO: gl.constexpr = WARP_SIZE // cfg.N_PER_TILE
        S_HI: gl.constexpr = cfg.NUM_SCALES // S_LO
        N_HI: gl.constexpr = cfg.BLOCK_KV // cfg.N_PER_TILE
        lr: gl.constexpr = _scale_group_layout(
            cfg.kv_scale_layout, cfg.N_PER_TILE, S_LO, S_HI, N_HI
        )
        b0 = gl.arange(0, N_HI, layout=_axis_layout(lr, 0, 4))[:, None, None, None]
        b2 = gl.arange(0, cfg.N_PER_TILE, layout=_axis_layout(lr, 2, 4))[
            None, None, :, None
        ]
        blk = gl.minimum(
            tile_pos // cfg.GATHER_BLOCK
            + (b0 * cfg.N_PER_TILE + b2) // cfg.GATHER_BLOCK,
            self.last_blk,
        )
        # A buffer load, for the reason gather_base gives.
        return gl.amd.cdna4.buffer_load(ptr=self.gs_ptr, offsets=blk)

    @gluon.jit
    def gather_tok(self, tile_pos, val_layout: gl.constexpr):
        """Both streams' per-column bases for one tile, as a carryable pair.

        Split out from the loads for the same reason page_token is: it is a
        dependent memory load standing directly in front of every KV address in
        the tile. GATHER_PIPE is what lets the walk issue it an iteration early.
        """
        if self.cfg.GATHER:
            # The scale base has to be built on whichever index space its load
            # uses: the four-dimensional one for the wide mode-1 read, the
            # plain [BLOCK_KV, NUM_SCALES] one otherwise.
            if self.cfg.PRESHUFFLE and self.cfg.SCALE_MODE == 1:
                return (
                    self.gather_base(tile_pos, self.gv_ptr, val_layout, 0),
                    self.gather_scale_base(tile_pos),
                )
            else:
                return (
                    self.gather_base(tile_pos, self.gv_ptr, val_layout, 0),
                    self.gather_base(
                        tile_pos, self.gs_ptr, self.cfg.kv_scale_layout, 1
                    ),
                )
        else:
            return tile_pos * 0, tile_pos * 0

    @gluon.jit
    def page_token(self, tile_pos):
        # Split out from the loads so the walk can issue it an iteration early
        if self.cfg.GATHER:
            # The candidate list already carries the page, so there is no table
            # read left to hoist.
            return tile_pos * 0
        else:
            return gl.load(
                self.blk_ptr
                + gl.minimum(tile_pos >> self.cfg.PAGE_SHIFT, self.last_page_row)
            )

    @gluon.jit
    def tile_ptr(
        self, base, tile_pos, page, PAGE_STRIDE: gl.constexpr, PER_TOKEN: gl.constexpr
    ):
        # 64-bit on purpose: one scalar multiply per tile keeps a cache far
        # above 2 GiB addressable while every per-element offset stays i32.
        return base + (
            page.to(gl.int64) * PAGE_STRIDE
            + (tile_pos & self.cfg.PAGE_MASK).to(gl.int64) * PER_TOKEN
        )

    @gluon.jit
    def page_off(
        self, tile_pos, page, PAGE_STRIDE: gl.constexpr, PER_TOKEN: gl.constexpr
    ):
        # The same address as a scalar to add into an offsets tensor, which is
        # what the LDS path needs; see LDSLoader.issue.
        in_page = tile_pos & self.cfg.PAGE_MASK
        if self.cfg.USE_BUFFER_LOAD:
            return page * PAGE_STRIDE + in_page * PER_TOKEN
        else:
            return page.to(gl.int64) * PAGE_STRIDE + in_page.to(gl.int64) * PER_TOKEN

    @gluon.jit
    def scales(self, tile_pos, page, gtok):
        """One [BLOCK_KV, NUM_SCALES] e8m0 tile in the MFMA's scale layout.

        Never goes through LDS on either path -- 256 bytes at head_size 128,
        loaded straight into the distribution the scaled MFMA wants.
        """
        if self.cfg.GATHER:
            # A per-column page, so the whole address goes in the offsets.
            # U is the value stream's D_PER_TILE move: the list is stored in
            # units of U so the multiply hands the 2-byte alignment back, and
            # without it each scale run splits into two buffer_load_ubyte.
            U: gl.constexpr = (
                2
                if (self.cfg.GATHER_BLOCK % 2 == 0 and self.cfg.NUM_SCALES % 2 == 0)
                else 1
            )
            if self.cfg.USE_BUFFER_LOAD:
                gs = gtok[1] * U
            else:
                gs = gtok[1].to(gl.int64) * U
            if self.cfg.PRESHUFFLE and self.cfg.SCALE_MODE == 1:
                return _load_scales_wide(self.cfg, self.kv_scales_ptr, gs)
            elif self.cfg.USE_BUFFER_LOAD:
                return gl.amd.cdna4.buffer_load(
                    ptr=self.kv_scales_ptr,
                    offsets=self.scale_offsets + gs[:, None],
                    cache=self.cfg.KV_CACHE,
                )
            else:
                return gl.load(
                    self.kv_scales_ptr + self.scale_offsets + gs[:, None],
                    cache_modifier=self.cfg.KV_CACHE,
                )
        elif self.cfg.PRESHUFFLE and self.cfg.SCALE_MODE == 1:
            return _load_scales_wide(
                self.cfg,
                self.tile_ptr(
                    self.kv_scales_ptr,
                    tile_pos,
                    page,
                    self.cfg.KVS_PAGE_STRIDE,
                    self.cfg.NUM_SCALES,
                ),
                0,
            )
        else:
            ptr = self.tile_ptr(
                self.kv_scales_ptr,
                tile_pos,
                page,
                self.cfg.KVS_PAGE_STRIDE,
                self.cfg.NUM_SCALES,
            )
            return gl.amd.cdna4.buffer_load(
                ptr=ptr, offsets=self.scale_offsets, cache=self.cfg.KV_CACHE
            )


@gluon.jit
def _load_scales_wide(cfg, ptr, gbase):
    """The whole scale tile, read over the run the stored order makes contiguous.

    `gbase` is the gather's per-candidate-block base on this same 4-D index
    space, or a scalar 0 when the walk is contiguous and the page rides in
    `ptr` instead. The stored order puts the token axis at stride S_HI inside a
    group of N_PER_TILE, so a candidate block of at most N_PER_TILE tokens is
    still one broadcast add, which is why the gather does not need a byte
    order of its own here.

    Grouping one MFMA tile keeps the order independent of BLOCK_KV and caps the
    run at S_HI bytes. Keep both page strides constexpr or the
    vectorizer loses the divisibility and every load collapses to a ubyte.
    """
    S_LO: gl.constexpr = WARP_SIZE // cfg.N_PER_TILE
    S_HI: gl.constexpr = cfg.NUM_SCALES // S_LO
    N_HI: gl.constexpr = cfg.BLOCK_KV // cfg.N_PER_TILE
    lr: gl.constexpr = _scale_group_layout(
        cfg.kv_scale_layout, cfg.N_PER_TILE, S_LO, S_HI, N_HI
    )
    b0 = gl.arange(0, N_HI, layout=_axis_layout(lr, 0, 4))[:, None, None, None]
    b1 = gl.arange(0, S_LO, layout=_axis_layout(lr, 1, 4))[None, :, None, None]
    b2 = gl.arange(0, cfg.N_PER_TILE, layout=_axis_layout(lr, 2, 4))[
        None, None, :, None
    ]
    b3 = gl.arange(0, S_HI, layout=_axis_layout(lr, 3, 4))[None, None, None, :]
    if cfg.GATHER:
        # The group term is in gbase, and the token index is block-local
        offs = (b2 % cfg.GATHER_BLOCK + b1 * cfg.N_PER_TILE) * S_HI + b3 + gbase
    else:
        offs = (
            b0 * (cfg.N_PER_TILE * cfg.NUM_SCALES)
            + (b2 + b1 * cfg.N_PER_TILE) * S_HI
            + b3
        )
    if cfg.GATHER and not cfg.USE_BUFFER_LOAD:
        raw = gl.load(ptr + offs, cache_modifier=cfg.KV_CACHE)
    else:
        raw = gl.amd.cdna4.buffer_load(ptr=ptr, offsets=offs, cache=cfg.KV_CACHE)
    return (
        raw.reshape([N_HI, S_LO, cfg.N_PER_TILE, S_HI])
        .permute((0, 2, 3, 1))
        .reshape([cfg.BLOCK_KV, cfg.NUM_SCALES])
    )


@aggregate
@strip_annotate
class RegLoader:
    """Preshuffled page straight into dot-operand registers. No LDS, no barrier,
    no ds_read: HBM already carries the byte order the matrix core wants."""

    st: KVState

    @gluon.constexpr_function
    def __init__(self, st):
        self.st = st

    @gluon.jit
    def initialize(
        cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row, gv_ptr, gs_ptr, last_blk
    ):
        return RegLoader(
            KVState.initialize(
                cfg,
                KV_ptr,
                kv_scales_ptr,
                blk_ptr,
                last_page_row,
                gv_ptr,
                gs_ptr,
                last_blk,
                cfg.dot_b,
            )
        )

    @gluon.jit
    def page_token(self, tile_pos):
        return self.st.page_token(tile_pos)

    @gluon.jit
    def gather_tok(self, tile_pos):
        return self.st.gather_tok(tile_pos, self.st.cfg.dot_b)

    @gluon.jit
    def scales(self, tile_pos, page, gtok):
        return self.st.scales(tile_pos, page, gtok)

    @gluon.jit
    def values(self, tile_pos, page, gtok):
        cfg: gl.constexpr = self.st.cfg
        if cfg.GATHER:
            # An offset arriving from memory carries no provable alignment, and
            # Triton then emits one buffer_load_ubyte per byte. Storing the list
            # in D_PER_TILE units and multiplying back hands the divisibility
            # over.
            if cfg.USE_BUFFER_LOAD:
                return gl.amd.cdna4.buffer_load(
                    ptr=self.st.KV_ptr,
                    offsets=self.st.val_offsets + (gtok[0] * cfg.D_PER_TILE)[None, :],
                    cache=cfg.KV_CACHE,
                )
            else:
                return gl.load(
                    self.st.KV_ptr
                    + self.st.val_offsets
                    + (gtok[0].to(gl.int64) * cfg.D_PER_TILE)[None, :],
                    cache_modifier=cfg.KV_CACHE,
                )
        else:
            return gl.amd.cdna4.buffer_load(
                ptr=self.st.tile_ptr(
                    self.st.KV_ptr, tile_pos, page, cfg.KV_PAGE_STRIDE, cfg.HEAD_BYTES
                ),
                offsets=self.st.val_offsets,
                cache=cfg.KV_CACHE,
            )


@aggregate
@strip_annotate
class LDSLoader:
    """Token-major page into LDS by async copy, then one ds_read into the dot
    operand. The path for an unshuffled cache; LDS is what rearranges the tile.
    """

    st: KVState
    kv_shared: gl.shared_memory_descriptor

    @gluon.constexpr_function
    def __init__(self, st, kv_shared):
        self.st = st
        self.kv_shared = kv_shared

    @gluon.jit
    def initialize(
        cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row, gv_ptr, gs_ptr, last_blk
    ):
        st = KVState.initialize(
            cfg,
            KV_ptr,
            kv_scales_ptr,
            blk_ptr,
            last_page_row,
            gv_ptr,
            gs_ptr,
            last_blk,
            cfg.blocked_kv,
        )
        shared = gl.allocate_shared_memory(
            KV_ptr.type.element_ty,
            [cfg.NUM_BUFFERS, cfg.HEAD_BYTES, cfg.BLOCK_KV],
            layout=cfg.shared_kv,
        )
        return LDSLoader(st, shared)

    @gluon.jit
    def page_token(self, tile_pos):
        return self.st.page_token(tile_pos)

    @gluon.jit
    def gather_tok(self, tile_pos):
        return self.st.gather_tok(tile_pos, self.st.cfg.blocked_kv)

    @gluon.jit
    def scales(self, tile_pos, page, gtok):
        return self.st.scales(tile_pos, page, gtok)

    @gluon.jit
    def issue(self, tile_pos, page, gtok, buffer_id):
        """Start one tile's global-to-LDS copy. No mask: see page_token.

        The page rides in the offsets, not the base pointer:
        buffer_load_to_shared crashes the backend on a runtime-varying base.
        """
        cfg: gl.constexpr = self.st.cfg
        dest = self.kv_shared.index(buffer_id)
        if cfg.GATHER:
            if cfg.USE_BUFFER_LOAD:
                offsets = self.st.val_offsets + (gtok[0] * cfg.D_PER_TILE)[None, :]
            else:
                offsets = (
                    self.st.val_offsets
                    + (gtok[0].to(gl.int64) * cfg.D_PER_TILE)[None, :]
                )
        else:
            offsets = self.st.val_offsets + self.st.page_off(
                tile_pos, page, cfg.KV_PAGE_STRIDE, cfg.HEAD_BYTES
            )
        if cfg.USE_BUFFER_LOAD:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                dest, self.st.KV_ptr, offsets, cache_modifier=cfg.KV_CACHE
            )
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                dest, self.st.KV_ptr + offsets, cache_modifier=cfg.KV_CACHE
            )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def consume(self, wait_count, buffer_id):
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return self.kv_shared.index(buffer_id).load(layout=self.st.cfg.dot_b)


@gluon.jit
def _dot(cfg, q, q_scale, k, k_scale, ROWS: gl.constexpr):
    acc = gl.zeros([ROWS, cfg.BLOCK_KV], dtype=gl.float32, layout=cfg.mfma_layout)
    return gl.amd.cdna4.mfma_scaled(
        a=q,
        a_scale=q_scale,
        a_format="e2m1",
        b=k,
        b_scale=k_scale,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _row_logits(cfg, q, q_scale, w, k, k_scale):
    if cfg.M_CHUNK > 0:
        # Fold each head group as its MFMA retires, so one group's accumulator
        # is live at a time and the lane crossing is paid once for the row.
        acc = None
        for c in gl.static_range(0, cfg.N_CHUNK):
            s = _dot(cfg, q[c], q_scale[c], k, k_scale, cfg.M_CHUNK)
            acc = _fold_heads(
                _relu(s, cfg.RELU_ADD),
                w[c],
                cfg.M_CHUNK,
                cfg.BLOCK_KV,
                cfg.mfma_layout,
                cfg.NUM_CHAINS,
                cfg.FOLD_ASM,
                acc,
                c > 0,
            )
        return _fold_tail(acc, cfg.mfma_layout)
    else:
        s = _dot(cfg, q, q_scale, k, k_scale, cfg.NUM_HEADS)
        acc = _fold_heads(
            _relu(s, cfg.RELU_ADD),
            w,
            cfg.NUM_HEADS,
            cfg.BLOCK_KV,
            cfg.mfma_layout,
            cfg.NUM_CHAINS,
            cfg.FOLD_ASM,
        )
        return _fold_tail(acc, cfg.mfma_layout)


@aggregate
@strip_annotate
class Program:
    """This workgroup's rows and the window they own."""

    cfg: Config
    out_ptr: gl.tensor
    stride_s: gl.tensor
    stride_k: gl.tensor
    tile_lo: gl.tensor
    tile_hi: gl.tensor
    store_hi: gl.tensor
    bs_ptr: gl.tensor
    bs_stride_s: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        out_ptr,
        stride_s,
        stride_k,
        tile_lo,
        tile_hi,
        store_hi,
        bs_ptr,
        bs_stride_s,
    ):
        self.cfg = cfg
        self.out_ptr = out_ptr
        self.stride_s = stride_s
        self.stride_k = stride_k
        self.tile_lo = tile_lo
        self.tile_hi = tile_hi
        self.store_hi = store_hi
        self.bs_ptr = bs_ptr
        self.bs_stride_s = bs_stride_s

    @gluon.jit
    def initialize(
        cfg,
        out_ptr,
        stride_s,
        stride_k,
        context_len,
        block_end,
        split_id,
        num_kv_splits,
        slice_idx,
        num_slices,
        bs_ptr,
        bs_stride_s,
        HAS_KV_SPLIT: gl.constexpr,
        DYNAMIC: gl.constexpr,
    ):
        if cfg.GATHER:
            # block_end counts candidate slots here, not key positions: the
            # walk is over a compact list the host already causally filtered,
            # so there is no context length to clamp it against.
            keys = gl.maximum(block_end, 0)
        else:
            # Walk only as far as the furthest of the block's rows can attend
            keys = gl.minimum(context_len, block_end)
        n_tiles = gl.maximum((keys + cfg.BLOCK_KV - 1) // cfg.BLOCK_KV, 0)
        if DYNAMIC:
            # Slice slice_idx of num_slices, converted against this layer's own
            # tile count
            per = (n_tiles + gl.maximum(num_slices, 1) - 1) // gl.maximum(num_slices, 1)
            tile_lo = slice_idx * per
            tile_hi = gl.maximum(gl.minimum(tile_lo + per, n_tiles), tile_lo)
        elif HAS_KV_SPLIT:
            per = (n_tiles + num_kv_splits - 1) // num_kv_splits
            tile_lo = split_id * per
            tile_hi = gl.minimum(tile_lo + per, n_tiles)
        else:
            tile_lo: gl.int32 = 0
            tile_hi = n_tiles
        # Stops spilling stores on each other's write region
        # The compact end under GATHER: output column j is candidate slot j.
        store_hi = gl.minimum(
            keys if cfg.GATHER else context_len, tile_hi * cfg.BLOCK_KV
        )
        return Program(
            cfg,
            out_ptr,
            stride_s,
            stride_k,
            tile_lo,
            tile_hi,
            store_hi,
            bs_ptr,
            bs_stride_s,
        )

    @gluon.jit
    def row_bounds(self, ends, BLOCK_M: gl.constexpr):
        # One exclusive bound per row
        out = ()
        for r in gl.static_range(0, BLOCK_M):
            if self.cfg.GATHER:
                out = out + (self.store_hi,)
            else:
                out = out + (gl.minimum(ends[r], self.store_hi),)
        return out

    @gluon.jit
    def block_max(self, scores, r: gl.constexpr, tile_pos):
        """A per-row max over each BSCORE_BLOCK columns of this tile.

        The candidate block divides BLOCK_KV, so a tile owns a whole number of
        blocks
        """
        cfg: gl.constexpr = self.cfg
        C: gl.constexpr = cfg.BSCORE_BLOCK
        BPT: gl.constexpr = cfg.BLOCKS_PER_TILE
        grouped = scores.reshape([BPT, C])
        # Adjacent columns are adjacent lanes under the MFMA layout, so this is
        # a cross-lane reduce: at C <= N_PER_TILE it stays inside a DPP row.
        best = gl.reduce(grouped, 1, _max_nan)
        blk = gl.arange(0, BPT, layout=gl.SliceLayout(1, grouped.type.layout))
        gl.amd.cdna4.buffer_store(
            best,
            ptr=self.bs_ptr + r * self.bs_stride_s,
            offsets=tile_pos // C + blk,
            mask=tile_pos + blk * C < self.store_hi,
        )

    @gluon.jit
    def emit(
        self, qs, qss, ws, row_hi, k, k_scale, tile_pos, MASKED: gl.constexpr = True
    ):
        # One row: the bound is exact, so it goes straight into the predicate.
        # Several: a per-row predicate would need one exec mask per row, so the
        # per-row part goes in a select under a shared predicate instead.
        # RELAXED_STORE drops that select
        cfg: gl.constexpr = self.cfg
        col = gl.arange(0, cfg.BLOCK_KV, layout=gl.SliceLayout(0, cfg.mfma_layout))
        pos = tile_pos + col
        if cfg.STORE_LOGITS:
            offsets = pos * self.stride_k
        for r in gl.static_range(0, cfg.BLOCK_M):
            scores = _row_logits(cfg, qs[r], qss[r], ws[r], k, k_scale)
            if cfg.BLOCK_M == 1:
                if cfg.BSCORE and MASKED:
                    scores = gl.where(pos < row_hi[r], scores, float("-inf"))
                if cfg.STORE_LOGITS:
                    mask = pos < row_hi[r]
            elif cfg.RELAXED_STORE:
                if cfg.BSCORE:
                    scores = gl.where(pos < row_hi[r], scores, float("-inf"))
                if cfg.STORE_LOGITS:
                    mask = pos < self.store_hi
            else:
                scores = gl.where(pos < row_hi[r], scores, float("-inf"))
                if cfg.STORE_LOGITS:
                    mask = pos < self.store_hi
            # A slot past the sequence is the next sequence's row: the shared
            # predicate only selects -inf, so it has to be masked out here.
            if cfg.VARLEN and cfg.BLOCK_M > 1 and cfg.STORE_LOGITS:
                mask = mask & (row_hi[r] > 0)
            if cfg.STORE_LOGITS:
                gl.amd.cdna4.buffer_store(
                    scores,
                    ptr=self.out_ptr + r * self.stride_s,
                    offsets=offsets,
                    mask=mask,
                )
            if cfg.BSCORE:
                if cfg.VARLEN and cfg.BLOCK_M > 1:
                    if row_hi[r] > 0:
                        self.block_max(scores, r, tile_pos)
                else:
                    self.block_max(scores, r, tile_pos)


@gluon.jit
def _loop_with_reg(pgm, loader, qs, qss, ws, row_hi):
    """DEPTH tiles in flight, straight into registers."""
    cfg: gl.constexpr = pgm.cfg
    BKV: gl.constexpr = cfg.BLOCK_KV
    AHEAD: gl.constexpr = cfg.DEPTH * BKV
    pos = pgm.tile_lo * BKV

    # vmcnt is a FIFO, so issue order is wait order: the 256-byte scale tile
    # goes before the 4 KB value tile, which makes the wait that releases it the
    # later and looser one.
    p0 = loader.page_token(pos)
    g0 = loader.gather_tok(pos)
    s0 = loader.scales(pos, p0, g0)
    k0 = loader.values(pos, p0, g0)
    if cfg.DEPTH == 2:
        p1 = loader.page_token(pos + BKV)
        g1 = loader.gather_tok(pos + BKV)
        s1 = loader.scales(pos + BKV, p1, g1)
        k1 = loader.values(pos + BKV, p1, g1)
    pn = loader.page_token(pos + AHEAD) if cfg.PAGE_PIPE else 0
    # The candidate list, an iteration ahead of the tile it addresses. Same
    # trade as PAGE_PIPE
    gn = loader.gather_tok(pos + AHEAD) if cfg.GATHER_PIPE else (0, 0)

    unroll: gl.constexpr = cfg.UNROLL if cfg.UNROLL > 1 else None
    for _i in tl.range(
        0, pgm.tile_hi - pgm.tile_lo - cfg.DEPTH, loop_unroll_factor=unroll
    ):
        if not cfg.PAGE_PIPE:
            pn = loader.page_token(pos + AHEAD)
        if cfg.GATHER_PIPE:
            gc = gn
        else:
            gc = loader.gather_tok(pos + AHEAD)
        s_new = loader.scales(pos + AHEAD, pn, gc)
        k_new = loader.values(pos + AHEAD, pn, gc)
        if cfg.GATHER_PIPE:
            gn = loader.gather_tok(pos + AHEAD + BKV)
        if cfg.PAGE_PIPE:
            pn = loader.page_token(pos + AHEAD + BKV)
        pgm.emit(qs, qss, ws, row_hi, k0, s0, pos, MASKED=False)
        if cfg.DEPTH == 2:
            k0, s0, k1, s1 = k1, s1, k_new, s_new
        else:
            k0, s0 = k_new, s_new
        pos += BKV

    pgm.emit(qs, qss, ws, row_hi, k0, s0, pos)
    if cfg.DEPTH == 2:
        pgm.emit(qs, qss, ws, row_hi, k1, s1, pos + BKV)


@gluon.jit
def _loop_with_lds(pgm, loader, qs, qss, ws, row_hi):
    """Double-buffered: wait for tile i, start i+2 into the buffer i vacated,
    read i out of LDS, then the MFMAs. The scale tile is fetched first so its
    HBM latency overlaps the wait_group."""
    cfg: gl.constexpr = pgm.cfg
    gl.static_assert(cfg.NUM_BUFFERS == 2, "the LDS walk assumes double buffering")
    BKV: gl.constexpr = cfg.BLOCK_KV
    n_tiles = pgm.tile_hi - pgm.tile_lo
    pos = pgm.tile_lo * BKV

    pa = loader.page_token(pos)
    ga = loader.gather_tok(pos)
    loader.issue(pos, pa, ga, 0)
    pb = loader.page_token(pos + BKV)
    gb = loader.gather_tok(pos + BKV)
    loader.issue(pos + BKV, pb, gb, 1)
    pn = loader.page_token(pos + 2 * BKV) if cfg.PAGE_PIPE else 0
    gn = loader.gather_tok(pos + 2 * BKV) if cfg.GATHER_PIPE else (0, 0)

    unroll: gl.constexpr = cfg.UNROLL if cfg.UNROLL > 1 else None
    buf: gl.int32 = 0
    for _i in tl.range(0, n_tiles - 2, loop_unroll_factor=unroll):
        if not cfg.PAGE_PIPE:
            pn = loader.page_token(pos + 2 * BKV)
        if not cfg.GATHER_PIPE:
            gn = loader.gather_tok(pos + 2 * BKV)
        kv_scale = loader.scales(pos, pa, ga)
        k = loader.consume(1, buf)
        loader.issue(pos + 2 * BKV, pn, gn, buf)
        if cfg.PAGE_PIPE:
            pa, pb, pn = pb, pn, loader.page_token(pos + 3 * BKV)
        else:
            pa, pb = pb, pn
        if cfg.GATHER_PIPE:
            ga, gb, gn = gb, gn, loader.gather_tok(pos + 3 * BKV)
        else:
            ga, gb = gb, gn
        pgm.emit(qs, qss, ws, row_hi, k, kv_scale, pos, MASKED=False)
        buf = 1 - buf
        pos += BKV

    # Two groups are outstanding whenever n_tiles >= 2; at one tile the second
    # was issued anyway and is simply never read.
    if n_tiles > 1:
        kv_scale = loader.scales(pos, pa, ga)
        k = loader.consume(1, buf)
        pgm.emit(qs, qss, ws, row_hi, k, kv_scale, pos)
        pos += BKV
        buf = 1 - buf
        pa = pb
        ga = gb
    kv_scale = loader.scales(pos, pa, ga)
    k = loader.consume(0, buf)
    pgm.emit(qs, qss, ws, row_hi, k, kv_scale, pos)


# the kernel

_repr = make_kernel_repr(
    "_pa_mqa_logits_mxfp4_kernel",
    [
        "NUM_HEADS",
        "HEAD_SIZE",
        "PAGE_SIZE",
        "BLOCK_KV",
        "BLOCK_M",
        "PRESHUFFLE",
        "GATHER",
        "BSCORE",
        "BSCORE_BLOCK",
        "DEPTH",
        "UNROLL",
        "M_CHUNK",
        "MFMA_NONK_DIM",
        "NUM_WARPS",
        "HAS_KV_SPLIT",
        "FOLD_ASM",
        "RELU_ADD",
    ],
)


# These move with the batch and with where a workspace slice starts, and none
# reaches the generated code, so specializing only recompiled it. A gl.int32
# equal to 1 is not folded either, which is why the ints drop both checks.
@gluon.jit(
    repr=_repr,
    do_not_specialize=[
        "next_n",
        "batch",
        "num_kv_splits",
        "stride_logits_s",
        "stride_bs_s",
    ],
    do_not_specialize_on_alignment=[
        "sched_ptr",
        "context_lens_ptr",
        "gather_v_ptr",
        "gather_s_ptr",
    ],
)
def _pa_mqa_logits_mxfp4_kernel(
    Q_ptr,  # uint8 [B, NEXT_N, H, D//2]  packed e2m1
    q_scales_ptr,  # uint8 [B, NEXT_N, H, D//32] e8m0
    KV_ptr,  # uint8 paged, page p's values at p * KV_PAGE_STRIDE
    kv_scales_ptr,  # uint8 paged, page p's e8m0 at p * KVS_PAGE_STRIDE
    weights_ptr,  # fp32  [B * NEXT_N, H]
    context_lens_ptr,  # int32 [B]
    query_start_loc_ptr,  # int32 [B + 1] row prefix sum, unused unless VARLEN
    row_ends_ptr,  # int32 [B * NEXT_N] exclusive row end, or the row's
    # candidate slot count under GATHER; None unless HAS_ROW_ENDS
    block_table_ptr,  # int32 [B, stride_blk_b]
    sched_ptr,  # int32 [grid, 4] descriptors; unused unless DYNAMIC
    gather_v_ptr,  # int32 [B*NEXT_N, blocks] resolved value offsets
    gather_s_ptr,  # int32 [B*NEXT_N, blocks] resolved scale byte offsets
    block_scores_ptr,  # fp32  [B * NEXT_N, ceil(max_model_len / BSCORE_BLOCK)]
    # per-row block maxima; None unless BSCORE
    logits_ptr,  # fp32  [B * NEXT_N, max_model_len]; None under
    # BSCORE == 2, which writes no logits at all
    next_n: gl.int32,
    batch: gl.int32,
    num_kv_splits: gl.int32,
    stride_q_b: gl.int32,
    stride_q_n: gl.int32,
    stride_qs_b: gl.int32,
    stride_qs_n: gl.int32,
    stride_w_s: gl.int32,
    stride_logits_s: gl.int32,
    stride_logits_k: gl.int32,
    stride_blk_b: gl.int32,
    stride_gather_r: gl.int32,
    stride_bs_s,  # None unless BSCORE, so it leaves no kernarg when off
    max_blocks: gl.int32,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    KV_PAGE_STRIDE: gl.constexpr,
    KVS_PAGE_STRIDE: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    DEPTH: gl.constexpr,
    UNROLL: gl.constexpr,
    PAGE_PIPE: gl.constexpr,
    M_CHUNK: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
    FOLD_ASM: gl.constexpr,
    RELU_ADD: gl.constexpr,
    PRESHUFFLE: gl.constexpr,
    SCALE_MODE: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    RELAXED_STORE: gl.constexpr,
    MFMA_NONK_DIM: gl.constexpr,
    HAS_KV_SPLIT: gl.constexpr,
    KV_REREAD: gl.constexpr,
    DYNAMIC: gl.constexpr,
    VARLEN: gl.constexpr,
    HAS_ROW_ENDS: gl.constexpr,
    GATHER: gl.constexpr,  # walk a host-resolved candidate list
    GATHER_BLOCK: gl.constexpr,  # KV tokens per candidate block
    GATHER_PIPE: gl.constexpr,  # hoist the candidate-list read one iteration
    BSCORE: gl.constexpr,  # 0 off, 1 block maxima beside the logits,
    # 2 block maxima and no logits store
    BSCORE_BLOCK: gl.constexpr,  # columns per candidate block
):
    gl.static_assert(BLOCK_KV % MFMA_NONK_DIM == 0)
    gl.static_assert(
        PAGE_SIZE % BLOCK_KV == 0,
        "BLOCK_KV must divide the page so a tile never spans two",
    )
    gl.static_assert(
        (GATHER == 0)
        | (
            (BLOCK_M == 1)
            & (BLOCK_KV % GATHER_BLOCK == 0)
            & (PAGE_SIZE % GATHER_BLOCK == 0)
            & (GATHER_BLOCK <= MFMA_NONK_DIM)
        ),
        "a candidate block must tile BLOCK_KV, sit inside one page and one "
        "shuffle group, and the gather is one query row per workgroup",
    )
    gl.static_assert(
        (GATHER == 0) | (HAS_ROW_ENDS == 1),
        "the gather takes its walk length from row_ends, read as a slot count",
    )
    gl.static_assert(
        (BSCORE == 0) | ((BSCORE_BLOCK <= BLOCK_KV) & (BLOCK_KV % BSCORE_BLOCK == 0)),
        "a candidate block must tile BLOCK_KV, so that no block straddles a "
        "tile or a KV split and the reduce stays local",
    )
    gl.static_assert(
        (BSCORE == 0) | (BSCORE == 1) | (BSCORE == 2),
        "BSCORE is 0 off, 1 block maxima beside the logits, "
        "2 block maxima instead of them",
    )
    gl.static_assert(
        (BSCORE == 0) | (GATHER == 0),
        "block maxima are for the dense producer, not the gather",
    )

    cfg = Config(
        NUM_HEADS,
        HEAD_SIZE,
        PAGE_SIZE,
        BLOCK_KV,
        BLOCK_M,
        KV_PAGE_STRIDE,
        KVS_PAGE_STRIDE,
        NUM_WARPS,
        NUM_BUFFERS,
        DEPTH,
        UNROLL,
        PAGE_PIPE,
        M_CHUNK,
        NUM_CHAINS,
        FOLD_ASM,
        RELU_ADD,
        PRESHUFFLE,
        SCALE_MODE,
        USE_BUFFER_LOAD,
        RELAXED_STORE,
        VARLEN,
        MFMA_NONK_DIM,
        HAS_KV_SPLIT,
        KV_REREAD,
        GATHER,
        GATHER_BLOCK,
        GATHER_PIPE,
        BSCORE,
        BSCORE_BLOCK,
    )

    if DYNAMIC:
        desc = sched_ptr + gl.program_id(0) * 4
        batch_id = gl.load(desc + 0)
        row_block = gl.load(desc + 1)
        slice_idx = gl.load(desc + 2)
        num_slices = gl.load(desc + 3)
        split_id: gl.int32 = 0
        if num_slices <= slice_idx:
            return
    elif VARLEN:
        # No rectangle to take the grid from when the rows are ragged, so the
        # unit space is query_start_loc[s] // BLOCK_M + s the way aiter's
        # unified attention does it: monotone, so a binary search takes a flat
        # id back to its sequence, and the spare unit per sequence -- which
        # absorbs a partial last block -- falls out past the row count below.
        unit = gl.program_id(0)
        lo: gl.int32 = 0
        hi = batch
        while lo < hi:
            mid = (lo + hi) // 2
            if gl.load(query_start_loc_ptr + mid) // BLOCK_M + mid <= unit:
                lo = mid + 1
            else:
                hi = mid
        batch_id = lo - 1
        row_block = unit - (
            gl.load(query_start_loc_ptr + batch_id) // BLOCK_M + batch_id
        )
        split_id = gl.program_id(2)
        slice_idx: gl.int32 = 0
        num_slices: gl.int32 = 0
    else:
        # Longest first keeps the grid's tail short.
        row_block = gl.num_programs(0) - 1 - gl.program_id(0)
        batch_id = gl.program_id(1)
        split_id = gl.program_id(2)
        slice_idx: gl.int32 = 0
        num_slices: gl.int32 = 0

    context_len = gl.load(context_lens_ptr + batch_id)
    if context_len <= 0:
        return

    if VARLEN:
        q_start = gl.load(query_start_loc_ptr + batch_id)
        rows = gl.load(query_start_loc_ptr + batch_id + 1) - q_start
        # Kept apart from the runtime test so the DYNAMIC build emits none of it
        if not DYNAMIC:  # noqa: SIM102
            # the spare unit, and any the bound over-provisioned
            if row_block * BLOCK_M >= rows:
                return
    else:
        q_start = batch_id * next_n
        rows = next_n

    # The trailing block still owns BLOCK_M real rows when BLOCK_M does not
    # divide the count. Rows two blocks share are computed twice, identically.
    # A varlen sequence can own fewer rows than BLOCK_M, so the floor matters.
    n0 = (
        row_block
        if BLOCK_M == 1
        else gl.maximum(gl.minimum(row_block * BLOCK_M, rows - BLOCK_M), 0)
    )
    if VARLEN:
        row0 = (q_start + n0).to(gl.int64)
        q_base, qs_base, w_row = row0 * stride_q_n, row0 * stride_qs_n, row0
    else:
        q_base = batch_id.to(gl.int64) * stride_q_b + n0.to(gl.int64) * stride_q_n
        qs_base = batch_id.to(gl.int64) * stride_qs_b + n0.to(gl.int64) * stride_qs_n
        w_row = batch_id.to(gl.int64) * next_n + n0

    mfma_qs, q_scales, w_blocks, ends = (), (), (), ()
    for r in gl.static_range(0, BLOCK_M):
        # A short varlen sequence repeats its last row rather than reading past
        # itself; end 0 then drops it at the store.
        if VARLEN:
            rr = gl.minimum(n0 + r, rows - 1) - n0
        else:
            rr = r
        q, qs, w = _load_q_row(
            cfg,
            Q_ptr + (q_base + rr * stride_q_n),
            q_scales_ptr + (qs_base + rr * stride_qs_n),
            weights_ptr + (w_row + rr) * stride_w_s,
        )
        mfma_qs, q_scales, w_blocks = mfma_qs + (q,), q_scales + (qs,), w_blocks + (w,)
        if HAS_ROW_ENDS:
            e = gl.load(row_ends_ptr + (w_row + rr))
        else:
            e = context_len - rows + n0 + r + 1
        if VARLEN:
            e = gl.where(n0 + r < rows, e, 0)
        ends = ends + (e,)

    if HAS_ROW_ENDS or VARLEN:
        block_end = ends[0]
        for r in gl.static_range(1, BLOCK_M):
            block_end = gl.maximum(block_end, ends[r])
    else:
        block_end = context_len - next_n + n0 + BLOCK_M

    if BSCORE:
        bs_ptr = block_scores_ptr + w_row.to(gl.int64) * stride_bs_s
        bs_stride_s = stride_bs_s
    else:
        bs_ptr = logits_ptr
        bs_stride_s: gl.int32 = 0
    if BSCORE == 2:
        # There is no logits row to address: logits_ptr arrives as None and
        # specializes away
        out_ptr = bs_ptr
        out_stride_s: gl.int32 = 0
        out_stride_k: gl.int32 = 0
    else:
        out_ptr = logits_ptr + w_row.to(gl.int64) * stride_logits_s
        out_stride_s = stride_logits_s
        out_stride_k = stride_logits_k
    pgm = Program.initialize(
        cfg,
        out_ptr,
        out_stride_s,
        out_stride_k,
        context_len,
        block_end,
        split_id,
        num_kv_splits,
        slice_idx,
        num_slices,
        bs_ptr,
        bs_stride_s,
        HAS_KV_SPLIT,
        DYNAMIC,
    )
    if pgm.tile_lo >= pgm.tile_hi:
        return
    row_hi = pgm.row_bounds(ends, BLOCK_M)

    blk_ptr = block_table_ptr + batch_id * stride_blk_b
    last_page_row = (
        gl.minimum((context_len + PAGE_SIZE - 1) // PAGE_SIZE, max_blocks) - 1
    )

    # The candidate list is per query row
    g_base = w_row.to(gl.int64) * stride_gather_r
    gv_ptr = gather_v_ptr + g_base
    gs_ptr = gather_s_ptr + g_base
    last_blk = gl.maximum((block_end + GATHER_BLOCK - 1) // GATHER_BLOCK - 1, 0)

    if PRESHUFFLE:
        loader = RegLoader.initialize(
            cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row, gv_ptr, gs_ptr, last_blk
        )
        _loop_with_reg(pgm, loader, mfma_qs, q_scales, w_blocks, row_hi)
    else:
        loader = LDSLoader.initialize(
            cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row, gv_ptr, gs_ptr, last_blk
        )
        _loop_with_lds(pgm, loader, mfma_qs, q_scales, w_blocks, row_hi)

    if BSCORE:
        # The row's newest block is a candidate whatever it scored
        gl.barrier()
        # One thread, not one element of a replicated tensor.
        NT: gl.constexpr = NUM_WARPS * WARP_SIZE
        lay: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[WARP_SIZE],
            warps_per_cta=[NUM_WARPS],
            order=[0],
        )
        tid = gl.arange(0, NT, layout=lay)
        inf = gl.full([NT], float("inf"), gl.float32, layout=lay)
        for r in gl.static_range(0, BLOCK_M):
            last = ends[r] - 1
            tile = last // BLOCK_KV
            own = (last >= 0) & (tile >= pgm.tile_lo) & (tile < pgm.tile_hi)
            gl.amd.cdna4.buffer_store(
                inf,
                ptr=pgm.bs_ptr + r * pgm.bs_stride_s,
                offsets=tid * 0 + last // BSCORE_BLOCK,
                mask=own & (tid == 0),
            )


# batch and the slot counts change every step and their alignment never reaches
# the code. next_n keeps its specialization: it folds at 1 here.
@triton.jit(
    do_not_specialize_on_alignment=["batch", "num_ctas", "num_units", "max_tiles"]
)
def _pa_mqa_logits_mxfp4_sched_kernel(
    context_lens_ptr,
    row_ends_ptr,
    query_start_loc_ptr,
    sched_ptr,
    batch,
    next_n,
    num_ctas,
    num_units,
    max_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    ROW_BLOCKS: tl.constexpr,
    ALIGN_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HAS_ROW_ENDS: tl.constexpr,
    GATHER: tl.constexpr,
    VARLEN: tl.constexpr,
    ALIGN_B: tl.constexpr,
    SLICE_ROOM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # A "unit" is one (sequence, row block) pair; a "slot" is one workgroup of
    # the launch. The job is to give every slot a slice of some unit's KV walk,
    # all slices about the same length. Steps 1 to 3 are cheap reductions over
    # all ALIGN_W units, so every program just redoes them, each then writes
    # descriptors for the BLOCK_P slots it owns.

    # Step 1: how many KV tiles each unit actually walks.
    unit = tl.arange(0, ALIGN_W)
    if VARLEN:
        b = tl.arange(0, ALIGN_B)
        blive = b < batch
        q0 = tl.load(query_start_loc_ptr + b, mask=blive, other=0)
        rows_of = tl.load(query_start_loc_ptr + b + 1, mask=blive, other=0) - q0
        # Unit space is query_start_loc[s] // BLOCK_M + s: the spare unit per
        # sequence absorbs its partial last block, which keeps the space
        # monotone for the search and bounds the count at
        # total_rows // BLOCK_M + batch.
        ustart = q0 // BLOCK_M + b
        found = (ustart[None, :] <= unit[:, None]) & blive[None, :]
        seq = tl.maximum(tl.sum(found.to(tl.int32), axis=1) - 1, 0)
        pick = (b[None, :] == seq[:, None]) & blive[None, :]
        row_blk = unit - tl.sum(tl.where(pick, ustart[None, :], 0), axis=1)
        rows = tl.sum(tl.where(pick, rows_of[None, :], 0), axis=1)
        q_start = tl.sum(tl.where(pick, q0[None, :], 0), axis=1)
        live = (unit < num_units) & (row_blk < tl.cdiv(rows, BLOCK_M))
    else:
        seq = unit // ROW_BLOCKS
        row_blk = unit % ROW_BLOCKS
        rows = next_n
        q_start = seq * next_n
        live = unit < batch * ROW_BLOCKS
    ctx = tl.load(context_lens_ptr + seq, mask=live, other=0)

    # The same causal trim the main kernel applies, clamped trailing row block
    # and all, so this counts tiles that are really walked. The count only sets
    # the balance: a live unit is floored at one tile below, so it keeps its slot
    # whatever the trim thinks, and the kernel re-derives the real walk.
    first_row = (
        row_blk
        if BLOCK_M == 1
        else tl.maximum(tl.minimum(row_blk * BLOCK_M, rows - BLOCK_M), 0)
    )
    if HAS_ROW_ENDS:
        block_end = tl.zeros([ALIGN_W], tl.int32)
        for r in tl.static_range(0, BLOCK_M):
            block_end = tl.maximum(
                block_end,
                tl.load(row_ends_ptr + q_start + first_row + r, mask=live, other=0),
            )
    else:
        block_end = ctx - rows + first_row + BLOCK_M
    # Under the gather block_end counts candidate slots, which the context
    # length does not bound.
    keys = block_end if GATHER else tl.minimum(ctx, block_end)
    # Floored at one so a live unit always owns a slot. Whether a unit is live
    # must not depend on the bound: the launch may hold a row_ends this kernel was
    # not given, and a unit with no slot has nothing to compute its rows.
    tiles = tl.maximum(tl.cdiv(keys, BLOCK_KV), 1)
    tiles = tl.where(live & (ctx > 0), tiles, 0)

    # Step 2: how many tiles go in a slice, the same for every slot
    total_tiles = tl.sum(tiles)
    live_units = tl.sum((tiles > 0).to(tl.int32))
    tiles_per_slice = tl.maximum(
        tl.cdiv(total_tiles, tl.maximum(num_ctas - live_units, 1)), 1
    )
    # Filling the machine once is not enough on a long walk: the static grid
    # caps a workgroup at max_tiles_per_split through _kv_splits and ends up
    # with far more, shorter workgroups, which this kernel keeps gaining from
    # past residency. The host sizes num_ctas so the extra slices fit.
    tiles_per_slice = tl.minimum(tiles_per_slice, max_tiles)
    # Bound one unit's slices so the write below stays a few bands. Measured
    # against its average share of the slots, not max_model_len, which on a
    # spread of lengths sits well above what the balance wants.
    share = tl.maximum((num_ctas - live_units) // tl.maximum(live_units, 1), 1)
    tiles_per_slice = tl.maximum(
        tiles_per_slice, tl.cdiv(tl.max(tiles), share * SLICE_ROOM)
    )

    # Step 3: lay each unit's slices end to end over the slots, so unit i owns
    # slots [first_slot_i, first_slot_i + n_slices_i).
    n_slices = tl.cdiv(tiles, tiles_per_slice)
    first_slot = tl.cumsum(n_slices) - n_slices  # exclusive prefix sum
    slots_used = tl.sum(n_slices)

    # Step 4: write the descriptors. A unit knows where its slices start, so
    # there is no owner to search for: one program takes a band of slice
    # indices and every unit writes the slices it has in that band. Slots past
    # slots_used keep the zeroed descriptor, which the walk reads as no work.
    if VARLEN:
        own_seq, own_blk = seq, row_blk
    else:
        own_seq, own_blk = unit // ROW_BLOCKS, unit % ROW_BLOCKS
    shape: tl.constexpr = (BLOCK_S, ALIGN_W)
    # The band strides, so the grid sizes the write rather than bounding what
    # a unit may own.
    widest = tl.max(n_slices)
    sl0 = tl.program_id(0) * BLOCK_S
    while sl0 < widest:
        sl = sl0 + tl.arange(0, BLOCK_S)
        slot = first_slot[None, :] + sl[:, None]
        live_slice = (sl[:, None] < n_slices[None, :]) & (slot < num_ctas)
        base = sched_ptr + slot * 4
        tl.store(base + 0, tl.broadcast_to(own_seq[None, :], shape), mask=live_slice)
        tl.store(base + 1, tl.broadcast_to(own_blk[None, :], shape), mask=live_slice)
        tl.store(base + 2, tl.broadcast_to(sl[:, None], shape), mask=live_slice)
        tl.store(base + 3, tl.broadcast_to(n_slices[None, :], shape), mask=live_slice)
        sl0 += tl.num_programs(0) * BLOCK_S

    # Step 5: the grid is num_ctas, so a slot past the last slice launches too
    # and has to read as no work. The walk returns on num_slices <= slice_idx,
    # so both have to be cleared -- a stale negative slice_idx would pass that
    # test and send the workgroup on with a garbage batch id. num_ctas is sized
    # to hold the slices, so this is usually empty and the loop does not run.
    t = slots_used + tl.program_id(0) * BLOCK_T
    while t < num_ctas:
        idx = t + tl.arange(0, BLOCK_T)
        keep = idx < num_ctas
        zero = tl.zeros([BLOCK_T], tl.int32)
        tl.store(sched_ptr + idx * 4 + 2, zero, mask=keep)
        tl.store(sched_ptr + idx * 4 + 3, zero, mask=keep)
        t += tl.num_programs(0) * BLOCK_T
