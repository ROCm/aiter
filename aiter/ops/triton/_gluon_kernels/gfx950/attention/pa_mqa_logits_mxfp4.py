# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged MXFP4 MQA-logits kernel for gfx950, plus its scheduler kernel.

    logits[b*next_n + n, t] = sum_h relu(q[b, n, h, :] . kv[t, :]) * w[b*next_n + n, h]
"""

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan
from triton.language.core import _aggregate as aggregate

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
        reg_bases=[[b[0] // n_per_tile, 0, 0, b[1] // s_lo]
                   for b in scale_layout.reg_bases],
        lane_bases=[[0, b[1] % s_lo, b[0] % n_per_tile, 0]
                    for b in scale_layout.lane_bases],
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
    lane = bases[i:i + lg2_ws]
    i += lg2_ws
    warp = bases[i:i + lg2_nw]
    i += lg2_nw
    warp = warp + [[0] * rank] * (lg2_nw - len(warp))
    return gl.DistributedLinearLayout(reg_bases=reg + bases[i:], lane_bases=lane,
                                      warp_bases=warp, block_bases=[], shape=shape)


@gluon.constexpr_function
def _staging_layouts(head_bytes, block_kv, num_warps, warp_size):
    # (blocked, shared) for a token-major [head_bytes, block_kv] tile. The XOR
    # term on the token axis is what keeps the ds_reads conflict free; the
    # padding pair covers the rest.
    c = 16  # 128-bit vector of 8-bit containers
    lg2_hs = head_bytes.bit_length() - 1
    lg2_ts = block_kv.bit_length() - 1
    hs_lane = lg2_hs - (c.bit_length() - 1)
    bases = ([[1 << i, 0] for i in range(lg2_hs)]
             + [[0, 1 << ((i + hs_lane) % lg2_ts)] for i in range(lg2_ts)])
    shared = gl.PaddedSharedLayout(interval_padding_pairs=[[1024, 16]],
                                   offset_bases=bases, cga_layout=[],
                                   shape=[head_bytes, block_kv])
    return _offset_bases_to_blocked(bases, c, num_warps, warp_size,
                                    [head_bytes, block_kv]), shared


@gluon.jit
def _shuffled_offsets(offs_d, offs_n, D_EXTENT: gl.constexpr,
                      N_PER_TILE: gl.constexpr, D_PER_TILE: gl.constexpr):
    # Byte offset of (row, container) inside one preshuffled page.
    TILE: gl.constexpr = N_PER_TILE * D_PER_TILE
    GROUP: gl.constexpr = N_PER_TILE * D_EXTENT
    return ((offs_d % D_PER_TILE)
            + (offs_d // D_PER_TILE) * TILE
            + (offs_n % N_PER_TILE) * D_PER_TILE
            + (offs_n // N_PER_TILE) * GROUP)


@gluon.jit
def _shuffled_scale_offsets(offs_n, offs_s, NUM_SCALES: gl.constexpr,
                            N_PER_TILE: gl.constexpr):
    # The e8m0 twin, used for Q. Grouping N_PER_TILE rows puts each register
    # element's 64 lanes on one contiguous 64-byte run.
    return ((offs_n % N_PER_TILE)
            + offs_s * N_PER_TILE
            + (offs_n // N_PER_TILE) * (N_PER_TILE * NUM_SCALES))


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
    return (tuple([2] * head_bits + [block_kv]),
            tuple(summed + [head_bits] + folded[:chain_bits] + folded[chain_bits:]),
            tuple([1 << len(summed), block_kv, num_chains] + [2] * depth),
            depth, 1 << depth)


@gluon.jit
def _relu(x, RELU_ADD: gl.constexpr):
    # RELU_ADD spells relu as (x + |x|)/2: one v_add_f32 with an abs source
    # modifier where max(x, 0) is a v_maximum3_f32, same instruction count and
    # registers, just a faster opcode here. The halving rides on the per-head
    # weight in the prologue.
    #
    # Two ways to lose it, both measured. Do not also wrap it in inline asm --
    # each asm block is a scheduling barrier, s_nop 29 -> 46 at 64 heads. And do
    # not let the SLP vectorizer reach it: v_pk_add_f32 has no abs modifier, so
    # packing forces an explicit v_and_b32 per element plus copies, +65-73% VALU
    # and 0.70-0.73x. FOLD_ASM is what keeps these adds scalar, for free.
    #
    # Differs from max(x, 0) only at x = -inf and |x| > 2**127; the launcher
    # docstring states the contract.
    if RELU_ADD:
        return x + gl.abs(x)
    return gl.maximum(x, 0, propagate_nan=_NAN_ALL)


@gluon.jit
def _fma_unpacked(a, b, c):
    # a * b + c as one v_fma_f32 the SLP vectorizer cannot pair. Packed FP32
    # cannot co-issue in an MFMA shadow on gfx950 and needs even-aligned pairs,
    # so keeping these scalar is what frees the registers BLOCK_M and
    # waves_per_eu spend. is_pure=False or LLVM dedups asm blocks it thinks equal.
    return gl.inline_asm_elementwise("v_fma_f32 $0, $1, $2, $3", "=v,v,v,v",
                                     [a, b, c], dtype=gl.float32,
                                     is_pure=False, pack=1)


@gluon.jit
def _split_leaf(x, IDX: gl.constexpr, DEPTH: gl.constexpr):
    for bit in gl.static_range(0, DEPTH):
        lo, hi = x.split()
        x = lo if (IDX // (2 ** bit)) % 2 == 0 else hi
    return x


@gluon.jit
def _fold_heads(s, w_col, NUM_HEADS: gl.constexpr, BLOCK_KV: gl.constexpr,
                mfma_layout: gl.constexpr, NUM_CHAINS: gl.constexpr,
                FOLD_ASM: gl.constexpr, acc=None, HAS_ACC: gl.constexpr = False):
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
    return gl.convert_layout(gl.sum(gl.sum(acc, axis=2), axis=0),
                             gl.SliceLayout(0, mfma_layout))


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
    USE_BUFFER_LOAD: gl.constexpr
    RELAXED_STORE: gl.constexpr
    N_PER_TILE: gl.constexpr
    D_PER_TILE: gl.constexpr
    Q_CACHE: gl.constexpr
    KV_CACHE: gl.constexpr
    mfma_layout: gl.constexpr
    dot_a: gl.constexpr
    dot_b: gl.constexpr
    q_scale_layout: gl.constexpr
    kv_scale_layout: gl.constexpr
    blocked_kv: gl.constexpr
    shared_kv: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, NUM_HEADS, HEAD_SIZE, PAGE_SIZE, BLOCK_KV, BLOCK_M,
                 KV_PAGE_STRIDE, KVS_PAGE_STRIDE, NUM_WARPS, NUM_BUFFERS, DEPTH,
                 UNROLL, PAGE_PIPE, M_CHUNK, NUM_CHAINS, FOLD_ASM, RELU_ADD,
                 PRESHUFFLE, USE_BUFFER_LOAD, RELAXED_STORE,
                 MFMA_NONK_DIM, HAS_KV_SPLIT,
                 KV_REREAD):
        self.NUM_HEADS = gl.constexpr(NUM_HEADS)
        self.HEAD_BYTES = gl.constexpr(HEAD_SIZE // 2)
        self.NUM_SCALES = gl.constexpr(HEAD_SIZE // SCALE_GROUP.value)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
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
        self.USE_BUFFER_LOAD = gl.constexpr(USE_BUFFER_LOAD)
        self.RELAXED_STORE = gl.constexpr(RELAXED_STORE)
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
            transposed=False, warps_per_cta=[1, NUM_WARPS])
        self.mfma_layout = gl.constexpr(mfma)
        self.dot_a = gl.constexpr(gl.DotOperandLayout(
            operand_index=0, parent=mfma, k_width=K_WIDTH.value))
        self.dot_b = gl.constexpr(gl.DotOperandLayout(
            operand_index=1, parent=mfma, k_width=K_WIDTH.value))

        A_ROWS = M_CHUNK if M_CHUNK > 0 else NUM_HEADS
        self.A_ROWS = gl.constexpr(A_ROWS)
        self.N_CHUNK = gl.constexpr(NUM_HEADS // A_ROWS)
        self.q_scale_layout = gl.constexpr(gl.amd.cdna4.get_mfma_scale_layout(
            self.dot_a.value, [A_ROWS, HEAD_SIZE // SCALE_GROUP.value]))
        self.kv_scale_layout = gl.constexpr(gl.amd.cdna4.get_mfma_scale_layout(
            self.dot_b.value, [BLOCK_KV, HEAD_SIZE // SCALE_GROUP.value]))

        # Only the unshuffled path stages through LDS, but the layouts are cheap
        # to build and keeping them unconditional avoids a constexpr branch in
        # the aggregate.
        blocked, shared = _staging_layouts(HEAD_SIZE // 2, BLOCK_KV, NUM_WARPS,
                                           WARP_SIZE.value)
        self.blocked_kv = gl.constexpr(blocked)
        self.shared_kv = gl.constexpr(shared)


# Q


@gluon.jit
def _load_q_chunk(cfg, q_ptr, qs_ptr, w_ptr, HEAD_OFFSET: gl.constexpr,
                  ROWS: gl.constexpr):
    # The query row rides in the pointer, not the offsets, so the i32 buffer
    # offset spans one row rather than the whole tensor.
    hs = gl.arange(0, ROWS, layout=gl.SliceLayout(1, cfg.q_scale_layout))[:, None]
    ss = gl.arange(0, cfg.NUM_SCALES,
                   layout=gl.SliceLayout(0, cfg.q_scale_layout))[None, :]
    # Q is token major and converted here. It is loaded once per row block
    # rather than once per tile, so the convert amortizes over the whole walk --
    # preshuffling Q measured within noise on every shape, which is why only the
    # cache has a shuffled layout.
    lay: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[WARP_SIZE // (cfg.HEAD_BYTES // 16),
                          cfg.HEAD_BYTES // 16],
        warps_per_cta=[cfg.NUM_WARPS, 1], order=[1, 0])
    offs_h = gl.arange(0, ROWS, layout=gl.SliceLayout(1, lay)) + HEAD_OFFSET
    offs_d = gl.arange(0, cfg.HEAD_BYTES, layout=gl.SliceLayout(0, lay))
    q = gl.convert_layout(
        gl.amd.cdna4.buffer_load(
            ptr=q_ptr,
            offsets=(offs_h * cfg.HEAD_BYTES)[:, None] + offs_d[None, :],
            cache=cfg.Q_CACHE),
        cfg.dot_a)
    qs = gl.amd.cdna4.buffer_load(
        ptr=qs_ptr,
        offsets=((hs + HEAD_OFFSET) * cfg.NUM_SCALES) + ss,
        cache=cfg.Q_CACHE)

    hw = gl.arange(0, ROWS, layout=gl.SliceLayout(1, cfg.mfma_layout))
    w = gl.amd.cdna4.buffer_load(ptr=w_ptr, offsets=(hw + HEAD_OFFSET)[:, None],
                                 cache=cfg.Q_CACHE)
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
            a, b, d = _load_q_chunk(cfg, q_ptr, qs_ptr, w_ptr,
                                    c * cfg.M_CHUNK, cfg.M_CHUNK)
            q, qs, w = q + (a,), qs + (b,), w + (d,)
        return q, qs, w
    else:
        return _load_q_chunk(cfg, q_ptr, qs_ptr, w_ptr, 0, cfg.NUM_HEADS)


# paged addressing, shared by both loaders


@aggregate
@strip_annotate
class KVState:
    """The cache, the block table and the offsets inside a page.

    Address is page_id * PAGE_STRIDE + (tile_pos % PAGE_SIZE) * PER_TOKEN; the
    second term is layout independent, since a tile is the same size either way.
    """

    cfg: Config
    KV_ptr: gl.tensor
    kv_scales_ptr: gl.tensor
    blk_ptr: gl.tensor
    last_page_row: gl.tensor
    val_offsets: gl.tensor
    scale_offsets: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row,
                 val_offsets, scale_offsets):
        self.cfg = cfg
        self.KV_ptr = KV_ptr
        self.kv_scales_ptr = kv_scales_ptr
        self.blk_ptr = blk_ptr
        self.last_page_row = last_page_row
        self.val_offsets = val_offsets
        self.scale_offsets = scale_offsets

    @gluon.jit
    def initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row,
                   val_layout: gl.constexpr):
        offs_d = gl.arange(0, cfg.HEAD_BYTES,
                           layout=gl.SliceLayout(1, val_layout))[:, None]
        offs_n = gl.arange(0, cfg.BLOCK_KV,
                           layout=gl.SliceLayout(0, val_layout))[None, :]
        if cfg.PRESHUFFLE:
            val = _shuffled_offsets(offs_d, offs_n, cfg.HEAD_BYTES,
                                    cfg.N_PER_TILE, cfg.D_PER_TILE)
        else:
            val = _token_major_offsets(offs_d, offs_n, cfg.HEAD_BYTES)

        sn = gl.arange(0, cfg.BLOCK_KV,
                       layout=gl.SliceLayout(1, cfg.kv_scale_layout))[:, None]
        ss = gl.arange(0, cfg.NUM_SCALES,
                       layout=gl.SliceLayout(0, cfg.kv_scale_layout))[None, :]
        if cfg.PRESHUFFLE:
            sc = _shuffled_scale_offsets(sn, ss, cfg.NUM_SCALES, cfg.N_PER_TILE)
        else:
            sc = _token_major_scale_offsets(sn, ss, cfg.NUM_SCALES)
        return KVState(cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row, val, sc)

    @gluon.jit
    def page_token(self, tile_pos):
        # Split out from the loads so the walk can issue it an iteration early:
        # it is a dependent load in front of every KV address in the tile, and
        # DEPTH does not hide it. Clamped to a page the sequence owns, which is
        # what lets every KV load run unmasked -- a tile past the context reads
        # real bytes and produces real, wrong logits that the store drops.
        return gl.load(self.blk_ptr + gl.minimum(tile_pos // self.cfg.PAGE_SIZE,
                                                 self.last_page_row))

    @gluon.jit
    def tile_ptr(self, base, tile_pos, page, PAGE_STRIDE: gl.constexpr,
                 PER_TOKEN: gl.constexpr):
        # 64-bit on purpose: one scalar multiply per tile keeps a cache far
        # above 2 GiB addressable while every per-element offset stays i32.
        return base + (page.to(gl.int64) * PAGE_STRIDE
                       + (tile_pos % self.cfg.PAGE_SIZE).to(gl.int64) * PER_TOKEN)

    @gluon.jit
    def page_off(self, tile_pos, page, PAGE_STRIDE: gl.constexpr,
                 PER_TOKEN: gl.constexpr):
        # The same address as a scalar to add into an offsets tensor, which is
        # what the LDS path needs; see LDSLoader.issue.
        in_page = tile_pos % self.cfg.PAGE_SIZE
        if self.cfg.USE_BUFFER_LOAD:
            return page * PAGE_STRIDE + in_page * PER_TOKEN
        else:
            return (page.to(gl.int64) * PAGE_STRIDE
                    + in_page.to(gl.int64) * PER_TOKEN)

    @gluon.jit
    def scales(self, tile_pos, page):
        """One [BLOCK_KV, NUM_SCALES] e8m0 tile in the MFMA's scale layout.

        Never goes through LDS on either path -- 256 bytes at head_size 128,
        loaded straight into the distribution the scaled MFMA wants.
        """
        if self.cfg.PRESHUFFLE:
            return _load_scales_wide(
                self.cfg,
                self.tile_ptr(self.kv_scales_ptr, tile_pos, page,
                              self.cfg.KVS_PAGE_STRIDE, self.cfg.NUM_SCALES))
        else:
            ptr = self.tile_ptr(self.kv_scales_ptr, tile_pos, page,
                                self.cfg.KVS_PAGE_STRIDE, self.cfg.NUM_SCALES)
            return gl.amd.cdna4.buffer_load(ptr=ptr, offsets=self.scale_offsets,
                                            cache=self.cfg.KV_CACHE)


@gluon.jit
def _load_scales_wide(cfg, ptr):
    """The whole scale tile, read over the run the stored order makes contiguous.

    Grouping one MFMA tile keeps the order independent of BLOCK_KV and caps the
    run at S_HI bytes -- a lane gets group_size * NUM_SCALES / 64, so widening
    would mean grouping more tokens than a tile has. Triton takes vector width
    from contiguity in index space, hence the reshaped view and the fold back,
    which is a register rename. Keep both page strides constexpr or the
    vectorizer loses the divisibility and every load collapses to a ubyte.
    """
    S_LO: gl.constexpr = WARP_SIZE // cfg.N_PER_TILE
    S_HI: gl.constexpr = cfg.NUM_SCALES // S_LO
    N_HI: gl.constexpr = cfg.BLOCK_KV // cfg.N_PER_TILE
    lr: gl.constexpr = _scale_group_layout(cfg.kv_scale_layout, cfg.N_PER_TILE,
                                           S_LO, S_HI, N_HI)
    b0 = gl.arange(0, N_HI, layout=_axis_layout(lr, 0, 4))[:, None, None, None]
    b1 = gl.arange(0, S_LO, layout=_axis_layout(lr, 1, 4))[None, :, None, None]
    b2 = gl.arange(0, cfg.N_PER_TILE,
                   layout=_axis_layout(lr, 2, 4))[None, None, :, None]
    b3 = gl.arange(0, S_HI, layout=_axis_layout(lr, 3, 4))[None, None, None, :]
    raw = gl.amd.cdna4.buffer_load(
        ptr=ptr,
        offsets=b0 * (cfg.N_PER_TILE * cfg.NUM_SCALES)
                + (b2 + b1 * cfg.N_PER_TILE) * S_HI + b3,
        cache=cfg.KV_CACHE)
    return (raw.reshape([N_HI, S_LO, cfg.N_PER_TILE, S_HI])
            .permute((0, 2, 3, 1))
            .reshape([cfg.BLOCK_KV, cfg.NUM_SCALES]))


# the two KV loaders
#
# Same vocabulary -- page_token, scales, and either values or issue/consume --
# so the walks read the same and nothing else branches on which one it got.


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
    def initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row):
        return RegLoader(KVState.initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr,
                                            last_page_row, cfg.dot_b))

    @gluon.jit
    def page_token(self, tile_pos):
        return self.st.page_token(tile_pos)

    @gluon.jit
    def scales(self, tile_pos, page):
        return self.st.scales(tile_pos, page)

    @gluon.jit
    def values(self, tile_pos, page):
        cfg: gl.constexpr = self.st.cfg
        return gl.amd.cdna4.buffer_load(
            ptr=self.st.tile_ptr(self.st.KV_ptr, tile_pos, page,
                                 cfg.KV_PAGE_STRIDE, cfg.HEAD_BYTES),
            offsets=self.st.val_offsets, cache=cfg.KV_CACHE)


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
    def initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr, last_page_row):
        st = KVState.initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr,
                                last_page_row, cfg.blocked_kv)
        shared = gl.allocate_shared_memory(
            KV_ptr.type.element_ty,
            [cfg.NUM_BUFFERS, cfg.HEAD_BYTES, cfg.BLOCK_KV], layout=cfg.shared_kv)
        return LDSLoader(st, shared)

    @gluon.jit
    def page_token(self, tile_pos):
        return self.st.page_token(tile_pos)

    @gluon.jit
    def scales(self, tile_pos, page):
        return self.st.scales(tile_pos, page)

    @gluon.jit
    def issue(self, tile_pos, page, buffer_id):
        """Start one tile's global-to-LDS copy. No mask: see page_token.

        The page rides in the offsets, not the base pointer -- the opposite of
        the register path -- because buffer_load_to_shared crashes the AMDGPU
        backend on a runtime-varying base when the loop also carries async
        copies. Cost: i32 offsets, so past 2 GiB the launcher clears
        USE_BUFFER_LOAD and the same arithmetic runs through plain pointers.
        """
        cfg: gl.constexpr = self.st.cfg
        dest = self.kv_shared.index(buffer_id)
        offsets = self.st.val_offsets + self.st.page_off(
            tile_pos, page, cfg.KV_PAGE_STRIDE, cfg.HEAD_BYTES)
        if cfg.USE_BUFFER_LOAD:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                dest, self.st.KV_ptr, offsets, cache_modifier=cfg.KV_CACHE)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                dest, self.st.KV_ptr + offsets, cache_modifier=cfg.KV_CACHE)
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def consume(self, wait_count, buffer_id):
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return self.kv_shared.index(buffer_id).load(layout=self.st.cfg.dot_b)


# the inner product and the store


@gluon.jit
def _dot(cfg, q, q_scale, k, k_scale, ROWS: gl.constexpr):
    # Both operands carry their own e8m0 block scales and the matrix core
    # applies both, so the accumulator comes out in real units and there is no
    # trailing per-token multiply.
    acc = gl.zeros([ROWS, cfg.BLOCK_KV], dtype=gl.float32, layout=cfg.mfma_layout)
    return gl.amd.cdna4.mfma_scaled(a=q, a_scale=q_scale, a_format="e2m1",
                                    b=k, b_scale=k_scale, b_format="e2m1",
                                    acc=acc)


@gluon.jit
def _row_logits(cfg, q, q_scale, w, k, k_scale):
    if cfg.M_CHUNK > 0:
        # Fold each head group as its MFMA retires, so one group's accumulator
        # is live at a time and the lane crossing is paid once for the row.
        acc = None
        for c in gl.static_range(0, cfg.N_CHUNK):
            s = _dot(cfg, q[c], q_scale[c], k, k_scale, cfg.M_CHUNK)
            acc = _fold_heads(_relu(s, cfg.RELU_ADD), w[c], cfg.M_CHUNK,
                              cfg.BLOCK_KV,
                              cfg.mfma_layout, cfg.NUM_CHAINS, cfg.FOLD_ASM,
                              acc, c > 0)
        return _fold_tail(acc, cfg.mfma_layout)
    else:
        s = _dot(cfg, q, q_scale, k, k_scale, cfg.NUM_HEADS)
        acc = _fold_heads(_relu(s, cfg.RELU_ADD), w, cfg.NUM_HEADS, cfg.BLOCK_KV,
                          cfg.mfma_layout, cfg.NUM_CHAINS, cfg.FOLD_ASM)
        return _fold_tail(acc, cfg.mfma_layout)


@aggregate
@strip_annotate
class Program:
    """This workgroup's rows and the window they own.

    A split partitions the output, not a sum -- two splits of one row write
    disjoint columns, so there is nothing to reduce afterwards.
    """

    cfg: Config
    out_ptr: gl.tensor
    stride_s: gl.tensor
    stride_k: gl.tensor
    tile_lo: gl.tensor
    tile_hi: gl.tensor
    store_hi: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg, out_ptr, stride_s, stride_k, tile_lo, tile_hi,
                 store_hi):
        self.cfg = cfg
        self.out_ptr = out_ptr
        self.stride_s = stride_s
        self.stride_k = stride_k
        self.tile_lo = tile_lo
        self.tile_hi = tile_hi
        self.store_hi = store_hi

    @gluon.jit
    def initialize(cfg, out_ptr, stride_s, stride_k, context_len, block_first,
                   split_id, num_kv_splits, slice_idx, num_slices,
                   HAS_KV_SPLIT: gl.constexpr, DYNAMIC: gl.constexpr):
        # Walk only as far as the block's last row can attend. For decode that
        # is the whole context; for a prefill chunk it is most of the work.
        keys = gl.minimum(context_len, block_first + cfg.BLOCK_M)
        n_tiles = gl.maximum((keys + cfg.BLOCK_KV - 1) // cfg.BLOCK_KV, 0)
        if DYNAMIC:
            # Slice slice_idx of num_slices, converted against this layer's own
            # tile count -- which is what lets one schedule serve layers whose
            # contexts differ by a compression factor.
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
        # The one column bound every store answers to: it keeps a walk that ran
        # past its own end out of the next split's columns and out of the region
        # the caller pre-filled with -inf.
        store_hi = gl.minimum(context_len, tile_hi * cfg.BLOCK_KV)
        return Program(cfg, out_ptr, stride_s, stride_k, tile_lo, tile_hi, store_hi)

    @gluon.jit
    def row_bounds(self, limits, BLOCK_M: gl.constexpr):
        # One exclusive bound per row, folding the causal limit, the split's
        # share and the end of the context. All loop invariant, so the walk
        # never recomputes a window.
        out = ()
        for r in gl.static_range(0, BLOCK_M):
            out = out + (gl.minimum(limits[r] + 1, self.store_hi),)
        return out

    @gluon.jit
    def emit(self, qs, qss, ws, row_hi, k, k_scale, tile_pos):
        # One row: the bound is exact, so it goes straight into the predicate.
        # Several: a per-row predicate would need one exec mask per row, so the
        # per-row part goes in a select under a shared predicate instead.
        # RELAXED_STORE drops that select -- those columns are unspecified with
        # clean_logits off. The union bound stays; it is what keeps the store
        # inside this row and this split.
        cfg: gl.constexpr = self.cfg
        col = gl.arange(0, cfg.BLOCK_KV, layout=gl.SliceLayout(0, cfg.mfma_layout))
        pos = tile_pos + col
        offsets = pos * self.stride_k
        for r in gl.static_range(0, cfg.BLOCK_M):
            scores = _row_logits(cfg, qs[r], qss[r], ws[r], k, k_scale)
            if cfg.BLOCK_M == 1:
                mask = pos < row_hi[r]
            elif cfg.RELAXED_STORE:
                mask = pos < self.store_hi
            else:
                scores = gl.where(pos < row_hi[r], scores, float("-inf"))
                mask = pos < self.store_hi
            gl.amd.cdna4.buffer_store(scores, ptr=self.out_ptr + r * self.stride_s,
                                      offsets=offsets, mask=mask)


# the two walks


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
    s0 = loader.scales(pos, p0)
    k0 = loader.values(pos, p0)
    if cfg.DEPTH == 2:
        p1 = loader.page_token(pos + BKV)
        s1 = loader.scales(pos + BKV, p1)
        k1 = loader.values(pos + BKV, p1)
    pn = loader.page_token(pos + AHEAD) if cfg.PAGE_PIPE else 0

    unroll: gl.constexpr = cfg.UNROLL if cfg.UNROLL > 1 else None
    for _i in tl.range(0, pgm.tile_hi - pgm.tile_lo - cfg.DEPTH,
                       loop_unroll_factor=unroll):
        if not cfg.PAGE_PIPE:
            pn = loader.page_token(pos + AHEAD)
        s_new = loader.scales(pos + AHEAD, pn)
        k_new = loader.values(pos + AHEAD, pn)
        if cfg.PAGE_PIPE:
            pn = loader.page_token(pos + AHEAD + BKV)
        pgm.emit(qs, qss, ws, row_hi, k0, s0, pos)
        if cfg.DEPTH == 2:
            k0, s0, k1, s1 = k1, s1, k_new, s_new
        else:
            k0, s0 = k_new, s_new
        pos += BKV

    # A segment shorter than the pipeline walks past its own end; every store
    # there is bounded by store_hi, so it costs work and nothing else.
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
    loader.issue(pos, pa, 0)
    pb = loader.page_token(pos + BKV)
    loader.issue(pos + BKV, pb, 1)
    pn = loader.page_token(pos + 2 * BKV) if cfg.PAGE_PIPE else 0

    unroll: gl.constexpr = cfg.UNROLL if cfg.UNROLL > 1 else None
    buf: gl.int32 = 0
    for _i in tl.range(0, n_tiles - 2, loop_unroll_factor=unroll):
        if not cfg.PAGE_PIPE:
            pn = loader.page_token(pos + 2 * BKV)
        kv_scale = loader.scales(pos, pa)
        k = loader.consume(1, buf)
        loader.issue(pos + 2 * BKV, pn, buf)
        if cfg.PAGE_PIPE:
            pa, pb, pn = pb, pn, loader.page_token(pos + 3 * BKV)
        else:
            pa, pb = pb, pn
        pgm.emit(qs, qss, ws, row_hi, k, kv_scale, pos)
        buf = 1 - buf
        pos += BKV

    # Two groups are outstanding whenever n_tiles >= 2; at one tile the second
    # was issued anyway and is simply never read.
    if n_tiles > 1:
        kv_scale = loader.scales(pos, pa)
        k = loader.consume(1, buf)
        pgm.emit(qs, qss, ws, row_hi, k, kv_scale, pos)
        pos += BKV
        buf = 1 - buf
        pa = pb
    kv_scale = loader.scales(pos, pa)
    k = loader.consume(0, buf)
    pgm.emit(qs, qss, ws, row_hi, k, kv_scale, pos)


# the kernel

_repr = make_kernel_repr("_pa_mqa_logits_mxfp4_kernel",
                         ["NUM_HEADS", "HEAD_SIZE", "PAGE_SIZE", "BLOCK_KV",
                          "BLOCK_M", "PRESHUFFLE", "DEPTH",
                          "UNROLL", "M_CHUNK", "MFMA_NONK_DIM", "NUM_WARPS",
                          "HAS_KV_SPLIT", "FOLD_ASM", "RELU_ADD"])


@gluon.jit(repr=_repr)
def _pa_mqa_logits_mxfp4_kernel(
    Q_ptr,             # uint8 [B, NEXT_N, H, D//2]  packed e2m1
    q_scales_ptr,      # uint8 [B, NEXT_N, H, D//32] e8m0
    KV_ptr,            # uint8 paged, page p's values at p * KV_PAGE_STRIDE
    kv_scales_ptr,     # uint8 paged, page p's e8m0 at p * KVS_PAGE_STRIDE
    weights_ptr,       # fp32  [B * NEXT_N, H]
    context_lens_ptr,  # int32 [B]
    block_table_ptr,   # int32 [B, stride_blk_b]
    sched_ptr,         # int32 [grid, 4] descriptors; unused unless DYNAMIC
    logits_ptr,        # fp32  [B * NEXT_N, max_model_len]
    next_n: gl.int32,
    num_kv_splits: gl.int32,
    stride_q_b: gl.int32, stride_q_n: gl.int32,
    stride_qs_b: gl.int32, stride_qs_n: gl.int32,
    stride_w_s: gl.int32,
    stride_logits_s: gl.int32, stride_logits_k: gl.int32,
    stride_blk_b: gl.int32,
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
    USE_BUFFER_LOAD: gl.constexpr,
    RELAXED_STORE: gl.constexpr,
    MFMA_NONK_DIM: gl.constexpr,
    HAS_KV_SPLIT: gl.constexpr,
    KV_REREAD: gl.constexpr,
    DYNAMIC: gl.constexpr,
):
    gl.static_assert(BLOCK_KV % MFMA_NONK_DIM == 0)
    gl.static_assert(PAGE_SIZE % BLOCK_KV == 0,
                     "BLOCK_KV must divide the page so a tile never spans two")

    cfg = Config(NUM_HEADS, HEAD_SIZE, PAGE_SIZE, BLOCK_KV, BLOCK_M,
                 KV_PAGE_STRIDE, KVS_PAGE_STRIDE, NUM_WARPS, NUM_BUFFERS, DEPTH,
                 UNROLL, PAGE_PIPE, M_CHUNK, NUM_CHAINS, FOLD_ASM, RELU_ADD,
                 PRESHUFFLE, USE_BUFFER_LOAD, RELAXED_STORE,
                 MFMA_NONK_DIM, HAS_KV_SPLIT,
                 KV_REREAD)

    if DYNAMIC:
        desc = sched_ptr + gl.program_id(0) * 4
        batch_id = gl.load(desc + 0)
        row_block = gl.load(desc + 1)
        slice_idx = gl.load(desc + 2)
        num_slices = gl.load(desc + 3)
        split_id: gl.int32 = 0
        if num_slices <= slice_idx:
            return
    else:
        # Reversed: the walk is trimmed at each block's causal limit, so block 0
        # of a prefill chunk has the least work and the last has the most.
        # Longest first keeps the grid's tail short.
        row_block = gl.num_programs(0) - 1 - gl.program_id(0)
        batch_id = gl.program_id(1)
        split_id = gl.program_id(2)
        slice_idx: gl.int32 = 0
        num_slices: gl.int32 = 0

    context_len = gl.load(context_lens_ptr + batch_id)
    if context_len <= 0:
        return

    # The trailing block still owns BLOCK_M real rows when BLOCK_M does not
    # divide next_n. Rows two blocks share are computed twice, identically.
    n0 = row_block if BLOCK_M == 1 else gl.minimum(row_block * BLOCK_M,
                                                   next_n - BLOCK_M)
    q_base = batch_id.to(gl.int64) * stride_q_b + n0.to(gl.int64) * stride_q_n
    qs_base = batch_id.to(gl.int64) * stride_qs_b + n0.to(gl.int64) * stride_qs_n
    w_row = batch_id.to(gl.int64) * next_n + n0

    mfma_qs, q_scales, w_blocks, limits = (), (), (), ()
    for r in gl.static_range(0, BLOCK_M):
        q, qs, w = _load_q_row(cfg, Q_ptr + (q_base + r * stride_q_n),
                               q_scales_ptr + (qs_base + r * stride_qs_n),
                               weights_ptr + (w_row + r) * stride_w_s)
        mfma_qs, q_scales, w_blocks = mfma_qs + (q,), q_scales + (qs,), w_blocks + (w,)
        limits = limits + (context_len - next_n + n0 + r,)

    pgm = Program.initialize(cfg, logits_ptr + w_row.to(gl.int64) * stride_logits_s,
                             stride_logits_s, stride_logits_k, context_len,
                             context_len - next_n + n0, split_id, num_kv_splits,
                             slice_idx, num_slices, HAS_KV_SPLIT, DYNAMIC)
    if pgm.tile_lo >= pgm.tile_hi:
        return
    row_hi = pgm.row_bounds(limits, BLOCK_M)

    blk_ptr = block_table_ptr + batch_id * stride_blk_b
    # Both bounds matter: the context says how many pages are filled, the
    # table's width how many exist.
    last_page_row = gl.minimum((context_len + PAGE_SIZE - 1) // PAGE_SIZE,
                               max_blocks) - 1

    if PRESHUFFLE:
        loader = RegLoader.initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr,
                                      last_page_row)
        _loop_with_reg(pgm, loader, mfma_qs, q_scales, w_blocks, row_hi)
    else:
        loader = LDSLoader.initialize(cfg, KV_ptr, kv_scales_ptr, blk_ptr,
                                      last_page_row)
        _loop_with_lds(pgm, loader, mfma_qs, q_scales, w_blocks, row_hi)


@triton.jit
def _pa_mqa_logits_mxfp4_sched_kernel(
    context_lens_ptr, sched_ptr, batch, next_n, num_ctas,
    BLOCK_M: tl.constexpr, BLOCK_KV: tl.constexpr, ROW_BLOCKS: tl.constexpr,
    ALIGN_W: tl.constexpr, BLOCK_P: tl.constexpr,
):
    # A "unit" is one (sequence, row block) pair; a "slot" is one workgroup of
    # the launch. The job is to give every slot a slice of some unit's KV walk,
    # all slices about the same length. Steps 1 to 3 are cheap reductions over
    # all ALIGN_W units, so every program just redoes them, each then writes
    # descriptors for the BLOCK_P slots it owns.

    # Step 1: how many KV tiles each unit actually walks.
    unit = tl.arange(0, ALIGN_W)
    seq = unit // ROW_BLOCKS
    row_blk = unit % ROW_BLOCKS
    live = unit < batch * ROW_BLOCKS
    ctx = tl.load(context_lens_ptr + seq, mask=live, other=0)

    # The same causal trim the main kernel applies, clamped trailing row block
    # and all, so this counts tiles that are really walked.
    first_row = (row_blk if BLOCK_M == 1
                 else tl.minimum(row_blk * BLOCK_M, next_n - BLOCK_M))
    keys = tl.minimum(ctx, ctx - next_n + first_row + BLOCK_M)
    tiles = tl.maximum(tl.cdiv(keys, BLOCK_KV), 0)
    tiles = tl.where(live & (ctx > 0), tiles, 0)

    # Step 2: how many tiles go in a slice, the same for every slot
    total_tiles = tl.sum(tiles)
    live_units = tl.sum((tiles > 0).to(tl.int32))
    tiles_per_slice = tl.maximum(
        tl.cdiv(total_tiles, tl.maximum(num_ctas - live_units, 1)), 1)

    # Step 3: lay each unit's slices end to end over the slots, so unit i owns
    # slots [first_slot_i, first_slot_i + n_slices_i).
    n_slices = tl.cdiv(tiles, tiles_per_slice)
    first_slot = tl.cumsum(n_slices) - n_slices      # exclusive prefix sum
    slots_used = tl.sum(n_slices)

    # Step 4: Calculate ownership.
    # An empty unit ends where it starts, so it is counted from its own slot onward and never owns one.
    slot = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    owner_unit = tl.sum(
        ((first_slot + n_slices)[None, :] <= slot[:, None]).to(tl.int32), axis=1)
    owner_unit = tl.minimum(owner_unit, ALIGN_W - 1)

    is_owner_unit = unit[None, :] == owner_unit[:, None]
    owner_first_slot = tl.sum(tl.where(is_owner_unit, first_slot[None, :], 0), axis=1)
    owner_slices = tl.sum(tl.where(is_owner_unit, n_slices[None, :], 0), axis=1)

    # Step 5: name the slice and write the descriptor.
    slice_idx = slot - owner_first_slot
    # A slot past the last slice owns nothing
    idle = (slot >= slots_used) | (slice_idx >= owner_slices)
    owner_unit = tl.where(idle, 0, owner_unit)
    # Which slice of how many
    slice_idx = tl.where(idle, 1, slice_idx)
    num_slices = tl.where(idle, 1, tl.maximum(owner_slices, 1))
    # save the schedule, ownership info and slide indices and num slices
    in_range = slot < num_ctas
    base = sched_ptr + slot * 4
    tl.store(base + 0, owner_unit // ROW_BLOCKS, mask=in_range)
    tl.store(base + 1, owner_unit % ROW_BLOCKS, mask=in_range)
    tl.store(base + 2, slice_idx, mask=in_range)
    tl.store(base + 3, num_slices, mask=in_range)
