"""
tl_lamport2.py -- all-reduce + residual-add + RMSNorm as 2 kernels, after
aiter CAR's structure (2026-11-16). Plain Triton: runs with Triton 3.4+.

Kernel 1, _ar_kernel: Lamport all-reduce of x plus the residual, element-wise
over the flat tensor in chunks of BLK words (2 bf16 per word), so any size
uses many programs. Writes residual_out (bf16).
  one-shot: push the chunk to every peer; poll; sum in rank order + residual.
  two-shot: the flat tensor is cut into NGPU parts; rank p owns part p.
            Phase 1 pushes chunk j of part p to rank p. The owner sums its
            chunk in rank order + residual and pushes the result to every
            peer (phase 2), which copy it into residual_out. Every rank
            receives ~2 (n-1)/n of the data instead of n-1 copies.
Kernel 2, _norm_kernel: local RMSNorm of residual_out, one program per row,
bf16 or static-scale fp8 out (as trms QUANT).

Polls are cache-volatile loads (cache_modifier=".cv"); see _poll.
Pushes are write-through stores (cache_modifier=".wt"): the inbox is uncached
on its owner, but the sender's IPC mapping can hold a pushed line in the
sender's L2 until eviction or kernel end. 2 ranks waiting for each other then
never see each other's data: an intermittent deadlock (2026-11-16).

Protocol details: uncached inboxes filled with 0xFFFF,
senders turn 0xFFFF halves into 0x7FFF, a value has arrived when it is not
0xFFFF, 3 rotating buffers, program j owns fixed positions and one round
counter, 16-byte polls (plain load + empty asm with a memory clobber),
offset and pointer hints so every access stays 16 bytes.

Inbox of each rank, int32 words: [one-shot | two-shot phase 1 | phase 2],
each [3 buffers][NGPU sources][SLOT words].
"""
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_GPUS = 8
_ipc = None


def _load_ipc():
    """JIT-build csrc/lamport_ipc.cpp into ~/.aiter/triton_rmsnorm/torch-<version>/
    (TRITON_RMSNORM_BUILD_DIR overrides): one directory per torch build, and
    the package directory may be read-only. Build once in one process before
    a multi-rank run."""
    global _ipc
    if _ipc is None:
        from torch.utils.cpp_extension import load
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        tag = torch.__version__.replace("+", "_")
        root = os.environ.get("TRITON_RMSNORM_BUILD_DIR",
                              os.path.join(os.path.expanduser("~"), ".aiter", "triton_rmsnorm"))
        build = os.path.join(root, f"torch-{tag}")
        os.makedirs(build, exist_ok=True)
        _ipc = load(name="_lamport_ipc",
                    sources=[os.path.join(_HERE, "csrc", "lamport_ipc.cpp")],
                    extra_include_paths=[os.path.join(_HERE, "csrc"),
                                         os.path.join(rocm, "include")],
                    extra_cflags=["-D__HIP_PLATFORM_AMD__"],
                    extra_ldflags=[f"-L{rocm}/lib", "-lamdhip64"],
                    build_directory=build, verbose=False)
    return _ipc


@triton.jit
def _pick(B0, B1, B2, B3, B4, B5, B6, B7, G: tl.constexpr):
    r = B0
    if G == 1:
        r = B1
    if G == 2:
        r = B2
    if G == 3:
        r = B3
    if G == 4:
        r = B4
    if G == 5:
        r = B5
    if G == 6:
        r = B6
    if G == 7:
        r = B7
    return r

ONE, TWO = 1, 2


def _no_buffer_ops():
    """Compile and launch with Triton's AMD buffer-ops pass off, for these
    kernels only. With it on (Triton 3.8 default), some masked inbox polls
    became buffer loads that never saw the peers' pushes: every rank but one
    waited forever (2026-11-16). Older Triton has no such knob."""
    import contextlib
    try:
        from triton import knobs
        amd = knobs.amd
        assert hasattr(amd, "use_buffer_ops")
    except Exception:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def scope():
        with amd.scope():
            amd.use_buffer_ops = False
            yield
    return scope()


@triton.jit
def _aptr(addr):
    return tl.multiple_of(addr.to(tl.pointer_type(tl.int32)), 16)


@triton.jit
def _hint(offs):
    return tl.max_contiguous(tl.multiple_of(offs, 4), 4)


@triton.jit
def _poll(ptrs, mask):
    # .cv: the load bypasses the caches. The inbox is uncached memory, but
    # without it a poll could keep reading a stale cached line on the
    # receiver: the data was in memory, the kernel never saw it (2026-11-16).
    w = tl.load(ptrs, mask=mask, other=0, cache_modifier=".cv")
    return tl.inline_asm_elementwise("", "=v,0,~{memory}", [w], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _sentinel(w):
    return ((w & 0xFFFF) == 0xFFFF) | ((w & -65536) == -65536)


@triton.jit
def _canon(w):
    # never send the sentinel: 0xFFFF halves become 0x7FFF (both bf16 NaN)
    lo = w & 0xFFFF
    hi = w & -65536
    lo = tl.where(lo == 0xFFFF, 0x7FFF, lo)
    hi = tl.where(hi == -65536, 0x7FFF0000, hi)
    return hi | lo


@triton.jit
def _lo(w):
    return (w << 16).to(tl.float32, bitcast=True)


@triton.jit
def _hi(w):
    return (w & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _pack(f_lo, f_hi):
    return (f_hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16) | (
        f_lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF)


@triton.jit
def _wait(me, base, offs, m, dbg_ptr, pid, NGPU: tl.constexpr, RANK: tl.constexpr,
          SLOT: tl.constexpr, MAXIT: tl.constexpr):
    # Wait until the slots of all peers in [base] hold data at offs (masked).
    # MAXIT > 0 (debug): stop after MAXIT polls and record, per peer, how
    # many values never arrived in dbg_ptr[pid * 8 + peer].
    pending = 1
    it = 0
    while pending > 0 and (MAXIT == 0 or it < MAXIT):
        pending = 0
        it += 1
        for g in tl.static_range(NGPU):
            if g != RANK:
                w = _poll(me + _hint(base + g * SLOT + offs), m)
                pending += tl.sum((_sentinel(w) & m).to(tl.int32), axis=0)
    if MAXIT > 0:
        for g in tl.static_range(NGPU):
            if g != RANK:
                w = _poll(me + _hint(base + g * SLOT + offs), m)
                tl.store(dbg_ptr + pid * 8 + g, tl.sum((_sentinel(w) & m).to(tl.int32), axis=0))


@triton.jit
def _rearm(me, base, offs, BLK: tl.constexpr, NGPU: tl.constexpr, RANK: tl.constexpr,
           SLOT: tl.constexpr):
    full = tl.full([BLK], -1, tl.int32)
    for g in tl.static_range(NGPU):
        if g != RANK:
            tl.store(me + _hint(base + g * SLOT + offs), full)


@triton.jit
def _ar_kernel(x_ptr, res_ptr, res_out_ptr, cnt_ptr, nw, dbg_ptr,
               B0, B1, B2, B3, B4, B5, B6, B7,
               RANK: tl.constexpr, NGPU: tl.constexpr, SLOT: tl.constexpr,
               BLK: tl.constexpr, MODE: tl.constexpr, HAS_RES: tl.constexpr,
               MAXIT: tl.constexpr = 0):
    # nw = words in the tensor; one program = BLK words of every part it owns.
    pid = tl.program_id(0)
    loc = _hint(pid * BLK + tl.arange(0, BLK))       # position inside a slot
    c = tl.load(cnt_ptr + pid)
    me = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, RANK))
    BUF: tl.constexpr = NGPU * SLOT                     # words of one buffer
    REGION: tl.constexpr = 3 * BUF                      # words of one region

    if MODE == 1:
        m = loc < nw
        xw = tl.load(x_ptr + loc, mask=m, other=0)
        send = _canon(xw)
        for g in tl.static_range(NGPU):
            if g != RANK:
                dst = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, g))
                tl.store(dst + _hint(c * BUF + RANK * SLOT + loc), send, mask=m,
                         cache_modifier=".wt")
        _wait(me, c * BUF, loc, m, dbg_ptr, pid, NGPU, RANK, SLOT, MAXIT)
        a_lo = tl.zeros([BLK], dtype=tl.float32)
        a_hi = tl.zeros([BLK], dtype=tl.float32)
        for g in tl.static_range(NGPU):
            if g == RANK:
                w = xw
            else:
                w = _poll(me + _hint(c * BUF + g * SLOT + loc), m)
            a_lo += _lo(w)
            a_hi += _hi(w)
        if HAS_RES:
            rw = tl.load(res_ptr + loc, mask=m, other=0)
            a_lo += _lo(rw)
            a_hi += _hi(rw)
        tl.store(res_out_ptr + loc, _pack(a_lo, a_hi), mask=m)
        _rearm(me, ((c + 2) % 3) * BUF, loc, BLK, NGPU, RANK, SLOT)
    else:
        # parts of PART words; part p = [p * PART, min((p + 1) * PART, nw))
        PART = (nw + NGPU - 1) // NGPU
        PART = tl.multiple_of((PART + BLK - 1) // BLK * BLK, BLK)
        R1 = REGION                                     # phase 1 region
        R2 = 2 * REGION                                 # phase 2 region
        # phase 1: chunk pid of part p goes to rank p
        for p in tl.static_range(NGPU):
            if p != RANK:
                gpos = _hint(p * PART + loc)
                mp = (loc < PART) & (gpos < nw)
                v = tl.load(x_ptr + gpos, mask=mp, other=0)
                dst = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, p))
                tl.store(dst + _hint(R1 + c * BUF + RANK * SLOT + loc), _canon(v), mask=mp,
                         cache_modifier=".wt")
        # reduce my part
        gme = _hint(RANK * PART + loc)
        mm = (loc < PART) & (gme < nw)
        xw = tl.load(x_ptr + gme, mask=mm, other=0)
        _wait(me, R1 + c * BUF, loc, mm, dbg_ptr, pid, NGPU, RANK, SLOT, MAXIT)
        a_lo = tl.zeros([BLK], dtype=tl.float32)
        a_hi = tl.zeros([BLK], dtype=tl.float32)
        for g in tl.static_range(NGPU):
            if g == RANK:
                w = xw
            else:
                w = _poll(me + _hint(R1 + c * BUF + g * SLOT + loc), mm)
            a_lo += _lo(w)
            a_hi += _hi(w)
        if HAS_RES:
            rw = tl.load(res_ptr + gme, mask=mm, other=0)
            a_lo += _lo(rw)
            a_hi += _hi(rw)
        red = _pack(a_lo, a_hi)
        tl.store(res_out_ptr + gme, red, mask=mm)
        # phase 2: my reduced chunk to every peer
        sred = _canon(red)
        for g in tl.static_range(NGPU):
            if g != RANK:
                dst = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, g))
                tl.store(dst + _hint(R2 + c * BUF + RANK * SLOT + loc), sred, mask=mm,
                         cache_modifier=".wt")
        # receive the other parts: one wait for all peers, then copy
        pending = 1
        while pending > 0:
            pending = 0
            for p in tl.static_range(NGPU):
                if p != RANK:
                    mp = (loc < PART) & (p * PART + loc < nw)
                    w = _poll(me + _hint(R2 + c * BUF + p * SLOT + loc), mp)
                    pending += tl.sum((_sentinel(w) & mp).to(tl.int32), axis=0)
        for p in tl.static_range(NGPU):
            if p != RANK:
                gpos = _hint(p * PART + loc)
                mp = (loc < PART) & (gpos < nw)
                w = _poll(me + _hint(R2 + c * BUF + p * SLOT + loc), mp)
                tl.store(res_out_ptr + gpos, w, mask=mp)
        r = (c + 2) % 3
        _rearm(me, R1 + r * BUF, loc, BLK, NGPU, RANK, SLOT)
        _rearm(me, R2 + r * BUF, loc, BLK, NGPU, RANK, SLOT)
    tl.debug_barrier()
    tl.store(cnt_ptr + pid, (c + 1) % 3)


@triton.jit
def _bases_vec(B0, B1, B2, B3, B4, B5, B6, B7, NGPU: tl.constexpr):
    # [NGPU] int64 vector of the ranks' inbox bases
    g = tl.arange(0, NGPU)
    v = tl.zeros([NGPU], dtype=tl.int64) + B0
    v = tl.where(g == 1, B1, v)
    v = tl.where(g == 2, B2, v)
    v = tl.where(g == 3, B3, v)
    v = tl.where(g == 4, B4, v)
    v = tl.where(g == 5, B5, v)
    v = tl.where(g == 6, B6, v)
    v = tl.where(g == 7, B7, v)
    return v


@triton.jit
def _ar_ws_kernel(x_ptr, res_ptr, res_out_ptr, cnt_ptr, nw,
                  B0, B1, B2, B3, B4, B5, B6, B7,
                  RANK: tl.constexpr, NGPU: tl.constexpr, SLOT: tl.constexpr,
                  BLK: tl.constexpr, HAS_RES: tl.constexpr):
    # One-shot with aiter's wave split: a [NGPU, BLK] tile, row g = GPU g.
    # With num_warps = NGPU and 4 words per lane, wave g owns row g, so each
    # lane pushes once, polls once and re-arms once. Row RANK is this rank's
    # x. Rows are not rotated: every rank sums in the same order, same bits.
    pid = tl.program_id(0)
    g = tl.arange(0, NGPU)[:, None]                       # [NGPU, 1]
    loc = _hint(pid * BLK + tl.arange(0, BLK))            # [BLK]
    loc2 = g * 0 + loc[None, :]                           # [NGPU, BLK]
    m = loc < nw
    m2 = loc2 < nw
    peer = (g != RANK) & m2
    c = tl.load(cnt_ptr + pid)
    BUF: tl.constexpr = NGPU * SLOT
    bases = _bases_vec(B0, B1, B2, B3, B4, B5, B6, B7, NGPU)[:, None]
    me = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, RANK))

    xw = tl.load(x_ptr + loc, mask=m, other=0)            # [BLK]
    x2 = g * 0 + xw[None, :]                              # [NGPU, BLK]
    # push: wave g sends x to GPU g, slot [c][RANK]
    dst = tl.multiple_of((bases + (c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                         [16, 1])
    tl.store(dst + loc2, _canon(x2), mask=peer, cache_modifier=".wt")
    # poll: wave g reads slot [c][g] of my inbox; carry the values
    src = me + c * BUF + g * SLOT + loc2
    # unmasked polls (slots are allocated in full; a tail mask miscompiled,
    # see aiter/ops/triton/triton_ar), a barrier per iteration (shared reduction scratch)
    w = tl.load(src, cache_modifier=".cv")
    while tl.sum(tl.sum((_sentinel(w) & peer).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()
        w = tl.load(src, cache_modifier=".cv")
    w = tl.where(g == RANK, x2, w)
    lo = tl.sum(_lo(w), axis=0)                            # [BLK]
    hi = tl.sum(_hi(w), axis=0)
    if HAS_RES:
        rw = tl.load(res_ptr + loc, mask=m, other=0)
        lo += _lo(rw)
        hi += _hi(rw)
    tl.store(res_out_ptr + loc, _pack(lo, hi), mask=m)
    # re-arm buffer (c+2)%3: wave g rewrites slot g
    full = tl.full([NGPU, BLK], -1, tl.int32)
    tl.store(me + ((c + 2) % 3) * BUF + g * SLOT + loc2, full, mask=(g != RANK), cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(cnt_ptr + pid, (c + 1) % 3)


@triton.jit
def _ar_ws2_kernel(x_ptr, res_ptr, res_out_ptr, cnt_ptr, nw,
                   B0, B1, B2, B3, B4, B5, B6, B7,
                   RANK: tl.constexpr, NGPU: tl.constexpr, SLOT: tl.constexpr,
                   BLK: tl.constexpr, HAS_RES: tl.constexpr, BASE: tl.constexpr):
    # Two-shot with the wave split: row g of the [NGPU, BLK] tile = GPU g.
    # Phase 1: wave g pushes chunk pid of part g to GPU g and polls GPU g's
    # chunk of my part. Phase 2: wave g pushes my reduced chunk to GPU g and
    # receives GPU g's reduced chunk. Inbox: phase 1 and phase 2 regions of
    # [3][NGPU][SLOT] words from word BASE (SLOT = one rank's part).
    pid = tl.program_id(0)
    g = tl.arange(0, NGPU)[:, None]
    loc = _hint(pid * BLK + tl.arange(0, BLK))
    loc2 = g * 0 + loc[None, :]
    c = tl.load(cnt_ptr + pid)
    BUF: tl.constexpr = NGPU * SLOT
    REGION: tl.constexpr = 3 * BUF
    R1: tl.constexpr = BASE
    R2: tl.constexpr = BASE + REGION
    PART = (nw + NGPU - 1) // NGPU
    PART = tl.multiple_of((PART + BLK - 1) // BLK * BLK, BLK)
    bases = _bases_vec(B0, B1, B2, B3, B4, B5, B6, B7, NGPU)[:, None]
    me = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, RANK))

    gpos2 = g * PART + loc2                                # row g: part g
    in2 = (loc2 < PART) & (gpos2 < nw)
    peer = (g != RANK) & in2
    # phase 1 push
    xv = tl.load(x_ptr + gpos2, mask=peer, other=0)
    dst1 = tl.multiple_of((bases + (R1 + c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                          [16, 1])
    tl.store(dst1 + loc2, _canon(xv), mask=peer, cache_modifier=".wt")
    # reduce my part: row g = GPU g's chunk of my part, row RANK = my x
    gme = _hint(RANK * PART + loc)
    mm = (loc < PART) & (gme < nw)
    mm2 = mm[None, :] & (g >= 0)                          # broadcast, stays boolean
    mine = (g != RANK) & mm2
    src1 = me + R1 + c * BUF + g * SLOT + loc2
    w = tl.load(src1, cache_modifier=".cv")
    while tl.sum(tl.sum((_sentinel(w) & mine).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()
        w = tl.load(src1, cache_modifier=".cv")
    xme = tl.load(x_ptr + (g * 0 + gme[None, :]), mask=(g == RANK) & mm2, other=0)
    w = tl.where(g == RANK, xme, w)
    lo = tl.sum(_lo(w), axis=0)
    hi = tl.sum(_hi(w), axis=0)
    if HAS_RES:
        rw = tl.load(res_ptr + gme, mask=mm, other=0)
        lo += _lo(rw)
        hi += _hi(rw)
    red = _pack(lo, hi)
    tl.store(res_out_ptr + gme, red, mask=mm)
    # phase 2 push: my reduced chunk to every GPU
    red2 = g * 0 + _canon(red)[None, :]
    dst2 = tl.multiple_of((bases + (R2 + c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                          [16, 1])
    tl.store(dst2 + loc2, red2, mask=mine, cache_modifier=".wt")
    # receive: row g = GPU g's reduced chunk of part g
    src2 = me + R2 + c * BUF + g * SLOT + loc2
    v = tl.load(src2, cache_modifier=".cv")
    while tl.sum(tl.sum((_sentinel(v) & peer).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()
        v = tl.load(src2, cache_modifier=".cv")
    tl.store(res_out_ptr + gpos2, v, mask=peer)
    # re-arm buffer (c+2)%3 of both regions: wave g rewrites slot g
    r = (c + 2) % 3
    full = tl.full([NGPU, BLK], -1, tl.int32)
    tl.store(me + R1 + r * BUF + g * SLOT + loc2, full, mask=(g != RANK), cache_modifier=".wt")
    tl.store(me + R2 + r * BUF + g * SLOT + loc2, full, mask=(g != RANK), cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(cnt_ptr + pid, (c + 1) % 3)


@triton.jit
def _norm_kernel(v_ptr, w_ptr, out_ptr, scale_ptr, eps,
                 W: tl.constexpr, QUANT: tl.constexpr, FP8_MAX: tl.constexpr):
    # One row: v = residual_out (bf16 pairs as int32 words).
    row = tl.program_id(0)
    cols = _hint(tl.arange(0, W))
    offs = _hint(row * W + cols)
    v = tl.load(v_ptr + offs)
    f_lo = _lo(v)
    f_hi = _hi(v)
    rn = tl.rsqrt(tl.sum(f_lo * f_lo + f_hi * f_hi, axis=0) / (2 * W) + eps)
    ww = tl.load(w_ptr + cols)
    y_lo = f_lo * rn * _lo(ww)
    y_hi = f_hi * rn * _hi(ww)
    if QUANT:
        inv = 1.0 / tl.load(scale_ptr)
        q_lo = tl.minimum(tl.maximum(y_lo * inv, -FP8_MAX), FP8_MAX).to(tl.float8e4nv)
        q_hi = tl.minimum(tl.maximum(y_hi * inv, -FP8_MAX), FP8_MAX).to(tl.float8e4nv)
        b_lo = q_lo.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
        b_hi = q_hi.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
        tl.store(out_ptr + offs, ((b_hi << 8) | b_lo).to(tl.int16))
    else:
        tl.store(out_ptr + offs, _pack(y_lo, y_hi))


class TlLamport2RMSNorm:
    """out, residual_out = RMSNorm(all-reduce(x) + residual), bf16 in, bf16 or
    static-scale fp8 out, at most max_rows rows of hidden values (power of 2,
    512 to 16384). one_shot_max_rows / two_shot_max_rows pick the mode."""

    # 256 words, 1 wave: best of the 2026-11-16 sweep (one wave also makes the
    # wait loop's reduction wave-local)
    def __init__(self, rank, nGPU, hidden, max_rows, blk_words=256, num_warps=1,
                 one_shot_max_rows=4, two_shot_max_rows=None):
        assert nGPU <= MAX_GPUS and hidden & (hidden - 1) == 0 and 512 <= hidden <= 16384
        self.rank, self.nGPU, self.hidden, self.max_rows = rank, nGPU, hidden, max_rows
        self.words = hidden // 2
        self.blk, self.num_warps = blk_words, num_warps
        self.one_max = one_shot_max_rows
        self.two_max = max_rows if two_shot_max_rows is None else two_shot_max_rows
        max_words = max_rows * self.words
        # slot: words a source can write at most (one-shot: all; two-shot: a part)
        self.slot = triton.cdiv(max_words, blk_words) * blk_words
        # wave-split chunk: 64 words per wave row (triton_ar sweep, 2026-11-16;
        # was 256). TL2_WS_BLK overrides.
        self.ws_blk = int(os.environ.get("TL2_WS_BLK", "64"))
        self.max_programs = triton.cdiv(max_words, min(blk_words, self.ws_blk))
        self.wave_split = os.environ.get("TL2_WS", "1") == "1" and nGPU in (2, 4, 8)
        if self.wave_split:
            # one-shot region for one_max rows, two-shot regions for one part
            ws = self.ws_blk
            self.slot1 = triton.cdiv(min(self.one_max, max_rows) * self.words, ws) * ws
            self.slot2 = triton.cdiv(triton.cdiv(max_words, nGPU), ws) * ws
            self.base2 = 3 * nGPU * self.slot1
            words_total = 3 * nGPU * (self.slot1 + 2 * self.slot2)
        else:
            words_total = 9 * nGPU * self.slot
        self.inbox_bytes = words_total * 4
        self._buf = _load_ipc().UncachedIpcBuffer(rank, nGPU, self.inbox_bytes, 0xFF)
        dev = torch.device("cuda", rank)
        # One counter row per mode: a round of one mode must not advance the
        # other mode's buffers, which only that mode re-arms.
        # row 2: the wave-split one-shot kernel (its own program geometry)
        # rows 2 / 3: the wave-split one-shot / two-shot kernels
        self.cnt = torch.zeros((4, self.max_programs), dtype=torch.int32, device=dev)

        self._bases = None
        self.norm_warps = min(16, self.words // 256)
        # debug: TL2_MAXIT > 0 caps the polls and records missing values
        self.maxit = int(os.environ.get("TL2_MAXIT", "0"))
        self.dbg = torch.zeros(self.max_programs * 8, dtype=torch.int32, device=dev)

    def register(self, gloo_group):
        self._buf.exchange(gloo_group)
        b = [int(v) for v in self._buf.get_peer_bases_cpu()]
        self._bases = b + [b[self.rank]] * (MAX_GPUS - len(b))

    def precompile(self, quant_modes=(False, True), has_res=(True, False)):
        """Compile both kernels for every mode without running them. Call on
        all ranks before the first collective call (debug aid for compile skew)."""
        dev = self.cnt.device
        z = torch.zeros(self.words * 8, dtype=torch.int32, device=dev)
        with _no_buffer_ops():
            for mode in (ONE, TWO):
                for hr in has_res:
                    _ar_kernel.warmup(z, z, z, self.cnt[0], 16, self.dbg, *([1 << 40] * 8),
                                      RANK=self.rank, NGPU=self.nGPU, SLOT=self.slot,
                                      BLK=self.blk, MODE=mode, HAS_RES=hr, MAXIT=self.maxit,
                                      num_warps=self.num_warps, grid=(1,))
            for q in quant_modes:
                _norm_kernel.warmup(z, z, z.view(torch.int16) if q else z,
                                    torch.ones(1, device=dev), 1e-6, W=self.words, QUANT=q,
                                    FP8_MAX=448.0, num_warps=self.norm_warps, grid=(1,))

    def mode(self, rows):
        if rows <= self.one_max or self.nGPU == 1:
            return ONE
        return TWO

    def __call__(self, x, residual, weight, eps, out_scale=None, parts=("ar", "norm")):
        rows, hidden = x.shape
        assert x.dtype == torch.bfloat16 and hidden == self.hidden and rows <= self.max_rows
        assert x.is_contiguous() and x.data_ptr() % 16 == 0
        quant = out_scale is not None
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn if quant else x.dtype,
                          device=x.device)
        res_out = torch.empty_like(x)
        if rows == 0:
            return out, (res_out if residual is not None else None)
        nw = rows * self.words
        with _no_buffer_ops():
            self._launch(x, residual, weight, eps, out_scale, out, res_out, rows, nw, quant,
                         parts)
        return out, (res_out if residual is not None else None)

    def _launch(self, x, residual, weight, eps, out_scale, out, res_out, rows, nw, quant,
                parts=("ar", "norm")):
        # parts: ("ar",) / ("norm",) run one kernel alone, for timing breakdowns
        iv = lambda t: t.view(torch.int32)
        mode = self.mode(rows)
        if mode == ONE:
            grid = triton.cdiv(nw, self.blk)
        else:
            part = triton.cdiv(triton.cdiv(nw, self.nGPU), self.blk) * self.blk
            grid = part // self.blk
        if "ar" in parts and mode == ONE and self.wave_split:
            _ar_ws_kernel[(triton.cdiv(nw, self.ws_blk),)](
                iv(x), iv(residual) if residual is not None else iv(x), iv(res_out),
                self.cnt[2], nw, *self._bases, RANK=self.rank, NGPU=self.nGPU,
                SLOT=self.slot1, BLK=self.ws_blk, HAS_RES=residual is not None,
                num_warps=self.nGPU)
        elif "ar" in parts and mode == TWO and self.wave_split:
            part = triton.cdiv(triton.cdiv(nw, self.nGPU), self.ws_blk) * self.ws_blk
            _ar_ws2_kernel[(part // self.ws_blk,)](
                iv(x), iv(residual) if residual is not None else iv(x), iv(res_out),
                self.cnt[3], nw, *self._bases, RANK=self.rank, NGPU=self.nGPU,
                SLOT=self.slot2, BLK=self.ws_blk, HAS_RES=residual is not None,
                BASE=self.base2, num_warps=self.nGPU)
        elif "ar" in parts:
            _ar_kernel[(grid,)](
                iv(x), iv(residual) if residual is not None else iv(x), iv(res_out),
                self.cnt[mode - 1], nw, self.dbg,
                *self._bases, RANK=self.rank, NGPU=self.nGPU, SLOT=self.slot, BLK=self.blk,
                MODE=mode, HAS_RES=residual is not None, MAXIT=self.maxit,
                num_warps=self.num_warps)
        if "norm" in parts:
            _norm_kernel[(rows,)](
                iv(res_out), iv(weight), out.view(torch.int16) if quant else iv(out),
                out_scale if quant else self.cnt, eps, W=self.words, QUANT=quant,
                FP8_MAX=float(torch.finfo(torch.float8_e4m3fn).max),
                num_warps=self.norm_warps)
