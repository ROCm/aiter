"""
triton_ar.py -- Triton all-reduce (sum) for AMD gfx950, after aiter CAR's
Lamport kernels. Plain Triton: Triton 3.4+. 2026-11-16.

    ar = TritonAllReduce(rank, world, max_bytes=64 << 20)
    ar.register(gloo_group)          # collective, eager
    y = ar(x)                        # bf16 / fp16, numel % 8 == 0, 16-byte aligned

Graph capture needs nothing extra: x is only read locally.

Protocol (from the fused all-reduce + RMSNorm kernels, measured against
aiter CAR): every rank pushes data into the peers' uncached inboxes; the
data is the arrival signal (inboxes start as 0xFFFF, senders turn 0xFFFF
halves into 0x7FFF -- both NaN); 3 rotating buffers replace barriers; each
program owns fixed positions and one round counter in device memory.
  one-shot (<= one_shot_bytes): push the chunk to every peer, sum all ranks.
  two-shot: rank p reduces part p of the tensor (phase 1), then pushes the
            result to every peer (phase 2). Each rank receives ~2 (n-1)/n of
            the data instead of n-1 copies.
Wave split (aiter): a [NGPU, BLK] tile with num_warps = NGPU, so wave g
handles GPU g only: one push, one poll and one re-arm per lane; waves combine
through shared memory. Rows are GPU ids, not rotated: all ranks add in the
same order and get the same bits (fp32 sum in rank order, rounded once).

Rules found the hard way (see README.md):
  * polls are cache_modifier=".cv" loads, pushes ".wt" stores, or a poll can
    keep reading a stale cached line and deadlock;
  * re-arm stores are ".wt" too: a plain store could stay in the receiver's
    cache while memory still held the old round's data, and a later poll
    took that stale data as new (wrong values, intermittent, 2026-11-16);
  * a poll loop whose exit test is a block-wide tl.sum needs a barrier per
    iteration: the waves share the reduction scratch, so a fast wave could
    overwrite it and a slow one read a 0 and leave with lanes still unwritten
    (sentinel copied to the output, uneven-part tails, 2026-11-16);
  * poll loads are unmasked (slots are always allocated in full; the mask
    only enters the exit test): with a tail mask the loop compiled to
    per-dword branches with irreducible control flow, and some lanes were
    never re-read -- the sentinel was copied to the output (2026-11-16);
  * fp8 kernel: re-load the inbox after the poll loops (loop-carried values
    that are reshaped afterwards came out stale), and the scale is polled in
    a [NGPU, 32] tile, not a 1-D [NGPU] vector (that deadlocked);
  * Triton's AMD buffer-ops pass is switched off for these kernels.
"""
import contextlib
import os

import torch
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_GPUS = 8
# words (2 values each) per wave row: one-shot, two-shot (TAR_BLK1/2 for sweeps)
BLK1 = int(os.environ.get("TAR_BLK1", "64"))
BLK2 = int(os.environ.get("TAR_BLK2", "64"))
# One BLK per mode: a program owns fixed inbox positions and one round
# counter, so mixing BLKs within a mode desynchronises the rounds (a 1-2 MB
# BLK2=256 variant, ~2% faster there, gave wrong values; 2026-11-16).
SLOT_ALIGN = 512  # slots are a multiple of every BLK
_ipc = None


def _load_ipc():
    """JIT-build csrc/lamport_ipc.cpp into ~/.aiter/triton_ar/torch-<version>/
    (TRITON_AR_BUILD_DIR overrides): one directory per torch build, and the
    package directory may be read-only. Build once in one process before a
    multi-rank run."""
    global _ipc
    if _ipc is None:
        from torch.utils.cpp_extension import load
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        tag = torch.__version__.replace("+", "_")
        root = os.environ.get("TRITON_AR_BUILD_DIR",
                              os.path.join(os.path.expanduser("~"), ".aiter", "triton_ar"))
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


def _no_buffer_ops():
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


@triton.jit
def _bases_vec(B0, B1, B2, B3, B4, B5, B6, B7, NGPU: tl.constexpr):
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
def _aptr(addr):
    return tl.multiple_of(addr.to(tl.pointer_type(tl.int32)), 16)


@triton.jit
def _hint(offs):
    return tl.max_contiguous(tl.multiple_of(offs, 4), 4)


@triton.jit
def _sentinel(w):
    return ((w & 0xFFFF) == 0xFFFF) | ((w & -65536) == -65536)


@triton.jit
def _canon(w):
    lo = w & 0xFFFF
    hi = w & -65536
    lo = tl.where(lo == 0xFFFF, 0x7FFF, lo)
    hi = tl.where(hi == -65536, 0x7FFF0000, hi)
    return hi | lo


@triton.jit
def _lo(w, BF16: tl.constexpr):
    if BF16:
        r = (w << 16).to(tl.float32, bitcast=True)
    else:
        r = (w & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    return r


@triton.jit
def _hi(w, BF16: tl.constexpr):
    if BF16:
        r = (w & -65536).to(tl.float32, bitcast=True)
    else:
        r = (w >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    return r


@triton.jit
def _pack(f_lo, f_hi, BF16: tl.constexpr):
    if BF16:
        a = f_lo.to(tl.bfloat16).to(tl.int16, bitcast=True)
        b = f_hi.to(tl.bfloat16).to(tl.int16, bitcast=True)
    else:
        a = f_lo.to(tl.float16).to(tl.int16, bitcast=True)
        b = f_hi.to(tl.float16).to(tl.int16, bitcast=True)
    return (b.to(tl.int32) << 16) | (a.to(tl.int32) & 0xFFFF)


@triton.jit
def _ar_one_shot(x_ptr, out_ptr, cnt_ptr, nw,
                 B0, B1, B2, B3, B4, B5, B6, B7,
                 RANK: tl.constexpr, NGPU: tl.constexpr, SLOT: tl.constexpr,
                 BLK: tl.constexpr, BF16: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.arange(0, NGPU)[:, None]
    loc = _hint(pid * BLK + tl.arange(0, BLK))
    loc2 = g * 0 + loc[None, :]
    m = loc < nw
    peer = (g != RANK) & (loc2 < nw)
    c = tl.load(cnt_ptr + pid)
    BUF: tl.constexpr = NGPU * SLOT
    bases = _bases_vec(B0, B1, B2, B3, B4, B5, B6, B7, NGPU)[:, None]
    me = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, RANK))

    xw = tl.load(x_ptr + loc, mask=m, other=0)
    x2 = g * 0 + xw[None, :]
    dst = tl.multiple_of((bases + (c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                         [16, 1])
    tl.store(dst + loc2, _canon(x2), mask=peer, cache_modifier=".wt")
    src = me + c * BUF + g * SLOT + loc2
    w = tl.load(src, cache_modifier=".cv")
    while tl.sum(tl.sum((_sentinel(w) & peer).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()  # the next test reuses the reduction scratch
        w = tl.load(src, cache_modifier=".cv")
    w = tl.where(g == RANK, x2, w)
    lo = tl.sum(_lo(w, BF16), axis=0)
    hi = tl.sum(_hi(w, BF16), axis=0)
    tl.store(out_ptr + loc, _pack(lo, hi, BF16), mask=m)
    full = tl.full([NGPU, BLK], -1, tl.int32)
    tl.store(me + ((c + 2) % 3) * BUF + g * SLOT + loc2, full, mask=(g != RANK), cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(cnt_ptr + pid, (c + 1) % 3)


@triton.jit
def _ar_two_shot(x_ptr, out_ptr, cnt_ptr, nw,
                 B0, B1, B2, B3, B4, B5, B6, B7,
                 RANK: tl.constexpr, NGPU: tl.constexpr, SLOT: tl.constexpr,
                 BLK: tl.constexpr, BF16: tl.constexpr, BASE: tl.constexpr):
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

    gpos2 = g * PART + loc2
    peer = (g != RANK) & (loc2 < PART) & (gpos2 < nw)
    xv = tl.load(x_ptr + gpos2, mask=peer, other=0)
    dst1 = tl.multiple_of((bases + (R1 + c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                          [16, 1])
    tl.store(dst1 + loc2, _canon(xv), mask=peer, cache_modifier=".wt")
    gme = _hint(RANK * PART + loc)
    mm = (loc < PART) & (gme < nw)
    mm2 = mm[None, :] & (g >= 0)
    mine = (g != RANK) & mm2
    src1 = me + R1 + c * BUF + g * SLOT + loc2
    w = tl.load(src1, cache_modifier=".cv")
    while tl.sum(tl.sum((_sentinel(w) & mine).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()  # the next test reuses the reduction scratch
        w = tl.load(src1, cache_modifier=".cv")
    xme = tl.load(x_ptr + (g * 0 + gme[None, :]), mask=(g == RANK) & mm2, other=0)
    w = tl.where(g == RANK, xme, w)
    red = _pack(tl.sum(_lo(w, BF16), axis=0), tl.sum(_hi(w, BF16), axis=0), BF16)
    tl.store(out_ptr + gme, red, mask=mm)
    red2 = g * 0 + _canon(red)[None, :]
    dst2 = tl.multiple_of((bases + (R2 + c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                          [16, 1])
    tl.store(dst2 + loc2, red2, mask=mine, cache_modifier=".wt")
    src2 = me + R2 + c * BUF + g * SLOT + loc2
    v = tl.load(src2, cache_modifier=".cv")
    while tl.sum(tl.sum((_sentinel(v) & peer).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()  # the next test reuses the reduction scratch
        v = tl.load(src2, cache_modifier=".cv")
    tl.store(out_ptr + gpos2, v, mask=peer)
    r = (c + 2) % 3
    full = tl.full([NGPU, BLK], -1, tl.int32)
    tl.store(me + R1 + r * BUF + g * SLOT + loc2, full, mask=(g != RANK), cache_modifier=".wt")
    tl.store(me + R2 + r * BUF + g * SLOT + loc2, full, mask=(g != RANK), cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(cnt_ptr + pid, (c + 1) % 3)


@triton.jit
def _ar_two_shot_fp8(x_ptr, out_ptr, cnt_ptr, nw,
                     B0, B1, B2, B3, B4, B5, B6, B7,
                     RANK: tl.constexpr, NGPU: tl.constexpr, SLOT: tl.constexpr,
                     QSLOT: tl.constexpr, BF16: tl.constexpr, BASE: tl.constexpr,
                     QBASE: tl.constexpr, dbg_ptr=None, MAXIT: tl.constexpr = 0):
    """Two-shot with an fp8 all-gather (aiter CAR's open_fp8_quant scheme):
    phase 1 pushes full precision; phase 2 pushes the reduced part as e4m3fn
    with one fp32 scale (amax / 448) per 128 values -- one 64-word tile. The
    owner writes the dequantized values too, so all ranks get the same bits.
    No canonicalisation: 0xFF is NaN in e4m3fn and the scale is never NaN,
    so a word of 4 values or a scale can never be the sentinel 0xFFFFFFFF."""
    BLK: tl.constexpr = 64
    QW: tl.constexpr = 32
    pid = tl.program_id(0)
    g = tl.arange(0, NGPU)[:, None]
    loc = _hint(pid * BLK + tl.arange(0, BLK))
    loc2 = g * 0 + loc[None, :]
    c = tl.load(cnt_ptr + pid)
    BUF: tl.constexpr = NGPU * SLOT
    QBUF: tl.constexpr = NGPU * QSLOT
    QD: tl.constexpr = SLOT // 2  # fp8 data words per GPU; scales follow
    PART = (nw + NGPU - 1) // NGPU
    PART = tl.multiple_of((PART + BLK - 1) // BLK * BLK, BLK)
    bases = _bases_vec(B0, B1, B2, B3, B4, B5, B6, B7, NGPU)[:, None]
    me = _aptr(_pick(B0, B1, B2, B3, B4, B5, B6, B7, RANK))

    # phase 1: reduce-scatter, full precision (as _ar_two_shot)
    gpos2 = g * PART + loc2
    peer = (g != RANK) & (gpos2 < nw)
    xv = tl.load(x_ptr + gpos2, mask=peer, other=0)
    dst1 = tl.multiple_of((bases + (BASE + c * BUF + RANK * SLOT) * 4).to(tl.pointer_type(tl.int32)),
                          [16, 1])
    tl.store(dst1 + loc2, _canon(xv), mask=peer, cache_modifier=".wt")
    gme = _hint(RANK * PART + loc)
    mm = gme < nw
    mm2 = mm[None, :] & (g >= 0)
    mine = (g != RANK) & mm2
    src1 = me + BASE + c * BUF + g * SLOT + loc2
    w = tl.load(src1, cache_modifier=".cv")
    it = 0
    while tl.sum(tl.sum((_sentinel(w) & mine).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()
        w = tl.load(src1, cache_modifier=".cv")
        if MAXIT > 0:
            it += 1
            if it > MAXIT:
                tl.atomic_add(dbg_ptr, tl.sum(tl.sum((_sentinel(w) & mine).to(tl.int32), axis=1), axis=0))
                w = tl.where(_sentinel(w), 0, w)
    xme = tl.load(x_ptr + (g * 0 + gme[None, :]), mask=(g == RANK) & mm2, other=0)
    w = tl.where(g == RANK, xme, tl.where(mine, w, 0))
    red = _pack(tl.sum(_lo(w, BF16), axis=0), tl.sum(_hi(w, BF16), axis=0), BF16)
    lo = _lo(red, BF16)
    hi = _hi(red, BF16)  # rounded to the output dtype first, as aiter
    amax = tl.maximum(tl.max(tl.abs(lo), axis=0), tl.max(tl.abs(hi), axis=0))
    scale = amax / 448.0
    scale = tl.where(scale > 0, scale, 1.0)
    qlo = tl.clamp(lo / scale, -448.0, 448.0).to(tl.float8e4nv)
    qhi = tl.clamp(hi / scale, -448.0, 448.0).to(tl.float8e4nv)
    blo = qlo.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
    bhi = qhi.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
    dq = _pack(qlo.to(tl.float32) * scale, qhi.to(tl.float32) * scale, BF16)
    tl.store(out_ptr + gme, dq, mask=mm)
    a, b = tl.split(tl.reshape(blo | (bhi << 8), [QW, 2]))
    qword = a | (b << 16)

    # phase 2: all-gather in fp8
    j = tl.arange(0, QW)
    qloc2 = g * 0 + (pid * QW + j)[None, :]
    qmine = (g != RANK) & ((RANK * PART + pid * BLK + 2 * j)[None, :] < nw)
    dq2 = tl.multiple_of((bases + (QBASE + c * QBUF + RANK * QSLOT) * 4).to(tl.pointer_type(tl.int32)),
                         [16, 1])
    tl.store(dq2 + qloc2, g * 0 + qword[None, :], mask=qmine, cache_modifier=".wt")
    # the scale word: lane j == 0 of row g (1-D [NGPU] vectors deadlocked)
    zq = qloc2 * 0
    sdst = (bases + (QBASE + c * QBUF + RANK * QSLOT + QD + pid) * 4).to(tl.pointer_type(tl.int32))
    tl.store(sdst + zq, zq + scale.to(tl.int32, bitcast=True),
             mask=(g != RANK) & (j[None, :] == 0) & (RANK * PART + pid * BLK < nw),
             cache_modifier=".wt")

    qsrc = me + QBASE + c * QBUF + g * QSLOT + qloc2
    ssrc = me + QBASE + c * QBUF + g * QSLOT + QD + pid + zq
    qpeer = (g != RANK) & ((g * PART + pid * BLK + 2 * j[None, :]) < nw)
    speer = (g != RANK) & (g * PART + pid * BLK < nw) & (j[None, :] >= 0)
    qv = tl.load(qsrc, cache_modifier=".cv")
    while tl.sum(tl.sum(((qv == -1) & qpeer).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()
        qv = tl.load(qsrc, cache_modifier=".cv")
    # the scale in a loop of its own: a combined exit test let the scale
    # through as the sentinel (NaN tiles, 2026-11-16)
    sv = tl.load(ssrc, cache_modifier=".cv")
    while tl.sum(tl.sum(((sv == -1) & speer).to(tl.int32), axis=1), axis=0) > 0:
        tl.debug_barrier()
        sv = tl.load(ssrc, cache_modifier=".cv")
    # load once more after the polls: the values carried out of the loops
    # were stale -- whole NaN tiles although the loops had seen the data and
    # the scale arrive (the decode after the loop reshapes them; 2026-11-16)
    qv = tl.load(qsrc, cache_modifier=".cv")
    sv = tl.load(ssrc, cache_modifier=".cv")
    p16 = tl.reshape(tl.join(qv & 0xFFFF, (qv >> 16) & 0xFFFF), [NGPU, BLK])
    sc = tl.reshape(tl.join(sv, sv), [NGPU, BLK]).to(tl.float32, bitcast=True)
    flo = (p16 & 0xFF).to(tl.int8).to(tl.float8e4nv, bitcast=True).to(tl.float32) * sc
    fhi = ((p16 >> 8) & 0xFF).to(tl.int8).to(tl.float8e4nv, bitcast=True).to(tl.float32) * sc
    tl.store(out_ptr + gpos2, _pack(flo, fhi, BF16), mask=peer)

    r = (c + 2) % 3
    tl.store(me + BASE + r * BUF + g * SLOT + loc2, tl.full([NGPU, BLK], -1, tl.int32),
             mask=(g != RANK), cache_modifier=".wt")
    tl.store(me + QBASE + r * QBUF + g * QSLOT + qloc2, tl.full([NGPU, QW], -1, tl.int32),
             mask=(g != RANK), cache_modifier=".wt")
    tl.store(me + QBASE + r * QBUF + g * QSLOT + QD + pid + zq, tl.full([NGPU, QW], -1, tl.int32),
             mask=(g != RANK) & (j[None, :] == 0), cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(cnt_ptr + pid, (c + 1) % 3)


class TritonAllReduce:
    """Sum of x over the ranks of the registered group. bf16 / fp16, numel a
    multiple of 8, 16-byte aligned, at most max_bytes. NGPU 2, 4 or 8.
    fp8=True allocates the inboxes of the lossy fp8 all-gather mode, used
    by ar(x, fp8=True) above one_shot_bytes."""

    def __init__(self, rank, nGPU, max_bytes=64 << 20, one_shot_bytes=64 << 10, fp8=False):
        assert nGPU in (2, 4, 8)
        self.rank, self.nGPU = rank, nGPU
        self.max_bytes, self.one_shot_bytes = max_bytes, one_shot_bytes
        max_words = triton.cdiv(max_bytes, 4)
        self.slot1 = triton.cdiv(min(one_shot_bytes, max_bytes) // 4, SLOT_ALIGN) * SLOT_ALIGN
        self.slot2 = triton.cdiv(triton.cdiv(max_words, nGPU), SLOT_ALIGN) * SLOT_ALIGN
        self.base2 = 3 * nGPU * self.slot1
        self.inbox_bytes = 3 * nGPU * (self.slot1 + 2 * self.slot2) * 4
        # fp8 mode: own phase-1 region and counters (rounds must not mix with
        # the bf16 two-shot), then fp8 data + one scale word per 64-word tile
        self.fp8 = fp8
        self.qslot = self.slot2 // 2 + self.slot2 // 64
        self.base_f1 = self.inbox_bytes // 4
        self.base_q = self.base_f1 + 3 * nGPU * self.slot2
        if fp8:
            self.inbox_bytes += 3 * nGPU * (self.slot2 + self.qslot) * 4
        self._buf = _load_ipc().UncachedIpcBuffer(rank, nGPU, self.inbox_bytes, 0xFF)
        dev = torch.device("cuda", rank)
        n_prog = max(self.slot1 // BLK1, self.slot2 // min(BLK2, 64))
        self.cnt = torch.zeros((3, n_prog), dtype=torch.int32, device=dev)
        self._bases = None

    def _dbg_kw(self):
        """TAR_MAXIT=k: give up a poll after k iterations and count the words
        still missing in self.dbg (phase 1, fp8 data, fp8 scales). Debug only."""
        k = int(os.environ.get("TAR_MAXIT", "0"))
        if not k:
            return {}
        if not hasattr(self, "dbg"):
            self.dbg = torch.zeros(3, dtype=torch.int32, device=self.cnt.device)
        return {"dbg_ptr": self.dbg, "MAXIT": k}

    def register(self, gloo_group):
        self._buf.exchange(gloo_group)
        b = [int(v) for v in self._buf.get_peer_bases_cpu()]
        self._bases = b + [b[self.rank]] * (MAX_GPUS - len(b))

    def __call__(self, x, out=None, fp8=False):
        assert x.dtype in (torch.bfloat16, torch.float16) and x.is_contiguous()
        n = x.numel()
        nbytes = n * x.element_size()
        assert n % 8 == 0 and x.data_ptr() % 16 == 0 and nbytes <= self.max_bytes
        out = torch.empty_like(x) if out is None else out
        if n == 0:
            return out
        nw = n // 2
        xi, oi = x.view(torch.int32), out.view(torch.int32)
        bf16 = x.dtype == torch.bfloat16
        with _no_buffer_ops():
            if fp8 and nbytes > self.one_shot_bytes:
                assert self.fp8, "construct with fp8=True"
                part = triton.cdiv(triton.cdiv(nw, self.nGPU), 64) * 64
                _ar_two_shot_fp8[(part // 64,)](
                    xi, oi, self.cnt[2], nw, *self._bases, RANK=self.rank, NGPU=self.nGPU,
                    SLOT=self.slot2, QSLOT=self.qslot, BF16=bf16, BASE=self.base_f1,
                    QBASE=self.base_q, num_warps=self.nGPU, **self._dbg_kw())
            elif nbytes <= self.one_shot_bytes:
                _ar_one_shot[(triton.cdiv(nw, BLK1),)](
                    xi, oi, self.cnt[0], nw, *self._bases, RANK=self.rank, NGPU=self.nGPU,
                    SLOT=self.slot1, BLK=BLK1, BF16=bf16, num_warps=self.nGPU)
            else:
                blk2 = BLK2
                part = triton.cdiv(triton.cdiv(nw, self.nGPU), blk2) * blk2
                _ar_two_shot[(part // blk2,)](
                    xi, oi, self.cnt[1], nw, *self._bases, RANK=self.rank, NGPU=self.nGPU,
                    SLOT=self.slot2, BLK=blk2, BF16=bf16, BASE=self.base2,
                    num_warps=self.nGPU)
        return out
