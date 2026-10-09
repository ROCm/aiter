# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""K0: routing plan + activation quantization, both FlyDSL, no host sync.

plan    : parallel counting sort of topk_ids into the compact expert-sorted row
          array (exactly T*k rows), plus BM tile lists and the inverse map used by
          the combine. Row order inside an expert is arbitrary; results are not
          affected because every row is computed independently and the combine
          sums a token's slots in fixed slot order.
quant   : x bf16 [T, H] -> MX fp4 [T, H/2] + e8m0 [T, H/32], one thread per 32-group,
          bit-exact with mx.quant under either scale rule.
scale_t : the activation scales in K-step-major compact-row order (stage-1 AST).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from . import primitives


def _lds_atomic_add(base_i32, byte_off, val):
    ptr = primitives.lds_llvm_ptr(base_i32, byte_off)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        primitives.raw(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
    ).result


def _clamp_e(e, E):
    # Out-of-range expert ids must not index the per-expert LDS counters.
    return fx.max(fx.min(e, fx.Int32(E - 1)), fx.Int32(0))


CHUNK = 4096  # routing entries per CTA in the parallel plan


def spec_of(b):
    """Tile-list spec: int BM (all rows) or (BM, kind): kind -2 -> routed experts only,
    -3 -> the shared expert (E-1) only (the fused-combine launches)."""
    return (b, 0) if isinstance(b, int) else (int(b[0]), int(b[1]))


def spec_tag(bms):
    names = {0: "", -2: "ro", -3: "sh"}
    return "_".join(f"{bm}{names[p]}" for bm, p in map(spec_of, bms))


def spec_max_tiles(b, R, E):
    bm, _ = spec_of(b)
    return (R + bm - 1) // bm + E


@functools.cache
def build_plan_par(
    E: int,
    k: int,
    bms: tuple,
    shared_last: bool = False,
    qH: int = 0,
    qeven: bool = False,
):
    """Parallel plan in 3 launches (hist -> prefix/tiles -> scatter).

    hist   : CTA c counts its CHUNK of topk_ids in LDS, then one global atomic per
             expert returns the CTA's base inside that expert (cbase[c, e]).
    prefix : one CTA turns gcount into expert offsets and tile lists; re-zeroes gcount.
    qH > 0: the activation quant (build_quant's ops, hidden size qH) runs as extra CTAs of the
    hist launch, so it needs no launch of its own and is ordered before stage 1 by the plan.
    scatter: CTA c re-walks its chunk; row = offs[e] + cbase[c, e] + local position.
    """
    NBM = len(bms)
    TH = 1024
    assert E <= 256, "k_prefix runs one thread per expert in a 256-thread CTA"

    @fx.struct
    class HistStorage:
        raw: fx.Array[fx.Uint8, ((2 * E * 4 + 15) // 16) * 16, 16]

    kname = (
        f"moe_a4w4_compact_plan_hist_e{E}"
        + (f"_k{k}sl" if shared_last else "")
        + (f"_q{qH}{'se' if qeven else ''}" if qH else "")
        + f"_{primitives.SRC_HASH}"
    )

    @flyc.kernel(name=kname, known_block_size=[TH, 1, 1])
    def k_hist(
        ids_ptr: fx.Int64,
        gcount_ptr: fx.Int64,
        cbase_ptr: fx.Int64,
        n_rows: fx.Int32,
        x_ptr: fx.Int64,
        q_ptr: fx.Int64,
        s_ptr: fx.Int64,
        n_groups: fx.Int32,
    ):
        if const_expr(kname == ""):  # name (incl. source hash) in the JIT cache key
            pass
        tid = fx.Int32(gpu.thread_id("x"))
        c = fx.Int32(gpu.block_id("x"))
        if const_expr(qH > 0):
            n_hist = (n_rows + (CHUNK - 1)) // CHUNK
            if c >= n_hist:
                gid = (c - n_hist) * TH + tid
                primitives.quant_group(
                    primitives.rsrc(x_ptr, fx.Int64(n_groups) * fx.Int64(64)),
                    primitives.rsrc(q_ptr, fx.Int64(n_groups) * fx.Int64(16)),
                    primitives.rsrc(s_ptr, fx.Int64(n_groups)),
                    gid,
                    qeven,
                    ok=gid < n_groups,
                )
            else:
                _hist_body(ids_ptr, gcount_ptr, cbase_ptr, n_rows, tid, c)
        else:
            _hist_body(ids_ptr, gcount_ptr, cbase_ptr, n_rows, tid, c)

    def _hist_body(ids_ptr, gcount_ptr, cbase_ptr, n_rows, tid, c):
        base = fx.Int32(
            fx.ptrtoint(fx.SharedAllocator().allocate(HistStorage).peek().raw.ptr)
        )
        r_ids = primitives.rsrc(ids_ptr, fx.Int64(n_rows) * fx.Int64(4))
        r_cb = primitives.rsrc(cbase_ptr)
        if tid < fx.Int32(E):
            primitives.lds_store(fx.Int32(0), base, tid * 4, align=4)
        gpu.barrier()
        for j in range_constexpr(CHUNK // TH):
            i = c * CHUNK + j * TH + tid
            if i < n_rows:
                e = _clamp_e(fx.Int32(primitives.bload(r_ids, i * 4, T.i32)), E)
                _lds_atomic_add(base, e * 4, 1)
                if const_expr(shared_last):  # noqa: SIM102 (const_expr guard)
                    if e == fx.Int32(E - 1):
                        # gcount[E] counts tokens' repeated shared-expert slots.
                        slot = i % k
                        dup = fx.Int32(0)
                        for s in range_constexpr(k - 1):
                            o = _clamp_e(
                                fx.Int32(
                                    primitives.bload(
                                        r_ids, (i - slot + fx.Int32(s)) * 4, T.i32
                                    )
                                ),
                                E,
                            )
                            hit = (o == fx.Int32(E - 1)) & (fx.Int32(s) < slot)
                            dup = dup + hit.select(fx.Int32(1), fx.Int32(0))
                        if dup > fx.Int32(0):
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.add,
                                primitives.raw(_global_ptr(gcount_ptr, E * 4)),
                                primitives.raw(fx.Int32(1)),
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                            )
        gpu.barrier()
        if tid < fx.Int32(E):
            cnt = fx.Int32(primitives.lds_load(base, tid * 4, T.i32, align=4))
            prev = fx.Int32(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    primitives.raw(_global_ptr(gcount_ptr, tid * 4)),
                    primitives.raw(cnt),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                ).result
            )
            primitives.bstore(prev, r_cb, (c * E + tid) * 4)

    _hist_body = ASTRewriter.transform(_hist_body)

    specs = tuple(
        spec_of(b) for b in bms
    )  # tuple: part of the JIT cache key (lists are not)

    def _nt(c, bm, parent, e, fc):
        """Tiles of expert e (thread id) in a spec's list. fc (0/1): the fused combine is
        valid for this batch; otherwise E-1 joins the routed list and -3 is empty."""
        n = (c + (bm - 1)) // bm
        if parent == 0:
            return n
        if parent == -2:
            return (e < fx.Int32(E - 1)).select(n, n * (fx.Int32(1) - fc))
        return (e == fx.Int32(E - 1)).select(n * fc, fx.Int32(0))

    ptag = (
        f"moe_a4w4_compact_plan_prefix_e{E}_{spec_tag(bms)}"
        + (f"_k{k}sl" if shared_last else "")
        + f"_{primitives.SRC_HASH}"
    )
    PT = 256
    NV = NBM + 1  # scanned vectors: row counts + one tile count per spec
    SCAN_B = (
        NV * PT * 4
    )  # one scan buffer; two alternate so each step needs one barrier
    X_OFF = 2 * SCAN_B  # per spec: exclusive tile prefix per expert, total at [E]
    C_OFF = X_OFF + NBM * (E + 1) * 4
    O_OFF = C_OFF + E * 4

    @fx.struct
    class PrefixStorage:
        raw: fx.Array[fx.Uint8, ((O_OFF + E * 4 + 15) // 16) * 16, 16]

    @flyc.kernel(name=ptag, known_block_size=[PT, 1, 1])
    def k_prefix(
        gcount_ptr: fx.Int64,
        offs_ptr: fx.Int64,
        tiles_ptr: fx.Int64,
        ntiles_ptr: fx.Int64,
        max_tiles0: fx.Int32,
        n_rows: fx.Int32,
    ):
        if const_expr(ptag == ""):  # tag string in the JIT cache key
            pass
        tid = fx.Int32(gpu.thread_id("x"))
        lb = fx.Int32(
            fx.ptrtoint(fx.SharedAllocator().allocate(PrefixStorage).peek().raw.ptr)
        )
        r_gc = primitives.rsrc(gcount_ptr)
        r_offs = primitives.rsrc(offs_ptr)
        r_tiles = primitives.rsrc(tiles_ptr)
        r_nt = primitives.rsrc(ntiles_ptr)
        valid = tid < fx.Int32(E)
        zero = fx.Int32(0)
        cnt = valid.select(
            fx.Int32(primitives.bload(r_gc, valid.select(tid, zero) * 4, T.i32)), zero
        )
        fc = zero
        if const_expr(shared_last):
            # count[E-1] == T and no repeats <=> E-1 exactly once per token.
            n_sh = fx.Int32(primitives.bload(r_gc, (E - 1) * 4, T.i32))
            n_dup = fx.Int32(primitives.bload(r_gc, E * 4, T.i32))
            ok = (n_dup == zero) & (n_sh == n_rows // k)
            fc = ok.select(fx.Int32(1), zero)
        own = [cnt] + [
            valid.select(_nt(cnt, *specs[b], tid, fc), zero)
            for b in range_constexpr(NBM)
        ]
        inc = list(own)
        # Hillis-Steele inclusive scan over the PT threads (experts >= E contribute 0).
        for step in range_constexpr((PT - 1).bit_length()):
            d, buf = 1 << step, step % 2
            for i in range_constexpr(NV):
                primitives.lds_store(
                    inc[i], lb, buf * SCAN_B + (i * PT) * 4 + tid * 4, align=4
                )
            gpu.barrier()
            src = fx.max(tid - d, zero)
            for i in range_constexpr(NV):
                o = fx.Int32(
                    primitives.lds_load(
                        lb, buf * SCAN_B + (i * PT) * 4 + src * 4, T.i32, align=4
                    )
                )
                inc[i] = inc[i] + (tid >= fx.Int32(d)).select(o, zero)
        off = inc[0] - own[0]
        if valid:
            primitives.bstore(off, r_offs, tid * 4)
            primitives.lds_store(cnt, lb, C_OFF + tid * 4, align=4)
            primitives.lds_store(off, lb, O_OFF + tid * 4, align=4)
            for b in range_constexpr(NBM):
                primitives.lds_store(
                    inc[b + 1] - own[b + 1],
                    lb,
                    X_OFF + (b * (E + 1)) * 4 + tid * 4,
                    align=4,
                )
                if tid == fx.Int32(E - 1):
                    primitives.lds_store(
                        inc[b + 1], lb, X_OFF + (b * (E + 1) + E) * 4, align=4
                    )
                    primitives.bstore(inc[b + 1], r_nt, b * 4)
        gpu.barrier()
        # Tile entries (expert, first row, rows, 0), spread over all threads: tile j of spec b
        # belongs to the expert e with X[e] <= j < X[e + 1] (binary search; X is monotone).
        for b in range_constexpr(NBM):
            bm = specs[b][0]
            xb = X_OFF + (b * (E + 1)) * 4
            tb = max_tiles0 * b  # spec b's tile list starts at b * stride
            ntot = fx.Int32(primitives.lds_load(lb, xb + E * 4, T.i32, align=4))
            for j in range(tid, ntot, PT):
                jj = fx.Int32(j)
                lo, hi = zero, fx.Int32(E)
                for _ in range_constexpr(max(1, (E - 1).bit_length())):
                    mid = (lo + hi) // 2
                    le = (
                        fx.Int32(primitives.lds_load(lb, xb + mid * 4, T.i32, align=4))
                        <= jj
                    )
                    lo = le.select(mid, lo)
                    hi = le.select(hi, mid)
                e = lo
                mi = jj - fx.Int32(primitives.lds_load(lb, xb + e * 4, T.i32, align=4))
                ce = fx.Int32(primitives.lds_load(lb, C_OFF + e * 4, T.i32, align=4))
                oe = fx.Int32(primitives.lds_load(lb, O_OFF + e * 4, T.i32, align=4))
                nr = fx.min(ce - mi * bm, fx.Int32(bm))
                v = fx.Vector.from_elements([e, oe + mi * bm, nr, zero], fx.Int32)
                primitives.bstore(v, r_tiles, (tb + jj) * 16)
        gpu.barrier()
        # Every thread has read every count above; re-zero here (single CTA, same launch
        # that consumed them) so an aborted later launch cannot leave gcount dirty.
        if tid < fx.Int32(E):
            primitives.bstore(fx.Int32(0), r_gc, tid * 4)
        if const_expr(shared_last):  # noqa: SIM102 (const_expr guard)
            if tid == zero:
                primitives.bstore(zero, r_gc, E * 4)
                primitives.bstore(fc, r_offs, E * 4)

    @fx.struct
    class ScatStorage:
        raw: fx.Array[fx.Uint8, ((2 * E * 4 + 15) // 16) * 16, 16]

    sname = f"moe_a4w4_compact_plan_scatter_e{E}_k{k}{'_sl' if shared_last else ''}_{primitives.SRC_HASH}"

    @flyc.kernel(name=sname, known_block_size=[TH, 1, 1])
    def k_scatter(
        ids_ptr: fx.Int64,
        w_ptr: fx.Int64,
        offs_ptr: fx.Int64,
        cbase_ptr: fx.Int64,
        rtok_ptr: fx.Int64,
        rw_ptr: fx.Int64,
        inv_ptr: fx.Int64,
        gcount_ptr: fx.Int64,
        n_rows: fx.Int32,
    ):
        if const_expr(sname == ""):  # name (incl. source hash) in the JIT cache key
            pass
        tid = fx.Int32(gpu.thread_id("x"))
        c = fx.Int32(gpu.block_id("x"))
        base = fx.Int32(
            fx.ptrtoint(fx.SharedAllocator().allocate(ScatStorage).peek().raw.ptr)
        )
        r_ids = primitives.rsrc(ids_ptr, fx.Int64(n_rows) * fx.Int64(4))
        r_w = primitives.rsrc(w_ptr, fx.Int64(n_rows) * fx.Int64(4))
        r_offs = primitives.rsrc(offs_ptr)
        r_cb = primitives.rsrc(cbase_ptr)
        r_rt = primitives.rsrc(rtok_ptr, fx.Int64(n_rows) * fx.Int64(4))
        r_rw = primitives.rsrc(rw_ptr, fx.Int64(n_rows) * fx.Int64(4))
        r_inv = primitives.rsrc(inv_ptr, fx.Int64(n_rows) * fx.Int64(4))
        if tid < fx.Int32(E):
            start = fx.Int32(primitives.bload(r_offs, tid * 4, T.i32)) + fx.Int32(
                primitives.bload(r_cb, (c * E + tid) * 4, T.i32)
            )
            primitives.lds_store(start, base, tid * 4, align=4)
        sh = fx.Int32(E)
        if const_expr(shared_last):
            # Fused combine valid: E-1 (once per token) gets token-ordered rows.
            fc = fx.Int32(primitives.bload(r_offs, E * 4, T.i32))
            sh = (fc != fx.Int32(0)).select(fx.Int32(E - 1), fx.Int32(E))
        gpu.barrier()
        for j in range_constexpr(CHUNK // TH):
            i = c * CHUNK + j * TH + tid
            if i < n_rows:
                e = _clamp_e(fx.Int32(primitives.bload(r_ids, i * 4, T.i32)), E)
                if const_expr(shared_last):
                    row = fx.Int32(0)
                    if e == sh:
                        row = (
                            fx.Int32(primitives.bload(r_offs, (E - 1) * 4, T.i32))
                            + i // k
                        )
                    else:
                        row = fx.Int32(_lds_atomic_add(base, e * 4, 1))
                else:
                    row = fx.Int32(_lds_atomic_add(base, e * 4, 1))
                primitives.bstore(i // k, r_rt, row * 4)
                primitives.bstore(
                    fx.Float32(primitives.bload(r_w, i * 4, T.f32)), r_rw, row * 4
                )
                primitives.bstore(row, r_inv, i * 4)

    @flyc.jit
    def launch(
        ids_ptr: fx.Int64,
        w_ptr: fx.Int64,
        rtok_ptr: fx.Int64,
        rw_ptr: fx.Int64,
        inv_ptr: fx.Int64,
        tiles_ptr: fx.Int64,
        ntiles_ptr: fx.Int64,
        gcount_ptr: fx.Int64,
        offs_ptr: fx.Int64,
        cbase_ptr: fx.Int64,
        n_rows: fx.Int32,
        n_cta: fx.Int32,
        max_tiles0: fx.Int32,
        x_ptr: fx.Int64,
        q_ptr: fx.Int64,
        s_ptr: fx.Int64,
        n_groups: fx.Int32,
        n_qcta: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        k_hist(
            ids_ptr, gcount_ptr, cbase_ptr, n_rows, x_ptr, q_ptr, s_ptr, n_groups
        ).launch(grid=(n_cta + n_qcta, 1, 1), block=(TH, 1, 1), stream=stream)
        k_prefix(
            gcount_ptr, offs_ptr, tiles_ptr, ntiles_ptr, max_tiles0, n_rows
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
        k_scatter(
            ids_ptr,
            w_ptr,
            offs_ptr,
            cbase_ptr,
            rtok_ptr,
            rw_ptr,
            inv_ptr,
            gcount_ptr,
            n_rows,
        ).launch(grid=(n_cta, 1, 1), block=(TH, 1, 1), stream=stream)

    return launch


def _global_ptr(addr_i64, byte_off):
    ptr_ty = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4)
    base = fx.inttoptr(
        fx.PointerType.get(T.i8, fx.AddressSpace.Global, 4), fx.Int64(addr_i64)
    )
    return fx.to_llvm_ptr(
        fx.recast_iter(ptr_ty, fx.add_offset(base, fx.Int32(byte_off)))
    )


@functools.cache
def build_scale_t(KG: int, threads: int = 256):
    """a_s [T, KG] bytes (token-major) -> a_s_t [KG/4, R, 4] (K-step-major, compact rows).

    With this layout a tile's A scales for one 128-K step are BM*4 contiguous bytes,
    so the GEMM fetches them with one 16 B/lane DMA instead of a 4 B-per-row gather.
    """
    assert KG % 16 == 0, "a token's scale row is read as 16 B chunks"

    kname = f"moe_a4w4_compact_scale_t_kg{KG}_{primitives.SRC_HASH}"

    @flyc.kernel(name=kname, known_block_size=[threads, 1, 1])
    def kern(
        as_ptr: fx.Int64,
        rtok_ptr: fx.Int64,
        ast_ptr: fx.Int64,
        n_rows: fx.Int32,
        n_tok: fx.Int32,
    ):
        if const_expr(kname == ""):  # name (incl. source hash) in the JIT cache key
            pass
        # One thread per compact row: the token's whole scale row in 16 B loads, then one
        # 4 B store per K step (consecutive rows -> coalesced across the wave).
        row = fx.Int32(gpu.block_id("x")) * threads + fx.Int32(gpu.thread_id("x"))
        r_tok = primitives.rsrc(rtok_ptr, fx.Int64(n_rows) * fx.Int64(4))
        r_as = primitives.rsrc(as_ptr, fx.Int64(n_tok) * fx.Int64(KG))
        r_ast = primitives.rsrc(ast_ptr, fx.Int64(n_rows) * fx.Int64(KG))
        if row < n_rows:
            t = fx.Int32(primitives.bload(r_tok, row * 4, T.i32))
            vs = [
                fx.Vector(primitives.bload(r_as, t * KG + q * 16, T.i32x4))
                for q in range_constexpr(KG // 16)
            ]
            for q in range_constexpr(KG // 16):
                for j in range_constexpr(4):
                    primitives.bstore(
                        fx.Int32(vs[q][j]), r_ast, ((q * 4 + j) * n_rows + row) * 4
                    )

    @flyc.jit
    def launch(
        as_ptr: fx.Int64,
        rtok_ptr: fx.Int64,
        ast_ptr: fx.Int64,
        n_rows: fx.Int32,
        n_tok: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(as_ptr, rtok_ptr, ast_ptr, n_rows, n_tok).launch(
            grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


def run_scale_t(a_s, row_tok, a_s_t, stream=None):
    import torch

    n_tok, KG = a_s.shape
    R = row_tok.numel()
    grid = (R + 255) // 256
    stream = torch.cuda.current_stream() if stream is None else stream
    _run(
        ("st", KG),
        build_scale_t(KG),
        (a_s.data_ptr(), row_tok.data_ptr(), a_s_t.data_ptr(), R, n_tok, grid, stream),
    )


@functools.cache
def build_quant(H: int, threads: int = 256, even: bool = False):
    name = (
        f"moe_a4w4_compact_quant_h{H}"
        + ("_se" if even else "")
        + f"_{primitives.SRC_HASH}"
    )

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kern(x_ptr: fx.Int64, q_ptr: fx.Int64, s_ptr: fx.Int64, n_groups: fx.Int32):
        if const_expr(name == ""):  # name in the JIT cache key
            pass
        gid = fx.Int32(gpu.block_id("x")) * threads + fx.Int32(gpu.thread_id("x"))
        r_x = primitives.rsrc(x_ptr, fx.Int64(n_groups) * fx.Int64(64))
        r_q = primitives.rsrc(q_ptr, fx.Int64(n_groups) * fx.Int64(16))
        r_s = primitives.rsrc(s_ptr, fx.Int64(n_groups))
        primitives.quant_group(r_x, r_q, r_s, gid, even)

    @flyc.jit
    def launch(
        x_ptr: fx.Int64,
        q_ptr: fx.Int64,
        s_ptr: fx.Int64,
        n_groups: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(x_ptr, q_ptr, s_ptr, n_groups).launch(
            grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


_cf = {}


def _run(key, launch, args):
    if key not in _cf:
        _cf[key] = flyc.compile(launch, *args)
    else:
        _cf[key](*args)


def run_quant(x, q, s, even=False, stream=None):
    import torch

    Tn, H = x.shape
    ng = Tn * (H // 32)
    assert ng * 64 < 2**31, "quant uses 32-bit byte offsets"
    grid = (ng + 255) // 256
    stream = torch.cuda.current_stream() if stream is None else stream
    _run(
        ("q", H, even),
        build_quant(H, 256, even),
        (x.data_ptr(), q.data_ptr(), s.data_ptr(), ng, grid, stream),
    )


def run_plan(
    ids_i32,
    w_f32,
    row_tok,
    row_w,
    inv,
    tiles,
    ntiles,
    E,
    k,
    bms,
    max_tiles0,
    scratch,
    shared_last=False,
    quant=None,
    stream=None,
):
    """scratch: (gcount [E+1] int32 zero-initialised once, offs [E+1], cbase [n_cta*E]).
    quant: (x, a_q, a_s, even) to run the activation quant inside the hist launch.
    shared_last: fused-combine plan; offs[E] = 1 when E-1 is exactly once per token
    (then E-1 rows are token-ordered and only in the -3 lists), else 0 (E-1 is routed).
    """
    import torch

    stream = torch.cuda.current_stream() if stream is None else stream
    n = ids_i32.numel()
    gcount, offs, cbase = scratch
    n_cta = (n + CHUNK - 1) // CHUNK
    if (
        cbase.numel() < n_cta * E
        or min(gcount.numel(), offs.numel()) < E + 1
        or max_tiles0 < max(spec_max_tiles(b, n, E) for b in bms)
        or tiles.numel() * tiles.element_size() < len(bms) * max_tiles0 * 16
    ):
        raise ValueError(f"plan buffers too small for {n} rows, E={E}, specs {bms}")
    if quant is not None:
        x, q, s, even = quant
        qH, ng = x.shape[1], x.numel() // 32
        assert ng * 64 < 2**31, "quant uses 32-bit byte offsets"
        qargs = (x.data_ptr(), q.data_ptr(), s.data_ptr(), ng, (ng + 1023) // 1024)
    else:
        qH, even, qargs = 0, False, (0, 0, 0, 0, 0)
    _run(
        ("pp", E, k, bms, shared_last, qH, even),
        build_plan_par(E, k, bms, shared_last, qH, even),
        (
            ids_i32.data_ptr(),
            w_f32.data_ptr(),
            row_tok.data_ptr(),
            row_w.data_ptr(),
            inv.data_ptr(),
            tiles.data_ptr(),
            ntiles.data_ptr(),
            gcount.data_ptr(),
            offs.data_ptr(),
            cbase.data_ptr(),
            n,
            n_cta,
            max_tiles0,
        )
        + qargs
        + (stream,),
    )


def plan_scratch(R, E, device):
    import torch

    n_cta = (R + CHUNK - 1) // CHUNK
    return (
        torch.zeros(E + 1, dtype=torch.int32, device=device),
        torch.zeros(E + 1, dtype=torch.int32, device=device),
        torch.empty(n_cta * E, dtype=torch.int32, device=device),
    )
