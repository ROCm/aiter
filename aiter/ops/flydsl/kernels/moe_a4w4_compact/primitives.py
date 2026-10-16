# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Thin hardware helpers over FlyDSL rocdl intrinsics (gfx950).

Everything here is plumbing: buffer descriptors, raw buffer loads/stores,
LDS pointers and the scaled fp4 MFMA. No MoE logic lives in this file.
"""

import contextlib
import functools
import hashlib
import inspect
import pathlib
import threading
import warnings

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _mrocdl
from flydsl.expr import rocdl
from flydsl.expr.typing import T

_CDNA_BUF_FLAGS = (7 << 12) | (4 << 15)
_AUX_IS_ATTR = (
    inspect.signature(_mrocdl.RawPtrBufferLoadOp).parameters["aux"].kind
    is inspect.Parameter.KEYWORD_ONLY
)

NT = 2  # non-temporal cache modifier
# FlyDSL's cache keys on a kernel's own source and closure only; kernel names carry this so
# edits to shared helpers (this file, gemm/prologue/combine) invalidate cached binaries.
SRC_HASH = hashlib.sha1(
    b"".join(
        (pathlib.Path(__file__).parent / f).read_bytes()
        for f in ("primitives.py", "gemm.py", "prologue.py", "combine.py")
    )
).hexdigest()[:8]
# Sized descriptors never cover more than this, so any voffset >= REC_CAP is out of range.
REC_CAP = 0x7FFFFF00
# voffset of masked lanes: past REC_CAP even with a 63 B chunk offset added.
OOB = REC_CAP + 0xC0


def raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    while not isinstance(v, ir.Value) and hasattr(v, "_value"):
        v = v._value
    return v


def e8m0_even(amax):
    """Biased e8m0 exponent under the OCP "even" rule (AITER runtime / checkpoint quant):
    floor(log2(amax rounded at mantissa 1.75)) - 2, clamped to [0, 254]."""
    bits = fx.Float32(amax).bitcast(fx.Int32)
    f = (bits + fx.Int32(0x200000)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    # field 255 (inf / amax >= 1.75 * 2^127): AITER's log2 -> inf, clamped to 254
    e = (f == fx.Int32(255)).select(fx.Int32(256), f) - fx.Int32(2)
    return fx.max(fx.min(e, fx.Int32(254)), fx.Int32(0))


def quant_group(r_x, r_q, r_s, gid, even, ok=None, cm=0):
    """MXFP4-quantize one 32-element group: bf16 x[gid * 32 : +32] -> fp4 q[gid * 16 : +16] and
    e8m0 s[gid] (even: e8m0_even, else ceil_pow2(amax / 6)). ok: lanes with ok False touch
    nothing, which requires r_x, r_q and r_s to be sized (rsrc with num_bytes). Returns the scale byte (i32); r_q None: q is not stored, returns (scale, q vector).
    """
    from flydsl.expr import math as fmath

    def at(o):
        return o if ok is None else ok.select(o, fx.Int32(OOB))

    vals = []
    for c in range(4):
        v = fx.Vector(bload(r_x, at(gid * 64 + c * 16), T.vec(8, T.bf16))).to(
            fx.Float32
        )
        for i in range(8):
            vals.append(fx.Float32(v[i]))
    m = fmath.absf(vals[0])
    for i in range(1, 32):
        m = fx.max(m, fmath.absf(vals[i]))
    if even:
        bexp = e8m0_even(m)
    else:
        bits = (m * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
        bexp = ((bits + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23))) & fx.Int32(0xFF)
        bexp = fx.min(bexp, fx.Int32(254))
    qs = (bexp << fx.Int32(23)).bitcast(fx.Float32)
    words = []
    for wi in range(4):
        pk = raw(fx.Int32(0))
        for p in range(4):
            pk = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                pk,
                raw(vals[wi * 8 + 2 * p]),
                raw(vals[wi * 8 + 2 * p + 1]),
                raw(qs),
                p,
            )
        words.append(fx.Int32(pk))
    qv = fx.Vector.from_elements(words, fx.Int32)
    if r_q is not None:
        bstore(qv, r_q, at(gid * 16), cm=cm)
    bstore(bexp.to(fx.Int8), r_s, at(gid), cm=cm)
    return bexp if r_q is not None else (bexp, qv)


def e8m0_even_small(amax):
    """e8m0_even for 0 <= amax < 1.75 * 2^125 (bounded, finite: e.g. clamped SwiGLU output):
    the field-255 case and the upper clamp cannot occur; 4 VALU ops instead of 7."""
    bits = fx.Float32(amax).bitcast(fx.Int32)
    return fx.max(
        (bits + fx.Int32(0x200000)).shrui(fx.Int32(23)) - fx.Int32(2), fx.Int32(0)
    )


def fmed3(x, lo, hi):
    """v_med3_f32: equals max(min(x, hi), lo) for non-NaN x when lo <= hi."""
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32,
            "llvm.amdgcn.fmed3.f32",
            [raw(fx.Float32(x)), raw(fx.Float32(lo)), raw(fx.Float32(hi))],
            [],
            [],
        )
    )


def select_rsrc(cond, a, b):
    """Uniform select between two buffer resources."""
    return llvm.SelectOp(raw(cond), a, b).result


def rsrc(addr_i64, num_bytes=None):
    """Buffer resource over a raw device address. OOB loads return 0 when num_bytes is set."""
    base = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), raw(fx.Int64(addr_i64))).result
    if num_bytes is None:
        n = fx.Int64(0xFFFFFFFF)
    elif isinstance(num_bytes, int):
        n = fx.Int64(max(0, min(num_bytes, REC_CAP)))
    else:
        n = fx.Int64(num_bytes)
        big = n > fx.Int64(REC_CAP)
        n = big.select(fx.Int64(REC_CAP), n)
    return _mrocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        base,
        raw(fx.Int16(0)),
        raw(n),
        raw(fx.Int32(_CDNA_BUF_FLAGS)),
    ).result


def _aux(cm):
    if _AUX_IS_ATTR:
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cm)
    return raw(fx.Int32(cm))


def bload(rs, byte_off, ty, soff=0, cm=0):
    """Raw buffer load at a byte offset. `ty` is an MLIR type (e.g. T.i32x4)."""
    return _mrocdl.RawPtrBufferLoadOp(
        ty, rs, raw(fx.Int32(byte_off)), raw(fx.Int32(soff)), aux=_aux(cm)
    ).result


def bstore(val, rs, byte_off, soff=0, cm=0):
    _mrocdl.RawPtrBufferStoreOp(
        raw(val), rs, raw(fx.Int32(byte_off)), raw(fx.Int32(soff)), aux=_aux(cm)
    )


def lds_llvm_ptr(base_i32, byte_off):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared)
    return fx.to_llvm_ptr(
        fx.inttoptr(ptr_ty, fx.Int64(fx.Int32(base_i32) + fx.Int32(byte_off)))
    )


def lds_load(base_i32, byte_off, res_ty, elem_ty=None, align=16):
    return llvm.LoadOp(res_ty, lds_llvm_ptr(base_i32, byte_off), alignment=align).result


def lds_load_u8(base_i32, byte_off):
    """ds_read_u8 as an i32 (explicit zext: no v_and 0xff after the load)."""
    v = llvm.LoadOp(T.i8, lds_llvm_ptr(base_i32, byte_off), alignment=1).result
    return fx.Int32(llvm.ZExtOp(T.i32, v).result)


def lds_store(val, base_i32, byte_off, elem_ty=None, align=16):
    llvm.StoreOp(raw(val), lds_llvm_ptr(base_i32, byte_off), alignment=align)


def dma_async(rs, lds_base_i32, lds_off, voff, soff=0, nbytes=16, cm=0):
    """Async global->LDS DMA (completion tracked by rocdl.asyncmark/wait_asyncmark).

    lds_off must be wave-uniform; lane i lands at lds_base + lds_off + i*nbytes.
    """
    lptr = lds_llvm_ptr(
        lds_base_i32, fx.Int32(rocdl.readfirstlane(T.i32, raw(fx.Int32(lds_off))))
    )
    _mrocdl.raw_ptr_buffer_load_async_lds(
        rs,
        lptr,
        raw(fx.Int32(nbytes)),
        raw(fx.Int32(voff)),
        raw(fx.Int32(soff)),
        raw(fx.Int32(0)),
        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cm),
    )


def mfma_fp4(acc, a, b, sa, sb, opsel_a=0, opsel_b=0):
    """v_mfma_scale_f32_16x16x128_f8f6f4 with fp4 (e2m1) A and B and e8m0 scales.

    a, b: i32x4 per lane (32 fp4 values). sa, sb: i32 holding 4 e8m0 bytes;
    opsel picks the byte. acc: f32x4.
    """
    return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a, b, acc, 4, 4, opsel_a, raw(fx.Int32(sa)), opsel_b, raw(fx.Int32(sb))],
    )


def mfma_fp4_agpr(acc, a, b, sa, sb, opsel_b=0, opsel_a=0):
    """mfma_fp4 as inline asm with the accumulator pinned to AGPRs (tied in/out) and A/B/scales
    in VGPRs; acc=None: srcC is the inline constant 0 (first K step, no zero-init copies).
    The hazard recognizer does not see inside the asm: finish with mfma_drain(accs)."""
    sel = f"op_sel:[{opsel_a & 1},{opsel_b & 1},0] op_sel_hi:[{opsel_a >> 1},{opsel_b >> 1},0]"
    ops = [raw(a), raw(b), raw(fx.Int32(sa)), raw(fx.Int32(sb))]
    if acc is None:
        return llvm.InlineAsmOp(
            T.f32x4,
            ops,
            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 {sel} cbsz:4 blgp:4",
            "=&a,v,v,v,v",
            has_side_effects=False,
        ).result
    return llvm.InlineAsmOp(
        T.f32x4,
        ops + [raw(acc)],
        f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 {sel} cbsz:4 blgp:4",
        "=a,v,v,v,v,0",
        has_side_effects=False,
    ).result


def mfma_drain(accs):
    """Wait states covering an XDL MFMA result read by VALU / v_accvgpr_read (>= 19), then an
    empty side-effecting asm re-defining each accumulator, so the compiler's AGPR reads of the
    results are ordered after the wait states. Returns the fenced accumulators."""
    for _ in range(3):
        rocdl.s_nop(7)
    return [
        llvm.InlineAsmOp(
            T.f32x4, [raw(v)], "; acc fence $0", "=a,0", has_side_effects=True
        ).result
        for v in accs
    ]


def s_opaque(x, dep):
    """x (uniform i32), made to look dependent on `dep`: values derived from it are not
    hoisted out of a loop over `dep` (and kept live across it) by LICM."""
    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [raw(fx.Int32(x)), raw(fx.Int32(dep))],
            "; opaque $0 $2",
            "=s,0,s",
            has_side_effects=False,
        ).result
    )


def v_opaque(x, dep):
    """Per-lane s_opaque."""
    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [raw(fx.Int32(x)), raw(fx.Int32(dep))],
            "; opaque $0 $2",
            "=v,0,s",
            has_side_effects=False,
        ).result
    )


def lane_id():
    """Lane id recomputed in place (opaque mbcnt): not CSE'd with, or kept live from, the
    thread id of an earlier phase."""
    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [],
            "v_mbcnt_lo_u32_b32 $0, -1, 0\n\tv_mbcnt_hi_u32_b32 $0, -1, $0",
            "=v",
            has_side_effects=False,
        ).result
    )


def dpp_i32(src, ctrl, row_mask=0xF, bank_mask=0xF, bound_ctrl=True):
    """llvm.amdgcn.update.dpp.i32 with old = src (lanes outside the pattern keep src)."""
    v = raw(fx.Int32(src))
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.update.dpp.i32",
            [
                v,
                v,
                raw(fx.Int32(ctrl)),
                raw(fx.Int32(row_mask)),
                raw(fx.Int32(bank_mask)),
                raw(fx.Boolean(bound_ctrl)),
            ],
            [],
            [],
        )
    )


def pair_cols_f32(a, b, odd):
    """Lane c of an n16 tile pair holds (a, b) = columns (c, 16 + c). Returns (lo, hi) =
    columns (c, c + 1) on even lanes and (15 + c, 16 + c) on odd lanes: adjacent natural
    columns, via one lane-pair exchange (DPP quad_perm [1, 0, 3, 2])."""
    x = odd.select(fx.Float32(a), fx.Float32(b))
    y = dpp_i32(x.bitcast(fx.Int32), 0xB1).bitcast(fx.Float32)
    return odd.select(y, fx.Float32(a)), odd.select(fx.Float32(b), y)


def pair_cols_bf16x2(p, sel):
    """pair_cols_f32 on a packed bf16x2 (lo = column c, hi = 16 + c): DPP + v_perm_b32.
    sel = pair_sel(odd)."""
    p = fx.Int32(p)
    q = dpp_i32(p, 0xB1)
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32, "llvm.amdgcn.perm", [raw(q), raw(p), raw(fx.Int32(sel))], [], []
        )
    )


def pair_sel(odd):
    """v_perm_b32 selector for pair_cols_bf16x2: even (p.lo, q.lo), odd (q.hi, p.hi)."""
    return odd.select(fx.Int32(0x03020706), fx.Int32(0x05040100))


def pair_col(c):
    """First natural column (within a 32-column tile pair) of lane c's pair_cols result."""
    return (c & fx.Int32(14)) + ((c & fx.Int32(1)) << fx.Int32(4))


def row16_max_nonneg_f32(x):
    """Max over each 16-lane DPP row for non-negative floats (int order == float order).

    quad_perm xor1, quad_perm xor2, row_half_mirror, row_mirror: after each step every
    lane holds the max of the lanes combined so far, so the mirrors act like xor4/xor8.
    """
    v = fx.Float32(x).bitcast(fx.Int32)
    for ctrl in (0xB1, 0x4E, 0x141, 0x140):
        o = dpp_i32(v, ctrl)
        v = (fx.Uint32(o) > fx.Uint32(v)).select(o, v)
    return v.bitcast(fx.Float32)


def row16_max_nonneg_f32_multi(xs):
    """row16_max_nonneg_f32 over several independent values, one DPP level at a time across
    all of them, so each DPP's VALU-write hazard is covered by the other chains' work.
    """
    vs = [fx.Float32(x).bitcast(fx.Int32) for x in xs]
    for ctrl in (0xB1, 0x4E, 0x141, 0x140):
        os_ = [dpp_i32(v, ctrl) for v in vs]
        vs = [(fx.Uint32(o) > fx.Uint32(v)).select(o, v) for o, v in zip(os_, vs)]
    return [v.bitcast(fx.Float32) for v in vs]


def _permlane_swap(name, v):
    """v_permlane{16,32}_swap_b32 with both operands = v: returns the (vdst, vsrc) results."""
    v = raw(fx.Int32(v))
    st = llvm.StructType.get_literal([T.i32, T.i32])
    r = llvm.call_intrinsic(
        st, name, [v, v, raw(fx.Boolean(False)), raw(fx.Boolean(False))], [], []
    )
    return (
        fx.Int32(llvm.extractvalue(T.i32, r, [0])),
        fx.Int32(llvm.extractvalue(T.i32, r, [1])),
    )


def rows4_max_nonneg_f32_multi(xs):
    """Max over the 4 lanes l, l^16, l^32, l^48 (the four 16-lane rows) for non-negative floats,
    several independent values level by level: permlane32_swap then permlane16_swap."""
    vs = [fx.Float32(x).bitcast(fx.Int32) for x in xs]
    for name in ("llvm.amdgcn.permlane32.swap", "llvm.amdgcn.permlane16.swap"):
        ps = [_permlane_swap(name, v) for v in vs]
        vs = [(fx.Uint32(a) > fx.Uint32(b)).select(a, b) for a, b in ps]
    return [v.bitcast(fx.Float32) for v in vs]


def fast_rcp(x):
    return fx.Float32(rocdl.rcp(T.f32, raw(fx.Float32(x))))


def _set_kernel_fn_attrs(module, attrs):
    """Add LLVM function attributes (llvm.func passthrough) to every kernel llvm.func."""
    from flydsl._mlir import ir as _ir

    with module.context:
        items = [
            _ir.ArrayAttr.get([_ir.StringAttr.get(k), _ir.StringAttr.get(str(v))])
            for k, v in attrs.items()
        ]
        for top in module.body.operations:
            if top.operation.name != "gpu.module":
                continue
            for f in top.regions[0].blocks[0].operations:
                if f.operation.name == "llvm.func" and "rocdl.kernel" in f.attributes:
                    old = (
                        list(f.attributes["passthrough"])
                        if "passthrough" in f.attributes
                        else []
                    )
                    f.attributes["passthrough"] = _ir.ArrayAttr.get(old + items)


_FN_ATTR_HOOK_LOCK = threading.Lock()
_FN_ATTR_HOOK_DEPTH = [0]
_FN_ATTR_HOOK_SAVED = []


def _fn_attr_hook_supported(jf):
    run = getattr(jf, "_run_pipeline", None)
    pm = getattr(jf, "PassManager", None)
    if run is None or pm is None or not callable(getattr(pm, "parse", None)):
        return False
    try:
        params = list(inspect.signature(run).parameters.values())
    except (TypeError, ValueError):
        return False
    kinds = [(p.name, p.kind) for p in params]
    P = inspect.Parameter
    return kinds == [
        ("module", P.POSITIONAL_OR_KEYWORD),
        ("fragments", P.POSITIONAL_OR_KEYWORD),
        ("verifier", P.KEYWORD_ONLY),
        ("print_after_all", P.KEYWORD_ONLY),
    ]


def _install_fn_attr_hook(jf):
    from flydsl.compiler.kernel_function import CompilationContext

    def _attrs():
        return (CompilationContext.get_compile_hints() or {}).get("fn_attrs")

    orig_run_pipeline = jf._run_pipeline
    OrigPM = jf.PassManager

    def run_pipeline(module, fragments, *, verifier, print_after_all):
        attrs = _attrs()
        if attrs and len(fragments) >= 2:
            orig_run_pipeline(
                module,
                fragments[:-1],
                verifier=verifier,
                print_after_all=print_after_all,
            )
            _set_kernel_fn_attrs(module, attrs)
            fragments = fragments[-1:]
        return orig_run_pipeline(
            module, fragments, verifier=verifier, print_after_all=print_after_all
        )

    class _ModView:
        def __init__(self, op):
            self.context = op.context
            self.body = op.regions[0].blocks[0]

    class _PMMeta(type):
        def __getattr__(cls, n):
            return getattr(OrigPM, n)

        def __instancecheck__(cls, inst):
            return isinstance(inst, OrigPM) or type.__instancecheck__(cls, inst)

    class PM(metaclass=_PMMeta):
        """Per-fragment path (FLYDSL_DUMP_IR): attach attrs before the binary fragment."""

        def __init__(self, pm, text):
            self._pm, self._text = pm, text

        @classmethod
        def parse(cls, text, *a, **k):
            return cls(OrigPM.parse(text, *a, **k), text)

        def run(self, op, *a, **k):
            attrs = _attrs()
            if attrs and "gpu-module-to-binary" in self._text:
                _set_kernel_fn_attrs(_ModView(op), attrs)
            return self._pm.run(op, *a, **k)

        def __getattr__(self, n):
            return getattr(self._pm, n)

    jf._run_pipeline = run_pipeline
    jf.PassManager = PM
    return orig_run_pipeline, OrigPM


@contextlib.contextmanager
def fn_attr_hook(launch):
    """Honour compile_hints["fn_attrs"] (LLVM function attributes, e.g.
    "amdgpu-agpr-alloc") for compiles inside this block.

    MLIR's gpu->rocdl lowering drops unknown function attributes and FlyDSL has no
    public hook for llvm.func passthrough, so FlyDSL's private pipeline runner is
    patched for the duration of the block only. Without fn_attrs it is a no-op; if
    the private API is not as expected the attributes are skipped with a warning.
    """
    if not (getattr(launch, "compile_hints", None) or {}).get("fn_attrs"):
        yield
        return
    from flydsl.compiler import jit_function as jf

    with _FN_ATTR_HOOK_LOCK:
        if _FN_ATTR_HOOK_DEPTH[0] == 0:
            if not _fn_attr_hook_supported(jf):
                _warn_fn_attr_hook_once()
                installed = False
            else:
                _FN_ATTR_HOOK_SAVED[:] = _install_fn_attr_hook(jf)
                installed = True
        else:
            installed = True
        if installed:
            _FN_ATTR_HOOK_DEPTH[0] += 1
    try:
        yield
    finally:
        if installed:
            with _FN_ATTR_HOOK_LOCK:
                _FN_ATTR_HOOK_DEPTH[0] -= 1
                if _FN_ATTR_HOOK_DEPTH[0] == 0:
                    jf._run_pipeline, jf.PassManager = _FN_ATTR_HOOK_SAVED
                    _FN_ATTR_HOOK_SAVED.clear()


@functools.lru_cache(maxsize=1)
def _warn_fn_attr_hook_once():
    warnings.warn(
        "moe_a4w4_compact: flydsl.compiler.jit_function._run_pipeline/PassManager "
        "not as expected; compiling without compile_hints['fn_attrs'] "
        "(kernels stay correct, possibly slower).",
        RuntimeWarning,
        stacklevel=3,
    )


def exp2_raw(x):
    """v_exp_f32 without LLVM's denormal-range fixup (valid for inputs well inside range)."""
    return fx.Float32(
        llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2", [raw(fx.Float32(x))], [], [])
    )


def gptr(addr_i64):
    """Global (addrspace 1) LLVM pointer from a 64-bit address (per-lane OK)."""
    return llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), raw(fx.Int64(addr_i64))
    ).result


def gload(addr_i64, res_ty, align=16, nt=False):
    if nt:
        return llvm.LoadOp(
            res_ty, gptr(addr_i64), alignment=align, nontemporal=True
        ).result
    return llvm.LoadOp(res_ty, gptr(addr_i64), alignment=align).result
