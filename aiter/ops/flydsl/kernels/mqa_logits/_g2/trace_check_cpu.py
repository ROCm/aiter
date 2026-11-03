# SPDX-License-Identifier: MIT
"""CPU *trace check* of the bf16 WMMA kernel body — executes the real source.

No GPU and no flydsl: a mock ``flydsl`` is installed, the kernel module is
loaded from its file, and the **actual** ``kernel`` function inside
``_build_wmma_kernel`` is re-compiled with every ``if <cond>:`` turned into an
execution-mask context (``with __qsa_mask(cond):``), which is what an ``scf.if``
on a lane-varying condition means on the GPU.  Every DSL value is a 32-lane
torch vector (one wave32), so one call simulates one wave; the grid is walked
exactly as the wrapper launches it (``wmma_grid``) with 4 waves per workgroup.

What it checks, for every case:

* every buffer **load** address of every lane (taken or not) is inside its
  plane and ``< 2**31`` (stricter than the HW, which skips masked lanes);
* every buffer **store** address of every lane (taken or not) is inside its
  plane: no sentinel / out-of-plane offsets exist anywhere in the body;
* every descriptor extent (``num_records``) equals the plane's true size;
* every i32 intermediate stays inside int32 (no silent wrap in addressing);
* ``fx.gemm`` (WMMA) only executes with a **full** 32-lane exec mask (WMMA
  needs EXEC = all ones), with the 16x16x32 bf16 -> f32 atom;
* every logit in ``[0, rows) x [0, columns)`` and every visible/row_starts entry
  is written exactly once, and the values match the public golden
  (``qsa_golden.reference_logits``) with ``-inf`` outside ``valid_page & c <
  visible`` — or, for page indices the golden cannot index, the contract.

The WMMA operand / accumulator layouts used by the mock ``fx.gemm`` are the
gfx1250 ones FlyDSL documents and tests (``tests/unit/test_gfx1250_mma_atom_
precision_check.py`` at v0.3.3.dev889)::

    A/B lane l, bf16 element v<16 -> row l%16, k = (v//8)*16 + (l//16)*8 + v%8
    C/D lane l, f32 element  v<8  -> m = (l//16)*8 + v,  n = l%16

    python trace_check_cpu.py        # r16, exit 0 on PASS
    QSA_BF16_K_LOAD=raw QSA_TRACE_ROWS_PER_BLOCK=32 python trace_check_cpu.py

The r32 mode checks the real two-M-tile shared-B path and mixed/partial fallback.
TDM itself is covered by the separate bitwise model and no-device ISA gate.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import math
import os
import sys
import textwrap
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.environ.get("QSA_GOLDEN_DIR", os.path.join(_HERE, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.abspath(_GOLDEN))
KERNEL_PATH = os.path.abspath(os.environ.get(
    "QSA_TRACE_KERNEL", os.path.join(_HERE, "..", "qsa_paged_mqa_logits_page4_gfx1250.py")))

import qsa_golden as G  # noqa: E402
import qsa_shapes as S  # noqa: E402

L = 32                       # wave32
I32_MIN, I32_MAX = -(2**31), 2**31 - 1


class TraceError(AssertionError):
    pass


# ---------------------------------------------------------------------------
# Mock numeric model: every value is a [L] torch tensor (int64 / float32 / bool)
# ---------------------------------------------------------------------------
class _T:
    """Mock scalar type (``fx.Int32`` ...): callable, hashable, has ir_type."""

    def __init__(self, name, kind, width):
        self.name, self.kind, self.width = name, kind, width
        self.ir_type = self

    def __call__(self, x):
        if isinstance(x, MV):
            if self.kind == "f" and x.kind != "f":
                raise TraceError(f"implicit int->float conversion via {self.name}")
            return MV(x.t, self) if self.kind == x.kind or self.kind == "f" else MV(x.t, self)
        if isinstance(x, bool):
            return MV(torch.full((L,), int(x), dtype=torch.int64), self)
        if self.kind == "f":
            return MV(torch.full((L,), float(x), dtype=torch.float32), self)
        return MV(torch.full((L,), int(x), dtype=torch.int64), self)

    def __repr__(self):
        return self.name


Int8 = _T("Int8", "i", 8)
Int16 = _T("Int16", "i", 16)
Int32 = _T("Int32", "i", 32)
Int64 = _T("Int64", "i", 64)
Float32 = _T("Float32", "f", 32)
BFloat16 = _T("BFloat16", "f", 16)
Boolean = _T("Boolean", "b", 1)


def _check_i32(t, what):
    if t.dtype == torch.int64 and t.numel():
        lo, hi = int(t.min()), int(t.max())
        if lo < I32_MIN or hi > I32_MAX:
            raise TraceError(f"int32 overflow in {what}: [{lo}, {hi}]")


class MV:
    """A wave-wide DSL value."""

    def __init__(self, t, ty):
        self.t, self.ty = t, ty
        if ty is Int32:
            _check_i32(t, "i32 value")

    kind = property(lambda self: self.ty.kind)

    def ir_value(self):
        return self

    @staticmethod
    def _lift(x, like):
        if isinstance(x, MV):
            return x
        if isinstance(x, bool):
            return MV(torch.full((L,), x, dtype=torch.bool), Boolean)
        if isinstance(x, float) or like.kind == "f":
            return Float32(float(x))
        return like.ty(int(x))

    def _bin(self, other, op, rev=False):
        o = MV._lift(other, self)
        a, b = (o, self) if rev else (self, o)
        if op in ("&", "|") and a.ty is Boolean and b.ty is Boolean:
            return MV(a.t & b.t if op == "&" else a.t | b.t, Boolean)
        if (a.kind == "f") != (b.kind == "f"):
            raise TraceError(f"mixed int/float operands for {op}")
        ty = Float32 if a.kind == "f" else (Int64 if Int64 in (a.ty, b.ty) else Int32)
        x, y = a.t, b.t
        if op == "+":
            r = x + y
        elif op == "-":
            r = x - y
        elif op == "*":
            r = x * y
        elif op == "//":
            if ty.kind == "f":
                raise TraceError("float floordiv")
            if bool((y == 0).any()):
                raise TraceError("division by zero")
            if bool((x < 0).any()) or bool((y < 0).any()):
                raise TraceError("signed division of a negative value (trunc vs floor)")
            r = x // y
        elif op == "%":
            if bool((x < 0).any()) or bool((y <= 0).any()):
                raise TraceError("signed remainder of a negative value")
            r = x % y
        elif op == "<<":
            r = x << y
        elif op == ">>":
            r = x >> y
        elif op == "&":
            r = x & y
        elif op == "|":
            r = x | y
        else:
            raise TraceError(op)
        if ty is Int32:
            _check_i32(r, f"'{op}'")
        return MV(r, ty)

    def _cmp(self, other, fn):
        o = MV._lift(other, self)
        return MV(fn(self.t, o.t), Boolean)

    __add__ = lambda s, o: s._bin(o, "+")
    __radd__ = lambda s, o: s._bin(o, "+", True)
    __sub__ = lambda s, o: s._bin(o, "-")
    __rsub__ = lambda s, o: s._bin(o, "-", True)
    __mul__ = lambda s, o: s._bin(o, "*")
    __rmul__ = lambda s, o: s._bin(o, "*", True)
    __floordiv__ = lambda s, o: s._bin(o, "//")
    __mod__ = lambda s, o: s._bin(o, "%")
    __lshift__ = lambda s, o: s._bin(o, "<<")
    __rshift__ = lambda s, o: s._bin(o, ">>")
    __and__ = lambda s, o: s._bin(o, "&")
    __rand__ = lambda s, o: s._bin(o, "&", True)
    __or__ = lambda s, o: s._bin(o, "|")
    __ror__ = lambda s, o: s._bin(o, "|", True)
    __lt__ = lambda s, o: s._cmp(o, torch.lt)
    __le__ = lambda s, o: s._cmp(o, torch.le)
    __gt__ = lambda s, o: s._cmp(o, torch.gt)
    __ge__ = lambda s, o: s._cmp(o, torch.ge)
    __eq__ = lambda s, o: s._cmp(o, torch.eq)
    __ne__ = lambda s, o: s._cmp(o, torch.ne)
    __hash__ = object.__hash__

    def __bool__(self):
        raise TraceError("a DSL value reached a Python truth test (if/and/or on a traced value)")

    def select(self, a, b):
        if self.ty is not Boolean:
            raise TraceError("select on a non-boolean")
        a = MV._lift(a, b if isinstance(b, MV) else self) if not isinstance(a, MV) else a
        b = MV._lift(b, a)
        if a.kind != b.kind:
            raise TraceError("select with mixed int/float branches")
        ty = a.ty if a.ty is b.ty else (Int64 if Int64 in (a.ty, b.ty) else a.ty)
        return MV(torch.where(self.t, a.t, b.t), ty)

    def maximumf(self, other):
        o = MV._lift(other, self)
        if self.kind != "f" or o.kind != "f":
            raise TraceError("maximumf on non-float")
        return MV(torch.maximum(self.t, o.t), Float32)


class MVec:
    """A wave-wide vector value: tensor [L, n]."""

    def __init__(self, t, ty):
        self.t, self.ty = t, ty

    def ir_value(self):
        return self

    def shuffle(self, other, mask):
        cat = torch.cat([self.t, other.t], dim=1)
        return MVec(cat[:, list(mask)], self.ty)

    def __getitem__(self, i):
        return MV(self.t[:, int(i)].clone(), self.ty)

    @classmethod
    def filled(cls, n, value, ty):
        dt = torch.float32 if ty.kind == "f" else torch.int64
        return cls(torch.full((L, n), value, dtype=dt), ty)


def _Vector(x, *a, **k):
    if isinstance(x, MVec):
        return x
    raise TraceError(f"Vector() of {type(x).__name__}")


_Vector.filled = MVec.filled
_Vector.from_elements = lambda els, dtype=None: MVec(torch.stack([e.t for e in els], 1), dtype)


class RTensor:
    """Register tensor (``make_rmem_tensor``)."""

    def __init__(self, n, ty):
        self.n, self.ty = n, ty
        self.v = None

    def store(self, vec):
        if vec.t.shape != (L, self.n):
            raise TraceError(f"rmem store of shape {tuple(vec.t.shape)} into {self.n}")
        self.v = vec.t.clone()

    def load(self):
        if self.v is None:
            raise TraceError("rmem load before store")
        return MVec(self.v.clone(), self.ty)


# ---------------------------------------------------------------------------
# Planes (buffer descriptors), loads, stores
# ---------------------------------------------------------------------------
class Plane:
    def __init__(self, name, u8, extent):
        self.name, self.u8, self.extent = name, u8, extent

    def check(self, offs, width, what):
        lo, hi = int(offs.min()), int(offs.max()) + width
        if lo < 0 or hi > self.extent:
            raise TraceError(f"{what} {self.name}: [{lo}, {hi}) outside extent {self.extent}")


class Sim:
    """Mutable simulation state shared with the mock modules."""

    planes: dict = {}
    names: dict = {}
    mask = [torch.ones(L, dtype=torch.bool)]
    tid = torch.arange(L)
    bid = (0, 0)
    writes: dict = {}
    n_loads = 0
    n_stores = 0
    n_gemm = 0


def _get_iter(t):
    return t


def _recast_iter(ty, t):
    return t


def _make_buffer_ptr(t, num_records_bytes=None):
    ext = num_records_bytes
    ext = int(ext.t[0]) if isinstance(ext, MV) else int(ext)
    if bool((num_records_bytes.t != num_records_bytes.t[0]).any()) if isinstance(num_records_bytes, MV) else False:
        raise TraceError("descriptor extent is not wave-uniform")
    name = Sim.names[id(t)]
    true_bytes = t.numel() * t.element_size()
    if ext > true_bytes:
        raise TraceError(f"descriptor extent {ext} > allocation {true_bytes} for {name}")
    if ext < true_bytes and not (name == "logits" or name == "kv" or name == "q"):
        pass
    p = Plane(name, t.contiguous().view(torch.uint8).reshape(-1), ext)
    Sim.planes[name] = p
    return p


def _raw_load(res_ty, plane, off, soff, aux=None):
    Sim.n_loads += 1
    offs = off.t
    if isinstance(res_ty, tuple):
        n, ety = res_ty
    else:
        n, ety = 1, res_ty
    nbytes = n * 4 if ety.width == 32 else n * ety.width // 8
    plane.check(offs, nbytes, "load")
    _check_i32(offs + nbytes, "load offset")
    if ety.width != 32:
        raise TraceError("only dword loads are mocked")
    idx = offs[:, None] + torch.arange(nbytes)[None, :]
    b = plane.u8[idx].to(torch.int64)
    words = b[:, 0::4] | (b[:, 1::4] << 8) | (b[:, 2::4] << 16) | (b[:, 3::4] << 24)
    words = torch.where(words >= 2**31, words - 2**32, words)
    if ety.kind == "f":
        f = words.to(torch.int32).view(torch.float32)
        return MVec(f, Float32) if isinstance(res_ty, tuple) else MV(f[:, 0].clone(), Float32)
    return MVec(words, Int32) if isinstance(res_ty, tuple) else MV(words[:, 0].clone(), Int32)


def _raw_store(value, plane, off, soff, aux=None):
    Sim.n_stores += 1
    offs = off.t
    plane.check(offs, 4, "store (all lanes, taken or not)")
    _check_i32(offs + 4, "store offset")
    m = Sim.mask[-1]
    if not bool(m.any()):
        return
    rec = Sim.writes.setdefault(plane.name, {})
    vals = value.t
    for lane in torch.nonzero(m).flatten().tolist():
        o = int(offs[lane])
        v = vals[lane].item()
        cnt, _ = rec.get(o, (0, None))
        rec[o] = (cnt + 1, v)


@contextlib.contextmanager
def _qsa_mask(cond):
    if isinstance(cond, MV):
        if cond.ty is not Boolean:
            raise TraceError("if on a non-boolean DSL value")
        m = Sim.mask[-1] & cond.t
    else:
        m = Sim.mask[-1] & torch.full((L,), bool(cond))
    Sim.mask.append(m)
    try:
        yield
    finally:
        Sim.mask.pop()


# ---------------------------------------------------------------------------
# WMMA mock (gfx1250 16x16x32 bf16 -> f32 layouts)
# ---------------------------------------------------------------------------
_LN = torch.arange(L)
_V16 = torch.arange(16)
_AB_ROW = (_LN[:, None] % 16).expand(L, 16)
_AB_K = (_V16[None, :] // 8) * 16 + (_LN[:, None] // 16) * 8 + _V16[None, :] % 8
_C_M = (_LN[:, None] // 16) * 8 + torch.arange(8)[None, :]
_C_N = (_LN[:, None] % 16).expand(L, 8)


def _bf16_of_words(w):
    w = w.to(torch.int64) & 0xFFFFFFFF
    lo = (w & 0xFFFF) << 16
    hi = ((w >> 16) & 0xFFFF) << 16
    both = torch.stack([lo, hi], dim=2).reshape(L, -1)
    both = torch.where(both >= 2**31, both - 2**32, both)
    return both.to(torch.int32).view(torch.float32)


class _WMMA:
    def __init__(self, m, n, k, ab, acc):
        if (m, n, k, ab, acc) != (16, 16, 32, BFloat16, Float32):
            raise TraceError(f"unexpected WMMA atom {(m, n, k, ab, acc)}")


def _make_mma_atom(op):
    if not isinstance(op, _WMMA):
        raise TraceError("mma atom is not the gfx1250 WMMA")
    return op


def _gemm(atom, d, a, b, c):
    m = Sim.mask[-1]
    if not bool(m.any()):
        return                      # branch not taken by this wave (uniform skip)
    if not bool(m.all()):
        raise TraceError("WMMA issued under a partial exec mask")
    Sim.n_gemm += 1
    if a.ty is not Int32 or b.ty is not Int32 or a.n != 8 or b.n != 8 or c.n != 8:
        raise TraceError("unexpected WMMA fragment shapes")
    av, bv = _bf16_of_words(a.v), _bf16_of_words(b.v)
    A = torch.zeros(16, 32)
    B = torch.zeros(16, 32)
    A[_AB_ROW, _AB_K] = av
    B[_AB_ROW, _AB_K] = bv
    C = torch.zeros(16, 16)
    C[_C_M, _C_N] = c.v
    D = A @ B.T + C
    d.v = D[_C_M, _C_N].clone()


# ---------------------------------------------------------------------------
# Mock module installation
# ---------------------------------------------------------------------------
def _install_mock():
    fx = types.ModuleType("flydsl.expr")
    for t in (Int8, Int16, Int32, Int64, Float32, BFloat16, Boolean):
        setattr(fx, t.name, t)
    fx.Uint8 = Int8
    fx.Uint32 = Int32
    fx.Float16 = BFloat16
    fx.Vector = _Vector
    fx.range_constexpr = range
    fx.get_iter = _get_iter
    fx.recast_iter = _recast_iter
    fx.make_rmem_tensor = lambda n, ty: RTensor(int(n), ty)
    fx.make_mma_atom = _make_mma_atom
    fx.gemm = _gemm

    class _Idx:
        @property
        def x(self):
            return MV(Sim.tid.clone(), Int32)

    class _Bid:
        @property
        def x(self):
            return MV(torch.full((L,), Sim.bid[0], dtype=torch.int64), Int32)

        @property
        def y(self):
            return MV(torch.full((L,), Sim.bid[1], dtype=torch.int64), Int32)

    fx.thread_idx = _Idx()
    fx.block_idx = _Bid()
    fx.Stream = lambda *a, **k: None

    rocdl = types.ModuleType("flydsl.expr.rocdl")
    rocdl.make_buffer_ptr = _make_buffer_ptr
    rocdl.get_buffer_rsrc = lambda p: p
    rocdl.WMMA = lambda m, n, k, ab, acc=None, **kw: _WMMA(m, n, k, ab, acc)

    def RawPtrBufferLoadOp(res, rsrc, off, soff, *, aux=None):  # noqa: N802
        raise TraceError("op class must not be called")

    rocdl.RawPtrBufferLoadOp = RawPtrBufferLoadOp
    rocdl.raw_ptr_buffer_load = _raw_load
    rocdl.raw_ptr_buffer_store = _raw_store
    fx.rocdl = rocdl

    mlir = types.ModuleType("flydsl._mlir")
    ir = types.ModuleType("flydsl._mlir.ir")

    class VectorType:
        @staticmethod
        def get(shape, ety):
            return (int(shape[0]), ety)

    class IntegerType:
        @staticmethod
        def get_signless(w):
            return ("i", w)

    class IntegerAttr:
        @staticmethod
        def get(ty, v):
            return ("attr", v)

    ir.VectorType, ir.IntegerType, ir.IntegerAttr = VectorType, IntegerType, IntegerAttr
    mlir.ir = ir

    compiler = types.ModuleType("flydsl.compiler")

    def kernel(name=None, known_block_size=None):
        def deco(fn):
            fn._qsa_name = name
            fn._qsa_block = known_block_size
            return fn
        return deco

    compiler.kernel = kernel
    compiler.jit = lambda fn: fn
    compiler.compile = lambda *a, **k: None
    flydsl = types.ModuleType("flydsl")
    flydsl.expr, flydsl.compiler, flydsl._mlir = fx, compiler, mlir
    flydsl.__version__ = "trace-mock"
    mods = {"flydsl": flydsl, "flydsl.expr": fx, "flydsl.expr.rocdl": rocdl,
            "flydsl.compiler": compiler, "flydsl._mlir": mlir, "flydsl._mlir.ir": ir}
    sys.modules.update(mods)
    return mods


class _IfToMask(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        if node.orelse:
            raise TraceError("if/else in the kernel body is not expected")
        call = ast.Call(func=ast.Name("__qsa_mask", ast.Load()), args=[node.test], keywords=[])
        return ast.copy_location(ast.With(items=[ast.withitem(call)], body=node.body), node)

    def visit_For(self, node):
        self.generic_visit(node)
        it = node.iter
        name = getattr(getattr(it, "func", None), "id", None)
        if name != "range_constexpr":
            raise TraceError("runtime loop in the kernel body (not expected in the WMMA path)")
        return node


def _instrument(fn):
    """Re-compile the kernel function with `if` -> exec-mask contexts."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fdef = tree.body[0]
    fdef.decorator_list = []
    _IfToMask().visit(fdef)
    ast.fix_missing_locations(tree)
    ns = dict(fn.__globals__)
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        ns[name] = cell.cell_contents
    ns["__qsa_mask"] = _qsa_mask
    exec(compile(tree, KERNEL_PATH, "exec"), ns)  # noqa: S102
    return ns[fdef.name]


def load_module():
    _install_mock()
    spec = importlib.util.spec_from_file_location("qsa_page4_trace", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_kernel(K, q, cache, table, t2r, qpos, ctx, columns, pos64=False):
    """``pos64=True`` (ROUND-NATIVE64): build the int64-position kernel and pass
    the int64 ``qpos`` exactly as the wrapper does (zero-copy int32 view)."""
    block_rows = int(os.environ.get("QSA_TRACE_ROWS_PER_BLOCK", "16"))
    if pos64:
        launch = K._build_wmma_kernel(math.sqrt(128), block_rows, pos64=True)
    else:
        launch = K._build_wmma_kernel(math.sqrt(128), block_rows)
    closure = dict(zip(launch.__code__.co_freevars,
                       (c.cell_contents for c in launch.__closure__)))
    # The r16/r32 shape selector renamed the launcher capture in round J.
    kern = closure.get("selected_kernel", closure.get("kernel"))
    if kern is None:
        raise TraceError("launcher has neither selected_kernel nor kernel capture")
    if kern._qsa_block != [K.WMMA_THREADS, 1, 1]:
        raise TraceError(f"known_block_size {kern._qsa_block}")
    body = _instrument(kern)
    rows = q.shape[0]
    num_pages = cache.shape[0]
    num_requests, width = table.shape
    tensors = {
        "q": q.contiguous().view(torch.uint8).view(rows, -1),
        "kv": cache.contiguous().view(torch.uint8).view(num_pages, -1),
        "pt": table.to(torch.int32).contiguous(),
        "t2r": t2r.to(torch.int32).contiguous(),
        "qp": (qpos.to(torch.int64).contiguous().view(torch.int32) if pos64
               else qpos.to(torch.int32).contiguous()),
        "cl": ctx.to(torch.int32).contiguous(),
        "logits": torch.zeros(rows, columns, dtype=torch.float32),
        "visible_groups": torch.zeros(rows, dtype=torch.int32),
        "row_starts": torch.zeros(rows, dtype=torch.int32),
    }
    Sim.names = {id(v): k for k, v in tensors.items()}
    Sim.writes = {}
    gx, gy = K.wmma_grid(rows, columns, block_rows)
    for by in range(gy):
        for bx in range(gx):
            Sim.bid = (bx, by)
            for w in range(K.WMMA_WAVES):
                Sim.tid = torch.arange(L) + w * L
                Sim.mask = [torch.ones(L, dtype=torch.bool)]
                body(tensors["q"], tensors["kv"], tensors["pt"], tensors["t2r"], tensors["qp"],
                     tensors["cl"], tensors["logits"], tensors["visible_groups"],
                     tensors["row_starts"], Int32(rows), Int32(columns), Int32(num_pages),
                     Int32(num_requests), Int32(width))
    expect = {"logits": (rows * columns * 4, "f"), "visible_groups": (rows * 4, "i"),
              "row_starts": (rows * 4, "i")}
    ext = {k: Sim.planes[k].extent for k in Sim.planes}
    want_ext = {"q": rows * 1024, "kv": num_pages * 1024, "pt": num_requests * width * 4,
                "t2r": rows * 4, "qp": rows * (8 if pos64 else 4), "cl": num_requests * 4,
                "logits": rows * columns * 4, "visible_groups": rows * 4, "row_starts": rows * 4}
    for k, v in want_ext.items():
        if ext.get(k) != max(v, 4):
            raise TraceError(f"descriptor extent of {k} = {ext.get(k)}, want {v}")
    outs = {}
    for name, (nbytes, kind) in expect.items():
        rec = Sim.writes.get(name, {})
        n = nbytes // 4
        counts = torch.zeros(n, dtype=torch.int64)
        vals = torch.zeros(n, dtype=torch.float64)
        for o, (cnt, v) in rec.items():
            if o % 4:
                raise TraceError(f"unaligned store in {name}")
            counts[o // 4] = cnt
            vals[o // 4] = v
        if not bool((counts == 1).all()):
            raise TraceError(f"{name}: writes per element min={int(counts.min())} max={int(counts.max())}")
        outs[name] = vals
    logits = outs["logits"].to(torch.float32).view(rows, columns)
    vis = outs["visible_groups"].to(torch.int32)
    rst = outs["row_starts"].to(torch.int32)
    return logits, vis, rst


def _fixture(regime, rows, requests, table_width, cache_pages, seed=0):
    case = S.make_case("trace", regime, rows=rows, requests=requests, max_ctx=16 * table_width,
                       index_dtype="bf16", table_width=table_width, cache_pages=cache_pages)
    data = G.build_case(case, device="cpu", seed=seed)
    return case, data


def run_case(K, name, regime, rows, requests, table_width, cache_pages, max_columns=None,
             patch=None, golden=True):
    case, data = _fixture(regime, rows, requests, table_width, cache_pages)
    t2r = data["token_to_request"].clone()
    qpos = data["query_positions"].clone()
    ctx = data["context_lens"].clone()
    table = data["page_table"].clone()
    if patch is not None:
        patch(t2r, qpos, ctx, table)
    cols = case.columns if max_columns is None else int(max_columns)
    Sim.n_loads = Sim.n_stores = Sim.n_gemm = 0
    try:
        out, vis, rst = run_kernel(K, data["q"], data["compressed_k_cache"], table, t2r, qpos,
                                   ctx, cols)
    except TraceError as exc:
        print("[FAIL] %-24s %s" % (name, exc))
        return False
    vis_ref = G.visible_groups(t2r, qpos, ctx, compress_ratio=S.COMPRESS_RATIO)
    num_pages = data["compressed_k_cache"].shape[0]
    ar = torch.arange(cols)
    logical = ar // 4
    req_ok = (t2r.to(torch.int64) >= 0) & (t2r.to(torch.int64) < requests)
    safe_req = t2r.to(torch.int64).clamp(0, requests - 1)
    entry = table.to(torch.int64)[safe_req][:, logical.clamp(max=table.shape[1] - 1)]
    page_ok = req_ok[:, None] & (logical[None, :] < table.shape[1]) & (entry >= 0) & (entry < num_pages)
    inside = page_ok & (ar[None, :] < vis_ref.to(torch.int64)[:, None])
    checks = {
        "row_starts_zero": bool((rst == 0).all()),
        "visible_equal": bool(torch.equal(vis, vis_ref.to(torch.int32))),
        "outside_neg_inf": bool((torch.isinf(out[~inside]) & (out[~inside] < 0)).all()),
        "no_nan": not bool(torch.isnan(out).any()),
    }
    detail = ""
    if golden:
        gold, gold_vis, valid = G.reference_logits(
            data["q"], data["compressed_k_cache"], table, t2r, qpos, ctx,
            compress_ratio=S.COMPRESS_RATIO, max_columns=cols)
        res = G.compare_logits(out, vis, gold, gold_vis, valid)
        checks["golden"] = bool(res.get("pass"))
        detail = "checked=%d max_abs=%.2e" % (res.get("checked", 0), res.get("max_abs_err", 0.0))
    else:
        checks["inside_finite"] = bool(torch.isfinite(out[inside]).all()) if bool(inside.any()) else True
        detail = "contract-only checked=%d" % int(inside.sum())
    ok = all(checks.values())
    print("[%s] %-24s rows=%3d cols=%5d loads=%6d stores=%6d wmma=%5d %s%s"
          % ("PASS" if ok else "FAIL", name, rows, cols, Sim.n_loads, Sim.n_stores, Sim.n_gemm,
             detail, "" if ok else "  " + str(checks)))
    return ok


def main() -> int:
    K = load_module()
    print("kernel module (mock flydsl, real source):", KERNEL_PATH)
    fails = 0

    def p_vis0(t2r, qpos, ctx, table):
        qpos[0] = 2
        qpos[1] = 0

    def p_t2r(t2r, qpos, ctx, table):
        t2r[0] = -1
        t2r[1] = 999

    def p_bad_page(t2r, qpos, ctx, table):
        table[0, 0] = -1

    def p_page_oob(t2r, qpos, ctx, table):
        table[0, 0] = 1 << 24
        table[0, 1] = table.shape[1] * 0 + 10**6

    def p_mixed(t2r, qpos, ctx, table):
        pattern = [0, 1, 1, 0, 1, 0, 0, 1]
        for i in range(t2r.numel()):
            t2r[i] = pattern[i % len(pattern)]
        t2r[5] = -1
        if t2r.numel() > 9:
            t2r[9] = 7

    def p_short(t2r, qpos, ctx, table):
        ctx[0] = 40
        qpos[-1] = 70

    cases = [
        ("decode_min", "decode", 4, 1, 16, 64, None, None, True),
        ("decode_64x16", "decode", 64, 16, 16, 512, None, None, True),
        ("draft_16x16", "draft", 16, 16, 16, 512, None, None, True),
        ("prefill_272", "prefill", 272, 1, 16, 64, None, None, True),
        ("rows7", "prefill", 7, 2, 32, 128, None, None, True),
        ("rows13_cols70", "prefill", 13, 2, 32, 128, 70, None, True),
        ("tail_cols_1000", "prefill", 8, 2, 256, 1024, 1000, None, True),
        ("tail_cols_1", "decode", 4, 1, 16, 64, 1, None, True),
        ("tail_cols_5", "prefill", 4, 1, 65, 256, 5, None, True),
        ("visible_zero", "decode", 4, 1, 16, 64, None, p_vis0, True),
        ("t2r_invalid", "decode", 4, 1, 16, 64, None, p_t2r, True),
        ("bad_page", "decode", 4, 1, 16, 64, None, p_bad_page, True),
        ("page_oob", "decode", 4, 1, 16, 64, None, p_page_oob, False),
        ("mixed_requests", "prefill", 16, 2, 32, 128, None, p_mixed, True),
        ("early_exit", "prefill", 13, 2, 64, 256, None, p_short, True),
        ("width1", "decode", 4, 1, 1, 4, None, None, True),
    ]
    for (name, regime, rows, reqs, width, pages, maxc, patch, golden) in cases:
        fails += 0 if run_case(K, name, regime, rows, reqs, width, pages, maxc, patch, golden) else 1

    # i32 addressing at the cap: rows * columns * 4 just below 2**31 must not
    # overflow any intermediate (a single wave, columns tail at the edge).
    print("\ntrace check: %s" % ("PASS" if fails == 0 else "%d FAILURE(S)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
