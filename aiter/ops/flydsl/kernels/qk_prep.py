# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Q/K prep that a GEMM epilogue applies to each head of a projection.

A ``PrepSpec`` names the prep's norm and RoPE, and whether it has a bias and a Hadamard.
``NORMS`` and ``ROPES`` map the names to ``PrepOp``s, so ``PrepSpec.ops`` is the norm's op,
the RoPE's and the Hadamard. ``StoreCRows`` (``qk_prep_epilogue``) adds the bias, then
applies the ops in order to the 8 FP32 values that each lane holds of one row of a
128-wide head; 16 lanes hold the head. Each op declares what it reads and what it leaves,
and the epilogue, the host GEMMs, the pre-attention monokernel and the tests derive the
rest from those declarations:

- ``columns``: names of per-column FP32 vectors, N values each, that the epilogue loads
  from the column operand after the scales and the bias, in spec order. ``pack_columns``
  multiplies each by the ``gain`` of every linear op after the op that reads it.
- ``tables``: names and widths of per-token FP32 tables, side by side in one buffer of
  ``[M, table_width]`` (``pack_tables``).
- ``row_factor``: the op leaves one scalar per row instead of applying it. The epilogue
  stores each head's partial (``leave``); the caller adds a row's partials and turns the
  sum into the factor (``finish``) before it quantizes the row. The ops after it must be
  ``linear``, so the factor still applies to their output.

The epilogue gives an op a context with ``lane`` (the lane in the wave), ``column(name)``
(the lane's 8 values of a column vector), ``table(name, row, index, count)`` (``count``
values, a multiple of 4, of a table row from ``index``) and ``leave(value)``.

Adding a capability is one ``PrepOp`` subclass, with its kernel math (``setup``, ``load``,
``apply``, ``finish``), its declarations, its kernel-name ``tag`` and its PyTorch
``reference``, and one entry in ``NORMS`` or ``ROPES``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import flydsl.expr as fx
import torch
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

HEAD_DIM = 128  # channels per head: 8 in each of 16 lanes
ROPE_PAIRS = HEAD_DIM // 2  # rotation pairs per head, one cos and sin each


# DPP moves within a 16-lane row and the lane bit that orders each pair: quad_perm
# [1,0,3,2] and [2,3,0,1] pair lanes i ^ 1 and i ^ 2, row_half_mirror i ^ 7 and
# row_ror:8 i ^ 8. The four masks span the row's lane bits, so a butterfly over them
# is a Hadamard transform, with rows in a different order than Sylvester's.
_ROW_BUTTERFLY = ((0xB1, 1), (0x4E, 2), (0x141, 4), (0x128, 8))
_ROW_ROR8 = 0x128  # lane i of a 16-lane row reads lane i ^ 8


def _row_move(value, ctrl):
    # FlyDSL 0.5 has no stable DPP move: rocdl.update_dpp is outside rocdl.__all__.
    # The stable cross-lane read, shuffle_xor with the same lane pairs, lowers to
    # ds_swizzle_b32, an LDS-unit permute that each consumer waits for (lgkmcnt): at
    # Wan's and FLUX.3's projections (1184x5120x5120, 576x3072x3072, persistent) the
    # prep epilogue then ran 4.6-13.6 % slower. Every lane of the row has a source under
    # these controls, so the zero fill (bound_ctrl) never shows; it lets LLVM fold the
    # move into the instruction that consumes it.
    bits = fx.Float32(value).bitcast(fx.Int32)
    moved = update_dpp_i32(bits, bits, ctrl, 0xF, 0xF, True)
    return fx.Int32(moved).bitcast(fx.Float32)


def _butterfly_signs(lane):
    """Per stage of _ROW_BUTTERFLY: +1 in the lane that keeps the sum, -1 in its partner."""
    return [
        ((lane & bit) == 0).select(fx.Float32(1.0), fx.Float32(-1.0))
        for _, bit in _ROW_BUTTERFLY
    ]


def _row_sum(value):
    """Sum over the 16 lanes of a DPP row, in every lane of it."""
    for ctrl, _ in _ROW_BUTTERFLY:
        value = value + _row_move(value, ctrl)
    return value


def _row_max(value):
    """Maximum over the 16 lanes of a DPP row, in every lane of it."""
    for ctrl, _ in _ROW_BUTTERFLY:
        value = value.maximumf(_row_move(value, ctrl))
    return value


def _max_abs(values):
    """Largest magnitude among the 8 values of a vector, as a tree."""
    level = [values[k].maximumf(-values[k]) for k in range(8)]
    while len(level) > 1:
        level = [level[k].maximumf(level[k + 1]) for k in range(0, len(level), 2)]
    return level[0]


def _hadamard128(values, signs):
    """Unnormalized Hadamard over one 128-wide head held as 8 values in each of 16 lanes.

    Q and K go through the same transform, so their attention scores are unchanged.
    ``signs`` comes from _butterfly_signs.
    """
    result = list(values)
    for stage in range(3):
        shift = 1 << stage
        result = [
            (
                result[i ^ shift] - result[i]
                if i & shift
                else result[i] + result[i ^ shift]
            )
            for i in range(8)
        ]
    for (ctrl, _), sign in zip(_ROW_BUTTERFLY, signs):
        result = [_row_move(value, ctrl) + sign * value for value in result]
    return result


def _lanes(x):
    """A lane's 8 values as a list."""
    return x if isinstance(x, list) else [x[k] for k in range(8)]


def _vector(x):
    """A lane's 8 values as a vector."""
    return Vec.from_elements(x, fx.Float32) if isinstance(x, list) else x


class PrepOp(ABC):
    """One capability of the Q/K prep; subclasses are frozen dataclasses.

    Traced into the epilogue, with ``ctx`` the epilogue's context (module docstring):
      - ``setup(ctx)``: once per epilogue call, before its tiles; returns the op's state.
      - ``load(ctx, row)``: per row, before the row's math, for an op with ``tables``;
        returns the row's data.
      - ``apply(x, ctx, state, data)``: the lane's 8 values of the row (a ``Vec`` or a
        list), transformed.
      - ``finish(total, count)``: for a ``row_factor`` op, the factor from the sum of a
        row's partials over ``count`` columns.

    On the host, ``reference(y, params, tables)`` applies the op to FP32
    ``[rows, heads, 128]`` values, with its folded columns in ``params`` and its
    ``[rows, width]`` tables in ``tables``, and returns the result and, for a ``row_factor``
    op, the per-head partials. ``reference_finish`` is ``finish`` in PyTorch.

    ``tag`` names the op in the kernel name. Names are for attribution in profiles and
    disassembly (FlyDSL keys compiled kernels on the traced code), so a tag spells out
    every parameter the kernel math depends on, in letters, digits and underscores.
    """

    columns: ClassVar[tuple] = ()
    tables: ClassVar[tuple] = ()
    row_factor: ClassVar[bool] = False
    linear: ClassVar[bool] = False
    gain: ClassVar[float] = 1.0

    def setup(self, ctx):
        return None

    def load(self, ctx, row):
        return None

    @abstractmethod
    def apply(self, x, ctx, state, data):
        pass

    def finish(self, total, count):
        raise NotImplementedError(f"{type(self).__name__} leaves no row factor")

    @abstractmethod
    def reference(self, y, params, tables):
        pass

    def reference_finish(self, total, count):
        raise NotImplementedError(f"{type(self).__name__} leaves no row factor")

    @property
    @abstractmethod
    def tag(self):
        pass


@dataclass(frozen=True)
class RMSNorm(PrepOp):
    """RMSNorm of each head, or of all heads of a row (``across_heads``), then the norm
    weight. Across heads the epilogue leaves the factor ``rsqrt(sum / N + eps)``: it stores
    each head's sum of squares of the values before the weight."""

    across_heads: bool
    eps: float

    columns: ClassVar[tuple] = ("weight",)

    @property
    def row_factor(self):
        return self.across_heads

    @property
    def tag(self):
        name = "nacross" if self.across_heads else "nhead"
        if self.eps == 1e-6:
            return name
        # 1e-05 is written 1em05 and 0.0001 0p0001.
        eps = f"{self.eps:g}".replace("-", "m").replace("+", "").replace(".", "p")
        return f"{name}_eps{eps}"

    def apply(self, x, ctx, state, data):
        x = _vector(x)
        sq = _row_sum(fx.Float32((x * x).reduce(ReductionOp.ADD)))
        if self.across_heads:
            ctx.leave(sq)
        else:
            x = x * Vec.filled(
                8, fmath.rsqrt(sq * (1.0 / HEAD_DIM) + self.eps), fx.Float32
            )
        return x * ctx.column("weight")

    def finish(self, total, count):
        return fmath.rsqrt(total * (1.0 / count) + self.eps)

    def reference(self, y, params, tables):
        sq = (y * y).sum(-1)
        if not self.across_heads:
            y = y * torch.rsqrt(sq / HEAD_DIM + self.eps)[..., None]
        y = y * params["weight"].view(y.shape[-2], HEAD_DIM)
        return y, (sq if self.across_heads else None)

    def reference_finish(self, total, count):
        return torch.rsqrt(total / count + self.eps)


@dataclass(frozen=True)
class RoPE(PrepOp):
    """RoPE from per-token cos and sin tables of one value per rotation pair, the same for
    every head: ``"interleaved"`` rotates channel pairs (2i, 2i + 1), ``"rotate_half"`` pairs
    i and i + 64."""

    kind: str

    tables: ClassVar[tuple] = (("cos", ROPE_PAIRS), ("sin", ROPE_PAIRS))
    linear: ClassVar[bool] = True

    def __post_init__(self):
        if self.kind not in ("interleaved", "rotate_half"):
            raise ValueError(f"unknown RoPE kind {self.kind!r}")

    @property
    def tag(self):
        return "rinter" if self.kind == "interleaved" else "rhalf"

    def load(self, ctx, row):
        """cos and sin of the rotation pairs this lane's 8 channels of ``row`` take part in.

        Interleaved: the 4 pairs the lane holds. Rotate-half: lanes l and l ^ 8 hold the two
        halves of 8 pairs, so both read the same 8 values.
        """
        if self.kind == "rotate_half":
            index = (ctx.lane % 8) * 8
            return tuple(ctx.table(name, row, index, 8) for name in ("cos", "sin"))
        index = (ctx.lane % 16) * 4
        return tuple(ctx.table(name, row, index, 4) for name in ("cos", "sin"))

    def apply(self, x, ctx, state, data):
        cos, sin = data
        rotated = []
        if self.kind == "interleaved":
            for k in range_constexpr(4):
                even, odd = x[2 * k], x[2 * k + 1]
                rotated += [even * cos[k] - odd * sin[k], odd * cos[k] + even * sin[k]]
            return rotated
        # Channel c < 64 takes -x[c + 64] sin, channel c >= 64 takes +x[c - 64] sin.
        low = (ctx.lane % 16) < 8
        for k in range_constexpr(8):
            partner = _row_move(x[k], _ROW_ROR8)
            rotated.append(x[k] * cos[k] + low.select(-partner, partner) * sin[k])
        return rotated

    def reference(self, y, params, tables):
        cos, sin = tables["cos"][:, None, :], tables["sin"][:, None, :]
        if self.kind == "interleaved":
            even, odd = y[..., 0::2], y[..., 1::2]
            y = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
            return y.flatten(-2), None
        low, high = y[..., :ROPE_PAIRS], y[..., ROPE_PAIRS:]
        return torch.cat((low * cos - high * sin, high * cos + low * sin), -1), None


@dataclass(frozen=True)
class Hadamard(PrepOp):
    """The unnormalized Hadamard of each head (``_hadamard128``): ``y = x H`` with
    ``H H^T = 128 I``. H is Sylvester's matrix with its rows permuted and 32 of them
    negated, the order the DPP butterflies give (``_ROW_BUTTERFLY``), so it is not
    interchangeable with a Sylvester-order transform such as ``rot_quant``'s or
    ``rope_rotate_activation``: attention scores are preserved only when Q and K both
    pass this one, as the prepped roles of one epilogue do. Its gain, 128 ** -0.5, folds
    into the columns of the ops before it (the norm weight); without such a column the
    output stays unnormalized."""

    linear: ClassVar[bool] = True
    gain: ClassVar[float] = HEAD_DIM**-0.5

    @property
    def tag(self):
        return "h"

    def setup(self, ctx):
        return _butterfly_signs(ctx.lane)

    def apply(self, x, ctx, signs, data):
        return _hadamard128(_lanes(x), signs)

    def reference(self, y, params, tables):
        # Channel 8 * lane + k of a head: three butterflies over a lane's 8 channels, then
        # four across the head's 16 lanes, pairing lanes i ^ 1, i ^ 2, i ^ 7 and i ^ 8 as
        # the kernel's DPP moves do.
        v = y.reshape(*y.shape[:-1], 16, 8)
        k = torch.arange(8, device=y.device)
        for shift in (1, 2, 4):
            partner = v[..., k ^ shift]
            v = torch.where((k & shift).bool(), partner - v, v + partner)
        lane = torch.arange(16, device=y.device)
        for mask, bit in ((1, 1), (2, 2), (7, 4), (8, 8)):
            sign = torch.where((lane & bit).bool(), -1.0, 1.0)[:, None]
            v = v[..., lane ^ mask, :] + sign * v
        return v.reshape(y.shape), None


# PrepSpec's norm and RoPE names and the ops they select, built from the spec.
NORMS = {
    "none": None,
    "rms_per_head": lambda spec: RMSNorm(False, spec.eps),
    "rms_across_heads": lambda spec: RMSNorm(True, spec.eps),
}
ROPES = {
    "none": None,
    "interleaved": lambda spec: RoPE("interleaved"),
    "rotate_half": lambda spec: RoPE("rotate_half"),
}


@dataclass(frozen=True)
class PrepSpec:
    """The Q/K prep of ``StoreCRows``, fixed at compile time.

    ``norm``: a ``NORMS`` name: ``"rms_across_heads"``, RMSNorm over every head of a row,
    whose per-row factor the epilogue leaves to the caller; ``"rms_per_head"``; or
    ``"none"``. ``rope``: a ``ROPES`` name: ``"interleaved"``, ``"rotate_half"`` or
    ``"none"``. ``bias``: a per-column bias before the norm. ``hadamard``: the per-head
    Hadamard after RoPE. ``eps``: the norm's epsilon.
    """

    norm: str
    rope: str
    bias: bool
    hadamard: bool
    eps: float = 1e-6

    def __post_init__(self):
        if self.norm not in NORMS:
            raise ValueError(f"unknown PrepSpec norm {self.norm!r}")
        if self.rope not in ROPES:
            raise ValueError(f"unknown PrepSpec rope {self.rope!r}")
        factors = [i for i, op in enumerate(self.ops) if op.row_factor]
        if len(factors) > 1:
            raise ValueError("a PrepSpec leaves at most one row factor")
        if factors and not all(op.linear for op in self.ops[factors[0] + 1 :]):
            raise ValueError("the ops after a row-factor op must be linear")
        names = list(self.columns) + [name for name, _ in self.tables]
        if len(set(names)) != len(names):
            raise ValueError(f"two ops declare the same input: {names}")

    @cached_property
    def ops(self):
        """The ops in order: the norm's, the RoPE's, then the Hadamard."""
        stages = (NORMS[self.norm], ROPES[self.rope])
        ops = tuple(build(self) for build in stages if build is not None)
        return ops + ((Hadamard(),) if self.hadamard else ())

    @property
    def columns(self):
        """The column vectors' names, in the order the column operand holds them."""
        return tuple(name for op in self.ops for name in op.columns)

    @property
    def tables(self):
        """The tables' names and widths, in the order a tables row holds them."""
        return tuple(table for op in self.ops for table in op.tables)

    @property
    def table_width(self):
        return sum(width for _, width in self.tables)

    def table_offset(self, name):
        offset = 0
        for table, width in self.tables:
            if table == name:
                return offset
            offset += width
        raise KeyError(name)

    @property
    def row_factor(self):
        """The op that leaves a row factor, or None."""
        return next((op for op in self.ops if op.row_factor), None)

    @property
    def suffix(self):
        """Kernel-name suffix after ``_prep``: the ops' tags in order."""
        return "".join(f"_{op.tag}" for op in self.ops)

    def gains(self):
        """Per column name, the product of the gains of the linear ops between the op that
        reads it and the next op that is not linear."""
        gains = {}
        for i, op in enumerate(self.ops):
            gain = 1.0
            for later in self.ops[i + 1 :]:
                if not later.linear:
                    break
                gain *= later.gain
            for name in op.columns:
                gains[name] = gain
        return gains

    def _vector(self, value, heads):
        value = value.float().reshape(-1)
        return value.repeat(heads) if value.numel() == HEAD_DIM else value

    def pack_columns(self, scales, bias, roles, heads, n=None):
        """The FP32 column operand of a projection with ``heads`` heads per role.

        ``scales``: the N column scales, or None for a host that applies none, which then
        passes N as ``n``. ``bias``: the N biases with ``self.bias``. ``roles``: per prepped
        role (one for a single projection; Q and K of a fused Q/K/V weight), a dict of the
        column vectors by name, ``[heads * 128]`` or one head's ``[128]``; the roles after
        them (V) take ones. Each vector is multiplied by its gain (``gains``).
        """
        parts = [] if scales is None else [scales.float().reshape(-1)]
        n = n if scales is None else scales.numel()
        if self.bias:
            parts.append(bias.float().reshape(-1))
        rest = n // (heads * HEAD_DIM) - len(roles)
        gains = self.gains()
        for name in self.columns:
            vectors = [self._vector(role[name], heads) * gains[name] for role in roles]
            parts += vectors
            if rest:
                device = vectors[0].device
                parts.append(torch.ones(rest * heads * HEAD_DIM, device=device))
        return torch.cat(parts).contiguous()

    def pack_tables(self, tables):
        """The tables buffer, FP32 ``[rows, table_width]``: each declared table's
        ``[rows, width]`` values side by side, or None without tables."""
        if not self.tables:
            return None
        return torch.cat(
            [tables[name].float().reshape(-1, width) for name, width in self.tables],
            dim=1,
        ).contiguous()

    def reference(self, y, params, tables):
        """The ops in PyTorch on FP32 ``[rows, heads, 128]`` values that hold the bias, with
        the column vectors in ``params`` (folded as ``pack_columns`` folds them) and the
        ``[rows, width]`` ``tables``. Returns the result and the row-factor op's per-head
        partials, or None."""
        heads = y.shape[-2]
        gains = self.gains()
        folded = {
            name: self._vector(params[name], heads) * gains[name]
            for name in self.columns
        }
        partials = None
        for op in self.ops:
            y, leaves = op.reference(y, folded, tables)
            if leaves is not None:
                partials = leaves
        return y, partials
