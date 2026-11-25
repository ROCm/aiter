# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The `Segment` interface, capabilities and parameters.

A segment is a thin adapter over one kernel family that covers a span of the op chain.
It declares what it can run as `Cap`s; anything a problem has that no `Cap` declares
(a feature, a shared expert, a tail, padding, an extra) rejects the segment, so a new
input never reaches a kernel that would silently ignore it.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from dataclasses import dataclass

from ..dispatch.ops import Op
from ..problem import FrozenMap, MoeProblem


class Params(FrozenMap):
    """A segment's or plan's parameters: an immutable mapping with attribute access."""

    __slots__ = ()

    def __init__(self, items=None, **kwargs):
        super().__init__({**(items or {}), **kwargs})

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def replace(self, **kwargs) -> "Params":
        return Params(self, **kwargs)

    def __repr__(self):
        return "Params(" + ", ".join(f"{k}={v!r}" for k, v in self.items()) + ")"


def _short(dtype):
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class Cap:
    """One combination a segment supports. `None` means "any"; sets mean membership.

    `features` follows the subset rule: every feature the problem uses must be listed.
    """

    arch: frozenset | None = None
    act_in: frozenset | None = None  # ScaleSpecs
    w1: frozenset | None = None
    w2: frozenset | None = None
    out_dtype: frozenset | None = None
    activation: frozenset | None = None  # ActSpec.kind values
    gate_mode: frozenset | None = None
    shuffle: frozenset | None = None
    g1u1: frozenset | None = None
    features: frozenset = frozenset()
    pads: bool = False  # hidden/intermediate padding supported
    extras: frozenset = frozenset()  # extras keys understood
    shared: bool = False
    parallel: bool = False
    tail: bool = False

    def reject_reason(self, p: MoeProblem) -> str | None:
        """Why this capability does not cover `p`, or None if it does."""
        layout = p.layout
        checks = (
            ("arch", self.arch, p.arch, str),
            ("act_in", self.act_in, p.act_in, str),
            ("w1", self.w1, p.w1, str),
            ("w2", self.w2, p.w2, str),
            ("out dtype", self.out_dtype, p.out_dtype, _short),
            ("activation", self.activation, p.activation.kind, str),
            ("gate mode", self.gate_mode, layout.gate_mode, str),
            ("weight shuffle", self.shuffle, layout.shuffle, str),
            ("g1u1", self.g1u1, layout.g1u1, str),
        )
        for what, allowed, value, show in checks:
            if allowed is not None and value not in allowed:
                return f"{what} {show(value)} not in {{{', '.join(sorted(map(show, allowed)))}}}"
        missing = sorted(p.features - self.features)
        if missing:
            return f"feature {', '.join(missing)} not supported"
        if not self.pads and (layout.hidden_pad or layout.intermediate_pad):
            return "hidden/intermediate padding not supported"
        unknown = sorted(set(p.extras) - self.extras)
        if unknown:
            return f"extras {', '.join(unknown)} not understood"
        for what, allowed, value in (
            ("shared expert", self.shared, p.shared),
            ("parallel spec", self.parallel, p.parallel),
            ("tail", self.tail, p.tail),
        ):
            if value is not None and not allowed:
                return f"{what} not supported"
        return None


@dataclass(frozen=True)
class RowRejection:
    """A tuned row a segment cannot run. `warning` is logged as is; None is silent."""

    warning: str | None


class Segment:
    """Base class of every segment.

    `name`: unique, used in plans and logs. `prefixes`: the kernel-name families
    (`kernel_names.FAMILIES`) whose tuned rows this segment runs. `covers`: the first and
    last op of the chain it implements. `capabilities`: the `Cap`s it supports.
    `structure_params`: parameters that change the boundary buffers (e.g. `block_m`), so
    every segment of a plan must agree on them. The remaining attributes describe stateful
    and collective segments; no ported segment sets them yet.
    """

    name: str = ""
    prefixes: tuple = ()
    covers: tuple = (Op.SORT, Op.SORT)
    capabilities: tuple = ()
    stateful: bool = False
    structure_params: frozenset = frozenset()
    rank_local_params: bool = False
    zero_tokens_ok: bool = True
    hints: frozenset = frozenset()
    output_lifetime: str = "caller"
    lossy_values: FrozenMap = FrozenMap()
    default_lossy: FrozenMap = FrozenMap()

    def reject_reason(self, p: MoeProblem) -> str | None:
        """None if some capability covers `p`, else the reason of the closest one."""
        if not self.capabilities:
            return "no capabilities declared"
        reasons = []
        for cap in self.capabilities:
            reason = cap.reject_reason(p)
            if reason is None:
                return None
            reasons.append(reason)
        return reasons[0] if len(reasons) == 1 else "; ".join(dict.fromkeys(reasons))

    def accepts(self, p: MoeProblem) -> bool:
        return self.reject_reason(p) is None

    def default_params(self, p: MoeProblem) -> Params:
        return Params()

    def owns(self, kernel) -> bool:
        """Whether this segment runs the parsed tuned kernel name `kernel`."""
        return kernel is not None and kernel.family in self.prefixes

    def produces(self, p: MoeProblem, params: Params):
        """The `Buf` this segment hands to the next one."""
        raise NotImplementedError(self.name)

    def check_row(self, p: MoeProblem, row, kernel_name1="") -> RowRejection | None:
        """Whether this segment can run tuned `row` for `p` (None: it can)."""
        return None

    def build(self, p: MoeProblem, params: Params):
        raise NotImplementedError(self.name)

    def __repr__(self):
        return f"<segment {self.name}>"
