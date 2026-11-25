# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel names in tuned tables: one parser for every family.

`parse(name)` returns a `KernelName` (family, stage and the fields the name encodes) or
None for an empty cell; `KernelName.format()` rebuilds the name from those fields, and
`parse(n).format() == n` holds for every name in today's tables. The legacy code keeps its
own `startswith` checks; v2 classifies names only through this module.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import math
import re
from dataclasses import dataclass

from ..problem import FrozenMap

ASM = "asm"  # _ZN5aiter<len><symbol>E, or a bare fmoe_<...> symbol
CK2STAGES = "ck2stages"  # moe_ck2stages_gemm{1,2}_...
CKTILE = "cktile"  # cktile_... and the swiglu_mxfp4_bf16_cktile alias
FLYDSL = "flydsl"  # flydsl_moe{1,2}_a.._w.._<out>_tMxNxK[_opts]
FLYDSL_LAYOUT = "flydsl_layout"  # flydsl_moe2_layout_a.._w.._<out>_tMxNxK[_opts]
MXMOE = "mxmoe"  # flydsl_mxmoe_g{1,2}_aXwY_MxNxK[_opts]
OPUS = "opus"  # opus_moe{1,2}[_layout]_...
IMPL = "impl"  # impl__<name>__<config>: a whole-graph implementation
GROUPED_GEMM = "grouped_gemm"  # grouped_gemm{1,2}_...
NULL = "null"  # the literal "Null" some 1-stage rows carry in kernelName2

FAMILIES = (
    ASM,
    CK2STAGES,
    CKTILE,
    FLYDSL,
    FLYDSL_LAYOUT,
    MXMOE,
    OPUS,
    IMPL,
    GROUPED_GEMM,
    NULL,
)


class KernelNameError(ValueError):
    pass


@dataclass(frozen=True)
class KernelName:
    family: str
    stage: int | None  # 1 or 2 for one GEMM stage; None for 1-stage and whole graph
    fields: FrozenMap

    def __getitem__(self, key):
        return self.fields[key]

    def get(self, key, default=None):
        return self.fields.get(key, default)

    def format(self) -> str:
        return _FORMATTERS[self.family](self)

    def __str__(self):
        return self.format()


def _opts(text):
    return tuple(t for t in text.split("_") if t) if text else ()


def _join_opts(opts):
    return "".join(f"_{o}" for o in opts)


_CK = re.compile(
    r"moe_ck2stages_gemm(?P<stage>[12])_"
    r"(?P<block>\d+)x(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<waves>\d+x\d+)_(?P<pipeline>[A-Za-z0-9]+)_v(?P<version>\d+)_"
    r"Nswizzle(?P<nswizzle>\d+)_Quant(?P<quant>\d+)_"
    r"MulRoutedWeight(?P<routed_weight>\d+)"
    r"(?:_(?P<act>[a-z]+))?"
    r"_(?P<a>[A-Z0-9]+)_(?P<b>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)"
)
_ASM = re.compile(r"_ZN5aiter(?P<length>\d+)(?P<symbol>\w+)E|(?P<bare>fmoe_\w+)")
_FLYDSL = re.compile(
    r"flydsl_moe(?P<stage>[12])(?P<layout>_layout)?_a(?P<a>[a-z0-9]+)_w(?P<w>[a-z0-9]+)_"
    r"(?P<out>[a-z0-9]+)_t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)"
    r"(?P<opts>(?:_[A-Za-z0-9]+)*)"
)
_MXMOE = re.compile(
    r"flydsl_mxmoe_g(?P<stage>[12])_a(?P<a>\d+)w(?P<w>\d+)_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)(?P<opts>(?:_[A-Za-z0-9]+)*)"
)
_OPUS = re.compile(
    r"opus_moe(?P<stage>[12])(?P<layout>_layout)?(?P<opts>(?:_[A-Za-z0-9]+)+)"
)
_GROUPED = re.compile(r"grouped_gemm(?P<stage>[12])(?P<opts>(?:_[A-Za-z0-9]+)+)")
_CKTILE = re.compile(r"cktile(?P<opts>(?:_[A-Za-z0-9]+)+)")
_CKTILE_ALIAS = "swiglu_mxfp4_bf16_cktile"
_IMPL_PREFIX = "impl__"


def _ints(match, *names):
    return {n: int(match[n]) for n in names}


def _parse_ck(m):
    fields = {
        **_ints(
            m, "block", "tile_m", "tile_n", "tile_k", "version", "nswizzle", "quant"
        ),
        "waves": m["waves"],
        "pipeline": m["pipeline"],
        "routed_weight": int(m["routed_weight"]),
        "act": m["act"],
        "a": m["a"],
        "b": m["b"],
        "out": m["out"],
    }
    return int(m["stage"]), fields


def _format_ck(k):
    f = k.fields
    act = f"_{f['act']}" if f["act"] else ""
    return (
        f"moe_ck2stages_gemm{k.stage}_{f['block']}x{f['tile_m']}x{f['tile_n']}x"
        f"{f['tile_k']}_{f['waves']}_{f['pipeline']}_v{f['version']}_"
        f"Nswizzle{f['nswizzle']}_Quant{f['quant']}_MulRoutedWeight{f['routed_weight']}"
        f"{act}_{f['a']}_{f['b']}_{f['out']}"
    )


def _parse_asm(m):
    symbol = m["symbol"] or m["bare"]
    tile = re.search(r"_(\d+)x(\d+)", symbol)
    fields = {
        "symbol": symbol,
        # Kept as written: one shipped name declares 47 for a 46-character symbol.
        # None for a bare symbol.
        "length": None if m["bare"] else int(m["length"]),
        "tile_m": int(tile[1]) if tile else None,
        "tile_n": int(tile[2]) if tile else None,
    }
    return (1 if symbol.startswith("fmoe_stage1_") else None), fields


def _format_asm(k):
    if k.fields["length"] is None:
        return k.fields["symbol"]
    return f"_ZN5aiter{k.fields['length']}{k.fields['symbol']}E"


def _parse_flydsl(m):
    fields = {
        **_ints(m, "tile_m", "tile_n", "tile_k"),
        "a": m["a"],
        "w": m["w"],
        "out": m["out"],
        "opts": _opts(m["opts"]),
    }
    return int(m["stage"]), fields


def _format_flydsl(k):
    f = k.fields
    layout = "_layout" if k.family == FLYDSL_LAYOUT else ""
    return (
        f"flydsl_moe{k.stage}{layout}_a{f['a']}_w{f['w']}_{f['out']}_"
        f"t{f['tile_m']}x{f['tile_n']}x{f['tile_k']}{_join_opts(f['opts'])}"
    )


def _parse_mxmoe(m):
    fields = {
        **_ints(m, "a", "w", "tile_m", "tile_n", "tile_k"),
        "opts": _opts(m["opts"]),
    }
    return int(m["stage"]), fields


def _format_mxmoe(k):
    f = k.fields
    return (
        f"flydsl_mxmoe_g{k.stage}_a{f['a']}w{f['w']}_{f['tile_m']}x{f['tile_n']}x"
        f"{f['tile_k']}{_join_opts(f['opts'])}"
    )


def _format_opus(k):
    layout = "_layout" if k.fields["layout"] else ""
    return f"opus_moe{k.stage}{layout}{_join_opts(k.fields['opts'])}"


def _format_grouped(k):
    return f"grouped_gemm{k.stage}{_join_opts(k.fields['opts'])}"


def _format_cktile(k):
    if k.fields["alias"]:
        return _CKTILE_ALIAS
    return f"cktile{_join_opts(k.fields['opts'])}"


def _format_impl(k):
    return f"{_IMPL_PREFIX}{k.fields['name']}__{k.fields['config']}"


_FORMATTERS = {
    ASM: _format_asm,
    CK2STAGES: _format_ck,
    CKTILE: _format_cktile,
    FLYDSL: _format_flydsl,
    FLYDSL_LAYOUT: _format_flydsl,
    MXMOE: _format_mxmoe,
    OPUS: _format_opus,
    IMPL: _format_impl,
    GROUPED_GEMM: _format_grouped,
    NULL: lambda k: "Null",
}


def is_empty(value) -> bool:
    """An empty tuned-table cell: None, NaN (pandas) or a blank string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def parse(name) -> KernelName | None:
    """Parse one kernel-name cell; None for an empty cell, KernelNameError if unknown."""
    if is_empty(name):
        return None
    if not isinstance(name, str):
        raise KernelNameError(f"kernel name is not a string: {name!r}")
    if name == "Null":
        return KernelName(NULL, None, FrozenMap())
    if name == _CKTILE_ALIAS:
        return KernelName(CKTILE, 2, FrozenMap({"alias": True, "opts": ()}))
    if name.startswith(_IMPL_PREFIX):
        impl, sep, config = name[len(_IMPL_PREFIX) :].partition("__")
        if not impl or not sep:
            raise KernelNameError(f"invalid whole-graph kernel name {name!r}")
        return KernelName(IMPL, None, FrozenMap({"name": impl, "config": config}))
    for family, pattern, parser in (
        (CK2STAGES, _CK, _parse_ck),
        (ASM, _ASM, _parse_asm),
        (MXMOE, _MXMOE, _parse_mxmoe),
    ):
        m = pattern.fullmatch(name)
        if m:
            stage, fields = parser(m)
            return KernelName(family, stage, FrozenMap(fields))
    m = _FLYDSL.fullmatch(name)
    if m:
        stage, fields = _parse_flydsl(m)
        family = FLYDSL_LAYOUT if m["layout"] else FLYDSL
        return KernelName(family, stage, FrozenMap(fields))
    m = _OPUS.fullmatch(name)
    if m:
        fields = {"layout": bool(m["layout"]), "opts": _opts(m["opts"])}
        return KernelName(OPUS, int(m["stage"]), FrozenMap(fields))
    m = _GROUPED.fullmatch(name)
    if m:
        return KernelName(
            GROUPED_GEMM, int(m["stage"]), FrozenMap({"opts": _opts(m["opts"])})
        )
    m = _CKTILE.fullmatch(name)
    if m:
        return KernelName(
            CKTILE, None, FrozenMap({"alias": False, "opts": _opts(m["opts"])})
        )
    raise KernelNameError(f"unknown kernel-name family: {name!r}")


def try_parse(name):
    """`parse`, but an unknown name gives None instead of raising."""
    try:
        return parse(name)
    except KernelNameError:
        return None
