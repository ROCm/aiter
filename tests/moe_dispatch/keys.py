# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch keys: every tuned row plus a grid of untuned keys, per simulated arch.

A key is the full argument set of one `get_2stage_cfgs` call, built the way
`_fused_moe_impl` and `aiter.fhmoe` build it. Keys come in three sections:

- `tuned`: one key per row of the merged runtime table for the arch, in three call
  variants (`base`, `ep`, `unshuffled`);
- `fhmoe`: one key per row of the dedicated FHMoE table (`config_file=` path);
- `grid`: the quant combinations found in any tuned table (fp8 mapped to the arch's
  fp8), times `GRID_SHAPES`, times `GRID_M`, in the `base` and `ep` variants.

The `base` tuned and grid keys of the quant types in `FEATURE_QUANT_TYPES` also come in
feature variants (`feature_keys`): the same call with one more `get_2stage_cfgs` feature
(`num_local_tokens`, activation scales, bias, padding, ...), so a changed decision for
it shows up as its own key.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

from dataclasses import dataclass, field

import torch

GRID_SHAPES = (
    # (model_dim, inter_dim, experts, topk)
    (7168, 256, 256, 8),  # DeepSeek-V3 / Kimi, TP8
    (7168, 2048, 256, 8),  # DeepSeek-V3, TP1
    (4096, 192, 128, 8),  # Qwen3-235B, TP8
    (4096, 1536, 128, 8),  # Qwen3-235B, TP1
    (2880, 2880, 128, 4),  # GPT-OSS-120B, TP1
    (2880, 360, 128, 4),  # GPT-OSS-120B, TP8 (inter_dim not 256-aligned)
    (4096, 14336, 8, 2),  # Mixtral-8x7B
    (2048, 768, 128, 8),  # Qwen3-30B-A3B
)

GRID_M = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    131072,
)

# Combinations no tuned table contains yet, but that dispatch has branches for.
EXTRA_COMBOS = (
    # (activation, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)
    ("Silu", "bfloat16", "int8", "int8", "per_Token", True, False),
    ("Silu", "bfloat16", "int8", "int8", "per_Token", False, False),
    ("Silu", "bfloat16", "bfloat16", "bfloat16", "No", False, False),
    ("Silu", "float16", "float16", "float16", "No", True, False),
    ("Relu2", "bfloat16", "bfloat16", "float4_e2m1fn_x2", "per_1x32", False, False),
    ("Silu", "bfloat16", "fp8", "fp8", "per_Tensor", True, False),
    ("Gelu", "bfloat16", "fp8", "fp8", "per_Token", True, False),
)

TUNED_VARIANTS = ("base", "ep", "unshuffled")
GRID_VARIANTS = ("base", "ep")
# Quant types whose keys get the feature variants; a v2 path adds its quant type.
FEATURE_QUANT_TYPES = ("per_Tensor",)

_FP8_NAMES = ("float8_e4m3fn", "float8_e4m3fnuz")


@dataclass(frozen=True)
class Key:
    section: str
    variant: str
    args: tuple
    kwargs: tuple = field(default=())

    def call_kwargs(self):
        return dict(self.kwargs)

    def label(self):
        a = self.args
        parts = [
            self.section,
            self.variant,
            f"M={a[0]}",
            f"H={a[1]}",
            f"I={a[2]}",
            f"E={a[3]}",
            f"k={a[4]}",
            _short(a[10]),
            _short(a[5]),
            _short(a[6]),
            _short(a[7]),
            _short(a[8]),
            f"g1u1={int(a[9])}",
            f"dw={int(a[11])}",
        ]
        if a[12] or a[13]:
            parts.append(f"pad={a[12]},{a[13]}")
        kw = dict(self.kwargs)
        if "gate_mode" in kw:
            parts.append(f"gate={_short(kw['gate_mode'])}")
        if kw.get("config_file"):
            parts.append("config=fhmoe")
        return "|".join(parts)


def _short(value):
    text = str(value)
    for prefix in ("torch.", "ActivationType.", "QuantType.", "GateMode."):
        text = text.replace(prefix, "")
    return text


def parse_dtype(text, arch_fp8=None):
    name = str(text).replace("torch.", "")
    if name == "fp8" or (arch_fp8 is not None and name in _FP8_NAMES):
        return arch_fp8
    return getattr(torch, name)


def _enum(enum_cls, text):
    return getattr(enum_cls, str(text).split(".")[-1])


def _activation_kwargs(activation):
    from aiter.fused_moe import _normalize_mxfp4_activation_params

    situ_beta, situ_linear_beta, swiglu_limit = _normalize_mxfp4_activation_params(
        activation, None, None, None
    )
    return {
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
        "swiglu_limit": swiglu_limit,
    }


def _variant_kwargs(variant, activation):
    from aiter.ops.flydsl.moe_common import GateMode

    shuffled = variant != "unshuffled"
    return {
        "is_shuffled": shuffled,
        "gate_mode": GateMode.SEPARATED,
        "is_ep": variant == "ep",
        "has_stage1_bias": False,
        "has_stage2_bias": False,
        **_activation_kwargs(activation),
        "opus_weights_shuffled": shuffled,
        "gate_up_interleave": None,
    }


def _make_key(section, variant, shape, combo, extra_kwargs=None):
    token, model_dim, inter_dim, expert, topk = shape
    activation, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight, pads = combo
    kwargs = _variant_kwargs(variant, activation)
    kwargs["input_dtype"] = dtype
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    args = (
        int(token),
        int(model_dim),
        int(inter_dim),
        int(expert),
        int(topk),
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        bool(use_g1u1),
        activation,
        bool(doweight),
        int(pads[0]),
        int(pads[1]),
    )
    return Key(section, variant, args, tuple(sorted(kwargs.items())))


def _row_combo(row, arch_fp8):
    from aiter import ActivationType, QuantType

    return (
        _enum(ActivationType, row["act_type"]),
        parse_dtype(row["dtype"]),
        parse_dtype(row["q_dtype_a"], arch_fp8),
        parse_dtype(row["q_dtype_w"], arch_fp8),
        _enum(QuantType, row["q_type"]),
        bool(int(row["use_g1u1"])),
        bool(int(row["doweight_stage1"])),
        (0, 0),
    )


def _row_shape(row):
    return (
        row["token"],
        row["model_dim"],
        row["inter_dim"],
        row["expert"],
        row["topk"],
    )


def tuned_table():
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS

    return pd.read_csv(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)


def fhmoe_table():
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS

    path = AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE
    return path, pd.read_csv(path)


def tuned_keys(gfx, cu_num, arch_fp8, table=None):
    df = tuned_table() if table is None else table
    rows = df[(df["gfx"] == gfx) & (df["cu_num"] == cu_num)]
    keys = []
    for _, row in rows.iterrows():
        combo = _row_combo(row, arch_fp8)
        for variant in TUNED_VARIANTS:
            keys.append(_make_key("tuned", variant, _row_shape(row), combo))
    return keys


def fhmoe_keys(gfx, cu_num, arch_fp8):
    from aiter.ops.flydsl.moe_common import GateMode

    path, df = fhmoe_table()
    rows = df[(df["gfx"] == gfx) & (df["cu_num"] == cu_num)]
    keys = []
    for _, row in rows.iterrows():
        combo = list(_row_combo(row, arch_fp8))
        combo[-1] = (row["hidden_pad"], row["intermediate_pad"])
        extra = {
            "gate_mode": _enum(GateMode, row["gate_mode"]),
            "config_file": path,
        }
        keys.append(_make_key("fhmoe", "base", _row_shape(row), tuple(combo), extra))
    return keys


def grid_combos(arch_fp8, table=None):
    from aiter import ActivationType, QuantType

    df = tuned_table() if table is None else table
    cols = [
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    combos = set()
    for _, row in df[cols].drop_duplicates().iterrows():
        combos.add(_row_combo(row, arch_fp8))
    for act, dtype, qa, qw, qt, g1u1, dw in EXTRA_COMBOS:
        combos.add(
            (
                getattr(ActivationType, act),
                parse_dtype(dtype),
                parse_dtype(qa, arch_fp8),
                parse_dtype(qw, arch_fp8),
                getattr(QuantType, qt),
                g1u1,
                dw,
                (0, 0),
            )
        )
    return sorted(combos, key=lambda c: tuple(_short(x) for x in c))


def grid_keys(arch_fp8, table=None):
    keys = []
    for combo in grid_combos(arch_fp8, table):
        for model_dim, inter_dim, expert, topk in GRID_SHAPES:
            for token in GRID_M:
                for variant in GRID_VARIANTS:
                    shape = (token, model_dim, inter_dim, expert, topk)
                    keys.append(_make_key("grid", variant, shape, combo))
    return keys


_USE_G1U1, _ACTIVATION, _DOWEIGHT, _HIDDEN_PAD, _INTER_PAD = 9, 10, 11, 12, 13


def _feature_variants(arch_fp8):
    """{variant: (positional argument overrides by index, keyword overrides)}."""
    from aiter import ActivationType
    from aiter.ops.flydsl.moe_common import GateMode

    return {
        "num_local_tokens": ({}, {"has_num_local_tokens": True}),
        "act_scales": ({}, {"has_activation_scales": True}),
        "input_act_scales": (
            {},
            {"has_activation_scales": True, "_has_input_activation_scales": True},
        ),
        "fp8_input": ({}, {"input_dtype": arch_fp8, "has_activation_scales": True}),
        "fp8_input_no_scales": ({}, {"input_dtype": arch_fp8}),
        "scatter": ({}, {"has_stage2_scatter": True}),
        "unshuffled": ({}, {"is_shuffled": False, "opus_weights_shuffled": False}),
        "half_shuffled": ({}, {"opus_weights_shuffled": False}),
        "bias1": ({}, {"has_stage1_bias": True}),
        "bias2": ({}, {"has_stage2_bias": True}),
        "gate_interleave": ({}, {"gate_mode": GateMode.INTERLEAVE}),
        "gate_up_interleave": ({}, {"gate_up_interleave": True}),
        "limit0": ({}, {"swiglu_limit": 0.0}),
        "limit7": ({}, {"swiglu_limit": 7.0}),
        "exclude_mxmoe": ({}, {"_exclude_mxmoe": True}),
        "q_dtype_a2": ({}, {"q_dtype_a2": arch_fp8}),
        "doweight": ({_DOWEIGHT: True}, {}),
        "gelu": ({_ACTIVATION: ActivationType.Gelu}, {}),
        "no_g1u1": ({_USE_G1U1: False}, {}),
        "pad": ({_HIDDEN_PAD: 128, _INTER_PAD: 128}, {}),
    }


def feature_keys(base_keys, arch_fp8):
    """The `base` tuned and grid keys of `FEATURE_QUANT_TYPES`, one feature added each."""
    from aiter import QuantType

    quant_types = {getattr(QuantType, q) for q in FEATURE_QUANT_TYPES}
    variants = _feature_variants(arch_fp8)
    keys = []
    for key in base_keys:
        if (
            key.variant != "base"
            or key.section not in ("tuned", "grid")
            or key.args[8] not in quant_types
        ):
            continue
        for name, (positional, keyword) in variants.items():
            if key.section == "tuned" and name in TUNED_VARIANTS:
                continue
            args = list(key.args)
            for index, value in positional.items():
                args[index] = value
            kwargs = key.call_kwargs()
            if _ACTIVATION in positional:
                kwargs.update(_activation_kwargs(args[_ACTIVATION]))
            kwargs.update(keyword)
            keys.append(
                Key(key.section, name, tuple(args), tuple(sorted(kwargs.items())))
            )
    return keys


ACT_DTYPE_QUANT = (
    # (q_type, q_dtype_w)
    ("No", "bfloat16"),
    ("per_Tensor", "fp8"),
    ("per_Token", "fp8"),
    ("per_Token", "int8"),
    ("per_1x128", "fp8"),
    ("per_128x128", "fp8"),
    ("per_1x32", "float4_e2m1fn_x2"),
    ("per_1x32", "fp8"),
    ("per_1x32", "int4"),
    ("per_1x32", "uint32"),
)
ACT_DTYPE_M = (None, 1, 255, 256, 4096)


def activation_dtype_keys(arch_fp8):
    """Argument sets for `resolve_activation_dtype` (the activation quant policy)."""
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    keys = []
    for q_type, q_dtype_w in ACT_DTYPE_QUANT:
        for activation in ("Silu", "Gelu", "Swiglu", "Situv2", "Relu2"):
            for gate_mode in (GateMode.SEPARATED, GateMode.INTERLEAVE):
                for interleave in (None, True, False):
                    for M in ACT_DTYPE_M:
                        for hidden, a1 in (("bfloat16", False), ("fp8", True)):
                            keys.append(
                                {
                                    "quant_type": getattr(QuantType, q_type),
                                    "q_dtype_w": parse_dtype(q_dtype_w, arch_fp8),
                                    "activation": getattr(ActivationType, activation),
                                    "gate_mode": gate_mode,
                                    "M": M,
                                    "hidden_dtype": parse_dtype(hidden, arch_fp8),
                                    "has_a1_scale": a1,
                                    "gate_up_interleave": interleave,
                                }
                            )
    return keys
