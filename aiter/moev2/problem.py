# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`MoeProblem`: what one MoE call must compute.

A problem is hashable and cheap: it holds formats, shapes and flags, never tensors or
tuning knobs. `MoeProblem.from_legacy_key` builds one from the arguments of the legacy
`get_2stage_cfgs`, and `legacy_key` gives them back, so v2 dispatch can be compared key
by key with the legacy code and can reproduce its log lines.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .shapes import get_padded_M

# `ScaleSpec.group`: rows x columns sharing one scale.
GROUP_NONE = ()
GROUP_PER_TENSOR = (0, 0)
GROUP_PER_TOKEN = (-1, 1)
GROUP_1X32 = (1, 32)
GROUP_1X128 = (1, 128)
GROUP_128X128 = (128, 128)
GROUP_256X128 = (256, 128)
GROUP_1024X128 = (1024, 128)
GROUP_IQ2R = ("iq2r_2bit",)  # codebook format, not a scale group

_GROUP_OF_QUANT = {
    QuantType.No: GROUP_NONE,
    QuantType.per_Tensor: GROUP_PER_TENSOR,
    QuantType.per_Token: GROUP_PER_TOKEN,
    QuantType.per_1x32: GROUP_1X32,
    QuantType.per_1x128: GROUP_1X128,
    QuantType.per_128x128: GROUP_128X128,
    QuantType.per_256x128: GROUP_256X128,
    QuantType.per_1024x128: GROUP_1024X128,
    QuantType.iq2r_2bit: GROUP_IQ2R,
}
_QUANT_OF_GROUP = {group: qt for qt, group in _GROUP_OF_QUANT.items()}
_GROUP_NAMES = {
    GROUP_NONE: "none",
    GROUP_PER_TENSOR: "per_tensor",
    GROUP_PER_TOKEN: "per_token",
    GROUP_1X32: "1x32",
    GROUP_1X128: "1x128",
    GROUP_128X128: "128x128",
    GROUP_256X128: "256x128",
    GROUP_1024X128: "1024x128",
    GROUP_IQ2R: "iq2r_2bit",
}
_UNQUANTIZED = (torch.bfloat16, torch.float16, torch.float32)


def _short(dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class ScaleSpec:
    """How a tensor is stored or quantized."""

    dtype: torch.dtype
    group: tuple[int, ...] = GROUP_NONE
    scale_dtype: torch.dtype | None = None

    @property
    def is_quantized(self) -> bool:
        return self.dtype not in _UNQUANTIZED

    def __str__(self):
        return f"{_short(self.dtype)}/{_GROUP_NAMES.get(self.group, self.group)}"


def scale_spec(quant_type: QuantType, dtype: torch.dtype) -> ScaleSpec:
    """The `ScaleSpec` a legacy `QuantType` gives a tensor of `dtype`.

    The group is kept even for bf16 tensors (a16w4 activations are `bf16/1x32`), so the
    legacy quant type can always be recovered from the weight spec.
    """
    quant_type = QuantType(quant_type)
    group = _GROUP_OF_QUANT[quant_type]
    if group == GROUP_NONE:
        scale = None
    elif group == GROUP_1X32:
        scale = dtypes.fp8_e8m0
    else:
        scale = dtypes.fp32
    return ScaleSpec(dtype, group, scale)


_ACT_BY_KIND = {name.lower(): m for name, m in ActivationType.__members__.items()}


@dataclass(frozen=True)
class ActSpec:
    kind: str  # ActivationType name in lower case: "silu", "gelu", "swiglu", ...
    swiglu_limit: float | None = None
    situ_beta: float | None = None
    situ_linear_beta: float | None = None

    @property
    def activation(self) -> ActivationType:
        return _ACT_BY_KIND[self.kind]


@dataclass(frozen=True)
class WeightLayout:
    shuffle: str  # "16x16" (w1 and w2 preshuffled), "mixed" (one of them), "none"
    gate_mode: str  # GateMode value: "separated", "interleave"
    hidden_pad: int = 0
    intermediate_pad: int = 0
    g1u1: bool = True  # w1 holds gate and up rows
    gate_up_interleave: bool | None = None  # w1.gate_up_interleave when tagged


@dataclass(frozen=True)
class SharedExpertSpec:
    placement: str  # "fused_row", "fused_row_per_rank", "external_partial", "separate"
    expert_id: int = -1
    w1: ScaleSpec | None = None
    w2: ScaleSpec | None = None
    clamp: bool = True


@dataclass(frozen=True)
class TailSpec:
    kind: str  # "add_rmsnorm_quant"
    norm: str  # "rms" or "gemma_rms"
    eps: float
    out_fmt: ScaleSpec
    keep_bf16_rows: bool = False


@dataclass(frozen=True)
class ParallelSpec:
    mode: str  # "ep" or "tp"
    world_size: int
    comm: str  # "a2a", "ag_rs", "ar"
    backend: str
    max_tokens_per_rank: int
    experts_per_rank: int
    input_replication: int = 1
    comm_precision: str = "kernel_default"


@dataclass(frozen=True)
class Shape:
    m: int  # the token count dispatch keys on (the tuning bucket for runtime calls)
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    logical_experts: int | None = None

    @property
    def m_bucket(self) -> int:
        return get_padded_M(self.m)


class FrozenMap(Mapping):
    """An immutable, hashable mapping (`MoeProblem.extras`)."""

    __slots__ = ("_hash", "_items")

    def __init__(self, items: Mapping | None = None):
        self._items = dict(sorted((items or {}).items()))
        self._hash = hash(tuple(self._items.items()))

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, FrozenMap) and self._items == other._items

    def __repr__(self):
        return f"FrozenMap({self._items!r})"


# `features`: optional inputs and modes a segment must declare to be chosen.
FEATURES = frozenset(
    {
        "bias1",
        "bias2",
        "doweight_stage1",
        "ep_mask",  # expert_mask: some routed experts live on other ranks
        "num_local_tokens",
        "scatter",  # stage2 writes into peers' combine buffers (stage2_scatter)
        "act_scales",  # caller passes a1_scale / a2_scale
        "a1_scale",  # caller passes a1_scale
        "global_scale",  # NVFP4 per-expert global weight scales
    }
)


@dataclass(frozen=True)
class MoeProblem:
    arch: str
    cu_num: int
    shape: Shape
    out_dtype: torch.dtype
    act_in: ScaleSpec  # activation format into GEMM1 (decided by policy.py)
    w1: ScaleSpec
    w2: ScaleSpec
    activation: ActSpec
    layout: WeightLayout | None  # None only in engine templates
    in_dtype: torch.dtype | None = None  # hidden_states dtype, when known
    inter: ScaleSpec | None = None  # GEMM2 input format when the caller fixes it
    shared: SharedExpertSpec | None = None
    parallel: ParallelSpec | None = None
    tail: TailSpec | None = None
    graph_capture: bool = False
    features: frozenset = frozenset()
    extras: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        unknown = self.features - FEATURES
        if unknown:
            raise ValueError(f"unknown MoeProblem features: {sorted(unknown)}")

    # Derived predicates. Adding one changes no key.

    @property
    def quant_type(self) -> QuantType:
        """The legacy `QuantType` (after `quant_remap`) the weights are quantized with."""
        return _QUANT_OF_GROUP[self.w1.group]

    @property
    def is_g1u1(self) -> bool:
        return self.layout.g1u1

    @property
    def is_ep(self) -> bool:
        return "ep_mask" in self.features

    @property
    def any_shuffled(self) -> bool:
        return self.layout.shuffle != "none"

    @property
    def fully_shuffled(self) -> bool:
        return self.layout.shuffle == "16x16"

    @property
    def gate_mode(self) -> GateMode:
        return GateMode(self.layout.gate_mode)

    @property
    def rows_per_expert(self) -> int:
        s = self.shape
        return s.m * s.topk // s.experts

    @property
    def is_a16w4_situv2(self) -> bool:
        """bf16 activations x MXFP4 weights with SiTUv2 (the format, any layout)."""
        return (
            self.quant_type == QuantType.per_1x32
            and self.w1.dtype == dtypes.fp4x2
            and self.act_in.dtype == dtypes.bf16
            and self.activation.kind == "situv2"
        )

    @property
    def a16w4_situv2_kernel_ok(self) -> bool:
        """`is_a16w4_situv2` with what the ported kernel also needs (out dtype, layout)."""
        return (
            self.is_a16w4_situv2
            and self.out_dtype in (dtypes.bf16, dtypes.fp16)
            and self.any_shuffled
            and self.is_g1u1
            and "doweight_stage1" not in self.features
        )

    def with_features(self, add=(), remove=()) -> "MoeProblem":
        return replace(self, features=(self.features | set(add)) - set(remove))

    def __str__(self):
        s = self.shape
        layout = self.layout
        shuffled = {
            "16x16": "shuffled",
            "mixed": "partly shuffled",
            "none": "unshuffled",
        }
        parts = [
            (
                f"{self.arch} M={s.m}(bucket {s.m_bucket}) H={s.model_dim} "
                f"I={s.inter_dim} E={s.experts} k={s.topk}"
            ),
            (
                f"act_in={self.act_in} w={self.w1} {self.activation.kind} "
                f"{layout.gate_mode} {shuffled[layout.shuffle]}"
            ),
        ]
        if self.features:
            parts.append("features=" + ",".join(sorted(self.features)))
        return "\n            ".join(parts)

    # Legacy key mapping.

    @classmethod
    def from_legacy_key(
        cls,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        is_shuffled=True,
        gate_mode=GateMode.SEPARATED.value,
        is_ep=False,
        has_stage1_bias=False,
        has_stage2_bias=False,
        situ_beta=1.0,
        situ_linear_beta=1.0,
        swiglu_limit=None,
        opus_weights_shuffled=None,
        gate_up_interleave=None,
        config_file=None,
        _disable_inline_sort=False,
        q_dtype_a2=None,
        input_dtype=None,
        has_stage2_scatter=False,
        has_activation_scales=False,
        has_num_local_tokens=False,
        _exclude_mxmoe=False,
        _has_input_activation_scales=False,
        has_global_scale=False,
        *,
        arch: str,
        cu_num: int,
    ) -> "MoeProblem":
        """A problem from the arguments of the legacy `get_2stage_cfgs`, same names."""
        fully = is_shuffled if opus_weights_shuffled is None else opus_weights_shuffled
        if fully and not is_shuffled:
            raise ValueError("both weights shuffled but neither is marked is_shuffled")
        shuffle = "16x16" if fully else ("mixed" if is_shuffled else "none")
        flags = {
            "ep_mask": is_ep,
            "bias1": has_stage1_bias,
            "bias2": has_stage2_bias,
            "doweight_stage1": doweight_stage1,
            "scatter": has_stage2_scatter,
            "act_scales": has_activation_scales,
            "num_local_tokens": has_num_local_tokens,
            "a1_scale": _has_input_activation_scales,
            "global_scale": has_global_scale,
        }
        extras = {}
        if config_file is not None:
            extras["tuned_table"] = config_file
        if _disable_inline_sort:
            extras["disable_inline_sort"] = True
        if _exclude_mxmoe:
            extras["exclude_mxmoe"] = True
        q_type = QuantType(q_type)
        return cls(
            arch=arch,
            cu_num=cu_num,
            shape=Shape(token, model_dim, inter_dim, expert, topk),
            out_dtype=dtype,
            act_in=scale_spec(q_type, q_dtype_a),
            w1=scale_spec(q_type, q_dtype_w),
            w2=scale_spec(q_type, q_dtype_w),
            activation=ActSpec(
                ActivationType(activation).name.lower(),
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            ),
            layout=WeightLayout(
                shuffle=shuffle,
                gate_mode=GateMode(gate_mode).value,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                g1u1=bool(use_g1u1),
                gate_up_interleave=gate_up_interleave,
            ),
            in_dtype=input_dtype,
            inter=None if q_dtype_a2 is None else scale_spec(q_type, q_dtype_a2),
            features=frozenset(name for name, on in flags.items() if on),
            extras=FrozenMap(extras),
        )

    def legacy_key(self) -> tuple[tuple, dict]:
        """`(args, kwargs)` for the legacy `get_2stage_cfgs`; inverse of `from_legacy_key`."""
        s, layout, act = self.shape, self.layout, self.activation
        args = (
            s.m,
            s.model_dim,
            s.inter_dim,
            s.experts,
            s.topk,
            self.out_dtype,
            self.act_in.dtype,
            self.w1.dtype,
            self.quant_type,
            layout.g1u1,
            act.activation,
            "doweight_stage1" in self.features,
            layout.hidden_pad,
            layout.intermediate_pad,
        )
        kwargs = {
            "is_shuffled": self.any_shuffled,
            "gate_mode": self.gate_mode,
            "is_ep": self.is_ep,
            "has_stage1_bias": "bias1" in self.features,
            "has_stage2_bias": "bias2" in self.features,
            "situ_beta": act.situ_beta,
            "situ_linear_beta": act.situ_linear_beta,
            "swiglu_limit": act.swiglu_limit,
            "opus_weights_shuffled": self.fully_shuffled,
            "gate_up_interleave": layout.gate_up_interleave,
            "config_file": self.extras.get("tuned_table"),
            "_disable_inline_sort": bool(self.extras.get("disable_inline_sort")),
            "q_dtype_a2": None if self.inter is None else self.inter.dtype,
            "input_dtype": self.in_dtype,
            "has_stage2_scatter": "scatter" in self.features,
            "has_activation_scales": "act_scales" in self.features,
            "has_num_local_tokens": "num_local_tokens" in self.features,
            "_exclude_mxmoe": bool(self.extras.get("exclude_mxmoe")),
            "_has_input_activation_scales": "a1_scale" in self.features,
            "has_global_scale": "global_scale" in self.features,
        }
        return args, kwargs

    def log_keys(self) -> tuple:
        """The `keys` tuple legacy dispatch prints in its log lines and tuned lookups."""
        s = self.shape
        keys = (
            self.arch,
            self.cu_num,
            s.m,
            s.model_dim,
            s.inter_dim,
            s.experts,
            s.topk,
            self.activation.activation,
            str(self.out_dtype),
            str(self.act_in.dtype),
            str(self.w1.dtype),
            str(self.quant_type),
            self.is_g1u1,
            "doweight_stage1" in self.features,
        )
        table = self.extras.get("tuned_table")
        if table is not None:
            keys += (
                s.experts - 1,
                self.layout.hidden_pad,
                self.layout.intermediate_pad,
                str(self.gate_mode),
            )
        return keys
