# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structured metadata for Opus MoE A8W4 stage2 kernels.

This module is intentionally torch-free. Runtime wrappers and csrc-side tuner /
codegen helpers can import it as the Python source of truth for A8W4 stage2
kids without pulling in JIT registration or opus arch dispatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

OPUS_A8W4_OUT_MODE_ATOMIC = 0
OPUS_A8W4_OUT_MODE_BF16 = 1
OPUS_A8W4_OUT_MODE_FP8 = 2

# Kid layout:
# - 2000-2099: K3 decode candidates, assigned contiguously.
# - 2100-2109: K5 direct atomic bring-up candidates.
# - 2110-2119: K5 route-output bring-up candidates.
OPUS_A8W4_KID_K3_ATOMIC_BM16_BN64_B3_WS2 = 2000
OPUS_A8W4_KID_K3_ROUTE_FP8_BM32_OCC4_RBN2240 = 2001
OPUS_A8W4_KID_K3_ROUTE_FP8_BM32_OCC5_RBN2304 = 2002
OPUS_A8W4_KID_K3_ROUTE_FP8_BM64_RBN3072 = 2003
OPUS_A8W4_KID_K3_ROUTE_FP8_BM64_RBN3584 = 2004
OPUS_A8W4_KID_K3_ROUTE_BF16_BM64_FULL_N7168 = 2005
OPUS_A8W4_KID_K5_ATOMIC_BM16_BN64_OCC6_B2_WS2 = 2100
OPUS_A8W4_KID_K5_ATOMIC_BM16_BN64_B3_WS2 = 2101
OPUS_A8W4_KID_K5_ROUTE_BF16_BM64_FULL_N7168 = 2110
OPUS_A8W4_KID_K5_ROUTE_FP8_BM64_RBN2816 = 2111

_OPUS_A8W4_REDUCE_BLOCK_N_RE = re.compile(r"_rbn(\d+)$")


@dataclass(frozen=True)
class OpusA8W4ShapeFamilyContract:
    name: str
    logical_inter_dim: int
    inter_dim_pad: int

    @property
    def effective_inter_dim(self) -> int:
        return self.logical_inter_dim - self.inter_dim_pad

    def matches(
        self,
        *,
        model_dim: int,
        inter_dim: int,
        expert: int,
        topk: int,
        block_n: Optional[int] = None,
    ) -> bool:
        model_dim = int(model_dim)
        inter_dim = int(inter_dim)
        expert = int(expert)
        topk = int(topk)
        if model_dim <= 0 or expert <= 0 or topk <= 0:
            return False
        if inter_dim != self.logical_inter_dim:
            return False
        if block_n is not None:
            block_n = int(block_n)
            if block_n <= 0 or model_dim % block_n != 0:
                return False
        return self.effective_inter_dim > 0


@dataclass(frozen=True)
class OpusA8W4KernelContract:
    name: str
    scale_group_logical_k: int
    fp4_values_per_byte: int
    vector_bytes: int
    default_block_m: int
    default_block_n: int
    default_cta_threads: int
    bk_logical: int
    mfma_m: int
    mfma_n: int
    mfma_k: int
    scale_groups_per_row_pack: int
    scale_words_per_group_pack: int
    c_vec: int
    c_values_per_atomic: int


@dataclass(frozen=True)
class OpusA8W4Stage2Instance:
    kid: int
    name: str
    out_mode: int
    block_m: int
    block_n: int
    block_k: int
    sort_block_m: int
    direct_atomic: bool
    pace_route_blocks_to_pow2: bool = False
    block_threads: int = 0
    min_blocks_per_cu: int = 0
    cachectl_b: int = 0
    cachectl_wscale: int = 0
    route_reduce: Optional[str] = None
    tuner_candidate: bool = True
    min_tuner_token: Optional[int] = None
    max_tuner_token: Optional[int] = None
    shape_family: str = "a8w4_decode_k3"
    kernel_contract: str = "gfx950_a8w4_decode_v1"
    mode_default: bool = False

    @property
    def route_out(self) -> bool:
        return self.out_mode != OPUS_A8W4_OUT_MODE_ATOMIC

    @property
    def route_out_fp8(self) -> bool:
        return self.out_mode == OPUS_A8W4_OUT_MODE_FP8

    @property
    def tuner_name(self) -> str:
        route_reduce = opus_a8w4_route_reduce(self.route_reduce)
        if route_reduce is None or route_reduce.suffix is None:
            return self.name
        return f"{self.name}_{route_reduce.suffix}"

    def tuner_params(self) -> dict[str, object]:
        route_reduce = opus_a8w4_route_reduce(self.route_reduce)
        params = {
            "kid": self.kid,
            "kernel_block_m": self.block_m,
            "sort_block_m": self.sort_block_m,
            "out_mode": self.out_mode,
            "route_out": self.route_out,
            "kernel_block_n": self.block_n,
            "shape_family": self.shape_family,
            "kernel_contract": self.kernel_contract,
        }
        if route_reduce is not None:
            params["route_reduce"] = route_reduce.name
            if route_reduce.suffix is not None:
                params["reduce_block_n"] = route_reduce.block_n
        return params

    def supports_tuner_token(self, token: Optional[int]) -> bool:
        if token is None:
            return True
        token = int(token)
        if self.min_tuner_token is not None and token < self.min_tuner_token:
            return False
        if self.max_tuner_token is not None and token > self.max_tuner_token:
            return False
        return True


@dataclass(frozen=True)
class OpusA8W4RouteReduceInstance:
    name: str
    block_n: int
    threads: int
    suffix: Optional[str] = None
    auto_model_dims: tuple[int, ...] = ()


OPUS_A8W4_K3_SHAPE_FAMILY_CONTRACT = OpusA8W4ShapeFamilyContract(
    name="a8w4_decode_k3",
    logical_inter_dim=512,
    inter_dim_pad=128,
)

# K5 is kept as a small generalized decode family. It is not part of the
# current DSV4 tuned target set, but it exercises the metadata/codegen path
# without requiring one-off shape-specific code.
OPUS_A8W4_K5_SHAPE_FAMILY_CONTRACT = OpusA8W4ShapeFamilyContract(
    name="a8w4_decode_k5",
    logical_inter_dim=768,
    inter_dim_pad=128,
)


OPUS_A8W4_DSV4_MODEL_DIM = 7168

OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT = OpusA8W4KernelContract(
    name="gfx950_a8w4_decode_v1",
    scale_group_logical_k=32,
    fp4_values_per_byte=2,
    vector_bytes=16,
    default_block_m=32,
    default_block_n=256,
    default_cta_threads=256,
    bk_logical=256,
    mfma_m=16,
    mfma_n=16,
    mfma_k=128,
    scale_groups_per_row_pack=8,
    scale_words_per_group_pack=64,
    c_vec=4,
    c_values_per_atomic=2,
)

OPUS_A8W4_SHAPE_FAMILY_CONTRACTS = {
    OPUS_A8W4_K3_SHAPE_FAMILY_CONTRACT.name: OPUS_A8W4_K3_SHAPE_FAMILY_CONTRACT,
    OPUS_A8W4_K5_SHAPE_FAMILY_CONTRACT.name: OPUS_A8W4_K5_SHAPE_FAMILY_CONTRACT,
}
OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT = OPUS_A8W4_K3_SHAPE_FAMILY_CONTRACT
OPUS_A8W4_ROUTE_REDUCE_INSTANCES = (
    OpusA8W4RouteReduceInstance(
        name="full_model_n7168",
        block_n=OPUS_A8W4_DSV4_MODEL_DIM,
        threads=448,
        auto_model_dims=(OPUS_A8W4_DSV4_MODEL_DIM,),
    ),
    OpusA8W4RouteReduceInstance(
        name="rbn2240",
        block_n=2240,
        threads=280,
        suffix="rbn2240",
    ),
    OpusA8W4RouteReduceInstance(
        name="rbn2304",
        block_n=2304,
        threads=288,
        suffix="rbn2304",
    ),
    OpusA8W4RouteReduceInstance(
        name="rbn2816",
        block_n=2816,
        threads=176,
        suffix="rbn2816",
    ),
    OpusA8W4RouteReduceInstance(
        name="rbn3072",
        block_n=3072,
        threads=384,
        suffix="rbn3072",
    ),
    OpusA8W4RouteReduceInstance(
        name="rbn3584",
        block_n=3584,
        threads=448,
        suffix="rbn3584",
    ),
)

OPUS_A8W4_ROUTE_REDUCE_BY_NAME = {
    inst.name: inst for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES
}
OPUS_A8W4_ROUTE_REDUCE_BY_SUFFIX = {
    inst.suffix: inst
    for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES
    if inst.suffix is not None
}


def _atomic_stage2_instance(
    *,
    kid: int,
    name: str,
    block_m: int,
    block_n: int,
    sort_block_m: int,
    shape_family: str,
    block_threads: int = 0,
    min_blocks_per_cu: int = 0,
    cachectl_b: int = 0,
    cachectl_wscale: int = 0,
    pace_route_blocks_to_pow2: bool = False,
    tuner_candidate: bool = True,
    min_tuner_token: Optional[int] = None,
    max_tuner_token: Optional[int] = None,
    mode_default: bool = False,
) -> OpusA8W4Stage2Instance:
    return OpusA8W4Stage2Instance(
        kid=kid,
        name=name,
        out_mode=OPUS_A8W4_OUT_MODE_ATOMIC,
        block_m=block_m,
        block_n=block_n,
        block_k=256,
        sort_block_m=sort_block_m,
        direct_atomic=True,
        pace_route_blocks_to_pow2=pace_route_blocks_to_pow2,
        block_threads=block_threads,
        min_blocks_per_cu=min_blocks_per_cu,
        cachectl_b=cachectl_b,
        cachectl_wscale=cachectl_wscale,
        tuner_candidate=tuner_candidate,
        min_tuner_token=min_tuner_token,
        max_tuner_token=max_tuner_token,
        shape_family=shape_family,
        mode_default=mode_default,
    )


def _route_stage2_instance(
    *,
    kid: int,
    name: str,
    out_mode: int,
    block_m: int,
    sort_block_m: int,
    shape_family: str,
    route_reduce: str,
    min_blocks_per_cu: int = 0,
    tuner_candidate: bool = True,
    min_tuner_token: Optional[int] = None,
    max_tuner_token: Optional[int] = None,
    mode_default: bool = False,
) -> OpusA8W4Stage2Instance:
    return OpusA8W4Stage2Instance(
        kid=kid,
        name=name,
        out_mode=out_mode,
        block_m=block_m,
        block_n=256,
        block_k=256,
        sort_block_m=sort_block_m,
        direct_atomic=False,
        min_blocks_per_cu=min_blocks_per_cu,
        route_reduce=route_reduce,
        tuner_candidate=tuner_candidate,
        min_tuner_token=min_tuner_token,
        max_tuner_token=max_tuner_token,
        shape_family=shape_family,
        mode_default=mode_default,
    )


_K3 = OPUS_A8W4_K3_SHAPE_FAMILY_CONTRACT.name
_K5 = OPUS_A8W4_K5_SHAPE_FAMILY_CONTRACT.name

OPUS_A8W4_K3_STAGE2_INSTANCES = (
    _atomic_stage2_instance(
        kid=OPUS_A8W4_KID_K3_ATOMIC_BM16_BN64_B3_WS2,
        name="opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_cache_b3_ws2",
        block_m=16,
        block_n=64,
        sort_block_m=16,
        block_threads=128,
        cachectl_b=3,
        cachectl_wscale=2,
        max_tuner_token=1024,
        shape_family=_K3,
        mode_default=True,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K3_ROUTE_FP8_BM32_OCC4_RBN2240,
        name="opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ4",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=32,
        sort_block_m=32,
        route_reduce="rbn2240",
        min_blocks_per_cu=4,
        min_tuner_token=512,
        max_tuner_token=4096,
        shape_family=_K3,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K3_ROUTE_FP8_BM32_OCC5_RBN2304,
        name="opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ5",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=32,
        sort_block_m=32,
        route_reduce="rbn2304",
        min_blocks_per_cu=5,
        min_tuner_token=512,
        max_tuner_token=4096,
        shape_family=_K3,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K3_ROUTE_FP8_BM64_RBN3072,
        name="opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=64,
        sort_block_m=64,
        route_reduce="rbn3072",
        shape_family=_K3,
        mode_default=True,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K3_ROUTE_FP8_BM64_RBN3584,
        name="opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=64,
        sort_block_m=64,
        route_reduce="rbn3584",
        shape_family=_K3,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K3_ROUTE_BF16_BM64_FULL_N7168,
        name="opus_moe2_afp8_wfp4_bf16_t64x256x256_sbm64",
        out_mode=OPUS_A8W4_OUT_MODE_BF16,
        block_m=64,
        sort_block_m=64,
        route_reduce="full_model_n7168",
        min_tuner_token=4096,
        shape_family=_K3,
        mode_default=True,
    ),
)

OPUS_A8W4_K5_STAGE2_INSTANCES = (
    _atomic_stage2_instance(
        kid=OPUS_A8W4_KID_K5_ATOMIC_BM16_BN64_OCC6_B2_WS2,
        name="opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_occ6_cache_b2_ws2_k5",
        block_m=16,
        block_n=64,
        sort_block_m=16,
        block_threads=128,
        min_blocks_per_cu=6,
        cachectl_b=2,
        cachectl_wscale=2,
        shape_family=_K5,
    ),
    _atomic_stage2_instance(
        kid=OPUS_A8W4_KID_K5_ATOMIC_BM16_BN64_B3_WS2,
        name="opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_cache_b3_ws2_k5",
        block_m=16,
        block_n=64,
        sort_block_m=16,
        block_threads=128,
        cachectl_b=3,
        cachectl_wscale=2,
        shape_family=_K5,
        mode_default=True,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K5_ROUTE_BF16_BM64_FULL_N7168,
        name="opus_moe2_afp8_wfp4_bf16_t64x256x256_sbm64_k5",
        out_mode=OPUS_A8W4_OUT_MODE_BF16,
        block_m=64,
        sort_block_m=64,
        route_reduce="full_model_n7168",
        shape_family=_K5,
        mode_default=True,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_K5_ROUTE_FP8_BM64_RBN2816,
        name="opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_k5",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=64,
        sort_block_m=64,
        route_reduce="rbn2816",
        shape_family=_K5,
        mode_default=True,
    ),
)

OPUS_A8W4_STAGE2_INSTANCES = tuple(
    sorted(
        (*OPUS_A8W4_K3_STAGE2_INSTANCES, *OPUS_A8W4_K5_STAGE2_INSTANCES),
        key=lambda inst: inst.kid,
    )
)


def _build_stage2_by_name() -> dict[str, OpusA8W4Stage2Instance]:
    by_name: dict[str, OpusA8W4Stage2Instance] = {}
    for inst in OPUS_A8W4_STAGE2_INSTANCES:
        route_reduce = OPUS_A8W4_ROUTE_REDUCE_BY_NAME.get(inst.route_reduce)
        if route_reduce is not None and route_reduce.suffix is not None:
            by_name[f"{inst.name}_{route_reduce.suffix}"] = inst
        by_name.setdefault(inst.name, inst)
    return by_name


OPUS_A8W4_STAGE2_BY_KID = {inst.kid: inst for inst in OPUS_A8W4_STAGE2_INSTANCES}
OPUS_A8W4_STAGE2_BY_NAME = _build_stage2_by_name()
OPUS_A8W4_TUNER_INSTANCES = tuple(
    inst for inst in OPUS_A8W4_STAGE2_INSTANCES if inst.tuner_candidate
)
OPUS_A8W4_SUPPORTED_BLOCK_MS = tuple(
    sorted({inst.block_m for inst in OPUS_A8W4_STAGE2_INSTANCES})
)


def _build_mode_default_by_family_mode_block_m() -> dict[tuple[str, int, int], int]:
    mode_defaults: dict[tuple[str, int, int], int] = {}
    for inst in OPUS_A8W4_STAGE2_INSTANCES:
        if not inst.mode_default:
            continue
        key = (inst.shape_family, inst.out_mode, inst.block_m)
        if key in mode_defaults:
            raise ValueError(
                "duplicate Opus A8W4 mode default for "
                f"shape_family={inst.shape_family}, out_mode={inst.out_mode}, "
                f"block_m={inst.block_m}"
            )
        mode_defaults[key] = inst.kid
    return mode_defaults


OPUS_A8W4_MODE_DEFAULT_BY_FAMILY_MODE_BLOCK_M = (
    _build_mode_default_by_family_mode_block_m()
)


def opus_a8w4_shape_family(
    name: str,
) -> Optional[OpusA8W4ShapeFamilyContract]:
    return OPUS_A8W4_SHAPE_FAMILY_CONTRACTS.get(str(name))


def opus_a8w4_route_reduce(
    name: Optional[str],
) -> Optional[OpusA8W4RouteReduceInstance]:
    if name is None:
        return None
    return OPUS_A8W4_ROUTE_REDUCE_BY_NAME.get(str(name))


def opus_a8w4_shape_family_for_shape(
    *,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    block_n: Optional[int] = None,
) -> Optional[OpusA8W4ShapeFamilyContract]:
    for contract in OPUS_A8W4_SHAPE_FAMILY_CONTRACTS.values():
        if contract.matches(
            model_dim=model_dim,
            inter_dim=inter_dim,
            expert=expert,
            topk=topk,
            block_n=block_n,
        ):
            return contract
    return None


def _opus_a8w4_stage2_instance(kid: int) -> Optional[OpusA8W4Stage2Instance]:
    return OPUS_A8W4_STAGE2_BY_KID.get(int(kid))


def opus_a8w4_base_name(name: str) -> str:
    return _OPUS_A8W4_REDUCE_BLOCK_N_RE.sub("", str(name).strip())


def opus_a8w4_reduce_block_n_from_name(name) -> Optional[int]:
    m = _OPUS_A8W4_REDUCE_BLOCK_N_RE.search(str(name).strip())
    if m is None:
        return None
    route_reduce = OPUS_A8W4_ROUTE_REDUCE_BY_SUFFIX.get(f"rbn{int(m.group(1))}")
    return int(m.group(1)) if route_reduce is None else route_reduce.block_n


def opus_a8w4_kid_from_name(name) -> Optional[int]:
    name = str(name).strip()
    inst = OPUS_A8W4_STAGE2_BY_NAME.get(name)
    if inst is None:
        inst = OPUS_A8W4_STAGE2_BY_NAME.get(opus_a8w4_base_name(name))
    return None if inst is None else inst.kid


def opus_a8w4_kid_name(kid: int) -> str:
    inst = _opus_a8w4_stage2_instance(kid)
    return "unknown" if inst is None else inst.name


def _require_a8w4_stage2_instance(kid: int) -> OpusA8W4Stage2Instance:
    inst = _opus_a8w4_stage2_instance(kid)
    if inst is None:
        raise ValueError(f"unsupported Opus A8W4 stage2 kid: {kid}")
    return inst


def opus_a8w4_decode_kid(
    out_mode: int,
    block_m: int,
    shape_family: str = OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT.name,
) -> int:
    out_mode = int(out_mode)
    block_m = int(block_m)
    shape_family = str(shape_family)
    kid = OPUS_A8W4_MODE_DEFAULT_BY_FAMILY_MODE_BLOCK_M.get(
        (shape_family, out_mode, block_m)
    )
    if kid is not None:
        return kid
    raise ValueError(
        "unsupported Opus A8W4 stage2 family/mode/block_m: "
        f"{shape_family}/{out_mode}/{block_m}"
    )


def opus_a8w4_kid_is_fp8(kid: int) -> bool:
    return _require_a8w4_stage2_instance(kid).route_out_fp8


def opus_a8w4_kid_uses_route(kid: int) -> bool:
    return _require_a8w4_stage2_instance(kid).route_out


def opus_a8w4_kid_block_m(kid: int) -> int:
    return _require_a8w4_stage2_instance(kid).block_m


def opus_a8w4_kid_reduce_block_n(kid: int) -> Optional[int]:
    route_reduce = opus_a8w4_route_reduce(
        _require_a8w4_stage2_instance(kid).route_reduce
    )
    return None if route_reduce is None else route_reduce.block_n


def opus_a8w4_supported_block_ms() -> tuple[int, ...]:
    return OPUS_A8W4_SUPPORTED_BLOCK_MS


def opus_a8w4_best_atomic_kid(
    token_num: int,
    shape_family: str = OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT.name,
) -> int:
    del token_num
    for block_m in (32, 16):
        kid = OPUS_A8W4_MODE_DEFAULT_BY_FAMILY_MODE_BLOCK_M.get(
            (str(shape_family), OPUS_A8W4_OUT_MODE_ATOMIC, block_m)
        )
        if kid is not None:
            return kid
    raise ValueError(f"unsupported Opus A8W4 atomic family: {shape_family}")


def get_opus_a8w4_stage2_kernels(
    shape_family: Optional[str] = None,
    token: Optional[int] = None,
) -> dict[str, dict[str, object]]:
    return {
        inst.tuner_name: inst.tuner_params()
        for inst in OPUS_A8W4_TUNER_INSTANCES
        if shape_family is None or inst.shape_family == shape_family
        if inst.supports_tuner_token(token)
    }
