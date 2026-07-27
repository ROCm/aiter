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
# - 2000-2099: A8W4 decode algorithm candidates. K is selected by runtime
#   effective inter dim at dispatch time, not encoded into the public kid.
OPUS_A8W4_KID_ATOMIC_BM16_BN64_B3_WS2 = 2000
OPUS_A8W4_KID_ROUTE_FP8_BM32_OCC4_RBN2240 = 2001
OPUS_A8W4_KID_ROUTE_FP8_BM32_OCC5_RBN2304 = 2002
OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072 = 2003
OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3584 = 2004
OPUS_A8W4_KID_ATOMIC_BM32_BN128_OCC1_B3_WS2 = 2005
OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072_B3 = 2006
OPUS_A8W4_KID_ROUTE_BF16_BM32_FULL_N7168_SMALL = 2007

_OPUS_A8W4_REDUCE_BLOCK_N_RE = re.compile(r"_rbn(\d+)$")


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
    sort_block_m: int
    direct_atomic: bool
    pace_route_blocks_to_pow2: bool = False
    block_threads: int = 0
    min_blocks_per_cu: int = 0
    cachectl_b: int = 0
    cachectl_wscale: int = 0
    route_reduce: Optional[str] = None
    min_tuner_token: Optional[int] = None
    max_tuner_token: Optional[int] = None
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
        }
        if route_reduce is not None:
            params["route_reduce"] = route_reduce.name
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


OPUS_A8W4_CODEGEN_SEED_EFFECTIVE_INTER_DIMS = (384,)


def _opus_a8w4_k_step_packed() -> int:
    k = OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
    if k.bk_logical % k.fp4_values_per_byte != 0:
        raise ValueError(
            "Opus A8W4 kernel contract requires bk_logical divisible by "
            "fp4_values_per_byte"
        )
    return k.bk_logical // k.fp4_values_per_byte


OPUS_A8W4_ROUTE_REDUCE_INSTANCES = (
    OpusA8W4RouteReduceInstance(
        name="full_model_n7168",
        block_n=7168,
        threads=448,
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
    block_threads: int = 0,
    min_blocks_per_cu: int = 0,
    cachectl_b: int = 0,
    cachectl_wscale: int = 0,
    pace_route_blocks_to_pow2: bool = False,
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
        sort_block_m=sort_block_m,
        direct_atomic=True,
        pace_route_blocks_to_pow2=pace_route_blocks_to_pow2,
        block_threads=block_threads,
        min_blocks_per_cu=min_blocks_per_cu,
        cachectl_b=cachectl_b,
        cachectl_wscale=cachectl_wscale,
        min_tuner_token=min_tuner_token,
        max_tuner_token=max_tuner_token,
        mode_default=mode_default,
    )


def _route_stage2_instance(
    *,
    kid: int,
    name: str,
    out_mode: int,
    block_m: int,
    sort_block_m: int,
    route_reduce: str,
    min_blocks_per_cu: int = 0,
    cachectl_b: int = 0,
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
        sort_block_m=sort_block_m,
        direct_atomic=False,
        min_blocks_per_cu=min_blocks_per_cu,
        cachectl_b=cachectl_b,
        route_reduce=route_reduce,
        min_tuner_token=min_tuner_token,
        max_tuner_token=max_tuner_token,
        mode_default=mode_default,
    )


OPUS_A8W4_DECODE_STAGE2_INSTANCES = (
    _atomic_stage2_instance(
        kid=OPUS_A8W4_KID_ATOMIC_BM16_BN64_B3_WS2,
        name="opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_cache_b3_ws2",
        block_m=16,
        block_n=64,
        sort_block_m=16,
        block_threads=128,
        cachectl_b=3,
        cachectl_wscale=2,
        max_tuner_token=1024,
        mode_default=True,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_ROUTE_FP8_BM32_OCC4_RBN2240,
        name="opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ4",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=32,
        sort_block_m=32,
        route_reduce="rbn2240",
        min_blocks_per_cu=4,
        min_tuner_token=8,
        max_tuner_token=4096,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_ROUTE_FP8_BM32_OCC5_RBN2304,
        name="opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ5",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=32,
        sort_block_m=32,
        route_reduce="rbn2304",
        min_blocks_per_cu=5,
        min_tuner_token=8,
        max_tuner_token=4096,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072,
        name="opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=64,
        sort_block_m=64,
        route_reduce="rbn3072",
        min_tuner_token=128,
        mode_default=True,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3584,
        name="opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=64,
        sort_block_m=64,
        route_reduce="rbn3584",
        min_tuner_token=128,
    ),
    _atomic_stage2_instance(
        kid=OPUS_A8W4_KID_ATOMIC_BM32_BN128_OCC1_B3_WS2,
        name="opus_moe2_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b3_ws2",
        block_m=32,
        block_n=128,
        sort_block_m=32,
        block_threads=128,
        min_blocks_per_cu=1,
        cachectl_b=3,
        cachectl_wscale=2,
        min_tuner_token=1,
        max_tuner_token=2048,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072_B3,
        name="opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_cache_b3",
        out_mode=OPUS_A8W4_OUT_MODE_FP8,
        block_m=64,
        sort_block_m=64,
        route_reduce="rbn3072",
        cachectl_b=3,
        min_tuner_token=128,
    ),
    _route_stage2_instance(
        kid=OPUS_A8W4_KID_ROUTE_BF16_BM32_FULL_N7168_SMALL,
        name="opus_moe2_afp8_wfp4_bf16_t32x256x256_sbm32_small",
        out_mode=OPUS_A8W4_OUT_MODE_BF16,
        block_m=32,
        sort_block_m=32,
        route_reduce="full_model_n7168",
        min_tuner_token=1,
        max_tuner_token=64,
    ),
)

OPUS_A8W4_STAGE2_INSTANCES = tuple(
    sorted(
        OPUS_A8W4_DECODE_STAGE2_INSTANCES,
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
OPUS_A8W4_SUPPORTED_BLOCK_MS = tuple(
    sorted({inst.block_m for inst in OPUS_A8W4_STAGE2_INSTANCES})
)


def _build_mode_default_by_mode_block_m() -> dict[tuple[int, int], int]:
    mode_defaults: dict[tuple[int, int], int] = {}
    for inst in OPUS_A8W4_STAGE2_INSTANCES:
        if not inst.mode_default:
            continue
        key = (inst.out_mode, inst.block_m)
        if key in mode_defaults:
            raise ValueError(
                "duplicate Opus A8W4 mode default for "
                f"out_mode={inst.out_mode}, block_m={inst.block_m}"
            )
        mode_defaults[key] = inst.kid
    return mode_defaults


OPUS_A8W4_MODE_DEFAULT_BY_MODE_BLOCK_M = _build_mode_default_by_mode_block_m()


def opus_a8w4_route_reduce(
    name: Optional[str],
) -> Optional[OpusA8W4RouteReduceInstance]:
    if name is None:
        return None
    return OPUS_A8W4_ROUTE_REDUCE_BY_NAME.get(str(name))


def opus_a8w4_effective_inter_dim(
    logical_inter_dim: int,
    inter_dim_pad: int,
) -> Optional[int]:
    logical_inter_dim = int(logical_inter_dim)
    inter_dim_pad = int(inter_dim_pad)
    if inter_dim_pad < 0 or logical_inter_dim <= inter_dim_pad:
        return None
    return logical_inter_dim - inter_dim_pad


def opus_a8w4_scale_cols_for_effective_inter_dim(effective_inter_dim: int) -> int:
    effective_inter_dim = int(effective_inter_dim)
    if effective_inter_dim <= 0:
        raise ValueError(
            f"effective_inter_dim must be positive, got {effective_inter_dim}"
        )
    k = OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
    k_step_packed = _opus_a8w4_k_step_packed()
    if effective_inter_dim % k_step_packed != 0:
        raise ValueError(
            "effective_inter_dim must be divisible by "
            f"K_STEP_PACKED={k_step_packed}, got {effective_inter_dim}"
        )
    k_tiles = effective_inter_dim // k_step_packed
    return ((k_tiles + 1) // 2) * k.scale_groups_per_row_pack


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
) -> int:
    out_mode = int(out_mode)
    block_m = int(block_m)
    kid = OPUS_A8W4_MODE_DEFAULT_BY_MODE_BLOCK_M.get((out_mode, block_m))
    if kid is not None:
        return kid
    raise ValueError(
        "unsupported Opus A8W4 stage2 mode/block_m: " f"{out_mode}/{block_m}"
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
) -> int:
    del token_num
    for block_m in (32, 16):
        kid = OPUS_A8W4_MODE_DEFAULT_BY_MODE_BLOCK_M.get(
            (OPUS_A8W4_OUT_MODE_ATOMIC, block_m)
        )
        if kid is not None:
            return kid
    raise ValueError("unsupported Opus A8W4 atomic kernel")


def get_opus_a8w4_stage2_kernels(
    token: Optional[int] = None,
) -> dict[str, dict[str, object]]:
    return {
        inst.tuner_name: inst.tuner_params()
        for inst in OPUS_A8W4_STAGE2_INSTANCES
        if inst.supports_tuner_token(token)
    }
