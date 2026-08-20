#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT adapters for fused heterogeneous MoE (FHMoE)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

from aiter.aot.flydsl.common import cu_num_to_arch, job_identity
from aiter.fhmoe_config import DSV4_I384_FHMOE_MAX_TOKENS

_DSV4_MODEL_DIM = 7168
_DSV4_NATIVE_INTER_DIM = 384
_DSV4_TUNED_INTER_DIM = 512
_DSV4_EXPERTS = 385
_DSV4_TOPK = 7
_DSV4_SHARED_EXPERT_ID = 384
_DSV4_I384_2048_HEURISTIC_KERNELS = (
    "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui",
    "flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic",
)


def _normalized_enum(value: str) -> str:
    return value.strip().split(".")[-1].lower()


def _is_dsv4_profile_row(row: dict[str, str]) -> bool:
    return (
        row.get("gfx", "") == "gfx950"
        and int(row["cu_num"]) == 256
        and int(row["model_dim"]) == _DSV4_MODEL_DIM
        and int(row["expert"]) == _DSV4_EXPERTS
        and int(row["topk"]) == _DSV4_TOPK
        and _normalized_enum(row.get("act_type", "")) == "silu"
        and row.get("dtype", "") == "torch.bfloat16"
        and row.get("use_g1u1", "") == "1"
        and row.get("doweight_stage1", "") == "0"
        and _normalized_enum(row.get("q_type", "")) == "per_1x32"
        and "float8_e4m3fn" in row.get("q_dtype_a", "")
        and "float4_e2m1fn_x2" in row.get("q_dtype_w", "")
    )


def _is_dsv4_fhmoe_row(row: dict[str, str]) -> bool:
    """Return whether a tuned row can supply DSV4 FHMoE kernels."""
    return (
        _is_dsv4_profile_row(row)
        and int(row["inter_dim"]) in (_DSV4_NATIVE_INTER_DIM, _DSV4_TUNED_INTER_DIM)
        and "_gui" in row.get("kernelName1", "")
    )


def _target_inter_dims(row: dict[str, str]) -> tuple[int, ...]:
    """Return physical FHMoE dimensions covered by a tuned config row."""
    if not _is_dsv4_fhmoe_row(row):
        return ()
    inter_dim = int(row["inter_dim"])
    token_num = int(row["token"])
    if inter_dim == _DSV4_TUNED_INTER_DIM:
        if 1 <= token_num <= DSV4_I384_FHMOE_MAX_TOKENS:
            return (_DSV4_TUNED_INTER_DIM, _DSV4_NATIVE_INTER_DIM)
        return (_DSV4_TUNED_INTER_DIM,)
    if 1 <= token_num <= DSV4_I384_FHMOE_MAX_TOKENS:
        return (_DSV4_NATIVE_INTER_DIM,)
    return ()


def _row_job_keys(row: dict[str, str]) -> set[tuple[Any, ...]]:
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    stage1_name = row.get("kernelName1", "").strip()
    stage1_params = get_flydsl_kernel_params(stage1_name)
    stage1_out_dtype = stage1_params.get("out_dtype") if stage1_params else None
    stage1_fuse_quant = stage1_out_dtype if stage1_out_dtype in ("fp4", "fp8") else None
    prefix = (
        int(row["token"]),
        int(row["model_dim"]),
        int(row["inter_dim"]),
        int(row["expert"]),
        int(row["topk"]),
        bool(int(row.get("doweight_stage1", "0"))),
        int(row.get("cu_num", "0")),
        int(row.get("block_m", "0") or "0"),
        _normalized_enum(row.get("act_type", "")),
    )
    return {
        (*prefix, name, fuse_quant)
        for name, fuse_quant in (
            (stage1_name, None),
            (row.get("kernelName2", "").strip(), stage1_fuse_quant),
        )
        if name.startswith("flydsl_")
    }


def _job_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return (
        job["token_num"],
        job["model_dim"],
        job["inter_dim"],
        job["experts"],
        job["topk"],
        job["doweight_stage1"],
        job["cu_num"],
        job["block_m"],
        job["act"],
        job["kernel_name"],
        job.get("stage1_fuse_quant"),
    )


def _dsv4_i384_2048_heuristic_jobs() -> list[dict[str, Any]]:
    """Cover the native-I384 metadata heuristic used at the 2048 bucket."""
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    stage1_params = get_flydsl_kernel_params(_DSV4_I384_2048_HEURISTIC_KERNELS[0])
    if stage1_params is None:
        raise ValueError("Missing DSV4 I384 FHMoE stage-1 heuristic kernel")
    stage1_fuse_quant = stage1_params.get("out_dtype")
    if stage1_fuse_quant not in ("fp4", "fp8"):
        stage1_fuse_quant = None

    jobs = []
    for kernel_name in _DSV4_I384_2048_HEURISTIC_KERNELS:
        params = get_flydsl_kernel_params(kernel_name)
        if params is None:
            raise ValueError(f"Missing DSV4 I384 FHMoE heuristic {kernel_name}")
        job = {
            "kernel_name": kernel_name,
            "model_dim": _DSV4_MODEL_DIM,
            "inter_dim": _DSV4_NATIVE_INTER_DIM,
            "experts": _DSV4_EXPERTS,
            "topk": _DSV4_TOPK,
            "doweight_stage1": False,
            "cu_num": 256,
            "act": "silu",
            "enable_bias": False,
            "token_num": DSV4_I384_FHMOE_MAX_TOKENS,
            "block_m": 64,
            "shared_expert_id": _DSV4_SHARED_EXPERT_ID,
            **params,
        }
        if params["stage"] == 2:
            job["stage1_fuse_quant"] = stage1_fuse_quant
        jobs.append(job)
    return jobs


def extend_fhmoe_jobs(
    csv_path: str,
    ordinary_jobs: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """Append FHMoE variants for eligible ordinary jobs in ``csv_path``."""
    eligible_targets: dict[tuple[Any, ...], set[int]] = {}
    has_native_i384_2048 = False
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            target_inter_dims = _target_inter_dims(row)
            for key in _row_job_keys(row) if target_inter_dims else ():
                eligible_targets.setdefault(key, set()).update(target_inter_dims)
            has_native_i384_2048 |= (
                _is_dsv4_profile_row(row)
                and int(row["inter_dim"]) == _DSV4_NATIVE_INTER_DIM
                and int(row["token"]) == DSV4_I384_FHMOE_MAX_TOKENS
            )

    fhmoe_jobs = []
    for job in ordinary_jobs:
        if job.get("stage") not in (1, 2) or job.get("enable_bias", False):
            continue
        target_inter_dims = eligible_targets.get(_job_key(job), ())
        for inter_dim in target_inter_dims:
            fhmoe_job = {
                **job,
                "inter_dim": inter_dim,
                "shared_expert_id": job["experts"] - 1,
            }
            key = job_identity(fhmoe_job)
            if key in seen:
                continue
            seen.add(key)
            fhmoe_jobs.append(fhmoe_job)

    if has_native_i384_2048:
        for fhmoe_job in _dsv4_i384_2048_heuristic_jobs():
            key = job_identity(fhmoe_job)
            if key in seen:
                continue
            seen.add(key)
            fhmoe_jobs.append(fhmoe_job)

    return [*ordinary_jobs, *fhmoe_jobs]


def _shared_weight(device, n_in: int, k_in: int):
    import torch

    return torch.zeros((1, n_in, k_in), dtype=torch.uint8, device=device)


def _shared_scale(device, n_in: int, k_in: int):
    import torch

    rows = (n_in + 255) // 256 * 256
    cols = ((k_in + 255) // 256 * 256) // 32
    return torch.zeros((rows, cols), dtype=torch.uint8, device=device)


@dataclass(frozen=True)
class _FHMoEAOTBackend:
    shared_expert_id: int

    def build_stage1_args(
        self,
        out,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        out_scale_sorted,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        dev,
        bias=None,
        stream=None,
        swiglu_limit=float("inf"),
        pass_swiglu_limit: bool = True,
    ):
        from aiter.ops.flydsl.fhmoe import _s1_args_fhmoe

        shared_w = _shared_weight(dev, n_in, k_in)
        shared_w_scale = _shared_scale(dev, n_in, k_in)
        return _s1_args_fhmoe(
            out,
            a,
            w,
            a_scale,
            w_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            out_scale_sorted,
            token_num,
            n_in,
            k_in,
            size_expert_ids_in,
            dev,
            bias=bias,
            stream=stream,
            swiglu_limit=swiglu_limit,
            pass_swiglu_limit=pass_swiglu_limit,
            shared_w=shared_w.view(-1),
            shared_w_scale=shared_w_scale.view(-1),
        )

    def build_stage2_args(
        self,
        target,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        blocks,
        dev,
        bias=None,
        stream=None,
    ):
        from aiter.ops.flydsl.fhmoe import _s2_args_fhmoe

        shared_w = _shared_weight(dev, n_in, k_in)
        shared_w_scale = _shared_scale(dev, n_in, k_in)
        return _s2_args_fhmoe(
            target,
            a,
            w,
            a_scale,
            w_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            token_num,
            n_in,
            k_in,
            blocks,
            dev,
            bias=bias,
            stream=stream,
            shared_w=shared_w,
            shared_w_scale=shared_w_scale,
        )

    def compile_stage1(self, **kwargs):
        from aiter.ops.flydsl.fhmoe import compile_flydsl_fhmoe_stage1

        return compile_flydsl_fhmoe_stage1(
            **kwargs,
            shared_expert_id=self.shared_expert_id,
        )

    def compile_stage2(self, **kwargs):
        from aiter.ops.flydsl.fhmoe import compile_flydsl_fhmoe_stage2

        return compile_flydsl_fhmoe_stage2(
            **kwargs,
            shared_expert_id=self.shared_expert_id,
        )


def precompile_fhmoe_to_cache(
    *,
    experts: int,
    shared_expert_id: int,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    act: str = "silu",
    cu_num: int = 0,
    enable_bias: bool = False,
    **kwargs,
):
    """Precompile one heterogeneous MoE job through the shared AOT harness."""
    if shared_expert_id != experts - 1:
        raise ValueError(
            "FHMoE AOT expects the shared expert to be the final logical expert; "
            f"got {shared_expert_id=} for {experts=}"
        )
    if a_dtype != "fp8" or b_dtype != "fp4":
        raise ValueError(
            "FHMoE AOT supports routed FP8 activations and MXFP4 weights; "
            f"got {a_dtype=} and {b_dtype=}"
        )
    if enable_bias:
        raise ValueError("FHMoE AOT does not support expert bias")
    if act != "silu":
        raise ValueError(f"FHMoE AOT supports only SiLU, got {act=}")
    if cu_num_to_arch(cu_num) != "gfx950":
        raise ValueError(f"FHMoE AOT supports only gfx950, got {cu_num=}")

    from aiter.aot.flydsl.moe import _precompile_to_cache

    return _precompile_to_cache(
        experts=experts,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        act=act,
        cu_num=cu_num,
        enable_bias=enable_bias,
        _aot_backend=_FHMoEAOTBackend(shared_expert_id),
        **kwargs,
    )
