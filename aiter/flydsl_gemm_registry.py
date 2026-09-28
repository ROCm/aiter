# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Registry of FlyDSL GEMM kernel families.

A family is one kernel implementation on a set of GPUs. A tuned CSV row names a
kernel, and the start of that name says which family it belongs to. The family
knows how to parse the name, check the row against the call being dispatched,
launch the kernel, offer tuning candidates and describe an AOT job. The
dispatcher, the tuner and the AOT builder only ask this registry, so adding a
family, or a GPU to an existing one, does not touch them.

Every family belongs to one op table: the tuned-CSV family it is dispatched
from. Today there is one, ``"a16w16"`` (the ``*bf16_tuned_gemm*`` files).
Quantized families (a8w8, a4w4, ...) are expected to register into their own
op table; ``GemmProblem`` already carries the scale kind and weight layout they
need, so they do not have to change the lookup or the launch signature.

Nothing here imports FlyDSL at module import time: callers check a registry
even when FlyDSL is not installed, and each family imports its kernels lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "A16W16",
    "Candidate",
    "FlydslGemmFamily",
    "GemmProblem",
    "families",
    "family_for_kernel",
    "get_family",
    "is_exact_m_decode_row",
    "register",
]

# Op table of the BF16/FP16 dense GEMM (the ``*bf16_tuned_gemm*`` CSV files).
A16W16 = "a16w16"

# The CSV libtype shared by every FlyDSL family.
FLYDSL_LIBTYPE = "flydsl"


@dataclass(frozen=True)
class GemmProblem:
    """What a caller asks for, independent of which kernel answers it."""

    m: int
    n: int
    k: int
    arch: str
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    has_bias: bool = False
    # Quantized op tables need these; A16W16 problems are always "none"/"plain".
    scale_kind: str = "none"  # none | per_tensor | per_token | block
    weight_layout: str = "plain"  # plain | preshuffle
    cu_num: int | None = None


@dataclass(frozen=True)
class Candidate:
    """One kernel a family offers the tuner for a problem."""

    kernel_name: str
    solidx: int
    split_k: int
    params: Any


@dataclass(frozen=True)
class FlydslGemmFamily:
    """One kernel implementation on a set of GPUs.

    ``parse``         kernel name -> params, or None if the name is not this family's
    ``check_row``     params, problem, CSV row -> None if the row fits the call,
                      otherwise a reason (logged by the dispatcher)
    ``launch``        run the kernel: (inp, weights, params, *, bias, out_dtype,
                      scale_a, scale_b, scale_c, bpreshuffle, out) -> Tensor
    ``candidates``    problem, policy -> tuning candidates; None means the family
                      has no tuner support yet (kept visible rather than hidden)
    ``aot_job``       CSV row, kernel name -> AOT job dict (see aiter.aot.flydsl.gemm)
    ``exact_m``       rows are exact-M specializations and are never reused for a
                      padded M (a padded row is skipped silently, not warned about)

    Every family's rows carry ``libtype = "flydsl"`` in the tuned CSV; the kernel
    name prefix, not the libtype, says which family a row belongs to.
    """

    name: str
    op: str
    archs: frozenset[str]
    prefix: str
    parse: Callable[[str], Any]
    check_row: Callable[[Any, GemmProblem, dict], str | None]
    launch: Callable[..., torch.Tensor]
    candidates: Callable[[GemmProblem, str], list[Candidate]] | None = None
    aot_job: Callable[[dict, str], dict] | None = None
    exact_m: bool = False
    tuner_tolerance: tuple[float, float] | None = None  # (rtol, atol); None = default


_FAMILIES: list[FlydslGemmFamily] = []


def register(family: FlydslGemmFamily) -> FlydslGemmFamily:
    """Add a family.

    Two families of one op may use the same prefix only on disjoint archs. A
    longer prefix (``flydsl_decode_fp8_`` next to ``flydsl_decode_``) is fine:
    lookup takes the longest prefix that matches.
    """
    for other in _FAMILIES:
        if other.name == family.name:
            raise ValueError(f"FlyDSL GEMM family {family.name!r} already registered")
        if (
            other.op == family.op
            and other.archs & family.archs
            and other.prefix == family.prefix
        ):
            raise ValueError(
                f"FlyDSL GEMM families {other.name!r} and {family.name!r} "
                f"overlap on {sorted(other.archs & family.archs)}"
            )
    _FAMILIES.append(family)
    return family


def families(op: str, arch: str | None = None) -> list[FlydslGemmFamily]:
    return [f for f in _FAMILIES if f.op == op and (arch is None or arch in f.archs)]


def get_family(name: str) -> FlydslGemmFamily:
    for f in _FAMILIES:
        if f.name == name:
            return f
    raise KeyError(f"unknown FlyDSL GEMM family {name!r}")


def family_for_kernel(
    op: str, kernel_name: str, arch: str | None
) -> FlydslGemmFamily | None:
    """The family whose prefix starts ``kernel_name``; ``arch=None`` ignores arch."""
    matches = [
        f
        for f in families(op, arch)
        if kernel_name and kernel_name.startswith(f.prefix)
    ]
    return max(matches, key=lambda f: len(f.prefix)) if matches else None


def is_exact_m_decode_row(row: dict | None, arch: str | None = None) -> bool:
    """True when a tuned A16W16 row selects a FlyDSL exact-M decode kernel.

    For callers outside aiter (vLLM) that must not depend on libtype labels.
    """
    if not row or row.get("libtype") != FLYDSL_LIBTYPE:
        return False
    family = family_for_kernel(A16W16, row.get("kernelName") or "", arch)
    return family is not None and family.exact_m


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gemm_kernels():
    from aiter.ops.flydsl import gemm_kernels

    return gemm_kernels


def _row_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("", "0", "false", "no"):
        return False
    if normalized in ("1", "true", "yes"):
        return True
    raise ValueError(f"Expected True/False, got {value!r}")


def _no_scaling(scale_a, scale_b, scale_c) -> bool:
    return scale_a is None and scale_b is None and scale_c is None


# ---------------------------------------------------------------------------
# hgemm: gfx950 A16W16 GEMM (Yutao Xu) and gfx1250 A16W16 GEMM (Omar Muhammad)
# ---------------------------------------------------------------------------


def _hgemm_parse(kernel_name: str):
    try:
        return _gemm_kernels().get_flydsl_hgemm_kernel_params(kernel_name)
    except ImportError:
        return None


def _hgemm_check_row(params, problem: GemmProblem, row: dict) -> str | None:
    if params.get("target_gfx") != problem.arch:
        return f"kernel targets {params.get('target_gfx')}, running on {problem.arch}"
    name_n, name_k = params.get("n"), params.get("k")
    if (
        name_n is not None
        and name_k is not None
        and (name_n, name_k)
        != (
            problem.n,
            problem.k,
        )
    ):
        return f"kernel shape N={name_n} K={name_k} does not match the call"
    if int(row["splitK"]) != params.get("split_k"):
        return f"row splitK={row['splitK']} does not match kernel split_k"
    return None


def _hgemm_launch(
    inp,
    weights,
    params,
    *,
    bias=None,
    out_dtype=None,
    scale_a=None,
    scale_b=None,
    scale_c=None,
    bpreshuffle=False,
    out=None,
):
    assert _no_scaling(
        scale_a, scale_b, scale_c
    ), "FlyDSL hgemm does not support scaling yet."
    fused_bias = None
    if (
        bias is not None
        and (out_dtype is None or out_dtype == inp.dtype)
        and bias.dtype == inp.dtype
    ):
        fused_bias = bias
    extra = {} if out is None else {"out": out}
    result = _gemm_kernels().flydsl_hgemm(
        inp,
        weights,
        bias=fused_bias,
        block_m=params["block_m"],
        block_n=params["block_n"],
        block_k=params["block_k"],
        split_k=params["split_k"],
        m_waves=params["m_waves"],
        n_waves=params["n_waves"],
        k_waves=params["k_waves"],
        stages=params["stages"],
        group_m=params["group_m"],
        policy="ht" if params["use_half_tile_interleaved"] else "ft",
        out_dtype=out_dtype,
        **extra,
    )
    if bias is not None and fused_bias is None:
        result = result.to(bias.dtype) + bias
    if out_dtype is not None and result.dtype != out_dtype:
        result = result.to(out_dtype)
    return result


def _hgemm_gfx950_candidates(problem: GemmProblem, policy: str) -> list[Candidate]:
    del policy  # the hgemm policy has a single catalog
    if (
        problem.scale_kind != "none"
        or problem.weight_layout != "plain"
        or problem.in_dtype != torch.bfloat16
    ):
        return []
    from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_configs

    gemm_kernels = _gemm_kernels()
    fused_bias = bool(problem.has_bias and problem.out_dtype == torch.bfloat16)
    configs = get_flydsl_a16w16_configs(
        problem.m,
        problem.n,
        problem.k,
        torch.bfloat16,
        problem.out_dtype,
        fused_bias,
    )
    by_name = {
        gemm_kernels.flydsl_hgemm_kernel_name(
            dtype=torch.bfloat16,
            out_dtype=problem.out_dtype,
            config=config,
            has_bias=fused_bias,
        ): config
        for config in configs
    }
    return [
        Candidate(name, idx, by_name[name]["split_k"], dict(by_name[name]))
        for idx, name in enumerate(sorted(by_name))
    ]


def _hgemm_aot_job(row: dict, kernel_name: str) -> dict:
    params = _hgemm_parse(kernel_name)
    if params is None:
        raise ValueError(f"Unknown FlyDSL HGEMM kernel name: {kernel_name}")
    params = dict(params)
    gfx = (row.get("gfx") or "").strip()
    if params["target_gfx"] == "gfx1250":
        params["kind"] = "a16w16_gfx1250"
    else:
        params["kind"] = "hgemm"
        if int(row.get("splitK", "0")) != params["split_k"]:
            raise ValueError("FlyDSL HGEMM CSV splitK does not match kernel name")
        if params["target_gfx"] != gfx:
            raise ValueError("FlyDSL HGEMM CSV architecture does not match kernel name")
    return params


# ---------------------------------------------------------------------------
# decode: exact-M BF16 decode GEMM for M=1..5 (Sami Remes)
# ---------------------------------------------------------------------------

DECODE_MAX_M = 5


def _decode_parse(kernel_name: str):
    try:
        arch, m, n, k, config, has_bias = _gemm_kernels().parse_gemm_decode_kernel_name(
            kernel_name
        )
    except (ImportError, ValueError):
        return None
    return {
        "arch": arch,
        "m": m,
        "n": n,
        "k": k,
        "config": config,
        "has_bias": has_bias,
    }


def _decode_check_row(params, problem: GemmProblem, row: dict) -> str | None:
    del row
    if (problem.in_dtype, problem.out_dtype) != (torch.bfloat16, torch.bfloat16):
        return f"decode is BF16 only, call is {problem.in_dtype}->{problem.out_dtype}"
    if problem.scale_kind != "none" or problem.weight_layout != "plain":
        return "decode supports neither scaling nor preshuffled weights"
    got = (params["arch"], params["m"], params["n"], params["k"], params["has_bias"])
    want = (problem.arch, problem.m, problem.n, problem.k, problem.has_bias)
    if got != want:
        return f"kernel identity {got} does not match the call {want}"
    return None


def _decode_launch(
    inp,
    weights,
    params,
    *,
    bias=None,
    out_dtype=None,
    scale_a=None,
    scale_b=None,
    scale_c=None,
    bpreshuffle=False,
    out=None,
):
    """Launch an exact-M/N/K decode kernel."""
    if not _no_scaling(scale_a, scale_b, scale_c):
        raise ValueError("FlyDSL decode does not support scaling")
    if bpreshuffle:
        raise ValueError("FlyDSL decode does not support preshuffled weights")
    if (out_dtype or inp.dtype) != torch.bfloat16:
        raise ValueError("FlyDSL decode requires BF16 output")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    expected = (
        get_gfx_runtime(),
        int(inp.shape[0]),
        int(weights.shape[0]),
        int(inp.shape[1]),
    )
    kernel = (params["arch"], params["m"], params["n"], params["k"])
    if kernel != expected:
        raise ValueError(
            "FlyDSL decode tuned kernel does not match the runtime "
            f"exact identity: kernel={kernel}, runtime={expected}"
        )
    if params["has_bias"] != (bias is not None):
        raise ValueError("FlyDSL decode kernel bias identity does not match launch")
    if out is None:
        out = torch.empty(
            (params["m"], params["n"]), dtype=torch.bfloat16, device=inp.device
        )
    # Looked up on the module at call time so tests can intercept the launcher.
    return _gemm_kernels().gemm_decode_bf16(
        inp, weights, out, params["config"], bias=bias
    )


def _round_robin_representatives(items, *, limit, bucket_key, priority_key):
    buckets = {}
    for item in items:
        buckets.setdefault(bucket_key(item), []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=priority_key)

    keys = sorted(buckets, key=str)
    offsets = {key: 0 for key in keys}
    selected = []
    while keys and len(selected) < limit:
        next_keys = []
        for key in keys:
            offset = offsets[key]
            bucket = buckets[key]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[key] = offset + 1
            if offsets[key] < len(bucket):
                next_keys.append(key)
            if len(selected) >= limit:
                break
        keys = next_keys
    return selected


def _bounded_decode_configs(configs: Iterable, limit: int = 12) -> list:
    """Pick a small but representative set of decode candidates.

    The catalog enumerates every Wave configuration before any BlockMFMA one,
    so a plain prefix (`list(configs)[:limit]`) is not a sample: it is always
    Wave-only, and BlockMFMA is never timed. Measured on gfx950, that excluded
    38% of the catalog on all 84 shape/M cells; on gfx942 the excluded family
    turned out to win 81 of 84 cells.

    Bucketing by family first and round-robining across buckets keeps the same
    candidate budget while guaranteeing both families are represented.
    """
    from aiter.ops.flydsl.gemm_kernels import WaveDecodeConfig

    def bucket(config):
        if isinstance(config, WaveDecodeConfig):
            return ("wave", config.contraction.value)
        return ("block", config.activation_source.value, bool(config.persistent_n))

    def priority(config):
        if isinstance(config, WaveDecodeConfig):
            return (
                -config.m_per_wave,
                config.n_per_wave != 1,
                config.kvec != 8,
                config.prefetch_depth != 1,
                config.waves_per_eu != 2,
                config.b_cache_modifier != 0,
                config.reduction.value != "dpp",
                repr(config),
            )
        return (
            config.waves_per_workgroup != 8,
            config.columns_per_wave != 1,
            config.b_load_width != 8,
            config.k_unroll != 2,
            config.waves_per_eu != 2,
            config.workgroups_per_cu != 1,
            config.b_cache_modifier != 0,
            repr(config),
        )

    return _round_robin_representatives(
        list(configs), limit=limit, bucket_key=bucket, priority_key=priority
    )


def _decode_candidates(problem: GemmProblem, policy: str) -> list[Candidate]:
    if (
        not 1 <= problem.m <= DECODE_MAX_M
        or problem.scale_kind != "none"
        or problem.weight_layout != "plain"
        or problem.in_dtype != torch.bfloat16
        or problem.out_dtype != torch.bfloat16
    ):
        return []
    from aiter.jit.utils.chip_info import get_gfx_runtime

    runtime_arch = get_gfx_runtime()
    if problem.arch != runtime_arch:
        raise ValueError(
            f"FlyDSL decode tuner row targets {problem.arch}, "
            f"but the runtime device is {runtime_arch}"
        )
    gemm_kernels = _gemm_kernels()
    configs = list(
        gemm_kernels.iter_gemm_decode_configs(
            problem.m, problem.n, problem.k, problem.arch, num_cus=problem.cu_num
        )
    )
    if policy == "bounded":
        configs = _bounded_decode_configs(configs)
    candidates = []
    for solidx, config in enumerate(configs):
        name = gemm_kernels.gemm_decode_kernel_name(
            problem.arch,
            problem.m,
            problem.n,
            problem.k,
            config,
            has_bias=problem.has_bias,
        )
        params = {
            "arch": problem.arch,
            "m": problem.m,
            "n": problem.n,
            "k": problem.k,
            "config": config,
            "has_bias": problem.has_bias,
        }
        candidates.append(Candidate(name, solidx, 0, params))
    return candidates


def _decode_aot_job(row: dict, kernel_name: str) -> dict:
    params = _decode_parse(kernel_name)
    if params is None:
        raise ValueError(f"invalid FlyDSL decode kernel name: {kernel_name!r}")
    m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
    csv_arch = (row.get("gfx") or "").strip()
    if (params["m"], params["n"], params["k"]) != (m, n, k):
        raise ValueError(
            "FlyDSL decode kernel name shape does not match CSV row: "
            f"name={(params['m'], params['n'], params['k'])}, row={(m, n, k)}"
        )
    if csv_arch and csv_arch != params["arch"]:
        raise ValueError(
            f"FlyDSL decode architecture mismatch: name={params['arch']}, csv={csv_arch}"
        )
    has_bias = _row_bool(row.get("bias"))
    if params["has_bias"] != has_bias:
        raise ValueError("FlyDSL decode CSV bias metadata does not match kernel name")
    if (row.get("dtype") or "").strip() != "torch.bfloat16":
        raise ValueError("FlyDSL decode AOT requires BF16 input dtype")
    if (row.get("outdtype") or "").strip() != "torch.bfloat16":
        raise ValueError("FlyDSL decode AOT requires BF16 output dtype")
    if _row_bool(row.get("scaleAB")):
        raise ValueError("FlyDSL decode AOT does not support scaling")
    if _row_bool(row.get("bpreshuffle")):
        raise ValueError("FlyDSL decode AOT does not support preshuffled weights")
    return {
        "kind": "decode",
        "config": params["config"],
        "gfx": csv_arch or params["arch"],
    }


# ---------------------------------------------------------------------------
# Built-in families
# ---------------------------------------------------------------------------

register(
    FlydslGemmFamily(
        name="hgemm_gfx950",
        op=A16W16,
        archs=frozenset({"gfx950"}),
        prefix="flydsl_hgemm_",
        parse=_hgemm_parse,
        check_row=_hgemm_check_row,
        launch=_hgemm_launch,
        candidates=_hgemm_gfx950_candidates,
        aot_job=_hgemm_aot_job,
    )
)

register(
    FlydslGemmFamily(
        name="hgemm_gfx1250",
        op=A16W16,
        archs=frozenset({"gfx1250"}),
        prefix="flydsl_hgemm_",
        parse=_hgemm_parse,
        check_row=_hgemm_check_row,
        launch=_hgemm_launch,
        # The gfx1250 kernel has no tuner catalog yet; its tuned rows were
        # measured outside the bf16 tuner.
        candidates=None,
        aot_job=_hgemm_aot_job,
    )
)

register(
    FlydslGemmFamily(
        name="decode",
        op=A16W16,
        archs=frozenset({"gfx942", "gfx950"}),
        prefix="flydsl_decode_",
        parse=_decode_parse,
        check_row=_decode_check_row,
        launch=_decode_launch,
        candidates=_decode_candidates,
        aot_job=_decode_aot_job,
        exact_m=True,
        tuner_tolerance=(0.01, 0.125),
    )
)
