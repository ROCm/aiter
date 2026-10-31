# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""can_use_flydsl_pa_decode agrees with what pa_decode accepts.

The host-only tests describe calls with CPU tensors and a stubbed arch, so
they run without a GPU. Run with:
python -m pytest -q op_tests/test_flydsl_pa_decode_supports.py
"""

import importlib.util

import pytest
import torch

if importlib.util.find_spec("flydsl") is None:
    pytest.skip("FlyDSL is not installed", allow_module_level=True)

import aiter.ops.flydsl.pa_decode as pa
from aiter.ops.flydsl.kernels.pa_decode.traits import (
    PaDecodeSchedule,
)
from aiter.ops.flydsl.kernels.pa_decode_plan import PADecodePlan

ARCHS = ("gfx942", "gfx950", "gfx1250")


class _Reached(Exception):
    """pa_decode accepted the arguments and went on to check the plan."""


class _StopPlan(PADecodePlan):
    def __init__(self):
        pass

    def validate(self, *args, **kwargs):
        raise _Reached


def _call(
    *,
    num_seqs=2,
    query_length=6,
    kv_heads=1,
    group=8,
    head_dim=256,
    block_size=16,
    kv="fp8",
    query_dtype=torch.bfloat16,
    scales="per-token",
    trans_v=True,
    device="cpu",
):
    """Arguments for one call; defaults are Qwen3.5 TP4 MTP-5 decode."""
    kv_dtype = torch.bfloat16 if kv == "bf16" else pa.dtypes.fp8
    x = 8 if kv == "bf16" else 16
    blocks = 4
    rows = num_seqs * query_length
    query = torch.zeros(
        rows, kv_heads * group, head_dim, dtype=query_dtype, device=device
    )
    key = torch.zeros(blocks, kv_heads, head_dim // x, block_size, x, device=device).to(
        kv_dtype
    )
    if trans_v:
        value_shape = (blocks, kv_heads, block_size // x, head_dim, x)
    else:
        value_shape = (blocks, kv_heads, head_dim, block_size)
    value = torch.zeros(value_shape, device=device).to(kv_dtype)
    args = [
        torch.empty_like(query),
        query,
        key,
        value,
        torch.ones(num_seqs, dtype=torch.int32, device=device),
        torch.zeros(num_seqs, blocks, dtype=torch.int32, device=device),
        query_length,
    ]
    options = {}
    if kv != "bf16" and scales == "per-token":
        shape = (blocks, kv_heads, block_size, 1)
        options["key_scale"] = torch.ones(shape, device=device)
        options["value_scale"] = torch.ones(shape, device=device)
    elif kv != "bf16" and scales == "per-tensor":
        options["key_scale"] = torch.ones(1, device=device)
        options["value_scale"] = torch.ones(1, device=device)
    return args, options


def _mutate(name, args, options):
    """Apply one change pa_decode has to reject."""
    output, query, key, value, lengths, table, query_length = args
    if name == "alibi":
        options["alibi_slopes"] = torch.ones(query.shape[1])
    elif name == "fp8-query":
        options["query_scale"] = torch.ones(1)
    elif name == "partition-512":
        options["context_partition_size"] = 512
    elif name == "rows-not-rectangular":
        args[6] = query_length + 1
    elif name == "int64-table":
        args[5] = table.long()
    elif name == "int64-lengths":
        args[4] = lengths.long()
    elif name == "table-rows":
        args[5] = table[:1]
    elif name == "noncontiguous-key":
        args[2] = torch.cat([key, key], dim=0)[::2]
    elif name == "output-dtype":
        args[0] = output.float()
    elif name == "query-dtype":
        args[1] = query.float()
        args[0] = output.float()
    elif name == "query-head-stride":
        args[1] = torch.cat([query, query], dim=2)[..., ::2]
    elif name == "gqa-not-divisible":
        # Eight query heads over three KV heads.
        args[2] = torch.cat([key] * 3, dim=1)
        args[3] = torch.cat([value] * 3, dim=1)
        options["key_scale"] = options["value_scale"] = None
    elif name == "one-scale":
        options["value_scale"] = None
    elif name == "scale-shape":
        options["key_scale"] = options["key_scale"][:, :, :-1]
        options["value_scale"] = options["value_scale"][:, :, :-1]
    elif name == "scale-dtype":
        options["key_scale"] = options["key_scale"].double()
        options["value_scale"] = options["value_scale"].double()
    elif name == "sinks-shape":
        options["sinks"] = torch.zeros(query.shape[1] + 1)
    elif name == "compute-type":
        options["compute_type"] = torch.float16
    elif name == "max-context-past-table":
        options["max_context_length"] = table.shape[1] * key.shape[-2] + 1
    else:
        raise AssertionError(name)


REJECTED = (
    "alibi",
    "fp8-query",
    "partition-512",
    "rows-not-rectangular",
    "int64-table",
    "int64-lengths",
    "table-rows",
    "noncontiguous-key",
    "output-dtype",
    "query-dtype",
    "query-head-stride",
    "gqa-not-divisible",
    "one-scale",
    "scale-shape",
    "scale-dtype",
    "sinks-shape",
    "compute-type",
    "max-context-past-table",
)

# (arch, call overrides) pa_decode's host checks reject.
REJECTED_SHAPES = (
    ("gfx90a", {}),
    ("gfx950", {"block_size": 32}),
    ("gfx950", {"head_dim": 96}),
    ("gfx950", {"head_dim": 192}),
    ("gfx950", {"head_dim": 2048}),
    ("gfx950", {"kv": "bf16", "query_dtype": torch.float16}),
    ("gfx950", {"kv": "bf16", "trans_v": False}),
    ("gfx1250", {"kv": "bf16", "head_dim": 192}),
)

ACCEPTED = (
    *[(arch, {}) for arch in ARCHS],
    *[(arch, {"scales": "per-tensor"}) for arch in ARCHS],
    *[(arch, {"scales": None}) for arch in ARCHS],
    ("gfx950", {"query_length": 1, "group": 16, "head_dim": 128, "block_size": 128}),
    ("gfx950", {"query_dtype": torch.float16, "trans_v": False}),
    ("gfx942", {"block_size": 64, "head_dim": 64}),
    ("gfx950", {"kv": "bf16", "head_dim": 128}),
    ("gfx950", {"kv": "bf16", "head_dim": 192}),
    ("gfx1250", {"kv": "bf16", "query_dtype": torch.float16, "trans_v": False}),
)


@pytest.fixture
def host_only(monkeypatch):
    """Describe calls with CPU tensors on a chosen arch."""
    monkeypatch.setattr(pa, "_is_cuda", lambda tensor: True)

    def use_arch(arch):
        monkeypatch.setattr(pa, "get_gfx_runtime", lambda: arch)

    return use_arch


def _pa_decode_accepts(args, options):
    try:
        pa.pa_decode(*args[:6], 1.0, args[6], work_plan=_StopPlan(), **options)
    except _Reached:
        return True
    except (NotImplementedError, TypeError, ValueError):
        return False
    raise AssertionError("pa_decode returned without checking its plan")


def _check(arch, overrides, mutation, host_only):
    host_only(arch)
    args, options = _call(**overrides)
    if mutation is not None:
        _mutate(mutation, args, options)
    supported = pa.can_use_flydsl_pa_decode(*args, **options)
    assert supported == _pa_decode_accepts(args, options)
    return supported


@pytest.mark.parametrize("arch,overrides", ACCEPTED)
def test_accepts(arch, overrides, host_only):
    assert _check(arch, overrides, None, host_only)


@pytest.mark.parametrize("mutation", REJECTED)
def test_rejects_what_pa_decode_rejects(mutation, host_only):
    assert not _check("gfx950", {}, mutation, host_only)


@pytest.mark.parametrize("arch,overrides", REJECTED_SHAPES)
def test_rejects_unsupported_shapes(arch, overrides, host_only):
    assert not _check(arch, overrides, None, host_only)


def test_bf16_query_group_past_lds(host_only, monkeypatch):
    """gfx950 BF16 KV: pa_decode's host checks pass, the scheduler rejects."""
    host_only("gfx950")
    args, options = _call(kv="bf16", head_dim=1024, group=32, query_length=1)
    assert _pa_decode_accepts(args, options)
    assert not pa.can_use_flydsl_pa_decode(*args, **options)

    from aiter.ops.flydsl.kernels.pa_decode import traits

    monkeypatch.setattr(traits, "get_rocm_arch", lambda: "gfx950")
    schedule = {
        "head_dim": 1024,
        "block_size": 16,
        "num_seqs": 2,
        "num_kv_heads": 1,
        "num_compute_units": 256,
        "work_capacity": 512,
        "query_dtype": "bf16",
        "kv_dtype": "bf16",
        "query_length": 1,
    }
    with pytest.raises(ValueError, match="64 KiB LDS"):
        PaDecodeSchedule.select(query_group_size=32, **schedule)
    # One row group smaller fits, and the predicate agrees.
    PaDecodeSchedule.select(query_group_size=16, **schedule)
    args, options = _call(kv="bf16", head_dim=1024, group=16, query_length=1)
    assert pa.can_use_flydsl_pa_decode(*args, **options)


def test_gfx1250_tiles_past_lds(host_only, monkeypatch):
    """gfx1250: the predicate applies the kernel builder's LDS limit.

    No supported shape reaches 320 KiB today (D1024 tops out near 309 KiB), so
    lower the limit in both places and check that they still agree.
    """
    from aiter.ops.flydsl.kernels import pa_decode_kernel as kernel

    host_only("gfx1250")
    limit = 200 * 1024
    monkeypatch.setattr(pa, "GFX1250_LDS_LIMIT_BYTES", limit)
    monkeypatch.setattr(kernel, "GFX1250_LDS_LIMIT_BYTES", limit)
    args, options = _call(head_dim=1024, query_length=1, group=16)
    assert _pa_decode_accepts(args, options)
    assert not pa.can_use_flydsl_pa_decode(*args, **options)
    with pytest.raises(NotImplementedError, match="LDS capacity"):
        pa.compile_pa_decode_tile(
            architecture="gfx1250",
            head_dim=1024,
            query_group_size=16,
            block_size=16,
            num_seqs=2,
            num_kv_heads=1,
            num_compute_units=256,
            num_partitions=2,
            softmax_scale=1.0,
            query_dtype="bf16",
            kv_dtype="fp8",
            per_token_kv=True,
            query_length=1,
            trans_v=True,
            work_capacity=512,
        )
    # The Qwen3.5 MTP-5 call stays under the lowered limit.
    args, options = _call()
    assert pa.can_use_flydsl_pa_decode(*args, **options)


def test_rejects_host_tensors(monkeypatch):
    monkeypatch.setattr(pa, "get_gfx_runtime", lambda: "gfx950")
    args, options = _call()
    assert not pa.can_use_flydsl_pa_decode(*args, **options)


def test_exported():
    from aiter.ops.flydsl import can_use_flydsl_pa_decode

    assert can_use_flydsl_pa_decode is pa.can_use_flydsl_pa_decode


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if pa.get_gfx_runtime() not in ARCHS:
        pytest.skip(f"pa_decode is unsupported on {pa.get_gfx_runtime()}")
    return torch.device("cuda")


def test_accepted_call_runs(device):
    """The default (Qwen3.5 MTP-5) call is accepted and pa_decode runs it."""
    args, options = _call(device=device)
    assert pa.can_use_flydsl_pa_decode(*args, **options)
    plan = pa.plan_pa_decode(args[4], args[2].shape[1], query_length=args[6])
    pa.pa_decode(*args[:6], 1.0, args[6], work_plan=plan, **options)
    torch.cuda.synchronize()
    assert torch.isfinite(args[0]).all()
