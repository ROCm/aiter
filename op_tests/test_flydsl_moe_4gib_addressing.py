# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Boundary-address coverage for FlyDSL MoE expert weight buffers.

Each allocation exceeds 4 GiB.  Boundary experts straddle and follow the 2/4
GiB offsets.  Every boundary expert has a distinct weight/scale signature;
its result is compared with a low-address oracle only after the high-address
launch, so an alias to expert zero or another boundary probe cannot pass.
"""

from __future__ import annotations

import math

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4

_GFX = get_gfx() if torch.cuda.is_available() else None
_A16_GPU_ONLY = pytest.mark.skipif(
    _GFX not in ("gfx942", "gfx950"),
    reason="gfx942 or gfx950 FlyDSL required",
)
_GFX950_ONLY = pytest.mark.skipif(_GFX != "gfx950", reason="gfx950 FlyDSL required")

MODEL_DIM = 7168
INTER_DIM = 3072
TILE_M = 32
M = 64
# BM=128 has 28 N tiles.  37 M tiles makes 1036 work tiles, exceeds the
# 1024-work-tile persistent threshold, and therefore enters the grid-stride loop.
PORT_PERSIST_M = 37 * 128
# The padded expert grid triggers the direct a16 persistent cap at 256 CTAs.
# Sixteen live M tiles produce 448 real work tiles, so the second loop executes.
A16_PORT_PERSIST_M = 16 * TILE_M
# More than 256 logical rows selects four nonpersistent loop iterations per CTA.
# Use full 32-row tiles so the oracle comparison does not include tail-row behavior.
COMMON_NONPERSIST_M = 288
# Common scaled-MFMA runs on gfx950; 257 M tiles exceed its 256-CU grid.
COMMON_PERSIST_M = 257 * TILE_M


def _expert_count(per_expert_bytes: int) -> int:
    return math.ceil(2**32 / per_expert_bytes) + 2


def _probes(per_expert_bytes: int, experts: int) -> list[int]:
    probes = [1, experts - 1]
    for boundary in (2**31, 2**32):
        first_after = math.ceil(boundary / per_expert_bytes)
        probes.extend((first_after - 1, first_after, first_after + 1))
    probes = sorted(set(probes))
    assert all(0 < probe < experts for probe in probes)
    return probes


def _require_memory(weight_bytes: int, scale_bytes: int = 0) -> None:
    # Account for allocator fragmentation, one local shuffled signature, and outputs.
    required = int((weight_bytes + scale_bytes) * 1.15 + 1024 * 2**20)
    free = torch.cuda.mem_get_info()[0]
    if free < required:
        pytest.skip(
            f"requires {required / 2**30:.1f} GiB free VRAM, found {free / 2**30:.1f}"
        )


def _routing(experts: int, m: int, tile_m: int) -> tuple[torch.Tensor, ...]:
    blocks = math.ceil(m / tile_m)
    sorted_ids = torch.full(
        ((experts + 1) * tile_m,), m, dtype=torch.int32, device="cuda"
    )
    sorted_ids[:m] = torch.arange(m, dtype=torch.int32, device="cuda")
    expert_ids = torch.full((experts + 1,), -1, dtype=torch.int32, device="cuda")
    valid = torch.tensor([m, m], dtype=torch.int32, device="cuda")
    return (
        sorted_ids,
        expert_ids,
        valid,
        torch.ones_like(sorted_ids, dtype=torch.float32),
        blocks,
    )


def _select_expert(expert_ids: torch.Tensor, blocks: int, expert: int) -> None:
    expert_ids.fill_(-1)
    expert_ids[:blocks] = expert


def _assert_boundary_results(
    run,
    weights: torch.Tensor,
    scales: torch.Tensor | None,
    probes: list[int],
    *,
    msg: str,
) -> None:
    """Compare each high expert with a separately compiled small oracle.

    The one-expert oracle is separately compiled. FP4/FP8 uses the small global
    resource path, while BF16 removes every nonzero expert displacement.
    """
    oracle_weights = torch.empty_like(weights[:1])
    oracle_scales = None if scales is None else torch.empty_like(scales[:1])
    background = tuple(value.clone() for value in run(0, weights, scales))
    previous = []
    for expert in probes:
        result = run(expert, weights, scales)

        oracle_weights.copy_(weights[expert : expert + 1])
        if scales is not None:
            oracle_scales.copy_(scales[expert : expert + 1])
        reference = run(0, oracle_weights, oracle_scales)

        for actual, expected in zip(result, reference):
            assert torch.isfinite(
                actual
            ).all(), f"{msg}, expert {expert}: non-finite output"
            torch.testing.assert_close(
                actual, expected, rtol=0, atol=0, msg=f"{msg}, expert {expert}"
            )
        assert any(
            not torch.equal(value, original)
            for value, original in zip(reference, background)
        ), f"{msg}, expert {expert}: signature collided with expert zero"
        assert all(
            any(not torch.equal(value, old) for value, old in zip(reference, earlier))
            for earlier in previous
        ), f"{msg}, expert {expert}: signature collided with an earlier probe"
        previous.append(tuple(value.clone() for value in reference))


def _make_mx_weights(n: int, k: int, *, fp8: bool, interleave: bool):
    per_expert = n * k // (1 if fp8 else 2)
    experts = _expert_count(per_expert)
    scale_bytes = experts * n * (k // 32)
    _require_memory(experts * per_expert, scale_bytes)
    probes = _probes(per_expert, experts)
    weights = torch.full(
        (experts, n, k // (1 if fp8 else 2)),
        0.5 if fp8 else 0x11,
        dtype=dtypes.fp8 if fp8 else torch.uint8,
        device="cuda",
    )
    scales = torch.full((experts, n, k // 32), 120, dtype=torch.uint8, device="cuda")
    for signature, expert in enumerate(probes, start=1):
        value = 1.0 + signature * 0.25 if fp8 else 0x11 * signature
        other = value + 0.5 if fp8 else min(value + 0x11, 0x77)
        local = torch.full_like(weights[:1], value)
        local[:, 1::2] = other
        weights[expert].copy_(shuffle_weight_a16w4(local, 16, interleave)[0])
        # 125..131 encode finite powers 2^-2..2^4, with a unique scale per probe.
        e8m0 = 124 + signature
        assert 0 < e8m0 < 255
        scales[expert].fill_(e8m0)
    return weights, scales, probes


def _make_bf16_weights(n: int, k: int):
    per_expert = n * k * torch.tensor([], dtype=torch.bfloat16).element_size()
    experts = _expert_count(per_expert)
    _require_memory(experts * per_expert)
    probes = _probes(per_expert, experts)
    weights = torch.zeros((experts, n, k), dtype=torch.bfloat16, device="cuda")
    for signature, expert in enumerate(probes, start=1):
        local = torch.full(
            (1, n, k), signature / 4, dtype=torch.bfloat16, device="cuda"
        )
        local[:, 1::2] = (signature + 1) / 4
        weights[expert].copy_(shuffle_weight(local, (16, 16))[0])
    return weights, probes


def _run_stage1_port(
    weights, scales, probes, *, m: int, w_dtype: str, interleave: bool = False
) -> None:
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm1

    experts = weights.shape[0]
    sorted_ids, expert_ids, valid, _, blocks = _routing(experts, m, TILE_M)
    activation = torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")

    def run(expert, run_weights, run_scales):
        _select_expert(expert_ids, blocks, expert)
        out = torch.empty((m, INTER_DIM), dtype=torch.bfloat16, device="cuda")
        flydsl_a16w4_gemm1(
            a_bf16=activation,
            w1_u8=run_weights.view(torch.uint8),
            w1_scale_u8=(
                run_scales.view(torch.uint8)
                if run_scales is not None
                else torch.empty(1, dtype=torch.uint8, device="cuda")
            ),
            sorted_expert_ids=expert_ids,
            cumsum_tensor=valid,
            m_indices=sorted_ids,
            inter_sorted_bf16=out,
            n_tokens=m,
            NE=run_weights.shape[0],
            D_HIDDEN=MODEL_DIM,
            D_INTER=INTER_DIM,
            topk=1,
            tile_m=TILE_M,
            tile_n=256,
            tile_k=256,
            w_dtype=w_dtype,
            w_layout="guinterleave" if interleave else "standard",
        )
        return (out,)

    _assert_boundary_results(run, weights, scales, probes, msg=f"stage1 {w_dtype}")


def _run_stage1_wrapper(weights, scales, probes) -> None:
    experts = weights.shape[0]
    sorted_ids, expert_ids, valid, sorted_weights, blocks = _routing(experts, M, TILE_M)
    activation = torch.ones((M, MODEL_DIM), dtype=torch.bfloat16, device="cuda")

    def run(expert, run_weights, run_scales):
        _select_expert(expert_ids, blocks, expert)
        out = flydsl_moe_stage1(
            a=activation,
            w1=run_weights,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=expert_ids,
            num_valid_ids=valid,
            topk=1,
            tile_m=TILE_M,
            tile_n=256,
            tile_k=256,
            a_dtype="bf16",
            b_dtype="fp4",
            out_dtype="bf16",
            w1_scale=run_scales,
            sorted_weights=sorted_weights,
        )
        return (out[:M].clone(),)

    _assert_boundary_results(run, weights, scales, probes, msg="stage1 a16w4 wrapper")


def _run_stage1_mxfp4_port(weights, scales, probes, *, interleave: bool) -> None:
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

    experts = weights.shape[0]
    sorted_ids, expert_ids, valid, _, blocks = _routing(experts, M, TILE_M)
    activation = torch.ones((M, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    aq = torch.full((M, MODEL_DIM // 2), 0x22, dtype=torch.uint8, device="cuda")
    asc = torch.full((M, MODEL_DIM // 32), 127, dtype=torch.uint8, device="cuda")

    def run(expert, run_weights, run_scales):
        _select_expert(expert_ids, blocks, expert)
        out = torch.empty((M, INTER_DIM // 2), dtype=torch.uint8, device="cuda")
        out_scale = torch.empty((M, INTER_DIM // 32), dtype=torch.uint8, device="cuda")
        flydsl_mxfp4_gemm1(
            a_quant=aq,
            a_scale_sorted_shuffled=asc,
            w1_u8=run_weights,
            w1_scale_u8=run_scales,
            sorted_expert_ids=expert_ids,
            cumsum_tensor=valid,
            m_indices=sorted_ids,
            inter_sorted_quant=out,
            inter_sorted_shuffled_scale=out_scale,
            hidden_states=activation,
            n_tokens=M,
            BM=TILE_M,
            use_nt=False,
            inline_quant=False,
            NE=run_weights.shape[0],
            D_HIDDEN=MODEL_DIM,
            D_INTER=INTER_DIM,
            topk=1,
            interleave=interleave,
        )
        return out, out_scale

    _assert_boundary_results(run, weights, scales, probes, msg="stage1 a4w4 port")


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("wrapper", marks=_A16_GPU_ONLY),
        pytest.param("direct", marks=_A16_GPU_ONLY),
        pytest.param("guinterleave", marks=_A16_GPU_ONLY),
        pytest.param("port", marks=_GFX950_ONLY),
        pytest.param("port_interleave", marks=_GFX950_ONLY),
    ],
)
def test_stage1_mxfp4_boundary_addresses(backend):
    interleave = backend in ("guinterleave", "port_interleave")
    weights, scales, probes = _make_mx_weights(
        2 * INTER_DIM, MODEL_DIM, fp8=False, interleave=interleave
    )
    try:
        if backend == "wrapper":
            _run_stage1_wrapper(weights, scales, probes)
        elif backend in ("port", "port_interleave"):
            _run_stage1_mxfp4_port(weights, scales, probes, interleave=interleave)
        else:
            _run_stage1_port(
                weights,
                scales,
                probes,
                m=M,
                w_dtype="fp4",
                interleave=interleave,
            )
    finally:
        weights = scales = None
        torch.cuda.empty_cache()


@_A16_GPU_ONLY
def test_stage1_bf16_boundary_addresses():
    weights, probes = _make_bf16_weights(2 * INTER_DIM, MODEL_DIM)
    try:
        _run_stage1_port(weights, None, probes, m=M, w_dtype="bf16")
    finally:
        weights = None
        torch.cuda.empty_cache()


def _run_stage2_a16_port(
    weights, scales, probes, *, w_dtype: str, persist: bool, m: int = M
) -> None:
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm2

    experts = weights.shape[0]
    sorted_ids, expert_ids, valid, sorted_weights, blocks = _routing(experts, m, TILE_M)
    activation = torch.ones((m, INTER_DIM), dtype=torch.bfloat16, device="cuda")

    def run(expert, run_weights, run_scales):
        _select_expert(expert_ids, blocks, expert)
        out = torch.zeros((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=activation,
            w2_u8=run_weights.view(torch.uint8),
            w2_scale_u8=(
                run_scales.view(torch.uint8)
                if run_scales is not None
                else torch.empty(1, dtype=torch.uint8, device="cuda")
            ),
            sorted_expert_ids=expert_ids,
            cumsum_tensor=valid,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            flat_out=out.view(-1),
            M_logical=m,
            max_sorted=m,
            NE=run_weights.shape[0],
            D_HIDDEN=MODEL_DIM,
            D_INTER=INTER_DIM,
            topk=1,
            tile_m=TILE_M,
            tile_n=256,
            tile_k=256,
            w_dtype=w_dtype,
            persist=persist,
        )
        return (out,)

    kind = "persistent" if persist else "nonpersistent"
    _assert_boundary_results(
        run, weights, scales, probes, msg=f"stage2 {w_dtype} {kind}"
    )


def _run_stage2_wrapper(
    weights,
    scales,
    probes,
    *,
    a_dtype: str,
    persist: bool,
    m: int = M,
) -> None:
    experts = weights.shape[0]
    sorted_ids, expert_ids, valid, sorted_weights, blocks = _routing(experts, m, TILE_M)
    if a_dtype == "bf16":
        activation = torch.ones((m, INTER_DIM), dtype=torch.bfloat16, device="cuda")
        activation_scale = None
    elif a_dtype == "fp8":
        activation = torch.full((m, 1, INTER_DIM), 1.0, dtype=dtypes.fp8, device="cuda")
        activation_scale = torch.full(
            (m, INTER_DIM // 32), 127, dtype=torch.uint8, device="cuda"
        )
    else:
        activation = torch.full(
            (m, 1, INTER_DIM // 2), 0x22, dtype=torch.uint8, device="cuda"
        )
        activation_scale = torch.full(
            (m, INTER_DIM // 32), 127, dtype=torch.uint8, device="cuda"
        )

    def run(expert, run_weights, run_scales):
        _select_expert(expert_ids, blocks, expert)
        out = torch.zeros((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
        flydsl_moe_stage2(
            inter_states=activation,
            w2=run_weights,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=expert_ids,
            num_valid_ids=valid,
            topk=1,
            tile_m=TILE_M,
            tile_n=256,
            tile_k=256,
            a_dtype=a_dtype,
            b_dtype="fp4",
            out_dtype="bf16",
            mode="atomic",
            w2_scale=run_scales,
            a2_scale=activation_scale,
            sorted_weights=sorted_weights,
            out=out,
            persist=persist,
        )
        return (out,)

    kind = "persistent" if persist else "nonpersistent"
    _assert_boundary_results(
        run, weights, scales, probes, msg=f"stage2 {a_dtype}w4 wrapper {kind}"
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("a16_direct", marks=_A16_GPU_ONLY),
        pytest.param("a16_direct_persistent", marks=_A16_GPU_ONLY),
        pytest.param("a16_wrapper", marks=_A16_GPU_ONLY),
        pytest.param("a4w4", marks=_GFX950_ONLY),
        pytest.param("a4w4_persistent", marks=_GFX950_ONLY),
        pytest.param("a8w4", marks=_GFX950_ONLY),
    ],
)
def test_stage2_mxfp4_boundary_addresses(backend):
    weights, scales, probes = _make_mx_weights(
        MODEL_DIM, INTER_DIM, fp8=False, interleave=False
    )
    try:
        if backend == "a16_direct":
            _run_stage2_a16_port(weights, scales, probes, w_dtype="fp4", persist=False)
        elif backend == "a16_direct_persistent":
            _run_stage2_a16_port(
                weights,
                scales,
                probes,
                w_dtype="fp4",
                persist=True,
                m=A16_PORT_PERSIST_M,
            )
        elif backend == "a16_wrapper":
            _run_stage2_wrapper(weights, scales, probes, a_dtype="bf16", persist=False)
        elif backend == "a4w4":
            _run_stage2_wrapper(
                weights,
                scales,
                probes,
                a_dtype="fp4",
                persist=False,
                m=COMMON_NONPERSIST_M,
            )
        elif backend == "a4w4_persistent":
            _run_stage2_wrapper(
                weights,
                scales,
                probes,
                a_dtype="fp4",
                persist=True,
                m=COMMON_PERSIST_M,
            )
        else:
            _run_stage2_wrapper(weights, scales, probes, a_dtype="fp8", persist=False)
    finally:
        weights = scales = None
        torch.cuda.empty_cache()


@_GFX950_ONLY
def test_stage2_mxfp8_boundary_addresses():
    """MXFP8 B takes the full-operand load branch in mixed_moe_gemm2_common."""
    weights, scales, probes = _make_mx_weights(
        MODEL_DIM, INTER_DIM, fp8=True, interleave=False
    )
    try:
        experts = weights.shape[0]
        sorted_ids, expert_ids, valid, sorted_weights, blocks = _routing(
            experts, M, TILE_M
        )
        activation = torch.full((M, 1, INTER_DIM), 1.0, dtype=dtypes.fp8, device="cuda")
        activation_scale = torch.full(
            (M, INTER_DIM // 32), 127, dtype=torch.uint8, device="cuda"
        )

        def run(expert, run_weights, run_scales):
            _select_expert(expert_ids, blocks, expert)
            out = torch.zeros((M, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
            flydsl_moe_stage2(
                inter_states=activation,
                w2=run_weights,
                sorted_token_ids=sorted_ids,
                sorted_expert_ids=expert_ids,
                num_valid_ids=valid,
                topk=1,
                tile_m=TILE_M,
                tile_n=256,
                tile_k=256,
                a_dtype="fp8",
                b_dtype="fp8",
                out_dtype="bf16",
                mode="atomic",
                w2_scale=run_scales,
                a2_scale=activation_scale,
                sorted_weights=sorted_weights,
                out=out,
                persist=False,
            )
            return (out,)

        _assert_boundary_results(
            run, weights, scales, probes, msg="stage2 a8w8 nonpersistent"
        )
    finally:
        weights = scales = None
        torch.cuda.empty_cache()


@_GFX950_ONLY
@pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "nonatomic"])
def test_stage2_mxfp4_port_modes_boundary_addresses(atomic):
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2

    weights, scales, probes = _make_mx_weights(
        MODEL_DIM, INTER_DIM, fp8=False, interleave=False
    )
    try:
        m, bm = (M, TILE_M) if atomic else (PORT_PERSIST_M, 128)
        sorted_ids, expert_ids, valid, sorted_weights, blocks = _routing(
            weights.shape[0], m, bm
        )
        a = torch.full((m, 1, INTER_DIM // 2), 0x22, dtype=torch.uint8, device="cuda")
        a_scale = torch.full(
            (m, INTER_DIM // 32), 127, dtype=torch.uint8, device="cuda"
        )

        def run(expert, run_weights, run_scales):
            _select_expert(expert_ids, blocks, expert)
            out = torch.zeros((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
            flydsl_mxfp4_gemm2(
                inter_sorted_quant=a,
                inter_sorted_shuffled_scale=a_scale,
                w2_u8=run_weights,
                w2_scale_u8=run_scales,
                sorted_expert_ids=expert_ids,
                cumsum_tensor=valid,
                sorted_token_ids=sorted_ids,
                sorted_weights=sorted_weights,
                flat_out=out,
                M_logical=m,
                max_sorted=m,
                BM=bm,
                use_nt=False,
                atomic=atomic,
                mxfp4out=False,
                NE=run_weights.shape[0],
                D_HIDDEN=MODEL_DIM,
                D_INTER=INTER_DIM,
                topk=1,
            )
            return (out,)

        mode = "atomic" if atomic else "nonatomic persistent"
        _assert_boundary_results(
            run, weights, scales, probes, msg=f"stage2 a4w4 port {mode}"
        )
    finally:
        weights = scales = None
        torch.cuda.empty_cache()


@_A16_GPU_ONLY
@pytest.mark.parametrize("persist", [False, True], ids=["nonpersistent", "persistent"])
def test_stage2_bf16_boundary_addresses(persist):
    weights, probes = _make_bf16_weights(MODEL_DIM, INTER_DIM)
    try:
        _run_stage2_a16_port(
            weights,
            None,
            probes,
            w_dtype="bf16",
            persist=persist,
            m=A16_PORT_PERSIST_M if persist else M,
        )
    finally:
        weights = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
