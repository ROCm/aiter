# SPDX-License-Identifier: MIT
"""Promotion gates for gfx942 exact-local P4 plus segment-range P3/P4."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiter.ops.flash_kda import (  # noqa: E402
    flash_kda_fwd as native_fwd,
    flash_kda_fwd_hip,
    flash_kda_workspace_size,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (  # noqa: E402
    flash_kda_fwd as triton_fwd,
)

D = 128
H = 12
LOWER_BOUND = -5.0


def _inputs(tokens: int, seed: int) -> dict[str, torch.Tensor | float]:
    torch.manual_seed(seed)
    shape = (1, tokens, H, D)

    def projection() -> torch.Tensor:
        return F.silu(
            torch.randn(shape, device="cuda", dtype=torch.float32)
        ).to(torch.bfloat16)

    return {
        "q": projection(),
        "k": projection(),
        "v": projection(),
        "g": torch.randn(shape, device="cuda", dtype=torch.float32).to(
            torch.bfloat16
        ),
        "beta": torch.randn(
            (1, tokens, H), device="cuda", dtype=torch.bfloat16
        ).float(),
        "A_log": torch.empty(H, device="cuda", dtype=torch.float32)
        .uniform_(1.0, 16.0)
        .log_(),
        "dt_bias": torch.randn(H * D, device="cuda", dtype=torch.float32),
        "scale": D**-0.5,
    }


def _state(dtype: torch.dtype, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    value = torch.randn((1, H, D, D), device="cuda", dtype=torch.float32) * 0.02
    value += torch.linspace(-0.04, 0.03, D, device="cuda").view(1, 1, D, 1)
    value += 0.37 * torch.linspace(0.02, -0.01, D, device="cuda").view(
        1, 1, 1, D
    )
    return value.to(dtype)


def _metadata(tokens: int, packed: bool) -> torch.Tensor | None:
    if not packed:
        return None
    return torch.tensor([0, tokens], device="cuda", dtype=torch.int32)


def _set_route(route: str, tokens: int) -> None:
    for name in (
        "FLASH_KDA_K2",
        "FLASH_KDA_GFX942_P3_PREFETCH",
        "FLASH_KDA_GFX942_FUSED_PREP_NEUMANN",
        "FLASH_KDA_GFX942_AQK_OVERLAP",
        "FLASH_KDA_GFX942_FUSED_OUT",
        "FLASH_KDA_GFX942_P3_PERSISTENT_MIXED",
        "FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE",
        "FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE_BATCH",
        "FLASH_KDA_GFX942_HYBRID_LOCAL_OUT",
        "FLASH_KDA_GFX942_P3_P4_PIPELINE",
        "FLASH_KDA_CS_SKIP_K1_PREP",
        "FLASH_KDA_CS_SKIP_K1_SOLVE",
        "FLASH_KDA_CS_SKIP_SCAN",
        "FLASH_KDA_CS_SKIP_OUT",
    ):
        os.environ.pop(name, None)
    if route == "automatic":
        return
    if route == "explicit_hybrid":
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE"] = "1"
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE_BATCH"] = (
            "48" if (tokens + 63) // 64 <= 128 else "72"
        )
    elif route == "full_hybrid":
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE"] = "0"
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_OUT"] = "1"
        os.environ["FLASH_KDA_GFX942_P3_P4_PIPELINE"] = "0"
    elif route == "rollback":
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE"] = "0"
    elif route == "explicit_old":
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_PIPELINE"] = "0"
        os.environ["FLASH_KDA_GFX942_HYBRID_LOCAL_OUT"] = "0"
        os.environ["FLASH_KDA_GFX942_P3_P4_PIPELINE"] = "1"
        os.environ["FLASH_KDA_GFX942_P3_PREFETCH"] = "1"
        os.environ["FLASH_KDA_GFX942_FUSED_PREP_NEUMANN"] = "1"
    else:
        raise ValueError(route)


def _native(
    data: dict[str, torch.Tensor | float], *, tokens: int, packed: bool,
    initial: torch.Tensor | None, final: bool, route: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _set_route(route, tokens)
    return native_fwd(
        **data,
        initial_state=initial,
        output_final_state=final,
        lower_bound=LOWER_BOUND,
        cu_seqlens=_metadata(tokens, packed),
    )


def _triton(
    data: dict[str, torch.Tensor | float], *, tokens: int, packed: bool,
    initial: torch.Tensor | None, final: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return triton_fwd(
        **data,
        initial_state=initial,
        output_final_state=final,
        state_v_first=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=_metadata(tokens, packed),
    )


def _rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float()
    e = expected.float()
    return float(
        ((a - e).square().mean().sqrt() /
         e.square().mean().sqrt().clamp_min(1.0e-8)).item()
    )


def _check_result(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
    *, bitwise: bool = False,
) -> None:
    out, state = actual
    ref_out, ref_state = expected
    assert torch.isfinite(out).all(), f"{label}: non-finite output"
    out_rms = _rms(out, ref_out)
    state_rms = 0.0
    assert (state is None) == (ref_state is None), f"{label}: state presence"
    if state is not None:
        assert ref_state is not None and torch.isfinite(state).all()
        state_rms = _rms(state, ref_state)
    if bitwise:
        assert torch.equal(out, ref_out), f"{label}: output not bitwise"
        if state is not None:
            assert torch.equal(state, ref_state), f"{label}: state not bitwise"
    else:
        assert max(out_rms, state_rms) < 2.5e-2, (
            f"{label}: out_rms={out_rms}, state_rms={state_rms}"
        )
    print(
        f"PASS {label:46s} out_rms={out_rms:.7f} "
        f"state_rms={state_rms:.7f} bitwise={int(bitwise)}"
    )


def check_boundaries(seed: int) -> None:
    for tokens in (8191, 8192, 8193):
        data = _inputs(tokens, seed + tokens)
        initial = _state(torch.float32, seed + tokens + 1)
        for packed in (False, True):
            actual = _native(
                data, tokens=tokens, packed=packed, initial=initial,
                final=True, route="automatic",
            )
            expected = _triton(
                data, tokens=tokens, packed=packed, initial=initial, final=True,
            )
            _check_result(
                f"boundary T={tokens} {'packed' if packed else 'dense'}",
                actual, expected,
            )
            if tokens == 8191:
                rollback = _native(
                    data, tokens=tokens, packed=packed, initial=initial,
                    final=True, route="rollback",
                )
                _check_result(
                    f"ineligible flag gate T={tokens} "
                    f"{'packed' if packed else 'dense'}",
                    actual, rollback, bitwise=True,
                )
        del data, initial
        torch.cuda.empty_cache()


def check_state_matrix(seed: int) -> None:
    tokens = 8192
    data = _inputs(tokens, seed)
    states = {
        "fresh": None,
        "fp32": _state(torch.float32, seed + 1),
        "bf16": _state(torch.bfloat16, seed + 2),
    }
    for packed in (False, True):
        for state_name, initial in states.items():
            for final in (False, True):
                pipeline = _native(
                    data, tokens=tokens, packed=packed, initial=initial,
                    final=final, route="automatic",
                )
                full = _native(
                    data, tokens=tokens, packed=packed, initial=initial,
                    final=final, route="full_hybrid",
                )
                _check_result(
                    f"state {state_name}/final={int(final)} "
                    f"{'packed' if packed else 'dense'}",
                    pipeline, full, bitwise=True,
                )
                expected_dtype = (
                    initial.dtype if initial is not None else torch.float32
                )
                if final:
                    assert pipeline[1] is not None
                    assert pipeline[1].dtype == expected_dtype
                else:
                    assert pipeline[1] is None

    # The packed N=1 adapter deliberately enters the dense launch geometry.
    initial = states["fp32"]
    packed_result = _native(
        data, tokens=tokens, packed=True, initial=initial,
        final=True, route="automatic",
    )
    dense_result = _native(
        data, tokens=tokens, packed=False, initial=initial,
        final=True, route="automatic",
    )
    _check_result("packed-N1 versus true-dense", packed_result, dense_result,
                  bitwise=True)

    rollback = _native(
        data, tokens=tokens, packed=True, initial=initial,
        final=True, route="rollback",
    )
    triton = _triton(
        data, tokens=tokens, packed=True, initial=initial, final=True,
    )
    _check_result("explicit flag=0 rollback", rollback, triton)
    assert torch.equal(rollback[1], packed_result[1]), (
        "rollback and range P3 must publish the same final state"
    )


def check_automatic_selection(seed: int) -> None:
    """Confirm unset defaults and explicit rollback at both promoted buckets."""
    for tokens in (8192, 16384):
        data = _inputs(tokens, seed + tokens)
        dtype = torch.float32 if tokens == 8192 else torch.bfloat16
        initial = _state(dtype, seed + tokens + 1)
        for packed in (False, True):
            automatic = _native(
                data, tokens=tokens, packed=packed, initial=initial,
                final=True, route="automatic",
            )
            explicit_hybrid = _native(
                data, tokens=tokens, packed=packed, initial=initial,
                final=True, route="explicit_hybrid",
            )
            _check_result(
                f"automatic versus explicit hybrid T={tokens} "
                f"{'packed' if packed else 'dense'}",
                automatic, explicit_hybrid, bitwise=True,
            )

            rollback = _native(
                data, tokens=tokens, packed=packed, initial=initial,
                final=True, route="rollback",
            )
            explicit_old = _native(
                data, tokens=tokens, packed=packed, initial=initial,
                final=True, route="explicit_old",
            )
            _check_result(
                f"flag=0 versus explicit old pipeline T={tokens} "
                f"{'packed' if packed else 'dense'}",
                rollback, explicit_old, bitwise=True,
            )
        del data, initial
        torch.cuda.empty_cache()


def _lowlevel_run(
    data: dict[str, torch.Tensor | float], *, tokens: int, packed: bool,
    has_input: bool, has_output: bool, dtype: torch.dtype,
    initial: torch.Tensor, route: str,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    _set_route(route, tokens)
    out = torch.empty_like(data["v"])
    workspace = torch.empty(
        flash_kda_workspace_size(tokens, H, 1),
        device="cuda", dtype=torch.uint8,
    )
    ignored_in = torch.full(
        (1, H, D, D), 7.0, device="cuda", dtype=dtype
    )
    state_in = initial if has_input else ignored_in
    state_out = torch.full(
        (1, H, D, D), -11.0, device="cuda", dtype=dtype
    )
    poison = state_out.clone()
    cu = _metadata(tokens, packed)
    if cu is None:
        cu = torch.empty(0, device="cuda", dtype=torch.int32)
    flash_kda_fwd_hip(
        data["q"], data["k"], data["v"], data["g"], data["beta"],
        out, workspace, data["A_log"], data["dt_bias"], state_in,
        state_out, cu, float(data["scale"]), LOWER_BOUND,
        has_input, has_output, packed,
    )
    torch.cuda.synchronize()
    ignored_unchanged = has_output or torch.equal(state_out, poison)
    return out, state_out if has_output else None, ignored_unchanged


def check_lowlevel_state_matrix(seed: int) -> None:
    """Cover all seven ABI state modes, including output-only BF16."""
    tokens = 8192
    data = _inputs(tokens, seed)
    modes = (
        ("none", False, False, torch.float32),
        ("out-bf16", False, True, torch.bfloat16),
        ("out-fp32", False, True, torch.float32),
        ("in-bf16", True, False, torch.bfloat16),
        ("in-fp32", True, False, torch.float32),
        ("in-out-bf16", True, True, torch.bfloat16),
        ("in-out-fp32", True, True, torch.float32),
    )
    for packed in (False, True):
        for index, (name, has_input, has_output, dtype) in enumerate(modes):
            initial = _state(dtype, seed + index + 1)
            actual = _lowlevel_run(
                data, tokens=tokens, packed=packed, has_input=has_input,
                has_output=has_output, dtype=dtype, initial=initial,
                route="automatic",
            )
            expected = _lowlevel_run(
                data, tokens=tokens, packed=packed, has_input=has_input,
                has_output=has_output, dtype=dtype, initial=initial,
                route="full_hybrid",
            )
            assert actual[2] and expected[2], (
                f"{name}: ignored final-state buffer was modified"
            )
            _check_result(
                f"lowlevel {name} {'packed' if packed else 'dense'}",
                actual[:2], expected[:2], bitwise=True,
            )


def check_graph(seed: int) -> None:
    tokens = 8192
    data = _inputs(tokens, seed)
    initial = _state(torch.float32, seed + 1)
    cu = _metadata(tokens, True)
    _set_route("automatic", tokens)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            warm = native_fwd(
                **data, initial_state=initial, output_final_state=True,
                lower_bound=LOWER_BOUND, cu_seqlens=cu,
            )
    stream.synchronize()
    del warm

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = native_fwd(
            **data, initial_state=initial, output_final_state=True,
            lower_bound=LOWER_BOUND, cu_seqlens=cu,
        )
    graph.replay()
    torch.cuda.synchronize()
    first = (captured[0].clone(), captured[1].clone())
    graph.replay()
    torch.cuda.synchronize()
    _check_result("graph capture/replay deterministic", captured, first,
                  bitwise=True)
    triton = _triton(
        data, tokens=tokens, packed=True, initial=initial, final=True,
    )
    _check_result("graph capture/replay versus Triton", captured, triton)


def check_multistream(seed: int) -> None:
    tokens = 8192
    data0 = _inputs(tokens, seed)
    data1 = _inputs(tokens, seed + 1)
    state0 = _state(torch.float32, seed + 2)
    state1 = _state(torch.bfloat16, seed + 3)
    ref0 = _native(
        data0, tokens=tokens, packed=True, initial=state0,
        final=True, route="automatic",
    )
    ref1 = _native(
        data1, tokens=tokens, packed=False, initial=state1,
        final=True, route="automatic",
    )
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    stream0.wait_stream(torch.cuda.current_stream())
    stream1.wait_stream(torch.cuda.current_stream())
    _set_route("automatic", tokens)
    with torch.cuda.stream(stream0):
        out0 = native_fwd(
            **data0, initial_state=state0, output_final_state=True,
            lower_bound=LOWER_BOUND, cu_seqlens=_metadata(tokens, True),
        )
    with torch.cuda.stream(stream1):
        out1 = native_fwd(
            **data1, initial_state=state1, output_final_state=True,
            lower_bound=LOWER_BOUND, cu_seqlens=None,
        )
    torch.cuda.current_stream().wait_stream(stream0)
    torch.cuda.current_stream().wait_stream(stream1)
    torch.cuda.synchronize()
    _check_result("multi-stream packed/fp32", out0, ref0, bitwise=True)
    _check_result("multi-stream dense/bf16", out1, ref1, bitwise=True)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(f"GPU={props.name} arch={props.gcnArchName}")
    check_boundaries(args.seed)
    check_state_matrix(args.seed + 100_000)
    check_lowlevel_state_matrix(args.seed + 150_000)
    check_automatic_selection(args.seed + 175_000)
    check_graph(args.seed + 200_000)
    check_multistream(args.seed + 300_000)
    print("ALL HYBRID PIPELINE PROMOTION GATES PASSED")


if __name__ == "__main__":
    main()
