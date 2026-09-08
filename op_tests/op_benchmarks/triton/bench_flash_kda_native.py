# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare native HIP FlashKDA with both aiter Triton KDA implementations.

All backends receive the same Kimi-K3 TP8-style packed tensors in one process:

* ``native`` calls :func:`aiter.ops.flash_kda.flash_kda_fwd` directly;
* ``triton`` selects the PR #4683 Triton path through the public wrapper; and
* ``baseline`` selects the original Triton chunk pipeline through that wrapper.

The default mode calls the native public operator and selects both Triton
implementations through the public KDA wrapper.  Pass ``--public-k3`` for the
final production-routing gate: its ``native`` row uses the zero-environment
public K3 default, proves it is bitwise equal to explicit native, and compares
it with the same public wrapper forced to Triton.
Pass ``--execution graph`` to compile and check every backend eagerly, capture
each backend in its own ROCm graph, verify replay against that backend's eager
result, and time only steady-state ``graph.replay()``.
Latencies are collected with alternating backend order and are reported
together with token throughput, speedup versus Triton FlashKDA, incremental
peak allocated memory, and output/final-state error.

Examples::

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --case single-2k --case ragged-16k --warmup 5 --repeat 30

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --seq-lens 127 511 1361 --resume --csv result.csv --json result.json

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --suite mixed --backend native --backend triton

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --suite mixed-boundary --backend native --backend triton

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --case resume-4x4k --execution graph --backend native --backend triton

    python op_tests/op_benchmarks/triton/bench_flash_kda_native.py \
        --suite core --heads 2 --value-heads 4 \
        --execution graph --public-k3 --backend native --backend triton \
        --min-speedup 1.0 --min-geomean-speedup 1.05 \
        --min-paired-win-fraction 0.75

For the complete gfx950 GVA correctness and performance acceptance matrix, run
``scripts/run_flash_kda_gva_acceptance.sh`` from a clean checkout.
For the three-seed Hq=HV=12 public-K3 promotion matrix, run
``scripts/run_flash_kda_public_k3_perf_acceptance.sh``.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make direct execution from any working directory resolve the source checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from aiter.ops.flash_kda import (
    flash_kda_fwd as flash_kda_native_fwd,
    flash_kda_native_supported,
)
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

HEAD_DIM = 128
K3_GLOBAL_HEADS = 96
K3_TP_SIZE = 8
K3_LOCAL_HEADS = K3_GLOBAL_HEADS // K3_TP_SIZE
LOWER_BOUND = -5.0
BACKEND_ORDER = ("native", "triton", "baseline")
EXECUTION_MODES = ("eager", "graph")


@dataclass(frozen=True)
class Case:
    name: str
    seq_lens: tuple[int, ...]
    resume: bool = False
    # Per-sequence state presence for ATOM mixed prefill/decode batches.  The
    # legacy ``resume`` flag remains the compact spelling for all-or-nothing
    # cases and for the ``--seq-lens --resume`` CLI.
    resume_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError(f"{self.name}: sequence lengths must be positive")
        if self.resume and self.resume_mask is not None:
            raise ValueError(f"{self.name}: use either resume or resume_mask")
        if self.resume_mask is not None and len(self.resume_mask) != len(
            self.seq_lens
        ):
            raise ValueError(
                f"{self.name}: resume_mask has {len(self.resume_mask)} entries "
                f"for {len(self.seq_lens)} sequences"
            )


@dataclass
class _CapturedBackend:
    """Own a graph and every object whose lifetime its replay depends on."""

    graph: torch.cuda.CUDAGraph
    output: tuple[torch.Tensor, torch.Tensor | None]
    stream: torch.cuda.Stream

    def replay(self) -> None:
        self.graph.replay()


class _Dim3(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    )


class _HipKernelNodeParams(ctypes.Structure):
    _fields_ = (
        ("block_dim", _Dim3),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("func", ctypes.c_void_p),
        ("grid_dim", _Dim3),
        ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
        ("shared_mem_bytes", ctypes.c_uint),
    )


# Shapes used by the K3 integration study.  Every case is packed B=1, including
# a single sequence, because that is the metadata/layout ATOM passes to prefill.
CORE_CASES = (
    Case("single-128", (128,)),
    Case("single-256", (256,)),
    Case("single-512", (512,)),
    Case("single-1k", (1024,)),
    Case("single-2k", (2048,)),
    Case("single-8k", (8192,)),
    Case("single-16k", (16384,)),
    Case("batch-16x1k", (1024,) * 16),
    Case("batch-64x256", (256,) * 64),
    Case("ragged-16k", (127, 255, 511, 1023, 2047, 3073, 4095, 5253)),
    Case("resume-4x4k", (4096,) * 4, resume=True),
)


def _mixed_case(decodes: int) -> Case:
    """One fresh long prefill sharing ATOM vLLM's 16K budget with decodes."""

    total_budget = 16384
    return Case(
        f"mixed-{decodes}d-budget16k",
        (1,) * decodes + (total_budget - decodes,),
        resume_mask=(True,) * decodes + (False,),
    )


# ATOM's vLLM path orders decode requests before prefills in a mixed batch.
# Keeping the total at max_num_batched_tokens=16384 models its token budget
# rather than an impossible 16K prefill plus extra tokens.  Preserve the
# requested D=8/32/64/128 sweep, and separately expose the production-real
# D=7/8/32/63 sweep: D=7->8 crosses affine to hybrid at N=9, while D=63 plus
# one prefill exactly reaches the published max_num_seqs=64 ceiling.  D=64
# and D=128 remain useful capacity stress cases beyond that recipe limit.
_MIXED_CASE_BY_DECODES = {
    decodes: _mixed_case(decodes) for decodes in (7, 8, 32, 63, 64, 128)
}
MIXED_CASES = tuple(
    _MIXED_CASE_BY_DECODES[decodes] for decodes in (8, 32, 64, 128)
)
MIXED_PRODUCTION_CASES = tuple(
    _MIXED_CASE_BY_DECODES[decodes] for decodes in (7, 8, 32, 63)
)
_ALL_MIXED_CASES = tuple(_MIXED_CASE_BY_DECODES.values())

# Hybrid context routing keeps sequences of at most 64 C16 chunks on its
# direct path.  N=16 also selects context automatically at this average length,
# so these otherwise-identical ATOM-shaped batches really take the direct and
# affine sides of the 1024/1025 boundary without growing the default core sweep.
MIXED_BOUNDARY_CASES = (
    Case(
        "mixed-15d-prefill-1024",
        (1,) * 15 + (1024,),
        resume_mask=(True,) * 15 + (False,),
    ),
    Case(
        "mixed-15d-prefill-1025",
        (1,) * 15 + (1025,),
        resume_mask=(True,) * 15 + (False,),
    ),
)

CASES = CORE_CASES + _ALL_MIXED_CASES + MIXED_BOUNDARY_CASES
CASE_SUITES = {
    "core": CORE_CASES,
    "mixed": MIXED_CASES,
    "mixed-production": MIXED_PRODUCTION_CASES,
    "mixed-boundary": MIXED_BOUNDARY_CASES,
    "all": CASES,
}


def _case_resume_mask(case: Case) -> tuple[bool, ...]:
    if case.resume_mask is not None:
        return case.resume_mask
    return (case.resume,) * len(case.seq_lens)


def _case_state_label(case: Case) -> str:
    mask = _case_resume_mask(case)
    if all(mask):
        return "resume"
    if any(mask):
        return "mixed"
    return "fresh"


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _errors(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = actual - reference
    relative_rms = (
        difference.square().mean().sqrt()
        / reference.square().mean().sqrt().clamp_min(1e-8)
    )
    return float(relative_rms.item()), float(difference.abs().max().item())


def _storage_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare exact logical tensor bytes without a dtype conversion."""

    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _assert_output_contract(
    case: Case,
    backend: str,
    output_pair: tuple[torch.Tensor, torch.Tensor | None],
    inputs: dict[str, object],
) -> dict[str, object]:
    """Fail closed on the public output dtype, shape, and device contract."""

    output, final_state = output_pair
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    if final_state is None:
        raise RuntimeError(
            f"{case.name}/{backend}: output_final_state=True returned None"
        )
    expected_output_shape = (*q.shape[:-1], v.shape[-1])
    expected_state_shape = (
        len(case.seq_lens),
        v.shape[-2],
        v.shape[-1],
        k.shape[-1],
    )
    failures = []
    if output.dtype != q.dtype:
        failures.append(f"output dtype {output.dtype} != q dtype {q.dtype}")
    if final_state.dtype != torch.float32:
        failures.append(f"final-state dtype {final_state.dtype} != torch.float32")
    if tuple(output.shape) != expected_output_shape:
        failures.append(
            f"output shape {tuple(output.shape)} != {expected_output_shape}"
        )
    if tuple(final_state.shape) != expected_state_shape:
        failures.append(
            "final-state shape "
            f"{tuple(final_state.shape)} != {expected_state_shape}"
        )
    if output.device != q.device or final_state.device != q.device:
        failures.append(
            f"output/state device {output.device}/{final_state.device} "
            f"!= q device {q.device}"
        )
    if failures:
        raise RuntimeError(
            f"{case.name}/{backend}: public output contract failed: "
            + "; ".join(failures)
        )
    return {
        "output_dtype": str(output.dtype),
        "final_state_dtype": str(final_state.dtype),
        "output_shape": list(output.shape),
        "final_state_shape": list(final_state.shape),
        "output_contract_verified": True,
    }


def _assert_initial_state_contract(
    case: Case,
    initial_state: torch.Tensor | None,
    *,
    literal_none: bool,
) -> dict[str, object]:
    """Verify fresh, resume, and mixed state presence/content semantics."""

    expected_mask = _case_resume_mask(case)
    if literal_none:
        if initial_state is not None or any(expected_mask):
            raise RuntimeError(
                f"{case.name}: literal initial_state=None contract failed"
            )
        return {
            "input_initial_state_literal_none": True,
            "input_initial_state_dtype": None,
            "input_resume_mask_verified": True,
        }
    if initial_state is None:
        raise RuntimeError(f"{case.name}: materialized initial state is missing")
    if initial_state.dtype != torch.float32:
        raise RuntimeError(
            f"{case.name}: initial-state dtype {initial_state.dtype} != "
            "torch.float32"
        )
    if initial_state.shape[0] != len(expected_mask):
        raise RuntimeError(
            f"{case.name}: initial-state batch {initial_state.shape[0]} != "
            f"{len(expected_mask)}"
        )
    actual_mask = tuple(
        bool(torch.count_nonzero(sequence).item()) for sequence in initial_state
    )
    if actual_mask != expected_mask:
        raise RuntimeError(
            f"{case.name}: initial-state nonzero mask {actual_mask} != "
            f"{expected_mask}"
        )
    return {
        "input_initial_state_literal_none": False,
        "input_initial_state_dtype": str(initial_state.dtype),
        "input_resume_mask_verified": True,
    }


def _max_sequence_relative_rms(
    actual: torch.Tensor, reference: torch.Tensor
) -> float:
    """Maximum relative RMS over the leading (sequence) dimension."""

    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = actual - reference
    reduction_dims = tuple(range(1, difference.ndim))
    relative_rms = difference.square().mean(dim=reduction_dims).sqrt() / (
        reference.square().mean(dim=reduction_dims).sqrt().clamp_min(1e-8)
    )
    return float(relative_rms.max().item())


def _packed_output_max_sequence_relative_rms(
    actual: torch.Tensor,
    reference: torch.Tensor,
    seq_lens: tuple[int, ...],
) -> float:
    """Maximum output relative RMS after slicing the packed token axis."""

    start = 0
    values = []
    for length in seq_lens:
        end = start + length
        relative_rms, _ = _errors(
            actual[:, start:end], reference[:, start:end]
        )
        values.append(relative_rms)
        start = end
    if start != actual.shape[1]:
        raise ValueError(
            f"packed lengths cover {start} tokens, output has {actual.shape[1]}"
        )
    return max(values)


def _mixed_decode_output(output: torch.Tensor, case: Case) -> torch.Tensor | None:
    """Select resumed one-token outputs from an explicitly mixed case."""

    if case.resume_mask is None or _case_state_label(case) != "mixed":
        return None
    pieces = []
    start = 0
    for length, resumed in zip(case.seq_lens, case.resume_mask):
        end = start + length
        if resumed and length == 1:
            pieces.append(output[:, start:end])
        start = end
    if not pieces:
        return None
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def _make_state(
    sequences: int,
    heads: int,
    *,
    resume_mask: tuple[bool, ...],
    device: torch.device,
) -> torch.Tensor:
    if len(resume_mask) != sequences:
        raise ValueError("resume_mask must have one entry per sequence")
    state = torch.zeros(
        sequences,
        heads,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    if any(resume_mask):
        state.normal_(mean=0.0, std=0.02)
        # Make V and K physically distinguishable even though both dimensions
        # happen to be 128 in K3.
        v_axis = torch.linspace(-0.04, 0.03, HEAD_DIM, device=device).view(
            1, 1, HEAD_DIM, 1
        )
        k_axis = torch.linspace(0.02, -0.01, HEAD_DIM, device=device).view(
            1, 1, 1, HEAD_DIM
        )
        state.add_(v_axis).add_(0.37 * k_axis)
        if not all(resume_mask):
            present = torch.tensor(
                resume_mask, dtype=torch.bool, device=device
            ).view(sequences, 1, 1, 1)
            state.masked_fill_(~present, 0.0)
    return state


def _make_inputs(
    case: Case,
    *,
    heads: int,
    value_heads: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    total = sum(case.seq_lens)
    qk_shape = (1, total, heads, HEAD_DIM)
    value_shape = (1, total, value_heads, HEAD_DIM)

    def projection(shape: tuple[int, ...]) -> torch.Tensor:
        return F.silu(torch.randn(shape, device=device, dtype=torch.float32)).to(
            torch.bfloat16
        )

    offsets = [0]
    for length in case.seq_lens:
        offsets.append(offsets[-1] + length)

    inputs: dict[str, object] = {
        "q": projection(qk_shape),
        "k": projection(qk_shape),
        "v": projection(value_shape),
        "g": torch.randn(value_shape, device=device, dtype=torch.bfloat16),
        # This mirrors ATOM: the beta projection is BF16 and is widened before
        # the fused in-kernel sigmoid.
        "beta": torch.randn(
            (1, total, value_heads), device=device, dtype=torch.bfloat16
        ).float(),
        "A_log": torch.empty(value_heads, device=device, dtype=torch.float32)
        .uniform_(1.0, 16.0)
        .log_(),
        "dt_bias": torch.randn(
            value_heads * HEAD_DIM, device=device, dtype=torch.float32
        ),
        "scale": HEAD_DIM**-0.5,
        "initial_state": _make_state(
            len(case.seq_lens),
            value_heads,
            resume_mask=_case_resume_mask(case),
            device=device,
        ),
        "cu_seqlens": torch.tensor(offsets, device=device, dtype=torch.int32),
        # ATOM/vLLM supplies the graph-bucket maximum from host metadata.  A
        # standalone eager benchmark has the exact batch lengths available,
        # so use their true maximum rather than silently exercising raw-v2
        # with its legacy zero hint.
        "max_seqlen_upper_bound": max(case.seq_lens),
    }
    assert inputs["dt_bias"].ndim == 1
    assert inputs["cu_seqlens"].dtype == torch.int32
    return inputs


def _native_supported(inputs: dict[str, object]) -> bool:
    return flash_kda_native_supported(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=inputs["cu_seqlens"],
        max_seqlen_upper_bound=inputs["max_seqlen_upper_bound"],
    )


def _build_backends(
    inputs: dict[str, object], selected: tuple[str, ...], *, public_k3: bool = False
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]]:
    def public(backend: str | None):
        # ATOM omits both ``backend`` and the default scale, but supplies the
        # static query-length bucket.  Preserve that resolver contract; only
        # comparison rows force a backend.
        backend_kwargs = {} if backend is None else {"backend": backend}
        return chunk_kimi_delta_attn(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            state_v_first=True,
            cu_seqlens=inputs["cu_seqlens"],
            max_seqlen_upper_bound=inputs["max_seqlen_upper_bound"],
            **backend_kwargs,
        )

    def native():
        if public_k3:
            # ``None`` deliberately omits the argument and exercises the
            # production zero-environment resolver used by ATOM.
            return public(None)
        return flash_kda_native_fwd(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            scale=inputs["scale"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            lower_bound=LOWER_BOUND,
            cu_seqlens=inputs["cu_seqlens"],
            max_seqlen_upper_bound=inputs["max_seqlen_upper_bound"],
        )

    def triton_flash():
        return public("triton")

    def baseline():
        return public("baseline")

    implementations = {
        "native": native,
        # Internal witness used by the formal public-K3 runner.  The ordinary
        # CLI intentionally keeps its historical backend choices unchanged.
        "explicit-native": lambda: public("native"),
        "triton": triton_flash,
        "baseline": baseline,
    }
    return {name: implementations[name] for name in selected}


def _time_once(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    milliseconds = float(start.elapsed_time(end))
    del result
    return milliseconds


def _peak_memory_mib(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    gc.collect()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    del result
    torch.cuda.synchronize()
    return peak / (1024**2)


def _check_graph_replay_matches_eager(
    case_name: str,
    backend: str,
    replay_output: tuple[torch.Tensor, torch.Tensor | None],
    eager_output: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, bool]:
    """Require static-input graph replay to be bitwise identical to eager."""

    replay_o, replay_state = replay_output
    eager_o, eager_state = eager_output
    if (replay_state is None) != (eager_state is None):
        raise RuntimeError(
            f"{case_name}/{backend}: graph replay and eager disagree on "
            "whether final state is present"
        )

    pairs = [("output", replay_o, eager_o)]
    if replay_state is not None:
        assert eager_state is not None
        pairs.append(("final_state", replay_state, eager_state))

    matches: dict[str, bool] = {}
    for tensor_name, replay_tensor, eager_tensor in pairs:
        if (
            replay_tensor.shape != eager_tensor.shape
            or replay_tensor.dtype != eager_tensor.dtype
            or replay_tensor.device != eager_tensor.device
        ):
            raise RuntimeError(
                f"{case_name}/{backend}: graph replay {tensor_name} metadata "
                f"{tuple(replay_tensor.shape)}/{replay_tensor.dtype}/"
                f"{replay_tensor.device} differs from eager "
                f"{tuple(eager_tensor.shape)}/{eager_tensor.dtype}/"
                f"{eager_tensor.device}"
            )
        if not bool(torch.isfinite(replay_tensor).all().item()):
            raise RuntimeError(
                f"{case_name}/{backend}: graph replay produced non-finite "
                f"{tensor_name}"
            )
        if not bool(torch.isfinite(eager_tensor).all().item()):
            raise RuntimeError(
                f"{case_name}/{backend}: eager reference produced non-finite "
                f"{tensor_name}"
            )
        equal = _storage_bytes_equal(replay_tensor, eager_tensor)
        matches[f"graph_eager_{tensor_name}_bitwise_equal"] = equal
        if not equal:
            relative_rms, max_abs = _errors(replay_tensor, eager_tensor)
            raise RuntimeError(
                f"{case_name}/{backend}: graph replay {tensor_name} is not "
                f"bitwise equal to eager (relative RMS={relative_rms:.6g}, "
                f"max abs={max_abs:.6g})"
            )
    return matches


def _capture_backends(
    case_name: str,
    backends: dict[
        str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]
    ],
    eager_outputs: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
    *,
    keep_graph: bool = False,
    input_state_guard: Callable[[str], None] | None = None,
) -> tuple[dict[str, _CapturedBackend], dict[str, dict[str, bool]]]:
    """Capture each backend independently after eager compilation/checking."""

    captured_backends: dict[str, _CapturedBackend] = {}
    replay_correctness: dict[str, dict[str, bool]] = {}
    for name, fn in backends.items():
        # A normal eager call above performs compilation and correctness.  One
        # additional side-stream call initializes any stream-local runtime
        # state before this backend enters capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            side_stream_output = fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        del side_stream_output
        if input_state_guard is not None:
            input_state_guard(f"after_{name}_side_stream_warmup")

        graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
        with torch.cuda.graph(graph, stream=stream):
            captured_output = fn()
        torch.cuda.synchronize()
        if input_state_guard is not None:
            input_state_guard(f"after_{name}_graph_capture")

        captured = _CapturedBackend(
            graph=graph, output=captured_output, stream=stream
        )
        # Poison both static outputs before replay.  Comparing capture-time
        # buffers without doing this cannot detect an empty/incomplete graph:
        # stale output would otherwise satisfy the eager comparison.
        for tensor in captured.output:
            if tensor is not None:
                tensor.fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        if input_state_guard is not None:
            input_state_guard(f"after_{name}_poison_graph_replay")
        replay_correctness[name] = _check_graph_replay_matches_eager(
            case_name, name, captured.output, eager_outputs[name]
        )
        captured_backends[name] = captured

    return captured_backends, replay_correctness


def _graph_records(
    captured: _CapturedBackend, device: torch.device
) -> list[dict[str, object]]:
    """Return HIP graph nodes and launch data for formal route auditing."""

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGraphGetNodes.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    )
    hip.hipGraphGetNodes.restype = ctypes.c_int
    hip.hipGraphNodeGetType.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )
    hip.hipGraphNodeGetType.restype = ctypes.c_int
    hip.hipGraphKernelNodeGetParams.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_HipKernelNodeParams),
    )
    hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
    hip.hipKernelNameRef.argtypes = (ctypes.c_void_p,)
    hip.hipKernelNameRef.restype = ctypes.c_char_p
    hip.hipKernelNameRefByPtr.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

    def check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with HIP status {status}")

    graph_handle = ctypes.c_void_p(captured.graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    check(
        hip.hipGraphGetNodes(graph_handle, None, ctypes.byref(count)),
        "hipGraphGetNodes(count)",
    )
    nodes = (ctypes.c_void_p * int(count.value))()
    check(
        hip.hipGraphGetNodes(graph_handle, nodes, ctypes.byref(count)),
        "hipGraphGetNodes(nodes)",
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    records: list[dict[str, object]] = []
    for index, node in enumerate(nodes[: int(count.value)]):
        node_type = ctypes.c_int()
        check(
            hip.hipGraphNodeGetType(node, ctypes.byref(node_type)),
            f"hipGraphNodeGetType({index})",
        )
        record: dict[str, object] = {
            "index": index,
            "node_type": int(node_type.value),
        }
        if node_type.value == 0:
            params = _HipKernelNodeParams()
            check(
                hip.hipGraphKernelNodeGetParams(node, ctypes.byref(params)),
                f"hipGraphKernelNodeGetParams({index})",
            )
            encoded_name = hip.hipKernelNameRefByPtr(params.func, stream)
            if encoded_name is None:
                encoded_name = hip.hipKernelNameRef(params.func)
            record.update(
                {
                    "name": (
                        encoded_name.decode(errors="replace")
                        if encoded_name is not None
                        else "<unknown>"
                    ),
                    "grid": [
                        int(params.grid_dim.x),
                        int(params.grid_dim.y),
                        int(params.grid_dim.z),
                    ],
                    "block": [
                        int(params.block_dim.x),
                        int(params.block_dim.y),
                        int(params.block_dim.z),
                    ],
                    "shared_mem_bytes": int(params.shared_mem_bytes),
                }
            )
        records.append(record)
    return records


def _canonical_graph_signature(
    records: list[dict[str, object]],
) -> tuple[tuple[object, ...], ...]:
    entries = [
        (
            record.get("node_type"),
            record.get("name"),
            tuple(record.get("grid", [])),
            tuple(record.get("block", [])),
            record.get("shared_mem_bytes"),
        )
        for record in records
    ]
    return tuple(sorted(entries, key=repr))


def _classify_graph_route(
    backend: str, records: list[dict[str, object]]
) -> dict[str, object]:
    kernels = [record for record in records if record.get("node_type") == 0]
    names = [str(record.get("name", "")) for record in kernels]
    lowered = [name.lower() for name in names]
    unresolved = [name for name in names if name in ("", "<unknown>")]
    native_k1 = [name for name, low in zip(names, lowered) if "k1_kda_" in low]
    native_k2 = [name for name, low in zip(names, lowered) if "k2_kda_" in low]
    triton_prepare = [
        name for name, low in zip(names, lowered) if "flash_kda_prepare" in low
    ]
    triton_segment = [
        name
        for name, low in zip(names, lowered)
        if "flash_kda_segment" in low or "flash_kda_seg_scan" in low
    ]
    native_route = backend in ("native", "explicit-native")
    verified = (
        not unresolved
        and bool(native_k1 and native_k2)
        and not (triton_prepare or triton_segment)
        if native_route
        else not unresolved
        and bool(triton_prepare and triton_segment)
        and not (native_k1 or native_k2)
    )
    if not verified:
        raise RuntimeError(
            f"{backend}: graph route verification failed; "
            f"native_k1={len(native_k1)}, native_k2={len(native_k2)}, "
            f"triton_prepare={len(triton_prepare)}, "
            f"triton_segment={len(triton_segment)}, "
            f"unresolved={len(unresolved)}, names={names}"
        )
    for record in kernels:
        for field in ("grid", "block"):
            dimensions = record.get(field)
            if not isinstance(dimensions, list) or len(dimensions) != 3 or any(
                type(value) is not int or value <= 0 for value in dimensions
            ):
                raise RuntimeError(
                    f"{backend}: invalid {field} for {record.get('name')}: "
                    f"{dimensions}"
                )
    return {
        "route": "native-hip-k1-k2"
        if native_route
        else "triton-flash-kda-prepare-segment",
        "route_verified": True,
        "node_count": len(records),
        "kernel_node_count": len(kernels),
        "kernel_names": names,
        "records": records,
    }


def _audit_captured_routes(
    captured_backends: dict[str, _CapturedBackend],
) -> dict[str, object]:
    required = {"native", "explicit-native", "triton"}
    if set(captured_backends) != required:
        raise RuntimeError(
            f"formal route audit requires {sorted(required)}, got "
            f"{sorted(captured_backends)}"
        )
    graph_ids = [
        int(captured.graph.raw_cuda_graph())
        for captured in captured_backends.values()
    ]
    stream_ids = [
        int(captured.stream.cuda_stream) for captured in captured_backends.values()
    ]
    if len(set(graph_ids)) != len(graph_ids) or len(set(stream_ids)) != len(
        stream_ids
    ):
        raise RuntimeError("formal route audit found aliased graphs or streams")

    device = torch.device("cuda", torch.cuda.current_device())
    records = {
        name: _graph_records(captured, device)
        for name, captured in captured_backends.items()
    }
    routes = {
        name: _classify_graph_route(name, backend_records)
        for name, backend_records in records.items()
    }
    signatures_equal = _canonical_graph_signature(
        records["native"]
    ) == _canonical_graph_signature(records["explicit-native"])
    if not signatures_equal:
        raise RuntimeError(
            "public default and explicit native graph signatures differ"
        )
    return {
        "all_routes_verified": True,
        "graphs_independent": True,
        "streams_independent": True,
        "public_explicit_graph_signatures_equal": True,
        "graph_handles": graph_ids,
        "stream_handles": stream_ids,
        "backends": routes,
    }


@torch.inference_mode()
def _benchmark_case(
    case: Case,
    *,
    heads: int,
    value_heads: int,
    selected: tuple[str, ...],
    warmup: int,
    repeat: int,
    seed: int,
    check: bool,
    tolerance: float,
    execution: str = "eager",
    public_k3: bool = False,
    initial_state_none: bool = False,
    check_input_state_immutability: bool = False,
    timed_selected: tuple[str, ...] | None = None,
    audit_graph_routes: bool = False,
    raw_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if execution not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {execution}")
    if timed_selected is not None and (
        not timed_selected or not set(timed_selected).issubset(selected)
    ):
        raise ValueError("timed_selected must be a nonempty subset of selected")
    if audit_graph_routes and (
        execution != "graph"
        or not public_k3
        or set(selected) != {"native", "explicit-native", "triton"}
    ):
        raise ValueError(
            "route audit requires public graph execution with native, "
            "explicit-native, and triton"
        )
    inputs = _make_inputs(
        case,
        heads=heads,
        value_heads=value_heads,
        seed=seed,
        device=torch.device("cuda"),
    )
    if initial_state_none:
        if any(_case_resume_mask(case)):
            raise ValueError(
                f"{case.name}: literal initial_state=None is only valid for "
                "a fully fresh case"
            )
        # The general benchmark materializes a zero FP32 state so it can
        # compare fresh and resume allocations uniformly.  Formal public-K3
        # coverage also needs the production call contract where fresh state
        # is absent, not merely numerically zero.
        inputs["initial_state"] = None

    original_initial_state = inputs["initial_state"]
    assert original_initial_state is None or isinstance(
        original_initial_state, torch.Tensor
    )
    initial_state_contract = _assert_initial_state_contract(
        case,
        original_initial_state,
        literal_none=initial_state_none,
    )
    initial_state_snapshot = (
        original_initial_state.detach().clone()
        if check_input_state_immutability and original_initial_state is not None
        else None
    )
    input_state_checks: list[str] = []

    def check_input_state(phase: str) -> None:
        if not check_input_state_immutability:
            return
        current = inputs["initial_state"]
        if original_initial_state is None:
            if current is not None:
                raise RuntimeError(
                    f"{case.name}/{phase}: initial_state=None was replaced"
                )
        else:
            if current is not original_initial_state:
                raise RuntimeError(
                    f"{case.name}/{phase}: initial_state object was replaced"
                )
            assert initial_state_snapshot is not None
            if (
                current.shape != initial_state_snapshot.shape
                or current.stride() != initial_state_snapshot.stride()
                or current.dtype != initial_state_snapshot.dtype
                or current.device != initial_state_snapshot.device
            ):
                raise RuntimeError(
                    f"{case.name}/{phase}: initial_state metadata changed"
                )
            if not _storage_bytes_equal(current, initial_state_snapshot):
                raise RuntimeError(
                    f"{case.name}/{phase}: initial_state storage bytes changed"
                )
        input_state_checks.append(phase)

    check_input_state("before_backend_setup")
    if "native" in selected and not _native_supported(inputs):
        raise RuntimeError(
            "native FlashKDA rejected the exact K3 inputs; check gfx942/gfx950, "
            "AITER_TRITON_ONLY, dtypes, and the extension build"
        )
    backends = _build_backends(inputs, selected, public_k3=public_k3)

    # Graph capture must never be the first call: finish compilation and retain
    # a same-backend eager reference even when cross-backend checking is
    # explicitly skipped.
    eager_outputs = None
    output_contracts: dict[str, dict[str, object]] = {}
    if check or execution == "graph":
        eager_outputs = {}
        for name, fn in backends.items():
            eager_outputs[name] = fn()
            torch.cuda.synchronize()
            output_contracts[name] = _assert_output_contract(
                case, name, eager_outputs[name], inputs
            )
            check_input_state(f"after_{name}_eager")
    if eager_outputs is not None:
        torch.cuda.synchronize()

    correctness: dict[str, dict[str, object]] = {}
    public_default_bitwise_native: bool | None = None
    if check:
        assert eager_outputs is not None
        outputs = eager_outputs
        if public_k3 and "native" in outputs:
            # Prove that the zero-env default actually selected native rather
            # than merely producing a numerically close Triton result.
            explicit_native = outputs.get("explicit-native")
            if explicit_native is None:
                explicit_native = chunk_kimi_delta_attn(
                    q=inputs["q"],
                    k=inputs["k"],
                    v=inputs["v"],
                    g=inputs["g"],
                    beta=inputs["beta"],
                    A_log=inputs["A_log"],
                    dt_bias=inputs["dt_bias"],
                    initial_state=inputs["initial_state"],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    safe_gate=True,
                    lower_bound=LOWER_BOUND,
                    state_v_first=True,
                    cu_seqlens=inputs["cu_seqlens"],
                    max_seqlen_upper_bound=inputs[
                        "max_seqlen_upper_bound"
                    ],
                    backend="native",
                )
                torch.cuda.synchronize()
                check_input_state("after_explicit_native_eager")
            default_o, default_state = outputs["native"]
            explicit_o, explicit_state = explicit_native
            if not (
                _storage_bytes_equal(default_o, explicit_o)
                and default_state is not None
                and explicit_state is not None
                and _storage_bytes_equal(default_state, explicit_state)
            ):
                raise RuntimeError(
                    f"{case.name}: zero-env K3 default is not bitwise native"
                )
            public_default_bitwise_native = True
            del explicit_native, explicit_o, explicit_state
        torch.cuda.synchronize()
        reference_name = "triton" if "triton" in outputs else selected[0]
        reference_o, reference_state = outputs[reference_name]
        assert reference_state is not None
        for name, (output, final_state) in outputs.items():
            assert final_state is not None
            for tensor_name, tensor in (
                ("output", output),
                ("final_state", final_state),
            ):
                if not bool(torch.isfinite(tensor).all().item()):
                    raise RuntimeError(
                        f"{case.name}/{name} produced non-finite {tensor_name}"
                    )
            output_rms, output_max = _errors(output, reference_o)
            output_max_sequence_rms = (
                _packed_output_max_sequence_relative_rms(
                    output, reference_o, case.seq_lens
                )
            )
            state_rms, state_max = _errors(final_state, reference_state)
            state_max_sequence_rms = _max_sequence_relative_rms(
                final_state, reference_state
            )
            metrics: dict[str, object] = {
                "error_reference": reference_name,
                "output_relative_rms": output_rms,
                "output_max_sequence_relative_rms": output_max_sequence_rms,
                "output_max_abs": output_max,
                "state_relative_rms": state_rms,
                "state_max_abs": state_max,
                "state_max_sequence_relative_rms": state_max_sequence_rms,
            }
            metrics.update(output_contracts[name])
            if name == "native" and public_default_bitwise_native is not None:
                metrics["public_default_bitwise_native"] = (
                    public_default_bitwise_native
                )
            checked_relative_rms = [
                output_rms,
                output_max_sequence_rms,
                state_rms,
                state_max_sequence_rms,
            ]
            decode_output = _mixed_decode_output(output, case)
            reference_decode_output = _mixed_decode_output(reference_o, case)
            if decode_output is not None:
                assert reference_decode_output is not None
                decode_rms, decode_max = _errors(
                    decode_output, reference_decode_output
                )
                metrics.update(
                    {
                        "decode_output_relative_rms": decode_rms,
                        "decode_output_max_abs": decode_max,
                    }
                )
                checked_relative_rms.append(decode_rms)
            correctness[name] = metrics
            has_nonfinite_error = any(
                not math.isfinite(value) for value in checked_relative_rms
            )
            if name != reference_name and (
                has_nonfinite_error
                or max(checked_relative_rms) > tolerance
            ):
                raise RuntimeError(
                    f"{case.name}/{name} exceeds relative-RMS tolerance "
                    f"{tolerance}: output={output_rms:.6g}, state={state_rms:.6g}, "
                    f"max_sequence_output={output_max_sequence_rms:.6g}, "
                    f"max_sequence_state={state_max_sequence_rms:.6g}, "
                    f"decode_output={metrics.get('decode_output_relative_rms', '-')}"
                )
        del outputs, reference_o, reference_state, output, final_state
        del decode_output, reference_decode_output

    captured_backends: dict[str, _CapturedBackend] = {}
    route_audit: dict[str, object] | None = None
    if execution == "graph":
        assert eager_outputs is not None
        captured_backends, graph_correctness = _capture_backends(
            case.name,
            backends,
            eager_outputs,
            keep_graph=audit_graph_routes,
            input_state_guard=check_input_state,
        )
        for name, metrics in graph_correctness.items():
            correctness.setdefault(name, {}).update(metrics)
        if audit_graph_routes:
            route_audit = _audit_captured_routes(captured_backends)
            public_output = captured_backends["native"].output
            explicit_output = captured_backends["explicit-native"].output
            for tensor_name, public_tensor, explicit_tensor in (
                ("output", public_output[0], explicit_output[0]),
                ("final_state", public_output[1], explicit_output[1]),
            ):
                if public_tensor is None or explicit_tensor is None:
                    raise RuntimeError(
                        f"{case.name}: graph {tensor_name} witness is missing"
                    )
                if not _storage_bytes_equal(public_tensor, explicit_tensor):
                    raise RuntimeError(
                        f"{case.name}: public and explicit native graph "
                        f"{tensor_name} bytes differ"
                    )
            check_input_state("after_graph_route_audit")
        # Keep captured_backends alive for the static outputs and streams, but
        # time the graph object's replay method directly: no Python operator
        # call or output allocation occurs inside the timed interval.
        timed_backends: dict[str, Callable[[], object]] = {
            name: captured.graph.replay
            for name, captured in captured_backends.items()
        }
    else:
        timed_backends = dict(backends)

    if timed_selected is not None:
        timed_backends = {name: timed_backends[name] for name in timed_selected}

    peak_memory = {}
    for name, fn in timed_backends.items():
        peak_memory[name] = _peak_memory_mib(fn)
        check_input_state(f"after_{name}_peak_memory_replay")

    # Rotate the first backend in every round.  This prevents one implementation
    # from always measuring at the same clock/temperature point.
    names = list(timed_backends)
    for index in range(warmup):
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            result = timed_backends[name]()
            del result
    torch.cuda.synchronize()
    check_input_state("after_timing_warmup")

    samples: dict[str, list[float]] = {name: [] for name in names}
    for index in range(repeat):
        offset = index % len(names)
        order = names[offset:] + names[:offset]
        for order_index, name in enumerate(order):
            elapsed = _time_once(timed_backends[name])
            samples[name].append(elapsed)
            if raw_rows is not None:
                raw_rows.append(
                    {
                        "case": case.name,
                        "backend": name,
                        "execution": execution,
                        "round": index,
                        "order": order_index,
                        "seed": seed,
                        "sequences": len(case.seq_lens),
                        "tokens": sum(case.seq_lens),
                        "heads": heads,
                        "value_heads": value_heads,
                        "state": _case_state_label(case),
                        "latency_ms": elapsed,
                    }
                )

    for name, values in samples.items():
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise RuntimeError(
                f"{case.name}/{name}: CUDA event produced a nonpositive or "
                "nonfinite latency sample"
            )
    check_input_state("after_measured_timing")

    if audit_graph_routes:
        assert eager_outputs is not None
        for name, captured in captured_backends.items():
            captured.replay()
            torch.cuda.synchronize()
            check_input_state(f"after_{name}_final_graph_replay")
            final_checks = _check_graph_replay_matches_eager(
                case.name, name, captured.output, eager_outputs[name]
            )
            correctness.setdefault(name, {}).update(
                {
                    key.replace("graph_eager_", "final_graph_eager_"): value
                    for key, value in final_checks.items()
                }
            )
        final_public = captured_backends["native"].output
        final_explicit = captured_backends["explicit-native"].output
        if any(
            left is None
            or right is None
            or not _storage_bytes_equal(left, right)
            for left, right in zip(final_public, final_explicit)
        ):
            raise RuntimeError(
                f"{case.name}: final public and explicit native graph bytes differ"
            )
    del eager_outputs

    medians = {name: statistics.median(values) for name, values in samples.items()}
    triton_median = medians.get("triton")
    paired_vs_triton: dict[str, dict[str, float]] = {}
    if "triton" in samples:
        triton_samples = samples["triton"]
        for name, candidate_samples in samples.items():
            if name == "triton":
                continue
            # Every backend is measured exactly once per rotated round.  Zip
            # equal round indices so clock/temperature drift is represented by
            # a paired endpoint rather than only by independent medians.
            delta_us = [
                (candidate - triton) * 1000.0
                for candidate, triton in zip(
                    candidate_samples, triton_samples
                )
            ]
            speedups = [
                triton / candidate
                for candidate, triton in zip(
                    candidate_samples, triton_samples
                )
            ]
            paired_vs_triton[name] = {
                "paired_candidate_minus_triton_p10_us": _percentile(
                    delta_us, 0.10
                ),
                "paired_candidate_minus_triton_median_us": (
                    statistics.median(delta_us)
                ),
                "paired_candidate_minus_triton_p90_us": _percentile(
                    delta_us, 0.90
                ),
                "paired_speedup_median": statistics.median(speedups),
                "paired_win_fraction": sum(
                    candidate < triton
                    for candidate, triton in zip(
                        candidate_samples, triton_samples
                    )
                )
                / len(candidate_samples),
            }
    total_tokens = sum(case.seq_lens)
    rows: list[dict[str, object]] = []
    for name in names:
        values = samples[name]
        median = medians[name]
        row: dict[str, object] = {
            "case": case.name,
            "backend": name,
            "execution": execution,
            "sequences": len(case.seq_lens),
            "tokens": total_tokens,
            "heads": heads,
            "value_heads": value_heads,
            "seed": seed,
            "state": _case_state_label(case),
            "latency_min_ms": min(values),
            "latency_p10_ms": _percentile(values, 0.10),
            "latency_median_ms": median,
            "latency_p90_ms": _percentile(values, 0.90),
            "latency_mean_ms": statistics.mean(values),
            "latency_std_ms": statistics.pstdev(values),
            "tokens_per_second": total_tokens * 1000.0 / median,
            "speedup_vs_triton": (
                triton_median / median if triton_median is not None else None
            ),
            "peak_memory_mib": peak_memory[name],
            "samples": len(values),
        }
        row.update(paired_vs_triton.get(name, {}))
        row.update(correctness.get(name, {}))
        if check_input_state_immutability:
            row["input_initial_state_unchanged"] = True
            row["input_state_immutability_checks"] = len(input_state_checks)
        row.update(initial_state_contract)
        if name == "native" and route_audit is not None:
            row["graph_route_audit"] = route_audit
        rows.append(row)

    del timed_backends, captured_backends, backends, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _print_environment(heads: int, value_heads: int) -> dict[str, object]:
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def source_fingerprint() -> str:
        source_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
        roots = (
            _REPO_ROOT / "csrc" / "kernels" / "flash_kda",
            _REPO_ROOT / "aiter" / "ops" / "triton" / "kimi_delta_attn",
            _REPO_ROOT
            / "aiter"
            / "ops"
            / "triton"
            / "_triton_kernels"
            / "chunk_delta_attn",
            _REPO_ROOT
            / "aiter"
            / "ops"
            / "triton"
            / "_triton_kernels"
            / "gated_delta_rule",
        )
        files = [
            _REPO_ROOT / "aiter" / "ops" / "flash_kda.py",
            _REPO_ROOT / "csrc" / "include" / "flash_kda.h",
            _REPO_ROOT / "csrc" / "pybind" / "flash_kda_pybind.cu",
        ]
        for root in roots:
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in source_suffixes
            )
        files.append(Path(__file__).resolve())
        digest = hashlib.sha256()
        for path in sorted(set(files)):
            relative = path.relative_to(_REPO_ROOT)
            digest.update(str(relative).encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
        return digest.hexdigest()

    def git_output(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch_detail = getattr(properties, "gcnArchName", "unknown")
    arch = arch_detail.split(":", 1)[0]
    jit_dir_value = os.getenv("AITER_JIT_DIR")
    module_path = (
        Path(jit_dir_value).resolve() / "module_flash_kda_hip.so"
        if jit_dir_value
        else None
    )
    module_sha256 = (
        sha256_file(module_path)
        if module_path is not None and module_path.is_file()
        else None
    )
    loaded_module_identities: dict[str, dict[str, object]] = {}
    if jit_dir_value:
        jit_dir = Path(jit_dir_value).resolve()
        for module_name in ("module_aiter_core", "module_flash_kda_hip"):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            loaded_raw = getattr(module, "__file__", None)
            if not loaded_raw:
                raise RuntimeError(
                    f"loaded {module_name} does not expose its file path"
                )
            loaded_path = Path(loaded_raw).resolve()
            expected_path = jit_dir / f"{module_name}.so"
            if loaded_path != expected_path:
                raise RuntimeError(
                    f"loaded {module_name} from {loaded_path}, expected "
                    f"{expected_path}"
                )
            loaded_module_identities[module_name] = {
                "path": str(loaded_path),
                "sha256": sha256_file(loaded_path),
                "matches_expected_jit_path": True,
            }
    controlled_environment = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith("FLASH_KDA_")
        or name
        in {
            "AITER_JIT_DIR",
            "AITER_KDA_BACKEND",
            "AITER_AOT_IMPORT",
            "AITER_REBUILD",
            "AITER_TRITON_ONLY",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "MAX_JOBS",
            "NUMEXPR_NUM_THREADS",
            "GPU_ARCHS",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "TRITON_CACHE_DIR",
        }
    }
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "arch": arch,
        "arch_detail": arch_detail,
        "compute_units": properties.multi_processor_count,
        "pytorch": torch.__version__,
        "rocm": torch.version.hip,
        "heads": heads,
        "value_heads": value_heads,
        "head_dim": HEAD_DIM,
        "tp_size": K3_TP_SIZE,
        "module_path": str(module_path) if module_path is not None else None,
        "module_sha256": module_sha256,
        "loaded_module_identities": loaded_module_identities,
        "source_fingerprint_sha256": source_fingerprint(),
        "benchmark_sha256": sha256_file(Path(__file__).resolve()),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--short"),
        "controlled_environment": controlled_environment,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    print(
        f"GPU: {metadata['gpu']} ({arch}, {metadata['compute_units']} CUs); "
        f"PyTorch {torch.__version__}; ROCm {torch.version.hip}"
    )
    print(
        f"KDA per rank: TP={K3_TP_SIZE}, Hq={heads}, HV={value_heads}, "
        f"K=V={HEAD_DIM}; "
        "packed cu_seqlens=int32, beta/state=fp32"
    )
    return metadata


def _print_rows(rows: list[dict[str, object]]) -> None:
    print(
        "\n| case | seed | backend | execution | N | tokens | state | "
        "p10 / p50 / p90 ms | "
        "Mtoken/s | vs Triton | candidate−Triton Δp50 us | paired x / win | "
        "peak MiB | out RMS | decode RMS | state RMS | "
        "max-seq state RMS |"
    )
    print(
        "| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        speedup = row["speedup_vs_triton"]
        speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
        paired_delta = row.get("paired_candidate_minus_triton_median_us")
        paired_delta_text = (
            "-" if paired_delta is None else f"{paired_delta:+.2f}"
        )
        paired_speedup = row.get("paired_speedup_median")
        paired_win_fraction = row.get("paired_win_fraction")
        paired_summary_text = (
            "-"
            if paired_speedup is None or paired_win_fraction is None
            else f"{paired_speedup:.3f}x / {paired_win_fraction:.0%}"
        )
        output_rms = row.get("output_relative_rms")
        decode_rms = row.get("decode_output_relative_rms")
        state_rms = row.get("state_relative_rms")
        state_max_sequence_rms = row.get("state_max_sequence_relative_rms")
        output_text = "-" if output_rms is None else f"{output_rms:.3e}"
        decode_text = "-" if decode_rms is None else f"{decode_rms:.3e}"
        state_text = "-" if state_rms is None else f"{state_rms:.3e}"
        state_max_sequence_text = (
            "-"
            if state_max_sequence_rms is None
            else f"{state_max_sequence_rms:.3e}"
        )
        print(
            f"| {row['case']} | {row['seed']} | {row['backend']} | "
            f"{row['execution']} | {row['sequences']} | "
            f"{row['tokens']} | {row['state']} | {row['latency_p10_ms']:.4f} / "
            f"{row['latency_median_ms']:.4f} / {row['latency_p90_ms']:.4f} | "
            f"{row['tokens_per_second'] / 1e6:.3f} | {speedup_text} | "
            f"{paired_delta_text} | {paired_summary_text} | "
            f"{row['peak_memory_mib']:.1f} | {output_text} | {decode_text} | "
            f"{state_text} | {state_max_sequence_text} |"
        )


def _check_performance_gate(
    rows: list[dict[str, object]],
    *,
    min_speedup: float | None,
    min_geomean_speedup: float | None,
    min_paired_win_fraction: float | None,
) -> None:
    """Fail when native misses an explicitly requested Triton-relative gate."""

    if all(
        threshold is None
        for threshold in (
            min_speedup,
            min_geomean_speedup,
            min_paired_win_fraction,
        )
    ):
        return

    native_rows = [row for row in rows if row["backend"] == "native"]
    if not native_rows:
        raise RuntimeError("performance gate has no native result rows")

    errors: list[str] = []
    speedups: list[float] = []
    win_fractions: list[float] = []
    for row in native_rows:
        label = (
            f"{row['case']} (Hq={row['heads']}, HV={row['value_heads']}, "
            f"seed={row['seed']})"
        )
        speedup_value = row.get("speedup_vs_triton")
        if not isinstance(speedup_value, (int, float)) or not math.isfinite(
            speedup_value
        ):
            errors.append(f"{label}: missing finite speedup versus Triton")
        else:
            speedup = float(speedup_value)
            speedups.append(speedup)
            if min_speedup is not None and speedup < min_speedup:
                errors.append(
                    f"{label}: speedup {speedup:.4f}x is below "
                    f"{min_speedup:.4f}x"
                )

        win_value = row.get("paired_win_fraction")
        if not isinstance(win_value, (int, float)) or not math.isfinite(
            win_value
        ):
            errors.append(f"{label}: missing finite paired win fraction")
        else:
            win_fraction = float(win_value)
            win_fractions.append(win_fraction)
            if (
                min_paired_win_fraction is not None
                and win_fraction < min_paired_win_fraction
            ):
                errors.append(
                    f"{label}: paired win fraction {win_fraction:.1%} is below "
                    f"{min_paired_win_fraction:.1%}"
                )

    geomean = (
        math.exp(
            math.fsum(math.log(speedup) for speedup in speedups) / len(speedups)
        )
        if speedups and all(speedup > 0.0 for speedup in speedups)
        else math.nan
    )
    if min_geomean_speedup is not None and (
        not math.isfinite(geomean) or geomean < min_geomean_speedup
    ):
        errors.append(
            f"geometric-mean speedup {geomean:.4f}x is below "
            f"{min_geomean_speedup:.4f}x"
        )

    minimum_speedup = min(speedups, default=math.nan)
    minimum_win_fraction = min(win_fractions, default=math.nan)
    print(
        "Performance gate: "
        f"minimum speedup={minimum_speedup:.4f}x, "
        f"geomean speedup={geomean:.4f}x, "
        f"minimum paired win fraction={minimum_win_fraction:.1%}"
    )
    if errors:
        raise RuntimeError("performance gate failed:\n  - " + "\n  - ".join(errors))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {path}")


def _write_json(
    path: Path,
    *,
    metadata: dict[str, object],
    args: argparse.Namespace,
    cases: list[Case],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": metadata,
        "configuration": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "heads": args.heads,
            "value_heads": args.value_heads,
            "execution": args.execution,
            # ``requested_suite`` records the CLI default even when --case or
            # --seq-lens overrides it.  Evidence gates must use
            # ``selection_mode`` plus the effective ``suite`` so a one-case
            # probe cannot masquerade as a complete named sweep.
            "selection_mode": (
                "seq-lens"
                if args.seq_lens is not None
                else "case"
                if args.case
                else "suite"
            ),
            "requested_suite": args.suite,
            "suite": (
                args.suite
                if args.seq_lens is None and not args.case
                else None
            ),
            "tolerance": args.tolerance,
            "backends": list(args.backend),
            "public_k3": args.public_k3,
            "min_speedup": args.min_speedup,
            "min_geomean_speedup": args.min_geomean_speedup,
            "min_paired_win_fraction": args.min_paired_win_fraction,
            "require_arch": args.require_arch,
            "require_compute_units": args.require_compute_units,
        },
        "cases": [
            {
                "name": case.name,
                "seq_lens": list(case.seq_lens),
                "resume": case.resume,
                "resume_mask": (
                    list(case.resume_mask) if case.resume_mask is not None else None
                ),
                "state": _case_state_label(case),
                "max_seqlen_upper_bound": max(case.seq_lens),
            }
            for case in cases
        ],
        "results": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote JSON: {path}")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Benchmark only this named case; repeat the option for multiple cases.",
    )
    parser.add_argument(
        "--suite",
        choices=tuple(CASE_SUITES),
        default="core",
        help="Named case suite used when neither --case nor --seq-lens is given.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        metavar="T",
        help="Use one custom packed case instead of the named case sweep.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Use a nonzero asymmetric V-first state for --seq-lens.",
    )
    parser.add_argument("--heads", type=int, default=K3_LOCAL_HEADS)
    parser.add_argument(
        "--value-heads",
        type=int,
        help="Value/gate/state heads; defaults to --heads (set larger for GVA).",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKEND_ORDER,
        help="Backend to run; repeat as needed. Defaults to all three.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execution",
        choices=EXECUTION_MODES,
        default="eager",
        help=(
            "Time direct eager calls or independently captured steady-state "
            "ROCm graph replays."
        ),
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help=(
            "Skip cross-backend output/state comparison before timing; graph "
            "mode still verifies every replay against its eager result."
        ),
    )
    parser.add_argument(
        "--public-k3",
        action="store_true",
        help=(
            "Time the public chunk_kimi_delta_attn resolver: the 'native' row "
            "uses its zero-env default and the 'triton' row forces Triton."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.04,
        help="Maximum output/state relative RMS versus Triton FlashKDA.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help="Fail if any native median speedup versus Triton is smaller.",
    )
    parser.add_argument(
        "--min-geomean-speedup",
        type=float,
        help="Fail if the native geometric-mean speedup is smaller.",
    )
    parser.add_argument(
        "--min-paired-win-fraction",
        type=float,
        help="Fail if any native paired-round win fraction is smaller.",
    )
    parser.add_argument(
        "--require-arch",
        help="Fail before timing unless the visible GPU has this GCN arch.",
    )
    parser.add_argument(
        "--require-compute-units",
        type=int,
        help="Fail before timing unless the visible GPU has this CU count.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="?",
        const=Path("bench_flash_kda_native.csv"),
        help="Write flat results to this path.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        help="Write one latency row per alternating backend/round.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        nargs="?",
        const=Path("bench_flash_kda_native.json"),
        help="Write metadata and results to this path.",
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="Print named cases and exit."
    )
    args = parser.parse_args(argv)
    if args.backend is None:
        args.backend = list(BACKEND_ORDER)
    else:
        args.backend = list(dict.fromkeys(args.backend))
    if args.value_heads is None:
        args.value_heads = args.heads
    if args.heads <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error(
            "--heads and --repeat must be positive; --warmup must be nonnegative"
        )
    if args.value_heads < args.heads or args.value_heads % args.heads != 0:
        parser.error("--value-heads must be a positive multiple of --heads")
    if args.seq_lens is not None and any(length <= 0 for length in args.seq_lens):
        parser.error("all --seq-lens values must be positive")
    if args.case and args.seq_lens:
        parser.error("--case and --seq-lens are mutually exclusive")
    if args.resume and args.seq_lens is None:
        parser.error("--resume only applies to a custom --seq-lens case")
    if args.public_k3 and args.skip_correctness:
        parser.error(
            "--public-k3 requires correctness checking to prove that the "
            "default resolver selected native"
        )
    gate_values = (args.min_speedup, args.min_geomean_speedup)
    if any(
        value is not None and (not math.isfinite(value) or value <= 0.0)
        for value in gate_values
    ):
        parser.error("speedup thresholds must be positive and finite")
    if args.min_paired_win_fraction is not None and (
        not math.isfinite(args.min_paired_win_fraction)
        or not 0.0 <= args.min_paired_win_fraction <= 1.0
    ):
        parser.error("--min-paired-win-fraction must be in [0, 1]")
    if args.require_compute_units is not None and args.require_compute_units <= 0:
        parser.error("--require-compute-units must be positive")
    if any(
        value is not None
        for value in (
            args.min_speedup,
            args.min_geomean_speedup,
            args.min_paired_win_fraction,
        )
    ) and not {"native", "triton"}.issubset(args.backend):
        parser.error("performance gates require --backend native and --backend triton")
    return args


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.list_cases:
        for case in CASE_SUITES[args.suite]:
            print(
                f"{case.name:16s} N={len(case.seq_lens):2d} "
                f"tokens={sum(case.seq_lens):6d} "
                f"state={_case_state_label(case)}"
            )
        return
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("This benchmark requires a ROCm GPU.")
    if os.getenv("AITER_TRITON_ONLY", "0") == "1" and "native" in args.backend:
        raise SystemExit("Unset AITER_TRITON_ONLY to benchmark native FlashKDA.")
    if args.public_k3 and os.getenv("AITER_KDA_BACKEND"):
        raise SystemExit("Unset AITER_KDA_BACKEND for a zero-env public K3 benchmark.")

    if args.seq_lens is not None:
        cases = [Case("custom", tuple(args.seq_lens), resume=args.resume)]
    elif args.case:
        selected_names = set(args.case)
        cases = [case for case in CASES if case.name in selected_names]
    else:
        cases = list(CASE_SUITES[args.suite])

    metadata = _print_environment(args.heads, args.value_heads)
    if args.require_arch is not None and metadata["arch"] != args.require_arch:
        raise SystemExit(
            f"required GPU arch {args.require_arch}, got {metadata['arch_detail']}"
        )
    if (
        args.require_compute_units is not None
        and metadata["compute_units"] != args.require_compute_units
    ):
        raise SystemExit(
            f"required {args.require_compute_units} compute units, "
            f"got {metadata['compute_units']}"
        )
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    selected_backends = tuple(args.backend)
    for case in cases:
        print(
            f"Running {case.name}: N={len(case.seq_lens)}, "
            f"tokens={sum(case.seq_lens)}, state={_case_state_label(case)}, "
            f"execution={args.execution}",
            flush=True,
        )
        rows.extend(
            _benchmark_case(
                case,
                heads=args.heads,
                value_heads=args.value_heads,
                selected=selected_backends,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                check=not args.skip_correctness,
                tolerance=args.tolerance,
                execution=args.execution,
                public_k3=args.public_k3,
                raw_rows=raw_rows,
            )
        )

    _print_rows(rows)
    if args.csv is not None:
        _write_csv(args.csv, rows)
    if args.raw_csv is not None:
        _write_csv(args.raw_csv, raw_rows)
    if args.json is not None:
        _write_json(
            args.json, metadata=metadata, args=args, cases=cases, rows=rows
        )
    _check_performance_gate(
        rows,
        min_speedup=args.min_speedup,
        min_geomean_speedup=args.min_geomean_speedup,
        min_paired_win_fraction=args.min_paired_win_fraction,
    )


if __name__ == "__main__":
    main()
