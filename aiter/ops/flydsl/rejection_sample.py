# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL standard speculative rejection sampling, 5 draft tokens, vocab 129280.

``flydsl_rejection_sample_standard`` verifies the draft tokens of one
speculative step as vLLM's ``rejection_sample`` does for the standard method
(Leviathan et al.) with a replicated draft-logits cache, no watermark, no block
verification and no FLy window, in two launches. Per request ``b`` (6 logits
rows: the 5 draft steps and the bonus row):

1. ``u = max(philox(seed, pos[6 b + j]), 2^-31)``, Triton's ``tl.rand``.
2. Temperature 0: accept iff the draft is the lowest-id argmax of the target
   row. Otherwise accept iff ``log p(x) > log u + log q(x)``, with
   ``q = softmax(draft / T)``. ``synthetic_rates``: accept iff ``u < rate[j]``.
3. The first rejected row (or the bonus row) gets one token: temperature 0 the
   target argmax; otherwise the argmax of ``residual + gumbel`` (residual
   ``log p + log1p(-q / p)`` where ``q < p``, the target row for the bonus row or
   a placeholder draft), with vLLM's murmur3 Gumbel draw of
   ``(seed, pos, vocab id)``.

The target rows may be the raw bf16 lm_head output (the kernel divides by the
request's temperature in fp32, as vLLM's ``apply_temperature``) or fp32 rows
with the temperature already applied. The kernel's fp32 sums and hardware
exp / log differ from Triton's in the last bits, so a decision whose margin is
at that rounding level can go the other way. The op test compares the tokens
with an fp64 reference (and with vLLM when it is installed) for every request
whose smallest decision margin exceeds 1e-4; closer decisions are not checked.

One workspace per device: launches on a device must be stream-ordered. Call
``warmup_rejection_sample_standard(device)`` before capturing a CUDA graph; it
compiles the kernels and allocates the workspace outside the capture.
"""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.rejection_sample_standard import (
    MAX_REQS,
    NB,
    NSPEC,
    ROWS,
    WORKSPACE_BYTES,
    V,
    build_rejection_resample,
    build_rejection_stats,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "REJECTION_SAMPLE_MAX_REQS",
    "REJECTION_SAMPLE_NUM_SPEC",
    "REJECTION_SAMPLE_VOCAB",
    "flydsl_rejection_sample_standard",
    "rejection_sample_standard_supported",
    "warmup_rejection_sample_standard",
]

REJECTION_SAMPLE_VOCAB = V
REJECTION_SAMPLE_NUM_SPEC = NSPEC
REJECTION_SAMPLE_MAX_REQS = MAX_REQS


# Per device: the workspace (stats, block partials, ticket counters; zeroed
# once, and every launch leaves its tickets at zero) and the placeholder for an
# absent ``synthetic_rates``.
_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _device_buffers(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bufs = _buffers.get(dev.index)
    if bufs is None:
        if torch.cuda.is_current_stream_capturing():
            # Allocated here, the buffers would come from the graph's private
            # pool and their zero-fill would be recorded in the graph.
            raise RuntimeError(
                "flydsl_rejection_sample_standard: call "
                "warmup_rejection_sample_standard(device) before graph capture"
            )
        bufs = (
            torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=dev),
            torch.zeros(NSPEC, dtype=torch.float32, device=dev),
        )
        _buffers[dev.index] = bufs
    return bufs


@functools.cache
def _stats_launcher(device_index: int, target_bf16: bool):
    # One compiled handle per device and variant.
    return build_rejection_stats(target_bf16)


@functools.cache
def _resample_launcher(device_index: int, synthetic: bool, target_bf16: bool):
    return build_rejection_resample(synthetic, target_bf16)


def rejection_sample_standard_supported(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    num_reqs: int,
    num_speculative_steps: int,
) -> bool:
    """Whether ``flydsl_rejection_sample_standard`` takes this step: gfx950,
    5 speculative steps, 1..1024 requests of exactly 6 logits rows each,
    contiguous fp32 or bf16 ``target_logits [6 B, 129280]`` and a contiguous bf16
    replicated draft cache ``[R, 5, 129280]`` indexed by request state."""
    return (
        target_logits.is_cuda
        and get_gfx_runtime() == "gfx950"
        and num_speculative_steps == NSPEC
        and 1 <= num_reqs <= MAX_REQS
        and target_logits.dtype in (torch.float32, torch.bfloat16)
        and tuple(target_logits.shape) == (num_reqs * ROWS, V)
        and target_logits.is_contiguous()
        and draft_logits.dtype == torch.bfloat16
        and draft_logits.dim() == 3
        and tuple(draft_logits.shape[1:]) == (NSPEC, V)
        and draft_logits.is_contiguous()
        and draft_logits.device == target_logits.device
    )


def flydsl_rejection_sample_standard(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    synthetic_rates: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard rejection sampling of 5 draft tokens per request.

    Args:
        target_logits: ``[B * 6, 129280]``, bf16 raw logits (the temperature is
            applied in the kernel) or fp32 with the temperature applied.
        draft_logits: bf16 ``[R, 5, 129280]`` pre-temperature draft logits, row
            ``idx_mapping[b]`` for request ``b``.
        draft_sampled: int64 ``[B * 6]``; entry ``6 b + j + 1`` is the draft token
            of step ``j`` (negative: a placeholder, always rejected).
        pos: int64 ``[B * 6]`` positions (the random streams).
        idx_mapping: int32 ``[B]`` request state of each request (>= 0).
        temperature: fp32 ``[R]``; 0 is greedy.
        seed: int64 ``[R]``.
        synthetic_rates: optional fp32 ``[5]`` conditional acceptance rates; the
            ratio test becomes ``u < rate[j]``.

    Returns:
        ``sampled`` int64 ``[B, 6]`` (the accepted drafts, the resampled token,
        then -1) and ``num_sampled`` int32 ``[B]``. CUDA-graph safe.
    """
    b = idx_mapping.shape[0]
    if not rejection_sample_standard_supported(target_logits, draft_logits, b, NSPEC):
        raise ValueError(
            "flydsl_rejection_sample_standard needs gfx950, 1..1024 requests of 6 "
            "contiguous fp32/bf16 target rows [6 B, 129280] and a contiguous bf16 "
            "draft cache [R, 5, 129280]"
        )
    dev = target_logits.device
    for t, dtype, numel, name in (
        (draft_sampled, torch.int64, b * ROWS, "draft_sampled"),
        (pos, torch.int64, b * ROWS, "pos"),
        (idx_mapping, torch.int32, b, "idx_mapping"),
        (temperature, torch.float32, None, "temperature"),
        (seed, torch.int64, None, "seed"),
    ):
        if not (
            t.dtype == dtype
            and t.dim() == 1
            and t.is_contiguous()
            and t.device == dev
            and (numel is None or t.numel() == numel)
        ):
            raise ValueError(
                f"{name} must be a contiguous 1-D {dtype} on the logits' device"
            )
    synthetic = synthetic_rates is not None
    if synthetic and not (
        synthetic_rates.dtype == torch.float32
        and tuple(synthetic_rates.shape) == (NSPEC,)
        and synthetic_rates.is_contiguous()
        and synthetic_rates.device == dev
    ):
        raise ValueError("synthetic_rates must be a contiguous fp32 [5]")
    target_bf16 = target_logits.dtype == torch.bfloat16
    ws, rates_placeholder = _device_buffers(dev)
    rates = synthetic_rates if synthetic else rates_placeholder
    sampled = torch.empty((b, ROWS), dtype=torch.int64, device=dev)
    num_sampled = torch.empty(b, dtype=torch.int32, device=dev)
    stream = torch.cuda.current_stream(dev)
    _run_compiled(
        _stats_launcher(dev.index, target_bf16),
        target_logits.data_ptr(),
        draft_logits.data_ptr(),
        idx_mapping.data_ptr(),
        temperature.data_ptr(),
        ws.data_ptr(),
        b * NSPEC,
        stream,
    )
    _run_compiled(
        _resample_launcher(dev.index, synthetic, target_bf16),
        target_logits.data_ptr(),
        draft_logits.data_ptr(),
        draft_sampled.data_ptr(),
        pos.data_ptr(),
        idx_mapping.data_ptr(),
        temperature.data_ptr(),
        seed.data_ptr(),
        rates.data_ptr(),
        ws.data_ptr(),
        sampled.data_ptr(),
        num_sampled.data_ptr(),
        b,
        NB * 8 * ((b + 7) // 8),
        stream,
    )
    return sampled, num_sampled


def warmup_rejection_sample_standard(device: torch.device | None = None) -> None:
    """Allocate the workspace on ``device`` and compile the four variants (fp32
    or bf16 target rows, ratio test or synthetic). Call it before capturing a
    CUDA graph that contains the op."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    _device_buffers(dev)
    z = functools.partial(torch.zeros, device=dev)
    for dtype in (torch.float32, torch.bfloat16):
        args = (
            z((ROWS, V), dtype=dtype),
            z((1, NSPEC, V), dtype=torch.bfloat16),
            z(ROWS, dtype=torch.int64),
            z(ROWS, dtype=torch.int64),
            z(1, dtype=torch.int32),
            z(1, dtype=torch.float32),
            z(1, dtype=torch.int64),
        )
        flydsl_rejection_sample_standard(*args)
        flydsl_rejection_sample_standard(*args, z(NSPEC, dtype=torch.float32))
    torch.cuda.synchronize(dev)
