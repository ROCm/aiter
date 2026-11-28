# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import functools

import torch
from torch import Generator, Tensor

from ..jit.core import compile_ops
from ..plugin import route as _route


def _philox_seed_offset(tensor: Tensor, generator: Generator | None = None):
    # The compiled kernels are torch-free and take a plain (seed, offset) pair
    # instead of an at::Generator. Mirror what the old C++ host code did with
    # gen->philox_cuda_state(counter_offset): read (seed, offset) from the torch
    # generator, then advance its offset by the same per-launch increment so that
    # back-to-back calls don't reuse the same random numbers.
    if generator is None:
        generator = torch.cuda.default_generators[tensor.device.index]
    numel = tensor.numel()
    M = tensor.size(0)
    # block_size=1024, unroll_factor=4, grid.x=M (see sample_kernels.cu)
    counter_offset = ((numel - 1) // (1024 * M * 4) + 1) * 4
    seed = generator.initial_seed()
    offset = generator.get_offset()
    generator.set_offset(offset + counter_offset)
    return int(seed), int(offset)


@compile_ops("module_sample", develop=True)
def greedy_sample(
    out: Tensor,
    input: Tensor,
) -> None: ...


@compile_ops("module_sample", develop=True)
def random_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperatures: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fc_name="random_sample", develop=True)
def _random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    eps: float = 1e-10,
) -> None: ...


def random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1.0,
    generator: Generator | None = None,
    eps: float = 1e-10,
) -> None:
    philox_seed, philox_offset = _philox_seed_offset(input, generator)
    _random_sample(out, input, temperatures, lambd, philox_seed, philox_offset, eps)


@compile_ops("module_sample", develop=True)
def mixed_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fc_name="mixed_sample", develop=True)
def _mixed_sample(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    eps: float = 1e-10,
) -> None: ...


def mixed_sample(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    generator: Generator | None = None,
    eps: float = 1e-10,
) -> None:
    philox_seed, philox_offset = _philox_seed_offset(input, generator)
    _mixed_sample(out, input, temperature, lambd, philox_seed, philox_offset, eps)


@compile_ops("module_sample", fc_name="exponential", develop=True)
def _exponential(
    out: Tensor,
    lambd: float = 1.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    eps: float = 1e-10,
) -> None: ...


def exponential(
    out: Tensor,
    lambd: float = 1.0,
    generator: Generator | None = None,
    eps: float = 1e-10,
) -> None:
    philox_seed, philox_offset = _philox_seed_offset(out, generator)
    _exponential(out, lambd, philox_seed, philox_offset, eps)


@compile_ops("module_sample", fc_name="spec_verify_pick", develop=True)
def _spec_verify_pick_hip(
    out_t: Tensor,
    out_b: Tensor,
    input: Tensor,
    expo_t: Tensor,
    expo_b: Tensor,
    temps: Tensor,
    target_idx: Tensor,
    bonus_idx: Tensor,
    cu_draft: Tensor,
    workspace: Tensor,
    split_rows: bool = False,
    eps: float = 1e-10,
    greedy_eps: float = 1e-10,
) -> None: ...


# int32 of workspace per row of a split launch (sample_kernels.cu, SvpWsPerRow).
_SPEC_VERIFY_PICK_WS_PER_ROW = 4 * 1024


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _spec_verify_pick_split_rows(rows: int, device: torch.device) -> bool:
    """Whether to spread each row over 16 one-wave workgroups.

    Unsplit, a row is one 1024-thread workgroup: 16 waves sharing a CU's 4
    SIMDs, ~33 us at V = 129280 whether the step has 1 row or one per CU. Split,
    the same 16 waves land on 16 SIMDs, ~14 us a row plus ~3 us of finishing
    pass while there is about a wave per SIMD; with more rows the waves share
    SIMDs again and the unsplit kernel, one pass, is as fast or faster. Split
    while the rows' waves (16 a row) fill at most half the SIMDs (4 a CU) --
    rows <= CUs / 2. The picks are the same either way.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    return 2 * rows <= _cu_count(index)


def _spec_verify_pick_fake(*args, **kwargs) -> None:
    """Both results are in-place writes (``out_t``, ``out_b``), so a trace of
    the op produces nothing."""
    return


def _spec_verify_pick_ref(
    input: Tensor,
    expo: Tensor,
    temps: Tensor,
    rows: Tensor,
    eps: float,
    greedy_eps: float,
) -> Tensor:
    """The picks of ``rows`` in torch: the definition the kernel implements,
    for inputs the kernel is not built for (a tensor off the GPU, a vocabulary
    past int32)."""
    logits = input.index_select(0, rows).float()
    greedy = logits.argmax(dim=-1)
    probs = torch.softmax(logits / temps.clamp_min(1e-5).unsqueeze(-1), dim=-1)
    sampled = (probs / (expo.to(probs.device) + eps)).argmax(dim=-1)
    return torch.where(temps <= greedy_eps, greedy, sampled).to(torch.int32)


@_route(
    op="spec_verify_pick",
    fake=_spec_verify_pick_fake,
    mutates_args=["out_t", "out_b"],
)
def spec_verify_pick(
    out_t: Tensor,
    out_b: Tensor,
    input: Tensor,
    expo_t: Tensor,
    expo_b: Tensor,
    temps: Tensor,
    target_idx: Tensor,
    bonus_idx: Tensor,
    cu_draft: Tensor,
    *,
    eps: float = 1e-10,
    greedy_eps: float = 1e-10,
) -> None:
    """The token picks of a sampled speculative-verification step, fused.

    Writes ``out_t`` and ``out_b`` in place and returns None.

    A step that verifies ``n_target`` draft rows for ``n_req`` requests picks
    a token on every verification row and a bonus token per request. Each row
    is read where it lies in ``input`` -- no ``index_select`` -- and picked as

    - ``temperature <= greedy_eps``: ``topk_select(row, 1, tie="low")``, the
      largest score under the integer-ordered key (-0.0 and 0.0 one score,
      NaN above +inf), ties to the smallest column;
    - otherwise: ``mixed_sample_outer_exponential``'s Gumbel-max race on the
      row's own noise row, by the same device function with the same block
      and vector sizes, so the pick is that op's pick on the same noise.

    A verification row takes the temperature of the request that owns it,
    ``torch.searchsorted(cu_draft, row, right=True)``. The noise comes from
    the caller, so the op touches no generator and a caller keeps its RNG
    stream: draw ``expo_t`` and ``expo_b`` in the order the unfused path
    would.

    Equivalent unfused chain (per row set)::

        owner = searchsorted(cu_draft, arange(n_target), right=True)
        mixed_sample_outer_exponential(o, input[target_idx], expo_t, temps[owner])
        out_t = where(temps[owner] <= greedy_eps, topk_select(..., tie="low"), o)
        (bonus rows likewise with input[bonus_idx], expo_b and temps)

    Args:
        out_t: ``[n_target]`` int32, the verification rows' picks.
        out_b: ``[n_req]`` int32, the bonus picks.
        input: ``[rows, V]`` fp32, fp16 or bf16 logits of the step.
        expo_t: ``[n_target, V]`` fp32 Exp(1) noise of the verification rows.
            A broadcast row (stride 0 in dim 0) is accepted.
        expo_b: ``[n_req, V]`` fp32 Exp(1) noise of the bonus rows.
        temps: ``[n_req]`` fp32 per-request temperatures.
        target_idx: ``[n_target]`` rows of ``input`` that are verification rows.
        bonus_idx: ``[n_req]`` rows of ``input`` that are bonus rows.
        cu_draft: ``[n_req]`` inclusive prefix sums of the draft counts.
        eps: keyword-only; added to the noise before the division.
        greedy_eps: keyword-only; rows at or below this temperature are
            greedy; 0.0 gives the ``temperature == 0`` rule of
            ``mixed_sample_outer_exponential``.

    Raises:
        ValueError: ``temps`` or ``cu_draft`` is not ``[n_req]`` -- a 0-dim
            temperature, or verification rows with no requests to own them.

    The contract, i.e. what a provider must keep and what a change to this op
    must treat as breaking (a new ``version=``):

    - Greedy rows: ``temperature <= greedy_eps`` (inclusive) picks the
      column of the largest score under the integer-ordered key: -0.0 and
      0.0 are one score, NaN (either sign) is above +inf, ties go to the
      smallest column (``tie="low"``; an all -inf row picks 0). Exact.
    - Sampled rows: the pick is ``argmax_j p_j / (expo_j + eps)`` with
      ``p = softmax(x / max(T, 1e-5))`` in fp32 -- the logits are scaled by
      ``1 / T`` first and the row max is subtracted after (the
      ``mixed_sample_outer_exponential`` order; ``fused_mixed_sample``
      subtracts first). ``T`` in ``(greedy_eps, 1e-5]`` samples at
      ``T = 1e-5``, on the GPU and off it. Ties of the race go to the
      smallest column. Changing the clamp, the eps placement or the scaling
      order is breaking; the rounding of the native kernel is not part of the
      contract (the torch definition divides by ``T`` where the kernel
      multiplies by ``1 / T``), so a provider matches the law, not the bits.
    - A verification row's temperature is ``temps[owner]``,
      ``owner = searchsorted(cu_draft, row, right=True)`` clamped to the last
      request; ``expo_t[i]`` / ``expo_b[j]`` is the noise of ``out_t[i]`` /
      ``out_b[j]``. The op draws no noise.

    One launch; a step with few rows (at most half the CU count) spreads each
    row over 16 workgroups and adds a finishing launch, with the same picks
    bit for bit -- the split re-runs the one-workgroup race thread for thread
    -- so a pick does not depend on how many rows the step has.

    The kernel runs on GPU tensors with the last dim of ``input`` and the
    noise contiguous; index lists, prefix sums, temperatures and outputs are
    brought to int64 / int32 / fp32 contiguous form first when they are not.
    Off the GPU, with logits of another dtype, or with a vocabulary past
    int32, the picks are computed in torch (``_spec_verify_pick_ref``) -- same
    rule, not bitwise the kernel.
    """
    n_t, n_req = target_idx.numel(), bonus_idx.numel()
    if n_t + n_req == 0:
        return
    # Host metadata only, so graph-safe. Without these a 0-dim or empty
    # ``temps`` fails as an IndexError deep in the torch path (``t[owner]``)
    # and as a kernel-side check on the GPU.
    if (
        n_req == 0
        or temps.dim() != 1
        or temps.numel() != n_req
        or cu_draft.numel() != n_req
    ):
        raise ValueError(
            "spec_verify_pick: verification rows need their requests, and temps "
            "and cu_draft must be [n_req] "
            f"(n_req = bonus_idx.numel() = {n_req}, n_target = {n_t}); got "
            f"temps {tuple(temps.shape)}, cu_draft {tuple(cu_draft.shape)}"
        )
    if (
        not input.is_cuda
        or input.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or input.size(1) >= 2**31
    ):
        device = input.device
        cu = cu_draft.to(device=device, dtype=torch.int64)
        owner = torch.searchsorted(
            cu, torch.arange(n_t, device=device), right=True
        ).clamp_max_(n_req - 1)
        t = temps.to(device=device, dtype=torch.float32)
        rows_t, rows_b = target_idx.to(device), bonus_idx.to(device)
        if n_t:
            out_t.copy_(
                _spec_verify_pick_ref(input, expo_t, t[owner], rows_t, eps, greedy_eps)
            )
        if n_req:
            out_b.copy_(
                _spec_verify_pick_ref(input, expo_b, t, rows_b, eps, greedy_eps)
            )
        return

    # The kernel's layout: noise and logits contiguous along V, index lists
    # int64, prefix sums and outputs int32, all on the logits' device. Each
    # conversion is a no-op (no copy, no launch) on inputs already in that form.
    device = input.device

    def rowwise(x: Tensor, dtype: torch.dtype) -> Tensor:
        x = x.to(device=device, dtype=dtype)
        return x if x.stride(-1) == 1 else x.contiguous()

    def dense(x: Tensor, dtype: torch.dtype) -> Tensor:
        return x.to(device=device, dtype=dtype).contiguous()

    def writable(x: Tensor) -> Tensor:
        if x.dtype == torch.int32 and x.device == device and x.is_contiguous():
            return x
        return torch.empty(x.shape, dtype=torch.int32, device=device)

    kt, kb = writable(out_t), writable(out_b)
    split_rows = _spec_verify_pick_split_rows(n_t + n_req, device)
    workspace = torch.empty(
        (n_t + n_req) * _SPEC_VERIFY_PICK_WS_PER_ROW if split_rows else 0,
        dtype=torch.int32,
        device=device,
    )
    _spec_verify_pick_hip(
        kt,
        kb,
        rowwise(input, input.dtype),
        rowwise(expo_t, torch.float32),
        rowwise(expo_b, torch.float32),
        dense(temps, torch.float32),
        dense(target_idx, torch.int64),
        dense(bonus_idx, torch.int64),
        dense(cu_draft, torch.int32),
        workspace,
        split_rows,
        eps,
        greedy_eps,
    )
    if kt is not out_t:
        out_t.copy_(kt)
    if kb is not out_b:
        out_b.copy_(kb)
