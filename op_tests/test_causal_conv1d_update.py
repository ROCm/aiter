# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for the AIter causal-conv1d update op."""

import argparse
import itertools
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

PAD_SLOT_ID = -1
SUPPORTED_GFX = ("gfx942", "gfx950")

_MAX_PERF_ROTATIONS = 32
_PERF_ROTATION_BUDGET = 256 * 1024 * 1024


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    pad_slot_id=-1,
):
    """Convenience wrapper around the preallocated-output AIter interface."""
    out = torch.zeros_like(x)
    _run_aiter(
        x,
        conv_state,
        weight,
        bias,
        out,
        activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    return out


def _run_aiter(
    x,
    conv_state,
    weight,
    bias,
    out,
    activation,
    cache_seqlens,
    conv_state_indices,
    pad_slot_id,
):
    bias_arg = (
        torch.empty(0, dtype=x.dtype, device=x.device)
        if bias is None
        else bias.to(dtype=x.dtype)
    )
    cache_arg = (
        torch.empty(0, dtype=torch.int32, device=x.device)
        if cache_seqlens is None
        else cache_seqlens
    )
    indices_arg = (
        torch.empty(0, dtype=torch.int32, device=x.device)
        if conv_state_indices is None
        else conv_state_indices
    )
    aiter.causal_conv1d_update(
        x,
        conv_state,
        weight.to(dtype=x.dtype),
        bias_arg,
        out,
        activation in ("silu", "swish"),
        cache_arg,
        indices_arg,
        pad_slot_id,
    )
    return out


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """Torch reference for direct state or circular-cache updates."""
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_update_with_indices_ref(
    x, conv_state, weight, bias=None, activation=None, conv_state_indices=None
):
    """Torch reference for model-style indexed linear state updates."""
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert weight.shape == (dim, width)
    assert state_len == width - 1

    if conv_state_indices is None:
        assert conv_state.shape == (batch, dim, state_len)
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        assert conv_state_indices.shape == (batch,)
        x_new_list = []
        for i in range(batch):
            slot = int(conv_state_indices[i].item())
            state_slot = conv_state[slot : slot + 1].clone()
            x_new_i = torch.cat(
                [state_slot.to(weight.dtype), x[i : i + 1].to(weight.dtype)], dim=-1
            )
            conv_state[slot].copy_(
                torch.cat([state_slot[:, :, 1:], x[i : i + 1]], dim=-1)[
                    :, :, -state_len:
                ].squeeze(0)
            )
            x_new_list.append(x_new_i)
        x_new = torch.cat(x_new_list, dim=0)

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _tolerances(dtype):
    if dtype == dtypes.fp32:
        return 3e-4, 1e-3
    if dtype == dtypes.bf16:
        return 1e-2, 5e-2
    return 3e-3, 5e-3


def _perf_rotation_count(state, out):
    bytes_per_call = max(1, state.nbytes + out.nbytes)
    return max(
        1,
        min(_MAX_PERF_ROTATIONS, _PERF_ROTATION_BUDGET // bytes_per_call),
    )


def _work(batch, dim, width, seqlen, dtype, has_bias, silu_activation):
    output_elements = batch * dim * seqlen
    flops_per_output = 2 * width + int(has_bias) + 4 * int(silu_activation)
    flops = output_elements * flops_per_output
    # Logical traffic: input/output, weights/bias, and the read/write state window.
    elements = (
        2 * output_elements
        + dim * width
        + dim * int(has_bias)
        + 2 * batch * dim * (width - 1)
    )
    nbytes = elements * torch.empty((), dtype=dtype).element_size()
    return flops, nbytes


def _measure_candidates(
    candidates,
    initial_state,
    output_template,
    reference_state,
    reference_out,
    flops,
    nbytes,
    rtol,
    atol,
    label,
    valid_output_rows=None,
):
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        # State and output are explicit timing arguments, allowing run_perftest
        # to rotate independent copies without timing a reset or clone.
        perf_state = initial_state.clone()
        perf_out = torch.zeros_like(output_template)
        _, us = run_perftest(
            candidate,
            perf_state,
            perf_out,
            num_rotate_args=_perf_rotation_count(perf_state, perf_out),
        )

        # Correctness is independent of state repeatedly mutated during timing.
        candidate_state = initial_state.clone()
        candidate_out = torch.zeros_like(output_template)
        candidate(candidate_state, candidate_out)
        checked_reference_out = (
            reference_out
            if valid_output_rows is None
            else reference_out[:valid_output_rows]
        )
        checked_candidate_out = (
            candidate_out
            if valid_output_rows is None
            else candidate_out[:valid_output_rows]
        )
        out_err = checkAllclose(
            checked_reference_out.to(dtypes.fp32),
            checked_candidate_out.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"{name}: {label} output ",
        )
        state_err = checkAllclose(
            reference_state.to(dtypes.fp32),
            candidate_state.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: {label} in-place state ",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(out_err, state_err)
    return ret


@benchmark()
def test_causal_conv1d_update_with_indices_decode(
    batch,
    dim,
    width,
    seqlen,
    has_bias=False,
    silu_activation=True,
    dtype=dtypes.bf16,
    num_cache_lines=128,
    with_padding=False,
    padding=5,
):
    """Indexed linear-shift mode used by continuous-batching decode."""
    seed_everything(0)
    padded_batch = batch + (padding if with_padding else 0)
    num_slots = max(num_cache_lines, batch)
    x = torch.randn(padded_batch, dim, seqlen, device="cuda", dtype=dtype)
    # Match the model: physical [slots, state, dim], viewed as [slots, dim, state].
    initial_state = (
        torch.randn(num_slots, width - 1, dim, device="cuda", dtype=dtype)
        .contiguous()
        .transpose(1, 2)
    )
    weight = torch.randn(dim, width, device="cuda", dtype=dtype)
    bias = torch.randn(dim, device="cuda", dtype=dtype) if has_bias else None
    activation = "silu" if silu_activation else None
    valid_indices = torch.randperm(num_slots, device="cuda", dtype=torch.int32)[:batch]
    indices = (
        torch.cat(
            [
                valid_indices,
                torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"),
            ]
        )
        if with_padding
        else valid_indices
    )
    pad_slot_id = PAD_SLOT_ID if with_padding else -1

    reference_state = initial_state.clone()
    reference_out = torch.zeros_like(x)
    reference_out[:batch] = causal_conv1d_update_with_indices_ref(
        x[:batch],
        reference_state,
        weight,
        bias,
        activation,
        valid_indices,
    )

    def run_aiter(candidate_state, candidate_out):
        return _run_aiter(
            x,
            candidate_state,
            weight,
            bias,
            candidate_out,
            activation,
            None,
            indices,
            pad_slot_id,
        )

    candidates = {"aiter": run_aiter}
    flops, nbytes = _work(batch, dim, width, seqlen, dtype, has_bias, silu_activation)
    rtol, atol = _tolerances(dtype)
    return _measure_candidates(
        candidates,
        initial_state,
        x,
        reference_state,
        reference_out,
        flops,
        nbytes,
        rtol,
        atol,
        "indices decode",
        batch,
    )


@benchmark()
def test_causal_conv1d_update_cache_seqlens_decode(
    batch,
    dim,
    width,
    seqlen,
    has_bias=False,
    silu_activation=True,
    dtype=dtypes.bf16,
    has_cache_seqlens=False,
):
    """Direct-state mode, optionally using circular cache positions."""
    seed_everything(0)
    state_len = width - 1
    x = torch.randn(batch, dim, seqlen, device="cuda", dtype=dtype)
    initial_state = (
        torch.randn(batch, state_len, dim, device="cuda", dtype=dtype)
        .contiguous()
        .transpose(1, 2)
    )
    weight = torch.randn(dim, width, device="cuda", dtype=dtype)
    bias = torch.randn(dim, device="cuda", dtype=dtype) if has_bias else None
    activation = "silu" if silu_activation else None
    cache_seqlens = (
        torch.randint(0, 1024, (batch,), dtype=torch.int32, device="cuda")
        if has_cache_seqlens
        else None
    )

    reference_state = initial_state.clone()
    reference_out = causal_conv1d_update_ref(
        x,
        reference_state,
        weight,
        bias,
        activation,
        cache_seqlens,
    )

    def run_aiter(candidate_state, candidate_out):
        return _run_aiter(
            x,
            candidate_state,
            weight,
            bias,
            candidate_out,
            activation,
            cache_seqlens,
            None,
            -1,
        )

    candidates = {"aiter": run_aiter}
    flops, nbytes = _work(batch, dim, width, seqlen, dtype, has_bias, silu_activation)
    rtol, atol = _tolerances(dtype)
    return _measure_candidates(
        candidates,
        initial_state,
        x,
        reference_state,
        reference_out,
        flops,
        nbytes,
        rtol,
        atol,
        "cache-seqlens decode",
    )


@benchmark()
def test_causal_conv1d_update_with_batch_gather_decode(
    batch_size,
    dim,
    width,
    seqlen,
    has_bias=False,
    silu_activation=True,
    dtype=dtypes.bf16,
    has_cache_seqlens=False,
    with_padding=False,
    padding=5,
    total_entries_scale=10,
):
    """Gathered state mode with optional circular positions and padding."""
    seed_everything(0)
    pad_count = padding if with_padding else 0
    padded_batch = batch_size + pad_count
    total_entries = total_entries_scale * batch_size
    state_len = width - 1
    x = torch.randn(padded_batch, dim, seqlen, device="cuda", dtype=dtype)
    indices = torch.randperm(total_entries, device="cuda", dtype=torch.int32)[
        :batch_size
    ]
    padded_indices = (
        torch.cat(
            [
                indices,
                torch.full((pad_count,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"),
            ]
        )
        if with_padding
        else indices
    )
    initial_state = (
        torch.randn(total_entries, state_len, dim, device="cuda", dtype=dtype)
        .contiguous()
        .transpose(1, 2)
    )
    weight = torch.randn(dim, width, device="cuda", dtype=dtype)
    bias = torch.randn(dim, device="cuda", dtype=dtype) if has_bias else None
    activation = "silu" if silu_activation else None
    real_cache_seqlens = (
        torch.randint(0, 1024, (batch_size,), dtype=torch.int32, device="cuda")
        if has_cache_seqlens
        else None
    )
    cache_seqlens = (
        torch.cat(
            [
                real_cache_seqlens,
                torch.zeros(pad_count, dtype=torch.int32, device="cuda"),
            ]
        )
        if has_cache_seqlens and with_padding
        else real_cache_seqlens
    )
    pad_slot_id = PAD_SLOT_ID if with_padding else -1

    reference_state = initial_state.clone()
    gathered_reference_state = reference_state[indices].clone()
    gathered_reference_out = causal_conv1d_update_ref(
        x[:batch_size],
        gathered_reference_state,
        weight,
        bias,
        activation,
        real_cache_seqlens,
    )
    reference_state[indices] = gathered_reference_state
    reference_out = torch.zeros_like(x)
    reference_out[:batch_size] = gathered_reference_out

    def run_aiter(candidate_state, candidate_out):
        return _run_aiter(
            x,
            candidate_state,
            weight,
            bias,
            candidate_out,
            activation,
            cache_seqlens,
            padded_indices,
            pad_slot_id,
        )

    candidates = {"aiter": run_aiter}
    flops, nbytes = _work(
        batch_size, dim, width, seqlen, dtype, has_bias, silu_activation
    )
    rtol, atol = _tolerances(dtype)
    return _measure_candidates(
        candidates,
        initial_state,
        x,
        reference_state,
        reference_out,
        flops,
        nbytes,
        rtol,
        atol,
        "batch-gather decode",
        batch_size,
    )


def _str2bool(value):
    value = value.lower()
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def _flatten_dtypes(values):
    return list(
        itertools.chain.from_iterable(
            value if isinstance(value, tuple) else (value,) for value in values
        )
    )


def _summarize(name, rows):
    frame = pd.DataFrame(rows)
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, frame.to_markdown(index=False)
    )


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("causal_conv1d_update is unsupported on %s; skipping", gfx)
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark causal_conv1d_update",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.bf16, dtypes.fp16, dtypes.fp32],
        default=[dtypes.bf16],
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 64, 256])
    parser.add_argument("--dim", type=int, nargs="*", default=[2048, 4096])
    parser.add_argument("-w", "--width", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("-s", "--seqlen", type=int, nargs="*", default=[1])
    parser.add_argument("--has-bias", type=_str2bool, nargs="*", default=[True, False])
    parser.add_argument(
        "--silu-activation", type=_str2bool, nargs="*", default=[True, False]
    )
    parser.add_argument("-n", "--num-cache-lines", type=int, nargs="*", default=[512])
    parser.add_argument(
        "--with-padding", type=_str2bool, nargs="*", default=[True, False]
    )
    parser.add_argument("--padding", type=int, nargs="*", default=[5])
    parser.add_argument(
        "--has-cache-seqlens", type=_str2bool, nargs="*", default=[True, False]
    )
    parser.add_argument(
        "--mode",
        nargs="*",
        choices=["indices", "cache_seqlens", "batch_gather"],
        default=["indices", "cache_seqlens", "batch_gather"],
    )
    parser.add_argument("--total-entries-scale", type=int, nargs="*", default=[10])
    args = parser.parse_args()
    dtype_values = _flatten_dtypes(args.dtype)

    if "indices" in args.mode:
        rows = [
            test_causal_conv1d_update_with_indices_decode(*case)
            for case in itertools.product(
                args.batch,
                args.dim,
                args.width,
                args.seqlen,
                args.has_bias,
                args.silu_activation,
                dtype_values,
                args.num_cache_lines,
                args.with_padding,
                args.padding,
            )
        ]
        _summarize("causal_conv1d_update indices", rows)

    if "cache_seqlens" in args.mode:
        rows = [
            test_causal_conv1d_update_cache_seqlens_decode(*case)
            for case in itertools.product(
                args.batch,
                args.dim,
                args.width,
                args.seqlen,
                args.has_bias,
                args.silu_activation,
                dtype_values,
                args.has_cache_seqlens,
            )
        ]
        _summarize("causal_conv1d_update cache_seqlens", rows)

    if "batch_gather" in args.mode:
        rows = [
            test_causal_conv1d_update_with_batch_gather_decode(*case)
            for case in itertools.product(
                args.batch,
                args.dim,
                args.width,
                args.seqlen,
                args.has_bias,
                args.silu_activation,
                dtype_values,
                args.has_cache_seqlens,
                args.with_padding,
                args.padding,
                args.total_entries_scale,
            )
        ]
        _summarize("causal_conv1d_update batch_gather", rows)


if __name__ == "__main__":
    main()
