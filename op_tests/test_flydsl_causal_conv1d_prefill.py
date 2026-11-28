# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf for the FlyDSL ``causal_conv1d_fn``-contract prefill.

``causal_conv1d_prefill_flydsl_fn`` returns one output tensor and updates
``conv_states`` in place. Every case is compared against an fp32 torch
reference built on ``F.conv1d``.

Usage:
    python op_tests/test_flydsl_causal_conv1d_prefill.py
    pytest -sv op_tests/test_flydsl_causal_conv1d_prefill.py
"""

import argparse
import contextlib
import itertools
import os

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.causal_conv1d_flydsl import causal_conv1d_prefill_flydsl_fn
from aiter.ops.flydsl.kernels import causal_conv1d_prefill_vec
from aiter.ops.flydsl.kernels.causal_conv1d_prefill import (
    create_causal_conv1d_prefill_kernel,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# The kernel is tuned and validated on gfx950 only.
SUPPORTED_GFX = ["gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="FlyDSL causal conv1d prefill is validated on gfx950 only",
)
PAD_SLOT_ID = -1
_TOL_ERR_RATIO = 1e-3
_MAX_PERF_ROTATIONS = 32
_PERF_ROTATION_BUDGET = 256 * 1024 * 1024


def make_inputs(
    seq_lens,
    dim,
    width,
    dtype,
    layout,
    with_bias=False,
    with_initial_state=True,
    seed=0,
):
    """Inputs as SGLang's GDN prefill passes them to ``causal_conv1d_fn``.

    ``layout`` is how x is stored: "token_major" is ``mixed_qkv.transpose(0, 1)``
    of the contiguous ``[tokens, dim]`` projection, which is what the GDN layer
    passes; "channel_major" is a contiguous ``[dim, tokens]`` tensor.
    ``conv_states`` is a contiguous ``[slots, dim, width - 1]`` pool, the layout
    of SGLang's MambaPool conv cache.
    """
    torch.manual_seed(seed)
    total = sum(seq_lens)
    batch = len(seq_lens)
    if layout == "token_major":
        x = torch.randn(total, dim, dtype=dtype).t()
    else:
        x = torch.randn(dim, total, dtype=dtype)
    offsets = [0]
    for n in seq_lens:
        offsets.append(offsets[-1] + n)
    return {
        "x": x,
        "weight": torch.randn(dim, width, dtype=dtype) * 0.5,
        "bias": torch.randn(dim, dtype=dtype) if with_bias else None,
        # One spare slot that no sequence maps to.
        "conv_states": torch.randn(batch + 1, dim, width - 1, dtype=dtype),
        "query_start_loc": torch.tensor(offsets, dtype=torch.int32),
        "seq_lens_cpu": list(seq_lens),
        "cache_indices": torch.arange(batch, 0, -1, dtype=torch.int32),
        "has_initial_state": torch.full((batch,), with_initial_state, dtype=torch.bool),
    }


def run_torch(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    seq_lens_cpu,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
):
    """fp32 reference. Returns the output and updates ``conv_states`` in place.

    Padded sequences produce zeros here; the kernel leaves them unspecified,
    so callers compare live tokens only.
    """
    dim, width = weight.shape
    out = torch.zeros(x.shape, dtype=torch.float32)
    start = 0
    for seq, length in enumerate(seq_lens_cpu):
        slot = seq if cache_indices is None else int(cache_indices[seq])
        if length == 0 or slot == pad_slot_id:
            start += length
            continue
        history = torch.zeros(dim, width - 1)
        if has_initial_state is not None and bool(has_initial_state[seq]):
            history = conv_states[slot, :, : width - 1].float()
        full = torch.cat([history, x[:, start : start + length].float()], dim=1)
        y = F.conv1d(
            full[None],
            weight.float()[:, None, :],
            bias.float() if bias is not None else None,
            groups=dim,
        )[0]
        out[:, start : start + length] = F.silu(y) if activation else y
        if conv_states is not None:
            conv_states[slot, :, : width - 1] = full[:, -(width - 1) :].to(
                conv_states.dtype
            )
        start += length
    return out


def live_token_mask(inputs, pad_slot_id=PAD_SLOT_ID):
    mask = torch.zeros(inputs["x"].shape[1], dtype=torch.bool)
    start = 0
    for seq, length in enumerate(inputs["seq_lens_cpu"]):
        if int(inputs["cache_indices"][seq]) != pad_slot_id:
            mask[start : start + length] = True
        start += length
    return mask


def check_against_reference(inputs, msg, activation="silu", **kwargs):
    """Run kernel and reference on separate state copies; return the error."""
    reference_state = inputs["conv_states"].clone()
    reference = run_torch(
        **{**inputs, "conv_states": reference_state}, activation=activation
    )
    state = inputs["conv_states"].clone()
    out = causal_conv1d_prefill_flydsl_fn(
        **{**inputs, "conv_states": state}, activation=activation, **kwargs
    )
    tol = 1e-2 if inputs["x"].dtype != torch.float32 else 1e-4
    return max(
        checkAllclose(
            reference,
            out.to(dtypes.fp32),
            rtol=tol,
            atol=tol,
            mask=live_token_mask(inputs)[None, :],
            msg=f"{msg} output ",
        ),
        checkAllclose(
            reference_state.to(dtypes.fp32),
            state.to(dtypes.fp32),
            rtol=tol,
            atol=tol,
            msg=f"{msg} conv_states ",
        ),
    )


@contextlib.contextmanager
def vec_conv(enabled):
    """AITER_CONV1D_PREFILL_VEC set to 1 (or unset) inside the block."""
    old = os.environ.pop("AITER_CONV1D_PREFILL_VEC", None)
    if enabled:
        os.environ["AITER_CONV1D_PREFILL_VEC"] = "1"
    try:
        yield
    finally:
        os.environ.pop("AITER_CONV1D_PREFILL_VEC", None)
        if old is not None:
            os.environ["AITER_CONV1D_PREFILL_VEC"] = old


@contextlib.contextmanager
def nan_outputs():
    """Outputs start NaN-filled, so a store the kernel skips shows."""
    empty_like = torch.empty_like

    def nan_empty_like(*args, **kwargs):
        t = empty_like(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    torch.empty_like = nan_empty_like
    try:
        yield
    finally:
        torch.empty_like = empty_like


# (sequence lengths, dim, width, dtype, token stride, has_initial_state, extras):
# Qwen3.8 TP1 conv views (dim 10240 inside the 16384-wide projection), varlen
# batches with an empty sequence and sequences shorter than the halo, a padded
# slot, bias, fp16 (exact SiLU), no SiLU, widths 2/3/5, int64 indices with
# integer history flags, and token rows not 16-byte aligned (base kernel).
VEC_CASES = [
    ([16384], 10240, 4, torch.bfloat16, 16384, [True], {}),
    ([6169], 10240, 4, torch.bfloat16, 16384, [False], {}),
    (
        [1, 2, 3, 0, 4, 70, 129, 64, 65, 1000],
        1024,
        4,
        torch.bfloat16,
        1024,
        [True, False, True, True, True, False, True, True, False, True],
        {},
    ),
    ([50, 300], 512, 4, torch.bfloat16, 512, [True, True], {"pad_first": True}),
    ([700, 33], 512, 4, torch.bfloat16, 512, [True, True], {"bias": True}),
    ([515, 2], 512, 4, torch.float16, 640, [True, True], {}),
    ([3, 200], 512, 2, torch.bfloat16, 512, [True, True], {}),
    ([1, 300], 512, 3, torch.bfloat16, 512, [True, True], {}),
    ([2, 77], 512, 5, torch.bfloat16, 512, [True, True], {}),
    ([257, 1], 512, 4, torch.bfloat16, 512, [True, True], {"activation": None}),
    ([2, 129], 512, 4, torch.bfloat16, 512, [True, False], {"int64": True}),
    ([5, 40], 512, 4, torch.bfloat16, 516, [True, True], {"fallback": True}),
]


def check_vec_matches_base():
    """AITER_CONV1D_PREFILL_VEC=1 gives the base kernel's outputs and states bit
    for bit, and the vectorized kernel ran on every case it supports."""
    vec_launches = causal_conv1d_prefill_vec.create_causal_conv1d_prefill_vec_kernel
    for lens, dim, width, dtype, stride, initial, extra in VEC_CASES:
        inputs = make_inputs(
            lens, dim, width, dtype, "token_major", extra.get("bias", False)
        )
        buf = torch.randn(sum(lens), stride, dtype=dtype)
        buf[:, :dim] = inputs["x"].t()
        inputs["x"] = buf[:, :dim].t()
        inputs["has_initial_state"] = torch.tensor(initial)
        if extra.get("pad_first"):
            inputs["cache_indices"][0] = PAD_SLOT_ID
        if extra.get("int64"):
            inputs["cache_indices"] = inputs["cache_indices"].long()
            inputs["has_initial_state"] = inputs["has_initial_state"].int()
        activation = extra.get("activation", "silu")
        results = []
        for enabled in (False, True):
            state = inputs["conv_states"].clone()
            calls = vec_launches.cache_info()
            with vec_conv(enabled), nan_outputs():
                out = causal_conv1d_prefill_flydsl_fn(
                    **{**inputs, "conv_states": state}, activation=activation
                )
            ran = sum(vec_launches.cache_info()[:2]) - sum(calls[:2])
            expected = int(enabled and not extra.get("fallback", False))
            assert ran == expected, f"{lens}: vectorized kernel ran {ran} times"
            results.append((out, state))
        (ref, ref_state), (out, state) = results
        live = live_token_mask(inputs)
        assert torch.equal(out[:, live], ref[:, live]), f"{lens}: output differs"
        assert torch.equal(state, ref_state), f"{lens}: conv_states differ"


def silu_mismatches(margin):
    """Inputs among all 2^32 fp32 bit patterns where the vectorized kernel's
    bf16(fast SiLU with its fallback) != bf16(exact SiLU)."""
    per, iters, block, vec = 1 << 28, 8, 256, causal_conv1d_prefill_vec.VEC
    nblocks = per // (iters * block * vec)
    counts = torch.zeros(nblocks * block, dtype=torch.int32)
    launch = causal_conv1d_prefill_vec.create_silu_check_kernel(margin, iters, block)
    mismatches = 0
    for c in range((1 << 32) // per):
        base = c * per - (1 << 32) if c * per >= 1 << 31 else c * per
        _run_compiled(launch, counts, base, nblocks, torch.cuda.current_stream())
        mismatches += int(counts.sum())
    return mismatches


def check_fast_silu_exhaustive():
    assert silu_mismatches(causal_conv1d_prefill_vec.FAST_TIE_MARGIN) == 0
    # Without the tie fallback the same count finds mismatches.
    assert silu_mismatches(-1) > 0


@benchmark()
def test_causal_conv1d_prefill(
    batch, seqlen, dim, width, dtype, layout, with_initial_state
):
    inputs = make_inputs(
        [seqlen] * batch, dim, width, dtype, layout, False, with_initial_state
    )
    candidates = {
        name: lambda state: causal_conv1d_prefill_flydsl_fn(
            **{**inputs, "conv_states": state}
        )
        for name in ("flydsl", "flydsl vec")
    }

    tokens = batch * seqlen
    flops = 2 * tokens * dim * width
    # x in, output out, weights, and the state window read and written.
    elements = 2 * tokens * dim + dim * width + 2 * batch * dim * (width - 1)
    nbytes = elements * inputs["x"].element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        with vec_conv(name == "flydsl vec"):
            err = check_against_reference(inputs, name)
            assert err <= _TOL_ERR_RATIO, f"{name}: mismatch ratio {err:.3e}"
            timing_state = inputs["conv_states"].clone()
            _, us = run_perftest(
                fn,
                timing_state,
                num_rotate_args=max(
                    1,
                    min(
                        _MAX_PERF_ROTATIONS,
                        _PERF_ROTATION_BUDGET // max(1, timing_state.nbytes),
                    ),
                ),
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# Driven by main(); pytest collects the fixed cases below.
test_causal_conv1d_prefill.__test__ = False


VARLEN_SHAPES = [
    [1],
    [2, 1, 3],  # every sequence shorter than width - 1 for width >= 4
    [5, 0, 37, 300],  # includes an empty sequence
    [1000, 63, 1000],
    [64, 128, 200, 5063],
]


@pytest.mark.parametrize("seq_lens", VARLEN_SHAPES)
@pytest.mark.parametrize("layout", ["token_major", "channel_major"])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_varlen_matches_reference(seq_lens, layout, with_initial_state):
    inputs = make_inputs(
        seq_lens, 1024, 4, torch.bfloat16, layout, True, with_initial_state
    )
    err = check_against_reference(inputs, f"{layout} {seq_lens}")
    assert err <= _TOL_ERR_RATIO


@pytest.mark.parametrize("width", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_widths_and_dtypes(width, dtype):
    inputs = make_inputs([3, 17, 260], 768, width, dtype, "token_major", True)
    err = check_against_reference(inputs, f"{width=} {dtype}")
    assert err <= _TOL_ERR_RATIO


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("channels_per_thread", [1, 2])
def test_activation_and_channel_mapping(activation, channels_per_thread):
    # 8192 tokens is where the default mapping switches to two channels.
    inputs = make_inputs([4096, 4000, 96], 512, 4, torch.bfloat16, "token_major")
    err = check_against_reference(
        inputs,
        f"{activation=} {channels_per_thread=}",
        activation=activation,
        channels_per_thread=channels_per_thread,
    )
    assert err <= _TOL_ERR_RATIO


def test_channel_contiguous_state_pool():
    """Same values, but a pool whose channel axis is the unit-stride one."""
    inputs = make_inputs([2, 9, 130], 512, 4, torch.bfloat16, "token_major")
    states = inputs["conv_states"]
    inputs["conv_states"] = states.transpose(1, 2).contiguous().transpose(1, 2)
    assert inputs["conv_states"].stride(1) == 1
    err = check_against_reference(inputs, "channel-contiguous state")
    assert err <= _TOL_ERR_RATIO


def test_padding_slot_leaves_state_untouched():
    inputs = make_inputs([5, 7, 9], 512, 4, torch.bfloat16, "token_major")
    inputs["cache_indices"] = torch.tensor([2, PAD_SLOT_ID, 0], dtype=torch.int32)
    state_before = inputs["conv_states"].clone()
    err = check_against_reference(inputs, "padding")
    assert err <= _TOL_ERR_RATIO
    # Slots 1 and 3 are mapped to no live sequence.
    state = inputs["conv_states"].clone()
    causal_conv1d_prefill_flydsl_fn(**{**inputs, "conv_states": state})
    untouched = torch.tensor([1, 3])
    assert torch.equal(state[untouched], state_before[untouched])


@pytest.mark.parametrize("layout", ["token_major", "channel_major"])
def test_new_token_count_reuses_kernel(layout):
    """x/out strides are launch arguments, so a new token count must neither
    build a new kernel nor index with a stale stride."""
    misses = None
    for total in (1000, 1001, 1537):
        inputs = make_inputs([total - 10, 10], 512, 4, torch.bfloat16, layout)
        err = check_against_reference(inputs, f"{layout} {total=}")
        assert err <= _TOL_ERR_RATIO
        info = create_causal_conv1d_prefill_kernel.cache_info()
        if misses is None:
            misses = info.misses
        assert info.misses == misses, f"{layout} {total=} built a new kernel"


def test_vec_matches_base():
    check_vec_matches_base()


def test_vec_fast_silu_exhaustive():
    check_fast_silu_exhaustive()


def test_rejects_unsupported_width():
    inputs = make_inputs([8], 64, 4, torch.bfloat16, "token_major")
    inputs["weight"] = torch.randn(64, 6, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="width 2..5"):
        causal_conv1d_prefill_flydsl_fn(**inputs)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL causal conv1d prefill unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL causal conv1d prefill correctness + performance sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="Input dtype. Example: -d bf16 fp16",
    )
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[1, 4, 16], help="Sequences."
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(256, 10240, 4), (1024, 10240, 4), (4096, 10240, 4)],
        metavar="SEQLEN,DIM,WIDTH",
        help="Tokens per sequence, conv channels, conv width. "
        "Default: Qwen3.8 at TP1 (dim 10240, width 4).",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        choices=["token_major", "channel_major"],
        default=["token_major"],
    )
    parser.add_argument(
        "--initial-state", type=int, nargs="*", choices=[0, 1], default=[1]
    )
    args = parser.parse_args()

    check_vec_matches_base()
    check_fast_silu_exhaustive()
    rows = []
    for dtype, batch, shape, layout, initial in itertools.product(
        args.dtype,
        args.batch,
        args.mnk,
        args.layout,
        args.initial_state,
    ):
        if len(shape) != 3:
            parser.error("-s/--mnk entries must be seqlen,dim,width")
        seqlen, dim, width = shape
        rows.append(
            test_causal_conv1d_prefill(
                batch, seqlen, dim, width, dtype, layout, bool(initial)
            )
        )
    aiter.logger.info(
        "FlyDSL causal conv1d prefill summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
