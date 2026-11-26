# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf of the FlyDSL causal conv1d split-qkv backward.

``causal_conv1d_split_qkv_bwd_flydsl_fn`` is checked against the fp32 torch
autograd reference of the Triton test, on the same cases, and timed against the
Triton kernel on the Kimi-K3 KDA short-conv shapes (q, k, v depthwise
``[12288, 1, 4]`` each; TP8 has 1536 channels per tensor).

Usage:
    python op_tests/test_flydsl_causal_conv1d_bwd.py
    pytest -sv op_tests/test_flydsl_causal_conv1d_bwd.py
"""

import argparse
import itertools
import types

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.causal_conv1d_bwd_flydsl import (
    causal_conv1d_split_qkv_bwd_flydsl_fn,
)
from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_bwd_triton_fn
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import dtypes
from op_tests.triton_tests.gated_delta_net.test_causal_conv1d_bwd import (
    KIMI_K3_DIM,
    KIMI_K3_DIM_TP8,
    PAD_SLOT_ID,
    _assert_close,
    _make_inputs,
    _reference,
)

torch.set_default_device("cuda")

# The kernel is validated on gfx950 only.
SUPPORTED_GFX = ["gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx_runtime() not in SUPPORTED_GFX,
    reason="FlyDSL causal conv1d backward is validated on gfx950 only",
)
fn = causal_conv1d_split_qkv_bwd_flydsl_fn


def _check(
    seqlens,
    has_init=None,
    width=4,
    x_dtype=torch.bfloat16,
    w_dtype=torch.float32,
    with_bias=False,
    activation="silu",
    channels_per_thread=None,
    **kwargs,
):
    inp = _make_inputs(
        seqlens,
        has_init,
        width=width,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        with_bias=with_bias,
        **kwargs,
    )
    dout = inp.pop("dout")
    x, weight, bias = inp["x"], inp["weight"], inp["bias"]
    dim, n_seqs = x.shape[0], len(seqlens)
    dx_r, dw_r, db_r, dinit_r = _reference(
        x,
        weight,
        bias,
        dout,
        inp["query_start_loc"],
        inp["initial_states"],
        inp["cache_indices"],
        inp["has_initial_state"],
        activation,
    )
    dx, dw, db, dinit = fn(
        **inp,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
        channels_per_thread=channels_per_thread,
    )
    assert dx.shape == x.shape and dx.dtype == x.dtype
    assert dw.shape == weight.shape and dw.dtype == weight.dtype
    tol = 2e-2 if x_dtype == torch.bfloat16 else 1e-4
    _assert_close(dx, dx_r, tol, "dx")
    _assert_close(dw, dw_r, 1e-2 if w_dtype == torch.bfloat16 else 1e-4, "dweight")
    if with_bias:
        assert db.shape == bias.shape and db.dtype == bias.dtype
        _assert_close(db, db_r, 1e-2 if w_dtype == torch.bfloat16 else 1e-4, "dbias")
    else:
        assert db is None
    if has_init is not None:
        assert dinit.shape == (n_seqs, dim, width - 1)
        _assert_close(dinit, dinit_r, tol, "dinitial_states")
    else:
        assert dinit is None


@pytest.mark.parametrize(
    "seqlens",
    [
        [7680],
        [5010, 2680],
        [1112] * 7,
        [1, 2, 3, 4, 5, 7, 8, 9],
        [63, 64, 65, 127, 128, 129, 255, 257],
        [10, 0, 5, 0, 40],  # zero-length sequences
    ],
)
@pytest.mark.parametrize("row_extra", [1680, 0])
def test_kimi_k3_tp8_shapes(seqlens, row_extra):
    _check(seqlens, row_extra=row_extra)


@pytest.mark.parametrize("seqlens", [[4096], [1024, 3, 1000]])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
def test_kimi_k3_full_dim(seqlens, w_dtype):
    # q, k and v with 12288 channels each (TP1).
    _check(
        seqlens,
        k_dim=KIMI_K3_DIM,
        v_dim=KIMI_K3_DIM,
        row_extra=0,
        w_dtype=w_dtype,
        with_bias=True,
    )


@pytest.mark.parametrize("sd_layout", [False, True])
@pytest.mark.parametrize("state_cols", [3, 7])
def test_initial_states(sd_layout, state_cols):
    _check(
        [1000, 1, 2, 3, 517],
        has_init=[True, True, False, True, True],
        sd_layout=sd_layout,
        state_cols=state_cols,
    )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_width_activation_bias(width, activation, with_bias):
    _check(
        [257, 3, 64, 1],
        has_init=[True, False, True, True],
        width=width,
        activation=activation,
        with_bias=with_bias,
    )


def test_pad_slots():
    _check([40, 2, 70, 1, 33], has_init=[True] * 5, pad_seqs=(0, 3))


def test_pad_slots_nan():
    # NaN in the x and dout rows of padded sequences must not reach any gradient.
    _check(
        [40, 2, 70, 1, 33],
        has_init=[True] * 5,
        pad_seqs=(0, 3),
        nan_pad=True,
        with_bias=True,
    )


def test_packed_dout_and_tail_tokens():
    # dq, dk, dv as column views of one [T, dim] buffer; tokens after the last sequence get zero.
    _check([300, 2, 129], packed_dout=True, tail=17)


def test_channel_major_strides():
    _check([300, 2, 129], has_init=[True, False, True], channel_major=True)


def test_large_channel_stride():
    # Token-contiguous x inside a [36864, 65536] buffer: channel offsets from 32768
    # on pass 2**31 elements, so they need int64 arithmetic.
    k_dim = v_dim = KIMI_K3_DIM
    dim, row, T = 3 * k_dim, 65536, 256
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("needs 8 GiB of free device memory")
    torch.manual_seed(0)
    buf = torch.empty((dim, row), dtype=torch.bfloat16, device="cuda")
    x = buf[:, :T]
    x.normal_()
    weight = torch.randn((dim, 4), dtype=torch.float32, device="cuda") * 0.5
    dout = torch.randn((T, dim), dtype=torch.bfloat16, device="cuda")
    qsl = torch.tensor([0, 100, T], dtype=torch.int32, device="cuda")
    slots = torch.arange(2, dtype=torch.int32, device="cuda")
    dx_r, dw_r, _, _ = _reference(x, weight, None, dout, qsl, None, slots, None, "silu")
    dx, dw, _, _ = fn(
        dout[:, :k_dim].contiguous(),
        dout[:, k_dim : 2 * k_dim].contiguous(),
        dout[:, 2 * k_dim :].contiguous(),
        x,
        weight,
        None,
        qsl,
        k_dim,
        v_dim,
    )
    _assert_close(dx, dx_r, 2e-2, "dx")
    _assert_close(dw, dw_r, 1e-4, "dweight")


@pytest.mark.parametrize("k_dim,v_dim", [(256, 512), (100, 260), (101, 261)])
def test_unaligned_split(k_dim, v_dim):
    # Channel blocks that straddle the q/k/v boundaries; odd splits take one channel per thread.
    _check([300, 2, 1, 129], k_dim=k_dim, v_dim=v_dim, row_extra=64, with_bias=True)


@pytest.mark.parametrize("channels_per_thread", [1, 2])
def test_channels_per_thread(channels_per_thread):
    _check(
        [513, 64, 1],
        has_init=[True, False, True],
        with_bias=True,
        channels_per_thread=channels_per_thread,
    )


def test_fp32():
    _check(
        [513, 7],
        has_init=[True, False],
        x_dtype=torch.float32,
        w_dtype=torch.float32,
        with_bias=True,
    )


def test_deterministic():
    torch.manual_seed(0)
    k_dim = v_dim = KIMI_K3_DIM_TP8
    dim, T = 3 * k_dim, 4096
    x = torch.randn((T, dim), dtype=torch.bfloat16, device="cuda").t()
    weight = torch.randn((dim, 4), dtype=torch.float32, device="cuda")
    dout = torch.randn((T, dim), dtype=torch.bfloat16, device="cuda")
    qsl = torch.tensor([0, 1000, T], dtype=torch.int32, device="cuda")
    args = (dout[:, :k_dim], dout[:, k_dim : 2 * k_dim], dout[:, 2 * k_dim :], x)
    first = fn(*args, weight, None, qsl, k_dim, v_dim)
    second = fn(*args, weight, None, qsl, k_dim, v_dim)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_nums_dict_schedule():
    # The second call reuses the schedule the first one stores, with no host sync.
    # Tokens after the last sequence must still get zero dx, whatever memory dx reuses.
    tail = 17
    inp = _make_inputs([700, 33, 1], has_init=[True, False, True], tail=tail)
    inp.pop("dout")
    ref = fn(**inp)
    metadata = types.SimpleNamespace(nums_dict={})
    fn(**inp, metadata=metadata)
    assert len(metadata.nums_dict) == 1
    poison = torch.full_like(ref[0].t(), float("nan"))
    del poison
    got = fn(**inp, metadata=metadata)
    assert not got[0][:, -tail:].any()
    for g, r in zip(got, ref):
        assert (g is None and r is None) or torch.equal(g, r)


def test_cuda_graph_replay():
    # A prefilled schedule makes the call capturable; replays track new inputs.
    inp = _make_inputs([2048, 1000, 3], has_init=[True, False, True], tail=5)
    inp.pop("dout")
    metadata = types.SimpleNamespace(nums_dict={})
    eager = fn(**inp, metadata=metadata)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(**inp, metadata=metadata)
    inp["dq"].mul_(-0.5)
    graph.replay()
    ref = fn(**inp)
    for g, r in zip(out, ref):
        assert (g is None and r is None) or torch.equal(g, r)
    assert not torch.equal(out[0], eager[0])


@benchmark()
def test_causal_conv1d_bwd(batch, seqlen, k_dim, row_stride, dtype):
    """Equal-length batch of ``batch`` x ``seqlen`` tokens, q/k/v of ``k_dim`` channels each."""
    seqlens = [seqlen] * batch
    dim = 3 * k_dim
    inp = _make_inputs(
        seqlens,
        k_dim=k_dim,
        v_dim=k_dim,
        row_extra=max(row_stride, dim) - dim,
        x_dtype=dtype,
        w_dtype=torch.float32,
    )
    dout = inp.pop("dout")
    dx_r, dw_r, _, _ = _reference(
        inp["x"],
        inp["weight"],
        None,
        dout,
        inp["query_start_loc"],
        None,
        inp["cache_indices"],
        None,
        "silu",
    )
    args = {k: inp[k] for k in ("dq", "dk", "dv", "x", "weight", "query_start_loc")}
    candidates = {
        "flydsl": causal_conv1d_split_qkv_bwd_flydsl_fn,
        "triton": causal_conv1d_split_qkv_bwd_triton_fn,
    }
    tokens = batch * seqlen
    # Per element of x: pre-activation (width MACs), SiLU gradient, dx (width MACs), dweight (width MACs).
    flops = tokens * dim * (6 * 4 + 8)
    # x and the output gradient read once, dx written once.
    nbytes = 3 * tokens * dim * dtype.itemsize

    ret = {"gfx": get_gfx_runtime()}
    for name, impl in candidates.items():
        metadata = types.SimpleNamespace(nums_dict={})
        (dx, dw, _, _), us = run_perftest(
            impl,
            **args,
            bias=None,
            k_dim=k_dim,
            v_dim=k_dim,
            metadata=metadata,
        )
        err = checkAllclose(
            dx_r, dx.to(dtypes.fp32), rtol=2e-2, atol=2e-2, msg=f"{name}: dx "
        )
        err_w = checkAllclose(
            dw_r, dw.to(dtypes.fp32), rtol=1e-3, atol=1e-3, msg=f"{name}: dweight "
        )
        assert err <= 1e-4 and err_w == 0, f"{name}: dx {err:.2e}, dweight {err_w:.2e}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# Driven by main(); pytest collects the fixed cases above.
test_causal_conv1d_bwd.__test__ = False


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL causal conv1d backward unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL causal conv1d backward correctness + performance sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="x and gradient dtype. Example: -d bf16",
    )
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[1], help="Sequences."
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (8192, KIMI_K3_DIM_TP8, 6288),
            (32768, KIMI_K3_DIM_TP8, 4608),
            (4096, KIMI_K3_DIM, 36864),
            (8192, KIMI_K3_DIM, 36864),
        ],
        metavar="SEQLEN,KDIM,ROW_STRIDE",
        help="Tokens per sequence, channels per q/k/v tensor, x row stride.\n"
        "Default: Kimi-K3 TP8 (1536) and TP1 (12288).",
    )
    args = parser.parse_args()

    rows = []
    for dtype, batch, shape in itertools.product(args.dtype, args.batch, args.mnk):
        if len(shape) != 3:
            parser.error("-s/--mnk entries must be seqlen,kdim,row_stride")
        seqlen, k_dim, row_stride = shape
        rows.append(test_causal_conv1d_bwd(batch, seqlen, k_dim, row_stride, dtype))
    aiter.logger.info(
        "FlyDSL causal conv1d backward summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
