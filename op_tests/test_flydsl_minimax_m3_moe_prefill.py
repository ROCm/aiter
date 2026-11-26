# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FlyDSL W4A4 MXFP4 prefill MoE chain (gfx950).

`a4w4_prefill_moe` = sort -> gemm1 (gate/up, SwiGLU-OAI, MXFP4 requant) -> gemm2
(routing-weighted partials) -> bf16 reduce, at the TP4 shapes of MiniMax-M3
(hidden 6144, intermediate 768, 128 experts, topk 4).

Reference: a torch MoE on the dequantized values, nothing from the kernel under
test. Weights and activations are quantized by aiter's `per_1x32_f4_quant`
(RoundUp) and dequantized with `fp4_utils`; per expert the reference runs
x @ W13^T -> SwiGLU-OAI (alpha 1.702, limit 7) -> bf16 -> MXFP4 round trip (the
kernel requantizes the intermediate) -> @ W2^T, weighted by the routing weight.

Every case is checked element-wise against that reference (a fraction of the
elements outside rtol/atol fails the run, so a few corrupted rows are caught) and
timed. The kernel writes into an output buffer pre-filled with NaN, as the model
passes one, so rows it leaves unwritten fail the comparison.

Run:
    python3 op_tests/test_flydsl_minimax_m3_moe_prefill.py
"""

import argparse
import contextlib
import os
import subprocess
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

SUPPORTED_GFX = ["gfx950"]  # the MXFP4 MFMA prefill kernels need gfx950

HIDDEN, INTER, EXPERTS, TOPK = 6144, 768, 128, 4
ALPHA, LIMIT = 1.702, 7.0
TOKENS = [3072, 5000, 16383, 16384, 65536]  # block_m 128 below 16384, 256 from it
# bf16 output, so one rounding step (2^-8 relative) separates two correct answers; the
# largest difference seen on the reference at 3072-16384 tokens is 0.016
RTOL, ATOL = 0.05, 0.05


def _quant(x):
    """bf16 [..., K] -> (fp4 bytes [..., K/2], e8m0 bytes [..., K/32])."""
    from aiter.ops.quant import per_1x32_f4_quant

    q, s = per_1x32_f4_quant(x.reshape(-1, x.shape[-1]))
    k = x.shape[-1]
    return (
        q.view(torch.uint8).reshape(*x.shape[:-1], k // 2),
        s.view(torch.uint8).reshape(*x.shape[:-1], k // 32),
    )


def _dequant(q, s):
    """Inverse of `_quant` -> fp32 [..., K]."""
    k = q.shape[-1] * 2
    f = fp4_utils.mxfp4_to_f32(q.reshape(-1, k // 2).clone())
    e = fp4_utils.e8m0_to_f32(s.reshape(-1, k // 32)).repeat_interleave(32, dim=-1)
    return (f * e).reshape(*q.shape[:-1], k)


def _swiglu_oai(g, u):
    g = g.clamp(max=LIMIT)
    u = u.clamp(-LIMIT, LIMIT)
    return g * torch.sigmoid(g * ALPHA) * (u + 1.0)


def _reference(x_q, x_s, w13, w2, topk_ids, topk_weights):
    """w13 / w2: (fp4, e8m0) of the raw [E, 2I, H] / [E, H, I] weights."""
    x = _dequant(x_q, x_s)
    out = torch.zeros_like(x)
    for e in range(EXPERTS):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gu = x[tok] @ _dequant(w13[0][e], w13[1][e]).T
        h = _swiglu_oai(gu[:, :INTER], gu[:, INTER:]).to(torch.bfloat16)
        h = _dequant(*_quant(h))
        y = h @ _dequant(w2[0][e], w2[1][e]).T
        out.index_add_(0, tok, y * topk_weights[tok, slot, None])
    return out


def _make_inputs(n_tokens, seed=0, device="cuda"):
    from aiter.ops.shuffle import shuffle_scale

    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(
            torch.bfloat16
        )

    x = randn(n_tokens, HIDDEN, scale=1.0)
    w13 = _quant(randn(EXPERTS, 2 * INTER, HIDDEN, scale=HIDDEN**-0.5))
    w2 = _quant(randn(EXPERTS, HIDDEN, INTER, scale=INTER**-0.5))
    logits = torch.rand(n_tokens, EXPERTS, device=device, generator=g)
    topk_ids = logits.topk(TOPK, dim=1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.rand(n_tokens, TOPK, device=device, generator=g), dim=1
    )

    def kernel_form(w):  # the layouts of the kernel docstrings
        q, s = w
        return (
            shuffle_weight(q.view(dtypes.fp4x2), layout=(16, 16)).view(torch.uint8),
            shuffle_scale(s.reshape(-1, s.shape[-1])),
        )

    return x, w13, w2, topk_ids, topk_weights, kernel_form(w13), kernel_form(w2)


@contextlib.contextmanager
def _rejects(name, first_arg=None):
    """The body must raise AssertionError (an input check), not run. With `first_arg`,
    the failed check must be the one whose message starts with it, so an earlier check
    cannot mask the one under test."""
    try:
        yield
    except AssertionError as e:
        got = e.args[0][0] if e.args and isinstance(e.args[0], tuple) else None
        if first_arg is not None and got != first_arg:
            raise RuntimeError(f"{name}: rejected by another check: {e.args}") from e
        return
    raise RuntimeError(f"accepted: {name}")


def _setup(n_tokens, device="cuda"):
    """-> (call, out, ref): the kernel call into a NaN-filled `out`, and the reference."""
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_mxfp4.prefill import (
        a4w4_prefill_moe,
    )

    x, w13, w2, ids, wts, (w13_k, w13_ks), (w2_k, w2_ks) = _make_inputs(
        n_tokens, device=device
    )
    x_q, x_s = _quant(x)
    out = torch.full((n_tokens, HIDDEN), float("nan"), dtype=dtypes.bf16, device=device)

    def call():
        return a4w4_prefill_moe(
            (x_q, x_s),
            w13_k,
            w13_ks,
            w2_k,
            w2_ks,
            wts,
            ids,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_experts=EXPERTS,
            out=out,
        )

    # the kernel returns bf16: round the reference the same way before comparing
    ref = _reference(x_q, x_s, w13, w2, ids, wts).to(dtypes.bf16)
    return call, out, ref


def _check(n_tokens, device="cuda"):
    call, out, ref = _setup(n_tokens, device)
    assert call() is out
    torch.cuda.synchronize(device)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg=f"{device} n_tokens {n_tokens}: ",
    )
    assert err == 0, f"{err:.2e} of the elements are outside rtol/atol"
    return call, err


@benchmark()
def test_a4w4_prefill_moe(n_tokens):
    call, err = _check(n_tokens)
    _, us = run_perftest(call)
    # gemm1 (gate and up, 2 * H * 2I) and gemm2 (2 * I * H) per routed token
    flops = n_tokens * TOPK * (4 * HIDDEN * INTER + 2 * INTER * HIDDEN)
    # activations, every expert's weights and scales read once, the bf16 partials written
    # by gemm2 and read by the reduce, the output
    nbytes = (
        n_tokens * (HIDDEN // 2 + HIDDEN // 32)
        + EXPERTS * 3 * INTER * HIDDEN // 2 * (1 + 1 / 16)
        + 2 * n_tokens * TOPK * HIDDEN * 2
        + n_tokens * HIDDEN * 2
    )
    return {
        "gfx": get_gfx_runtime(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def test_mxfp4_quant():
    """`mxfp4_quant` prepares the chain's activations: its codes and scales must equal
    the independent `_quant` reference, into a fresh buffer and a caller-provided one.
    """
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_mxfp4.prefill import (
        mxfp4_quant,
    )

    g = torch.Generator(device="cuda").manual_seed(1)
    for n_tokens in (3072, 5000):
        x = (torch.randn(n_tokens, HIDDEN, device="cuda", generator=g) * 2).to(
            torch.bfloat16
        )
        ref_q, ref_s = _quant(x)

        q, s = mxfp4_quant(x, None, None)
        assert q.shape == (n_tokens, HIDDEN // 2) and q.dtype == torch.uint8
        assert s.shape == (n_tokens, HIDDEN // 32) and s.dtype == torch.uint8
        assert torch.equal(q, ref_q), f"allocated codes differ at {n_tokens} tokens"
        assert torch.equal(s, ref_s), f"allocated scales differ at {n_tokens} tokens"

        # caller-provided buffers, pre-filled so unwritten bytes show up
        q_buf = torch.full_like(q, 0xA5)
        s_buf = torch.full_like(s, 0xA5)
        q2, s2 = mxfp4_quant(x, q_buf, s_buf)
        assert q2 is q_buf and s2 is s_buf
        assert torch.equal(q_buf, ref_q), f"provided codes differ at {n_tokens} tokens"
        assert torch.equal(s_buf, ref_s), f"provided scales differ at {n_tokens} tokens"


def test_a4w4_prefill_moe_non_current_device():
    if torch.cuda.device_count() < 2:
        aiter.logger.warning("non-current device check needs two GPUs; skipping")
        return
    # tensors on cuda:1 while cuda:0 is current: a fresh process, as FlyDSL binds a
    # compiled kernel to the first device that runs it
    here, name = os.path.split(
        os.path.splitext(__file__)[0]
    )  # __name__ is __main__ here
    code = f"import importlib; importlib.import_module({name!r})._check(3072, 'cuda:1')"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([here, *sys.path])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_a4w4_prefill_moe_rejects_bad_inputs():
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_mxfp4.prefill import (
        MIN_PREFILL_TOKENS,
        a4w4_prefill_moe,
        supports_shapes,
    )

    n = MIN_PREFILL_TOKENS
    dev = "cuda"

    def zeros(*shape, dtype=torch.uint8, device=dev):
        return torch.zeros(*shape, dtype=dtype, device=device)

    def empty(
        *shape, dtype=torch.uint8, device=dev
    ):  # the checks never read the contents
        return torch.empty(*shape, dtype=dtype, device=device)

    # 31775 tokens at topk 11: the padding sentinel's partial row ends past 2^32 bytes
    m = 31775

    good = {
        "x_quant": (zeros(n, HIDDEN // 2), zeros(n, HIDDEN // 32)),
        "w13": zeros(EXPERTS, 2 * INTER, HIDDEN // 2),
        "w13_scale": zeros(EXPERTS * 2 * INTER, HIDDEN // 32),
        "w2": zeros(EXPERTS, HIDDEN, INTER // 2),
        "w2_scale": zeros(EXPERTS * HIDDEN, INTER // 32),
        "topk_weights": zeros(n, TOPK, dtype=torch.float32),
        "topk_ids": zeros(n, TOPK, dtype=torch.int32),
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_experts": EXPERTS,
    }
    # control: all-zero inputs of the right sizes are accepted and give zeros
    assert not a4w4_prefill_moe(**good).any()

    bad = {
        "out of the wrong length": {"out": zeros(n - 1, HIDDEN, dtype=torch.bfloat16)},
        "out of the wrong dtype": {"out": zeros(n, HIDDEN, dtype=torch.float32)},
        "w13 of another dtype, same shape": {
            "w13": zeros(EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.bfloat16)
        },
        "w2_scale of another dtype": {
            "w2_scale": zeros(EXPERTS * HIDDEN, INTER // 32, dtype=torch.float32)
        },
        "strided w13_scale": {
            "w13_scale": zeros(EXPERTS * 2 * INTER, HIDDEN // 16)[:, ::2]
        },
        "w13 without its last expert": {"w13": good["w13"][:-1]},
        "w2 without its last expert": {"w2": good["w2"][:-1]},
        "undersized w13_scale": {"w13_scale": good["w13_scale"][:-1]},
        "undersized w2_scale": {"w2_scale": good["w2_scale"][:-1]},
        "routing weights on the CPU": {
            "topk_weights": zeros(n, TOPK, dtype=torch.float32, device="cpu")
        },
        "partial rows past 2^32 bytes": {
            "x_quant": (zeros(m, HIDDEN // 2), zeros(m, HIDDEN // 32)),
            "topk_ids": zeros(m, 11, dtype=torch.int32),
            "topk_weights": zeros(m, 11, dtype=torch.float32),
        },
        "too few tokens": {
            "x_quant": (zeros(n - 1, HIDDEN // 2), zeros(n - 1, HIDDEN // 32))
        },
    }
    for name, change in bad.items():
        try:
            a4w4_prefill_moe(**{**good, **change})
        except AssertionError:
            continue
        raise RuntimeError(f"accepted: {name}")

    # the scale sort takes rows up to 16384 wide: supports_shapes stops at that ceiling
    # (17408 satisfies every other shape rule) and the launcher refuses the shape
    assert supports_shapes(16384, INTER) and not supports_shapes(17408, INTER)
    assert not supports_shapes(65536, INTER)
    sort_h = 17408
    too_wide = {
        **good,
        "hidden_size": sort_h,
        "x_quant": (zeros(n, sort_h // 2), zeros(n, sort_h // 32)),
        "w13": empty(EXPERTS, 2 * INTER, sort_h // 2),
        "w13_scale": empty(EXPERTS * 2 * INTER, sort_h // 32),
        "w2": empty(EXPERTS, sort_h, INTER // 2),
        "w2_scale": empty(EXPERTS * sort_h, INTER // 32),
    }
    with _rejects("hidden past the scale sort's 16384", first_arg=sort_h):
        a4w4_prefill_moe(**too_wide)
    del too_wide

    # hidden 16384 (the widest the scale sort takes) with 512 experts passes every shape
    # rule, but w13 is 6.4e9 bytes: its offsets would wrap. first_arg checks that it is
    # the byte bound that rejects, not the hidden-size guard above.
    wide_h, wide_e = 16384, 512
    wide = {
        **good,
        "hidden_size": wide_h,
        "num_experts": wide_e,
        "x_quant": (zeros(n, wide_h // 2), zeros(n, wide_h // 32)),
        "w13": empty(wide_e, 2 * INTER, wide_h // 2),
        "w13_scale": empty(wide_e * 2 * INTER, wide_h // 32),
        "w2": empty(wide_e, wide_h, INTER // 2),
        "w2_scale": empty(wide_e * wide_h, INTER // 32),
    }
    with _rejects("w13 bytes past 2^32", first_arg="w13"):
        a4w4_prefill_moe(**wide)
    del wide

    # only gfx950 has this kernel's LDS and MFMA; the rest of the inputs are valid
    from unittest import mock

    arch = mock.patch(
        "aiter.jit.utils.chip_info.get_gfx_runtime", return_value="gfx942"
    )
    with arch, _rejects("gfx942"):
        a4w4_prefill_moe(**good)


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "minimax_m3 prefill moe unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=TOKENS)
    args = parser.parse_args()
    df = pd.DataFrame([test_a4w4_prefill_moe(n) for n in args.tokens])
    aiter.logger.info(
        "minimax_m3 prefill moe summary (markdown):\n%s", df.to_markdown(index=False)
    )
    test_mxfp4_quant()
    test_a4w4_prefill_moe_rejects_bad_inputs()
    test_a4w4_prefill_moe_non_current_device()


if __name__ == "__main__":
    main()
