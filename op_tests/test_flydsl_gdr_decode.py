# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI-visible numerics and contract checks for FlyDSL GDR decode, including KDA.

Run:
    python3 op_tests/test_flydsl_gdr_decode.py
    python3 op_tests/test_flydsl_gdr_decode.py -d bf16 -b 1 4
"""

from __future__ import annotations

import argparse
import re
import time

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_gdr_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
G_MIN = -5.0
K = V = 128
RTOL = ATOL = 1e-3
_TOL_ERR_RATIO = 1e-3
_PERF_ROTATION_BUDGET = 1024**3
_MAX_PERF_ROTATIONS = 101

try:
    from kda_ref import kda_gate, l2norm, naive_recurrent_kda
except ModuleNotFoundError as e:
    if e.name != "kda_ref":
        raise
    from op_tests.kda_ref import kda_gate, l2norm, naive_recurrent_kda


def _perf_rotation_count(*tensors):
    bytes_per_call = max(1, sum(t.nbytes for t in tensors))
    return max(1, min(_MAX_PERF_ROTATIONS, _PERF_ROTATION_BUDGET // bytes_per_call))


def _kda_decode_work(B, H, query, state, A_log, indices):
    tokens = B
    flops = tokens * H * (7 * K * V + 2 * V) + tokens * H * 6 * K
    data_elements = 3 * tokens * H * K + tokens * H * V + tokens * H
    state_elements = B * H * K * V
    nbytes = (
        data_elements * query.element_size()
        + 2 * state_elements * state.element_size()
        + H * K * 4
        + H * A_log.element_size()
        + B * indices.element_size()
    )
    return flops, nbytes


def _gdr_decode_work(b, sq, num_k_heads, num_v_heads, query, state, A_log, indices):
    tokens = b * sq
    flops = tokens * num_v_heads * (7 * K * V + 2 * V) + tokens * num_k_heads * 6 * K
    data_elements = (
        2 * tokens * num_k_heads * K
        + 2 * tokens * num_v_heads * V
        + 2 * tokens * num_v_heads
        + num_v_heads
    )
    state_elements = b * num_v_heads * K * V
    nbytes = (
        data_elements * query.element_size()
        + 2 * state_elements * state.element_size()
        + num_v_heads * A_log.element_size()
        + b * indices.element_size()
    )
    return flops, nbytes


def _kda_inputs(B, H, dt, first_index, padded, shuffle, seed=0, indices_stride=1):
    """Build one KDA decode case.

    ``shuffle`` picks the state layout the caller holds: False is already
    (D_v, D_k), True is (D_k, D_v) and the wrapper transposes. With K == V the
    two are shape-identical, so feeding the wrong one tests the harness.
    """
    torch.manual_seed(seed)
    T = 1
    args = {
        "q": torch.randn(B, T, H, K, dtype=dt, device="cuda"),
        "k": torch.randn(B, T, H, K, dtype=dt, device="cuda"),
        "v": torch.randn(B, T, H, V, dtype=dt, device="cuda"),
        "a": torch.randn(B, T, H, K, dtype=dt, device="cuda"),
        "b": torch.randn(B, T, H, dtype=dt, device="cuda"),
        "dt_bias": torch.randn(H, K, dtype=torch.float32, device="cuda") * 0.1,
        "A_log": torch.randn(H, dtype=torch.float32, device="cuda") * 0.5,
        "out": torch.zeros(B, T, H, V, dtype=dt, device="cuda"),
    }

    d0, d1 = (K, V) if shuffle else (V, K)
    n_slots = B + first_index
    if padded:
        # Serving stacks pass the pool as one field of a wider per-slot allocation.
        storage = torch.randn(
            n_slots, H * K * V + 17, dtype=torch.float32, device="cuda"
        )
        pool = storage[:, : H * K * V].view(n_slots, H, d0, d1)
        assert not pool.is_contiguous()
        assert pool.stride()[1:] == (d0 * d1, d1, 1)
    else:
        pool = torch.randn(n_slots, H, d0, d1, dtype=torch.float32, device="cuda")

    if indices_stride > 1:
        storage = torch.zeros(B, indices_stride, dtype=torch.int32, device="cuda")
        indices = storage[:, 0]
    else:
        indices = torch.empty(B, dtype=torch.int32, device="cuda")
    indices.copy_(
        torch.arange(first_index, first_index + B, dtype=torch.int32, device="cuda")
    )
    return args, pool, indices


def _clone_pool(pool):
    """Copy the pool keeping its layout; ``clone()`` densifies a padded view."""
    out = torch.empty_strided(
        pool.shape, pool.stride(), dtype=pool.dtype, device=pool.device
    )
    out.copy_(pool)
    assert out.stride() == pool.stride(), "kernel pool lost the padded stride"
    return out


def _kda_reference(args, initial_state):
    return naive_recurrent_kda(
        l2norm(args["q"]),
        l2norm(args["k"]),
        args["v"],
        kda_gate(args["a"], args["A_log"], args["dt_bias"], g_min=G_MIN),
        args["b"].float().sigmoid(),
        scale=K**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )


def _run_kda(args, indices, pool, out, shuffle):
    flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        pool,
        out,
        use_qk_l2norm=True,
        need_shuffle_state=shuffle,
    )


def _gdr_reference(
    query,
    key,
    value,
    a,
    b,
    dt_bias,
    A_log,
    indices,
    state,
    out,
    num_k_heads,
    num_v_heads,
):
    q = query.float()
    k = key.float()
    v = value.float()
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (K**-0.5)
    heads_per_k_head = num_v_heads // num_k_heads
    q = q.repeat_interleave(heads_per_k_head, dim=2)
    k = k.repeat_interleave(heads_per_k_head, dim=2)
    decay = torch.exp(
        -torch.exp(A_log.float())
        * F.softplus(a.float() + dt_bias.float(), beta=1.0, threshold=20.0)
    )
    beta = torch.sigmoid(b.float())
    for batch_idx, state_idx in enumerate(indices.tolist()):
        if state_idx < 0:
            continue
        h = state[state_idx].float()
        h = h * decay[batch_idx, 0, :, None, None]
        residual = v[batch_idx, 0] - torch.einsum("hkv,hk->hv", h, k[batch_idx, 0])
        residual = residual * beta[batch_idx, 0, :, None]
        h = h + k[batch_idx, 0, :, :, None] * residual[:, None, :]
        out[batch_idx, 0].copy_(torch.einsum("hkv,hk->hv", h, q[batch_idx, 0]))
        state[state_idx].copy_(h)


def _expect_raises(exc_type, pattern, fn):
    try:
        fn()
    except exc_type as exc:
        if re.search(pattern, str(exc)) is None:
            raise AssertionError(
                f"expected {exc_type.__name__} matching {pattern!r}, got {exc!r}"
            ) from exc
        return
    raise AssertionError(f"expected {exc_type.__name__} matching {pattern!r}")


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def _print_passed(passed, elapsed):
    print(f" {passed} passed in {elapsed:.2f}s ".center(78, "="))


@benchmark()
def test_kda_decode(b, h, dtype, shuffle, padded, first_index, indices_stride):
    args, pool, indices = _kda_inputs(
        b, h, dtype, first_index, padded, shuffle, indices_stride=indices_stride
    )
    initial_state = pool[indices.long()].clone()
    if not shuffle:
        initial_state = initial_state.transpose(-1, -2)
    ref_out, ref_state = _kda_reference(args, initial_state)

    def run(candidate_pool, candidate_out):
        _run_kda(args, indices, candidate_pool, candidate_out, shuffle)

    flops, nbytes = _kda_decode_work(b, h, args["q"], pool, args["A_log"], indices)
    ret = {"gfx": get_gfx()}
    perf_pool = _clone_pool(pool)
    perf_out = torch.zeros_like(args["out"])
    _, us = run_perftest(
        run,
        perf_pool,
        perf_out,
        num_rotate_args=_perf_rotation_count(perf_pool, perf_out),
    )
    kernel_pool = _clone_pool(pool)
    kernel_out = torch.zeros_like(args["out"])
    run(kernel_pool, kernel_out)
    got_state = kernel_pool[indices.long()]
    if not shuffle:
        got_state = got_state.transpose(-1, -2)
    err_out = checkAllclose(
        ref_out.to(dtypes.fp32),
        kernel_out.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="flydsl: KDA decode output ",
    )
    err_state = checkAllclose(
        ref_state.to(dtypes.fp32),
        got_state.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="flydsl: KDA decode state ",
    )
    assert err_out <= _TOL_ERR_RATIO and err_state <= _TOL_ERR_RATIO, (
        f"mismatch ratio exceeds {_TOL_ERR_RATIO:g} "
        f"(output {err_out:.3e}, state {err_state:.3e})"
    )
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = flops / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    ret["flydsl err"] = max(err_out, err_state)
    return ret


@benchmark()
def test_gdr_decode(b, dt_bias_dtype):
    num_k_heads, num_v_heads, sq = 2, 8, 1
    dtype = torch.bfloat16
    query = torch.randn(b, sq, num_k_heads, K, dtype=dtype, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn(b, sq, num_v_heads, V, dtype=dtype, device="cuda")
    a = torch.randn(b, sq, num_v_heads, dtype=dtype, device="cuda")
    beta = torch.randn_like(a)
    dt_bias = torch.randn(num_v_heads, dtype=dt_bias_dtype, device="cuda")
    dt_bias.uniform_(1, 2)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device="cuda")
    A_log.uniform_(0, 16)
    indices = torch.arange(b - 1, -1, -1, dtype=torch.int32, device="cuda")
    state = torch.randn(b, num_v_heads, K, V, dtype=torch.float32, device="cuda")
    ref_state = state.clone()
    ref_out = torch.zeros(b, sq, num_v_heads, V, dtype=dtype, device="cuda")
    _gdr_reference(
        query,
        key,
        value,
        a,
        beta,
        dt_bias,
        A_log,
        indices,
        ref_state,
        ref_out,
        num_k_heads,
        num_v_heads,
    )

    def run(candidate_state, candidate_out):
        flydsl_gdr_decode(
            query,
            key,
            value,
            a,
            beta,
            dt_bias,
            A_log,
            indices,
            candidate_state,
            candidate_out,
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )

    flops, nbytes = _gdr_decode_work(
        b, sq, num_k_heads, num_v_heads, query, state, A_log, indices
    )
    ret = {"gfx": get_gfx()}
    perf_state = state.clone()
    perf_out = torch.zeros_like(ref_out)
    _, us = run_perftest(
        run,
        perf_state,
        perf_out,
        num_rotate_args=_perf_rotation_count(perf_state, perf_out),
    )
    got_state = state.clone()
    got_out = torch.zeros_like(ref_out)
    run(got_state, got_out)
    err_out = checkAllclose(
        ref_out.to(dtypes.fp32),
        got_out.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="flydsl: GDR decode output ",
    )
    err_state = checkAllclose(
        ref_state.to(dtypes.fp32),
        got_state.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="flydsl: GDR decode state ",
    )
    assert err_out <= _TOL_ERR_RATIO and err_state <= _TOL_ERR_RATIO, (
        f"mismatch ratio exceeds {_TOL_ERR_RATIO:g} "
        f"(output {err_out:.3e}, state {err_state:.3e})"
    )
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = flops / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    ret["flydsl err"] = max(err_out, err_state)
    return ret


def test_channel_strided_a_is_rejected():
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )
    wide = torch.randn(B, 1, H, 2 * K, dtype=dt, device="cuda")
    strided_a = wide[..., ::2]
    assert strided_a.shape == args["a"].shape and strided_a.stride(-1) == 2

    def run():
        flydsl_gdr_decode(
            args["q"],
            args["k"],
            args["v"],
            strided_a,
            args["b"],
            args["dt_bias"],
            args["A_log"],
            indices,
            pool,
            args["out"],
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )

    _expect_raises(AssertionError, "dense along D_k", run)


def test_consumer_native_a_layout_is_rejected():
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )
    consumer_native = args["a"].transpose(0, 1)
    assert consumer_native.shape == (1, B, H, K)
    assert consumer_native.stride(-1) == 1

    def run():
        flydsl_gdr_decode(
            args["q"],
            args["k"],
            args["v"],
            consumer_native,
            args["b"],
            args["dt_bias"],
            args["A_log"],
            indices,
            pool,
            args["out"],
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )

    _expect_raises(ValueError, r"`a` must have shape", run)


def test_staging_copies_are_ordered_against_a_caller_supplied_stream():
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )
    wide_bias = torch.randn(H, 2 * K, dtype=torch.float32, device="cuda") * 0.1
    args["dt_bias"] = wide_bias[:, ::2]
    assert not args["dt_bias"].is_contiguous()
    initial_state = pool[indices.long()].clone()
    kernel_pool = pool.clone()
    # Warm the config so JIT does not overlap the current-stream sleep below.
    _run_kda(args, indices, pool.clone(), torch.empty_like(args["out"]), True)

    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda._sleep(100_000_000)
    flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        kernel_pool,
        args["out"],
        use_qk_l2norm=True,
        need_shuffle_state=True,
        stream=side,
    )
    torch.cuda.synchronize()
    ref_out, ref_state = _kda_reference(args, initial_state)
    err_out = checkAllclose(
        ref_out.to(dtypes.fp32),
        args["out"].to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="stream o ",
    )
    err_state = checkAllclose(
        ref_state.to(dtypes.fp32),
        kernel_pool[indices.long()].to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="stream ht ",
    )
    assert err_out <= _TOL_ERR_RATIO and err_state <= _TOL_ERR_RATIO


def test_negative_slot_is_skipped_and_zero_is_not():
    B, H, dt = 4, 12, torch.bfloat16
    args, pool, _ = _kda_inputs(B, H, dt, first_index=0, padded=False, shuffle=True)
    indices = torch.tensor([-1, 0, 1, 2], dtype=torch.int32, device="cuda")
    args["out"].fill_(7.0)
    kernel_pool = pool.clone()
    _run_kda(args, indices, kernel_pool, args["out"], True)

    assert flydsl_gdr_decode.zeroes_invalid_output
    assert (args["out"][0] == 0).all()
    assert torch.equal(kernel_pool[3], pool[3])

    live_args = dict(args)
    for name in ("q", "k", "v", "a", "b"):
        live_args[name] = args[name][1:]
    ref_out, ref_state = _kda_reference(live_args, pool[:3])
    err_out = checkAllclose(
        ref_out.to(dtypes.fp32),
        args["out"][1:].to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="live o ",
    )
    err_state = checkAllclose(
        ref_state.to(dtypes.fp32),
        kernel_pool[:3].to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="live ht ",
    )
    assert err_out <= _TOL_ERR_RATIO and err_state <= _TOL_ERR_RATIO


def test_only_kda_uses_the_kda_tiling_table():
    from aiter.ops.flydsl import linear_attention_kernels as lak

    kda_config = {"NUM_BLOCKS_PER_V_DIM": 4, "NUM_WARPS": 2, "WARP_THREADS_K": 32}
    dtypes_key = ("torch.bfloat16", "torch.float32")
    geometry = (4, 1, 12, 12, 128, 128)
    part = (lak.GDR_GPU_ARCH, lak.get_num_sms())
    saved = lak._KDA_DECODE_BY_PART
    try:
        lak._KDA_DECODE_BY_PART = {part: {4: (4, 2, 32)}}
        assert lak.get_default_kwargs(*dtypes_key, *geometry, "kda") == kda_config
        assert lak.get_default_kwargs(
            *dtypes_key, *geometry, "gdr"
        ) == lak._decode_tiling(
            geometry[0], geometry[3], geometry[4], geometry[5], dtypes_key[1]
        )
    finally:
        lak._KDA_DECODE_BY_PART = saved


def test_a_row_swept_on_another_part_is_not_reused():
    from aiter.ops.flydsl import linear_attention_kernels as lak

    dtypes_key = ("torch.bfloat16", "torch.float32")
    geometry = (4, 1, 12, 12, 128, 128)
    foreign = (lak.GDR_GPU_ARCH, lak.get_num_sms() + 1)
    saved = lak._KDA_DECODE_BY_PART
    try:
        lak._KDA_DECODE_BY_PART = {foreign: {4: (4, 2, 32)}}
        assert lak.get_default_kwargs(
            *dtypes_key, *geometry, "kda"
        ) == lak._decode_tiling(
            geometry[0], geometry[3], geometry[4], geometry[5], dtypes_key[1]
        )
    finally:
        lak._KDA_DECODE_BY_PART = saved


def test_state_store_follows_the_pool_dtype_not_the_activations(act_dtype):
    B, Sq, H, D = 2, 1, 4, 128

    def run(state_dtype):
        torch.manual_seed(0)
        kw = {"dtype": act_dtype, "device": "cuda"}
        q, k, v = (torch.randn(B, Sq, H, D, **kw) for _ in range(3))
        a, b = (torch.randn(B, Sq, H, **kw) for _ in range(2))
        dt_bias = torch.rand(H, dtype=torch.float32, device="cuda") + 1.0
        A_log = torch.rand(H, dtype=torch.float32, device="cuda") * 4.0
        indices = torch.arange(B, dtype=torch.int32, device="cuda")
        torch.manual_seed(99)
        state = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda").to(
            state_dtype
        )
        out = torch.zeros(B, Sq, H, D, **kw)
        flydsl_gdr_decode(
            q,
            k,
            v,
            a,
            b,
            dt_bias,
            A_log,
            indices,
            state,
            out,
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )
        return state.float()

    ref = run(torch.float32)
    got = run(torch.bfloat16)
    assert (got - ref).abs().max() < 0.05 * ref.abs().max()


def test_kda_supports_bf16_state():
    def run(state_dtype):
        args, pool, indices = _kda_inputs(
            B=2,
            H=4,
            dt=torch.bfloat16,
            first_index=0,
            padded=False,
            shuffle=True,
            seed=0,
        )
        pool = pool.to(state_dtype)
        _run_kda(args, indices, pool, args["out"], True)
        return args["out"].float(), pool.float()

    ref_out, ref_state = run(torch.float32)
    got_out, got_state = run(torch.bfloat16)
    assert (got_out - ref_out).abs().max() < 0.05 * ref_out.abs().max()
    assert (got_state - ref_state).abs().max() < 0.05 * ref_state.abs().max()


def test_fp32_activations_are_rejected():
    B, Sq, H, D = 1, 1, 4, 128
    kw = {"dtype": torch.float32, "device": "cuda"}
    q, k, v = (torch.randn(B, Sq, H, D, **kw) for _ in range(3))
    a, b = (torch.randn(B, Sq, H, **kw) for _ in range(2))

    def run():
        flydsl_gdr_decode(
            q,
            k,
            v,
            a,
            b,
            torch.rand(H, dtype=torch.float32, device="cuda"),
            torch.rand(H, dtype=torch.float32, device="cuda"),
            torch.zeros(B, dtype=torch.int32, device="cuda"),
            torch.randn(B, H, D, D, dtype=torch.float32, device="cuda"),
            torch.zeros(B, Sq, H, D, **kw),
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )

    _expect_raises(ValueError, r"`query` must be fp16 or bf16", run)


_KDA_CASES = [
    (1, 8, torch.bfloat16, True, False, 0, 1),
    (4, 12, torch.bfloat16, True, False, 0, 1),
    (2, 12, torch.float16, True, False, 0, 1),
    (4, 12, torch.bfloat16, False, True, 1, 8),
    (4, 12, torch.bfloat16, False, False, 1, 1),
    (4, 12, torch.bfloat16, True, True, 1, 1),
    (4, 12, torch.bfloat16, True, False, 1, 8),
    (1, 12, torch.bfloat16, True, False, 1, 1),
    (64, 12, torch.bfloat16, True, False, 1, 1),
    (256, 12, torch.bfloat16, True, False, 1, 1),
    (2, 64, torch.bfloat16, True, False, 1, 1),
]
_GDR_CASES = [
    (1, torch.float32),
    (1, torch.bfloat16),
    (128, torch.bfloat16),
]


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL GDR decode unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16, dtypes.fp16],
        nargs="*",
        default=[dtypes.bf16, dtypes.fp16],
        help="Activation dtype.\n        e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 4, 64, 128, 256],
        help="Batch sizes.\n        e.g.: -b 1 4",
    )
    args = parser.parse_args()
    passed = 0
    t0 = time.perf_counter()

    kda_rows = []
    for b, h, dtype, shuffle, padded, first_index, indices_stride in _KDA_CASES:
        if dtype not in args.dtype or b not in args.batch:
            continue
        kda_rows.append(
            test_kda_decode(b, h, dtype, shuffle, padded, first_index, indices_stride)
        )
        passed += 1
    if kda_rows:
        summarize("flydsl_gdr_decode kda", kda_rows)

    gdr_rows = []
    for b, dt_bias_dtype in _GDR_CASES:
        if torch.bfloat16 not in args.dtype or b not in args.batch:
            continue
        gdr_rows.append(test_gdr_decode(b, dt_bias_dtype))
        passed += 1
    if gdr_rows:
        summarize("flydsl_gdr_decode gdr", gdr_rows)

    test_channel_strided_a_is_rejected()
    test_consumer_native_a_layout_is_rejected()
    test_staging_copies_are_ordered_against_a_caller_supplied_stream()
    test_negative_slot_is_skipped_and_zero_is_not()
    test_only_kda_uses_the_kda_tiling_table()
    test_a_row_swept_on_another_part_is_not_reused()
    passed += 6
    for act_dtype in args.dtype:
        test_state_store_follows_the_pool_dtype_not_the_activations(act_dtype)
        passed += 1
    test_kda_supports_bf16_state()
    test_fp32_activations_are_rejected()
    passed += 2
    _print_passed(passed, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
