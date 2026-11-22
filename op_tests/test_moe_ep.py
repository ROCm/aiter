# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import math
import os
from functools import partial
from itertools import product
from pathlib import Path
from statistics import median

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, pertoken_quant
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.fused_moe_bf16_asm import asm_moe
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import (
    moe_shuffle_scale,
    moe_shuffle_weight,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    perftest,
    run_perftest,
)
from aiter.utility import fp4_utils

logger = aiter.logger

BLOCK_SIZE_M = 32
MAX_TOKENS = 4096 * 4


@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        expert_mask,
        activation=activation,
    )


@perftest()
def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    a16=False,
    expert_mask=None,
    local_expert_hash=None,
):

    return asm_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        a16,
        expert_mask=expert_mask,
        local_expert_hash=local_expert_hash,
    )


quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
]


def test_fmoe_ep(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=2,
    ep=8,
):
    # This gpu id in EP, this example use the last id
    ep_id = ep - 1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
    # expert_mask = torch.randint(
    #     0, 2, (E + shared_E + 1,), dtype=dtypes.i32, device="cuda"
    # )
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1
    # # Get local expert Number in this gpu
    local_E = torch.sum(expert_mask).item()
    # The last expert
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1

    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    quantstr = quant_algo[quantAlgoId]
    quant_dtype = dtypes.i8 if quantstr.startswith("int8") else dtypes.fp8
    use_smooth = "smooth" in quantstr

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = (
            torch.randn(
                (local_E + shared_E, inter_dim * 2, model_dim),
                dtype=dtype,
                device="cuda",
            )
            / 10
        )
    else:
        w1 = (
            torch.randn(
                (local_E + shared_E, inter_dim, model_dim), dtype=dtype, device="cuda"
            )
            / 10
        )
    w2 = (
        torch.randn(
            (local_E + shared_E, model_dim, inter_dim), dtype=dtype, device="cuda"
        )
        / 10
    )
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    # if shared_E > 0:
    shared_E_score = 0.1
    # init total_topk_ids, inference time you just need to fill ns_topk_ids in total_topk_ids
    total_topk_ids = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.i32, device=input.device
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
    shared_expert_ids = [E + i for i in range(shared_E + 1)]
    s_topk_ids_list = [[fake_expertid] * (shared_E + 1)] * MAX_TOKENS
    for i in range(ep_id, MAX_TOKENS, ep):
        s_topk_ids_list[i] = shared_expert_ids
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=dtypes.i32, device=input.device)

    # init total_topk_weights, inference time you just need to fill ns_topk_weights in total_topk_weights
    total_topk_weights = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.fp32, device=input.device
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [topk, shared_E + 1], dim=1
    )
    s_topk_weights[:] = shared_E_score

    # inference time, use fused_topk to fill ns_topk_ids and ns_topk_weights
    fused_topk(input, score, topk, True, ns_topk_ids, ns_topk_weights)
    # inference time, topk_ids simply slices total_topk_ids into the number of input tokens, same for topk_weights
    topk_ids = total_topk_ids[:token]
    topk_weights = total_topk_weights[:token]

    # else:
    #     topk_ids, topk_weights = fused_topk(input, score, topk, True)

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input, w1, w2, topk_weights, topk_ids, expert_mask=expert_mask
        )

        # b implement
        torch_quant = aiter.get_torch_quant(aiter.QuantType.No)
        w1_qt, _w1_scale = torch_quant(w1, quant_dtype=None)
        w2_qt, _w2_scale = torch_quant(w2, quant_dtype=None)
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))

        # if use_g1u1:
        #     out_b = ref2
        #     avg_b = 9999
        #     print("asm g1u1 only support quant/smoothquant Now")
        # else:
        #     out_b, avg_b = asm_moe_test(
        #         input,
        #         w1_qt_aiter,
        #         w2_qt_aiter,
        #         topk_weights,
        #         topk_ids,
        #         expert_mask=expert_mask,
        #     )

        # test ck moe
        out_ck, _avg_ck = run_perftest(
            fused_moe,
            input,
            w1_qt_aiter,
            w2_qt_aiter,
            topk_weights,
            topk_ids,
            expert_mask,
            w1_scale=None,
            w2_scale=None,
            quant_type=aiter.QuantType.No,
            activation=ActivationType.Silu,
            doweight_stage1=False,
        )

        # msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us, ck_avg: {avg_ck:>8.2f} us, uplift: {avg_c/avg_b-1:.1%}"
        # checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        checkAllclose(ref2, out_ck, rtol=0.01, atol=10, msg="ck check")

    else:
        w1, fc1_scale = pertoken_quant(w1, quant_dtype=quant_dtype)
        w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

        sp1 = (local_E + shared_E, inter_dim)
        sp2 = (local_E + shared_E, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            # [expert, 1, model_dim]
            fc1_smooth_scale = torch.randn(sp2, dtype=dtypes.fp32, device="cuda")
            # [expert, 1, inter_dim]
            fc2_smooth_scale = torch.randn(sp1, dtype=dtypes.fp32, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            expert_mask,
        )

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        local_expert_hash = None
        if expert_mask is not None and use_smooth:
            local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
            local_expert_hash[local_expert_hash > 0] -= 1
            local_expert_hash[expert_mask == 0] = -1
        out_b, avg_b = asm_moe_test(
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            expert_mask=expert_mask,
            local_expert_hash=local_expert_hash,
        )

        def calculateTensorsSize(*args):
            num_btype = 0
            for el in args:
                if isinstance(el, torch.Tensor):
                    num_btype += el.element_size() * el.numel()
            return num_btype

        num_tb = calculateTensorsSize(
            input,
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
        ) / (1024 * 1024 * 1024 * 1024.0)
        bw = num_tb * 1e6 / avg_b
        print(
            f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, asm_bandwidth: {bw:>8.2f}TB/s"
        )

        if use_smooth and (
            (
                (inter_dim % 512 == 0 or inter_dim % 320 == 0)
                and (w1b.dtype == dtypes.fp8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 256 == 0 or inter_dim % 320 == 0 or inter_dim % 384 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 512 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim == w1b.shape[1])
            )
        ):
            out_b2, avg_b2 = asm_moe_test(
                input,
                w1b,
                w2b,
                topk_weights,
                topk_ids,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                a16=True,
                expert_mask=expert_mask,
            )
            msg = f"[perf] a8w8 asm: {avg_b:>8.2f} vs a16w8 asm: {avg_b2:>8.2f} ......"
            checkAllclose(out_b, out_b2, atol=10, msg=msg)

        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us ...... uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        # checkAllclose(ref2, avg_ck, rtol=0.01, atol=10)


# ---------------------------------------------------------------------------
# EP end-to-end with per_1x32 mxfp4 (a8w4 / a4w4) via fused_moe
# ---------------------------------------------------------------------------


def _per_1x32_mxfp4_quant(w):
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w_qt, w_scale = torch_quant(w, quant_dtype=dtypes.fp4x2)
    w_qt = w_qt.view(w.shape[0], w.shape[1], w.shape[2] // 2)
    return w_qt, w_scale


def _randn_or_const(shape, *, const_init, scale=1.0, dtype=dtypes.bf16, device="cuda"):
    """randn (scaled) by default; a constant VALUE tensor when const_init is set.

    Mirrors flydsl_tests/test_flydsl_grouped_gemm.py's --const-init: the const path
    fills with VALUE exactly (the ``scale`` only applies to the random path)."""
    if const_init is not None:
        return torch.full(shape, float(const_init), dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device) * scale


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """1 - cosine-similarity in double; matches test_moe_2stage.calc_diff."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    # Under EP a single token can route entirely to non-local experts, so both
    # ref and out are all-zero and denom==0. That is a correct (empty) result,
    # not a mismatch — guard the division so it doesn't report a spurious NaN.
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


summary_table = []

DEFAULT_EP_CSV_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "aiter", "configs", name)
    for name in (
        "tuned_grouped_fmoe.csv",
        "model_configs/kimik3_gfx1250_tuned_grouped_fmoe.csv",
    )
]


def _ep_token_counts(token, ep, running_tokens_across_dp=None):
    """CPU-known counts; do not truncate a non-divisible global token count."""
    if token < 1 or ep < 1:
        raise ValueError("token and ep must be positive")
    counts = running_tokens_across_dp
    if counts is None:
        counts = [token // ep + (rank < token % ep) for rank in range(ep)]
    if (
        len(counts) != ep
        or any(not isinstance(n, int) or n < 0 for n in counts)
        or sum(counts) != token
    ):
        raise ValueError(
            "running_tokens_across_dp must have ep nonnegative counts summing to token"
        )
    return list(counts)


def _gpu_events(prof):
    # PyTorch reports HIP device events as DeviceType.CUDA too.
    events = [e for e in prof.events() if str(e.device_type).split(".")[-1] == "CUDA"]
    if not events:
        raise RuntimeError(
            "torch.profiler returned no GPU events; check the profiler/runtime"
        )
    return events


def _profile_ep_graph(replay, names, num_warmup, num_iters, trace_path=None):
    for _ in range(num_warmup):
        replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(num_iters):
            replay()
        torch.cuda.synchronize()
    events = _gpu_events(prof)
    ret = {"us": sum(e.self_device_time_total for e in events) / num_iters}
    for stage, name in names.items():
        kernels = [e for e in events if e.name == name]
        if len(kernels) != num_iters:
            raise RuntimeError(
                f"{stage}: expected {num_iters} graph kernel events, got {len(kernels)} ({name})"
            )
        ret[f"{stage}_us"] = sum(e.self_device_time_total for e in kernels) / num_iters
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    return ret


def _benchmark_ep_call(
    moe_kwargs, *, bench_mode, num_warmup, num_iters, repeats, trace_dir=None
):
    """Capture the real launchers once; production graph captures one complete MOE."""
    if num_warmup < 0 or num_iters < 2 or repeats < 1:
        raise ValueError("warmup >= 0, iters >= 2 and repeats >= 1 are required")
    if bench_mode not in ("kernel", "fused", "graph"):
        raise ValueError(f"unknown bench_mode: {bench_mode}")
    if trace_dir and bench_mode != "graph":
        raise ValueError("trace_dir requires bench_mode='graph'")
    call = partial(fused_moe, **moe_kwargs)
    launches = []
    if get_gfx_runtime().split("-")[0] == "gfx1250":
        from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

        previous = grouped.kernel_bench_callable
        grouped.kernel_bench_callable = launches
        try:
            out = call()
        finally:
            grouped.kernel_bench_callable = previous
    else:
        if bench_mode != "fused":
            raise ValueError("per-kernel and graph EP timing require gfx1250")
        out = call()
    torch.cuda.synchronize()
    selected = [(name, fn) for name, fn in launches if name in ("gemm1", "gemm2")]
    launchers = dict(selected)
    names = {}
    if bench_mode != "fused":
        if len(selected) != 2 or set(launchers) != {"gemm1", "gemm2"}:
            raise RuntimeError(f"Missing GEMM launch capture: {tuple(launchers)}")
        for stage, fn in launchers.items():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                fn()
                torch.cuda.synchronize()
            kernels = _gpu_events(prof)
            if len(kernels) != 1:
                raise RuntimeError(
                    f"{stage}: expected one GPU launch, got {[e.name for e in kernels]}"
                )
            names[stage] = kernels[0].name
        if len(set(names.values())) != 2:
            raise RuntimeError(f"GEMM symbols must be distinguishable: {names}")
        for stage, name in names.items():
            print(f"[ep-kernel-name] {stage}: {name}", flush=True)
    if bench_mode == "graph":
        # JIT/cache initialization precedes capture; graph owns all intermediate buffers.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(max(1, num_warmup)):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
    samples = []
    for repeat in range(repeats):
        if bench_mode == "graph":
            path = None
            if trace_dir:
                from time import time_ns

                path = Path(trace_dir) / f"moe_ep_{time_ns()}_{repeat + 1}.json"
            sample = _profile_ep_graph(graph.replay, names, num_warmup, num_iters, path)
        elif bench_mode == "kernel":
            sample = {}
            for stage, fn in launchers.items():
                _, us = run_perftest(
                    fn,
                    num_warmup=num_warmup,
                    num_iters=num_iters,
                    num_rotate_args=1,
                    testGraph=False,
                )
                sample[f"{stage}_us"] = us
            sample["us"] = sample["gemm1_us"] + sample["gemm2_us"]
        else:
            out, us = run_perftest(
                call,
                num_warmup=num_warmup,
                num_iters=num_iters,
                num_rotate_args=1,
                testGraph=False,
            )
            sample = {"us": us}
        if any(
            not isinstance(u, (int, float)) or not math.isfinite(u) or u <= 0
            for u in sample.values()
        ):
            raise RuntimeError(f"Invalid {bench_mode} timings: {sample}")
        print(
            f"[ep-bench] mode={bench_mode} repeat={repeat + 1}/{repeats} {sample}",
            flush=True,
        )
        samples.append(sample)
    ret = {key: median(sample[key] for sample in samples) for key in samples[0]}
    ret.update({f"{stage}_kernel": name for stage, name in names.items()})
    ret.update(
        samples=samples,
        bench_mode=bench_mode,
        timing_scope="gemm1+gemm2" if bench_mode == "kernel" else "fused_moe",
    )
    return out, ret


def _ep_work_metrics(moe_kwargs, timings):
    """Useful routed FLOPs and logical bytes; TB/s is not measured HBM traffic."""
    ids = moe_kwargs["topk_ids"].long()
    mask = moe_kwargs["expert_mask"]
    valid = (ids >= 0) & (ids < mask.numel())
    valid &= mask[ids.clamp(0, mask.numel() - 1)].bool()
    nvt = moe_kwargs.get("num_local_tokens")
    if nvt is not None:
        valid &= (
            torch.arange(ids.shape[0], device=ids.device)[:, None] < nvt.reshape(-1)[0]
        )
    routes = int(valid.sum().item())
    active = int(ids[valid].unique().numel())
    model_dim = moe_kwargs["hidden_states"].shape[1]
    inter_dim = moe_kwargs["w1"].shape[1] // 2
    a_bytes = 0.5 if moe_kwargs.get("quant_dtype_a") == dtypes.fp4x2 else 1.0
    w1_bytes = active * 2 * inter_dim * model_dim * (0.5 + 1 / 32)
    w2_bytes = active * inter_dim * model_dim * (0.5 + 1 / 32)
    work = {
        "gemm1": (
            4 * routes * model_dim * inter_dim,
            w1_bytes + routes * (model_dim + inter_dim) * (a_bytes + 1 / 32),
        ),
        "gemm2": (
            2 * routes * model_dim * inter_dim,
            w2_bytes + routes * (inter_dim * (a_bytes + 1 / 32) + model_dim * 2),
        ),
    }
    ret = {
        "local_routes": routes,
        "active_experts": active,
        "TFLOPS": sum(flops for flops, _ in work.values()) / timings["us"] / 1e6,
        "TB/s": sum(nbytes for _, nbytes in work.values()) / timings["us"] / 1e6,
    }
    for stage, (flops, nbytes) in work.items():
        us = timings.get(f"{stage}_us")
        if us is not None:
            ret[f"{stage}_TFLOPS"] = flops / us / 1e6
            ret[f"{stage}_TB/s"] = nbytes / us / 1e6
    return ret


def save_ep_moe_fixture(path, moe_kwargs, reference, metadata=None):
    """Call outside capture before arena reuse. torch.save preserves storage/strides."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "save the fixture outside CUDA graph capture, before arena reuse"
        )
    torch.cuda.synchronize()
    args = dict(moe_kwargs)
    for key in ("activation", "quant_type", "quant_type_a"):
        if args.get(key) is not None:
            args[key] = int(args[key])
    if isinstance(args.get("gate_mode"), GateMode):
        args["gate_mode"] = args["gate_mode"].value
    attrs = {
        name: {
            key: getattr(tensor, key)
            for key in ("is_shuffled", "gate_up_interleave")
            if hasattr(tensor, key)
        }
        for name, tensor in args.items()
        if isinstance(tensor, torch.Tensor)
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": args,
            "reference": reference,
            "tensor_attributes": attrs,
            "metadata": metadata or {},
        },
        path,
    )


@benchmark()
def test_fmoe_ep_fixture(
    fixture,
    *,
    bench_mode="graph",
    num_warmup=2,
    num_iters=101,
    repeats=1,
    trace_dir=None,
):
    """Replay the actual model call and compare with its saved output, without reshuffling."""
    payload = torch.load(fixture, map_location="cuda", weights_only=True)
    moe_kwargs = payload["args"]
    # output= may alias the saved reference; freeze the oracle before replay writes it.
    ref = payload["reference"].detach().clone()
    for key, enum in (
        ("activation", ActivationType),
        ("quant_type", QuantType),
        ("quant_type_a", QuantType),
    ):
        if moe_kwargs.get(key) is not None:
            moe_kwargs[key] = enum(moe_kwargs[key])
    required = {
        "hidden_states",
        "w1",
        "w2",
        "topk_weight",
        "topk_ids",
        "expert_mask",
        "quant_dtype_a",
        "gate_mode",
    }
    if not required.issubset(moe_kwargs):
        raise ValueError(f"Missing fixture arguments: {required - moe_kwargs.keys()}")
    if (
        moe_kwargs["hidden_states"].dtype != dtypes.bf16
        or moe_kwargs.get("num_local_tokens") is not None
        or moe_kwargs.get("stage2_scatter") is not None
        or GateMode(moe_kwargs["gate_mode"]) != GateMode.INTERLEAVE
        or moe_kwargs["quant_dtype_a"] not in (dtypes.fp4x2, dtypes.fp8)
        or QuantType(moe_kwargs.get("quant_type", QuantType.No)) != QuantType.per_1x32
        or any(moe_kwargs.get(key) is not None for key in ("a1_scale", "a2_scale"))
        or moe_kwargs.get("doweight_stage1", False)
        or moe_kwargs.get("hidden_pad", 0)
        or moe_kwargs.get("intermediate_pad", 0)
        or moe_kwargs.get("quant_dtype_a2") not in (None, moe_kwargs["quant_dtype_a"])
    ):
        raise ValueError(
            "fixture must contain an unfused BF16 gather + grouped A4W4/A8W4 call"
        )
    hidden, w1, w2 = (moe_kwargs[key] for key in ("hidden_states", "w1", "w2"))
    ids, weights = (moe_kwargs[key] for key in ("topk_ids", "topk_weight"))
    mask = moe_kwargs["expert_mask"]
    if (
        hidden.ndim != 2
        or ids.ndim != 2
        or weights.shape != ids.shape
        or ids.shape[0] != hidden.shape[0]
        or ref.shape != hidden.shape
        or ids.dtype != dtypes.i32
        or weights.dtype != dtypes.fp32
        or mask is None
        or mask.ndim != 1
        or mask.dtype != dtypes.i32
        or w1.ndim != 3
        or w2.ndim != 3
        or w1.shape[0] != w2.shape[0]
        or w1.dtype not in (torch.uint8, dtypes.fp4x2)
        or w2.dtype not in (torch.uint8, dtypes.fp4x2)
        or w1.shape[1] != w2.shape[2] * 4
        or w1.shape[2] * 2 != hidden.shape[1]
        or w2.shape[1] != hidden.shape[1]
        or int(mask.sum().item()) != w1.shape[0]
        or not all(
            isinstance(moe_kwargs.get(key), torch.Tensor)
            for key in ("w1_scale", "w2_scale")
        )
    ):
        raise ValueError(
            "fixture tensor shapes/dtypes do not match an EP grouped MXFP4 call"
        )
    for name, attrs in payload.get("tensor_attributes", {}).items():
        for key, value in attrs.items():
            if key not in ("is_shuffled", "gate_up_interleave"):
                raise ValueError(f"Unsupported tensor attribute: {key}")
            setattr(moe_kwargs[name], key, value)
    out, timings = _benchmark_ep_call(
        moe_kwargs,
        bench_mode=bench_mode,
        num_warmup=num_warmup,
        num_iters=num_iters,
        repeats=repeats,
        trace_dir=trace_dir,
    )
    err = checkAllclose(
        ref.float(),
        out.float(),
        atol=1e-3,
        rtol=1e-3,
        tol_err_ratio=0,
        msg="EP model fixture replay",
    )
    assert err == 0, f"fixture replay mismatch: {err}"
    return {
        "gfx": get_gfx_runtime(),
        "tokens": moe_kwargs["topk_ids"].shape[0],
        "topk": moe_kwargs["topk_ids"].shape[1],
        "model_dim": hidden.shape[1],
        "inter_dim": w1.shape[1] // 2,
        "local_experts": w1.shape[0],
        "swiglu_limit": moe_kwargs.get("swiglu_limit"),
        "err": err,
        **timings,
        **_ep_work_metrics(moe_kwargs, timings),
    }


def run_ep_csv(args):
    """Sweep the requested CSV or all default gfx1250 EP CSVs."""
    if get_gfx_runtime().split("-")[0] != "gfx1250":
        raise RuntimeError(f"EP CSV requires gfx1250, got {get_gfx_runtime()}")
    csv_paths = [args.csv_path] if args.csv_path is not None else DEFAULT_EP_CSV_PATHS
    failures = []
    for csv_path in csv_paths:
        try:
            _run_ep_csv_file(args, csv_path)
        except (AssertionError, OSError, ValueError) as exc:
            failures.append((csv_path, str(exc)))
            logger.exception("[ep-csv] %s: FAILED", csv_path)
    if failures:
        raise AssertionError(f"EP CSV failures: {failures}")


def _run_ep_csv_file(args, csv_path):
    """Check every EP row using this file's runtime tuning configuration."""
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    os.environ["AITER_CONFIG_GROUPED_FMOE"] = csv_path
    # The lookup cache is keyed by workload, so invalidate it when switching CSVs.
    grouped._find_grouped_config.cache_clear()
    print(f"[ep-csv] sweeping settings from {csv_path}", flush=True)
    with open(csv_path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    ep_rows = [(index, row) for index, row in enumerate(rows) if row["topk"] == "-1"]
    if not ep_rows:
        raise ValueError(f"No EP rows (topk=-1) in {csv_path}")

    failures = []
    for index, row in ep_rows:
        try:
            local_experts = int(row["expert"])
            # The 65-expert configs include one shared expert on each rank.
            shared_experts = 1 if local_experts == 65 else 0
            global_experts = (local_experts - shared_experts) * args.csv_ep
            quant_label = {
                "torch.float8_e4m3fn": "a8w4_mxfp4",
                "torch.float4_e2m1fn_x2": "a4w4_mxfp4",
            }[row["q_dtype_a"]]
            activation = {
                "ActivationType.Silu": ActivationType.Silu,
                "ActivationType.Swiglu": ActivationType.Swiglu,
                "ActivationType.Situv2": ActivationType.Situv2,
            }[row["act_type"]]
            print(
                f"\n[ep-csv] row {index}: {quant_label} token={row['token']} "
                f"model_dim={row['model_dim']} inter_dim={row['inter_dim']} "
                f"local_experts={local_experts} ep={args.csv_ep} "
                f"topk={args.csv_topk}",
                flush=True,
            )
            for input_mode, rank, mode in product(
                args.input_mode, args.ep_rank, args.bench_mode or [None]
            ):
                test_fmoe_ep_mxfp4(
                    quant_label,
                    int(row["token"]),
                    int(row["model_dim"]),
                    int(row["inter_dim"]),
                    global_experts,
                    args.csv_topk,
                    shared_E=shared_experts,
                    ep=args.csv_ep,
                    ep_mode=args.ep_mode,
                    fake_ep_rank=args.fake_ep_rank,
                    const_init=args.const_init,
                    activation=activation,
                    input_mode=input_mode,
                    ep_rank=rank,
                    swiglu_limit=args.swiglu_limit,
                    bench_mode=mode,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    repeats=args.repeats,
                    trace_dir=args.trace_dir,
                    running_tokens_across_dp=args.running_tokens_across_dp,
                    arena_rows=args.arena_rows,
                    seed=args.seed,
                )
        except Exception as exc:  # Keep checking the remaining CSV rows.
            failures.append((index, str(exc)))
            logger.exception("[ep-csv] row %s: FAILED", index)

    print(
        f"[ep-csv] {csv_path}: "
        f"{len(ep_rows) - len(failures)}/{len(ep_rows)} rows passed",
        flush=True,
    )
    if failures:
        raise AssertionError(f"EP CSV failures: {failures}")


@benchmark()
def test_fmoe_ep_mxfp4(
    quant_label,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    shared_E=2,
    ep=8,
    ep_mode="real",
    fake_ep_rank=0,
    const_init=None,
    activation=ActivationType.Silu,
    *,
    input_mode="legacy",
    ep_rank=None,
    running_tokens_across_dp=None,
    arena_rows=None,
    swiglu_limit=None,
    bench_mode=None,
    num_warmup=2,
    num_iters=101,
    repeats=1,
    trace_dir=None,
    seed=None,
):
    """End-to-end EP fused_moe with per_1x32 mxfp4 weights.
    quant_label ∈ {"a8w4_mxfp4", "a4w4_mxfp4"}.

    ep_mode selects how tokens/routing reach fused_moe:
      * "real" (default): simulate MORI dispatch. `token` is the GLOBAL token count;
        a source token is received iff it owns >=1 local expert (deduplicated), so
        total_recv <= token and non-local routes are masked. Buffer is trimmed to
        trim_M = token with a padded tail. AITER_MOE_EXPERT_BALANCE=true uses
        synthetic routing balanced across global experts and EP ranks, matching
        ATOM's init_balance_router_logits; otherwise routing scores are random.
      * "fake": mirror ATOM fake-EP (ATOM_FAKE_EP + --fake-eplb,
        atom/model_ops/moe.py:121-136). `token` is the PER-RANK batch M; every
        token's topk picks are redirected onto this rank's local expert block via a
        rolling window, so all M*topk routes are local (no mask, no dedup) and the
        grouped GEMM runs the full M*topk load. total_recv == trim_M == M.
        fake_ep_rank selects the local block [rank*L,(rank+1)*L) by default (0,
        matches ATOM_FAKE_EP_RANK). ep_rank overrides the rank in either mode.

    In real mode `token` is the global token count across all EP ranks. The
    unfused gather shape model after dispatch+trim is:

      graph_bs    = running_tokens_across_dp[ep_rank] (this rank before dispatch)
      n_src       = sum(running_tokens_across_dp)     (all source ranks)
      total_recv  = #tokens routed to THIS rank (MORI's total_recv_t, <= n_src)
      trim_M      = min(arena_rows, n_src) if arena_rows is given, else n_src

    Unlike a balanced "every token arrives" assumption, `total_recv` is derived
    from the actual routing exactly like MORI dispatch (and run_ref() in
    test_mori_all2all.py): a source token is received iff it owns >=1 *local*
    expert (routed or active shared), deduplicated to one buffer row per token.
    Legacy tests forward MORI's device scalar `total_recv_t` as
    `num_local_tokens` (mirrors
    bench_mega_moe.py's DeviceMoEPipeline._layer_step, where total_recv_t
    comes straight from op.dispatch and feeds moe_forward's num_local_tokens).

    The dispatch buffer has `trim_M` rows; legacy adds shared/fake route columns.
    Only the first `total_recv` rows carry valid data; the `[total_recv, trim_M)`
    tail is dead padding (legacy fused_moe uses `num_local_tokens` to skip it).
    expert_mask filters non-local experts so fused_moe only computes on this rank's
    experts. `input_mode="atom"` instead mirrors the current unfused FlyDSL
    gather: no shared/fake column, masked-sentinel padding, CPU-known static
    receive bound, `num_local_tokens=None` and explicit activation dtype. Pass
    `swiglu_limit=10` for Flash; other models retain their own clamp semantics.
    The actual model's routed tensors/weights can be replayed with a fixture.
    """
    _gfx = get_gfx_runtime().split("-")[0]
    if _gfx not in ["gfx950", "gfx1250"]:
        print(f"skip {quant_label}: mxfp4 requires gfx950/gfx1250, got {_gfx}")
        return
    if input_mode not in ("legacy", "atom"):
        raise ValueError(f"unknown input_mode: {input_mode}")
    if ep_mode not in ("real", "fake"):
        raise ValueError(f"unknown ep_mode: {ep_mode}")
    if quant_label not in ("a4w4_mxfp4", "a8w4_mxfp4"):
        raise ValueError(f"unknown quant_label: {quant_label}")
    if input_mode == "atom" and (_gfx != "gfx1250" or ep_mode != "real" or shared_E):
        raise ValueError("ATOM input mode requires gfx1250, real EP and shared_E=0")
    counts = _ep_token_counts(token, ep, running_tokens_across_dp)
    if E < 1 or E % ep or not 1 <= topk <= E or shared_E < 0:
        raise ValueError("E must be divisible by ep, and 1 <= topk <= E")
    if min(model_dim, inter_dim) < 1 or model_dim % 32 or inter_dim % 32:
        raise ValueError("model_dim and inter_dim must be positive multiples of 32")
    if arena_rows is not None and arena_rows < 1:
        raise ValueError("arena_rows must be positive")
    if seed is not None:
        torch.manual_seed(seed)
    if ep_mode == "fake" and (
        running_tokens_across_dp is not None or arena_rows is not None
    ):
        raise ValueError(
            "running_tokens_across_dp and arena_rows only apply to real EP"
        )
    if swiglu_limit is not None and (
        swiglu_limit < 0 or not math.isfinite(swiglu_limit)
    ):
        raise ValueError("swiglu_limit must be finite and nonnegative")
    if bench_mode is None:
        kernel_bench = os.environ.get("AITER_EP_KERNEL_BENCH", "1") in (
            "1",
            "true",
            "True",
            "yes",
        )
        bench_mode = (
            "graph"
            if input_mode == "atom"
            else ("kernel" if kernel_bench and _gfx == "gfx1250" else "fused")
        )
    # ---------- ATOM shape model (MoriV2ModularKernel) ----------
    # Before MORI dispatch: each of `ep` ranks holds `graph_bs` tokens, each
    #   with `topk` *global* expert ids (spanning all E experts).
    # MORI dispatch: sends each token to every rank that owns at least one of
    #   its topk experts. The token (hidden_states row) is sent once per unique
    #   dest_pe (deduplicated); all topk idx/weight travel with it.
    # After dispatch: recv buffer shape (mr, hidden_dim) with (mr, topk) ids.
    #   total_recv = number of unique tokens that landed on this rank.
    # After trim: the buffer retains one row per source token. Top-k remains a
    # separate routing dimension and does not enlarge the activation buffer.
    #
    # Realistic simulation:
    #   experts_per_rank = E // ep. total_recv is computed from the actual routing
    #   (tokens owning >=1 local expert, deduplicated) -- MORI's total_recv_t --
    #   rather than assuming every token arrives. total_recv <= n_src = token.
    experts_per_rank = E // ep

    # Which EP rank this process simulates. real mode keeps the historical last
    # rank (ep-1); fake mode uses fake_ep_rank to match ATOM_FAKE_EP_RANK (default
    # 0) so the local expert block lines up block-for-block with the fake-EP
    # server (init_balance_router_logits' fake_ep_rank, moe.py:134).
    ep_id = fake_ep_rank if ep_mode == "fake" else ep - 1
    if ep_rank is not None:
        ep_id = ep_rank
    if not 0 <= ep_id < ep:
        raise ValueError(f"ep_id={ep_id} out of range [0,{ep})")
    local_expert_start = ep_id * experts_per_rank
    local_expert_end = (ep_id + 1) * experts_per_rank
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[local_expert_start:local_expert_end] = 1
    local_E = int(torch.sum(expert_mask).item())
    fake_expertid = expert_mask.numel() - 1
    expert_mask[-1] = 0
    expert_mask[E:-1] = 1

    dtype = dtypes.bf16
    balanced = os.environ.get("AITER_MOE_EXPERT_BALANCE", "false").lower() == "true"
    routing = "balanced" if balanced or ep_mode == "fake" else "random"

    if ep_mode == "fake":
        # ---------- ATOM fake-EP (ATOM_FAKE_EP + --fake-eplb) shape model ----------
        # Mirrors init_balance_router_logits()'s fake_ep_rank branch,
        # atom/model_ops/moe.py:121-136. Under ATOM_FAKE_EP the parallel config is
        # forced to dp_size=1 (FusedMoEParallelConfig.make, moe.py:191-202), so
        # use_all2all_kernels is False (moe.py:171-173) and there is NO MORI
        # dispatch/combine: the full per-rank batch flows straight into fused_moe
        # (forward_impl, moe.py:3654-3687). --fake-eplb then rewrites the router so
        # every token's topk picks land on THIS rank's local block via a rolling
        # window (moe.py:131-135). Consequences vs the real path:
        #   * `token` is the PER-RANK forward batch M (server-side M), not a global
        #     count -- nothing shrinks it.
        #   * all M*topk routes are local => no expert_mask drop, no dedup.
        #   * total_recv == M and trim_M == M (no CUDAGraph pad tail); the grouped
        #     GEMM computes the full M*topk load (a fully-loaded EP rank).
        M = token
        L = experts_per_rank
        graph_bs = M
        n_src = M
        total_recv = M
        trim_M = M

        input_ = _randn_or_const((M, model_dim), const_init=const_init, dtype=dtype)

        # Rolling-window local routing == moe.py:131-135:
        #   local_slot = (t*topk + k) % L ; expert_ids = ep_id*L + local_slot
        # topk <= L keeps the topk picks distinct within a token, and the window
        # advancing by topk each row keeps the L local experts balanced.
        assert topk <= L, f"fake-eplb requires topk({topk}) <= experts_per_rank({L})"
        t = torch.arange(M, device="cuda").unsqueeze(1)
        k = torch.arange(topk, device="cuda").unsqueeze(0)
        local_slot = (t * topk + k) % L
        expert_ids = ep_id * L + local_slot  # (M, topk) int64, all local

        # Faithful replay of moe.py:130-135 followed by the REAL routing primitive:
        # scatter 1.0 into synthetic router_logits exactly like the fake-eplb branch
        # (moe.py builds a full (max_num_tokens, E) table then slices [:M]; because
        # the formula is per-row, building M rows here is identical), then run
        # fused_topk with renormalize=True -- the same topk the real branch uses --
        # so topk_ids AND topk_weights come from the actual kernel, not hardcoded.
        # Equal selected logits => softmax+renormalize gives uniform weights, but
        # now via the real path. (routed_scaling_factor is a later model-forward
        # multiply, not part of routing, and the real branch doesn't model it
        # either.)
        router_logits = torch.zeros((M, E), dtype=dtype, device="cuda")
        router_logits.scatter_(1, expert_ids, 1.0)
        # No shared/fake column: moe_sorting reads topk from topk_ids.shape[1]
        # (aiter/fused_moe.py:396,527,569), so (M, topk) expands to exactly M*topk
        # local routes -- no masked padding column.
        topk_ids = torch.empty((M, topk), dtype=dtypes.i32, device="cuda")
        topk_weights = torch.empty((M, topk), dtype=dtypes.fp32, device="cuda")
        fused_topk(input_, router_logits, topk, True, topk_ids, topk_weights)

        # Every row is valid; no padded tail to skip.
        num_local_tokens = torch.tensor([total_recv], dtype=dtypes.i32, device="cuda")

        # Reference sees exactly the same batch + routing.
        ref_input = input_
        ref_topk_ids = topk_ids
        ref_topk_weights = topk_weights

        print(
            f"  [EP sim/fake] per_rank_M={M} topk={topk} ep={ep} L={L} "
            f"local_experts=[{local_expert_start},{local_expert_end}) "
            f"routes={M * topk} (all local, no mask/dedup) trim_M={trim_M}"
        )
    else:
        graph_bs = counts[ep_id]
        n_src = sum(counts)
        trim_M = min(n_src, arena_rows) if arena_rows is not None else n_src

        # ---------- Simulate MORI dispatch output ----------
        # Step 1: generate the `n_src` *source* tokens (what all EP ranks hold
        #   before dispatch). Each token picks topk experts from ALL E experts via
        #   fused_topk (the router runs *before* dispatch on the source rank).
        src_input = _randn_or_const(
            (n_src, model_dim), const_init=const_init, dtype=dtype
        )
        if balanced:
            if E % ep != 0 or not 1 <= topk <= E:
                raise ValueError(
                    "Balanced real EP routing requires E divisible by ep and "
                    f"1 <= topk <= E, got E={E}, ep={ep}, topk={topk}"
                )
            # Match ATOM's global balanced router: walk ranks first, then
            # local expert slots. Each expert gets floor/ceil(n_src*topk/E)
            # routes, and each token's routes span EP ranks evenly.
            t = torch.arange(n_src, device="cuda").unsqueeze(1)
            k = torch.arange(topk, device="cuda").unsqueeze(0)
            p = t * topk + k
            expert_ids = (p % ep) * experts_per_rank + (p // ep) % experts_per_rank
            src_score = torch.full((n_src, E), -10.0, dtype=dtype, device="cuda")
            src_score.scatter_(1, expert_ids, 10.0)
        else:
            src_score = torch.randn((n_src, E), dtype=dtype, device="cuda")
        extra_slots = shared_E + 1 if input_mode == "legacy" else 0
        src_topk_ids = torch.empty(
            (n_src, topk + extra_slots), dtype=dtypes.i32, device="cuda"
        )
        src_topk_weights = torch.empty(
            (n_src, topk + extra_slots), dtype=dtypes.fp32, device="cuda"
        )
        ns_ids, ns_wts = src_topk_ids[:, :topk], src_topk_weights[:, :topk]
        if extra_slots:
            shared_expert_ids = [E + i for i in range(extra_slots)]
            s_ids_list = [[fake_expertid] * extra_slots] * n_src
            for i in range(ep_id, n_src, ep):
                s_ids_list[i] = shared_expert_ids
            src_topk_ids[:, topk:] = torch.tensor(
                s_ids_list, dtype=dtypes.i32, device="cuda"
            )
            src_topk_weights[:, topk:] = 0.1
        fused_topk(src_input, src_score, topk, True, ns_ids, ns_wts)

        # Step 2: MORI dispatch keeps only the tokens that own >=1 *local* expert
        #   on this rank (routed OR active shared), deduplicated to one row per
        #   token -- exactly run_ref()'s `mask.any(1)` selection in
        #   test_mori_all2all.py. The received count is the device scalar MORI
        #   returns as total_recv_t.
        recv_mask = expert_mask[src_topk_ids.long()].bool().any(dim=1)
        recv_input = src_input[recv_mask].contiguous()
        recv_topk_ids = src_topk_ids[recv_mask].contiguous()
        recv_topk_weights = src_topk_weights[recv_mask].contiguous()
        total_recv = int(recv_input.shape[0])
        assert total_recv <= trim_M, f"total_recv={total_recv} exceeds trim_M={trim_M}"
        print(
            f"  [EP sim] global_token={token} topk={topk} ep={ep} graph_bs={graph_bs} "
            f"rank={ep_id} running_tokens_across_dp={counts} n_src={n_src} "
            f"total_recv={total_recv} trim_M={trim_M} routing={routing}"
        )

        # Step 3: build the dispatch buffer at trim_M (ATOM's CUDAGraph bound).
        #   First total_recv rows = valid received tokens (from step 2).
        #   Rows [total_recv, trim_M) = dead padding (MORI arena tail).
        input_ = _randn_or_const(
            (trim_M, model_dim), const_init=const_init, dtype=dtype
        )
        input_[:total_recv] = recv_input

        total_topk_ids = torch.full(
            (trim_M, topk + extra_slots),
            fake_expertid if input_mode == "atom" else 0,
            dtype=dtypes.i32,
            device="cuda",
        )
        total_topk_ids[:total_recv] = recv_topk_ids

        total_topk_weights = torch.zeros(
            (trim_M, topk + extra_slots), dtype=dtypes.fp32, device="cuda"
        )
        total_topk_weights[:total_recv] = recv_topk_weights

        topk_ids = total_topk_ids
        topk_weights = total_topk_weights

        # Current ATOM FlyDSL gather passes None; invalid tail ids mask themselves.
        num_local_tokens = (
            None
            if input_mode == "atom"
            else torch.tensor([total_recv], dtype=dtypes.i32, device="cuda")
        )

        # Reference: same received tokens, same routing, same expert_mask.
        ref_input = recv_input
        ref_topk_ids = recv_topk_ids
        ref_topk_weights = recv_topk_weights

    total_local = local_E + shared_E
    w1 = _randn_or_const(
        (total_local, inter_dim * 2, model_dim),
        const_init=const_init,
        scale=0.1,
        dtype=dtype,
    )
    w2 = _randn_or_const(
        (total_local, model_dim, inter_dim),
        const_init=const_init,
        scale=0.1,
        dtype=dtype,
    )

    w1_qt, w1_scale = _per_1x32_mxfp4_quant(w1)
    w2_qt, w2_scale = _per_1x32_mxfp4_quant(w2)

    # Reference uses the dequantized mxfp4 weights so the comparison is
    # apples-to-apples (kernel sees the same quantized values). Mirrors the
    # mxfp4_to_f32 + e8m0_to_f32 dequant in aiter/fused_moe.py:2018-2021 and
    # the quantized-weight reference path in test_moe_2stage.py:333-345.
    def _dequant(w_qt, w_scale, orig_shape):
        wf = fp4_utils.mxfp4_to_f32(w_qt).view(*orig_shape)
        sf = fp4_utils.e8m0_to_f32(w_scale).view(orig_shape[0], orig_shape[1], -1)
        sf = sf.unsqueeze(-1).expand(-1, -1, -1, 32).reshape(*orig_shape)
        return (wf * sf).to(dtype)

    if total_recv and not (_gfx == "gfx1250" and quant_label == "a4w4_mxfp4"):
        w1_deq = _dequant(w1_qt, w1_scale, w1.shape)
        w2_deq = _dequant(w2_qt, w2_scale, w2.shape)
    local_topk_ids = torch.where(
        (ref_topk_ids >= local_expert_start) & (ref_topk_ids < local_expert_end),
        ref_topk_ids - local_expert_start,
        torch.where(
            (ref_topk_ids >= E) & (ref_topk_ids < E + shared_E),
            ref_topk_ids - E + local_E,
            -1,
        ),
    )
    if total_recv == 0:
        ref = torch.empty((0, model_dim), dtype=dtype, device="cuda")
    elif _gfx == "gfx1250" and quant_label == "a4w4_mxfp4":
        # Match both activation quantization points in the grouped A4W4 path.
        # The generic BF16 reference above otherwise compares against A16W4
        # and overstates the numerical error from the two MXFP4 round trips.
        a1_q, a1_scale = per_1x32_f4_quant(
            ref_input, quant_dtype=dtypes.fp4x2, shuffle=False
        )
        a2 = torch_moe_stage1(
            a1_q,
            w1_qt,
            w2_qt,
            ref_topk_weights,
            local_topk_ids,
            dtype=dtype,
            activation=activation,
            quant_type=QuantType.per_1x32,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            swiglu_limit=swiglu_limit,
        )
        ref_rows, ref_topk = local_topk_ids.shape
        a2_q, a2_scale = per_1x32_f4_quant(
            a2.contiguous().view(ref_rows * ref_topk, inter_dim),
            quant_dtype=dtypes.fp4x2,
            shuffle=False,
        )
        ref = torch_moe_stage2(
            a2_q.view(ref_rows, ref_topk, inter_dim // 2),
            w1_qt,
            w2_qt,
            ref_topk_weights,
            local_topk_ids,
            dtype=dtype,
            quant_type=QuantType.per_1x32,
            w2_scale=w2_scale,
            a2_scale=a2_scale,
            doweight=True,
        )
    elif swiglu_limit:
        a2 = torch_moe_stage1(
            ref_input,
            w1_deq,
            w2_deq,
            ref_topk_weights,
            local_topk_ids,
            dtype=dtype,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        ref = torch_moe_stage2(
            a2,
            w1_deq,
            w2_deq,
            ref_topk_weights,
            local_topk_ids,
            dtype=dtype,
            doweight=True,
        )
    else:
        ref = torch_moe(
            ref_input,
            w1_deq,
            w2_deq,
            ref_topk_weights,
            ref_topk_ids,
            expert_mask=expert_mask,
            activation=activation,
        )

    if _gfx == "gfx1250":
        # gfx1250 grouped GEMM path: FlyDSL grouped layout. Weights stay uint8;
        # quant_dtype_a explicitly selects FP8 or FP4 below.
        w1_u8 = w1_qt.view(torch.uint8)
        w2_u8 = w2_qt.view(torch.uint8)
        # gugu (INTERLEAVE) stage1 layout so the EP path is routed through the
        # TDM batched GEMM (_grouped_a8w4_tdm_moe, gugu-only). gate/up are
        # row-interleaved ([g0,u0,g1,u1,...]) inside moe_shuffle_weight/scale,
        # matching flydsl_tests/test_flydsl_grouped_gemm.py.
        w1_a = moe_shuffle_weight(
            w1_u8, experts_cnt=total_local, is_guinterleave=True, gate_up=True
        )
        w2_a = shuffle_weight(w2_u8, layout=(16, 16))
        w1_s = moe_shuffle_scale(
            w1_scale.contiguous(),
            experts_cnt=total_local,
            is_guinterleave=True,
            gate_up=True,
        )
        w2_s = moe_shuffle_scale(w2_scale.contiguous(), experts_cnt=total_local)
        gate_mode = GateMode.INTERLEAVE.value
        act = activation
        os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
    elif quant_label == "a8w4_mxfp4":
        # gfx950 a8w4 (fp8 activations, mxfp4 weights): use the CK a16w4 layout
        # — weights interleaved on N (gate/up) — paired with gate_mode=INTERLEAVE
        # at the call site. The FlyDSL fp4 (16,16) shuffle is separated and
        # would mis-route gate/up here.
        w1_a = shuffle_weight_a16w4(w1_qt, 16, True)
        w1_s = shuffle_scale_a16w4(w1_scale, total_local, True)
        w2_a = shuffle_weight_a16w4(w2_qt, 16, False)
        w2_s = shuffle_scale_a16w4(w2_scale, total_local, False)
        act = activation
        gate_mode = GateMode.INTERLEAVE.value
        # The explicit FP8 dtype below avoids the small-M BF16/A16W4 fallback.
    elif quant_label == "a4w4_mxfp4":
        # gfx950 a4w4 (fp4 activations, mxfp4 weights): FlyDSL fp4/fp4 layout —
        # shuffle (16,16) + e8m0 scale shuffle, gate/up separated. Matches
        # test_moe_2stage.py:251-255 and pairs with gate_mode=SEPARATED.
        w1_a = shuffle_weight(w1_qt, layout=(16, 16))
        w2_a = shuffle_weight(w2_qt, layout=(16, 16))
        w1_s = fp4_utils.e8m0_shuffle(w1_scale)
        w2_s = fp4_utils.e8m0_shuffle(w2_scale)
        w1_a.is_shuffled = True
        w2_a.is_shuffled = True
        act = activation
        gate_mode = GateMode.SEPARATED.value
    else:
        raise ValueError(f"unknown quant_label: {quant_label}")
    activation_params = (
        {"beta": 4.0, "linear_beta": 25.0}
        if activation == ActivationType.Situv2
        else {}
    )
    # Correctness, eager timing and full-graph replay use exactly the same call.
    moe_kwargs = dict(
        hidden_states=input_,
        w1=w1_a,
        w2=w2_a,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=act,
        gate_mode=gate_mode,
        quant_type=QuantType.per_1x32,
        quant_dtype_a=dtypes.fp4x2 if quant_label == "a4w4_mxfp4" else dtypes.fp8,
        dtype=dtype,
        swiglu_limit=swiglu_limit,
        doweight_stage1=False,
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=None,
        a2_scale=None,
        num_local_tokens=num_local_tokens,
        stage2_scatter=None,
        hidden_pad=0,
        intermediate_pad=0,
        bias1=None,
        bias2=None,
        **activation_params,
    )
    out, timings = _benchmark_ep_call(
        moe_kwargs,
        bench_mode=bench_mode,
        num_warmup=num_warmup,
        num_iters=num_iters,
        repeats=repeats,
        trace_dir=trace_dir,
    )
    work_metrics = _ep_work_metrics(moe_kwargs, timings)
    us = timings["us"]
    gemm1_us, gemm2_us = (timings.get(f"{stage}_us") for stage in ("gemm1", "gemm2"))
    if gemm1_us is not None:
        print(
            f"[ep-kernel-bench] {quant_label} token={token}(local={total_recv}) "
            f"mode={bench_mode} gemm1 us = {gemm1_us:.2f} gemm2 us = {gemm2_us:.2f} "
            f"gemm1 TFLOPS = {work_metrics['gemm1_TFLOPS']:.2f} "
            f"gemm2 TFLOPS = {work_metrics['gemm2_TFLOPS']:.2f}",
            flush=True,
        )

    # Trim to valid prefix (total_recv rows); the [total_recv, trim_M) tail is padding.
    if input_mode == "atom":
        tail = out[total_recv:]
        tail_err = checkAllclose(
            torch.zeros_like(tail).float(),
            tail.float(),
            atol=0,
            rtol=0,
            tol_err_ratio=0,
            msg="EP masked receive tail",
        )
        assert tail_err == 0, f"masked tail contributed routed work: {tail_err}"
    out = out[:total_recv].float()
    ref = ref.float()
    assert (
        torch.isfinite(out).all() and torch.isfinite(ref).all()
    ), "nonfinite EP output/reference"
    diff = ref - out
    abs_err = diff.abs()
    abs_mean = abs_err.mean().item() if total_recv else 0.0
    abs_max = abs_err.max().item() if total_recv else 0.0
    rel_mean = (abs_err / (ref.abs() + 1e-6)).mean().item() if total_recv else 0.0
    logits_diff = _calc_diff(ref, out)

    _msg = (
        f"{quant_label} ep={ep} token={token}(recv={total_recv}, trim_M={trim_M}) "
        f"model_dim={model_dim} inter_dim={inter_dim} E={E} topk={topk}"
    )
    max_row_rel_l2 = None
    if input_mode == "atom":
        # The BF16 stage1 reference rounds before the second MXFP4 quantization.
        # Bound every token's residual, so a corrupted row cannot hide in a batch average.
        row_rel_l2 = diff.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)
        max_row_rel_l2 = row_rel_l2.max().item() if total_recv else 0.0
        row_err = checkAllclose(
            torch.zeros_like(row_rel_l2),
            row_rel_l2,
            atol=0.1,
            rtol=0,
            tol_err_ratio=0,
            msg=f"{_msg}: per-token relative L2",
        )
        assert row_err == 0, f"{_msg}: max row rel_l2={max_row_rel_l2} exceeds 0.1"
    if _gfx == "gfx1250":
        # The grouped a8w4 path quantizes activations to fp8, so the reference
        # (bf16 activations) differs elementwise by more than atol/rtol=5e-2
        # even without EP. Use the grouped tests' cosine criterion instead
        # (flydsl_tests/test_flydsl_grouped_gemm.py: logits_diff < 0.01).
        _logits_diff_tol = 0.01
        err = row_err if input_mode == "atom" else logits_diff
        _verdict = "PASSED" if logits_diff < _logits_diff_tol else "FAILED"
        _timing_label = (
            "GEMM1 + GEMM2" if bench_mode == "kernel" else "MOE GPU kernel total"
        )
        print(
            f"[aiter] {_msg} {_timing_label} us = {us:.2f} "
            f"logits_diff={logits_diff:.6f} "
            f"(tol {_logits_diff_tol}) {_verdict}"
        )
        assert (
            logits_diff < _logits_diff_tol
        ), f"{_msg}: logits_diff={logits_diff} exceeds {_logits_diff_tol}"
    else:
        err = checkAllclose(ref, out, atol=5e-2, rtol=5e-2, msg=_msg)
        assert err <= 0.05, f"{_msg}: {err:.2%} of elements outside rtol/atol"

    ret = {
        "gfx": _gfx,
        "quant": quant_label,
        "ep_mode": ep_mode,
        "input_mode": input_mode,
        "ep_rank": ep_id,
        "activation": str(activation),
        "routing": routing,
        "global_token": token,
        "running_tokens_across_dp": counts if ep_mode == "real" else None,
        "total_recv": total_recv,
        "trim_M": trim_M,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E": E,
        "topk": topk,
        "ep": ep,
        "swiglu_limit": swiglu_limit,
        "warmup": num_warmup,
        "iters": num_iters,
        "repeats": repeats,
        "seed": seed,
        "logits_diff": logits_diff,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "rel_mean": rel_mean,
        "max_row_rel_l2": max_row_rel_l2,
        "checkAllclose_err": err,
        **timings,
        **work_metrics,
    }
    summary_table.append(ret)
    return ret


def _print_ep_summary(name, rows):
    if not rows:
        return
    try:
        import pandas as pd

        # Symbols and individual samples are already printed above the table.
        df = pd.DataFrame(rows).drop(
            columns=["samples", "gemm1_kernel", "gemm2_kernel"], errors="ignore"
        )
        logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))
    except ImportError:
        print(f"{name} summary (pandas/tabulate unavailable):")
        for row in rows:
            print(row)


def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="EP MOE correctness and performance tests",
    )
    parser.add_argument(
        "-t",
        "--test",
        nargs="*",
        default=[
            "test_fmoe_16_bit",
            "g1u1_no_quant",
            "g1u1_int8quant",
            "g1u1_fp8quant",
            "g1u0_int8smoothquant",
            "g1u1_int8smoothquant",
            "g1u1_fp8smoothquant",
            "g1u1_a8w4_mxfp4",
            "g1u1_a4w4_mxfp4",
        ],
        help="Select test names, or ep_csv to sweep grouped EP CSV rows.",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "-m",
        "--token",
        type=int,
        nargs="*",
        default=[128],
        help="Global tokens in real EP; per-rank tokens in fake EP.",
    )
    parser.add_argument("-hd", "--hidden_dim", type=int, nargs="*", default=[4096])
    parser.add_argument("-id", "--inter_dim", type=int, nargs="*", default=[1024])
    parser.add_argument(
        "-e", "--expert", type=int, nargs="?", default=32, help="Global routed experts."
    )
    parser.add_argument("-k", "--topk", type=int, nargs="?", default=5)
    parser.add_argument("-ep", "--expert_parallelism", type=int, nargs="*", default=[8])
    parser.add_argument(
        "--ep-mode",
        choices=["real", "fake"],
        default="real",
        help="real: simulate deduplicated MORI dispatch; fake: redirect all routes locally.\n"
        "AITER_MOE_EXPERT_BALANCE=true selects balanced real routing; otherwise random.",
    )
    parser.add_argument(
        "--fake-ep-rank",
        type=int,
        default=0,
        help="Default fake EP rank (matches ATOM_FAKE_EP_RANK).",
    )
    parser.add_argument(
        "--const-init",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        metavar="VALUE",
        help="Constant activations/weights instead of random values; routing is unaffected.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random input and routing seed."
    )
    parser.add_argument(
        "--input-mode",
        choices=["legacy", "atom"],
        nargs="+",
        default=["legacy"],
        help="legacy: device count + fake route column; atom: unfused BF16 gather, true topk,\n"
        "masked-sentinel tail and num_local_tokens=None (gfx1250, real EP, no shared expert).",
    )
    parser.add_argument(
        "--ep-rank",
        type=int,
        nargs="+",
        default=[None],
        help="Ranks to simulate; default is ep-1 for real EP, --fake-ep-rank for fake EP.",
    )
    parser.add_argument(
        "--running-tokens-across-dp",
        type=int,
        nargs="+",
        default=None,
        help="CPU-known per-rank counts summing to -m; default distributes global tokens evenly.",
    )
    parser.add_argument(
        "--arena-rows",
        type=int,
        default=None,
        help="Static receive bound is min(arena_rows, global running tokens), real EP only.",
    )
    parser.add_argument(
        "--swiglu-limit",
        type=float,
        default=None,
        help="Same gate/up clamp in kernel and reference (Flash: 10). Default is unclamped.",
    )
    parser.add_argument(
        "--bench-mode",
        choices=["kernel", "fused", "graph"],
        nargs="+",
        default=None,
        help="kernel: isolated GEMMs; fused: eager MOE; graph: replay one complete MOE capture.\n"
        "Default: graph for atom; legacy follows AITER_EP_KERNEL_BENCH.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Report the median of repeated measurements.",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Export one Chrome trace per graph measurement.",
    )
    parser.add_argument(
        "--fixture",
        nargs="+",
        metavar="PATH",
        help="Replay saved model calls exclusively; use their recorded args/layouts, not CLI shapes.",
    )
    parser.add_argument(
        "--csv-path", default=None, help="Grouped tuning CSV to sweep with -t ep_csv."
    )
    parser.add_argument(
        "--csv-ep",
        type=int,
        default=4,
        help="Ranks used to reconstruct global experts in CSV rows.",
    )
    parser.add_argument(
        "--csv-topk",
        type=int,
        default=8,
        help="Actual routing topk; EP CSV uses -1 as a wildcard.",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters < 2 or args.repeats < 1:
        parser.error("--warmup >= 0, --iters >= 2 and --repeats >= 1 are required")
    gpu_arch = get_gfx_runtime().split("-")[0]
    torch.manual_seed(args.seed)
    summary_table.clear()

    if args.fixture:
        if "ep_csv" in args.test:
            parser.error("--fixture and -t ep_csv cannot be combined")
        if gpu_arch != "gfx1250":
            logger.warning(
                "EP model fixture replay requires gfx1250, got %s; skipping", gpu_arch
            )
            return
        rows = [
            test_fmoe_ep_fixture(
                fixture,
                bench_mode=mode,
                num_warmup=args.warmup,
                num_iters=args.iters,
                repeats=args.repeats,
                trace_dir=args.trace_dir,
            )
            for fixture, mode in product(args.fixture, args.bench_mode or ["graph"])
        ]
        _print_ep_summary("EP model fixture", rows)
        return

    if "ep_csv" in args.test:
        if len(args.test) != 1:
            parser.error("ep_csv must be selected alone")
        if args.csv_ep < 1 or args.csv_topk < 1:
            parser.error("--csv-ep and --csv-topk must be positive")
        if gpu_arch != "gfx1250":
            logger.warning("EP CSV sweep requires gfx1250, got %s; skipping", gpu_arch)
            return
        run_ep_csv(args)
        _print_ep_summary("EP MXFP4", summary_table)
        return

    legacy_tests = {
        "g1u1_no_quant": ("No", True, 2),
        "g1u1_int8quant": ("int8quant", True, 2),
        "g1u1_fp8quant": ("fp8quant", True, 2),
        "g1u0_int8smoothquant": ("int8smoothquant", False, 2),
        "g1u1_int8smoothquant": ("int8smoothquant", True, 0),
        "g1u1_fp8smoothquant": ("fp8smoothquant", True, 2),
    }
    for test in args.test:
        print(f"\nRunning test: {test}")
        if test == "test_fmoe_16_bit":
            print("test test_fmoe 16 bit")
        elif test in ("g1u1_a8w4_mxfp4", "g1u1_a4w4_mxfp4"):
            if gpu_arch not in ("gfx950", "gfx1250"):
                logger.warning(
                    "%s requires gfx950/gfx1250, got %s; skipping", test, gpu_arch
                )
                continue
            label = test.removeprefix("g1u1_")
            for m, hdim, idim, ep, input_mode, rank, mode in product(
                args.token,
                args.hidden_dim,
                args.inter_dim,
                args.expert_parallelism,
                args.input_mode,
                args.ep_rank,
                args.bench_mode or [None],
            ):
                if gpu_arch != "gfx1250" and (
                    input_mode == "atom" or mode in ("kernel", "graph")
                ):
                    logger.warning(
                        "%s input / %s timing requires gfx1250; skipping",
                        input_mode,
                        mode,
                    )
                    continue
                test_fmoe_ep_mxfp4(
                    label,
                    m,
                    hdim,
                    idim,
                    args.expert,
                    args.topk,
                    shared_E=0,
                    ep=ep,
                    ep_mode=args.ep_mode,
                    fake_ep_rank=args.fake_ep_rank,
                    const_init=args.const_init,
                    input_mode=input_mode,
                    ep_rank=rank,
                    running_tokens_across_dp=args.running_tokens_across_dp,
                    arena_rows=args.arena_rows,
                    swiglu_limit=args.swiglu_limit,
                    bench_mode=mode,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    repeats=args.repeats,
                    trace_dir=args.trace_dir,
                    seed=args.seed,
                )
        elif test in legacy_tests:
            if test == "g1u0_int8smoothquant" and gpu_arch != "gfx942":
                logger.warning(
                    "%s only runs on gfx942, got %s; skipping", test, gpu_arch
                )
                continue
            quant, use_g1u1, shared_E = legacy_tests[test]
            for dtype, m, hdim, idim, ep in product(
                args.dtype,
                args.token,
                args.hidden_dim,
                args.inter_dim,
                args.expert_parallelism,
            ):
                test_fmoe_ep(
                    dtype,
                    m,
                    hdim,
                    idim,
                    args.expert,
                    args.topk,
                    quant=quant,
                    use_g1u1=use_g1u1,
                    shared_E=shared_E,
                    ep=ep,
                )
        else:
            parser.error(f"Unknown test: {test}")
    _print_ep_summary("EP MXFP4", summary_table)


if __name__ == "__main__":
    main()
