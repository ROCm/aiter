# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""General gfx1250 ASM MoE UT: func, perf, and profile.

M in --shape is valid token-expert pairs per expert; tokens=M*experts/topk.
Stage1 uses N=2*inter_dim, K=hidden_size; stage2 uses N=hidden_size, K=inter_dim.
--tile-m controls sorting/kernel tiles independently of M and input tokens.

Stage1 computes silu(min(gate,10))*clamp(up,-10,10), with BF16 or fused MXFP4
output. Stage2 weights expert contributions and accumulates in BF16 atomics.
Inputs use row-major activations, 16-row gate/up blocks and n32k4 scales;
packing is delegated to aiter's public shuffle helpers.
Scales default to a -5 exponent bias, as in the MoE PoC's calibration, to
exercise SwiGLU's useful range. --scale-bias 0 keeps unshifted GEMM scales.

Examples (run from the aiter checkout with a ROCm Python environment):
    python op_tests/test_moe_asm.py --mode perf
    python op_tests/test_moe_asm.py --mode func
    python op_tests/test_moe_asm.py --mode profile --stage 1 --shape 32,6144,7168

PoC M32/M64 experiment: --stage 1 --shape 32,8192,7168 --experts 96 --topk 6
--tile-m 32 64 --intype a4w4 --data-init constant --scale-init constant
--scale-bias -5 --warmup 10 --iters 10 --rounds 6 --timer events --rotate 1
--no-check. Both tiles keep tokens=512 and 32 valid routes per expert. The
constant inputs are FP4 0.5 with E8M0 exponent -5 (PoC init=10). Routing uses
the PoC expert_init=2 shuffle. Omit --no-check to validate the same data.

Timings include the public operation's output clear. asm TB/s uses the PoC
input-byte model, not HBM counters. M64 also changes the kernel's LDS path.
"""

import argparse
import csv
import ctypes
import functools
import hashlib
import itertools
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.benchmark_data_init import (
    DATA_DISTS,
    E8M0_SCALE_DISTS,
    fill_fp4,
    fill_fp8,
    fill_scale_e8m0,
    make_generator,
)
from aiter.jit.core import AITER_ASM_DIR
from aiter.jit.utils.asm_guard import is_gfx1250_asm_supported
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.moe_op_gfx1250 import (
    _moe_stage1_asm,
    _moe_stage2_asm,
    moe_stage1_asm,
    moe_stage2_asm,
    shuffle_moe_input_scale_gfx1250,
    shuffle_moe_weights_gfx1250,
)
from aiter.test_common import run_perftest
from aiter.utility import fp4_utils

# Like test_f4gemm.py, run this CLI directly; --mode func includes API checks.
__test__ = False
SUPPORTED_GFX = ["gfx1250"]
PERF_SHAPES = {
    1: [(m, 6144, 7168) for m in (1, 2, 4, 8, 16, 32, 64)],
    2: [(m, 7168, 3072) for m in (1, 2, 4, 8, 16, 32, 64)],
}
# N6144 supports every shipped N384/N512 and 4x1/1x1 variant. The larger
# M crosses sorting-block boundaries; additional cases cover short K and N.
FUNC_SHAPES = {
    1: [(1, 6144, 1024), (7, 6144, 1536), (33, 6144, 2048)],
    2: [(1, 6144, 1024), (7, 6144, 1536), (33, 6144, 2048)],
}
_INTYPE = {"a4w4": "mxfp4", "a8w4": "mxfp8"}
_OUT_DTYPE = {"bf16": dtypes.bf16, "mxfp4": dtypes.fp4x2}


@functools.lru_cache
def _configs(stage):
    path = (
        Path(AITER_ASM_DIR) / get_gfx() / f"moe_stage{stage}" / f"moe_stage{stage}.csv"
    )
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in (
            "tile_m",
            "tile_n",
            "tile_k",
            "pf_tiles",
            "cluster_x",
            "cluster_y",
            "priority",
        ):
            row[key] = int(row[key])
        if not (path.parent / row["co_name"]).is_file():
            raise FileNotFoundError(path.parent / row["co_name"])
    return rows


def _supports(cfg, n, k):
    return (
        n > 0
        and n % (cfg["tile_n"] * cfg["cluster_x"]) == 0
        and k % cfg["tile_k"] == 0
        and k >= cfg["tile_k"] * cfg["pf_tiles"]
    )


def _routing(tokens, experts, topk, expert_init, seed):
    """PoC-compatible routing; DATA RNG is separate from glibc's routing RNG."""
    total = tokens * topk
    libc = ctypes.CDLL(None)
    libc.srand.argtypes, libc.srand.restype = [ctypes.c_uint], None
    libc.rand.argtypes, libc.rand.restype = [], ctypes.c_int
    # A, E weights, AS, WS, then srand(++seed) in the PoC host.
    libc.srand(seed + experts + 4)
    if expert_init in (2, 3):
        if total % experts:
            raise ValueError("balanced routing requires (tokens*topk)%experts==0")
        cap = total // experts
        pool = [i // cap for i in range(total)] if total else []
        if expert_init == 2:
            for i in range(total - 1, 0, -1):
                j = libc.rand() % (i + 1)
                pool[i], pool[j] = pool[j], pool[i]
    elif expert_init == 0:
        pool = [i % experts for i in range(total)]
    else:
        pool = [libc.rand() % experts for _ in range(total)]
    rows = [pool[t * topk : (t + 1) * topk] for t in range(tokens)]
    if expert_init == 2:
        for t in range(tokens):
            for first in range(topk):
                for second in range(first + 1, topk):
                    if rows[t][first] != rows[t][second]:
                        continue
                    old = rows[t][second]
                    repaired = False
                    for other in range(tokens):
                        for slot in range(topk):
                            new = rows[other][slot]
                            if (
                                other == t
                                or new == old
                                or new in rows[t]
                                or old in rows[other]
                            ):
                                continue
                            rows[t][second], rows[other][slot] = new, old
                            repaired = True
                            break
                        if repaired:
                            break
                    if not repaired:
                        raise RuntimeError(
                            "PoC balanced-random duplicate repair failed"
                        )
    ids = torch.tensor(rows, dtype=torch.int32, device="cpu").reshape(tokens, topk)
    counts = torch.bincount(ids.flatten().long(), minlength=experts)
    if expert_init == 2:
        assert torch.all(counts == total // experts)
        assert all(len(set(row)) == topk for row in rows)
    gen = make_generator(seed, device="cpu")
    weights = torch.rand((tokens, topk), generator=gen, device="cpu") + 0.125
    weights /= weights.sum(dim=1, keepdim=True)
    return ids, weights, counts


def _dequant(data, scale, intype):
    values = fp4_utils.mxfp4_to_f32(data) if intype == "a4w4" else data.float()
    return values * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(
        32, dim=-1
    )


def _reference(p, activation=None):
    """Golden GEMMs use unshuffled operands; dequantize one expert at a time."""
    t, topk, n, k = p["tokens"], p["topk"], p["N"], p["K"]
    products = torch.zeros((t, topk, n), dtype=torch.float32, device="cuda")
    analytic = p["data_init"] == "zero" or (
        p["data_init"] == "constant" and p["scale_init"] in ("constant", "zero")
    )
    if activation is None and analytic:
        exponent = (
            max(
                0,
                min(
                    254, (127 if p["scale_init"] == "constant" else 0) + p["scale_bias"]
                ),
            )
            - 127
        )
        value = 0.0 if p["data_init"] == "zero" else k * math.ldexp(0.5, exponent) ** 2
        products.fill_(value)
    else:
        activation = (
            _dequant(p["A"], p["raw_as"], p["intype"])
            if activation is None
            else activation
        )
        for expert in range(p["experts"]):
            token, slot = (p["ids"] == expert).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            token, slot = token.cuda(), slot.cuda()
            a = activation[token] if p["stage"] == 1 else activation[token, slot]
            b = _dequant(p["raw_w"][expert], p["raw_ws"][expert], "a4w4")
            products[token, slot] = a @ b.T
    if p["stage"] == 1:
        gate, up = products.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    contributions = (products * p["weights"].cuda()[..., None]).to(dtypes.bf16).float()
    # Compare against the exact sum of BF16 contributions, allowing the
    # mathematically bounded rounding of any BF16 atomic accumulation order.
    return contributions.sum(dim=1), contributions.abs().sum(dim=1)


@functools.lru_cache(maxsize=1)
def _prepare(
    stage,
    m,
    n,
    k,
    intype,
    experts,
    topk,
    tokens,
    expert_init,
    data_init,
    scale_init,
    scale_bias,
    seed,
    check,
):
    tokens = m * experts // topk if tokens is None else tokens
    ids, weights, counts = _routing(tokens, experts, topk, expert_init, seed)
    gen = make_generator(seed)
    a_rows = tokens if stage == 1 else tokens * topk
    if intype == "a4w4":
        a = fill_fp4((a_rows, k), data_init, gen, constant=0x11).view(dtypes.fp4x2)
    else:
        a = fill_fp8((a_rows, k), data_init, gen, constant=0.5)
    w = (
        fill_fp4((experts * n, k), data_init, gen, constant=0x11)
        .reshape(experts, n, k // 2)
        .view(dtypes.fp4x2)
    )
    a_scale = fill_scale_e8m0((a_rows, k // 32), scale_init, gen)
    w_scale = fill_scale_e8m0((experts, n, k // 32), scale_init, gen)
    if scale_bias:
        a_scale = (a_scale.to(torch.int16) + scale_bias).clamp(0, 254).to(torch.uint8)
        w_scale = (w_scale.to(torch.int16) + scale_bias).clamp(0, 254).to(torch.uint8)
    if stage == 2:
        a = a.reshape(tokens, topk, k // 2 if intype == "a4w4" else k)
        a_scale = a_scale.reshape(tokens, topk, k // 32)
    packed_w, packed_ws = shuffle_moe_weights_gfx1250(w, w_scale, gate_up=stage == 1)
    p = {
        "stage": stage,
        "M": m,
        "N": n,
        "K": k,
        "intype": intype,
        "experts": experts,
        "topk": topk,
        "tokens": tokens,
        "expert_init": expert_init,
        "data_init": data_init,
        "scale_init": scale_init,
        "scale_bias": scale_bias,
        "seed": seed,
        "A": a,
        "raw_as": a_scale,
        "raw_w": w,
        "raw_ws": w_scale,
        "W": packed_w,
        "WS": packed_ws,
        "ids": ids,
        "weights": weights,
        "counts": counts,
        "layouts": {},
    }
    p["routing_hash"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    p["reference"] = _reference(p) if check else None
    return p


def _layout(p, tile_m):
    if tile_m in p["layouts"]:
        return p["layouts"][tile_m]
    t, topk, e = p["tokens"], p["topk"], p["experts"]
    capacity = max(0, t * topk + e * tile_m - topk)
    sid = torch.full((capacity,), t | (topk << 24), dtype=torch.int32, device="cpu")
    eid = torch.zeros(
        (capacity + tile_m - 1) // tile_m, dtype=torch.int32, device="cpu"
    )
    weights = torch.zeros(capacity, dtype=torch.float32, device="cpu")
    cursor = 0
    for expert in range(e):
        pos = (p["ids"] == expert).nonzero()
        count = pos.shape[0]
        padded = (count + tile_m - 1) // tile_m * tile_m
        sid[cursor : cursor + count] = pos[:, 0] | (pos[:, 1] << 24)
        weights[cursor : cursor + count] = p["weights"][pos[:, 0], pos[:, 1]]
        eid[cursor // tile_m : (cursor + padded) // tile_m] = expert
        cursor += padded
    rows = ((sid & 0xFFFFFF) < t).nonzero().flatten()
    logical = sid[rows].long()
    sid_gpu = sid.cuda()
    layout = {
        "sid": sid_gpu,
        "eid": eid.cuda(),
        "weights": weights.cuda(),
        "valid": torch.tensor([cursor, t], dtype=torch.int32, device="cuda"),
        "scale": shuffle_moe_input_scale_gfx1250(p["raw_as"], sid_gpu),
        "post_pad": cursor,
        "row": rows.cuda(),
        "token": (logical & 0xFFFFFF).cuda(),
        "slot": ((logical >> 24) & 0xFF).cuda(),
    }
    p["layouts"][tile_m] = layout
    return layout


def _dispatch(
    a,
    w,
    sa,
    sw,
    sid,
    eid,
    valid,
    weights,
    out,
    out_scale,
    *,
    stage,
    topk,
    tile_m,
    outtype,
    knl_name,
):
    if stage == 1:
        return moe_stage1_asm(
            a,
            w,
            sa,
            sw,
            sid,
            eid,
            valid,
            topk,
            tile_m,
            _OUT_DTYPE[outtype],
            knl_name,
            y=out,
            out_scale=out_scale,
        )
    return moe_stage2_asm(
        a, w, sa, sw, sid, eid, valid, weights, tile_m, knl_name, y=out
    )


def _runner(p, tile_m, outtype, knl_name=""):
    layout = _layout(p, tile_m)
    n_out = p["N"] // 2 if p["stage"] == 1 else p["N"]
    shape = (
        (p["tokens"], p["topk"], n_out // 2 if outtype == "mxfp4" else n_out)
        if p["stage"] == 1
        else (p["tokens"], n_out)
    )
    out = torch.empty(shape, dtype=_OUT_DTYPE[outtype], device="cuda")
    scale = None
    if outtype == "mxfp4":
        rows = (layout["sid"].numel() + 31) // 32 * 32
        scale = torch.empty((rows, n_out // 32), dtype=dtypes.fp8_e8m0, device="cuda")
    fn = functools.partial(
        _dispatch,
        stage=p["stage"],
        topk=p["topk"],
        tile_m=tile_m,
        outtype=outtype,
        knl_name=knl_name,
    )
    fn.__name__ = f"moe_stage{p['stage']}_{p['intype']}_{outtype}_M{p['M']}_N{p['N']}_K{p['K']}_BM{tile_m}"
    args = (
        p["A"],
        p["W"],
        layout["scale"],
        p["WS"],
        layout["sid"],
        layout["eid"],
        layout["valid"],
        layout["weights"],
        out,
        scale,
    )
    return fn, args, layout


def _fq_reference(values):
    groups = values.reshape(*values.shape[:-1], -1, 32)
    amax = groups.abs().amax(dim=-1)
    exponent = (((amax.view(torch.int32).long() + 0x400000) & 0xFF800000) >> 23) - 2
    scales = exponent.clamp(0, 255).to(torch.uint8)
    scale = torch.exp2(scales.float() - 127)
    normalized = (groups / scale[..., None]).abs().clamp(max=6)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    best = torch.zeros_like(normalized, dtype=torch.long)
    distance = normalized.abs()
    for index in range(1, 8):
        delta = (normalized - table[index]).abs()
        use = (delta < distance) | ((delta == distance) & (index % 2 == 0))
        best = torch.where(use, index, best)
        distance = torch.minimum(distance, delta)
    decoded = table[best] * scale[..., None] * torch.where(groups < 0, -1, 1)
    return decoded.reshape(values.shape), scales


def _decode_fq(result, layout):
    data, scale = result
    t, topk, packed_n = data.shape
    cols = packed_n * 2 // 32
    raw = scale.view(torch.uint8)
    unpacked = (
        raw.reshape(raw.shape[0] // 32, cols // 4, 32, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, cols)
    )
    logical = torch.full((t, topk, cols), 127, dtype=torch.uint8, device=data.device)
    logical[layout["token"], layout["slot"]] = unpacked[layout["row"]]
    return _dequant(data, logical, "a4w4"), logical


def _accuracy(p, outtype, result, layout, reference=None):
    reference = p["reference"] if reference is None else reference
    for tensor in reference if isinstance(reference, tuple) else (reference,):
        assert torch.isfinite(tensor).all(), "nonfinite reference"
    if p["stage"] == 2:
        expected, sum_abs = reference
        actual = result.float()
        unit_roundoff = 2**-8
        gamma = (p["topk"] - 1) * unit_roundoff / (1 - (p["topk"] - 1) * unit_roundoff)
        # Include one BF16 step in the expert contributions themselves when
        # CPU/GPU GEMM rounding lands on opposite sides of a BF16 boundary.
        tolerance = (gamma + 2 * unit_roundoff) * sum_abs + 0.004
        mismatch = (actual - expected).abs() > tolerance
    elif outtype == "mxfp4":
        actual, scales = _decode_fq(result, layout)
        expected, expected_scales = _fq_reference(reference)
        scale_err = (scales != expected_scales).float().mean().item()
        nrms = (
            (actual - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp(min=1e-20)
        ).item()
        assert torch.isfinite(actual).all(), "nonfinite FQ output"
        assert (
            nrms < 0.035 and scale_err < 0.002
        ), f"FQ nrms={nrms}, scale mismatch={scale_err}"
        return {"asm err": scale_err, "asm nrms": nrms}
    else:
        actual, expected = result.float(), reference.to(dtypes.bf16).float()
        mismatch = (actual - expected).abs() > (0.004 + 0.02 * expected.abs())
    assert torch.isfinite(actual).all(), "nonfinite output"
    err = mismatch.float().mean().item() if actual.numel() else 0.0
    nrms = (
        (
            (actual - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp(min=1e-20)
        ).item()
        if actual.numel()
        else 0.0
    )
    assert (
        err == 0
    ), f"mismatch={err:.6%}, nrms={nrms:.6g}, max_abs={(actual-expected).abs().max().item():.6g}"
    return {"asm err": err, "asm nrms": nrms}


_OBSERVED = {}


def _observed_config(p, tile_m, outtype, fn, args):
    key = (p["stage"], p["intype"], outtype, tile_m, p["N"], p["K"])
    if key not in _OBSERVED:
        fn(*args)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            fn(*args)
            torch.cuda.synchronize()
        names = {
            event.name.removeprefix("aiter::")
            for event in prof.events()
            if str(event.device_type).endswith("CUDA")
            and f"moe_stage{p['stage']}_" in event.name
        }
        matches = [
            cfg
            for cfg in _configs(p["stage"])
            if cfg["co_name"].removesuffix(".co") in names or cfg["knl_name"] in names
        ]
        assert len(matches) == 1, f"expected one MoE ASM kernel in trace, got {names}"
        _OBSERVED[key] = matches[0]
    return _OBSERVED[key]


def _event_time(fn, args, warmup, iters, graph):
    fn(*args)  # module load is outside the requested warmup/timed region
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            for _ in range(iters):
                result = fn(*args)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    if graph:
        capture.replay()
    else:
        for _ in range(iters):
            result = fn(*args)
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end) * 1000 / iters


def test_moe(p, tile_m, outtype, cfg, args, round_id):
    fn, tensors, layout = _runner(p, tile_m, outtype, cfg["knl_name"] if cfg else "")
    actual_cfg = cfg or _observed_config(p, tile_m, outtype, fn, tensors)
    row = {
        key: p[key]
        for key in (
            "stage",
            "M",
            "N",
            "K",
            "tokens",
            "experts",
            "topk",
            "intype",
            "data_init",
            "scale_init",
            "scale_bias",
            "expert_init",
            "seed",
            "routing_hash",
        )
    }
    row.update(
        gfx=get_gfx(),
        outtype=outtype,
        mode=args.mode,
        timer=args.timer,
        num_warmup=args.warmup,
        num_iters=args.iters,
        num_rotate=args.rotate,
        test_graph=args.graph,
        check=not args.no_check,
        round=round_id,
        tile_m=tile_m,
        tile_n=actual_cfg["tile_n"],
        tile_k=actual_cfg["tile_k"],
        pf_tiles=actual_cfg["pf_tiles"],
        cluster=f"{actual_cfg['cluster_x']}x{actual_cfg['cluster_y']}",
        knl_name=actual_cfg["knl_name"],
        post_pad=layout["post_pad"],
        padding=layout["post_pad"] - p["tokens"] * p["topk"],
        valid_min=int(p["counts"].min()),
        valid_max=int(p["counts"].max()),
        work_units=(p["N"] // actual_cfg["tile_n"]) * (layout["post_pad"] // tile_m),
        timing_scope="gpu_op_including_output_clear",
    )
    aiter.logger.info(
        "stage%s %s/%s M=%s N=%s K=%s tokens=%s E=%s topk=%s tile=%sx%s cluster=%s init=%s/%s round=%s",
        p["stage"],
        p["intype"],
        outtype,
        p["M"],
        p["N"],
        p["K"],
        p["tokens"],
        p["experts"],
        p["topk"],
        tile_m,
        actual_cfg["tile_n"],
        row["cluster"],
        p["data_init"],
        p["scale_init"],
        round_id,
    )
    if args.mode == "func":
        result, us = fn(*tensors), float("nan")
    elif args.timer == "events":
        result, us = _event_time(fn, tensors, args.warmup, args.iters, args.graph)
    else:
        if args.mode == "profile":
            fn = torch.profiler.record_function(fn.__name__)(fn)
        result, us = run_perftest(
            fn,
            *tensors,
            num_iters=args.iters,
            num_warmup=args.warmup,
            testGraph=args.graph,
            num_rotate_args=args.rotate,
            needTrace=args.mode == "profile",
        )
    active = int((p["counts"] > 0).sum())
    a_rows = p["tokens"] * (p["topk"] if p["stage"] == 2 else 1)
    input_bytes = active * p["N"] * (p["K"] // 2 + p["K"] // 32) + a_rows * (
        p["K"] // (2 if p["intype"] == "a4w4" else 1) + p["K"] // 32
    )
    row.update(
        {
            "input_bytes": input_bytes,
            "asm us": us,
            "asm TFLOPS": 2 * p["tokens"] * p["topk"] * p["N"] * p["K"] / us / 1e6,
            "asm TB/s": input_bytes / us / 1e6,
            "asm result": "not checked" if args.no_check else "pass",
        }
    )
    if args.mode == "profile":
        row["asm trace"] = f"./aiter_logs/gpu_id_{torch.cuda.current_device()}"
    if not args.no_check:
        try:
            row.update(_accuracy(p, outtype, result, layout))
        except AssertionError as exc:
            row.update({"asm result": "failed", "accuracy_error": str(exc)})
            aiter.logger.error("MoE accuracy failed: %s", exc)
    return row


def _expect_error(fn, message, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (RuntimeError, ValueError) as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected an error containing {message!r}")


def _check_api_contracts(stages):
    """Small integration regressions run by --mode func, outside perf/profile."""
    for stage in stages:
        p = _prepare(
            stage,
            2,
            3072 if stage == 1 else 1024,
            1024,
            "a8w4",
            4,
            1,
            None,
            2,
            "uniform",
            "auto",
            -5,
            7,
            True,
        )
        fn, tensors, layout = _runner(p, 16, "bf16")
        expected = fn(*tensors).clone()
        _accuracy(p, "bf16", expected, layout)
        tensors[-2].fill_(123)
        torch.testing.assert_close(fn(*tensors), expected, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(*tensors)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*tensors)
        for _ in range(2):
            tensors[-2].fill_(321)
            graph.replay()
            torch.testing.assert_close(tensors[-2], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.compile(fn, backend="eager", fullgraph=True)(*tensors),
            expected,
            rtol=0,
            atol=0,
        )
        saved = tensors[6].clone()
        tensors[6].zero_()
        graph.replay()
        assert torch.count_nonzero(tensors[-2]).item() == 0
        tensors[6].copy_(saved)
        for index, value, error in (
            (2, tensors[2][:16], "a_scale must have shape"),
            (4, tensors[4].long(), "must be int32"),
            (5, tensors[5][:1], "one expert per sorting block"),
        ):
            altered = list(tensors)
            altered[index] = value
            _expect_error(fn, error, *altered)
        _expect_error(fn, "block_m must be", *tensors, tile_m=17)
        _expect_error(fn, "unregistered", *tensors, knl_name="missing_kernel")
        wrong = next(c for c in _configs(stage) if c["a_intype"] == "mxfp4")
        _expect_error(fn, "does not match", *tensors, knl_name=wrong["knl_name"])
        raw = list(tensors)
        raw[-2] = torch.empty_like(tensors[-2], dtype=torch.float32)
        if stage == 1:
            _expect_error(
                _moe_stage1_asm,
                "output must be",
                *raw[:7],
                raw[8],
                None,
                1,
                16,
                None,
            )
        else:
            _expect_error(_moe_stage2_asm, "output must be", *raw[:9], 16, None)
        empty = _prepare(
            stage,
            0,
            3072 if stage == 1 else 1024,
            1024,
            "a8w4",
            4,
            1,
            0,
            2,
            "uniform",
            "auto",
            -5,
            7,
            False,
        )
        empty_fn, empty_args, _ = _runner(empty, 16, "bf16")
        assert empty_fn(*empty_args).numel() == 0
        # A4 byte carriers and typed E8M0 scales must retain the same ABI.
        byte_p = _prepare(
            stage,
            1,
            3072 if stage == 1 else 1024,
            1024,
            "a4w4",
            3,
            1,
            None,
            2,
            "uniform",
            "auto",
            -5,
            9,
            True,
        )
        byte_fn, byte_args, byte_layout = _runner(byte_p, 16, "bf16")
        byte_args = list(byte_args)
        byte_args[0] = byte_args[0].view(torch.uint8)
        byte_args[1] = byte_args[1].view(torch.uint8)
        byte_args[2] = byte_args[2].view(dtypes.fp8_e8m0)
        byte_args[3] = byte_args[3].view(dtypes.fp8_e8m0)
        assert byte_layout["post_pad"] == 48 and byte_args[2].shape[0] % 32 == 0
        _accuracy(byte_p, "bf16", byte_fn(*byte_args), byte_layout)

    if 1 in stages:
        from aiter.fused_moe import moe_sorting

        p = _prepare(
            1, 2, 3072, 1024, "a8w4", 4, 2, None, 2, "uniform", "auto", -5, 0, True
        )
        fn, tensors, layout = _runner(p, 16, "bf16")
        sid, weights, eid, valid, _ = moe_sorting(
            p["ids"].cuda(), p["weights"].cuda(), 4, 1536, dtypes.bf16, 16
        )
        actual = list(tensors)
        actual[2] = shuffle_moe_input_scale_gfx1250(p["raw_as"], sid)
        actual[4:8] = [sid, eid, valid, weights]
        _accuracy(p, "bf16", fn(*actual), layout)
        fq_fn, fq_args, fq_layout = _runner(p, 16, "mxfp4")
        fq = fq_fn(*fq_args)
        _accuracy(p, "mxfp4", fq, fq_layout)
        _accuracy(
            p,
            "mxfp4",
            torch.compile(fq_fn, backend="eager", fullgraph=True)(*fq_args),
            fq_layout,
        )
        if 2 in stages:
            p2 = _prepare(
                2, 2, 1024, 1536, "a4w4", 4, 2, None, 2, "uniform", "auto", -5, 0, True
            )
            fn2, args2, layout2 = _runner(p2, 16, "bf16")
            args2 = list(args2)
            args2[0], args2[2] = fq
            reference = _reference(p2, _decode_fq(fq, fq_layout)[0])
            _accuracy(p2, "bf16", fn2(*args2), layout2, reference)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fq_fn(*fq_args)
                fn2(*args2)
            graph.replay()
            _accuracy(p2, "bf16", args2[-2], layout2, reference)
    aiter.logger.info(
        "MoE API checks passed for stages %s (reuse/stream/graph/compile/routing/errors/empty/byte carriers/FQ chain)",
        stages,
    )


def _plans(args, stage, intype, outtype, n, k):
    configs = [
        c
        for c in _configs(stage)
        if c["a_intype"] == _INTYPE[intype] and c["outtype"] == outtype
    ]
    if args.knl_name not in (None, "auto", "all"):
        cfg = next((c for c in _configs(stage) if c["knl_name"] == args.knl_name), None)
        if cfg is None:
            raise ValueError(f"unregistered stage{stage} --knl-name {args.knl_name}")
        if (
            cfg not in configs
            or not _supports(cfg, n, k)
            or (args.tile_m is not None and cfg["tile_m"] not in args.tile_m)
            or (args.tile_n is not None and cfg["tile_n"] not in args.tile_n)
            or (
                args.cluster is not None
                and f"{cfg['cluster_x']}x{cfg['cluster_y']}" not in args.cluster
            )
        ):
            raise ValueError(
                "explicit --knl-name is incompatible with the requested case"
            )
        return [(cfg["tile_m"], cfg)]
    configs = [
        c
        for c in configs
        if _supports(c, n, k)
        and (args.tile_m is None or c["tile_m"] in args.tile_m)
        and (args.tile_n is None or c["tile_n"] in args.tile_n)
        and (
            args.cluster is None or f"{c['cluster_x']}x{c['cluster_y']}" in args.cluster
        )
    ]
    if args.knl_name == "all" or (args.mode == "func" and args.knl_name is None):
        return [(c["tile_m"], c) for c in configs]
    result = []
    for tile_m in args.tile_m or [16, 32, 64]:
        candidates = [c for c in configs if c["tile_m"] == tile_m]
        if candidates:
            if args.tile_n is not None or args.cluster is not None:
                for tile_n, cx in sorted(
                    {(c["tile_n"], c["cluster_x"]) for c in candidates}
                ):
                    selected = min(
                        (
                            c
                            for c in candidates
                            if c["tile_n"] == tile_n and c["cluster_x"] == cx
                        ),
                        key=lambda c: (c["priority"], c["knl_name"]),
                    )
                    result.append((tile_m, selected))
            else:
                result.append((tile_m, None))  # observe the native heuristic
    return result


def _parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("func", "perf", "profile"),
        default="perf",
        help="func=accuracy+API cover; perf=accuracy+timing; profile=perf+trace",
    )
    p.add_argument("--stage", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    p.add_argument(
        "--intype", nargs="+", choices=tuple(_INTYPE), default=["a4w4", "a8w4"]
    )
    p.add_argument(
        "--outtype",
        nargs="+",
        choices=tuple(_OUT_DTYPE),
        default=None,
        help="default: bf16 for perf/profile; all supported formats for func",
    )
    p.add_argument(
        "-s",
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="+",
        help="M,N,K tuples; M=valid routes/expert, stage1 N=gate+up, stage2 N=hidden",
    )
    p.add_argument(
        "--experts",
        type=int,
        default=None,
        help="default: 96 for perf/profile, 4 for func",
    )
    p.add_argument(
        "--topk", type=int, default=None, help="default: 6 for perf/profile, 2 for func"
    )
    p.add_argument(
        "--tokens",
        type=int,
        help="optional assertion of input T; must equal M*experts/topk",
    )
    p.add_argument(
        "--tile-m",
        type=int,
        nargs="+",
        choices=(16, 32, 64),
        help="sorting/kernel tile sweep, independent of M; default: 16 32 64",
    )
    p.add_argument(
        "--tile-n",
        type=int,
        nargs="+",
        choices=(384, 512),
        help="optional manifest filter; unset lets native dispatch choose N tile",
    )
    p.add_argument(
        "--cluster", nargs="+", choices=("1x1", "4x1"), help="optional cluster filter"
    )
    p.add_argument(
        "--knl-name",
        help="auto=native heuristic; all=all compatible manifest kernels (func default); otherwise exact symbol",
    )
    p.add_argument(
        "--expert-init",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help="PoC routing: 0=round-robin, 1=random, 2=balanced shuffled, 3=balanced contiguous",
    )
    p.add_argument(
        "--data-init",
        nargs="+",
        choices=DATA_DISTS,
        help="paired with --scale-init; defaults: constant uniform for perf/profile, uniform for func",
    )
    p.add_argument(
        "--scale-init",
        nargs="+",
        choices=E8M0_SCALE_DISTS,
        help="paired with --data-init; defaults: constant auto for perf/profile, auto for func",
    )
    p.add_argument(
        "--scale-bias",
        type=int,
        default=-5,
        help="E8M0 exponent bias (default -5, MoE calibration); constant/-5 reproduces PoC init=10; 0 uses unshifted GEMM scales",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--iters",
        type=int,
        help="timed iterations: 101 by default (func performs accuracy checks without timing)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="repeat each shape, reversing tile order every other round",
    )
    p.add_argument(
        "--timer",
        choices=("profiler", "events"),
        default="profiler",
        help="profiler=shared GEMM UT timer; events=one event interval over --iters calls (PoC-style)",
    )
    p.add_argument("--graph", action="store_true", help="time HIP graph replay")
    p.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="run_perftest input rotation count; 0=auto, 1=disabled; events requires 1",
    )
    p.add_argument(
        "--no-check",
        action="store_true",
        help="skip accuracy for perf/profile (PoC vld=0); forbidden in func",
    )
    p.add_argument(
        "--json", dest="json_out", help="write the full summary as JSON records"
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="print the resolved case matrix without allocating inputs or launching kernels",
    )
    return p


def main():
    parser = _parser()
    args = parser.parse_args()
    args.experts = (
        args.experts if args.experts is not None else (4 if args.mode == "func" else 96)
    )
    args.topk = (
        args.topk if args.topk is not None else (2 if args.mode == "func" else 6)
    )
    args.iters = args.iters if args.iters is not None else 101
    if not 0 < args.topk <= args.experts or args.topk >= 256:
        parser.error("require 0 < topk <= experts and topk < 256")
    if (
        args.iters < (2 if args.timer == "profiler" and args.mode != "func" else 1)
        or args.warmup < 0
        or args.rounds < 1
        or args.rotate < 0
    ):
        parser.error("invalid timing counts; profiler needs iters>=2")
    if args.mode == "func" and args.no_check:
        parser.error("func cover requires accuracy checks")
    if args.timer == "events" and (args.rotate != 1 or args.mode == "profile"):
        parser.error("events requires --rotate 1; profile uses --timer profiler")
    if (
        get_gfx() not in SUPPORTED_GFX
        or not torch.cuda.is_available()
        or not is_gfx1250_asm_supported()
    ):
        parser.error("MoE ASM requires an available gfx1250 B0+ GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_printoptions(sci_mode=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    di = args.data_init or (
        ["uniform"] if args.mode == "func" else ["constant", "uniform"]
    )
    si = args.scale_init or (["auto"] if args.mode == "func" else ["constant", "auto"])
    if len(di) == 1:
        di *= len(si)
    if len(si) == 1:
        si *= len(di)
    if len(di) != len(si):
        parser.error("data-init/scale-init lengths must match (length 1 broadcasts)")
    outtypes = args.outtype or (["bf16", "mxfp4"] if args.mode == "func" else ["bf16"])
    jobs = []
    for stage, intype, (data_init, scale_init) in itertools.product(
        args.stage, args.intype, zip(di, si)
    ):
        shapes = args.shape or (
            FUNC_SHAPES[stage] if args.mode == "func" else PERF_SHAPES[stage]
        )
        for shape in shapes:
            if len(shape) != 3 or min(shape) <= 0:
                parser.error("each shape must contain three positive integers M,N,K")
            m, n, k = shape
            if m * args.experts % args.topk:
                parser.error(
                    "M*experts must be divisible by topk to derive input tokens"
                )
            tokens = m * args.experts // args.topk
            if tokens >= 2**24 or (args.tokens is not None and args.tokens != tokens):
                parser.error(
                    "tokens must equal M*experts/topk and fit in 24 bits; tile-m never changes tokens"
                )
            if n % 32 or k % 128:
                parser.error("weight packing requires N%32==0 and K%128==0")
            for outtype in outtypes:
                if outtype == "mxfp4" and (stage != 1 or intype != "a8w4" or n % 256):
                    continue
                for tile_m, cfg in _plans(args, stage, intype, outtype, n, k):
                    jobs.append(
                        (
                            stage,
                            m,
                            n,
                            k,
                            intype,
                            outtype,
                            tile_m,
                            cfg,
                            data_init,
                            scale_init,
                        )
                    )
        if args.mode == "func" and args.shape is None:
            # Exercise the six short-prefetch families at their true minimum K
            # (both cluster variants), rather than only the common K1024 case.
            for cfg in _configs(stage):
                k = cfg["tile_k"] * cfg["pf_tiles"]
                if (
                    k >= 1024
                    or cfg["a_intype"] != _INTYPE[intype]
                    or cfg["outtype"] not in outtypes
                ):
                    continue
                if (cfg["tile_m"], cfg) in _plans(
                    args, stage, intype, cfg["outtype"], 6144, k
                ):
                    jobs.append(
                        (
                            stage,
                            1,
                            6144,
                            k,
                            intype,
                            cfg["outtype"],
                            cfg["tile_m"],
                            cfg,
                            data_init,
                            scale_init,
                        )
                    )
    if not jobs:
        parser.error("no compatible MoE ASM cases selected")
    aiter.logger.info(
        "MoE ASM %s: %s cases/round; M=valid routes/expert, tokens=M*E/topk. Timing includes output clear; BW models input bytes.",
        args.mode,
        len(jobs),
    )
    if args.list_cases:
        records = [
            {
                "stage": s,
                "M": m,
                "N": n,
                "K": k,
                "intype": it,
                "outtype": ot,
                "tile_m": bm,
                "tokens": m * args.experts // args.topk,
                "experts": args.experts,
                "topk": args.topk,
                "data_init": di,
                "scale_init": si,
                "knl_name": c["knl_name"] if c else "auto",
            }
            for s, m, n, k, it, ot, bm, c, di, si in jobs
        ]
        print(pd.DataFrame(records).to_markdown(index=False))
        if args.json_out:
            pd.DataFrame(records).to_json(args.json_out, orient="records", indent=2)
        return
    if args.mode == "func":
        _check_api_contracts(args.stage)
    # Group tile/output variants so they share identical data and routing.
    groups = {}
    for stage, m, n, k, intype, outtype, tile_m, cfg, data_init, scale_init in jobs:
        key = (stage, m, n, k, intype, data_init, scale_init)
        groups.setdefault(key, []).append((tile_m, outtype, cfg))
    rows = []
    for (stage, m, n, k, intype, data_init, scale_init), candidates in groups.items():
        p = _prepare(
            stage,
            m,
            n,
            k,
            intype,
            args.experts,
            args.topk,
            args.tokens,
            args.expert_init,
            data_init,
            scale_init,
            args.scale_bias,
            args.seed,
            not args.no_check,
        )
        for r in range(args.rounds):
            for tile_m, outtype, cfg in (
                candidates if r % 2 == 0 else candidates[::-1]
            ):
                rows.append(test_moe(p, tile_m, outtype, cfg, args, r + 1))
    table = pd.DataFrame(rows)
    if args.json_out:
        table.to_json(args.json_out, orient="records", indent=2)
        aiter.logger.info("wrote %s records to %s", len(rows), args.json_out)
    columns = [
        "stage",
        "intype",
        "outtype",
        "M",
        "N",
        "K",
        "tokens",
        "experts",
        "topk",
        "tile_m",
        "tile_n",
        "cluster",
        "data_init",
        "scale_init",
        "round",
        "post_pad",
        "asm us",
        "asm TFLOPS",
        "asm TB/s",
        "asm err",
        "asm result",
    ]
    aiter.logger.info(
        "MoE ASM summary:\n%s",
        table[[c for c in columns if c in table]].to_markdown(index=False),
    )
    if args.rounds > 1:
        keys = [
            "stage",
            "intype",
            "outtype",
            "M",
            "N",
            "K",
            "tokens",
            "tile_m",
            "tile_n",
            "cluster",
            "data_init",
            "scale_init",
        ]
        medians = (
            table.groupby(keys, dropna=False)[["asm us", "asm TB/s"]]
            .median()
            .reset_index()
        )
        aiter.logger.info("Across-round medians:\n%s", medians.to_markdown(index=False))
    if any(row["asm result"] == "failed" for row in rows):
        raise SystemExit("MoE accuracy failed; see summary/JSON")
    if args.mode == "profile":
        aiter.logger.info(
            "profiler traces written under ./aiter_logs/gpu_id_%s",
            torch.cuda.current_device(),
        )
    aiter.logger.info(
        "MoE ASM completed: %s cases, %s",
        len(rows),
        "accuracy skipped" if args.no_check else "all passed",
    )


if __name__ == "__main__":
    main()
