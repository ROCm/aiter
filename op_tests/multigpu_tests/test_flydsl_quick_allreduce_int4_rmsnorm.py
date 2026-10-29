# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for ``QuickAllReduceInt4.allreduce_rmsnorm`` and
``allreduce_rmsnorm_mxfp4``.

Every rank feeds its own input plus a shared residual and weight. Checks per
case:

* against an fp32 NCCL all-reduce followed by fp32 add + RMSNorm (SQNR, the
  INT4 codec is lossy);
* against unfused ``QuickAllReduceInt4.allreduce`` + fp32 add + RMSNorm
  (tight: the only difference is the bf16 rounding of the reduced sum that
  the fused kernel skips);
* bitwise equality of ``out`` / ``residual_out`` across ranks;
* where the fused RMSNorm has its own ladder, the same checks again on the
  mesh ladder, so the fused builds of its super-tiles stay covered;
* the first fused call inside a HIP graph capture, with no eager call
  before it: every fused binary of a hidden declared in ``rmsnorm_hidden`` is
  captured in one graph, at an epsilon other than the default;
* on gfx950, ``allreduce_rmsnorm_mxfp4``: the same ``out`` / ``residual_out``
  bit for bit, and fp4 bytes plus shuffled e8m0 scales equal to
  ``per_1x32_f4_quant_hip(out, shuffle=True)``.

The sweep times fused vs unfused (FlyDSL all-reduce + AITER
``rmsnorm2d_fwd_with_add``, then ``per_1x32_f4_quant_hip`` for MXFP4).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

import aiter
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import checkAllclose, run_perftest

set_start_method("spawn", force=True)

from aiter.ops.flydsl.allreduce_policy import (
    mesh_norm_ladder,
    mesh_st_ladder,
    resolve_link,
)
from aiter.ops.flydsl.kernels.comm.allreduce_shared import WORLD
from aiter.ops.flydsl.kernels.comm.quick_allreduce_int4 import mxfp4_scale_shape

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")
EPS = 1e-6
SQNR_MIN_DB = 18.0
# vs unfused FlyDSL: differs only by bf16 rounding of the reduced sum.
TIGHT_RTOL = 2e-2
TIGHT_ATOL = 2e-2
TIGHT_ERR_RATIO = 0.01
# (tokens, hidden, gemma_norm, inplace_residual). tokens=1 at hidden=2048 is a
# single partial tile; 37 rows leave a partial last tile at every hidden. Up to
# 512 tiles (16 MiB) on a 256-CU part run the ST=1 latency build; 1536x8192 and
# 3072x4096 (768 tiles) the occupancy-pinned ST=1 build; larger ones ST=8 where
# the ladder has it.
_VALIDITY_CASES = [
    (1, 2048, False, False),
    (37, 2048, False, False),
    (37, 4096, True, False),
    (37, 8192, False, True),
    (37, 16384, True, True),
    (512, 4096, False, False),
    (1536, 8192, False, True),
    (3072, 4096, True, False),
    (4096, 8192, False, False),
    (4096, 8192, True, True),
    (16384, 8192, False, False),
]


def _sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    mse = ((got - ref) ** 2).mean()
    pow_ = (ref * ref).mean()
    if float(mse) == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(pow_ / mse))


def _ref_add_rmsnorm(ar, res, weight, gemma_norm, eps=EPS):
    z = ar.float() + res.float()
    w = weight.float() + 1.0 if gemma_norm else weight.float()
    rstd = torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (z * rstd * w).to(torch.bfloat16), z.to(torch.bfloat16)


def _live_scale_idx(rows, hidden, device):
    """Flat indices of the real rows in the shuffled e8m0 scale buffer."""
    scale_n = mxfp4_scale_shape(rows, hidden)[1]
    x = torch.arange(rows, device=device)[:, None]
    y = torch.arange(hidden // 32, device=device)[None, :]
    idx = (
        (x // 32) * scale_n * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )
    return idx.flatten()


def _mxfp4_buffers(rows, hidden, device):
    return (
        torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device),
        torch.empty(mxfp4_scale_shape(rows, hidden), dtype=torch.uint8, device=device),
    )


def _fused(fly, inp, res, weight, out, ar, res_t, *_, gemma_norm):
    fly.allreduce_rmsnorm(inp, res, weight, EPS, out, res_t, gemma_norm=gemma_norm)


def _unfused(fly, inp, res, weight, out, ar, res_t, *_, gemma_norm):
    fly.allreduce(inp, ar)
    aiter.rmsnorm2d_fwd_with_add(out, ar, res, res_t, weight, EPS, gemma_norm)


def _fused_mxfp4(fly, inp, res, weight, out, ar, res_t, q, q_s, *, gemma_norm):
    fly.allreduce_rmsnorm_mxfp4(
        inp, res, weight, EPS, out, res_t, q, q_s, gemma_norm=gemma_norm
    )


def _unfused_mxfp4(fly, inp, res, weight, out, ar, res_t, q, q_s, *, gemma_norm):
    _unfused(fly, inp, res, weight, out, ar, res_t, gemma_norm=gemma_norm)
    aiter.per_1x32_f4_quant_hip(out, shuffle=True)


def _run_rank(rank, tp, init_method, cases, time_it, norm_ladder):
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD
    fly = QuickAllReduceInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        min_bytes=0,
        norm_ladder=norm_ladder,
        rmsnorm_hidden=tuple(sorted({c[1] for c in cases})),
    )
    rows = []
    # The constructor must prepare every fused build the cases pick: a case
    # that adds one to the cache compiled it on its first call.
    warmed = set(fly._norm_launch)
    try:
        for ntok, hidden, gemma_norm, inplace in cases:
            gen = torch.Generator().manual_seed(1234 + rank)
            inp = (torch.randn((ntok, hidden), generator=gen) * 0.1).to(
                device=device, dtype=torch.bfloat16
            )
            shared = torch.Generator().manual_seed(99)
            res = torch.randn((ntok, hidden), generator=shared).to(
                device=device, dtype=torch.bfloat16
            )
            weight = (
                torch.randn((hidden,), generator=shared) * 0.2
                + (0.0 if gemma_norm else 1.0)
            ).to(device=device, dtype=torch.bfloat16)

            ref_ar = inp.float()
            dist.all_reduce(ref_ar, group=group)
            ref_out, ref_res = _ref_add_rmsnorm(ref_ar, res, weight, gemma_norm)

            ar = torch.empty_like(inp)
            fly.allreduce(inp, ar)
            unf_out, unf_res = _ref_add_rmsnorm(ar, res, weight, gemma_norm)

            out = torch.empty_like(inp)
            res_in = res.clone()
            res_out = res_in if inplace else torch.empty_like(inp)
            fly.allreduce_rmsnorm(
                inp, res_in, weight, EPS, out, res_out, gemma_norm=gemma_norm
            )
            torch.cuda.synchronize()

            gathered = [torch.empty_like(out) for _ in range(tp)]
            dist.all_gather(gathered, out, group=group)
            same_out = all(torch.equal(g, gathered[0]) for g in gathered)
            gathered = [torch.empty_like(res_out) for _ in range(tp)]
            dist.all_gather(gathered, res_out, group=group)
            same_res = all(torch.equal(g, gathered[0]) for g in gathered)

            # The unfused all-reduce takes the mesh ladder's pick and the fused
            # one its own ladder's. The block sets the tile and so the
            # quantization; when the two differ, only the SQNR check against
            # the fp32 reference is meaningful.
            nbytes = inp.numel() * inp.element_size()
            fused_cfg = fly._norm_cfg(nbytes, hidden)[0]
            same_cfg = fly._pick_cfg(nbytes)[0][1] == fused_cfg[1]

            tag = f"rank {rank} {ntok}x{hidden} gemma={gemma_norm}"
            row = {
                "tokens": ntok,
                "hidden": hidden,
                "gemma": gemma_norm,
                "inplace": inplace,
                "fused_cfg": fused_cfg,
                "sqnr_out_db": _sqnr_db(out.float(), ref_out.float()),
                "sqnr_res_db": _sqnr_db(res_out.float(), ref_res.float()),
                "err_out_vs_unfused": (
                    0.0
                    if not same_cfg
                    else float(
                        checkAllclose(
                            unf_out.float(),
                            out.float(),
                            rtol=TIGHT_RTOL,
                            atol=TIGHT_ATOL,
                            tol_err_ratio=TIGHT_ERR_RATIO,
                            printLog=False,
                            msg=f"{tag} out",
                        )
                    )
                ),
                "err_res_vs_unfused": (
                    0.0
                    if not same_cfg
                    else float(
                        checkAllclose(
                            unf_res.float(),
                            res_out.float(),
                            rtol=TIGHT_RTOL,
                            atol=TIGHT_ATOL,
                            tol_err_ratio=TIGHT_ERR_RATIO,
                            printLog=False,
                            msg=f"{tag} residual_out",
                        )
                    )
                ),
                "ranks_equal": same_out and same_res,
                "mxfp4_equal": None,
                "fused_us": None,
                "unfused_us": None,
                "mxfp4_us": None,
                "unfused_mxfp4_us": None,
            }
            if fly.supports_mxfp4:
                out_q = torch.empty_like(inp)
                res_in_q = res.clone()
                res_out_q = res_in_q if inplace else torch.empty_like(inp)
                q, q_s = _mxfp4_buffers(ntok, hidden, device)
                fly.allreduce_rmsnorm_mxfp4(
                    inp,
                    res_in_q,
                    weight,
                    EPS,
                    out_q,
                    res_out_q,
                    q,
                    q_s,
                    gemma_norm=gemma_norm,
                )
                ref_q, ref_s = aiter.per_1x32_f4_quant_hip(out_q, shuffle=True)
                live = _live_scale_idx(ntok, hidden, device)
                row["mxfp4_equal"] = (
                    torch.equal(out_q, out)
                    and torch.equal(res_out_q, res_out)
                    and torch.equal(q, ref_q.view(torch.uint8))
                    and torch.equal(
                        q_s.flatten()[live], ref_s.view(torch.uint8).flatten()[live]
                    )
                )
            if time_it:
                args = (fly, inp, res, weight, out, ar, torch.empty_like(inp))
                variants = {"fused_us": _fused, "unfused_us": _unfused}
                if fly.supports_mxfp4:
                    args += (q, q_s)
                    variants.update(
                        mxfp4_us=_fused_mxfp4, unfused_mxfp4_us=_unfused_mxfp4
                    )
                for key, fn in variants.items():
                    dist.barrier(group=group)
                    torch.cuda.synchronize()
                    # Bound, not passed: run_perftest deep-copies its args.
                    _, row[key] = run_perftest(
                        functools.partial(fn, *args, gemma_norm=gemma_norm),
                        use_cuda_event=True,
                    )
            rows.append(row)
            dist.barrier(group=group)
        if set(fly._norm_launch) != warmed:
            raise AssertionError("a case compiled a fused build rmsnorm_hidden missed")
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(tp, cases, *, time_it, norm_ladder=None):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = Pool(processes=tp)
    try:
        futs = [
            pool.apply_async(
                _run_rank,
                args=(rank, tp, init_method, cases, time_it, norm_ladder),
            )
            for rank in range(tp)
        ]
        ranks = [f.get(timeout=timeout) for f in futs]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return ranks

CAPTURE_HIDDEN = 4096
CAPTURE_TOKENS = (37, 3072)
CAPTURE_EPS = 0.5


def _capture_first_rank(rank, tp, init_method, norm_ladder):
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4
    from aiter.ops.flydsl.allreduce_host_shared import launcher_loaded

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD
    fly = QuickAllReduceInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        min_bytes=0,
        norm_ladder=norm_ladder,
        rmsnorm_hidden=(CAPTURE_HIDDEN,),
    )
    res = {"capture_error": None, "rows": []}
    try:
        built = set(fly._norm_launch)
        res["unloaded"] = sum(not launcher_loaded(fly._norm_launch[k]) for k in built)
        calls = []
        for ntok in CAPTURE_TOKENS:
            for gemma_norm in (False, True):
                for mxfp4 in (False, True) if fly.supports_mxfp4 else (False,):
                    gen = torch.Generator().manual_seed(1234 + rank + ntok)
                    inp = (torch.randn((ntok, CAPTURE_HIDDEN), generator=gen) * 0.1).to(
                        device=device, dtype=torch.bfloat16
                    )
                    shared = torch.Generator().manual_seed(99 + ntok)
                    resid = torch.randn((ntok, CAPTURE_HIDDEN), generator=shared).to(
                        device=device, dtype=torch.bfloat16
                    )
                    weight = (
                        torch.randn((CAPTURE_HIDDEN,), generator=shared) * 0.2
                        + (0.0 if gemma_norm else 1.0)
                    ).to(device=device, dtype=torch.bfloat16)
                    ref_ar = inp.float()
                    dist.all_reduce(ref_ar, group=group)
                    calls.append(
                        {
                            "tokens": ntok,
                            "gemma": gemma_norm,
                            "mxfp4": mxfp4,
                            "args": (inp, resid, weight),
                            "out": torch.empty_like(inp),
                            "res_out": torch.empty_like(inp),
                            "q": _mxfp4_buffers(ntok, CAPTURE_HIDDEN, device),
                            "ref": _ref_add_rmsnorm(
                                ref_ar, resid, weight, gemma_norm, CAPTURE_EPS
                            ),
                        }
                    )
        torch.cuda.synchronize()

        def _call(c):
            inp, resid, weight = c["args"]
            if c["mxfp4"]:
                fly.allreduce_rmsnorm_mxfp4(
                    inp,
                    resid,
                    weight,
                    CAPTURE_EPS,
                    c["out"],
                    c["res_out"],
                    *c["q"],
                    gemma_norm=c["gemma"],
                )
            else:
                fly.allreduce_rmsnorm(
                    inp,
                    resid,
                    weight,
                    CAPTURE_EPS,
                    c["out"],
                    c["res_out"],
                    gemma_norm=c["gemma"],
                )

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                for c in calls:
                    _call(c)
        except Exception as exc:  # noqa: BLE001 -- reported, and ends the test
            res["capture_error"] = f"{type(exc).__name__}: {exc}"
        errors = [None] * tp
        dist.all_gather_object(errors, res["capture_error"], group=gloo)
        # Replay only if every rank captured: the graph launches peers' kernels.
        if not any(errors):
            graph.replay()
            torch.cuda.synchronize()
            for c in calls:
                ref_out, ref_res = c["ref"]
                row = {
                    "tokens": c["tokens"],
                    "gemma": c["gemma"],
                    "mxfp4": c["mxfp4"],
                    "sqnr_out_db": _sqnr_db(c["out"].float(), ref_out.float()),
                    "sqnr_res_db": _sqnr_db(c["res_out"].float(), ref_res.float()),
                    "mxfp4_equal": None,
                }
                if c["mxfp4"]:
                    ref_q, ref_s = aiter.per_1x32_f4_quant_hip(c["out"], shuffle=True)
                    live = _live_scale_idx(c["tokens"], CAPTURE_HIDDEN, device)
                    q, q_s = c["q"]
                    row["mxfp4_equal"] = torch.equal(
                        q, ref_q.view(torch.uint8)
                    ) and torch.equal(
                        q_s.flatten()[live], ref_s.view(torch.uint8).flatten()[live]
                    )
                res["rows"].append(row)
        res["compiled_in_capture"] = len(set(fly._norm_launch) - built)

        # A width that was never prepared cannot be prepared under capture.
        inp = torch.zeros((1, 2 * CAPTURE_HIDDEN), dtype=torch.bfloat16, device=device)
        weight = torch.zeros(2 * CAPTURE_HIDDEN, dtype=torch.bfloat16, device=device)
        res["unprepared_error"] = None
        try:
            with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                fly.allreduce_rmsnorm(
                    inp, inp.clone(), weight, CAPTURE_EPS, inp.clone(), inp.clone()
                )
        except RuntimeError as exc:
            res["unprepared_error"] = str(exc)
        dist.barrier(group=gloo)
    finally:
        fly.close()
        dist.destroy_process_group()
    return res


def test_first_call_in_capture(tp, norm_ladder=None):
    """The first fused call is inside a graph capture."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        futs = [
            pool.apply_async(
                _capture_first_rank, args=(rank, tp, init_method, norm_ladder)
            )
            for rank in range(tp)
        ]
        ranks = [
            f.get(timeout=float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600")))
            for f in futs
        ]
    fails = []
    for rank, r in enumerate(ranks):
        where = f"rank {rank}"
        if r["capture_error"]:
            fails.append(f"{where}: capture failed: {r['capture_error']}")
        if r["unloaded"]:
            fails.append(f"{where}: {r['unloaded']} fused builds not loaded at init")
        if r["compiled_in_capture"]:
            fails.append(f"{where}: the capture built {r['compiled_in_capture']}")
        if not r["unprepared_error"] or "rmsnorm_hidden" not in r["unprepared_error"]:
            fails.append(f"{where}: unprepared hidden under capture did not raise")
        for row in r["rows"]:
            for key in ("sqnr_out_db", "sqnr_res_db"):
                if row[key] < SQNR_MIN_DB:
                    fails.append(f"{where} {row}: {key} < {SQNR_MIN_DB}")
            if row["mxfp4_equal"] is False:
                fails.append(f"{where} {row}: MXFP4 differs")
    if fails:
        raise AssertionError("; ".join(fails))
    return ranks[0]["rows"]


def _assert_valid(ranks, cases):
    fails = []
    for rank, rows in enumerate(ranks):
        for case, row in zip(cases, rows, strict=True):
            where = f"rank {rank} {case}"
            for key in ("sqnr_out_db", "sqnr_res_db"):
                if row[key] < SQNR_MIN_DB:
                    fails.append(f"{where}: {key} {row[key]:.2f} < {SQNR_MIN_DB}")
            for key in ("err_out_vs_unfused", "err_res_vs_unfused"):
                if row[key] >= TIGHT_ERR_RATIO:
                    fails.append(f"{where}: {key} {row[key]:.4f}")
            if not row["ranks_equal"]:
                fails.append(f"{where}: ranks disagree")
            if row["mxfp4_equal"] is False:
                fails.append(f"{where}: MXFP4 differs from per_1x32_f4_quant_hip")
    if fails:
        raise AssertionError("; ".join(fails))


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt4 unsupported on %s; skipping", ARCH)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=WORLD)
    parser.add_argument(
        "--bench",
        type=str,
        nargs="*",
        default=["2048,8192", "8192,8192", "16384,8192", "16384,4096"],
        help="tokens,hidden pairs to time (fused vs unfused).",
    )
    args = parser.parse_args()
    if torch.cuda.device_count() < args.tp:
        aiter.logger.warning("need %s GPUs; skipping", args.tp)
        return

    ranks = _spawn(args.tp, _VALIDITY_CASES, time_it=False)
    _assert_valid(ranks, _VALIDITY_CASES)
    aiter.logger.info(
        "validity (rank 0):\n%s", pd.DataFrame(ranks[0]).to_markdown(index=False)
    )
    link = resolve_link(None)
    mesh = mesh_st_ladder(world_size=args.tp, link=link)
    own_ladder = mesh_norm_ladder(world_size=args.tp, link=link) != mesh
    if own_ladder:
        ranks = _spawn(args.tp, _VALIDITY_CASES, time_it=False, norm_ladder=mesh)
        _assert_valid(ranks, _VALIDITY_CASES)
        aiter.logger.info(
            "validity on the mesh ladder (rank 0):\n%s",
            pd.DataFrame(ranks[0]).to_markdown(index=False),
        )
    for ladder, label in (
        (None, ""),
        *(((mesh, " on the mesh ladder"),) if own_ladder else ()),
    ):
        rows = test_first_call_in_capture(args.tp, norm_ladder=ladder)
        aiter.logger.info(
            "first call in capture%s (rank 0):\n%s",
            label,
            pd.DataFrame(rows).to_markdown(index=False),
        )

    bench = []
    for pair in args.bench:
        t, h = (int(x) for x in pair.split(","))
        bench.append((t, h, False, False))
    if bench:
        ranks = _spawn(args.tp, bench, time_it=True)
        _assert_valid(ranks, bench)
        has_mxfp4 = ARCH == "gfx950"
        df = pd.DataFrame(
            [
                {
                    "tokens": c[0],
                    "hidden": c[1],
                    "fused_us": max(r[i]["fused_us"] for r in ranks),
                    "unfused_us": max(r[i]["unfused_us"] for r in ranks),
                    **(
                        {
                            "mxfp4_us": max(r[i]["mxfp4_us"] for r in ranks),
                            "unfused_mxfp4_us": max(
                                r[i]["unfused_mxfp4_us"] for r in ranks
                            ),
                        }
                        if has_mxfp4
                        else {}
                    ),
                    "sqnr_out_db": min(r[i]["sqnr_out_db"] for r in ranks),
                }
                for i, c in enumerate(bench)
            ]
        )
        df["speedup"] = df["unfused_us"] / df["fused_us"]
        if has_mxfp4:
            df["mxfp4_speedup"] = df["unfused_mxfp4_us"] / df["mxfp4_us"]
        aiter.logger.info("bench tp=%s:\n%s", args.tp, df.to_markdown(index=False))


if __name__ == "__main__":
    freeze_support()
    main()
