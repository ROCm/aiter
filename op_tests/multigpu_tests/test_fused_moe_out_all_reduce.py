# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

""" Retarget the fused-MoE combined output into the
CustomAllreduce registered IPC buffer and cross-rank reduce it.

Fold the routed-expert weighted topk-sum *and* the shared-expert add
into the down-GEMM epilogue: ``aiter.fused_moe(out=buf, residual=shared)`` writes
this rank's combined ``[tokens, hidden]`` straight into ``buf`` (validated
single-GPU in ``test_fused_moe_out_residual_single_gpu.py`` / ``test_moe_2stage.py``).

The only cross-rank step left is a plain all-reduce of that combined per-rank
output — *no dedicated kernel*. If ``buf`` aliases the AR registered input buffer
(``CustomAllreduce.moe_out_registered_buffer``) then
``CustomAllreduce.fused_moe_out_all_reduce`` reduces it across ranks with the
copy-in stage skipped.

This test validates that the registered-buffer reduce is bit-comparable to a
reference all-reduce (RCCL) of the *same* per-rank tensor. Both reducers consume
an identical local snapshot, so any producer nondeterminism (a4w4 atomic order)
cancels and only the AR summation path is compared.

Run (4 GPUs):
      python op_tests/multigpu_tests/test_fused_moe_out_all_reduce.py -t 4
    # a4w4 (MXFP4) producer instead of bf16:
      python op_tests/multigpu_tests/test_fused_moe_out_all_reduce.py -t 4 -q a4w4
"""

import os
import argparse
import logging
from typing import Optional

import torch
import torch.distributed as dist
from multiprocessing import Pool, set_start_method, freeze_support

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose
from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def _build_moe_weights(quant, E, model_dim, inter_dim, dtype, device, seed):
    """Build (input, w1, w2, topk_weights, topk_ids, fused_moe_kwargs) for the
    requested producer. ``a4w4`` mirrors the MXFP4 per_1x32/fp4x2 preshuffle
    setup in op_tests/test_moe_2stage.py; ``bf16`` is dense QuantType.No."""
    torch.manual_seed(seed)
    token = 128
    topk = 4
    x = torch.randn((token, model_dim), dtype=dtype, device=device)
    # g1u1 (gate+up) weights so Silu has both halves.
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device=device) / 10.0
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device=device) / 10.0
    score = torch.randn((token, E), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    if quant == "bf16":
        kwargs = dict(
            quant_type=aiter.QuantType.No,
            activation=aiter.ActivationType.Silu,
        )
        return x, w1, w2, topk_weights, topk_ids, kwargs

    # a4w4 (MXFP4): per_1x32 fp4x2 weights, bf16 activations quantized inside
    # fused_moe. Follows the preshuffle fp4x2 branch of test_moe_2stage.
    torch_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
    w1_qt_aiter = shuffle_weight(w1_qt, layout=(16, 16))
    w2_qt_aiter = shuffle_weight(w2_qt, layout=(16, 16))
    w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    kwargs = dict(
        w1_scale=w1_scale_aiter,
        w2_scale=w2_scale_aiter,
        quant_type=aiter.QuantType.per_1x32,
        activation=aiter.ActivationType.Silu,
    )
    return x, w1_qt_aiter, w2_qt_aiter, topk_weights, topk_ids, kwargs


def _worker(
    tp_size,
    pp_size,
    rankID,
    quant,
    E,
    model_dim,
    inter_dim,
    dtype,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    logger.info(f"RANK {rankID}/{tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    try:
        tp = get_tp_group()
        group = tp.device_group
        ca = tp.device_communicator.ca_comm
        # warmup / align ranks
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        torch.cuda.synchronize()

        if ca is None or ca.disabled:
            return {"skipped": "custom AR disabled", "rank": rankID}

        # Per-rank inputs (distinct so the cross-rank sum is non-trivial).
        x, w1, w2, tw, tid, kwargs = _build_moe_weights(
            quant, E, model_dim, inter_dim, dtype, device, seed=1000 + rankID
        )
        token = x.shape[0]
        # Shared-expert output (the residual folded into the combine).
        residual = torch.randn((token, model_dim), dtype=dtype, device=device) / 4.0

        # fused_moe writes combined (routed + shared) straight into
        # the registered IPC buffer, then cross-rank reduce it (copy-in skipped).
        reg = ca.moe_out_registered_buffer(token, model_dim, dtype)
        if reg is None:
            return {"skipped": "registered buffer unavailable", "rank": rankID}
        combined = fused_moe(x, w1, w2, tw, tid, out=reg, residual=residual, **kwargs)
        assert combined.data_ptr() == reg.data_ptr(), "fused_moe must write into reg"
        # Snapshot this rank's pre-reduce combined output for the reference.
        local = reg.clone()

        out_buf = torch.empty_like(reg)
        reduced = ca.fused_moe_out_all_reduce(reg, out=out_buf)

        # --- Reference: reduce the SAME local snapshot via RCCL. Both paths sum
        # identical per-rank tensors, so only the AR summation is under test.
        ref = local.clone()
        dist.all_reduce(ref, group=group)
        torch.cuda.synchronize()

        result = {
            "rank": rankID,
            "reduced": reduced.detach().to("cpu"),
            "ref": ref.detach().to("cpu"),
        }

        # full-pipeline parity vs the conventional "today" path:
        # combined = fused_moe(...); combined += shared; all_reduce(combined).
        # Only meaningful when the producer is deterministic (bf16); the a4w4
        # atomic accumulation order varies run-to-run, so a second fused_moe call
        # would not reproduce `local` bit-for-bit.
        if quant == "bf16":
            plain = fused_moe(x, w1, w2, tw, tid, **kwargs)  # routed combine only
            baseline_local = plain + residual  # shared-add done the old way
            baseline_reduced = tp.all_reduce(baseline_local)  # framework AR
            torch.cuda.synchronize()
            result["baseline_reduced"] = baseline_reduced.detach().to("cpu")

        # CUDA-graph capture/replay of the route-B registered reduce.
        # `reg` still holds this rank's pre-reduce combined output (all_reduce is
        # out-of-place), and it is the address-stable registered IPC buffer, so
        # the registered-input reduce is graph-safe. Capture only the reduce (the
        # new code path); the JIT/flydsl producer is not graph-captured here.
        try:
            graph_out = torch.empty_like(reg)
            with ca.capture():
                # warm-up (pre-capture): mimic allocations / init lazily.
                for _ in range(3):
                    ca.custom_fused_moe_out_all_reduce(reg, out=graph_out)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    ca.custom_fused_moe_out_all_reduce(reg, out=graph_out)
            g.replay()
            torch.cuda.synchronize()
            result["graph_reduced"] = graph_out.detach().to("cpu")
        except Exception as e:  # capture unsupported on this stack -> report
            result["graph_error"] = f"{type(e).__name__}: {e}"

        # The concrete pass route B removes is the AR copy-in
        # (cudaMemcpy of the combined output into the registered pool). Time the
        # registered reduce (copy-in skipped) vs a copy-in reduce of the same
        # data, on this rank, and report per-call microseconds.
        def _time(fn, iters=50, warmup=10):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters * 1e3  # us/call

        try:
            o = torch.empty_like(reg)
            copyin_src = local.clone()  # a normal (unregistered) tensor
            us_reg = _time(lambda: ca.all_reduce(reg, out=o, registered_input=True))
            us_copyin = _time(
                lambda: ca.all_reduce(copyin_src, out=o, registered_input=False)
            )
            result["us_registered"] = us_reg
            result["us_copyin"] = us_copyin
        except Exception as e:
            result["bench_error"] = f"{type(e).__name__}: {e}"

        return result
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()


def run_case(tp_size, quant, E, model_dim, inter_dim, dtype, distributed_init_method):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "49377")
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                _worker,
                args=(
                    tp_size,
                    1,
                    rank,
                    quant,
                    E,
                    model_dim,
                    inter_dim,
                    dtype,
                    distributed_init_method,
                ),
            )
            for rank in range(tp_size)
        ]
        rets = [r.get() for r in rets]

    for r in rets:
        if "skipped" in r:
            logger.warning("rank %s skipped: %s", r.get("rank"), r["skipped"])
            return {"quant": quant, "skipped": r["skipped"], "err": float("nan")}

    # bf16 all-reduce differs from RCCL only by summation-order / accumulation
    # rounding; the established bf16 AR tests (test_fused_ar_rms) use 5e-2.
    atol = rtol = 5e-2
    max_err = 0.0  # registered reduce vs RCCL of same snapshot
    max_err_pipe = 0.0  # full vs conventional sum-then-add-then-AR
    max_err_graph = 0.0  # graph replay vs eager reduce
    graph_ok = True
    us_reg = []
    us_copyin = []
    for r in rets:
        err = checkAllclose(
            r["ref"].float(),
            r["reduced"].float(),
            msg=f"route-B fused_moe_out_all_reduce (quant={quant}) rank={r['rank']}",
            atol=atol,
            rtol=rtol,
        )
        max_err = max(max_err, err)

        if "baseline_reduced" in r:
            e_pipe = checkAllclose(
                r["baseline_reduced"].float(),
                r["reduced"].float(),
                msg=f"#3 route-B == sum-then-add-then-AR rank={r['rank']}",
                atol=atol,
                rtol=rtol,
            )
            max_err_pipe = max(max_err_pipe, e_pipe)

        if "graph_error" in r:
            graph_ok = False
            logger.warning(
                "rank %s graph capture error: %s", r["rank"], r["graph_error"]
            )
        elif "graph_reduced" in r:
            e_graph = checkAllclose(
                r["reduced"].float(),
                r["graph_reduced"].float(),
                msg=f"#5 cudagraph replay == eager reduce rank={r['rank']}",
                atol=atol,
                rtol=rtol,
            )
            max_err_graph = max(max_err_graph, e_graph)

        if "us_registered" in r:
            us_reg.append(r["us_registered"])
            us_copyin.append(r["us_copyin"])
        if "bench_error" in r:
            logger.warning("rank %s bench error: %s", r["rank"], r["bench_error"])

    out = {"quant": quant, "tp": tp_size, "model_dim": model_dim, "err": max_err}
    if quant == "bf16":
        out["err_pipeline_#3"] = max_err_pipe
    out["graph_ok_#5"] = graph_ok
    if graph_ok:
        out["err_graph_#5"] = max_err_graph
    if us_reg:
        out["us_registered_#6"] = round(sum(us_reg) / len(us_reg), 2)
        out["us_copyin_#6"] = round(sum(us_copyin) / len(us_copyin), 2)
        out["copyin_overhead_us_#6"] = round(
            out["us_copyin_#6"] - out["us_registered_#6"], 2
        )
    return out


parser = argparse.ArgumentParser(description="SILO801 route B #3 TP validation")
parser.add_argument("-t", "--tp", type=int, default=4, help="tensor-parallel size")
parser.add_argument(
    "-q", "--quant", choices=["bf16", "a4w4"], default="bf16", help="producer path"
)
parser.add_argument("-e", "--expert", type=int, default=None, help="num experts")
parser.add_argument(
    "-dim",
    type=int,
    default=None,
    help="model_dim (hidden). must give 16B-multiple rows",
)
parser.add_argument("-inter", type=int, default=None, help="inter_dim")


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if torch.cuda.device_count() < args.tp:
        raise SystemExit(f"need >= {args.tp} GPUs, have {torch.cuda.device_count()}")
    # a4w4 needs a tuned MXFP4 config; default to the validated Kimi-ish shape.
    if args.quant == "a4w4":
        E = args.expert or 257
        model_dim = args.dim or 7168
        inter_dim = args.inter or 256
    else:
        E = args.expert or 8
        model_dim = args.dim or 4096
        inter_dim = args.inter or 1024
    ret = run_case(
        tp_size=args.tp,
        quant=args.quant,
        E=E,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtypes.bf16,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
    )
    logger.info("Result: %s", ret)
    print("[aiter] Result:", ret)
    if ret.get("err") == ret.get("err") and "skipped" not in ret:  # not NaN
        assert ret["err"] < 5e-2, f"route-B AR mismatch: {ret}"
        if "err_pipeline_#3" in ret:
            assert ret["err_pipeline_#3"] < 5e-2, f"#3 pipeline mismatch: {ret}"
        assert ret.get("graph_ok_#5", False), f"#5 graph capture failed: {ret}"
        if "err_graph_#5" in ret:
            assert ret["err_graph_#5"] < 5e-2, f"#5 graph replay mismatch: {ret}"
        print("[aiter] PASS")
