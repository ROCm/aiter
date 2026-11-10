# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for ``QuickAllReduceInt4.allreduce_norm_residual_norm``.

The post-norm must see the bf16 values plain ``allreduce`` stores. Each case
checks the boundary outputs against that unfused pair (tight) and that every
rank produces the same tensors. SQNR against an fp32 NCCL sum is reported
only: INT4 is lossy, and the fused kernel is supposed to match the lossy
all-reduce, not the exact sum.
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import aiter
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

set_start_method("spawn", force=True)

from aiter.ops.flydsl.kernels.comm.allreduce_shared import WORLD

try:
    ARCH = get_gfx()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")
POST_EPS = 1e-6
PRE_EPS = 1e-5
TIGHT_RTOL = 2e-2
TIGHT_ATOL = 2e-2
TIGHT_ERR_RATIO = 0.01
# (tokens, hidden, per_channel, post_weight, post_input_scale, gate_stream, gain)
_CASES = [
    (1, 2048, True, True, False, True, 0.0),
    (37, 8192, True, True, False, True, 0.0),
    (37, 4096, False, True, True, False, 0.5),
    (64, 8192, True, False, False, True, 0.0),
]


def _coeff(gen, hidden, per_channel, device):
    n = hidden if per_channel else 1
    return torch.rand((n,), generator=gen, device=device, dtype=torch.float32)


def _boundary(branch, stream, post_w, pre_w, alpha, beta, post_gain, pre_gain, post_is):
    x = branch.float()
    if post_w is not None:
        pw = post_w.float() + post_gain
        if post_is:
            x = x * pw
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + POST_EPS)
    if post_w is not None and not post_is:
        y = y * (post_w.float() + post_gain)
    new_stream = alpha * stream + beta * y
    z = new_stream
    rw = pre_w.float() + pre_gain
    z = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + PRE_EPS)
    z = z * rw
    return z.to(torch.bfloat16), new_stream


def _run_rank(rank, tp, init_method, cases):
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
    fly = QuickAllReduceInt4(
        group=gloo, device=device, rank=rank, world_size=tp, min_bytes=0
    )
    rows = []
    try:
        for ntok, hidden, per_channel, post_w, post_is, gate, gain in cases:
            gen = torch.Generator(device="cpu").manual_seed(1234 + rank)
            shared = torch.Generator(device="cpu").manual_seed(99)
            inp = (torch.randn((ntok, hidden), generator=gen) * 0.1).to(
                device=device, dtype=torch.bfloat16
            )
            stream = torch.randn(
                (ntok, hidden), generator=shared, dtype=torch.float32, device=device
            )
            pre_w = (torch.randn((hidden,), generator=shared) * 0.2 + 1).to(
                device=device, dtype=torch.bfloat16
            )
            post = None
            if post_w:
                post = (torch.randn((hidden,), generator=shared) * 0.2 + 1).to(
                    device=device, dtype=torch.bfloat16
                )
            beta = _coeff(shared, hidden, per_channel, device)
            alpha = _coeff(shared, hidden, per_channel, device) if gate else 1.0
            ar = torch.empty_like(inp)
            fly.allreduce(inp, ar)
            ref_out, ref_stream = _boundary(
                ar, stream, post, pre_w, alpha, beta, gain, 0.0, post_is
            )
            out = torch.empty_like(inp)
            new_stream = torch.empty_like(stream)
            fly.allreduce_norm_residual_norm(
                inp,
                stream,
                post,
                pre_w,
                POST_EPS,
                PRE_EPS,
                out,
                new_stream,
                alpha=alpha,
                beta=beta,
                post_gain_center=gain,
                pre_gain_center=0.0,
                post_input_scale=post_is,
            )
            torch.cuda.synchronize()
            gathered = [torch.empty_like(out) for _ in range(tp)]
            dist.all_gather(gathered, out, group=dist.group.WORLD)
            same_out = all(torch.equal(g, gathered[0]) for g in gathered)
            gathered = [torch.empty_like(new_stream) for _ in range(tp)]
            dist.all_gather(gathered, new_stream, group=dist.group.WORLD)
            same_stream = all(torch.equal(g, gathered[0]) for g in gathered)
            tag = f"rank {rank} {ntok}x{hidden}"
            rows.append(
                {
                    "err_out": float(
                        checkAllclose(
                            ref_out.float(),
                            out.float(),
                            rtol=TIGHT_RTOL,
                            atol=TIGHT_ATOL,
                            tol_err_ratio=TIGHT_ERR_RATIO,
                            printLog=False,
                            msg=f"{tag} out",
                        )
                    ),
                    "err_stream": float(
                        checkAllclose(
                            ref_stream,
                            new_stream,
                            rtol=TIGHT_RTOL,
                            atol=TIGHT_ATOL,
                            tol_err_ratio=TIGHT_ERR_RATIO,
                            printLog=False,
                            msg=f"{tag} new_stream",
                        )
                    ),
                    "ranks_equal": same_out and same_stream,
                }
            )
            dist.barrier()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(tp, cases):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = Pool(processes=tp)
    try:
        futs = [
            pool.apply_async(_run_rank, args=(rank, tp, init_method, cases))
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


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt4 unsupported on %s; skipping", ARCH)
        return
    tp = int(os.environ.get("FLYDSL_QR_TP", WORLD))
    if torch.cuda.device_count() < tp:
        aiter.logger.warning("need %s GPUs; skipping", tp)
        return
    ranks = _spawn(tp, _CASES)
    fails = []
    for rank, rows in enumerate(ranks):
        for case, row in zip(_CASES, rows, strict=True):
            where = f"rank {rank} {case}"
            if (
                row["err_out"] >= TIGHT_ERR_RATIO
                or row["err_stream"] >= TIGHT_ERR_RATIO
            ):
                fails.append(
                    f"{where}: out {row['err_out']} stream {row['err_stream']}"
                )
            if not row["ranks_equal"]:
                fails.append(f"{where}: ranks disagree")
    if fails:
        raise AssertionError("; ".join(fails))
    aiter.logger.info("boundary validity passed tp=%s cases=%s", tp, len(_CASES))


if __name__ == "__main__":
    freeze_support()
    main()
