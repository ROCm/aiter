# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for custom all-reduce residual fusion and the
fused AR+RMSNorm ``num_norm_rows`` / ``skip_residual`` options.

Inputs are small integers so every sum is exact in bf16; reduced tensors are
compared with atol=0, RMSNorm outputs with a bf16 tolerance.
"""

import argparse
import logging

import torch
import torch.distributed as dist

from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

logger = logging.getLogger("aiter")

EPS = 1e-6
HIDDEN = 7168


def _int_inputs(seed, tp, rank, shape, device):
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randint(-8, 9, (tp, *shape), generator=gen, dtype=torch.int16)
    residual = torch.randint(-8, 9, shape, generator=gen, dtype=torch.int16)
    x = parts[rank].to(device, torch.bfloat16)
    reduced = parts.sum(dim=0, dtype=torch.float32).to(device)
    return x, reduced, residual.to(device, torch.bfloat16)


def _rms_ref(x32, w):
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _run_graph(ca, fn, inputs_src, inputs_dst):
    """Capture ``fn`` once, refresh inputs, replay, and return its outputs."""
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.graph(graph, stream=stream):
        outs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    outs = outs if isinstance(outs, tuple) else (outs,)
    for o in outs:
        if all(o.data_ptr() != d.data_ptr() for d in inputs_dst):
            o.fill_(float("nan"))
    for s, d in zip(inputs_src, inputs_dst):
        d.copy_(s)
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    return graph, outs


def check_all_reduce_residual(ca, tp, rank, device, graphs):
    # 1-stage below the size crossover, 2-stage above it. The last shape is a
    # 2-stage size whose 16-byte packs do not split evenly across ranks, so it
    # takes the naive kernel (as does every size at TP6).
    shapes = [(m, HIDDEN) for m in (1, 8, 64, 512)] + [(8 * (tp * 12800 + 1),)]
    for shape in shapes:
        m = shape[0]
        x, reduced, residual = _int_inputs(m, tp, rank, shape, device)
        ref = (reduced + residual.float()).to(torch.bfloat16)
        out = ca.custom_all_reduce_residual(x, residual)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"eager {shape=}")

        x_in = torch.zeros_like(x)
        graph, (gout,) = _run_graph(
            ca,
            lambda x_in=x_in, residual=residual: ca.custom_all_reduce_residual(
                x_in, residual
            ),
            [x],
            [x_in],
        )
        graphs.append((graph, x_in, gout))
        torch.testing.assert_close(gout, ref, rtol=0, atol=0, msg=f"graph {shape=}")
        if rank == 0:
            logger.info("all_reduce_residual passed: shape=%s, TP%d", shape, tp)


def check_residual_rejects_bad_out(ca, rank, device):
    """The native op validates `out` before launching, not just `residual`."""
    from aiter.ops import custom_all_reduce as ops

    x = torch.ones((4, HIDDEN), dtype=torch.bfloat16, device=device)
    bad_outs = {
        "undersized": torch.empty((2, HIDDEN), dtype=x.dtype, device=device),
        "wrong dtype": torch.empty(x.shape, dtype=torch.float32, device=device),
    }
    for name, out in bad_outs.items():
        try:
            ops.all_reduce_residual(
                ca._ptr,
                x,
                out,
                x,
                True,
                ca._pool["input"].data_ptr,
                ca._pool["input"].max_size,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"all_reduce_residual accepted a {name} out")
    if rank == 0:
        logger.info("all_reduce_residual out checks passed")


def check_norm_rows_skip_residual(ca, tp, rank, device, graphs):
    w = (torch.arange(HIDDEN, device=device) % 7 + 1).to(torch.bfloat16) / 4
    # 16 rows use the single-kernel 2-stage path, 160 rows use
    # reduce-scatter + local RMSNorm.
    for m in (16, 160):
        for norm_rows, skip in ((m // 4, True), (m // 4, False), (m, True)):
            x, reduced, residual = _int_inputs(
                m * 3 + norm_rows, tp, rank, (m, HIDDEN), device
            )
            r32 = reduced if skip else reduced + residual.float()
            ref_norm = _rms_ref(r32, w)[:norm_rows]

            def call(inp, x=x, residual=residual, norm_rows=norm_rows, skip=skip):
                out = torch.full(
                    (norm_rows, HIDDEN), float("nan"), dtype=x.dtype, device=device
                )
                return ca.custom_fused_ar_rms(
                    inp,
                    residual,
                    w,
                    EPS,
                    False,
                    out=out,
                    num_norm_rows=norm_rows,
                    skip_residual=skip,
                )

            ctx = f"{m=} {norm_rows=} {skip=}"
            out, res_out = call(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                res_out.float(), r32, rtol=0, atol=0, msg=f"eager res_out {ctx}"
            )
            torch.testing.assert_close(
                out.float(), ref_norm, rtol=2e-2, atol=2e-2, msg=f"eager out {ctx}"
            )

            x_in = torch.zeros_like(x)
            graph, (gout, gres) = _run_graph(
                ca, lambda x_in=x_in: call(x_in), [x], [x_in]
            )
            graphs.append((graph, x_in, gout, gres))
            torch.testing.assert_close(
                gres.float(), r32, rtol=0, atol=0, msg=f"graph res_out {ctx}"
            )
            torch.testing.assert_close(
                gout.float(), ref_norm, rtol=2e-2, atol=2e-2, msg=f"graph out {ctx}"
            )
            if rank == 0:
                logger.info("fused_ar_rms options passed: %s, TP%d", ctx, tp)


def check_uses_1stage_ar_gfx1250(ca, rank, device):
    """gfx1250 runs only one-shot allreduce kernels, at every size."""
    big = torch.empty((4096, HIDDEN), dtype=torch.bfloat16, device=device)
    saved = ca._is_gfx1250
    try:
        ca._is_gfx1250 = True
        assert ca.uses_1stage_ar(big), "gfx1250 reported a 2-stage allreduce"
    finally:
        ca._is_gfx1250 = saved
    if rank == 0:
        logger.info("uses_1stage_ar gfx1250 check passed")


def check_norm_rejects_short_buffers(ca, rank, device):
    """The native op checks buffer extents against m and num_norm_rows."""
    from aiter.ops import custom_all_reduce as ops

    m, norm_rows = 16, 4
    x = torch.ones((m, HIDDEN), dtype=torch.bfloat16, device=device)
    w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)

    def rows(k):
        return torch.empty((k, HIDDEN), dtype=x.dtype, device=device)

    cases = {
        "out shorter than num_norm_rows": (x, rows(m), rows(norm_rows - 1), False),
        "res_out shorter than m": (x, rows(m - 1), rows(norm_rows), False),
        "res_inp shorter than m": (rows(m - 1), rows(m), rows(norm_rows), False),
    }
    for name, (res_inp, res_out, out, skip) in cases.items():
        try:
            ops.fused_allreduce_rmsnorm(
                ca._ptr,
                x,
                res_inp,
                res_out,
                out,
                w,
                EPS,
                ca._pool["input"].data_ptr,
                ca._pool["input"].max_size,
                False,
                False,
                norm_rows,
                skip,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"fused_allreduce_rmsnorm accepted {name}")
    if rank == 0:
        logger.info("fused_ar_rms buffer extent checks passed")


def check_1stage_rejects_options(ca, rank, device):
    from aiter.ops import custom_all_reduce as ops

    m = 4
    x = torch.ones((m, HIDDEN), dtype=torch.bfloat16, device=device)
    w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    for kwargs in ({"skip_residual": True}, {"num_norm_rows": 2}):
        try:
            ca.custom_fused_ar_rms(x, x, w, EPS, True, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"use_1stage=True accepted {kwargs}")
    # A padded output width takes the pad kernel, which has no row options,
    # including when fused_ar_rms allocates that output itself.
    for kwargs in ({"skip_residual": True}, {"num_norm_rows": 2}):
        try:
            ca.custom_fused_ar_rms(
                x, x, w, EPS, False, out_hidden_dim=HIDDEN + 128, **kwargs
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"padded out_hidden_dim accepted {kwargs}")
    # The native op checks too, for callers that bypass the Python wrapper.
    out, res_out = torch.empty_like(x), torch.empty_like(x)
    try:
        ops.fused_allreduce_rmsnorm(
            ca._ptr,
            x,
            x,
            res_out,
            out,
            w,
            EPS,
            ca._pool["input"].data_ptr,
            ca._pool["input"].max_size,
            True,
            False,
            -1,
            True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("native use_1stage=True accepted skip_residual")
    # num_norm_rows == m is a no-op and stays allowed on the 1-stage path.
    ca.custom_fused_ar_rms(x, x, w, EPS, True, num_norm_rows=m)
    torch.cuda.synchronize()
    if rank == 0:
        logger.info("1-stage option checks passed")


def _worker(rank, tp, init_method):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("gloo", world_size=tp, rank=rank, init_method=init_method)
    ca = None
    graphs = []
    try:
        ca = CustomAllreduce(dist.group.WORLD, device)
        assert not ca.disabled, "requires custom allreduce"
        check_all_reduce_residual(ca, tp, rank, device, graphs)
        check_residual_rejects_bad_out(ca, rank, device)
        check_uses_1stage_ar_gfx1250(ca, rank, device)
        # The fused allreduce + RMSNorm op rejects TP6.
        if tp != 6:
            check_norm_rows_skip_residual(ca, tp, rank, device, graphs)
            check_1stage_rejects_options(ca, rank, device)
            check_norm_rejects_short_buffers(ca, rank, device)
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        graphs.clear()
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


parser = argparse.ArgumentParser(description="custom AR residual / norm-row options")
parser.add_argument("-t", "--tp-size", type=int, choices=[2, 4, 6, 8], default=8)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.tp_size, get_distributed_init_method(get_ip(), get_open_port())),
        nprocs=args.tp_size,
        join=True,
    )
