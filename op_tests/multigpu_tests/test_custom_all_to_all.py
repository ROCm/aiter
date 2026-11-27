# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU tests and timing of the general all-to-all
(``tensor_model_parallel_all_to_all`` / ``GroupCoordinator.all_to_all``).

Every case compares the custom_all_reduce IPC kernel (and the NCCL fallback) with
a host reference built from all ranks' inputs, bit for bit; the unpack layout
``[rows, W * C]`` is also compared with RCCL ``all_to_all_single`` plus a
transpose. Each op runs many times in a row with new data on the same buffer
(race on the reused buffer), from a captured graph, and with the producer writing
straight into the registered buffer.

Usage:
    pytest op_tests/multigpu_tests/test_custom_all_to_all.py
    python op_tests/multigpu_tests/test_custom_all_to_all.py -n 4 --perf
"""

import argparse
import socket

import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import ensure_spawn_method

SUPPORTED_GFX = ["gfx942", "gfx950"]
ITERS = 12
HIDDEN = 6144  # MXFP8 attention-output width at the SP prefill shapes
ATTN_D = 2048


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _data(gen, shape, dtype, device):
    """Random bytes in the dtype: every bit pattern is a valid payload."""
    raw = torch.randint(
        0,
        256,
        (*shape[:-1], shape[-1] * dtype.itemsize),
        generator=gen,
        dtype=torch.uint8,
        device=device,
    )
    return raw.view(dtype)


def _reference(peers, rank, world, scatter_dim, gather_dim):
    """What rank `rank` must receive: chunk `rank` of every rank's input along
    scatter_dim, concatenated in rank order along gather_dim."""
    chunks = [
        torch.chunk(
            p.view(torch.uint8) if p.dtype.itemsize == 1 else p, world, scatter_dim
        )[rank]
        for p in peers
    ]
    return torch.cat(chunks, dim=gather_dim)


def _all_inputs(x):
    return _gather_host(x.cpu().view(torch.uint8) if x.dtype.itemsize == 1 else x.cpu())


_CPU_GROUP = None


def _gather_host(x):
    peers = [None] * dist.get_world_size(_CPU_GROUP)
    dist.all_gather_object(peers, x, group=_CPU_GROUP)
    return peers


def _same(out, ref):
    out = out.cpu()
    out = out.view(torch.uint8) if out.dtype.itemsize == 1 else out
    return out.shape == ref.shape and torch.equal(
        out.view(torch.uint8), ref.view(torch.uint8)
    )


def _rccl_unpack(tp, x, world):
    """RCCL all_to_all_single + transpose into [rows, W * C]."""
    rows = x.shape[0] // world
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=tp.device_group)
    return out.view(world, rows, -1).transpose(0, 1).reshape(rows, -1)


def _check_layouts(tp, rank, world, device, rows, width, dtype):
    from aiter.dist.communication_op import tensor_model_parallel_all_to_all

    gen = torch.Generator(device=device)
    for gather_dim in (-1, 0):
        for it in range(ITERS):
            gen.manual_seed(1000 * it + rank)
            x = _data(gen, (world * rows, width), dtype, device)
            out = tensor_model_parallel_all_to_all(x, 0, gather_dim)
            peers = _all_inputs(x)
            ref = _reference(peers, rank, world, 0, gather_dim)
            assert _same(out, ref), f"layout gather_dim={gather_dim} differs, iter {it}"
            if gather_dim == -1:
                rccl = _rccl_unpack(tp, x, world)
                assert torch.equal(
                    out.view(torch.uint8), rccl.view(torch.uint8)
                ), "differs from RCCL + transpose"
    ca = tp.device_communicator.ca_comm
    assert ca.should_custom_a2a(x), "fast path not taken for a 16-byte-row matrix"


def _check_dtypes(tp, rank, world, device):
    """Dtypes whose aiter-pybind id is not registered must still take the custom
    byte-view fast path, not raise. Before the byte-view fix should_custom_a2a
    returned True for these yet custom_all_to_all raised
    ``AssertionError: Unsupported dtype`` (torch_to_aiter_pybind only knows the
    arch-native fp8), so the NCCL fallback the PR promises was unreachable."""
    from aiter.dist.communication_op import tensor_model_parallel_all_to_all

    ca = tp.device_communicator.ca_comm
    # (dtype, width) with 16-byte rows. Each of these raised before the fix:
    # e5m2 / bool / uint16 / float64 on any arch, e4m3fn on gfx942 where native
    # fp8 is fnuz. Both 1-byte and multi-byte element sizes are covered.
    cases = [
        ("float8_e5m2", 32),
        ("float8_e4m3fn", 32),  # non-native (fnuz) fp8 on gfx942
        ("bool", 64),
        ("uint16", 16),
        ("float64", 4),
    ]
    gen = torch.Generator(device=device)
    for name, width in cases:
        dtype = getattr(torch, name, None)
        if dtype is None:
            continue
        for it in range(2):
            gen.manual_seed(5000 * it + rank)
            x = _data(gen, (world * 8, width), dtype, device)
            assert ca.should_custom_a2a(x), f"{name}: fast path not taken"
            out = tensor_model_parallel_all_to_all(x, 0, -1)
            # Compare in raw bytes: gather the uint8 view, since all_gather_object
            # cannot serialize the newer typed storages (e.g. uint16 / float8) --
            # that is a torch limitation, unrelated to the kernel under test.
            ref = _reference(
                _gather_host(x.view(torch.uint8).cpu()), rank, world, 0, -1
            )
            assert torch.equal(
                out.cpu().view(torch.uint8), ref
            ), f"{name}: custom byte-view all-to-all differs from reference"


def _check_scales(tp, rank, world, device, rows, d):
    """Block-scaled payload: fp8 codes [W*rows, d] and e8m0 scales [W*rows, d/32]."""
    from aiter.dist.communication_op import tensor_model_parallel_all_to_all

    gen = torch.Generator(device=device)
    for it in range(ITERS):
        gen.manual_seed(2000 * it + rank)
        codes = _data(gen, (world * rows, d), torch.float8_e4m3fn, device)
        scales = _data(gen, (world * rows, d // 32), torch.float8_e8m0fnu, device)
        out_c, out_s = tensor_model_parallel_all_to_all(codes, 0, -1, scales=scales)
        assert out_c.shape == (rows, world * d)
        assert out_s.shape == (rows, world * d // 32)
        assert _same(out_c, _reference(_all_inputs(codes), rank, world, 0, -1))
        assert _same(out_s, _reference(_all_inputs(scales), rank, world, 0, -1))
        rc = _rccl_unpack(tp, codes.view(torch.uint8), world)
        rs = _rccl_unpack(tp, scales.view(torch.uint8), world)
        assert torch.equal(out_c.view(torch.uint8), rc), "codes differ from RCCL"
        assert torch.equal(out_s.view(torch.uint8), rs), "scales differ from RCCL"


def _check_registered(tp, rank, world, device, rows, d):
    """The producer writes the registered buffer; calls follow each other with
    new data in the same buffer and no host sync (a race on the reused buffer
    would corrupt a result)."""
    gen = torch.Generator(device=device)
    outs, expect = [], []
    for it in range(ITERS):
        gen.manual_seed(3000 * it + rank)
        codes = _data(gen, (world * rows, d), torch.uint8, device)
        scales = _data(gen, (world * rows, d // 32), torch.uint8, device)
        off = (codes.numel() + 15) // 16 * 16
        bc = tp.all_to_all_buffer(codes.shape, torch.uint8, 0)
        bs = tp.all_to_all_buffer(scales.shape, torch.uint8, off)
        bc.copy_(codes)
        bs.copy_(scales)
        outs.append(tp.all_to_all(bc, 0, -1, scales=bs, registered=True))
        expect.append((codes, scales))  # stay on the GPU: no host wait inside the loop
    torch.cuda.synchronize()
    for it, ((oc, os_), (c, s)) in enumerate(zip(outs, expect)):
        peers_c, peers_s = _gather_host(c.cpu()), _gather_host(s.cpu())
        assert _same(
            oc, _reference(peers_c, rank, world, 0, -1)
        ), f"registered codes differ, iter {it}"
        assert _same(
            os_, _reference(peers_s, rank, world, 0, -1)
        ), f"registered scales differ, iter {it}"


def _check_graph(tp, rank, world, device, rows, d):
    """Staged and registered all-to-all captured in a graph, replayed on new data."""
    from aiter.dist.parallel_state import graph_capture

    gen = torch.Generator(device=device)
    x = _data(gen, (world * rows, d), torch.uint8, device)
    s = _data(gen, (world * rows, d // 32), torch.uint8, device)
    off = (x.numel() + 15) // 16 * 16
    bx = tp.all_to_all_buffer(x.shape, torch.uint8, 0)
    bs = tp.all_to_all_buffer(s.shape, torch.uint8, off)
    bx.copy_(x)
    bs.copy_(s)
    # The staged call may copy into the same pool as bx/bs (it stages in the pool
    # unless capture reads x in place): x and s carry the same bytes, so either
    # way the registered call reads the right data.
    tp.all_to_all(x, 0, -1, scales=s)  # warm up before the capture
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        staged = tp.all_to_all(x, 0, -1, scales=s)
        reg = tp.all_to_all(bx, 0, -1, scales=bs, registered=True)
    torch.cuda.synchronize()
    for gen_i in range(3):
        g = torch.Generator(device=device).manual_seed(4000 * (gen_i + 1) + rank)
        x.copy_(_data(g, x.shape, torch.uint8, device))
        s.copy_(_data(g, s.shape, torch.uint8, device))
        bx.copy_(x)
        bs.copy_(s)
        peers_x, peers_s = _all_inputs(x), _all_inputs(s)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        for name, (oc, os_) in (("staged", staged), ("registered", reg)):
            assert _same(
                oc, _reference(peers_x, rank, world, 0, -1)
            ), f"graph {name} codes differ, gen {gen_i}"
            assert _same(
                os_, _reference(peers_s, rank, world, 0, -1)
            ), f"graph {name} scales differ, gen {gen_i}"


def _check_fallback(tp, rank, world, device):
    """Shapes the kernel does not take go through NCCL and still follow the contract."""
    from aiter.dist.communication_op import tensor_model_parallel_all_to_all

    gen = torch.Generator(device=device).manual_seed(77 + rank)
    cases = [
        ((world * 3, 10), torch.bfloat16, 0, -1),  # 20-byte rows
        ((world * 4, 5, 8), torch.float32, 0, 1),  # 3D
        ((6, world * 3, 4), torch.bfloat16, 1, 0),
        ((6, world * 3, 4), torch.bfloat16, 1, 2),
        ((world * 2, 64), torch.float16, 0, 0),
        ((4, world * 2), torch.uint8, 1, 0),
    ]
    for shape, dtype, sd, gd in cases:
        x = _data(gen, shape, dtype, device)
        out = tensor_model_parallel_all_to_all(x, sd, gd)
        assert _same(
            out, _reference(_all_inputs(x), rank, world, sd, gd)
        ), f"fallback {shape} {sd}->{gd}"
    # explicit NCCL on a fast-path shape
    x = _data(gen, (world * 8, 64), torch.bfloat16, device)
    out = tensor_model_parallel_all_to_all(x, 0, -1, use_custom=False)
    assert _same(out, _reference(_all_inputs(x), rank, world, 0, -1))
    # scales through the NCCL path
    sc = _data(gen, (world * 8, 4), torch.uint8, device)
    out, sout = tensor_model_parallel_all_to_all(x, 0, -1, scales=sc, use_custom=False)
    assert _same(sout, _reference(_all_inputs(sc), rank, world, 0, -1))


def _check_errors(tp, world, device):
    from aiter.dist.communication_op import tensor_model_parallel_all_to_all

    x = torch.zeros((world * 4, 64), dtype=torch.bfloat16, device=device)
    with pytest.raises(AssertionError):
        tensor_model_parallel_all_to_all(
            torch.zeros((world * 4 + 1, 64), dtype=torch.bfloat16, device=device)
        )
    with pytest.raises(AssertionError):
        tensor_model_parallel_all_to_all(x, 0, 2)
    with pytest.raises(AssertionError):
        tensor_model_parallel_all_to_all(
            x, 0, -1, scales=torch.zeros((world * 4 + world, 4), device=device)
        )
    with pytest.raises(ValueError):
        tensor_model_parallel_all_to_all(x, 0, -1, use_custom=False, registered=True)
    odd = torch.zeros((world * 4, 5), dtype=torch.uint8, device=device)  # 5-byte rows
    with pytest.raises(ValueError):
        tensor_model_parallel_all_to_all(odd, 0, -1, registered=True)
    with pytest.raises(ValueError):
        tp.all_to_all_buffer((1 << 40,), torch.uint8)


def _worker(rank, world, port, perf_tokens, results):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}"
    )
    ensure_model_parallel_initialized(world, 1)
    tp = get_tp_group()
    global _CPU_GROUP
    _CPU_GROUP = tp.cpu_group
    assert (
        tp.device_communicator.ca_comm is not None
        and not tp.device_communicator.ca_comm.disabled
    )
    for rows in (16, 256, 1031):
        for dtype, width in (
            (torch.bfloat16, 256),
            (torch.uint8, 2048),
            (torch.float32, 64),
        ):
            _check_layouts(tp, rank, world, device, rows, width, dtype)
    _check_dtypes(tp, rank, world, device)
    _check_scales(tp, rank, world, device, 1024, ATTN_D)
    _check_scales(tp, rank, world, device, 77, 512)
    _check_registered(tp, rank, world, device, 512, ATTN_D)
    _check_graph(tp, rank, world, device, 512, ATTN_D)
    _check_fallback(tp, rank, world, device)
    _check_errors(tp, world, device)
    if perf_tokens:
        results.put((rank, _perf(tp, rank, world, device, perf_tokens)))
    torch.cuda.synchronize()
    tp.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


def _time_us(fn, iters=20, reps=7):
    for _ in range(3):
        fn()
    best = []
    for _ in range(reps):
        dist.barrier()
        torch.cuda.synchronize()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        best.append(a.elapsed_time(b) / iters * 1000.0)
    return min(best)


def _perf_row(tp, world, device, t):
    rows = t // world
    codes = torch.randint(0, 255, (t, ATTN_D), dtype=torch.uint8, device=device)
    scales = torch.randint(0, 255, (t, ATTN_D // 32), dtype=torch.uint8, device=device)
    off = (codes.numel() + 15) // 16 * 16
    bc = tp.all_to_all_buffer(codes.shape, torch.uint8, 0)
    bs = tp.all_to_all_buffer(scales.shape, torch.uint8, off)
    bc.copy_(codes)
    bs.copy_(scales)
    rc, rs = torch.empty_like(codes), torch.empty_like(scales)

    def rccl_two():
        dist.all_to_all_single(rc, codes, group=tp.device_group)
        dist.all_to_all_single(rs, scales, group=tp.device_group)
        rc.view(world, rows, ATTN_D).transpose(0, 1).reshape(rows, -1)
        rs.view(world, rows, -1).transpose(0, 1).reshape(rows, -1)

    comm = tp.device_communicator.pynccl_comm

    def grouped_send_recv():
        # the grouped pynccl send/recv hand-off of ATOM's SP prefill, plus the unpack
        comm.group_start()
        for x, o in ((codes, rc), (scales, rs)):
            xb, ob = x.reshape(-1), o.reshape(-1)
            n = xb.numel() // world
            for r in range(world):
                comm.send(xb[r * n : (r + 1) * n], r)
                comm.recv(ob[r * n : (r + 1) * n], r)
        comm.group_end()
        rc.view(world, rows, ATTN_D).transpose(0, 1).reshape(rows, -1)
        rs.view(world, rows, -1).transpose(0, 1).reshape(rows, -1)

    cands = {
        "custom registered (zero-copy)": lambda: tp.all_to_all(
            bc, 0, -1, scales=bs, registered=True
        ),
        "custom staged (copy-in)": lambda: tp.all_to_all(codes, 0, -1, scales=scales),
        "rccl a2a + transpose": rccl_two,
        "pynccl grouped send/recv + transpose": grouped_send_recv,
        "nccl fallback in all_to_all": lambda: tp.all_to_all(
            codes, 0, -1, scales=scales, use_custom=False
        ),
    }
    row = {"tokens": t, "world": world}
    for name, fn in cands.items():
        row[f"{name} us"] = _time_us(fn)
    return row


def _perf(tp, rank, world, device, tokens):
    return [_perf_row(tp, world, device, t) for t in tokens]


def _run(world, perf_tokens=()):
    ensure_spawn_method()
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, port, perf_tokens, results))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    perf = []
    if perf_tokens:
        perf = [results.get(timeout=1200) for _ in range(world)]
    for p in procs:
        p.join(timeout=300)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    return perf


@pytest.mark.parametrize("world", [2, 4, 8])
def test_custom_all_to_all(world):
    if torch.version.hip is None or torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} ROCm GPUs with peer access")
    if get_gfx_runtime() not in SUPPORTED_GFX:
        pytest.skip("custom all-to-all is validated on gfx942 and gfx950")
    _run(world)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--world-size", type=int, nargs="+", default=[2, 4])
    parser.add_argument(
        "--perf", action="store_true", help="also time the MXFP8 exchange"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8192, 16384])
    args = parser.parse_args()
    if get_gfx_runtime() not in SUPPORTED_GFX:
        logger.info("gfx %s not supported, nothing to run", get_gfx_runtime())
        return
    frames = []
    for world in args.world_size:
        perf = _run(world, args.tokens if args.perf else ())
        logger.info("world size %d: correctness passed", world)
        if perf:
            # the slowest rank sets the time of a collective
            df = pd.DataFrame([r for _, rows in perf for r in rows])
            frames.append(df.groupby(["world", "tokens"], as_index=False).max())
    if frames:
        logger.info(
            "all-to-all timing (max over ranks):\n%s",
            pd.concat(frames).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
