# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which backend serves a plain all-reduce, end to end.

Run:
    pytest op_tests/multigpu_tests/test_allreduce_routing.py
"""

import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_open_port

try:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    _ARCH = get_gfx_runtime()
except Exception:  # noqa: BLE001
    _ARCH = ""

# The archs the FlyDSL schedules are tuned for; elsewhere both slots stay on
# their HIP backends and there is no routing decision to observe.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")
_SUPPORTED_WORLDS = (2, 4, 8)

pytestmark = pytest.mark.skipif(
    _ARCH not in _SUPPORTED_ARCHS,
    reason=f"FlyDSL all-reduce needs {_SUPPORTED_ARCHS}, got {_ARCH!r}",
)

# Routing outcomes, named for the backend that ends up moving the bytes.
_FLY_QUANT = "fly_quant"  # FlyDSL mesh/ring, in the quick-reduce slot
_FLY_ONESHOT = "fly_oneshot"  # FlyDSL exact one-shot, in the custom-AR slot
_HIP_QR = "hip_qr"  # HIP quick-reduce
_CDR = "cdr"  # HIP cross_device_reduce
_FALLBACK = "fallback"  # neither slot took it: RCCL / torch.distributed


def _route(qr, ca, t) -> str:
    """The backend ``CudaCommunicator.all_reduce`` would reach for *t*."""
    if qr is not None and not qr.disabled and qr.should_quick_allreduce(t):
        return _FLY_QUANT if qr._should_fly(t) else _HIP_QR
    if ca is not None and not ca.disabled and ca.should_custom_ar(t):
        return _FLY_ONESHOT if ca._should_fly_oneshot(t, True, False) else _CDR
    return _FALLBACK


def _bf16(nbytes: int, device, fill: float = 0.0):
    return torch.full((max(8, nbytes // 2),), fill, dtype=torch.bfloat16, device=device)


def _cases(qr, ca, device):
    """(label, tensor, expected route) for this rank's live policy. """
    floor = qr._fly_policy.floor  # exclusive: quant slot serves strictly above
    ceil = ca._fly_policy.max_bytes  # inclusive: one-shot serves up to here
    out = [
        # Above the quant floor, and the quick-reduce slot is tried first.
        ("above-floor-bf16", _bf16(floor * 2, device), _FLY_QUANT),
        # Below it the quant slot declines, which is *how* the one-shot in the
        # next slot gets the payload.
        ("below-floor-bf16", _bf16(_round16(floor // 4), device), _FLY_ONESHOT),
        # FlyDSL is bf16-only and cannot honour
        # AITER_QUICK_REDUCE_CAST_BF16_TO_FP16, so fp16 must fall to the HIP
        # kernel *within* the same slot. This is the case that breaks if the
        # slot ever selects one backend exclusively.
        (
            "fp16",
            torch.zeros((16 << 20) // 2, dtype=torch.float16, device=device),
            _HIP_QR,
        ),
        # Strided: every FlyDSL schedule reads 16 B atoms and both slots require
        # (weak) contiguity, so this leaves the chain entirely.
        (
            "strided-bf16",
            torch.zeros((4096, 64), dtype=torch.bfloat16, device=device)[:, :32],
            _FALLBACK,
        ),
    ]
    if ceil < floor:
        # The two ceilings are fitted against different alternatives and do not
        # order (today only xgmi/4 inverts). In the band between them the
        # one-shot still beats the mesh but cross_device_reduce already beats
        # the one-shot, so both FlyDSL families decline on purpose.
        out.append(
            ("ceiling-gap-bf16", _bf16(_round16((ceil + floor) // 2), device), _CDR)
        )
    return out


def _round16(n: int) -> int:
    return max(16, n // 16 * 16)


def _routing_worker(rank, world_size, port, expect_fly):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    try:
        set_custom_all_reduce(True)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="nccl",
        )
        ensure_model_parallel_initialized(world_size, 1)
        dc = get_tp_group().device_communicator
        qr, ca = dc.qr_comm, dc.ca_comm

        built = (qr is not None and qr._fly_policy is not None) and (
            ca is not None and ca._fly_oneshot is not None
        )
        assert built is expect_fly, (
            f"rank {rank}: AITER_FLY_AR={os.environ.get('AITER_FLY_AR')!r} but "
            f"fly_quant_built={qr is not None and qr._fly_policy is not None}, "
            f"fly_oneshot_built={ca is not None and ca._fly_oneshot is not None}"
        )

        if not expect_fly:
            # Gate closed: nothing may route to a FlyDSL backend, at any size.
            # The HIP backends keep serving, so this is also the proof that
            # enabling FlyDSL is what changed routing and not some side effect.
            for nbytes in (4 << 10, 256 << 10, 4 << 20, 64 << 20):
                got = _route(qr, ca, _bf16(nbytes, device))
                assert got not in (_FLY_QUANT, _FLY_ONESHOT), (
                    f"rank {rank}: {nbytes} B routed to {got} with AITER_FLY_AR unset"
                )
            _teardown()
            return

        for label, t, want in _cases(qr, ca, device):
            got = _route(qr, ca, t)
            assert got == want, (
                f"rank {rank}: {label} ({t.numel() * t.element_size()} B, "
                f"{t.dtype}) routed to {got!r}, expected {want!r}"
            )
            del t

        _assert_reduces(qr, ca, device, world_size, rank)
        _assert_reduces_under_graph(qr, ca, device, world_size, rank)
        _teardown()
    except Exception:
        _teardown()
        raise


def _expected_sum(world_size: int) -> float:
    """Ranks contribute rank+1, so the all-reduce is the triangular number."""
    return float(world_size * (world_size + 1) // 2)


def _assert_reduces(qr, ca, device, world_size, rank):
    """Each routed payload must still produce the right sum.

    Constant fills: exactly representable in INT4, so a quantizing backend is
    held to the same equality as an exact one and the check stays about whether
    the bytes moved correctly.
    """
    want = _expected_sum(world_size)
    for label, t, route in _cases(qr, ca, device):
        if t.dtype is not torch.bfloat16 or not t.is_contiguous():
            del t
            continue
        t.fill_(float(rank + 1))
        got = tensor_model_parallel_all_reduce(t)
        err = (got.float() - want).abs().max().item()
        assert err == 0.0, f"rank {rank}: {label} via {route} gave max err {err}"
        del t, got


def _assert_reduces_under_graph(qr, ca, device, world_size, rank):
    """The one-shot branch sits ahead of CustomAllreduce's ``_IS_CAPTURING``
    check, so it has to be correct in all three states: the warmup pass (where
    the HIP path deliberately returns zeros), inside capture, and on replay.

    Sized below the one-shot ceiling so it routes there -- the quick-reduce slot
    needs no graph handling either way, since both of its backends use a static
    IPC inbox.
    """
    want = _expected_sum(world_size)
    nbytes = _round16(min(ca._fly_policy.max_bytes, qr._fly_policy.floor) // 4)
    x = _bf16(nbytes, device, fill=float(rank + 1))
    assert _route(qr, ca, x) == _FLY_ONESHOT, "graph case did not route to the one-shot"

    with graph_capture() as gc:
        warm = tensor_model_parallel_all_reduce(x)
        torch.cuda.synchronize()
        err = (warm.float() - want).abs().max().item()
        assert err == 0.0, f"rank {rank}: graph warmup gave max err {err}"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gc.stream):
            out = tensor_model_parallel_all_reduce(x)
    out.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    err = (out.float() - want).abs().max().item()
    assert err == 0.0, f"rank {rank}: graph replay gave max err {err}"
    del x


def _teardown():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
    torch.cuda.empty_cache()


def _spawn(worker, world_size, *args, timeout=1800):
    """Fresh port per spawn, with a deadline.

    Mirrors ``test_reuse_identical_rank_groups._spawn``. The deadline matters
    more here: the failure mode this suite guards against is ranks disagreeing
    on a backend, and two backends with incompatible wire protocols hang rather
    than return a wrong answer. The budget is generous because a cold FlyDSL
    JIT compiles every ladder rung at communicator init.
    """
    port = get_open_port()
    ctx = mp.spawn(
        worker, args=(world_size, port, *args), nprocs=world_size, join=False
    )
    deadline = time.monotonic() + timeout
    while not ctx.join(timeout=1):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.terminate()
            raise TimeoutError(
                f"{worker.__name__} did not finish within {timeout}s "
                "(ranks likely disagree on which backend serves a payload)"
            )


def _need(world_size):
    have = torch.cuda.device_count()
    if have < world_size:
        pytest.skip(f"needs {world_size} GPUs, have {have}")


@pytest.mark.parametrize("world_size", _SUPPORTED_WORLDS)
def test_routing_with_flydsl_enabled(monkeypatch, world_size):
    """Each payload class reaches the backend the design says it should.

    ``monkeypatch`` rather than a bare ``os.environ`` write: both variables are
    read when ``quick_all_reduce`` is first imported, and the spawned children
    inherit the parent's environment at interpreter start.
    """
    _need(world_size)
    monkeypatch.setenv("AITER_FLY_AR", "1")
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    _spawn(_routing_worker, world_size, True)


def test_routing_without_flydsl_is_unchanged(monkeypatch):
    """The gate closed: no payload reaches a FlyDSL backend.

    The negative control for the suite above -- without it, a routing table that
    sent everything to FlyDSL would pass every positive assertion.
    """
    _need(2)
    monkeypatch.delenv("AITER_FLY_AR", raising=False)
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    _spawn(_routing_worker, 2, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
