# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rank agreement in the FlyDSL init of ``CustomAllreduce`` and
``QuickAllReduce``: the exact one-shot, the quantized mesh/ring and the exact
fused all-reduce + RMSNorm.

Each rank decides locally (environment, arch, link) whether to build FlyDSL
engines, and the engine constructors are collectives. A rank that declines
alone must not leave its peers waiting in one: the init votes on every rank's
plan first, and builds only if all of them match.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import socket
from datetime import timedelta
from unittest import mock

import pytest

WORLD = 2
# The gloo timeout, in seconds. A rank stuck in a collective its peers never
# enter raises after this instead of hanging the test.
GLOO_TIMEOUT_S = 30
# Seconds to wait for each rank, import of aiter included.
RANK_TIMEOUT_S = 300

# The host every rank reports, unless a skew says otherwise.
ARCH = "gfx950"
LINK = "pcie"

# Skews: one rank (the last) differs from the others in one input to its plan.
# Each maps to ``(env overrides, plan-input patches)`` for that rank.
COMMON_SKEWS = {
    "fly_ar_off": ({"AITER_FLY_AR": "0"}, {}),
    "link": ({}, {"link": "xgmi"}),
    "import_failed": ({}, {"import_ok": False}),
}
ONESHOT_SKEWS = {
    **COMMON_SKEWS,
    "window": ({"AITER_FLY_AR_ONESHOT_MAX_BYTES": str(32 << 10)}, {}),
    "custom_ar_max_size": ({"AITER_CUSTOM_AR_MAX_SIZE": str(64 << 10)}, {}),
}
QUICK_SKEWS = {
    **COMMON_SKEWS,
    "window": ({"AITER_FLY_AR_MESH_MIN_BYTES": str(1 << 20)}, {}),
    "quantization": ({"AITER_QUICK_REDUCE_QUANTIZATION": "INT6"}, {}),
}
FUSED_SKEWS = {
    **COMMON_SKEWS,
    "window": ({"AITER_FLY_AR_FUSED_ONESHOT_MAX_BYTES": str(32 << 10)}, {}),
}
SKEWS = {"oneshot": ONESHOT_SKEWS, "quick": QUICK_SKEWS, "fused": FUSED_SKEWS}
OWNERS = list(SKEWS)


class _FakeEngine:
    """Stand-in for ``OneShotAllReduce``, ``FlyQuickAllReduce`` and
    ``OneShotAllReduceRMSNorm``.

    Its constructor is a collective, like the real IPC handle exchange, and it
    raises unless every rank builds the same engine (the constructor kwargs,
    minus the per-rank ones). A peer in a different collective shows up as a
    mismatch too.
    """

    inbox_bytes = 0

    def __init__(self, *, group, rank, device, **kwargs):
        import torch.distributed as dist

        mine = ("engine", sorted(kwargs.items()))
        everyone = [None] * dist.get_world_size(group)
        dist.all_gather_object(everyone, mine, group=group)
        if any(e != mine for e in everyone):
            raise RuntimeError(f"ranks build different engines: {everyone}")

    def preload(self, *args, payload_range=None):
        pass

    def close(self):
        pass


class _Disagreements(logging.Handler):
    """Counts the warnings the plan vote logs."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "disagree" in record.getMessage():
            self.count += 1


def _build(owner: str, group, rank: int):
    """Run the FlyDSL init of *owner* on a stand-in communicator. Returns
    whether it built the FlyDSL backend."""
    import torch

    if owner == "oneshot":
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce,
            _resolve_car_max_size,
            _resolve_car_min_size,
        )

        ca = CustomAllreduce.__new__(CustomAllreduce)
        ca.disabled = False
        ca.group = group
        ca.rank = rank
        ca.world_size = WORLD
        ca.device = torch.device("cpu")
        ca.fully_connected = True
        ca._car_max_size = _resolve_car_max_size(128 << 20)
        ca._car_min_size = _resolve_car_min_size()
        ca._fly_oneshot = None
        ca._fly_window = None
        ca._init_fly_oneshot()
        return ca._fly_oneshot is not None

    if owner == "fused":
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        ca = CustomAllreduce.__new__(CustomAllreduce)
        ca.disabled = False
        ca.group = group
        ca.rank = rank
        ca.world_size = WORLD
        ca.device = torch.device("cpu")
        ca._fly_rms = None
        ca._init_fly_rms()
        return ca._fly_rms is not None

    from aiter.dist.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        QuickReduceRegime,
    )

    qr = QuickAllReduce.__new__(QuickAllReduce)
    qr.group = group
    qr.rank = rank
    qr.world_size = WORLD
    qr.device = torch.device("cpu")
    qr._fly_ar = None
    qr._fly_rms = None
    qr.qr_quant_level = QuickReduceRegime[os.environ["AITER_QUICK_REDUCE_QUANTIZATION"]]
    qr._init_fly_backend()
    return qr._fly_ar is not None


def _rank(rank: int, port: int, owner: str, env: dict, patches: dict) -> dict:
    """One rank: join the gloo group, run the FlyDSL init, and report."""
    os.environ["AITER_FLY_AR"] = "1"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    os.environ.update(env)

    import torch.distributed as dist

    from aiter.dist.device_communicators import custom_all_reduce as car
    from aiter.dist.device_communicators import flydsl_backends as backends
    from aiter.ops.flydsl import allreduce_policy as fly_policy

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=GLOO_TIMEOUT_S),
    )
    group = dist.group.WORLD
    # The module that imports flydsl for this owner, and the engine it builds.
    module, engine = {
        "oneshot": (car, "OneShotAllReduce"),
        "quick": (backends, "FlyQuickAllReduce"),
        "fused": (backends, "OneShotAllReduceRMSNorm"),
    }[owner]
    disagreements = _Disagreements()
    logging.getLogger("aiter").addHandler(disagreements)
    try:
        with (
            mock.patch("aiter.jit.utils.chip_info.get_gfx_runtime", return_value=ARCH),
            mock.patch.object(
                fly_policy, "detect_link", return_value=patches.get("link", LINK)
            ),
            mock.patch.object(module, engine, _FakeEngine),
            mock.patch.object(module, "_FLY_IMPORT_OK", patches.get("import_ok", True)),
            # The one-shot's plan reads its own module's flag, the fused
            # backend's reads the backends module's; the skew covers both.
            mock.patch.object(car, "_FLY_IMPORT_OK", patches.get("import_ok", True)),
        ):
            built = _build(owner, group, rank)
        # A rank still inside an engine collective pairs up with this gather
        # and raises there.
        everyone = [None] * WORLD
        dist.all_gather_object(everyone, ("done", built), group=group)
        if any(e[0] != "done" for e in everyone):
            raise RuntimeError(f"ranks left the init out of step: {everyone}")
        return {"built": built, "disagreements": disagreements.count}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        logging.getLogger("aiter").removeHandler(disagreements)
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(owner: str, skew: tuple[dict, dict] | None, everyone: dict | None = None):
    """Every rank's report. *skew* applies to the last rank only, *everyone*
    (env overrides) to all of them."""
    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=WORLD) as pool:
        results = []
        for rank in range(WORLD):
            env, patches = dict(everyone or {}), {}
            if skew is not None and rank == WORLD - 1:
                env.update(skew[0])
                patches = skew[1]
            results.append(pool.apply_async(_rank, (rank, port, owner, env, patches)))
        reports = [r.get(timeout=RANK_TIMEOUT_S) for r in results]
    errors = [f"rank {r}: {p['error']}" for r, p in enumerate(reports) if "error" in p]
    assert not errors, "; ".join(errors)
    return reports


@pytest.mark.parametrize("owner", OWNERS)
def test_matching_ranks_build(owner):
    """The stand-in setup is valid: with nothing skewed, every rank builds."""
    reports = _spawn(owner, None)
    assert all(r["built"] for r in reports)
    assert not any(r["disagreements"] for r in reports)


@pytest.mark.parametrize("owner", OWNERS)
def test_flydsl_off_everywhere_is_silent(owner):
    """``AITER_FLY_AR`` unset on every rank, the default: nothing is built,
    and the vote logs no disagreement."""
    reports = _spawn(owner, None, everyone={"AITER_FLY_AR": ""})
    assert not any(r["built"] for r in reports)
    assert not any(r["disagreements"] for r in reports)


@pytest.mark.parametrize(
    "owner,skew",
    [(owner, s) for owner in OWNERS for s in SKEWS[owner]],
)
def test_one_skewed_rank_disables_every_rank(owner, skew):
    """One rank's plan differs: the init returns on every rank without
    building, and every rank logs why."""
    reports = _spawn(owner, SKEWS[owner][skew])
    assert not any(r["built"] for r in reports)
    assert all(r["disagreements"] for r in reports)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
