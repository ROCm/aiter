# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FlyDSL all-to-all policy (``alltoall_policy``) and the rank agreement in
the init of ``QuickAllToAll``. CPU only.

The policy tests check the tables and the environment overrides. The rank
agreement tests run the init on a two-rank gloo group with a stand-in engine:
each rank decides locally (environment, arch, link) whether to build, the
engine constructor is a collective, and a rank that declines alone must not
leave its peer waiting in it.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import socket
from datetime import timedelta
from unittest import mock

import pytest

from aiter.ops.flydsl import alltoall_policy as a2a

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
LINKS = ("pcie", "xgmi")
LADDERS = {
    "mesh": (a2a.mesh_st_ladder, a2a.A2A_MESH_SUPER_TILES),
    "ring": (a2a.ring_st_ladder, a2a.A2A_RING_SUPER_TILES),
}


@pytest.mark.parametrize("algorithm", list(LADDERS))
@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("world_size", a2a.SUPPORTED_WORLDS)
def test_ladders_are_buildable(algorithm, link, world_size):
    """Every ladder starts at 0, ascends, and names only values the kernel
    builds."""
    ladder_of, super_tiles = LADDERS[algorithm]
    ladder = ladder_of(world_size, link)
    assert ladder, f"no {algorithm} ladder for {link}/TP{world_size}"
    assert ladder[0].min_bytes == 0
    floors = [r.min_bytes for r in ladder]
    assert floors == sorted(set(floors))
    for rung in ladder:
        assert rung.super_tile in super_tiles
        assert rung.block in a2a.SUPPORTED_BLOCKS
        assert rung.grid_cap >= 1


def test_unknown_key_has_no_ladder():
    assert a2a.mesh_st_ladder(3, "pcie") == ()
    assert a2a.ring_st_ladder(8, "nvlink") == ()


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
_ENV_VARS = (
    a2a.ENABLE_VAR,
    a2a.CODEC_VAR,
    a2a.ALGORITHM_VAR,
    a2a.MIN_BYTES_VAR,
    a2a.MAX_BYTES_VAR,
    a2a.RING_MIN_BYTES_VAR,
)


@pytest.fixture
def env(monkeypatch):
    """A clean A2A environment; returns ``monkeypatch.setenv``."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch.setenv


@pytest.mark.parametrize(
    "value,on", [(None, False), ("0", False), ("true", False), ("1", True)]
)
def test_only_1_enables(env, value, on):
    if value is not None:
        env(a2a.ENABLE_VAR, value)
    assert a2a.enabled() is on


def test_defaults(env):
    pcie4 = a2a.resolve(4, link="pcie")
    assert (pcie4.algorithm, pcie4.codec) == ("auto", "none")
    assert (pcie4.min_bytes, pcie4.max_bytes) == (0, a2a.NO_MAX)
    assert pcie4.ring_from == a2a.RING_FROM_BYTES[("pcie", 4)]
    assert pcie4.families() == ("mesh", "ring")
    # No measured crossover: the mesh at every size.
    xgmi8 = a2a.resolve(8, link="xgmi")
    assert xgmi8.ring_from is None
    assert xgmi8.families() == ("mesh",)


def test_overrides(env):
    env(a2a.CODEC_VAR, "INT6")
    env(a2a.ALGORITHM_VAR, "ring")
    env(a2a.MIN_BYTES_VAR, str(1 << 20))
    env(a2a.MAX_BYTES_VAR, str(64 << 20))
    p = a2a.resolve(4, link="pcie")
    assert (p.codec, p.algorithm) == ("int6", "ring")
    assert (p.min_bytes, p.max_bytes) == (1 << 20, 64 << 20)
    assert p.families() == ("ring",)
    assert not p.routes((1 << 20) - 1)
    assert p.routes(1 << 20) and p.routes(64 << 20)
    assert not p.routes((64 << 20) + 1)


@pytest.mark.parametrize(
    "var,value", [(a2a.CODEC_VAR, "fp8"), (a2a.ALGORITHM_VAR, "tree")]
)
def test_unknown_choice_falls_back_with_a_warning(env, caplog, var, value):
    env(var, value)
    # The aiter logger does not propagate to the root, where caplog listens.
    aiter_logger = logging.getLogger("aiter")
    aiter_logger.addHandler(caplog.handler)
    try:
        p = a2a.resolve(4, link="pcie")
    finally:
        aiter_logger.removeHandler(caplog.handler)
    assert (p.codec, p.algorithm) == ("none", "auto")
    assert value in caplog.text


@pytest.mark.parametrize("value", ["-1", "lots"])
def test_bad_or_default_bound_keeps_the_table(env, value):
    env(a2a.MIN_BYTES_VAR, value)
    assert a2a.resolve(4, link="pcie").min_bytes == 0


def test_caller_codec_pins_the_wire(env):
    env(a2a.CODEC_VAR, "int4")
    assert a2a.resolve(4, link="pcie", codec="none").codec == "none"
    with pytest.raises(ValueError):
        a2a.resolve(4, link="pcie", codec="fp8")


@pytest.mark.parametrize("ring_from", [None, 4 << 20])
def test_family_ranges_partition_the_window(env, ring_from):
    """Under ``auto``, the preload range of each family is exactly the payloads
    ``pick`` sends it, so no served binary is left to compile under capture."""
    env(a2a.MIN_BYTES_VAR, str(64 << 10))
    if ring_from is None:
        env(a2a.ALGORITHM_VAR, "mesh")
    else:
        env(a2a.RING_MIN_BYTES_VAR, str(ring_from))
    p = a2a.resolve(4, link="pcie")
    for nbytes in (64 << 10, (4 << 20) - 1, 4 << 20, 1 << 30):
        family = p.pick(nbytes)
        assert family in p.families()
        lo, hi = p.family_range(family)
        assert lo <= nbytes <= hi
        for other in p.families():
            if other != family:
                olo, ohi = p.family_range(other)
                assert not olo <= nbytes <= ohi


# --------------------------------------------------------------------------- #
# Rank agreement in QuickAllToAll
# --------------------------------------------------------------------------- #
WORLD = 2
# A rank stuck in a collective its peer never enters raises after this.
GLOO_TIMEOUT_S = 30
# Seconds to wait for each rank, import of aiter included.
RANK_TIMEOUT_S = 300
ARCH = "gfx950"
LINK = "pcie"

# One rank (the last) differs from the other in one input to its plan:
# ``(env overrides, plan-input patches)``.
SKEWS = {
    "fly_a2a_off": ({"AITER_FLY_A2A": "0"}, {}),
    "link": ({}, {"link": "xgmi"}),
    "import_failed": ({}, {"import_ok": False}),
    "codec": ({"AITER_FLY_A2A_CODEC": "int4"}, {}),
    "window": ({"AITER_FLY_A2A_MIN_BYTES": str(1 << 20)}, {}),
}


class _FakeEngine:
    """Stand-in for ``FlyQuickAllToAll``. Its constructor is a collective, like
    the real IPC handle exchange, and it raises unless every rank builds the
    same engine."""

    inbox_bytes = 0

    def __init__(self, *, group, rank, device, **kwargs):
        import torch.distributed as dist

        mine = ("engine", sorted(kwargs.items()))
        everyone = [None] * dist.get_world_size(group)
        dist.all_gather_object(everyone, mine, group=group)
        if any(e != mine for e in everyone):
            raise RuntimeError(f"ranks build different engines: {everyone}")

    def preload(self, *, payload_range=None):
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


def _rank(rank: int, port: int, env: dict, patches: dict) -> dict:
    """One rank: join the gloo group, run the ``QuickAllToAll`` init, report."""
    os.environ["AITER_FLY_A2A"] = "1"
    os.environ.update(env)

    import torch
    import torch.distributed as dist

    from aiter.dist.device_communicators import quick_all_to_all as qa2a

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=GLOO_TIMEOUT_S),
    )
    group = dist.group.WORLD
    disagreements = _Disagreements()
    logging.getLogger("aiter").addHandler(disagreements)
    try:
        with (
            mock.patch("aiter.jit.utils.chip_info.get_gfx_runtime", return_value=ARCH),
            mock.patch.object(
                a2a, "detect_link", return_value=patches.get("link", LINK)
            ),
            mock.patch.object(qa2a, "FlyQuickAllToAll", _FakeEngine),
            mock.patch.object(qa2a, "_FLY_IMPORT_OK", patches.get("import_ok", True)),
        ):
            comm = qa2a.QuickAllToAll(group, torch.device("cpu"))
        built = not comm.disabled
        comm.close()
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


def _spawn(skew: tuple[dict, dict] | None, everyone: dict | None = None):
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
            results.append(pool.apply_async(_rank, (rank, port, env, patches)))
        reports = [r.get(timeout=RANK_TIMEOUT_S) for r in results]
    errors = [f"rank {r}: {p['error']}" for r, p in enumerate(reports) if "error" in p]
    assert not errors, "; ".join(errors)
    return reports


def test_matching_ranks_build():
    """The stand-in setup is valid: with nothing skewed, every rank builds."""
    reports = _spawn(None)
    assert all(r["built"] for r in reports)
    assert not any(r["disagreements"] for r in reports)


def test_fly_a2a_off_everywhere_is_silent():
    """``AITER_FLY_A2A`` unset on every rank, the default: nothing is built,
    and the vote logs no disagreement."""
    reports = _spawn(None, everyone={"AITER_FLY_A2A": ""})
    assert not any(r["built"] for r in reports)
    assert not any(r["disagreements"] for r in reports)


@pytest.mark.parametrize("skew", list(SKEWS))
def test_one_skewed_rank_disables_every_rank(skew):
    """One rank's plan differs: the init returns on every rank without
    building, and every rank logs why."""
    reports = _spawn(SKEWS[skew])
    assert not any(r["built"] for r in reports)
    assert all(r["disagreements"] for r in reports)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
