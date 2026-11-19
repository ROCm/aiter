# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx1250 behaviour on gfx1250: what a change to its dispatch/combine
wiring must keep.

    python3 test_mega_moe_gfx1250.py [--world N]

Spawns one rank per GPU (4 by default, 2 is the minimum). Anywhere else -- not
gfx1250, fewer than two GPUs, no mori -- it prints why and exits 0, since CI's
MULTIGPU job runs every file in this directory on its gfx950 runners.

Covered, on the fp4 and fp8 dispatch wires (a4w4 and a8w4) plus a bf16 wire:
  * every dispatch/combine pairing against the fp32 reference, and a captured
    graph's replay against the eager result, bit for bit; the fp8 wire runs
    with AITER_FORCE_A8W4=0, since the wire picks the GEMM's A operand;
  * pairings MegaMoE cannot run fail before the symmetric arena is allocated,
    and a failure after it frees what was built;
  * mori's combine reads fused_moe's rows in place (zero-copy staging);
  * close() releases the plans before the window and the window before its memory;
  * the members ATOM's Triton path reaches (_dispatch, _combine, _arena, _config);
  * the deprecated keyword constructor builds what from_recipe builds, and
    required_vmm_bytes covers exactly what the arena allocates;
  * a caller's a1_scale on a quantizing dispatch wire is refused.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback
import warnings
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# The weights, routings, fp32 reference and its tolerances are the bench's.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HIDDEN, INTER, TOPK, TOKENS = 7168, 1024, 8, 32
EXPERTS_PER_RANK = 8
# The quant key a dispatch wire runs as; bf16 has the GEMM quantize to fp4.
QUANT_KEY = {"bf16": "a4w4_mxfp4", "fp4": "a4w4_mxfp4", "fp8": "a8w4_mxfp4"}
# (label, dispatch, combine, dispatch_wire, the step's combine wire)
PAIRINGS = (
    ("mori+fused bf16", "mori", "fused_scatter", "bf16", "bf16"),
    ("mori+fused", "mori", "fused_scatter", "fp4", "bf16"),
    ("mori+fused fp4", "mori", "fused_scatter", "fp4", "fp4"),
    ("mori+fused fp8 combine", "mori", "fused_scatter", "fp4", "fp8"),
    ("mori+fused fp8", "mori", "fused_scatter", "fp8", "bf16"),
    ("mori+mori bf16", "mori", "mori", "bf16", "bf16"),
    ("mori+mori", "mori", "mori", "fp4", "bf16"),
    ("mori+mori fp4", "mori", "mori", "fp4", "fp4"),
    ("mori+mori fp8", "mori", "mori", "fp8", "bf16"),
    ("compact+fused", "compact", "fused_scatter", "fp4", "bf16"),
    ("compact+fused fp4", "compact", "fused_scatter", "fp4", "fp4"),
    ("compact+fused fp8", "compact", "fused_scatter", "fp8", "fp8"),
)
# The combine wires in the legacy combine_quant spelling.
LEGACY_QUANT = {"bf16": "none", "fp8": "mxfp8", "fp4": "mxfp4"}
# Rejected configurations and the message each must raise with: recipe fields,
# or the keyword constructor's flags.
REJECTED = (
    ("recipe", {"dispatch": "compact", "combine": "mori"}, "combine=mori does not"),
    (
        "recipe",
        {"dispatch": "compact", "dispatch_wire": "bf16"},
        "dispatch=compact carries an fp8 or fp4 dispatch_wire",
    ),
    (
        "recipe",
        {"combine": "mori", "combine_wires": ("bf16", "fp8")},
        "combine=mori offers the ('bf16', 'fp4') wires",
    ),
    ("legacy", {"dispatch_backend": "flydsl"}, "does not match"),
)


def _fields(dispatch, combine, wire, step_wire):
    """The recipe fields a pairing names."""
    wires = ("bf16",) if step_wire == "bf16" else ("bf16", step_wire)
    return {
        "dispatch": dispatch,
        "combine": combine,
        "dispatch_wire": wire,
        "combine_wires": wires,
    }


class _Checks:
    """Rank-local checks raised on every rank together, so a rank that fails
    one does not leave its peers inside a collective."""

    def __init__(self, label):
        self.label, self.failed = label, []

    def __call__(self, ok, what):
        if not ok:
            self.failed.append(what)

    def settle(self):
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, self.failed)
        self.failed = []
        bad = {rank: failed for rank, failed in enumerate(gathered) if failed}
        if bad:
            raise AssertionError(f"{self.label}: {bad}")


def _skip_reason() -> str | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return "needs two or more GPUs"
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        return f"MegaMoEGfx1250 is gfx1250-only, this is {get_gfx()}"
    try:
        import mori.cco  # noqa: F401
        from mori.ops.dispatch_combine_v2 import ep_plans  # noqa: F401
    except (ImportError, OSError) as error:
        return f"needs mori with cco and dispatch_combine_v2 ({error})"
    return None


class _LoggedClose:
    """Wraps a cco handle so close() is recorded in ``log`` before it runs."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def close(self):
        self._log.append(self._name)
        self._inner.close()


class _CountingCommunicator:
    """Forwards to the real communicator, counting alloc_mem and logging closes."""

    def __init__(self, inner):
        self._inner = inner
        self.allocs = 0
        self.alloc_bytes = []
        self.close_log = []

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def alloc_mem(self, nbytes):
        self.allocs += 1
        self.alloc_bytes.append(nbytes)
        return _LoggedClose(self._inner.alloc_mem(nbytes), "memory", self.close_log)

    def register_window(self, ptr, nbytes):
        return _LoggedClose(
            self._inner.register_window(ptr, nbytes), "window", self.close_log
        )


class _Ctx:
    def __init__(self, rank, world, comm, device):
        import bench_mega_moe as B

        self.B = B
        self.rank, self.world, self.comm, self.device = rank, world, comm, device
        self.experts = EXPERTS_PER_RANK * world
        # The expert-GEMM fields are the same for every quant key.
        self.spec = B.resolve_spec(QUANT_KEY["fp4"])
        w1, w2, _, _ = B.make_shared_weights(
            self.experts, HIDDEN, INTER, torch.bfloat16, device
        )
        lo, hi = rank * EXPERTS_PER_RANK, (rank + 1) * EXPERTS_PER_RANK
        q1, s1, q2, s2 = B.raw_quant_weights(
            w1[lo:hi].contiguous(), w2[lo:hi].contiguous()
        )
        self.weights = {
            key: B.shuffle_group(q1, s1, q2, s2, B.resolve_spec(key), EXPERTS_PER_RANK)
            for key in set(QUANT_KEY.values())
        }
        self.ids, self.wts = B.make_routings(
            1, TOKENS, self.experts, TOPK, device, seed=11
        )[0]
        gen = torch.Generator().manual_seed(500 + rank)
        x = torch.randn(TOKENS, HIDDEN, generator=gen).to(torch.bfloat16)
        self.x = x.to(device)
        ref = B.RefModel(w1, w2, None, None, self.spec, device)
        self.ref = (self.x.float() + ref.layer(self.x.float(), self.ids, self.wts)).to(
            torch.bfloat16
        )

    def recipe(self, **fields):
        from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import (
            ExpertComputeConfig,
            MegaMoERecipe,
        )

        compute = ExpertComputeConfig(
            activation=self.spec["activation"],
            gate_mode=self.spec["gate_mode"],
            quant_type=self.spec["aiter_qtype"],
        )
        return MegaMoERecipe(
            **{**_fields("mori", "fused_scatter", "fp4", "bf16"), **fields},
            compute=compute,
        )

    def geometry(self):
        from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGeometry

        return MegaMoEGeometry(
            rank=self.rank,
            world_size=self.world,
            model_dim=HIDDEN,
            inter_dim=INTER,
            experts=self.experts,
            topk=TOPK,
            max_tokens_per_rank=TOKENS,
        )

    def build(self, communicator=None, **fields):
        from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250

        return MegaMoEGfx1250.from_recipe(
            self.recipe(**fields),
            self.geometry(),
            communicator=self.comm if communicator is None else communicator,
        )

    def build_legacy(self, communicator=None, **overrides):
        """The deprecated keyword constructor; its warning is the caller's to
        check."""
        from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250

        kw = {
            "communicator": self.comm if communicator is None else communicator,
            "rank": self.rank,
            "world_size": self.world,
            "model_dim": HIDDEN,
            "inter_dim": INTER,
            "experts": self.experts,
            "topk": TOPK,
            "max_tokens_per_rank": TOKENS,
            "activation": self.spec["activation"],
            "gate_mode": self.spec["gate_mode"].value,
            "quant_type": self.spec["aiter_qtype"],
            "dispatch_wire": "fp4",
            "stage1_fused": False,
            "stage2_fused": True,
            "combine_quant": "none",
        }
        kw.update(overrides)
        return MegaMoEGfx1250(**kw)

    def layer(self, mega, combine_wire=None, **step):
        w1, w2, w1s, w2s = self.weights[QUANT_KEY[mega.recipe.dispatch_wire]]
        y = mega(
            self.B._rmsnorm(self.x),
            self.wts,
            self.ids,
            w1=w1,
            w2=w2,
            w1_scale=w1s,
            w2_scale=w2s,
            combine_wire=combine_wire,
            **step,
        )
        return self.x + y


def _check_pairings(ctx):
    """Each pairing against the reference, then a graph replay against eager."""
    for label, *pairing in PAIRINGS:
        wire, step_cq = pairing[2], pairing[-1]
        check = _Checks(label)
        mega = ctx.build(**_fields(*pairing))
        try:
            out = ctx.layer(mega, step_cq).clone()
            torch.cuda.synchronize()
            diff = ctx.B._calc_diff(out, ctx.ref)
            tol = ctx.B.default_logits_tol(QUANT_KEY[wire], LEGACY_QUANT[step_cq], 1)
            check(diff <= tol, f"diff {diff:.5f} > tol {tol:.5f}")
            check.settle()

            ctx.comm.barrier()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                ctx.layer(mega, step_cq)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            ctx.comm.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = ctx.layer(mega, step_cq)
            graph.replay()
            torch.cuda.synchronize()
            check(torch.equal(captured, out), "graph replay != eager")
            check.settle()
            del graph
        finally:
            torch.cuda.synchronize()
            ctx.comm.barrier()
            mega.close()
            ctx.comm.barrier()


def _check_rejected_before_alloc(ctx):
    check = _Checks("rejected before alloc")
    for entry, overrides, message in REJECTED:
        counting = _CountingCommunicator(ctx.comm)
        build = ctx.build if entry == "recipe" else ctx.build_legacy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                build(communicator=counting, **overrides).close()
        except ValueError as error:
            check(message in str(error), f"{overrides}: {error}")
        else:
            check(False, f"{overrides} was accepted")
        check(counting.allocs == 0, f"{overrides}: allocated before failing")
        check.settle()


def _check_unwind(ctx):
    """A failure after the arena is allocated releases the arena and the plans."""
    from mori.ops.dispatch_combine_v2 import ep_plans

    counting = _CountingCommunicator(ctx.comm)
    original = ep_plans.EpCombinePlan.bind

    def bind(self, **kwargs):
        raise RuntimeError("injected bind failure")

    ep_plans.EpCombinePlan.bind = bind
    try:
        ctx.build(communicator=counting, combine="mori").close()
    except RuntimeError as error:
        assert "injected" in str(error), error
    else:
        raise AssertionError("the injected failure did not surface")
    finally:
        ep_plans.EpCombinePlan.bind = original
        ctx.comm.barrier()
    log = counting.close_log
    assert counting.allocs == 1, counting.allocs
    assert log == ["window", "memory"], log


def _check_zero_copy(ctx):
    """With mori's combine, fused_moe writes its rows straight into comb_inp."""
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import mega_moe as mm

    mega = ctx.build(combine="mori")
    seen = {}
    real = mm.fused_moe

    def spy(*args, **kwargs):
        seen["output"] = kwargs.get("output")
        out = real(*args, **kwargs)
        seen["returned"] = out
        return out

    mm.fused_moe = spy
    try:
        ctx.layer(mega)
        torch.cuda.synchronize()
        staging = mega._arena.local_ptr("comb_inp")
        assert seen["output"] is not None
        assert seen["output"].data_ptr() == staging
        assert seen["returned"].data_ptr() == staging
    finally:
        mm.fused_moe = real
        ctx.comm.barrier()
        mega.close()
        ctx.comm.barrier()


def _check_close_order(ctx):
    from mori.ops.dispatch_combine_v2 import ep_plans

    counting = _CountingCommunicator(ctx.comm)
    log = counting.close_log
    patched = {}
    for cls in (ep_plans.EpDispatchPlan, ep_plans.EpCombinePlan):
        original = cls.close

        def close(self, _original=original, _name=cls.__name__):
            if getattr(self, "_handle", None) is not None:
                log.append(_name)
            _original(self)

        patched[cls] = original
        cls.close = close
    try:
        mega = ctx.build(communicator=counting, combine="mori")
        ctx.layer(mega)
        torch.cuda.synchronize()
        ctx.comm.barrier()
        mega.close()
        mega.close()  # a second close is a no-op
        ctx.comm.barrier()
    finally:
        for cls, original in patched.items():
            cls.close = original
    assert log.count("window") == 1 and log.count("memory") == 1, log
    window = log.index("window")
    assert all(i < window for i, n in enumerate(log) if n.startswith("Ep")), log
    assert window < log.index("memory"), log


def _check_atom_surface(ctx):
    """ATOM's Triton experts drive dispatch and combine themselves."""
    mega = ctx.build()
    check = _Checks("ATOM surface")
    try:
        mega._atom_per_rank_size = 1  # ATOM stamps its own attributes on
        cfg = mega._config
        check(
            (cfg.rank, cfg.world_size, cfg.hidden_dim) == (ctx.rank, ctx.world, HIDDEN),
            "rank, world_size, hidden_dim",
        )
        check(
            (cfg.max_tokens_per_rank, cfg.topk) == (TOKENS, TOPK),
            "max_tokens_per_rank, topk",
        )
        check(cfg.stage1_fused is False and cfg.stage2_fused is True, "stage flags")
        check(cfg.dispatch_wire == "fp4" and cfg.is_quant_dispatch_wire, "wire")
        check(cfg.dispatch_backend == "mori", "dispatch_backend")
        check(
            cfg.combine_slot_stride_bytes == 1 << (HIDDEN * 2 - 1).bit_length(),
            "combine_slot_stride_bytes",
        )
        check(mega._arena.local_ptr("comb_inp") > 0, "_arena")
        check.settle()
        recv_x, recv_w, recv_ids, total, routing = mega._dispatch(
            ctx.B._rmsnorm(ctx.x), ctx.wts, ctx.ids
        )
        check(recv_x.shape == (ctx.world * TOKENS, HIDDEN // 2), "recv_x")
        check(
            recv_w.shape == recv_ids.shape == (ctx.world * TOKENS, TOPK),
            "recv_w, recv_ids",
        )
        check(total.shape == (1,) and routing.token_count == TOKENS, "routing")
        scales = mega._recv_dispatch_scales()
        check(scales is not None and scales.shape[0] == ctx.world * TOKENS, "scales")
        check.settle()
        out = mega._combine(routing)
        torch.cuda.synchronize()
        check(out.shape == (TOKENS, HIDDEN) and out.dtype == torch.bfloat16, "combine")
        check(mega.max_tokens_per_rank == TOKENS and mega.topk == TOPK, "facade")
        check.settle()
    finally:
        torch.cuda.synchronize()
        ctx.comm.barrier()
        mega.close()
        ctx.comm.barrier()


def _check_keyword_constructor(ctx):
    """The deprecated keyword constructor builds the recipe its flags name, and
    runs bit for bit as from_recipe does; combine_quant= steps as combine_wire=."""
    for label, dispatch, combine, wire, step_wire in PAIRINGS:
        cq = LEGACY_QUANT[step_wire]
        check = _Checks(label)
        outs = []
        for entry in ("legacy", "recipe"):
            if entry == "legacy":
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", DeprecationWarning)
                    mega = ctx.build_legacy(
                        stage1_fused=dispatch == "compact",
                        stage2_fused=combine == "fused_scatter",
                        dispatch_wire=wire,
                        combine_quant=cq,
                    )
                check(
                    any(issubclass(w.category, DeprecationWarning) for w in caught),
                    "no DeprecationWarning",
                )
                check(
                    mega.recipe
                    == ctx.recipe(**_fields(dispatch, combine, wire, step_wire)),
                    f"built {mega.recipe}",
                )
                step = {"combine_quant": None if cq == "none" else cq}
            else:
                mega = ctx.build(**_fields(dispatch, combine, wire, step_wire))
                step = {"combine_wire": step_wire}
            try:
                check.settle()
                outs.append(ctx.layer(mega, **step).clone())
                torch.cuda.synchronize()
            finally:
                ctx.comm.barrier()
                mega.close()
                ctx.comm.barrier()
        check(torch.equal(outs[0], outs[1]), "keyword != from_recipe")
        check.settle()


def _check_a1_scale_refused(ctx):
    """The fp4 wire makes its own scales; a caller's would be dropped."""
    mega = ctx.build()
    try:
        ctx.layer(mega, a1_scale=torch.ones(1, device=ctx.device))
    except ValueError as error:
        assert "a1_scale is produced by the quantizing dispatch wire" in str(error)
    else:
        raise AssertionError("a1_scale was accepted")
    finally:
        ctx.comm.barrier()
        mega.close()
        ctx.comm.barrier()


def _check_vmm_bound(ctx):
    """required_vmm_bytes is the arena's allocation as cco rounds it."""
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250

    granularity = 2 << 20
    for label, *pairing in PAIRINGS:
        check = _Checks(label)
        counting = _CountingCommunicator(ctx.comm)
        mega = ctx.build(communicator=counting, **_fields(*pairing))
        try:
            (allocated,) = counting.alloc_bytes
            need = MegaMoEGfx1250.required_vmm_bytes(mega.recipe, mega.geometry)
            rounded = -(-allocated // granularity) * granularity
            check(need == rounded, f"{need} != {rounded} ({allocated})")
            bound = MegaMoEGfx1250.required_vmm_bytes(
                world_size=ctx.world,
                hidden_dim=HIDDEN,
                max_tokens_per_rank=TOKENS,
                experts=ctx.experts,
                topk=TOPK,
                stage1_fused=pairing[0] == "compact",
            )
            check(bound >= need, f"keyword bound {bound} < {need}")
        finally:
            ctx.comm.barrier()
            mega.close()
            ctx.comm.barrier()
        check.settle()



def _check_pull(
    ctx, *, a_preshuffled=True, fused=False, expected_cluster=None,
    expected_affinity=None, ready_first=False, ready_probe=None,
    expected_producer_group=None,
    expected_tile_ready=None,
    src_rank_shuffle=False,
):
    """Validate remote FP4 bytes, inverse maps, and changing graph inputs."""
    from aiter.test_common import checkAllclose

    fields = _fields("compact", "fused_scatter", "fp4", "bf16")
    check = _Checks("compact pull/preshuffle")
    saved_ids, saved_x = ctx.ids.clone(), ctx.x.clone()
    saved_weights = ctx.wts.clone()
    push = pull = None
    try:
        with mock.patch.dict(os.environ, {
            "AITER_MEGA_S1F": "0", "AITER_MEGA_DISPATCH_PULL": "0",
            "AITER_MEGA_S1F_PULL": "0",
        }):
            push = ctx.build(**fields)
        with mock.patch.dict(os.environ, {
            "AITER_MEGA_S1F": "1" if fused else "0",
            "AITER_MEGA_DISPATCH_PULL": "1",
            "AITER_MEGA_S1F_PULL": "1" if fused else "0",
            "AITER_MEGA_PULL_TDM_GATHER": "1" if fused else os.environ.get(
                "AITER_MEGA_PULL_TDM_GATHER", "0"),
        }):
            pull = ctx.build(**fields)
        check(pull._ep._dispatch._pull is not None, "pull was not selected")
        check(pull._ep._dispatch._pull.fused == fused, "wrong fusion mode")
        if ready_first:
            check(pull._ep._dispatch._pull.ready_first, "ready-first not selected")
        if expected_producer_group is not None:
            check(pull._ep._dispatch._pull.producer_group == expected_producer_group,
                  "cooperative producer mode not selected")
        if os.environ.get("AITER_MOE_S1F_CLUSTER_LOCAL", "0") == "1":
            check(pull._ep._dispatch._pull.cluster_local,
                  "cluster-local scheduler not selected")
        check.settle()
        # Warm the same addresses used in capture. The input contents change
        # afterward, exercising generations and stale padding/empty experts.
        if ready_first:
            # First launch has no old A flags. Give rank0 work independent of
            # the delayed rank so the readiness test can observe real progress.
            own = (
                torch.arange(TOKENS * TOPK, device=ctx.device)
                % EXPERTS_PER_RANK + ctx.rank * EXPERTS_PER_RANK
            ).reshape_as(ctx.ids)
            ctx.ids.copy_(own)
        ctx.layer(push, "bf16")
        ctx.layer(pull, "bf16")
        if ready_first:
            if ready_probe is not None:
                ready_probe(pull)
            ctx.ids.copy_(saved_ids)
        if expected_affinity is not None:
            selected = pull._ep._dispatch._pull.affinity_selection
            check(selected == expected_affinity,
                  f"affinity {selected} != {expected_affinity}")
            check.settle()
        if expected_cluster is not None:
            selected = pull._ep._dispatch._pull.cluster_selection
            check(selected[:2] == expected_cluster,
                  f"cluster {selected} != {expected_cluster}")
            check.settle()
        tile_flags = None
        if expected_tile_ready is not None:
            helper = pull._ep._dispatch._pull
            configs = [dict(key[-1]) for key in helper._fused_launches]
            check(bool(configs), "missing S1F compiled configuration")
            check(all(cfg.get("tile_ready") == expected_tile_ready for cfg in configs),
                  "wrong M256 ready selection")
            check.settle()
            if expected_tile_ready:
                from aiter.ops.flydsl.kernels.mega_moe_gfx1250.arena import _from_gpu_ptr
                from aiter.ops.flydsl.kernels.mega_moe_gfx1250.s1f_stage1 import FLAG_STRIDE

                tile_flags = _from_gpu_ptr(
                    helper.arena.local_ptr("pull_tile_flag"),
                    (helper.tile_info.numel(), FLAG_STRIDE // 4), torch.int32,
                )
        torch.cuda.synchronize()
        ctx.comm.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = ctx.layer(pull, "bf16")
        for iteration in range(6 if ready_first or src_rank_shuffle else 4):
            if (ready_first and iteration == 2) or (src_rank_shuffle and iteration == 4):
                # Disjoint source/destination ownership and empty segments.
                hot = (
                    torch.arange(TOPK, device=ctx.device, dtype=torch.int32)
                    + ctx.rank * EXPERTS_PER_RANK
                )
                ctx.ids.copy_(hot.expand_as(ctx.ids))
            elif (ready_first and iteration == 3) or (src_rank_shuffle and iteration == 5):
                # Unequal, non-M16 source boundaries within otherwise hot
                # experts. Each rank contributes a different random tail.
                hot = torch.arange(TOPK, device=ctx.device, dtype=torch.int32)
                ctx.ids.copy_(hot.expand_as(ctx.ids))
                tail = min(TOKENS, 3 + ctx.rank * 7)
                ctx.ids[-tail:].copy_(saved_ids[-tail:])
            elif iteration % 2:
                hot = torch.arange(TOPK, device=ctx.device, dtype=torch.int32)
                ctx.ids.copy_(hot.expand_as(ctx.ids))
            else:
                ctx.ids.copy_(saved_ids)
            ctx.x.copy_(torch.roll(saved_x, iteration * 17, dims=1))
            ctx.wts.copy_(torch.roll(saved_weights, iteration, dims=1))
            expected = ctx.layer(push, "bf16").clone()
            if tile_flags is not None:
                # A new generation must be published; unused M64/source slots
                # must stay untouched, so the legacy protocol cannot pass.
                tile_flags.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            err = checkAllclose(
                expected.float(), captured.float(), rtol=1e-2, atol=1e-2,
                msg=f"pull vs push iteration {iteration}",
            )
            check(err == 0, f"pull output mismatch {err}")

            stage = pull._ep._dispatch
            helper = stage._pull
            quant, _ = stage._quantize(ctx.B._rmsnorm(ctx.x), TOKENS)
            torch.cuda.synchronize()
            local = (
                quant.view(torch.uint8).cpu(),
                stage._quant_scales.view(torch.uint8).cpu(),
                ctx.wts.view(torch.int32).cpu().reshape(-1),
                ctx.ids.cpu().reshape(-1),
            )
            sources = [None] * ctx.world
            dist.all_gather_object(sources, local)
            counts = helper.masked_m.cpu().tolist()
            ends = helper.psum.cpu().tolist()
            if tile_flags is not None:
                flags = tile_flags.cpu()
                m_tiles = (ends[-1] + 255) // 256
                check(bool((flags[:m_tiles * 4:4, ctx.rank] != -1).all()),
                      "missing M256 generation publication")
                flags[:m_tiles * 4:4, ctx.rank] = -1
                check(bool((flags == -1).all()),
                      "M256 ready wrote an M64 or foreign-source flag")
            valid = torch.cat([
                torch.arange(end - count, end) for count, end in zip(counts, ends)
            ]).long()
            if src_rank_shuffle:
                full_map = stage._ep_rowmap.cpu()
                valid = torch.nonzero(full_map[:ends[-1], 0] >= 0).flatten()
                check(helper.src_rank_shuffle, "source shuffle not selected")
                for begin in range(0, ends[-1], 256):
                    routes = full_map[begin:begin + 256, 0]
                    routes = routes[routes >= 0]
                    if routes.numel():
                        owners = routes // (TOKENS * TOPK)
                        check(bool((owners == owners[0]).all()),
                              f"M256 at {begin} crosses source ranks")
                for source in range(ctx.world):
                    expect = torch.nonzero(
                        sources[source][3] // EXPERTS_PER_RANK == ctx.rank
                    ).flatten()
                    received = full_map[valid, 0].long()
                    received = received[
                        received // (TOKENS * TOPK) == source
                    ] % (TOKENS * TOPK)
                    check(torch.equal(received.sort().values, expect),
                          f"source {source}: missing or duplicate route")
            if fused:
                # GEMM1 can use a smaller tile than the plan/GEMM2 alignment.
                # Metadata must count actual rows, excluding alignment holes.
                metadata = helper.tile_info.cpu().to(torch.int64)
                expected_counts = torch.zeros_like(metadata)
                for count, end in zip(counts, ends):
                    start = end - count
                    for tile in range(start // 64, (end + 63) // 64):
                        if tile < metadata.numel():
                            expected_counts[tile] += max(
                                0, min(end, (tile + 1) * 64) - max(start, tile * 64)
                            )
                if src_rank_shuffle:
                    expected_counts = torch.bincount(
                        valid // 64, minlength=metadata.numel()
                    ).to(torch.int64)
                check(torch.equal((metadata >> 16) & 0xFFFF, expected_counts),
                      "pull tile metadata includes padding or misses live rows")
                check(torch.equal((metadata & 0xFF) != 0, expected_counts > 0),
                      "empty pull tiles must not wait for producer flags")
                if os.environ.get("AITER_MOE_S1F_READY_QUEUE", "0") == "1":
                    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import ready_queue

                    check(helper.ready_queue, "ready queue not selected")
                    configs = [dict(key[-1]) for key in helper._fused_launches]
                    tile_sizes = {cfg["tile_m"] for cfg in configs}
                    check(len(tile_sizes) == 1, "ambiguous queue tile size")
                    tile_m = tile_sizes.pop()
                    sub = tile_m // 64
                    m_tiles = (ends[-1] + tile_m - 1) // tile_m
                    cap = metadata.numel()
                    queue = helper.affinity_heads.cpu()[ready_queue.AFFINITY_WORDS:]
                    check(int(queue[ready_queue.TAIL]) == m_tiles,
                          "ready queue lost or duplicated an A tile")
                    entries = queue[ready_queue.COUNTS + cap:][:m_tiles]
                    check(torch.equal(entries.sort().values, torch.arange(1, m_tiles + 1)),
                          "ready queue did not publish every live A tile exactly once")
                    expected_parts = torch.zeros(m_tiles, dtype=torch.int32)
                    for part in range(sub):
                        live = expected_counts[part:m_tiles * sub:sub] > 0
                        expected_parts[:live.numel()] += live.to(torch.int32)
                    completed = queue[ready_queue.COUNTS:][:m_tiles]
                    check(torch.equal(completed, expected_parts),
                          "ready queue completion count includes padding or duplicates")
                if ready_first:
                    # Grade local-only chunk intervals against actual owners,
                    # independently of histogram-derived planner bounds.
                    full_map = stage._ep_rowmap.cpu()
                    max_routes = stage.compact.max_routes
                    owners = full_map[valid, 0].long() // max_routes
                    chunks = valid // 16
                    extent = metadata.numel() * 4
                    live = torch.bincount(chunks, minlength=extent).reshape(-1, 4)
                    remote = torch.bincount(
                        chunks[owners != ctx.rank], minlength=extent
                    ).reshape(-1, 4)
                    local = (live > 0) & (remote == 0)
                    index = torch.arange(4).expand_as(local)
                    first = torch.where(local, index, 4).amin(dim=1)
                    end = torch.where(local, index + 1, 0).amax(dim=1)
                    expected = first | (end << 16)
                    actual = helper.tile_local_chunks.cpu().to(torch.int64)
                    check(torch.equal(actual, expected),
                          "local M16 interval disagrees with received rows")
            rowmap = stage._ep_rowmap.cpu()[valid]
            if a_preshuffled:
                packed = helper.payload.cpu().reshape(-1, HIDDEN // 32, 16, 16)
                plain = packed.permute(0, 2, 1, 3).reshape(-1, HIDDEN // 2)
            else:
                plain = helper.payload.cpu().reshape(-1, HIDDEN // 2)
            actual_payload = plain[valid]
            actual_scales = helper.scales.cpu()[valid]
            for source in range(ctx.world):
                mask = rowmap[:, 0] // (TOKENS * TOPK) == source
                routes = rowmap[mask, 0] % (TOKENS * TOPK)
                tokens = routes // TOPK
                check(torch.equal(actual_payload[mask], sources[source][0][tokens]),
                      f"source {source}: pulled FP4 payload differs")
                check(torch.equal(actual_scales[mask], sources[source][1][tokens]),
                      f"source {source}: pulled scales differ")
                check(torch.equal(rowmap[mask, 1], sources[source][2][routes]),
                      f"source {source}: routing weight bits differ")
            if iteration == 0:
                # The original single-GPU entry point must keep working after
                # its payload operations are shared with remote pull dispatch.
                import flydsl.expr as fx

                from aiter.ops.flydsl.kernels.moe_apreshuffled_quant_scatter import (
                    build_moe_apre_scatter_module,
                )
                from aiter.ops.flydsl.kernels.tensor_shim import (
                    _run_compiled,
                    ptr_arg,
                )

                rows = stage._step.rows
                source_payload = torch.cat([s[0] for s in sources]).to(ctx.device)
                source_scales = torch.cat([s[1] for s in sources]).to(ctx.device)
                local_map = torch.full(
                    (rows,), -1, dtype=torch.int32, device=ctx.device
                )
                local_map[valid.to(ctx.device)] = (
                    rowmap[:, 0] // TOPK
                ).to(ctx.device)
                local_payload = torch.empty_like(helper.payload[:rows])
                local_scales = torch.empty_like(helper.scales[:rows])
                _run_compiled(
                    build_moe_apre_scatter_module(HIDDEN, 1),
                    ptr_arg(source_payload.reshape(-1)),
                    ptr_arg(source_scales.reshape(-1)),
                    ptr_arg(local_payload.reshape(-1)),
                    ptr_arg(local_scales.reshape(-1)),
                    ptr_arg(local_map),
                    TOKENS * ctx.world,
                    rows,
                    rows // 16,
                    fx.Stream(torch.cuda.current_stream()),
                )
                local_plain = local_payload.cpu().reshape(
                    -1, HIDDEN // 32, 16, 16
                )
                local_plain = local_plain.permute(0, 2, 1, 3).reshape(
                    -1, HIDDEN // 2
                )
                local_scale_plain = local_scales.cpu().reshape(
                    -1, HIDDEN // 128, 16, 4
                )
                local_scale_plain = local_scale_plain.permute(0, 2, 1, 3).reshape(
                    -1, HIDDEN // 32
                )
                check(torch.equal(local_plain[valid], actual_payload),
                      "original local preshuffle payload differs from pull")
                check(torch.equal(local_scale_plain[valid], actual_scales),
                      "original local preshuffle scales differ from pull")
            check.settle()
        del graph
    finally:
        ctx.ids.copy_(saved_ids)
        ctx.x.copy_(saved_x)
        ctx.wts.copy_(saved_weights)
        torch.cuda.synchronize()
        ctx.comm.barrier()
        for mega in (pull, push):
            if mega is not None:
                mega.close()
        ctx.comm.barrier()

def _check_pull_s1f(ctx):
    """Exercise fused pull/GEMM1 with changing inputs and empty ranks."""
    _check_pull(ctx, fused=True)


def _check_pull_s1f_affinity(ctx):
    """Check mixed 3P/5C clusters, stealing, tails and changing graph inputs."""
    with mock.patch.dict(os.environ, {
        "AITER_MOE_S1F_AFFINITY": "1",
        "AITER_MEGA_S1F_PULL_PRODUCERS": "96",
        "AITER_MEGA_S1F_PULL_CLUSTER": "0",
    }):
        _check_pull(ctx, fused=True, expected_affinity=(8, 3))


def _check_pull_s1f_ready_first(ctx, *, expected_tile_ready=None):
    """Check local-first M16 tails, empty segments and readiness generations."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import mori.cco.device.flydsl as cco

    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import s1f_pull
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.arena import _from_gpu_ptr
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.stages import pull as pull_module
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.stages.pull import PullDispatch

    compile_original = s1f_pull.compile_s1f_pull
    init_original = PullDispatch.__init__
    progress_views = []
    probe_storage = []
    delayed_launchers = {}
    run_original = pull_module._run_compiled
    producers = int(os.environ.get("AITER_MEGA_S1F_PULL_PRODUCERS", "96"))
    expected_affinity = s1f_pull.pull_affinity_geometry(256, producers)
    probe_flag_offset = 0

    def with_progress(self, dispatch):
        init_original(self, dispatch)
        self.send_base = torch.cat((
            self.send_base,
            torch.zeros(3, dtype=torch.int32, device=self.send_base.device),
        ))
        progress_views.append(self.send_base[-3:])
        probe_storage.append(self.send_base)

    def arm_probe(pull):
        # Every rank must finish JIT and its first execution before timing a
        # source delay; otherwise compilation skew can exhaust the probe.
        torch.cuda.synchronize()
        ctx.comm.barrier()
        previous = [None]
        if ctx.rank == 0:
            address = (pull._ep._dispatch._pull.arena.local_ptr("pull_tile_flag")
                       + probe_flag_offset)
            previous[0] = _from_gpu_ptr(address, (1,), torch.int32).item()
        dist.broadcast_object_list(previous, src=0)
        progress_views[0].copy_(torch.tensor(
            [1, 0, previous[0]], dtype=torch.int32, device=ctx.device
        ))
        torch.cuda.synchronize()
        ctx.comm.barrier()
        ctx.layer(pull, "bf16")
        torch.cuda.synchronize()
        ctx.comm.barrier()

    def dsl_types(**annotations):
        # This module postpones annotations; nested FX imports cannot be
        # resolved by FlyDSL's signature parser. Supply actual runtime types.
        def annotate(function):
            function.__annotations__.update(annotations)
            return function
        return annotate

    def compile_delayed(**kwargs):
        nonlocal probe_flag_offset
        # Cluster 1's first M tile belongs to a nonleader producer, which must
        # progress while cluster 0's producer leader waits on the delayed rank.
        probe_flag_offset = kwargs["tile_m"] // 64 * s1f_pull.FLAG_STRIDE
        ready, fused = compile_original(**kwargs)
        if ctx.rank != ctx.world - 1:
            return ready, fused
        flag_offset = kwargs["off_tile_flag"] + probe_flag_offset
        probe_offset = kwargs["world"] * kwargs["experts"]

        @flyc.kernel(known_block_size=(32, 1, 1))
        @dsl_types(arena=fx.Int64, bounds=fx.Tensor)
        def delay_source(arena: fx.Int64, bounds: fx.Tensor):
            if fx.thread_idx.x == 0:
                ptr = fx.PointerType.get(
                    fx.Int32.ir_type, fx.AddressSpace.Global, 4
                )
                if bounds[probe_offset] == 1:
                    # Rank0's flag must advance from the completed warmup
                    # generation before this source publishes readiness.
                    window = cco.CachedWindow(arena)
                    address = fx.inttoptr(ptr, window.base + flag_offset)
                    previous = bounds[probe_offset + 2]
                    value = previous
                    tries = fx.Int32(0)
                    while (value == previous) & (tries < 200000):
                        value = fx.generic_load(
                            address, memory_order=fx.AtomicOrdering.Acquire,
                            syncscope="",
                        )
                        fx.rocdl.s_sleep(1)
                        tries += 1
                    bounds[probe_offset + 1] = fx.Int32(value != previous)
                    bounds[probe_offset] = 2

        @flyc.jit
        @dsl_types(arena=fx.Int64, bounds=fx.Tensor, stream=fx.Stream)
        def delayed_entry(arena: fx.Int64, bounds: fx.Tensor, stream: fx.Stream):
            delay_source(arena, bounds).launch(
                grid=(1, 1, 1), block=(32, 1, 1), stream=stream
            )

        delayed_launchers[fused] = delayed_entry
        return ready, fused

    def run_with_delay(function, *args):
        delay = delayed_launchers.get(function)
        if delay is not None:
            # Delay the source before S1F publishes its epoch. Production has
            # no standalone ready kernel; the test-only delay remains bounded.
            run_original(delay, args[7], probe_storage[0], args[-1])
        return run_original(function, *args)

    with mock.patch.dict(os.environ, {
        "AITER_MOE_S1F_READY_FIRST": "1",
        "AITER_MOE_S1F_AFFINITY": "1",
        "AITER_MEGA_S1F_PULL_PRODUCERS": str(producers),
        "AITER_MEGA_S1F_PULL_CLUSTER": "0",
    }), mock.patch.object(
        s1f_pull, "compile_s1f_pull", compile_delayed
    ), mock.patch.object(PullDispatch, "__init__", with_progress), mock.patch.object(
        pull_module, "_run_compiled", run_with_delay
    ):
        _check_pull(
            ctx, fused=True, expected_affinity=expected_affinity, ready_first=True,
            ready_probe=arm_probe,
            expected_tile_ready=expected_tile_ready,
        )
    check = _Checks("ready-source progress before delayed source")
    if ctx.rank == ctx.world - 1:
        check(progress_views[0].cpu().tolist()[:2] == [2, 1],
              "producer waited for every rank")
    check.settle()


def _check_pull_s1f_ready_queue(ctx):
    """Exercise completion-driven scheduling across changing graph routing."""
    with mock.patch.dict(os.environ, {"AITER_MOE_S1F_READY_QUEUE": "1"}):
        _check_pull_s1f_ready_first(ctx)


def _check_pull_s1f_cluster_local(ctx):
    """Exercise hardware cluster rounds, partial batches and changing routing."""
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.s1f_pull import pull_affinity_geometry

    producers = int(os.environ.get("AITER_MEGA_S1F_PULL_PRODUCERS", "96"))
    with mock.patch.dict(os.environ, {
        "AITER_MOE_S1F_CLUSTER_LOCAL": "1",
        "AITER_MOE_S1F_READY_FIRST": "1",
        "AITER_MOE_S1F_READY_QUEUE": "0",
        "AITER_MOE_S1F_AFFINITY": "1",
        "AITER_MEGA_S1F_PULL_PRODUCERS": str(producers),
        "AITER_MEGA_S1F_PULL_CLUSTER": "0",
    }):
        _check_pull(
            ctx, fused=True, ready_first=True,
            expected_affinity=pull_affinity_geometry(256, producers),
        )


def _check_pull_s1f_producer_group(ctx):
    """Check four-TG publication, empty members and independent consumers."""
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    original_find = grouped._find_grouped_config

    def configured(**kwargs):
        row = dict(original_find(**kwargs) or {})
        # Scale loads must follow A-ready in S1F, including small CSV buckets.
        row["tdm_as_in_prologue"] = "0"
        return row

    # Pin the tiling so this collective test does not depend on CSV changes.
    # P100 also verifies that four-TG grouping does not require P % 8 == 0.
    producers = int(os.environ.get("CHECK_PRODUCERS", "100"))
    for ready_first in (False, True):
        with mock.patch.dict(os.environ, {
            "AITER_MOE_S1F_PRODUCER_GROUP": "4",
            "AITER_MOE_S1F_CLUSTER_LOCAL": "0",
            "AITER_MOE_S1F_READY_QUEUE": "0",
            "AITER_MOE_S1F_READY_FIRST": str(int(ready_first)),
            "AITER_MEGA_S1F_PULL_PRODUCERS": str(producers),
            "AITER_TDM_TILE_M": "256", "AITER_TDM_TILE_N": "256",
            "AITER_TDM_TILE_K": "256",
            # Small-token push skips A preshuffle and needs extra scale LDS.
            "AITER_TDM_NUM_BUFFERS": "4" if TOKENS >= 4096 else "2",
            "AITER_TDM_M_WARP": "2", "AITER_TDM_N_WARP": "2",
        }), mock.patch.object(grouped, "_find_grouped_config", configured):
            _check_pull(
                ctx, fused=True, ready_first=ready_first,
                expected_producer_group=4, expected_cluster=(1, 1),
                expected_affinity=(1, 0),
            )


def _check_pull_s1f_tile_ready(ctx, *, ready_first=False):
    """Check single M256 publication, graph generations and partial tiles."""
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    original_find = grouped._find_grouped_config

    def configured(**kwargs):
        row = dict(original_find(**kwargs) or {})
        row["tdm_as_in_prologue"] = "0"
        return row

    with mock.patch.dict(os.environ, {
        "AITER_MOE_S1F_TILE_READY": "1",
        "AITER_MOE_S1F_PRODUCER_GROUP": "1",
        "AITER_MOE_S1F_CLUSTER_LOCAL": "0",
        "AITER_MOE_S1F_READY_QUEUE": "0",
        "AITER_MOE_S1F_READY_FIRST": str(int(ready_first)),
        "AITER_MOE_S1F_AFFINITY": os.environ.get("CHECK_AFFINITY", "1"),
        "AITER_MEGA_S1F_PULL_PRODUCERS": "96",
        "AITER_MEGA_S1F_PULL_CLUSTER": "0",
        "AITER_TDM_TILE_M": "256", "AITER_TDM_TILE_N": "256",
        "AITER_TDM_TILE_K": "256",
        "AITER_TDM_NUM_BUFFERS": "4" if TOKENS >= 4096 else "2",
        "AITER_TDM_M_WARP": "2", "AITER_TDM_N_WARP": "2",
    }), mock.patch.object(grouped, "_find_grouped_config", configured):
        if ready_first:
            _check_pull_s1f_ready_first(ctx, expected_tile_ready=True)
        else:
            _check_pull(ctx, fused=True, expected_tile_ready=True)


def _check_pull_s1f_src_rank_shuffle(ctx):
    """Check source-homogeneous M256 tiles with arbitrary routing and padding."""
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    original_find = grouped._find_grouped_config

    def configured(**kwargs):
        row = dict(original_find(**kwargs) or {})
        row["tdm_as_in_prologue"] = "0"
        return row

    with mock.patch.dict(os.environ, {
        "AITER_MEGA_PULL_PLAN_PACK": "1",
        "AITER_MEGA_PULL_SRC_SHUFFLE": "1",
        "AITER_MOE_S1F_TILE_READY": "0",
        "AITER_MOE_S1F_READY_FIRST": "0",
        "AITER_MOE_S1F_PRODUCER_GROUP": "1",
        "AITER_MOE_S1F_CLUSTER_LOCAL": "0",
        "AITER_MOE_S1F_READY_QUEUE": "0",
        "AITER_MOE_S1F_AFFINITY": "1",
        "AITER_MEGA_S1F_PULL_PRODUCERS": "96",
        "AITER_MEGA_S1F_PULL_CLUSTER": "0",
        "AITER_TDM_TILE_M": "256", "AITER_TDM_TILE_N": "256",
        "AITER_TDM_TILE_K": "256",
        "AITER_TDM_NUM_BUFFERS": "4" if TOKENS >= 4096 else "2",
        "AITER_TDM_M_WARP": "2", "AITER_TDM_N_WARP": "2",
    }), mock.patch.object(grouped, "_find_grouped_config", configured):
        _check_pull(ctx, fused=True, src_rank_shuffle=True)


def _check_pull_s1f_tile_ready_leader(ctx):
    """Check M256 publication while the leader waits for a delayed source."""
    _check_pull_s1f_tile_ready(ctx, ready_first=True)


def _check_pull_s1f_clusters(ctx):
    """Exercise physical cluster roles, multicast, tails and graph generations."""
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.stages.pull import PullDispatch

    original = PullDispatch.gemm1
    for cm, cn in ((1, 2), (2, 2), (4, 2), (1, 4), (4, 4)):
        def configured(self, *args, **kwargs):
            kwargs.update(cluster_m=cm, cluster_n=cn)
            return original(self, *args, **kwargs)

        expected = (1, cn) if (2 * INTER // 256) % cn == 0 else (1, 1)
        with mock.patch.dict(os.environ, {
            "AITER_MEGA_S1F_PULL_CLUSTER": "1",
            "AITER_MEGA_S1F_PULL_PRODUCERS": "96",
        }), mock.patch.object(PullDispatch, "gemm1", configured):
            _check_pull(ctx, fused=True, expected_cluster=expected)
        if ctx.rank == 0:
            print(f"S1F cluster requested={cm}x{cn}, effective={expected}: passed",
                  flush=True)


def _check_pull_s1f_cluster_m_projection(ctx):
    """An MxN request projects onto the supported 1xN consumer layout."""
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.stages.pull import PullDispatch

    original = PullDispatch.gemm1

    def configured(self, *args, **kwargs):
        kwargs.update(cluster_m=4, cluster_n=4)
        return original(self, *args, **kwargs)

    with mock.patch.dict(os.environ, {
        "AITER_MEGA_S1F_PULL_CLUSTER": "1",
        "AITER_MEGA_S1F_PULL_PRODUCERS": "104",
    }), mock.patch.object(PullDispatch, "gemm1", configured):
        expected = (1, 4) if (2 * INTER // 256) % 4 == 0 else (1, 1)
        _check_pull(ctx, fused=True, expected_cluster=expected)


CHECKS = (
    _check_pull,
    _check_pull_s1f,
    _check_pull_s1f_affinity,
    _check_pull_s1f_ready_first,
    _check_pull_s1f_ready_queue,
    _check_pull_s1f_cluster_local,
    _check_pull_s1f_producer_group,
    _check_pull_s1f_tile_ready,
    _check_pull_s1f_tile_ready_leader,
    _check_pull_s1f_src_rank_shuffle,
    _check_pull_s1f_clusters,
    _check_pull_s1f_cluster_m_projection,
    _check_pairings,
    _check_rejected_before_alloc,
    _check_unwind,
    _check_zero_copy,
    _check_close_order,
    _check_atom_surface,
    _check_keyword_constructor,
    _check_a1_scale_refused,
    _check_vmm_bound,
)


def _worker(local_rank, world, port, failures):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(local_rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(local_rank),
        AITER_FORCE_A8W4="0",
        MORI_EP_TOKOFF_EXT="0",
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", rank=local_rank, world_size=world)
    from mori.cco import Communicator

    uid = [Communicator.get_unique_id() if local_rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = Communicator.init(world, local_rank, uid[0], per_rank_vmm=8 * 1024**3)
    comm.barrier()
    ctx = _Ctx(local_rank, world, comm, device)
    failed = []
    for check in CHECKS:
        try:
            check(ctx)
            status = "ok"
        except Exception:  # noqa: BLE001
            failed.append(check.__name__)
            status = "FAILED\n" + traceback.format_exc()
        if local_rank == 0:
            print(f"[{check.__name__}] {status}", flush=True)
        dist.barrier()
    comm.destroy()
    dist.destroy_process_group()
    failures[local_rank] = len(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    args = parser.parse_args()
    reason = _skip_reason()
    if reason is not None:
        print(f"SKIP test_mega_moe_gfx1250: {reason}")
        return 0
    world = min(args.world, torch.cuda.device_count())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    failures = mp.get_context("spawn").Manager().dict()
    mp.spawn(_worker, args=(world, port, failures), nprocs=world, join=True)
    bad = sum(failures.values())
    print("test_mega_moe_gfx1250:", "FAILED" if bad else "passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
