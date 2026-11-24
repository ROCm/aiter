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


CHECKS = (
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
