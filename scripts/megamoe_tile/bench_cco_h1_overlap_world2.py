# SPDX-License-Identifier: MIT
"""WORLD2 CCO/H1 control-overlap benchmark for the target MegaMoE shape.

This benchmark intentionally does not time token packing, expert sorting, or
route construction.  All A4W4 H1 inputs and the 64 KiB CCO payload are built
before the measured loops.  Each communication-bearing iteration measures:

    CCO post_dispatch (4 QPs, 8 x 2048 B per QP)
      -> wait for the peer's trailing per-QP generation
      -> system-scope publication of every H1 input-ready count

Credit return and request/ring-slot reclaim are required for correctness but
are outside the measured interval.  The four reported cases are:

* comm_only: real 64 KiB CCO send plus receive-ready/control publication;
* h1_only: ready-H1 with input-ready counts pre-published before timing;
* serial: comm/control then ready-H1 on one stream;
* overlap: ready-H1 is enqueued first on the compute stream, where it waits
  for flags published by CCO send/mark on the communication stream.

A delayed GPU event gate lets all measured work be enqueued before the common
start timestamp executes.  This avoids charging FlyDSL Python enqueue gaps to
the kernels.  For the two-stream case, elapsed time is the maximum of the
common-start-to-compute-end and common-start-to-communication-end spans.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass

import torch

from mori.cco import Communicator, UniqueId


NUM_QP = 4
BATCH_PER_QP = 8
SEGMENT_BYTES = 2048
TRAFFIC_BYTES = 64 * 1024
RING_DEPTH = 8
SLOT = 0

TOKENS = 8
D_HIDDEN = 3584
D_INTER = 384
EXPERTS = 56
TOPK = 16
BM = 32
MAX_SOURCE_TOKENS = 64
MAX_H1_N_BLOCKS = 3

EXPECTED_KERNEL = (
    "megamoe_tile_h1_ready_a4w4_silu_h3584_i384_e56_k16_"
    "bm32_wpe2_nt1_sc2_pw0"
)


def _bootstrap() -> tuple[int, int, UniqueId]:
    rank = int(os.environ["CCO_RANK"])
    world = int(os.environ["CCO_WORLD"])
    path = os.environ["CCO_UID_FILE"]
    while not os.path.exists(path) or os.path.getsize(path) != 128:
        time.sleep(0.05)
    with open(path, "rb") as handle:
        uid = UniqueId.from_bytes(handle.read())
    return rank, world, uid


def _payload(rank: int) -> tuple[int, ...]:
    """Return a rank-distinct, fixed 64 KiB payload."""

    prefix = (int(rank) + 1) << 56
    return tuple(prefix | index for index in range(TRAFFIC_BYTES // 8))


@dataclass(frozen=True)
class _H1Inputs:
    a1q: torch.Tensor
    a1ss: torch.Tensor
    w1: torch.Tensor
    w1_scale: torch.Tensor
    sorted_eids: torch.Tensor
    nvalid: torch.Tensor
    m_indices: torch.Tensor
    result_q: torch.Tensor
    result_s: torch.Tensor
    dummy_hidden: torch.Tensor
    active_m_tiles: int
    max_m_tiles: int


def _prepare_target_h1(rank: int, seed: int, device: torch.device) -> _H1Inputs:
    """Pre-pack the target H1 operands; no work here belongs to timing."""

    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.megamoe_tile import prepare_local_a4w4_weights
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    torch.manual_seed(int(seed) + int(rank))
    x = (torch.randn(TOKENS, D_HIDDEN, device=device) * 0.1).to(
        torch.bfloat16
    )
    w1 = (
        torch.randn(EXPERTS, 2 * D_INTER, D_HIDDEN, device=device) * 0.03
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(EXPERTS, D_HIDDEN, D_INTER, device=device) * 0.03
    ).to(torch.bfloat16)
    score = torch.rand(TOKENS, EXPERTS, device=device)
    values, ids = torch.topk(score, TOPK, dim=1)
    route_weights = torch.softmax(values, dim=1).float()

    prepared = prepare_local_a4w4_weights(w1, w2)
    sorted_ids, _, sorted_eids, nvalid, _ = moe_sorting(
        ids.to(torch.int32),
        route_weights,
        EXPERTS,
        D_HIDDEN,
        torch.bfloat16,
        BM,
        accumulate=False,
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(
        a1s.view(TOKENS, 1, D_HIDDEN // 32),
        sorted_ids,
        nvalid,
        TOKENS,
        BM,
    )
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()

    max_sorted = int(sorted_ids.shape[0])
    max_m_tiles = (max_sorted + BM - 1) // BM
    active_m_tiles = int(nvalid[0].item()) // BM
    if active_m_tiles <= 0:
        raise RuntimeError("target routing unexpectedly produced no H1 M tiles")
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((D_INTER // 32) + 7) // 8 * 8

    # Drop the BF16 source weights once the packed operands exist.  The
    # returned tensors below retain every object used by the H1 launch.
    del x, w1, w2, score, values, ids, route_weights, a1s, sorted_ids
    return _H1Inputs(
        a1q=a1q,
        a1ss=a1ss,
        w1=prepared.w1,
        w1_scale=prepared.w1_scale,
        sorted_eids=sorted_eids,
        nvalid=nvalid,
        m_indices=m_indices,
        result_q=torch.zeros(
            (max_sorted, D_INTER // 2), dtype=torch.uint8, device=device
        ),
        result_s=torch.zeros(
            scale_rows * scale_cols, dtype=torch.uint8, device=device
        ),
        dummy_hidden=torch.zeros(
            (TOKENS, D_HIDDEN), dtype=torch.bfloat16, device=device
        ),
        active_m_tiles=active_m_tiles,
        max_m_tiles=max_m_tiles,
    )


def _arm_event_gate(
    control_stream: torch.cuda.Stream,
    gate_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gate_repeats: int,
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """Queue a delayed common start and release event on ``control_stream``."""

    if gate_repeats <= 0:
        raise ValueError("gate_repeats must be positive")
    gate_a, gate_b, gate_out = gate_tensors
    start = torch.cuda.Event(enable_timing=True)
    release = torch.cuda.Event(enable_timing=False)
    with torch.cuda.stream(control_stream):
        # torch.cuda._sleep maps to an unsupported HIP runtime path on ROCm.
        # A few pre-warmed BF16 GEMMs keep the GPU busy while Python enqueues
        # every measured stream, then the common start/release events fire.
        for _ in range(gate_repeats):
            torch.mm(gate_a, gate_b, out=gate_out)
        start.record(control_stream)
        release.record(control_stream)
    return start, release


def _event_max_us(
    start: torch.cuda.Event, ends: tuple[torch.cuda.Event, ...]
) -> float:
    if not ends:
        raise ValueError("at least one completion event is required")
    for event in ends:
        event.synchronize()
    return max(float(start.elapsed_time(event) * 1000.0) for event in ends)


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_us": float(statistics.median(samples)),
        "min_us": float(min(samples)),
        "max_us": float(max(samples)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("MEGAMOE_OVERLAP_WARMUP", "5")),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=int(os.environ.get("MEGAMOE_OVERLAP_ITERS", "20")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("MEGAMOE_OVERLAP_WORKERS", "192")),
    )
    parser.add_argument(
        "--gate-repeats",
        type=int,
        default=int(os.environ.get("MEGAMOE_OVERLAP_GATE_REPEATS", "1")),
        help="Pre-warmed BF16 GEMMs before the common start event.",
    )
    parser.add_argument(
        "--gate-dim",
        type=int,
        default=int(os.environ.get("MEGAMOE_OVERLAP_GATE_DIM", "1024")),
    )
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.iters <= 0
        or args.workers <= 0
        or args.gate_repeats <= 0
        or args.gate_dim <= 0
    ):
        raise ValueError("warmup/workers must be non-negative and iters positive")

    rank, world, uid = _bootstrap()
    if world != 2:
        raise ValueError("CCO/H1 overlap benchmark requires exactly two ranks")
    for name, expected in (
        ("MEGAMOE_CCO_QP", NUM_QP),
        ("MEGAMOE_CCO_BATCH", BATCH_PER_QP),
        ("MEGAMOE_CCO_CHUNK", TRAFFIC_BYTES),
    ):
        if name in os.environ and int(os.environ[name]) != expected:
            raise ValueError(f"{name} must be {expected} for equal 64 KiB traffic")

    os.environ["MXFP4_G2_SPART"] = "0"
    os.environ["MXFP4_G2_BF16_LDS"] = "1"
    os.environ["MXFP4_G2_KSTATIC"] = "1"

    from aiter.ops.flydsl.kernels.megamoe_tile import cco
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_hier_epoch_module,
        compile_hier_stage1_ready_a4w4,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    gpu = int(os.environ.get("CCO_GPU", "0"))
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    peer = 1 - rank

    inputs = _prepare_target_h1(rank, args.seed, device)
    ready_h1 = compile_hier_stage1_ready_a4w4(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=EXPERTS,
        TOPK=TOPK,
        activation="silu",
        waves_per_eu_hint=2,
        wait_plan=False,
    )
    if ready_h1.kernel_name != EXPECTED_KERNEL:
        raise AssertionError(
            f"expected exact kernel {EXPECTED_KERNEL}, got {ready_h1.kernel_name}"
        )

    layout = HierCcoArenaLayout.create(
        ring_depth=RING_DEPTH,
        num_qp=NUM_QP,
        chunk_bytes=TRAFFIC_BYTES,
        max_m_tiles=inputs.max_m_tiles,
        max_source_tokens=MAX_SOURCE_TOKENS,
        max_h1_n_blocks=MAX_H1_N_BLOCKS,
    )
    sidecar = cco.CcoStage1Sidecar.create(
        layout,
        batch_per_qp=BATCH_PER_QP,
        segment_bytes=SEGMENT_BYTES,
        team=cco.TEAM_WORLD,
    )
    if sidecar.module.payload_bytes != TRAFFIC_BYTES:
        raise AssertionError("stage1 sidecar must carry exactly 64 KiB")
    reset_epoch, _, _ = build_hier_epoch_module(
        max_m_tiles=inputs.max_m_tiles,
        max_source_tokens=MAX_SOURCE_TOKENS,
    )

    setup_stream = torch.cuda.Stream(device=device)
    control_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    gate_dim = args.gate_dim
    gate_tensors = (
        torch.ones((gate_dim, gate_dim), dtype=torch.bfloat16, device=device),
        torch.ones((gate_dim, gate_dim), dtype=torch.bfloat16, device=device),
        torch.empty((gate_dim, gate_dim), dtype=torch.bfloat16, device=device),
    )
    with torch.cuda.stream(control_stream):
        torch.mm(gate_tensors[0], gate_tensors[1], out=gate_tensors[2])
    control_stream.synchronize()

    def launch_h1(ptrs, generation: int, stream: torch.cuda.Stream) -> None:
        ready_h1(
            ptrs.plan_ready,
            ptrs.h1_input_ready,
            ptrs.h1_input_expected,
            ptrs.h1_output_done,
            ptrs.h1_output_ready,
            generation,
            inputs.a1q.data_ptr(),
            inputs.a1ss.data_ptr(),
            inputs.w1.data_ptr(),
            inputs.w1_scale.data_ptr(),
            inputs.sorted_eids.data_ptr(),
            inputs.nvalid.data_ptr(),
            inputs.m_indices.data_ptr(),
            TOKENS,
            args.workers,
            inputs.result_q.data_ptr(),
            inputs.result_s.data_ptr(),
            inputs.dummy_hidden.data_ptr(),
            stream=stream,
        )

    generation = 0
    verified_payload = False

    with Communicator.init(
        world, rank, uid, per_rank_vmm=64 * 1024 * 1024
    ) as communicator:
        resources = cco.create_transport_resources(
            communicator,
            layout.total_bytes,
            num_qp=NUM_QP,
            team=cco.TEAM_WORLD,
        )
        arena_win = resources.window
        dev_comm = resources.dev_comm
        cco.zero_window(arena_win.local_ptr, layout.total_bytes)
        tx_ptr = arena_win.local_ptr + layout.ring_chunk_offset(
            "dispatch_tx", SLOT
        )
        rx_ptr = arena_win.local_ptr + layout.ring_chunk_offset(
            "dispatch_rx", SLOT
        )
        cco.write_window_u64(tx_ptr, _payload(rank))
        torch.cuda.synchronize(device)
        communicator.barrier()

        def setup_epoch(*, input_ready: bool):
            nonlocal generation
            generation += 1
            ptrs = layout.epoch_pointers(arena_win.local_ptr, generation)
            reset_epoch(
                ptrs.h1_input_expected,
                ptrs.h1_input_ready,
                ptrs.h1_output_done,
                ptrs.h2_output_done,
                ptrs.rank_route_expected,
                ptrs.rank_route_ready,
                inputs.active_m_tiles,
                TOKENS,
                1,
                stream=setup_stream,
            )
            sidecar.publish_plan_expected(
                generation,
                ptrs,
                inputs.active_m_tiles,
                expected_per_tile=1,
                stream=setup_stream,
            )
            setup_stream.synchronize()
            if input_ready:
                # Synchronous host-to-device copy is deliberately before the
                # start gate: h1_only measures compute, not flag publication.
                cco.write_window_u32(
                    ptrs.h1_input_ready, (1,) * inputs.active_m_tiles
                )
            return ptrs, generation

        # Compile/warm the exact H1 launch without communication.  This also
        # verifies that pre-published input counts let the consumer terminate.
        ptrs, warm_generation = setup_epoch(input_ready=True)
        launch_h1(ptrs, warm_generation, compute_stream)
        compute_stream.synchronize()
        communicator.barrier()

        def run_once(case: str) -> float:
            nonlocal verified_payload
            uses_comm = case in ("comm_only", "serial", "overlap")
            uses_h1 = case in ("h1_only", "serial", "overlap")
            ptrs, current_generation = setup_epoch(input_ready=case == "h1_only")
            communicator.barrier()

            start, release = _arm_event_gate(
                control_stream, gate_tensors, args.gate_repeats
            )
            ends: list[torch.cuda.Event] = []

            if case == "comm_only":
                comm_stream.wait_event(release)
                sidecar.post_dispatch(
                    dev_comm.ptr,
                    arena_win.handle,
                    arena_win.local_ptr,
                    peer,
                    SLOT,
                    current_generation,
                    stream=comm_stream,
                )
                sidecar.mark_chunk_ready(
                    arena_win.local_ptr,
                    SLOT,
                    current_generation,
                    ptrs,
                    0,
                    inputs.active_m_tiles,
                    delta=1,
                    stream=comm_stream,
                )
                comm_end = torch.cuda.Event(enable_timing=True)
                comm_end.record(comm_stream)
                ends.append(comm_end)
            elif case == "h1_only":
                compute_stream.wait_event(release)
                launch_h1(ptrs, current_generation, compute_stream)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_end.record(compute_stream)
                ends.append(compute_end)
            elif case == "serial":
                compute_stream.wait_event(release)
                sidecar.post_dispatch(
                    dev_comm.ptr,
                    arena_win.handle,
                    arena_win.local_ptr,
                    peer,
                    SLOT,
                    current_generation,
                    stream=compute_stream,
                )
                sidecar.mark_chunk_ready(
                    arena_win.local_ptr,
                    SLOT,
                    current_generation,
                    ptrs,
                    0,
                    inputs.active_m_tiles,
                    delta=1,
                    stream=compute_stream,
                )
                launch_h1(ptrs, current_generation, compute_stream)
                serial_end = torch.cuda.Event(enable_timing=True)
                serial_end.record(compute_stream)
                ends.append(serial_end)
            elif case == "overlap":
                # ROCm can serialize a later producer behind an already
                # resident spin-wait consumer. Queue the high-priority CCO
                # producer first; both streams still share the same GPU gate.
                comm_stream.wait_event(release)
                sidecar.post_dispatch(
                    dev_comm.ptr,
                    arena_win.handle,
                    arena_win.local_ptr,
                    peer,
                    SLOT,
                    current_generation,
                    stream=comm_stream,
                )
                sidecar.mark_chunk_ready(
                    arena_win.local_ptr,
                    SLOT,
                    current_generation,
                    ptrs,
                    0,
                    inputs.active_m_tiles,
                    delta=1,
                    stream=comm_stream,
                )
                comm_end = torch.cuda.Event(enable_timing=True)
                comm_end.record(comm_stream)

                compute_stream.wait_event(release)
                launch_h1(ptrs, current_generation, compute_stream)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_end.record(compute_stream)
                ends.extend((compute_end, comm_end))
            else:
                raise ValueError(f"unknown benchmark case {case!r}")

            elapsed_us = _event_max_us(start, tuple(ends))
            if uses_comm and not verified_payload:
                got = cco.read_window_u64(rx_ptr, TRAFFIC_BYTES // 8)
                if got != _payload(peer):
                    raise AssertionError("WORLD2 CCO 64 KiB payload mismatch")
                verified_payload = True

            # Slot credit and retained flush request retirement are lifecycle
            # costs, not part of the requested communication/control interval.
            if uses_comm:
                communicator.barrier()
                sidecar.return_credit(
                    dev_comm.ptr,
                    arena_win.handle,
                    peer,
                    SLOT,
                    current_generation,
                    stream=comm_stream,
                )
                sidecar.reclaim_dispatch(
                    dev_comm.ptr,
                    arena_win.local_ptr,
                    SLOT,
                    current_generation,
                    stream=comm_stream,
                )
                comm_stream.synchronize()
            if uses_h1:
                compute_stream.synchronize()
            communicator.barrier()
            return elapsed_us

        cases = ("comm_only", "h1_only", "serial", "overlap")
        samples: dict[str, list[float]] = {}
        for case in cases:
            print(f"MEGAMOE_OVERLAP_CASE_BEGIN rank={rank} case={case}", flush=True)
            for _ in range(args.warmup):
                run_once(case)
            samples[case] = [run_once(case) for _ in range(args.iters)]
            print(f"MEGAMOE_OVERLAP_CASE_END rank={rank} case={case}", flush=True)

        medians = {
            case: float(statistics.median(values))
            for case, values in samples.items()
        }
        tcomm = medians["comm_only"]
        tcompute = medians["h1_only"]
        tserial = medians["serial"]
        toverlap = medians["overlap"]
        # Fraction of the isolated communication latency removed from the
        # measured serial critical path by two-stream execution.
        hidden_fraction_raw = (tserial - toverlap) / tcomm
        hidden_fraction = min(1.0, max(0.0, hidden_fraction_raw))

        output = {
            "rank": rank,
            "peer": peer,
            "kernel": ready_h1.kernel_name,
            "tokens": TOKENS,
            "active_m_tiles": inputs.active_m_tiles,
            "workers": args.workers,
            "wpe": 2,
            "traffic_bytes": TRAFFIC_BYTES,
            "num_qp": NUM_QP,
            "batch_per_qp": BATCH_PER_QP,
            "segment_bytes": SEGMENT_BYTES,
            "warmup": args.warmup,
            "iters": args.iters,
            "gate_repeats": args.gate_repeats,
            "gate_dim": args.gate_dim,
            "Tcomm_us": tcomm,
            "Tcompute_us": tcompute,
            "Tserial_us": tserial,
            "Toverlap_us": toverlap,
            "hidden_fraction": hidden_fraction,
            "hidden_fraction_raw": hidden_fraction_raw,
            "hidden_fraction_formula": "clamp((Tserial-Toverlap)/Tcomm,0,1)",
            "ideal_sum_us": tcomm + tcompute,
            "payload_verified": verified_payload,
            "timing": "delayed-common-start HIP events; overlap=max(end spans)",
            "overlap_enqueue": "high-priority comm first, then ready-H1",
            "timed_comm_scope": "send+remote-ready-wait+input-ready-mark",
            "excluded": "packing,sorting,plan/reset,credit,request-reclaim",
            "cases": {case: _summary(values) for case, values in samples.items()},
        }
        print(
            "MEGAMOE_CCO_H1_OVERLAP " + json.dumps(output, sort_keys=True),
            flush=True,
        )
        print(
            "MEGAMOE_CCO_TRANSPORT_PASS "
            f"rank={rank} benchmark=h1_overlap traffic={TRAFFIC_BYTES}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
