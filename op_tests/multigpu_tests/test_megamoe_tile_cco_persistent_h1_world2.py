# SPDX-License-Identifier: MIT
"""WORLD2 target-shape smoke for the single-kernel persistent CCO H1.

Uses the same ``CCO_RANK/CCO_WORLD/CCO_UID_FILE`` launcher contract as the
other WORLD2 CCO tests. Compute tensors are deliberately preloaded; the real
64-KiB reciprocal CCO dispatch gates their plan/readiness publication.
"""

from __future__ import annotations

import os
import sys
import time

import torch

from mori.cco import Communicator, UniqueId

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.megamoe_tile import cco, prepare_local_a4w4_weights
from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
    compile_hier_stage1_persistent_cco_a4w4,
)
from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout
from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort


TOKENS, HIDDEN, INTER, EXPERTS, TOPK, BM = 8, 3584, 384, 56, 16, 32
NUM_QP = 4
WORKERS = int(os.environ.get("MEGAMOE_CCO_H1_WORKERS", "240"))
WPE = int(os.environ.get("MEGAMOE_CCO_H1_WPE", "2"))
GENERATION = int(os.environ.get("MEGAMOE_CCO_GENERATION", "7"))
WARMUP = int(os.environ.get("MEGAMOE_CCO_H1_WARMUP", "0"))
ITERS = int(os.environ.get("MEGAMOE_CCO_H1_ITERS", "1"))
SLOT = 0
CHUNK_BYTES = 64 * 1024


def _bootstrap():
    rank = int(os.environ["CCO_RANK"])
    world = int(os.environ["CCO_WORLD"])
    path = os.environ["CCO_UID_FILE"]
    while not os.path.exists(path) or os.path.getsize(path) != 128:
        time.sleep(0.05)
    with open(path, "rb") as file:
        return rank, world, UniqueId.from_bytes(file.read())


def _pattern(rank: int, generation: int, words: int):
    base = rank * 1_000_000 + generation * 100_000
    return tuple(base + index for index in range(words))


def main() -> int:
    rank, world, uid = _bootstrap()
    if world != 2:
        raise ValueError("persistent CCO H1 smoke requires WORLD2")
    gpu = int(os.environ.get("CCO_GPU", "0"))
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    stream = torch.cuda.current_stream(device)
    peer = 1 - rank

    torch.manual_seed(101 + rank)
    x = (torch.randn(TOKENS, HIDDEN, device=device) * 0.1).to(torch.bfloat16)
    w1 = (
        torch.randn(EXPERTS, 2 * INTER, HIDDEN, device=device) * 0.03
    ).to(torch.bfloat16)
    w2 = (torch.randn(EXPERTS, HIDDEN, INTER, device=device) * 0.03).to(
        torch.bfloat16
    )
    score = torch.rand(TOKENS, EXPERTS, device=device)
    values, ids = torch.topk(score, TOPK, dim=1)
    route_weights = torch.softmax(values, dim=1).float()
    prepared = prepare_local_a4w4_weights(w1, w2)
    sorted_ids, _, sorted_eids, nvalid, _ = moe_sorting(
        ids.to(torch.int32),
        route_weights,
        EXPERTS,
        HIDDEN,
        torch.bfloat16,
        BM,
        accumulate=False,
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(
        a1s.view(TOKENS, 1, HIDDEN // 32),
        sorted_ids,
        nvalid,
        TOKENS,
        BM,
    )
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()
    max_sorted = int(sorted_ids.shape[0])
    max_m_tiles = (max_sorted + BM - 1) // BM
    active_m_tiles = int(nvalid[0].item()) // BM
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((INTER // 32) + 7) // 8 * 8
    hidden_dummy = torch.zeros(
        (TOKENS, HIDDEN), dtype=torch.bfloat16, device=device
    )

    reference_q = torch.zeros(
        (max_sorted, INTER // 2), dtype=torch.uint8, device=device
    )
    reference_s = torch.zeros(
        scale_rows * scale_cols, dtype=torch.uint8, device=device
    )
    flydsl_mxfp4_gemm1(
        a_quant=a1q,
        a_scale_sorted_shuffled=a1ss,
        w1_u8=prepared.w1.view(torch.uint8),
        w1_scale_u8=prepared.w1_scale.view(torch.uint8),
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=nvalid,
        m_indices=m_indices,
        inter_sorted_quant=reference_q,
        inter_sorted_shuffled_scale=reference_s,
        hidden_states=hidden_dummy,
        n_tokens=TOKENS,
        BM=BM,
        use_nt=True,
        inline_quant=False,
        NE=EXPERTS,
        D_HIDDEN=HIDDEN,
        D_INTER=INTER,
        topk=TOPK,
        act="silu",
    )
    torch.cuda.synchronize(device)

    layout = HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=NUM_QP,
        chunk_bytes=CHUNK_BYTES,
        max_m_tiles=max_m_tiles,
        max_source_tokens=32,
        max_h1_n_blocks=4,
    )
    kernel = compile_hier_stage1_persistent_cco_a4w4(
        layout,
        D_HIDDEN=HIDDEN,
        D_INTER=INTER,
        NE=EXPERTS,
        TOPK=TOPK,
        WORK_SHARDS=8,
        waves_per_eu_hint=WPE,
        team=cco.TEAM_WORLD,
        activation="silu",
    )
    reclaim = cco.CcoStage1Sidecar.create(
        layout,
        batch_per_qp=8,
        segment_bytes=2048,
        team=cco.TEAM_WORLD,
    )
    entry_count = torch.zeros(1, dtype=torch.int64, device=device)
    epoch_gate = torch.zeros(1, dtype=torch.int32, device=device)
    work_head = torch.zeros(8 * 16, dtype=torch.int32, device=device)
    actual_q = torch.zeros_like(reference_q)
    actual_s = torch.zeros_like(reference_s)
    failures = 0

    with Communicator.init(
        world, rank, uid, per_rank_vmm=64 * 1024 * 1024
    ) as comm:
        resources = cco.create_transport_resources(
            comm,
            layout.total_bytes,
            num_qp=NUM_QP,
            team=cco.TEAM_WORLD,
        )
        window = resources.window
        dc = resources.dev_comm
        cco.clear_hip_last_error()
        cco.zero_window(window.local_ptr, layout.total_bytes)
        tx_ptr = window.local_ptr + layout.ring_chunk_offset("dispatch_tx", SLOT)
        rx_ptr = window.local_ptr + layout.ring_chunk_offset("dispatch_rx", SLOT)
        words = CHUNK_BYTES // 8
        request_ptr = window.local_ptr + layout.ring_qp_offset(
            "dispatch_request", SLOT, 0
        )
        launches = WARMUP + ITERS
        if launches <= 0:
            raise ValueError("WARMUP + ITERS must be positive")
        for iteration in range(launches):
            generation = GENERATION + iteration
            cco.write_window_u64(
                tx_ptr, _pattern(rank, generation, words)
            )
            cco.zero_window(rx_ptr, CHUNK_BYTES)
            torch.cuda.synchronize(device)
            comm.barrier()

            kernel(
                dc.ptr,
                window.handle,
                window.local_ptr,
                peer,
                SLOT,
                generation,
                active_m_tiles,
                1,
                entry_count.data_ptr(),
                epoch_gate.data_ptr(),
                work_head.data_ptr(),
                a1q.data_ptr(),
                a1ss.data_ptr(),
                prepared.w1.data_ptr(),
                prepared.w1_scale.data_ptr(),
                sorted_eids.data_ptr(),
                nvalid.data_ptr(),
                m_indices.data_ptr(),
                TOKENS,
                WORKERS,
                actual_q.data_ptr(),
                actual_s.data_ptr(),
                hidden_dummy.data_ptr(),
                stream=stream,
            )
            torch.cuda.synchronize(device)

            expected_payload = _pattern(peer, generation, words)
            received = cco.read_window_u64(rx_ptr, words)
            failures += sum(
                int(lhs != rhs)
                for lhs, rhs in zip(received, expected_payload)
            )
            failures += int(not torch.equal(actual_q, reference_q))
            failures += int(not torch.equal(actual_s, reference_s))
            ptrs = layout.epoch_pointers(window.local_ptr, generation)
            expected = cco.read_window_u32(
                ptrs.h1_input_expected, active_m_tiles
            )
            ready = cco.read_window_u32(
                ptrs.h1_input_ready, active_m_tiles
            )
            plan = cco.read_window_u64(ptrs.plan_ready, 1)[0]
            failures += sum(int(value != 1) for value in expected)
            failures += sum(int(value != 1) for value in ready)
            failures += int(plan != generation)
            requests = cco.read_window_u64(request_ptr, NUM_QP)
            failures += sum(int(value == 0) for value in requests)

            # External retirement: return credit, then wait credit + retained
            # local request and clear it before reusing the same ring slot.
            reclaim.module.launch_return_credit(
                dc.ptr,
                window.handle,
                peer,
                SLOT,
                generation,
                stream=stream,
            )
            reclaim.module.launch_reclaim(
                dc.ptr,
                window.local_ptr,
                SLOT,
                generation,
                stream=stream,
            )
            torch.cuda.synchronize(device)
            failures += sum(
                int(value != 0)
                for value in cco.read_window_u64(request_ptr, NUM_QP)
            )
            comm.barrier()

        print(
            f"MEGAMOE_CCO_PERSISTENT_H1_{'PASS' if failures == 0 else 'FAIL'} "
            f"rank={rank} peer={peer} kernel={kernel.kernel_name} "
            f"active_m_tiles={active_m_tiles} workers={WORKERS} "
            f"wpe={WPE} "
            f"warmup={WARMUP} iters={ITERS} "
            f"preloaded_seam={kernel.preloaded_compute_seam} "
            f"external_reclaim={kernel.request_reclaim_external} "
            f"failures={failures}",
            flush=True,
        )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
