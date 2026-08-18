# SPDX-License-Identifier: MIT
"""Target-shape correctness and kernel-only timing for ready-aware H1.

Control-plane reset/expected/ready publication is deliberately outside each
timed interval.  The reported ready latency therefore covers only the ready
consumer kernel after all input tiles are already visible.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch


def _event_time_us(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("small", "target"), default="target")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--wpe", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--scheduler", choices=("direct", "queue"), default="direct"
    )
    parser.add_argument(
        "--wait-plan",
        action="store_true",
        help="Enable the debug plan-ready wait; production default is off.",
    )
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    # Keep GMM2's accumulation schedule deterministic for the semantic check.
    os.environ["MXFP4_G2_SPART"] = "0"
    os.environ["MXFP4_G2_BF16_LDS"] = "1"
    os.environ["MXFP4_G2_KSTATIC"] = "1"

    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.megamoe_tile import prepare_local_a4w4_weights
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_h1_ready_queue_publisher,
        build_hier_epoch_module,
        compile_hier_stage1_queue_a4w4,
        compile_hier_stage1_ready_a4w4,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)

    if args.shape == "small":
        m, h, inter, experts, topk, bm = args.tokens, 1024, 256, 8, 2, 32
    else:
        m, h, inter, experts, topk, bm = args.tokens, 3584, 384, 56, 16, 32
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * inter, h, device=dev) * 0.03).to(
        torch.bfloat16
    )
    w2 = (torch.randn(experts, h, inter, device=dev) * 0.03).to(
        torch.bfloat16
    )
    score = torch.rand(m, experts, device=dev)
    values, ids = torch.topk(score, topk, dim=1)
    route_weights = torch.softmax(values, dim=1).float()

    prepared = prepare_local_a4w4_weights(w1, w2)
    sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
        ids.to(torch.int32),
        route_weights,
        experts,
        h,
        torch.bfloat16,
        bm,
        accumulate=False,
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(
        a1s.view(m, 1, h // 32), sorted_ids, nvalid, m, bm
    )
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()

    max_sorted = int(sorted_ids.shape[0])
    max_m_tiles = (max_sorted + bm - 1) // bm
    active_m_tiles = int(nvalid[0].item()) // bm
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((inter // 32) + 7) // 8 * 8
    dummy_hidden = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)

    reference_q = torch.zeros(
        (max_sorted, inter // 2), dtype=torch.uint8, device=dev
    )
    reference_s = torch.zeros(
        scale_rows * scale_cols, dtype=torch.uint8, device=dev
    )

    def launch_core() -> None:
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
            hidden_states=dummy_hidden,
            n_tokens=m,
            BM=bm,
            use_nt=True,
            inline_quant=False,
            NE=experts,
            D_HIDDEN=h,
            D_INTER=inter,
            topk=topk,
            act="silu",
            persistent=True,
            persistent_blocks=args.workers,
            stream=stream,
        )

    # Compile and establish the bitwise reference before timing.
    launch_core()
    torch.cuda.synchronize(dev)

    layout = HierCcoArenaLayout.create(
        max_m_tiles=max_m_tiles,
        max_source_tokens=max(32, m),
    )
    arena = layout.allocate_local(device=dev)
    reset, publish_plan, _ = build_hier_epoch_module(
        max_m_tiles=max_m_tiles,
        max_source_tokens=max(32, m),
    )
    queue_capacity = layout.view(arena, "h1_ready_queue", parity=0).numel()
    if args.scheduler == "queue":
        queue_init, queue_publish, queue_finish = build_h1_ready_queue_publisher(
            num_n_blocks=(2 * inter) // 256,
            max_work=queue_capacity,
        )
        ready_h1 = compile_hier_stage1_queue_a4w4(
            D_HIDDEN=h,
            D_INTER=inter,
            NE=experts,
            TOPK=topk,
            max_work=queue_capacity,
            activation="silu",
            waves_per_eu_hint=args.wpe,
            wait_epoch=args.wait_plan,
        )
    else:
        queue_init = queue_publish = queue_finish = None
        ready_h1 = compile_hier_stage1_ready_a4w4(
            D_HIDDEN=h,
            D_INTER=inter,
            NE=experts,
            TOPK=topk,
            activation="silu",
            waves_per_eu_hint=args.wpe,
            wait_plan=args.wait_plan,
        )
    result_q = torch.zeros_like(reference_q)
    result_s = torch.zeros_like(reference_s)

    def prepare_epoch(generation: int):
        ptrs = layout.epoch_pointers(arena.data_ptr(), generation)
        if args.scheduler == "queue":
            total_work = active_m_tiles * ready_h1.num_n_blocks
            queue_init(
                ptrs.h1_queue_header,
                generation,
                total_work,
                stream=stream,
            )
            for m_tile in range(active_m_tiles):
                queue_publish(
                    ptrs.h1_queue_header,
                    ptrs.h1_ready_queue,
                    m_tile,
                    stream=stream,
                )
            queue_finish(ptrs.h1_queue_header, generation, stream=stream)
            return ptrs
        reset(
            ptrs.h1_input_expected,
            ptrs.h1_input_ready,
            ptrs.h1_output_done,
            ptrs.h2_output_done,
            ptrs.rank_route_expected,
            ptrs.rank_route_ready,
            active_m_tiles,
            m,
            1,
            stream=stream,
        )
        # Same-stream stores are complete before the start event below.  This
        # models a sidecar that has already published every input M tile.
        expected = layout.view(
            arena, "h1_input_expected", parity=ptrs.parity
        )
        ready = layout.view(arena, "h1_input_ready", parity=ptrs.parity)
        expected[:active_m_tiles].fill_(1)
        ready[:active_m_tiles].fill_(1)
        publish_plan(ptrs.plan_ready, generation, stream=stream)
        return ptrs

    def launch_ready(ptrs, generation: int) -> None:
        if args.scheduler == "queue":
            ready_h1(
                ptrs.h1_queue_header,
                ptrs.h1_ready_queue,
                generation,
                a1q.data_ptr(),
                a1ss.data_ptr(),
                prepared.w1.data_ptr(),
                prepared.w1_scale.data_ptr(),
                sorted_eids.data_ptr(),
                nvalid.data_ptr(),
                m_indices.data_ptr(),
                m,
                args.workers,
                result_q.data_ptr(),
                result_s.data_ptr(),
                dummy_hidden.data_ptr(),
                stream=stream,
            )
            return
        ready_h1(
            ptrs.plan_ready,
            ptrs.h1_input_ready,
            ptrs.h1_input_expected,
            ptrs.h1_output_done,
            ptrs.h1_output_ready,
            generation,
            a1q.data_ptr(),
            a1ss.data_ptr(),
            prepared.w1.data_ptr(),
            prepared.w1_scale.data_ptr(),
            sorted_eids.data_ptr(),
            nvalid.data_ptr(),
            m_indices.data_ptr(),
            m,
            args.workers,
            result_q.data_ptr(),
            result_s.data_ptr(),
            dummy_hidden.data_ptr(),
            stream=stream,
        )

    ptrs = prepare_epoch(1)
    launch_ready(ptrs, 1)
    torch.cuda.synchronize(dev)
    q_equal = torch.equal(result_q, reference_q)
    s_equal = torch.equal(result_s, reference_s)
    output_done = layout.view(arena, "h1_output_done", parity=ptrs.parity)
    output_ready = layout.view(arena, "h1_output_ready", parity=ptrs.parity)
    diagnostic = {
        "q_diff": int(torch.count_nonzero(result_q != reference_q).item()),
        "scale_diff": int(torch.count_nonzero(result_s != reference_s).item()),
        "done_min": (
            int(output_done[:active_m_tiles].min().item())
            if args.scheduler == "direct"
            else -1
        ),
        "done_max": (
            int(output_done[:active_m_tiles].max().item())
            if args.scheduler == "direct"
            else -1
        ),
        "ready_count": (
            int(torch.count_nonzero(output_ready[:active_m_tiles] == 1).item())
            if args.scheduler == "direct"
            else -1
        ),
        "active_m_tiles": active_m_tiles,
        "num_n_blocks": ready_h1.num_n_blocks,
    }
    if not q_equal or not s_equal:
        diff_index = torch.nonzero(result_s != reference_s, as_tuple=False).flatten()
        # The v2 scale store addresses a 32-row M chunk with
        # (chunk, ku, wave_group, row_pair). For I=384 the current stride is
        # 64 dwords for both chunk and ku, so ku=1 aliases chunk+1,ku=0.
        mapping = []
        for index in diff_index[:32].tolist():
            dword = index // 4
            storage_chunk = dword // 64
            rem = dword % 64
            mapping.append(
                {
                    "byte_index": index,
                    "storage_chunk": storage_chunk,
                    "wave_group": rem // 16,
                    "row_pair_lane": rem % 16,
                    "byte_in_dword": index % 4,
                    "valid_writers": [
                        # N blocks 0/1 use ku=0 in this storage chunk.
                        {"m_tile": storage_chunk, "n_blocks": [0, 1]},
                        # N block 2 uses ku=1 and aliases the next chunk.
                        {"m_tile": storage_chunk - 1, "n_blocks": [2]},
                    ],
                }
            )
        diagnostic["scale_diff_mapping"] = mapping
        diagnostic["scale_diff_in_aliased_active_region"] = int(
            sum(
                1
                for index in diff_index.tolist()
                if 1 <= (index // 4) // 64 < active_m_tiles
            )
        )

        def run_h2(scale, output):
            mxfp4_moe_gemm2(
                inter_sorted_quant=reference_q,
                inter_sorted_shuffled_scale=scale,
                w2_u8=prepared.w2.view(torch.uint8),
                w2_scale_u8=prepared.w2_scale.view(torch.uint8),
                sorted_expert_ids=sorted_eids,
                cumsum_tensor=nvalid,
                sorted_token_ids=sorted_ids,
                sorted_weights=sorted_weights,
                out=output,
                M_logical=m,
                max_sorted=max_sorted,
                NE=experts,
                D_HIDDEN=h,
                D_INTER=inter,
                topk=topk,
                BM=bm,
                BN=128,
                BK=128,
                a_dtype="fp4",
                epilog="atomic",
                SBM=bm,
                HIDDEN_MAX=h,
                INTER_MAX=inter,
            )

        h2_reference = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)
        h2_ready = torch.zeros_like(h2_reference)
        run_h2(reference_s, h2_reference)
        run_h2(result_s, h2_ready)
        torch.cuda.synchronize(dev)
        xref = h2_reference.float().double()
        xgot = h2_ready.float().double()
        denominator = xref.square().sum() + xgot.square().sum()
        diagnostic["h2_bitwise"] = bool(torch.equal(h2_reference, h2_ready))
        diagnostic["h2_logits_diff"] = float(
            1.0 - 2.0 * (xref * xgot).sum() / denominator
        )
        diagnostic["h2_max_abs"] = float(
            (xref - xgot).abs().max().item()
        )
        print("READY_H1_MISMATCH", json.dumps(diagnostic, sort_keys=True), flush=True)
        raise AssertionError("ready H1 differs from persistent core")

    def run_h2_success(scale, output):
        mxfp4_moe_gemm2(
            inter_sorted_quant=reference_q,
            inter_sorted_shuffled_scale=scale,
            w2_u8=prepared.w2.view(torch.uint8),
            w2_scale_u8=prepared.w2_scale.view(torch.uint8),
            sorted_expert_ids=sorted_eids,
            cumsum_tensor=nvalid,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            out=output,
            M_logical=m,
            max_sorted=max_sorted,
            NE=experts,
            D_HIDDEN=h,
            D_INTER=inter,
            topk=topk,
            BM=bm,
            BN=128,
            BK=128,
            a_dtype="fp4",
            epilog="atomic",
            SBM=bm,
            HIDDEN_MAX=h,
            INTER_MAX=inter,
        )

    h2_reference = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)
    h2_ready = torch.zeros_like(h2_reference)
    run_h2_success(reference_s, h2_reference)
    run_h2_success(result_s, h2_ready)
    torch.cuda.synchronize(dev)
    xref = h2_reference.float().double()
    xgot = h2_ready.float().double()
    h2_denominator = xref.square().sum() + xgot.square().sum()
    h2_logits_diff = float(
        1.0 - 2.0 * (xref * xgot).sum() / h2_denominator
    )
    if h2_logits_diff >= 1.0e-2:
        raise AssertionError(f"ready H1 fails H2 semantic check: {h2_logits_diff}")

    for i in range(args.warmup):
        generation = 2 + i
        ptrs = prepare_epoch(generation)
        launch_ready(ptrs, generation)
    torch.cuda.synchronize(dev)

    core_times = [_event_time_us(launch_core) for _ in range(args.iters)]
    ready_times = []
    generation = 2 + args.warmup
    for _ in range(args.iters):
        generation += 1
        ptrs = prepare_epoch(generation)
        # The start event is ordered after reset/readiness publication.
        ready_times.append(
            _event_time_us(lambda p=ptrs, g=generation: launch_ready(p, g))
        )

    output = {
        "kernel": ready_h1.kernel_name,
        "scheduler": args.scheduler,
        "shape": args.shape,
        "tokens": m,
        "active_m_tiles": active_m_tiles,
        "gemm_tiles": active_m_tiles * ready_h1.num_n_blocks,
        "workers": args.workers,
        "wpe": args.wpe,
        "wait_plan": args.wait_plan,
        "warmup": args.warmup,
        "iters": args.iters,
        "core_median_us": statistics.median(core_times),
        "ready_median_us": statistics.median(ready_times),
        "ready_min_us": min(ready_times),
        "ready_max_us": max(ready_times),
        "overhead_pct": (
            statistics.median(ready_times) / statistics.median(core_times) - 1.0
        )
        * 100.0,
        "bitwise": True,
        "h2_bitwise": bool(torch.equal(h2_reference, h2_ready)),
        "h2_logits_diff": h2_logits_diff,
    }
    print("READY_H1_TIMING", json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
