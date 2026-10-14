"""TP8 fusion ladder with identical Gluon arithmetic at every split boundary."""

import argparse
import json
from pathlib import Path
import socket
import statistics
from contextlib import ExitStack
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale
from hip_ipc import PeerMemory, DevicePointer
from k3_gluon.complete_host import workspace, arguments
from k3_gluon.moe_complete import pack_dense
from k3_gluon.perf_reference_chain import chain
from k3_gluon.serving import allocate_workspace
from validate_complete_moe import checkpoint_weights, layer_weight
from k3_gluon_hybrid_sort_b8.pipeline import launch_hybrid
from hybrid_reference import check, expert_weights
from k3_gluon_hybrid_sort_b8.host import prepare_metadata, run_prepared_experts


def worker(rank, port, args):
    import importlib
    from triton.experimental.gluon import language as gl

    package = "k3_gluon_hybrid_sort_b8" if args.coherent_payloads else "k3_gluon"
    if args.grid != 256:
        assert (
            args.tuned
            and args.normal_launch
            and not args.phase_barrier
            and not args.compare_barrier
        )
        for module in (
            "perf_reference_up",
            "perf_reference_tail",
            "perf_timeline",
            "perf_front32",
        ):
            importlib.import_module(package + "." + module).PERSISTENT_CTAS = (
                gl.constexpr(args.grid)
            )
    if args.profile_all_ctas:
        assert args.profile_stages
        importlib.import_module(package + ".perf_timeline").ALL_CTAS = gl.constexpr(
            True
        )
    if args.pair_topk:
        importlib.import_module(package + ".moe_front").PAIR_TOPK = gl.constexpr(True)
    if args.load_selected_bias:
        importlib.import_module(package + ".moe_front").LOAD_SELECTED_BIAS = (
            gl.constexpr(True)
        )
    if args.collapse_columns:
        assert args.batches == [1]
        importlib.import_module(package + ".perf_reference_tail").COLLAPSE_COLUMNS = (
            gl.constexpr(True)
        )
    if args.skip_unused_quant:
        assert not args.native_dense
        importlib.import_module(package + ".perf_reference_chain").SKIP_UNUSED_QUANT = (
            gl.constexpr(True)
        )
    if args.partial_norm:
        assert (
            args.batches == [1]
            and args.dense_lds
            and not args.native_dense
            and args.tail_rows != 64
        )
    if args.coherent_payloads:
        from k3_gluon_hybrid_sort_b8.perf_reference_chain import chain as test_chain
    else:
        test_chain = chain
    if args.dense_lds or args.projection_unroll != 1 or args.projection_buffer:
        from triton.experimental.gluon import language as gl

        if args.coherent_payloads:
            from k3_gluon_hybrid_sort_b8 import projection
        else:
            from k3_gluon import projection
        projection.STAGE_INPUTS = gl.constexpr(args.dense_lds)
        projection.PROJECTION_UNROLL = gl.constexpr(args.projection_unroll)
        projection.PROJECTION_BUFFER = gl.constexpr(args.projection_buffer)
    if args.front32:
        import importlib
        from triton.experimental.gluon import language as gl

        package = "k3_gluon_hybrid_sort_b8" if args.coherent_payloads else "k3_gluon"
        front = importlib.import_module(package + ".perf_front32")
        module = importlib.import_module(package + ".perf_reference_chain")
        module.FRONT32 = gl.constexpr(True)
        front.STAGE_INPUTS = gl.constexpr(args.dense_lds)
        front.PROJECTION_UNROLL = gl.constexpr(
            args.front_unroll or args.projection_unroll
        )
        front.PROJECTION_BUFFER = gl.constexpr(args.projection_buffer)
        assert all(
            1 <= b <= 8 and ((b + 3) // 4) * 188 <= args.grid for b in args.batches
        )
        assert args.world == 8 and not args.native_dense
    if args.poll_sleep:
        from triton.experimental.gluon import language as gl
        from k3_gluon import primitives

        for world in (1, 4, 8):
            assembly = primitives._timed_asm(world).replace(
                "s_add_u32", f"s_sleep {args.poll_sleep}\ns_add_u32"
            )
            setattr(primitives, f"TIMED{world}", gl.constexpr(assembly))
        for world in (4, 8):
            assembly = primitives._poll_asm(world).replace(
                "s_add_u32", f"s_sleep {args.poll_sleep}\ns_add_u32"
            )
            setattr(primitives, f"POLL{world}_ASM", gl.constexpr(assembly))
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=args.world
    )
    cpu = dist.new_group(backend="gloo")
    w, ref = checkpoint_weights(rank, args.world, args.layer)
    for label, q, s in [("Up", ref["qu"], ref["su"]), ("Down", ref["qd"], ref["ss"])]:
        w["W" + label] = shuffle_weight_a16w4(q, 16, False).view(torch.uint8)
        w["S" + label] = shuffle_scale(s.reshape(-1, s.shape[-1])).view(torch.uint8)
    expert = expert_weights(w)
    native_extra = {}
    projection = layer_weight(args.layer, "self_attn.o_proj")[
        :, rank * 12288 // args.world : (rank + 1) * 12288 // args.world
    ].contiguous()
    reports = []
    for batch in args.batches:
        with ExitStack() as stack:
            down = stack.enter_context(
                PeerMemory(
                    2 * args.world * batch * 1792 * 8, rank, args.world, group=cpu
                )
            )
            final = stack.enter_context(
                PeerMemory(
                    2 * args.world * batch * 3584 * 8, rank, args.world, group=cpu
                )
            )
            proj = stack.enter_context(
                PeerMemory(
                    2 * args.world * batch * 3584 * 8, rank, args.world, group=cpu
                )
            )
            state = workspace(batch, args.world)
            outputs = allocate_workspace(batch)
            if args.grid != 256:
                state["Epochs"] = torch.zeros(
                    args.grid, device="cuda", dtype=torch.int32
                )
                state["Errors"] = torch.zeros(
                    args.grid * 512, device="cuda", dtype=torch.int32
                )
            if args.partial_norm:
                # A latent consumer accidentally using the obsolete global
                # normalized buffer must fail the independent output check.
                state["Normed"].fill_(float("nan"))
            if args.front32:
                state["LatentReady"] = torch.zeros(
                    (batch, 112), device="cuda", dtype=torch.uint64
                )
                state["SharedReady"] = torch.zeros(
                    (batch, 384 // args.world), device="cuda", dtype=torch.uint64
                )
            torch.manual_seed(302 + rank)
            core = torch.randn(batch, 12288 // args.world, device="cuda").bfloat16()
            torch.manual_seed(402)
            prefix = torch.randn(batch, 7168, device="cuda").bfloat16() * 0.1
            blocks = torch.randn(batch, 8, 7168, device="cuda").bfloat16() * 0.1
            blocks[:, args.blocks :] = float("nan")

            assert batch == 8 and args.expert_precision == "a4w4"
            metadata = prepare_metadata(state["IDs"], state["Weights"])
            local_routed = run_prepared_experts(
                state["Latent"], expert, state["IDs"], state["Weights"], metadata
            )
            delta = torch.empty_like(prefix)
            opts = dict(
                WORLD=args.world,
                WATCHDOG_TICKS=0 if args.iteration_watchdog else 500000000,
                AITER_LAYOUT=True,
                TUNED=args.tuned,
                PhaseReady=torch.zeros(args.grid, device="cuda", dtype=torch.uint64),
                ProjectionX=core,
                ProjectionW=pack_dense(projection),
                PeersProjection=proj.addresses,
                LocalProjection=DevicePointer(proj.ptr.value),
                Delta=delta,
                Attention=torch.zeros(batch * 3584, device="cuda", dtype=torch.uint64),
                Prefix=prefix,
                Blocks=blocks,
                Norm=layer_weight(args.layer, "mlp_res_norm").flatten(),
                Query=layer_weight(args.layer, "mlp_res_proj").flatten(),
                OutNorm=layer_weight(args.layer, "post_attention_layernorm").flatten(),
                Updated=outputs[0],
                Quant=outputs[2],
                Scales=outputs[3],
                Stats=outputs[4],
                InputReady=outputs[5],
                N_BLOCKS=args.blocks,
                BOUNDARY=False,
                BLOCK_STRIDE_M=blocks.stride(0),
                Residual=outputs[0],
                FinalPrefix=state["FinalPrefix"],
                num_warps=8,
                enable_fp_fusion=False,
                launch_cooperative_grid=not args.normal_launch,
                BUFFER=args.buffer,
                LDS=args.lds,
                ZERO_GUARD=(
                    3
                    if args.word_zero_guard
                    else 2 if args.packed_zero_guard else int(not args.elide_zero_guard)
                ),
                DOWN_CHUNK=args.down_chunk,
                DOWN_UNROLL=args.down_unroll,
                DOWN_LDS=args.down_lds,
                WAVE_PUBLISH=args.wave_publish,
                SINGLE_SAMPLE=args.single_sample_tail,
                BATCH_TAIL=args.batch_tail,
                DIRECT_ACC=args.direct_acc,
                Timeline=(
                    torch.zeros((8, args.grid), device="cuda", dtype=torch.uint64)
                    if args.profile_stages
                    else None
                ),
            )
            opts.update(
                NATIVE_DENSE=args.native_dense,
                NATIVE_LDS=args.native_lds,
                TAIL_ROWS=args.tail_rows,
                MFMA_LANES=args.mfma_lanes,
                **native_extra,
            )
            if args.partial_norm:
                opts["NormStats"] = torch.zeros(
                    (batch, 224), device="cuda", dtype=torch.uint64
                )
            if args.serving_tune:
                opts.update(BATCH_TAIL=batch > 1)
                if batch == 1:
                    opts.update(
                        BUFFER=True,
                        LDS=True,
                        ZERO_GUARD=3,
                        DOWN_CHUNK=128,
                        DOWN_UNROLL=3,
                        DOWN_LDS=True,
                    )
            opts.update(metadata)
            call = arguments(outputs[1], w, state, down, final, rank, args.world)
            variants = {
                "moe_fused": [(2, 2), (4, 4)],
                "projection_attnres_moe_fused": [(0, 2), (4, 4)],
            }
            if args.compare_barrier:
                for name, stages in list(variants.items()):
                    if any(start <= 3 and end >= 4 for start, end in stages):
                        variants[name + "_barrier"] = stages
            if args.reverse:
                variants = dict(reversed(list(variants.items())))
            if args.variants:
                variants = {
                    name: stages
                    for name, stages in variants.items()
                    if name in args.variants
                }
                assert set(variants) == set(args.variants), "Unknown variant"
            kernels = {}
            for name, stages in variants.items():
                barrier = args.phase_barrier or name.endswith("_barrier")
                for i, (start, end) in enumerate(stages):
                    key = (start, end, i == 0, barrier)
                    if key not in kernels:
                        k = test_chain.warmup(
                            *call,
                            START=start,
                            END=end,
                            BEGIN=i == 0,
                            PHASE_BARRIER=barrier,
                            grid=(args.grid,),
                            ExternalRouted=local_routed if start == 4 else None,
                            **opts,
                        )
                        _ = k.run
                        kernels[key] = dict(
                            registers=k.n_regs,
                            spills=k.n_spills,
                            shared=k.metadata.shared,
                        )
                        if args.normal_launch:
                            import ctypes as C

                            occupancy = (
                                down.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
                            )
                            occupancy.argtypes = [
                                C.POINTER(C.c_int),
                                C.c_void_p,
                                C.c_int,
                                C.c_size_t,
                            ]
                            occupancy.restype = C.c_int
                            resident = C.c_int()
                            down.call(
                                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                                C.byref(resident),
                                C.c_void_p(k.function),
                                512,
                                k.metadata.shared,
                            )
                            cus = torch.cuda.get_device_properties(
                                rank
                            ).multi_processor_count
                            assert resident.value * cus >= args.grid, (
                                resident.value,
                                cus,
                                args.grid,
                            )
                            kernels[key].update(
                                resident_blocks_per_cu=resident.value, num_cus=cus
                            )
                        if rank == 0:
                            Path(args.output).with_suffix(
                                f".b{batch}.s{start}-{end}.amdgcn"
                            ).write_text(k.asm["amdgcn"])
            # The independent check always runs the complete chain, even when
            # the requested timed subset omits it. Compile before rank sync.
            for start, end, begin in [(0, 2, True), (4, 4, False)]:
                k = test_chain.warmup(
                    *call,
                    START=start,
                    END=end,
                    BEGIN=begin,
                    grid=(args.grid,),
                    ExternalRouted=local_routed if start == 4 else None,
                    **opts,
                )
                _ = k.run

            def launch(stages, barrier=args.phase_barrier):
                launch_hybrid(
                    test_chain,
                    call,
                    dict(opts, START=stages[0][0]),
                    grid=args.grid,
                    state=state,
                    weights=expert,
                    metadata=metadata,
                    experts=run_prepared_experts,
                )

            torch.cuda.synchronize()
            dist.barrier(group=cpu)
            launch([(0, 4)])
            torch.cuda.synchronize()
            reference = state["Out"].clone()
            prefix_reference = state["FinalPrefix"].clone()
            reference_check = check
            check_options = dict(expert=expert, precision=args.expert_precision)
            numerical = reference_check(
                outputs[1], state, ref, rank=rank, world=args.world, **check_options
            )
            result = dict(
                batch=batch,
                unique_experts=int(state["IDs"].unique().numel()),
                numerical=numerical,
                kernels={str(k): v for k, v in kernels.items()},
                timings={},
            )
            if args.profile_stages:
                timeline = opts["Timeline"].cpu().tolist()
                all_timeline = [None] * args.world
                dist.all_gather_object(all_timeline, timeline, group=cpu)
                result["timeline_ticks_by_rank"] = all_timeline
                result["wall_clock_khz"] = down.wall_clock_khz
            for name, stages in variants.items():
                barrier = args.phase_barrier or name.endswith("_barrier")
                torch.cuda.synchronize()
                dist.barrier(group=cpu)
                launch(stages, barrier)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    state["Out"], reference, rtol=0.02, atol=0.003
                )
                torch.testing.assert_close(
                    state["FinalPrefix"], prefix_reference, rtol=0.02, atol=0.003
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.count):
                        launch(stages, barrier)
                torch.cuda.synchronize()
                dist.barrier(group=cpu)
                graph.replay()
                torch.cuda.synchronize()
                start_event, end_event = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)
                start_event.record()
                end_event.record()
                end_event.synchronize()
                times = []
                for _ in range(args.repeats):
                    dist.barrier(group=cpu)
                    start_event.record()
                    graph.replay()
                    end_event.record()
                    end_event.synchronize()
                    times.append(
                        start_event.elapsed_time(end_event) * 1000 / args.count
                    )
                assert not state["Errors"].count_nonzero().item()
                torch.testing.assert_close(
                    state["Out"], reference, rtol=0.02, atol=0.003
                )
                all_times = [None] * args.world
                dist.all_gather_object(all_times, times, group=cpu)
                slowest = [max(t[i] for t in all_times) for i in range(args.repeats)]
                row = dict(
                    median_max_rank_us=statistics.median(slowest),
                    min_max_rank_us=min(slowest),
                    max_max_rank_us=max(slowest),
                    rank_samples_us=all_times,
                )
                if args.profile_stages:
                    local_timeline = opts["Timeline"][:, 0].cpu().tolist()
                    all_timeline = [None] * args.world
                    dist.all_gather_object(all_timeline, local_timeline, group=cpu)
                    row["last_timeline_cta0_ticks_by_rank"] = all_timeline
                    if args.profile_all_ctas:
                        all_timeline = [None] * args.world
                        dist.all_gather_object(
                            all_timeline, opts["Timeline"].cpu().tolist(), group=cpu
                        )
                        row["last_timeline_all_ctas_ticks_by_rank"] = all_timeline
                if stages[0][0] == 0:
                    # The temporary Python output tensor has gone out of scope.
                    # Replay with changed inputs and allocator churn to verify
                    # that the graph owns the AITER result consumed by the tail.
                    saved_core = core.clone()
                    core.mul_(0.75)
                    churn = [
                        torch.empty(
                            (batch, 3584), device="cuda", dtype=torch.bfloat16
                        ).fill_(float("nan"))
                        for _ in range(16)
                    ]
                    del churn
                    torch.cuda.synchronize()
                    dist.barrier(group=cpu)
                    graph.replay()
                    torch.cuda.synchronize()
                    result["changed_input_graph_check"] = reference_check(
                        outputs[1],
                        state,
                        ref,
                        rank=rank,
                        world=args.world,
                        **check_options,
                    )
                    core.copy_(saved_core)
                    torch.cuda.synchronize()
                    dist.barrier(group=cpu)
                    graph.replay()
                    torch.cuda.synchronize()
                result["timings"][name] = row
                if rank == 0:
                    print(
                        dict(batch=batch, variant=name, us=row["median_max_rank_us"]),
                        flush=True,
                    )
            core.mul_(0.75)
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
            launch([(0, 4)])
            torch.cuda.synchronize()
            result["changed_input_check"] = reference_check(
                outputs[1], state, ref, rank=rank, world=args.world, **check_options
            )
            result["mailbox_errors"] = 0
            reports.append(result)
            if rank == 0:
                Path(args.output).write_text(
                    json.dumps(dict(args=vars(args), results=reports), indent=2)
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--grid", type=int, choices=[256, 512], default=256)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--output", default="results/chain.json")
    p.add_argument("--tuned", action="store_true")
    p.add_argument("--phase-barrier", action="store_true")
    p.add_argument("--compare-barrier", action="store_true")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--expert-precision", choices=["a4w4", "a16w4"], default="a4w4")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--buffer", action="store_true")
    p.add_argument("--lds", action="store_true")
    p.add_argument("--word-zero-guard", action="store_true")
    p.add_argument("--packed-zero-guard", action="store_true")
    p.add_argument("--elide-zero-guard", action="store_true")
    p.add_argument("--down-lds", action="store_true")
    p.add_argument("--down-chunk", type=int, default=32)
    p.add_argument("--down-unroll", type=int, default=1)
    p.add_argument("--wave-publish", action="store_true")
    p.add_argument("--profile-stages", action="store_true")
    p.add_argument("--profile-all-ctas", action="store_true")
    p.add_argument("--pair-topk", action="store_true")
    p.add_argument("--load-selected-bias", action="store_true")
    p.add_argument("--collapse-columns", action="store_true")
    p.add_argument("--skip-unused-quant", action="store_true")
    p.add_argument("--partial-norm", action="store_true")
    p.add_argument("--single-sample-tail", action="store_true")
    p.add_argument("--batch-tail", action="store_true")
    p.add_argument("--direct-acc", action="store_true")
    p.add_argument("--serving-tune", action="store_true")
    p.add_argument("--native-dense", action="store_true")
    p.add_argument("--native-unroll", type=int, default=1)
    p.add_argument("--native-router-unroll", type=int, default=0)
    p.add_argument("--native-buffer", action="store_true")
    p.add_argument("--native-slice-lds", action="store_true")
    p.add_argument("--native-b1-lds-mode", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--native-repeat-unused-columns", action="store_true")
    p.add_argument("--native-lds", action="store_true")
    p.add_argument("--dense-lds", action="store_true")
    p.add_argument("--projection-unroll", type=int, default=1)
    p.add_argument("--front32", action="store_true")
    p.add_argument("--front-unroll", type=int, default=0)
    p.add_argument("--projection-buffer", action="store_true")
    p.add_argument("--iteration-watchdog", action="store_true")
    p.add_argument("--coherent-payloads", action="store_true")
    p.add_argument("--normal-launch", action="store_true")
    p.add_argument("--flydsl-replay-input", action="store_true")
    p.add_argument("--mfma-lanes", action="store_true")
    p.add_argument("--poll-sleep", type=int, choices=[0, 1, 2, 4], default=0)
    p.add_argument("--tail-rows", type=int, choices=[16, 32, 64], default=64)
    args = p.parse_args()
    assert (
        not args.native_dense and not args.flydsl_replay_input
    ), "Historical ablations are not included"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=args.world, join=True)
