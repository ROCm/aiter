# SPDX-License-Identifier: MIT
"""Validate C1 allocation/region addressing without launching fused GEMMs.

Use torchrun with eight ranks on each of two or four MI355 hosts. The logical
layout can use four GPUs per node. Large buffers stay on the test hosts. --plan-only
checks host layout invariants without initializing a process group or GPU.
"""

import argparse
import importlib
import json
import os
import sys
import types
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def _modules(kernel_dir):
    # Load only the layout/resource modules, so this transport probe also runs
    # with the isolated MORI runtime before the full AITER/FlyDSL JIT is ready.
    package = types.ModuleType("_megamoe_c1_probe")
    package.__path__ = [str(kernel_dir)]
    sys.modules[package.__name__] = package
    return (
        importlib.import_module(f"{package.__name__}.layout"),
        importlib.import_module(f"{package.__name__}.cco_buffers"),
    )


def _make_plan(layout, buffers, model, tokens, world, gpus, quant):
    # Match NETWORKS in test_megamoe_internode.py: K3 and V4.1 Flash.
    hidden, inter, experts, topk = {
        "kimi_k3": (3584, 3072, 896, 16),
        "dsv41_flash": (5120, 2304, 384, 6),
    }[model]
    s1 = layout.Stage1ArenaLayout.create(
        hidden=hidden,
        inter=inter,
        experts=experts,
        world_size=world,
        gpus_per_node=gpus,
        topk=topk,
        max_tokens=tokens,
        tile_group=4,
        max_recv_routes_per_rank=3 * tokens * topk // 2,
        activation_dtype="fp8" if quant == "a8w4" else "fp4",
    )
    s2 = layout.Stage2ArenaLayout.create(
        hidden=hidden,
        topk=topk,
        max_tokens=tokens,
        world_size=world,
        gpus_per_node=gpus,
        source_nodes=world // gpus,
        rail_record_bytes=hidden + hidden // 32,
    )
    plan = buffers.CcoBufferPlan.create(s1, s2)
    expected = {
        (stage, r.name): r
        for stage, arena in ((1, s1), (2, s2))
        for r in arena.regions
        if r.name != "stage2_embed"
    }
    actual = {(r.stage, r.name): r for w in plan.windows for r in w.regions}
    if expected.keys() != actual.keys() or len(actual) != sum(
        len(w.regions) for w in plan.windows
    ):
        raise AssertionError("missing/duplicate regions in buffer plan")
    for key, old in expected.items():
        new = actual[key]
        if (old.nbytes, old.shape, old.dtype) != (new.nbytes, new.shape, new.dtype):
            raise AssertionError(f"region geometry changed: {key}")
        if old.shape[0] == 2:
            for parity in (0, 1):
                if new.parity_offset(parity) - new.offset != parity * old.nbytes // 2:
                    raise AssertionError(f"parity stride changed: {key}")
        else:
            try:
                new.parity_offset(0)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"persistent state treated as parity indexed: {key}"
                )
    for window in plan.windows:
        end = 0
        for region in window.regions:
            if region.offset < end or region.offset % region.alignment:
                raise AssertionError(f"overlap or misalignment: {region}")
            end = region.offset + region.nbytes
        if (
            end > window.registered_bytes
            or window.registered_bytes > window.allocation_bytes
        ):
            raise AssertionError(f"region outside window: {window.name}")
        if (
            window.rdma_register
            and window.registered_bytes > buffers.MAX_CCO_RAIL_MR_BYTES
        ):
            raise AssertionError("MR exceeds the guarded CCO registration size")
    rail_sizes = {w.allocation_bytes for w in plan.windows if w.rdma_register}
    if len(rail_sizes) != 1:
        raise AssertionError("rail allocations must have equal padded sizes")
    return plan


def _summary(plan):
    return {
        "allocation_MiB": plan.allocation_bytes // 2**20,
        "windows": [
            {
                "name": w.name,
                "allocation_MiB": w.allocation_bytes // 2**20,
                "registered_MiB": w.registered_bytes // 2**20,
                "rdma": w.rdma_register,
                "regions": len(w.regions),
            }
            for w in plan.windows
        ],
    }


class _CudaBytes:
    def __init__(self, ptr, count):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": (count,),
            "typestr": "|u1",
            "version": 3,
        }


def _view(ptr, count):
    return torch.as_tensor(_CudaBytes(ptr, count), device="cuda")


def _transfer_launcher(qp, *, rail_team=False, caller_ordered=False):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import mori.cco.device.flydsl as cco

    default_stream = fx.Stream(None)
    team_options = {"team": cco.Team.GDA} if rail_team else {}
    post_options = dict(team_options)
    flush_options = dict(team_options)
    handle_options = {}
    if caller_ordered:
        # This probe has one warp posting and flushing one transfer per QP.
        # The post returns before that same warp flushes, satisfying PR895's
        # caller-ordered publication contract without a separate CTA counter.
        if hasattr(cco, "GdaDoorbell"):
            handle_options["doorbell"] = cco.GdaDoorbell.CALLER_ORDERED
        else:
            post_options.update(defer_doorbell=True, posted_before_flush=True)
            flush_options.update(posted_before_flush=True)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def transfer(
        dev_comm: fx.Int64,
        peer: fx.Int32,
        dst_win: fx.Int64,
        dst_off: fx.Int64,
        src_win: fx.Int64,
        src_off: fx.Int64,
        nbytes: fx.Int64,
    ):
        gda = cco.DevComm(dev_comm).gda(qp, **handle_options)
        gda.put(
            peer,
            dst_win,
            dst_off,
            src_win,
            src_off,
            nbytes,
            coop=cco.CoopScope.WARP,
            **post_options,
        )
        gda.flush_peer(peer, coop=cco.CoopScope.WARP, **flush_options)

    @flyc.jit
    def launch(
        dev_comm: fx.Int64,
        peer: fx.Int32,
        dst_win: fx.Int64,
        dst_off: fx.Int64,
        src_win: fx.Int64,
        src_off: fx.Int64,
        nbytes: fx.Int64,
        stream=default_stream,
    ):
        transfer(dev_comm, peer, dst_win, dst_off, src_win, src_off, nbytes).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


def _check_local(storage, rank, lane, generation):
    samples = []
    for window in storage.plan.windows:
        for region in window.regions:
            depth = region.parity_depth if region.shape[0] == region.parity_depth else 1
            plane = region.nbytes // depth
            # Distinct markers at the head/tail must not overlap. A 13 KiB
            # metadata plane cannot hold two independent 8 KiB samples.
            width = plane if plane <= 8192 else min(8192, plane // 2)
            for parity in range(depth):
                start = region.parity_offset(parity) if depth > 1 else region.offset
                for offset in sorted({start, start + plane - width}):
                    samples.append((window.name, offset, width))
    own = [
        _view(storage.windows[name].local_ptr + off, width)
        for name, off, width in samples
    ]
    for shift in range(1, 8):
        for view in own:
            view.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        target_lane = (lane + shift) % 8
        for index, (name, off, width) in enumerate(samples):
            spec = next(w for w in storage.plan.windows if w.name == name)
            dev = (
                storage.rail_dev_comm if spec.rdma_register else storage.local_dev_comm
            )
            ptr = (
                storage.windows[name].local_ptr
                + (target_lane - lane) * dev.per_rank_size
                + off
            )
            _view(ptr, width).fill_(1 + (generation * 17 + rank * 7 + index) % 254)
        torch.cuda.synchronize()
        dist.barrier()
        source = rank - lane + (lane - shift) % 8
        for index, view in enumerate(own):
            value = 1 + (generation * 17 + source * 7 + index) % 254
            if not bool((view == value).all().item()):
                raise AssertionError(
                    f"LSA region sample {samples[index]}, source {source}"
                )
    return len(samples) * 7


def _address_launcher(addressing, region_count):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    default_stream = fx.Stream(None)

    def store_address(address, value):
        pointer_type = fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8)
        fx.ptr_store(value, fx.inttoptr(pointer_type, address))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def resolve(table: fx.Int64, output: fx.Int64):
        for entry in range(fx.thread_idx.x, region_count * 16, 256):
            region = entry // fx.Int32(16)
            parity = fx.Int64((entry // fx.Int32(8)) % fx.Int32(2))
            peer = entry % fx.Int32(8)
            handle, offset = addressing.region_window_address(table, region, parity)
            dst = output + fx.Int64(entry) * fx.Int64(32)
            store_address(dst, addressing.region_local_address(table, region, parity))
            store_address(
                dst + fx.Int64(8),
                addressing.region_lsa_address(table, region, parity, peer),
            )
            store_address(dst + fx.Int64(16), handle)
            store_address(dst + fx.Int64(24), offset)

    @flyc.jit
    def launch(table: fx.Int64, output: fx.Int64, stream=default_stream):
        resolve(table, output).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def _check_device_addresses(storage, lane):
    addressing = importlib.import_module("_megamoe_c1_probe.buffer_addressing")
    regions = storage.plan.regions
    result = torch.empty(
        (len(regions) * 16, 4), dtype=torch.int64, device=storage.device
    )
    _address_launcher(addressing, len(regions))(
        storage.device_table_ptr, result.data_ptr()
    )
    expected = []
    for region in regions:
        window = storage.windows[region.window]
        spec = next(w for w in storage.plan.windows if w.name == region.window)
        dev = storage.rail_dev_comm if spec.rdma_register else storage.local_dev_comm
        for parity in (0, 1):
            offset = region.offset + parity * region.plane_bytes
            local = window.local_ptr + offset
            for peer in range(8):
                expected.append(
                    (
                        local,
                        local + (peer - lane) * dev.per_rank_size,
                        window.handle,
                        offset,
                    )
                )
    observed = result.cpu()
    if not torch.equal(observed, torch.tensor(expected, dtype=torch.int64)):
        mismatch = (observed != torch.tensor(expected, dtype=torch.int64)).nonzero()[0]
        raise AssertionError(f"GPU region address resolution: {mismatch.tolist()}")
    return result.numel()


def _table_transfer_launcher(qp, *, rail_team, caller_ordered):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import mori.cco.device.flydsl as cco

    addressing = importlib.import_module("_megamoe_c1_probe.buffer_addressing")
    default_stream = fx.Stream(None)
    options = {"team": cco.Team.GDA} if rail_team else {}
    post_options, flush_options, handle_options = dict(options), dict(options), {}
    if caller_ordered:
        if hasattr(cco, "GdaDoorbell"):
            handle_options["doorbell"] = cco.GdaDoorbell.CALLER_ORDERED
        else:
            post_options.update(defer_doorbell=True, posted_before_flush=True)
            flush_options.update(posted_before_flush=True)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def transfer(
        dev_comm: fx.Int64,
        peer: fx.Int32,
        table: fx.Int64,
        dst_region: fx.Int32,
        dst_delta: fx.Int64,
        src_region: fx.Int32,
        src_delta: fx.Int64,
        parity: fx.Int64,
        nbytes: fx.Int64,
    ):
        dst_win, dst_off = addressing.region_window_address(table, dst_region, parity)
        src_win, src_off = addressing.region_window_address(table, src_region, parity)
        gda = cco.DevComm(dev_comm).gda(qp, **handle_options)
        gda.put(
            peer,
            dst_win,
            dst_off + dst_delta,
            src_win,
            src_off + src_delta,
            nbytes,
            coop=cco.CoopScope.WARP,
            **post_options,
        )
        gda.flush_peer(peer, coop=cco.CoopScope.WARP, **flush_options)

    @flyc.jit
    def launch(
        dev_comm: fx.Int64,
        peer: fx.Int32,
        table: fx.Int64,
        dst_region: fx.Int32,
        dst_delta: fx.Int64,
        src_region: fx.Int32,
        src_delta: fx.Int64,
        parity: fx.Int64,
        nbytes: fx.Int64,
        stream=default_stream,
    ):
        transfer(
            dev_comm,
            peer,
            table,
            dst_region,
            dst_delta,
            src_region,
            src_delta,
            parity,
            nbytes,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def _check_rail(
    storage,
    rank,
    peer,
    generation,
    *,
    receive_peer=None,
    rail_team=False,
    caller_ordered=False,
):
    # Copy 8 KiB at a high node_accumulator offset into remote_partial_rx.
    # DS large layouts put these regions in different MRs, exercising both
    # the source lkey and target rkey independently, through every GDA context.
    source = storage.plan.region(2, "node_accumulator")
    target = storage.plan.region(2, "remote_partial_rx")
    width = min(8192, source.nbytes // 2, target.nbytes // 2)
    checks = 0
    # Team.GDA on a physical RAIL DevComm addresses the peer's physical node,
    # while expected markers continue to use the peer's world rank.
    transport_peer = peer // 8 if rail_team else peer
    # A directed ring is a permutation: each rank sends to one node and
    # receives from one node. They coincide only with two physical nodes.
    receive_peer = peer if receive_peer is None else receive_peer
    # Every rail region/parity and the registered prefix tail must receive a
    # NIC write. Keep the eight-byte source away from those target samples.
    src_off = source.offset + source.nbytes // 4
    targets = set()
    for window in storage.plan.windows:
        if window.rdma_register:
            targets.add((window.name, window.registered_bytes - 8))
            for region in window.regions:
                for parity in (0, 1):
                    targets.add(
                        (
                            window.name,
                            region.parity_offset(parity) + region.nbytes // 2 - 8,
                        )
                    )
    views = [
        _view(storage.windows[name].local_ptr + offset, 8)
        for name, offset in sorted(targets)
    ]
    for view in views:
        view.zero_()
    _view(storage.windows[source.window].local_ptr + src_off, 8).fill_(1 + rank)
    torch.cuda.synchronize()
    dist.barrier()
    launch = _transfer_launcher(0, rail_team=rail_team, caller_ordered=caller_ordered)
    for name, offset in sorted(targets):
        launch(
            storage.rail_dev_comm.ptr,
            transport_peer,
            storage.windows[name].handle,
            offset,
            storage.windows[source.window].handle,
            src_off,
            8,
        )
    torch.cuda.synchronize()
    dist.barrier()
    for index, view in enumerate(views):
        if not bool((view == 1 + receive_peer).all().item()):
            raise AssertionError(f"rail tail {sorted(targets)[index]}")
        checks += 1
    for qp in range(storage.gda_context_count):
        launch = _table_transfer_launcher(
            qp, rail_team=rail_team, caller_ordered=caller_ordered
        )
        for parity in (0, 1):
            src_off = source.parity_offset(parity) + source.nbytes // 2 - width
            dst_off = target.parity_offset(parity) + target.nbytes // 2 - width
            src = _view(storage.windows[source.window].local_ptr + src_off, width)
            dst = _view(storage.windows[target.window].local_ptr + dst_off, width)
            src.fill_(1 + (generation * 17 + rank * 7 + qp) % 254)
            dst.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            launch(
                storage.rail_dev_comm.ptr,
                transport_peer,
                storage.device_table_ptr,
                storage.plan.region_index(2, "remote_partial_rx"),
                target.nbytes // 2 - width,
                storage.plan.region_index(2, "node_accumulator"),
                source.nbytes // 2 - width,
                parity,
                width,
            )
            torch.cuda.synchronize()
            dist.barrier()
            expected = 1 + (generation * 17 + receive_peer * 7 + qp) % 254
            if not bool((dst == expected).all().item()):
                raise AssertionError(f"rail payload qp={qp}, parity={parity}")
            checks += 1
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "aiter/ops/flydsl/kernels/mega_moe_internode",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("kimi_k3", "dsv41_flash"),
        default=["kimi_k3", "dsv41_flash"],
    )
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--quant", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--logical-peer-windows",
        action="store_true",
        help="check all logical peer payload/ready/credit slices in the operator windows",
    )
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument(
        "--rail-team",
        action="store_true",
        help="address physical rail peers with PR895 Team.GDA",
    )
    parser.add_argument(
        "--caller-ordered",
        action="store_true",
        help="exercise PR895 deferred posts and caller-ordered flushes",
    )
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("generations must be positive")
    if args.delay_ms < 0 or not -1 <= args.delay_rank < args.world_size:
        parser.error("invalid delay rank/time")
    if args.logical_peer_windows and (not args.rail_team or not args.caller_ordered):
        parser.error("logical peer windows require --rail-team --caller-ordered")
    layout, buffers = _modules(args.kernel_dir)
    plans = [
        (
            model,
            _make_plan(
                layout,
                buffers,
                model,
                args.tokens,
                args.world_size,
                args.gpus_per_node,
                args.quant,
            ),
        )
        for model in args.models
    ]
    if args.plan_only:
        for model, plan in plans:
            if args.logical_peer_windows:
                peer_layout = importlib.import_module("_megamoe_c1_probe.peer_layout")
                plan, _ = peer_layout.add_bridge_window(plan, buffers)
            print(json.dumps({"model": model, **_summary(plan)}), flush=True)
        return
    from mori.cco import Communicator, UniqueId

    rank, lane = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    expected_bitcode_ccqe = os.environ.get("CCO_TEST_BITCODE_CCQE")
    if expected_bitcode_ccqe is not None:
        from mori.jit.core import is_ccqe_enabled

        if expected_bitcode_ccqe not in ("0", "1"):
            raise ValueError("CCO_TEST_BITCODE_CCQE must be 0 or 1")
        actual_ccqe = is_ccqe_enabled()
        if actual_ccqe != bool(int(expected_bitcode_ccqe)):
            raise RuntimeError(
                f"explicit device bitcode CCQE={expected_bitcode_ccqe} disagrees "
                f"with this host's Ionic CQ mode CCQE={int(actual_ccqe)}"
            )
        if lane == 0:
            print(f"CCO bitcode/CQ mode matched: CCQE={int(actual_ccqe)}", flush=True)
    expected_library_dir = os.environ.get("CCO_TEST_LIBRARY_DIR")
    if expected_library_dir:
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libmori_cco.so" in line
            }
        )
        expected_library = str(Path(expected_library_dir) / "libmori_cco.so")
        if libraries != [expected_library]:
            raise RuntimeError(f"unexpected CCO runtime libraries: {libraries}")
        if rank == 0:
            print(f"CCO runtime libraries: {libraries}", flush=True)
    torch.cuda.set_device(lane)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        if (
            dist.get_world_size() not in (16, 32)
            or args.world_size != dist.get_world_size()
            or lane != rank % 8
        ):
            raise ValueError(
                "GPU probe requires two or four physical hosts with eight ranks each"
            )
        physical_nodes = dist.get_world_size() // 8
        node = rank // 8
        for model, plan in plans:
            if rank == 0:
                print(json.dumps({"model": model, **_summary(plan)}), flush=True)
            with ExitStack() as stack:
                if args.logical_peer_windows:
                    from megamoe_peer_window_probe import (
                        add_bridge_window,
                        check_logical_peers,
                    )

                    plan, bridge_area = add_bridge_window(plan, buffers)
                comms = []
                for rail in (False, True):
                    payload = [
                        bytes(Communicator.get_unique_id()) if rank == 0 else None
                    ]
                    dist.broadcast_object_list(payload, src=0)
                    comms.append(
                        stack.enter_context(
                            Communicator.init(
                                args.world_size,
                                rank,
                                UniqueId.from_bytes(payload[0]),
                                per_rank_vmm=plan.per_rank_vmm(rail=rail),
                            )
                        )
                    )
                storage = stack.enter_context(buffers.CcoBufferSet(plan, *comms))
                if args.logical_peer_windows:
                    for window in plan.windows:
                        _view(
                            storage.windows[window.name].local_ptr,
                            window.allocation_bytes,
                        ).zero_()
                    torch.cuda.synchronize()
                    dist.barrier()
                print(
                    f"rank {rank} {model}: allocated {len(storage.windows)} windows",
                    flush=True,
                )
                checks = _check_device_addresses(storage, lane)
                if rank == 0:
                    print(f"C1 {model}: GPU region addresses PASS", flush=True)
                # With logical transfers, repeat the established C1 preflight
                # twice and spend further generations on peer reuse/credits.
                c1_generations = (
                    min(args.generations, 2)
                    if args.logical_peer_windows
                    else args.generations
                )
                for generation in range(1, c1_generations + 1):
                    checks += _check_local(storage, rank, lane, generation)
                    if rank == 0:
                        print(
                            f"C1 {model}: generation {generation} LSA PASS", flush=True
                        )
                    # Every nonzero ring shift covers a different directed
                    # physical-node pair without two senders sharing an RX tail.
                    for shift in range(1, physical_nodes):
                        checks += _check_rail(
                            storage,
                            rank,
                            ((node + shift) % physical_nodes) * 8 + lane,
                            generation,
                            receive_peer=((node - shift) % physical_nodes) * 8 + lane,
                            rail_team=args.rail_team,
                            caller_ordered=args.caller_ordered,
                        )
                    if rank == 0:
                        print(
                            f"C1 {model}: generation {generation} rail PASS", flush=True
                        )
                if args.logical_peer_windows:
                    # The preceding C1 tests intentionally write metadata markers;
                    # start peer generations from a clean control baseline once.
                    for window in plan.windows:
                        _view(
                            storage.windows[window.name].local_ptr,
                            window.allocation_bytes,
                        ).zero_()
                    torch.cuda.synchronize()
                    dist.barrier()
                    peer_checks = 0
                    for generation in range(1, args.generations + 1):
                        for stage in (1, 2):
                            peer_checks += check_logical_peers(
                                storage,
                                rank,
                                stage,
                                generation,
                                bridge_area,
                                delay_rank=args.delay_rank,
                                delay_ms=args.delay_ms,
                            )
                    peer_total = torch.tensor([peer_checks], dtype=torch.int64)
                    dist.all_reduce(peer_total)
                    checks += peer_checks
                    if rank == 0:
                        print(
                            f"C2/C3 {model}: {int(peer_total[0])} logical peer payload/credit checks PASS",
                            flush=True,
                        )
                torch.cuda.synchronize()
                dist.barrier()
                total = torch.tensor([checks], dtype=torch.int64)
                dist.all_reduce(total)
                if rank == 0:
                    print(
                        f"C1 {model}: {int(total[0])} region/payload checks PASS",
                        flush=True,
                    )
            dist.barrier()  # Complete collective teardown before the next model.
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
