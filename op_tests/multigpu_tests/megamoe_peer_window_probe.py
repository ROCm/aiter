# SPDX-License-Identifier: MIT
"""Logical-peer payload and credit checks in the actual C1 operator windows.

Imported by probe_megamoe_cco_buffers; no GEMM or forward claim follows from
these transfers. A bounded extra rail window holds relay payloads so a relay
cannot overwrite an unrelated peer's RX slice.
"""

import importlib
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
import torch
import torch.distributed as dist
from probe_megamoe_virtual4_transport import _load_flag, _store_flag

_DEFAULT_STREAM = fx.Stream(None)


def add_bridge_window(plan, buffers):
    peer_layout = importlib.import_module("_megamoe_c1_probe.peer_layout")
    return peer_layout.add_bridge_window(plan, buffers)


def _pointer(address):
    return fx.inttoptr(
        fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), address
    )


@flyc.jit
def _wait(pointer, generation):
    seen = fx.Int64(_load_flag(pointer))
    while seen < generation:
        seen = fx.Int64(_load_flag(pointer))


@flyc.kernel(known_block_size=[256, 1, 1])
def _transfer(
    dev_comm: fx.Int64,
    src_win: fx.Int64,
    dst_win: fx.Int64,
    ready_win: fx.Int64,
    src_lsa: fx.Int32,
    dst_lsa: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    ready_off: fx.Int64,
    nbytes: fx.Int64,
    generation: fx.Int64,
    qp: fx.Int32,
    rail_peer: fx.Int32,
    use_rail: fx.Int32,
    credit_ptr: fx.Int64,
    credit_generation: fx.Int64,
):
    tid = fx.thread_idx.x
    if tid < fx.Int32(64):
        _wait(credit_ptr, credit_generation)
    fx.barrier()
    if use_rail != fx.Int32(0):
        if tid < fx.Int32(64):
            gda = cco.DevComm(dev_comm).gda(0, doorbell=cco.GdaDoorbell.CALLER_ORDERED)
            gda.put(
                rail_peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                coop=cco.CoopScope.WARP,
                team=cco.Team.GDA,
                ctx=qp,
            )
            gda.put_value(
                rail_peer,
                ready_win,
                ready_off,
                generation,
                coop=cco.CoopScope.WARP,
                team=cco.Team.GDA,
                ctx=qp,
            )
            gda.flush_peer(
                rail_peer, coop=cco.CoopScope.WARP, team=cco.Team.GDA, ctx=qp
            )
    else:
        src = fx.Int64(cco.Window(src_win).lsa_ptr(src_lsa, src_off))
        dst = fx.Int64(cco.Window(dst_win).lsa_ptr(dst_lsa, dst_off))
        for word in range(tid, fx.Int32(nbytes // fx.Int64(8)), fx.Int32(256)):
            delta = fx.Int64(word) * fx.Int64(8)
            fx.ptr_store(fx.ptr_load(_pointer(src + delta)), _pointer(dst + delta))
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        if tid == fx.Int32(0):
            _store_flag(cco.Window(ready_win).lsa_ptr(dst_lsa, ready_off), generation)


@flyc.kernel(known_block_size=[256, 1, 1])
def _relay(
    bridge_win: fx.Int64,
    dst_win: fx.Int64,
    ready_win: fx.Int64,
    src_lsa: fx.Int32,
    dst_lsa: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    ready_off: fx.Int64,
    nbytes: fx.Int64,
    bridge_ready_off: fx.Int64,
    bridge_generation: fx.Int64,
    generation: fx.Int64,
):
    tid = fx.thread_idx.x
    src = fx.Int64(cco.Window(bridge_win).lsa_ptr(src_lsa, src_off))
    if tid < fx.Int32(64):
        _wait(
            cco.Window(bridge_win).lsa_ptr(src_lsa, bridge_ready_off), bridge_generation
        )
    fx.barrier()
    dst = fx.Int64(cco.Window(dst_win).lsa_ptr(dst_lsa, dst_off))
    for word in range(tid, fx.Int32(nbytes // fx.Int64(8)), fx.Int32(256)):
        delta = fx.Int64(word) * fx.Int64(8)
        fx.ptr_store(fx.ptr_load(_pointer(src + delta)), _pointer(dst + delta))
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    if tid == fx.Int32(0):
        _store_flag(cco.Window(ready_win).lsa_ptr(dst_lsa, ready_off), generation)


@flyc.jit
def _launch_transfer(
    dev_comm: fx.Int64,
    src_win: fx.Int64,
    dst_win: fx.Int64,
    ready_win: fx.Int64,
    src_lsa: fx.Int32,
    dst_lsa: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    ready_off: fx.Int64,
    nbytes: fx.Int64,
    generation: fx.Int64,
    qp: fx.Int32,
    rail_peer: fx.Int32,
    use_rail: fx.Int32,
    credit_ptr: fx.Int64,
    credit_generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    _transfer(
        dev_comm,
        src_win,
        dst_win,
        ready_win,
        src_lsa,
        dst_lsa,
        src_off,
        dst_off,
        ready_off,
        nbytes,
        generation,
        qp,
        rail_peer,
        use_rail,
        credit_ptr,
        credit_generation,
    ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.jit
def _launch_relay(
    bridge_win: fx.Int64,
    dst_win: fx.Int64,
    ready_win: fx.Int64,
    src_lsa: fx.Int32,
    dst_lsa: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    ready_off: fx.Int64,
    nbytes: fx.Int64,
    bridge_ready_off: fx.Int64,
    bridge_generation: fx.Int64,
    generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    _relay(
        bridge_win,
        dst_win,
        ready_win,
        src_lsa,
        dst_lsa,
        src_off,
        dst_off,
        ready_off,
        nbytes,
        bridge_ready_off,
        bridge_generation,
        generation,
    ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


def check_logical_peers(
    storage, rank, stage, generation, area, *, delay_rank, delay_ms
):
    from probe_megamoe_cco_buffers import _view

    package = "_megamoe_c1_probe"
    topology_module = importlib.import_module(f"{package}.virtual_topology")
    peer_layout = importlib.import_module(f"{package}.peer_layout")
    plan = storage.plan
    topology = topology_module.VirtualNodeTopology(
        plan.world_size, plan.gpus_per_node, 8
    )
    protocol = peer_layout.PeerProtocolLayout.create(plan, stage)
    node, lane = topology.logical_node(rank), rank % 8
    nodes, parity = protocol.nodes, generation % 2
    qp = (generation - 1) % storage.gda_context_count
    bridge = storage.windows["peer_bridge"]
    bridge_plane = 2 * (nodes - 1) * area + 128
    marker_off = parity * bridge_plane + 2 * (nodes - 1) * area
    marker = _view(bridge.local_ptr + marker_off, 8).view(torch.int64)
    marker.fill_(generation)
    source = plan.region(
        stage, "dispatch_staging" if stage == 1 else "node_accumulator"
    )
    width = protocol.payload.peer_bytes
    source_plane = source.plane_bytes if stage == 1 else source.plane_bytes // nodes
    # Full allocation-shaped peer payloads: no partial sample can hide an
    # overlap between peers. This does not validate the Stage2 encoded-row ABI.
    for destination in range(nodes):
        if destination == node:
            continue
        off = source.parity_offset(parity) + (
            destination * source_plane if stage == 2 else 0
        )
        _view(storage.windows[source.window].local_ptr + off, width).fill_(
            1 + (generation * 37 + rank * 11 + (destination if stage == 2 else 0)) % 254
        )
    torch.cuda.synchronize()
    dist.barrier()

    def bridge_offset(sender_node, destination_node, bank):
        shift = (destination_node - sender_node) % nodes - 1
        return parity * bridge_plane + (bank * (nodes - 1) + shift) * area

    def send_phase(bank):
        if rank == delay_rank and bank == 0:
            time.sleep(delay_ms / 1000)
        for destination in range(nodes):
            if destination == node:
                continue
            route = topology.aligned_peer(rank, destination)
            target = protocol.payload.region if bank == 0 else protocol.credit.region
            ready = protocol.ready.region if bank == 0 else protocol.credit.region
            target_view = protocol.payload if bank == 0 else protocol.credit
            ready_view = protocol.ready if bank == 0 else protocol.credit
            target_off = target_view.offset(destination, node, parity)
            ready_off = ready_view.offset(destination, node, parity)
            target_win, ready_win = (
                storage.windows[target.window],
                storage.windows[ready.window],
            )
            count = width if bank == 0 else 8
            source_win = storage.windows[source.window] if bank == 0 else bridge
            source_off = (
                source.parity_offset(parity)
                + (destination * source_plane if stage == 2 else 0)
                if bank == 0
                else marker_off
            )
            stamp = generation
            if route.kind == "rail_relay":
                target_win = ready_win = bridge
                target_off = bridge_offset(node, destination, bank)
                ready_off = target_off + area - 8
                stamp = generation * 2 + stage
            credit_off = protocol.credit.offset(node, destination, parity)
            credit_ptr = (
                storage.windows[protocol.credit.region.window].local_ptr + credit_off
            )
            _launch_transfer(
                storage.rail_dev_comm.ptr,
                source_win.handle,
                target_win.handle,
                ready_win.handle,
                lane,
                route.destination_rank % 8,
                source_off,
                target_off,
                ready_off,
                count,
                stamp,
                qp,
                route.rail_node or 0,
                int(route.kind != "lsa"),
                credit_ptr,
                generation - 2 if bank == 0 else 0,
            )
        torch.cuda.synchronize()
        dist.barrier()
        # Every bridge rank executes only relays addressed to itself. Direct
        # RX slots stay intact while all three logical peers are in flight.
        for sender in range(plan.world_size):
            sender_node = topology.logical_node(sender)
            for destination in range(nodes):
                if sender_node == destination:
                    continue
                route = topology.aligned_peer(sender, destination)
                if route.kind != "rail_relay" or route.bridge_rank != rank:
                    continue
                target = protocol.payload if bank == 0 else protocol.credit
                ready = protocol.ready if bank == 0 else protocol.credit
                off = bridge_offset(sender_node, destination, bank)
                _launch_relay(
                    bridge.handle,
                    storage.windows[target.region.window].handle,
                    storage.windows[ready.region.window].handle,
                    lane,
                    route.destination_rank % 8,
                    off,
                    target.offset(destination, sender_node, parity),
                    ready.offset(destination, sender_node, parity),
                    width if bank == 0 else 8,
                    off + area - 8,
                    generation * 2 + stage,
                    generation,
                )
        torch.cuda.synchronize()
        dist.barrier()

    send_phase(0)
    checks = 0
    for sender_node in range(nodes):
        if sender_node == node:
            continue
        sender = sender_node * plan.gpus_per_node + rank % plan.gpus_per_node
        data = _view(
            storage.windows[protocol.payload.region.window].local_ptr
            + protocol.payload.offset(node, sender_node, parity),
            width,
        )
        expected = (
            1 + (generation * 37 + sender * 11 + (node if stage == 2 else 0)) % 254
        )
        ready = _view(
            storage.windows[protocol.ready.region.window].local_ptr
            + protocol.ready.offset(node, sender_node, parity),
            8,
        ).view(torch.int64)
        if not bool((data == expected).all().item()) or int(ready[0]) != generation:
            raise AssertionError(
                f"stage{stage} peer RX: rank={rank}, source={sender_node}, gen={generation}"
            )
        checks += 1
    send_phase(1)
    for consumer in range(nodes):
        if consumer != node:
            credit = _view(
                storage.windows[protocol.credit.region.window].local_ptr
                + protocol.credit.offset(node, consumer, parity),
                8,
            ).view(torch.int64)
            if int(credit[0]) != generation:
                raise AssertionError(
                    f"stage{stage} peer credit: rank={rank}, consumer={consumer}, gen={generation}"
                )
            checks += 1
    return checks
