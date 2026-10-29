# SPDX-License-Identifier: MIT
"""A test launch of the MoE front stages alone (xq, router, route, table) on
upstream's placements, for compile probes and region diffs."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.epoch import epoch_begin, load_epoch
from deepseek_v4_gluon.common.sync import SMEM_FLAT, T512, deadline, poll1, fail
from deepseek_v4_gluon.v41 import moe_front as mf
from deepseek_v4_gluon.v41.ffn import first_task


@gluon.jit
def k_front(x, gate_w, bias, scratch, epoch, health, rank, tabdump,
            S: gl.constexpr, HIDDEN: gl.constexpr, EXPERTS: gl.constexpr,
            TOPK: gl.constexpr, GROUPS: gl.constexpr, PER: gl.constexpr,
            ROUTER0: gl.constexpr, ROUTE0: gl.constexpr, ROUTER_TASKS: gl.constexpr,
            MAX_U: gl.constexpr, MULTI: gl.constexpr, SLOT_TILES: gl.constexpr,
            EXPERT_SLOT0: gl.constexpr, ORDER0: gl.constexpr, FIRST0: gl.constexpr,
            MASK0: gl.constexpr, TILE_FIRST0: gl.constexpr, TILE_SLOT0: gl.constexpr,
            TABW: gl.constexpr, X8R: gl.constexpr, X8RS: gl.constexpr, X8: gl.constexpr,
            X8S: gl.constexpr, X8RDY: gl.constexpr, LOGIT: gl.constexpr,
            ROUTE: gl.constexpr, XRDY: gl.constexpr):  # fmt: skip
    tid = gl.arange(0, 512, layout=T512)
    lane = tid % 64
    wave = tid // 64
    bid = gl.program_id(0)
    ep = load_epoch(epoch)
    epoch_begin(epoch, ep, bid, tid)
    tag = (ep << 1) + 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    tab = gl.allocate_shared_memory(gl.int32, [TABW], SMEM_FLAT)
    sb = scratch.to(gl.uint64)
    for t in range(first_task(bid, 0), S, 256):
        mf.stage_xq(t, x, sb + X8R, sb + X8RS, sb + X8, sb + X8S, sb + X8RDY, tag, tid,
                    HIDDEN)
    for task in range(first_task(bid, ROUTER0), ROUTER_TASKS, 256):
        mf.stage_router(task, gate_w, x, sb + LOGIT, red, tag, tid, S, EXPERTS, GROUPS,
                        PER, HIDDEN)
    for t in range(first_task(bid, ROUTE0), S, 256):
        mf.stage_route(t, sb + LOGIT, sb + ROUTE, bias, tag, tid, lane, wave, dl, health,
                       rank, S, EXPERTS, TOPK, 1.5)
    mf.load_route(tab, red, sb + ROUTE, bid, tag, tid, lane, wave, dl, health, rank, S,
                  TOPK, EXPERTS, MAX_U, MULTI, SLOT_TILES, EXPERT_SLOT0, ORDER0, FIRST0,
                  MASK0, TILE_FIRST0, TILE_SLOT0, TABW - 1)
    if bid == 0:
        for pas in gl.static_range(0, TABW, 512):
            idx = gl.minimum(pas + tid, TABW - 1)
            gl.store(tabdump.to(gl.pointer_type(gl.int32)) + idx, tab.gather(idx, 0))
