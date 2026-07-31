# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Cross-CTA global-memory ring correctness SPIKE (gfx1250, Task 0).

Proves that a producer-CTA / consumer-CTA hand-off over a global-memory ring
works on gfx1250 through FlyDSL, using ONLY the flydsl_prims primitives that
already back production dispatch (system-scope atomics / release+acquire fences
/ volatile spin) -- NO named barriers, NO intra-CTA warp specialization.

Model (deterministic grid-stride task partition; monotonic per-slot generation):
  grid = P_BLK producer CTAs + C_BLK consumer CTAs (bid < P_BLK == producer).
  task t in [0, G): slot = t % S, gen = t // S + 1.
    producer block (t % P_BLK owns t via `range(bid, G, P_BLK)`):
        spin freed[slot] == gen-1;  ring_a[slot] = t*7+13;
        fence_release; ready[slot] = gen  (release store).
    consumer block (t % C_BLK owns t via `range(bid-P_BLK, G, C_BLK)`):
        spin ready[slot] == gen;  fence_acquire; v = ring_a[slot];
        consumed[t] = v;  fence_release; freed[slot] = gen.
  Requires S >= C_BLK (else a slot is reclaimed while still needed -> hang).
  Grid must be co-resident (P_BLK+C_BLK small) -- spin-wait across CTAs assumes
  all blocks are scheduled; keep totals well under CU count.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, T
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import Int32, Int64

from aiter.ops.flydsl.dispatch_combine_v2 import flydsl_prims as P
from aiter.ops.flydsl.dispatch_combine_v2.intranode_kernels import WAVE


def _make_ring_spike(*, S, P_BLK, C_BLK):
    grid_blocks = P_BLK + C_BLK

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def ring_spike(
        a_addr: Int64,
        ready_addr: Int64,
        freed_addr: Int64,
        consumed_addr: Int64,
        G: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        rsrc_a = create_buffer_resource_from_addr(a_addr)
        rsrc_consumed = create_buffer_resource_from_addr(consumed_addr)

        if tid == 0:
            if bid < P_BLK:
                # ── producer CTA: owns tasks {bid, bid+P_BLK, ...} ──
                for t in range(bid, G, P_BLK):
                    slot = t % S
                    prev_gen = t // S            # == gen - 1
                    gen = prev_gen + 1
                    freed_slot = fx.Int64(freed_addr) + fx.Int64(slot) * fx.Int64(4)
                    # wait until the previous occupant of this slot was consumed
                    P.spin_until_eq_i32(freed_slot, prev_gen)
                    buffer_store(t * 7 + 13, rsrc_a, slot)
                    P.fence_system_release()
                    P.store_i32_system(ready_addr, slot, gen)
            else:
                # ── consumer CTA: owns tasks {bid-P_BLK, +C_BLK, ...} ──
                for t in range(bid - P_BLK, G, C_BLK):
                    slot = t % S
                    gen = t // S + 1
                    ready_slot = fx.Int64(ready_addr) + fx.Int64(slot) * fx.Int64(4)
                    P.spin_until_eq_i32(ready_slot, gen)
                    P.fence_system_acquire()
                    v = buffer_load(rsrc_a, slot, vec_width=1, dtype=T.i32())
                    buffer_store(v, rsrc_consumed, t)
                    P.fence_system_release()
                    P.store_i32_system(freed_addr, slot, gen)

    @flyc.jit
    def run(
        a_addr: Int64,
        ready_addr: Int64,
        freed_addr: Int64,
        consumed_addr: Int64,
        G: Int32,
        stream=fx.Stream(None),
    ):
        ring_spike(a_addr, ready_addr, freed_addr, consumed_addr, G).launch(
            grid=(grid_blocks, 1, 1),
            block=[WAVE, 1, 1],
            stream=stream,
        )

    return run


_CACHE = {}


def run_crosscta_ring_spike(*, G, S, P_BLK, C_BLK):
    """Launch the spike and return consumed[G] (int32, cuda)."""
    import torch

    assert S >= C_BLK, f"ring depth S={S} must be >= consumer CTAs C_BLK={C_BLK}"
    dev = torch.device("cuda")
    ring_a = torch.zeros(S, dtype=torch.int32, device=dev)
    ready = torch.zeros(S, dtype=torch.int32, device=dev)
    freed = torch.zeros(S, dtype=torch.int32, device=dev)
    consumed = torch.full((G,), -1, dtype=torch.int32, device=dev)

    key = (S, P_BLK, C_BLK)
    run = _CACHE.get(key)
    if run is None:
        run = _make_ring_spike(S=S, P_BLK=P_BLK, C_BLK=C_BLK)
        _CACHE[key] = run

    run(
        ring_a.data_ptr(),
        ready.data_ptr(),
        freed.data_ptr(),
        consumed.data_ptr(),
        int(G),
    )
    torch.cuda.synchronize()
    return consumed
