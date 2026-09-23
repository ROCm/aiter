# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-stage latency of the MegaMoE gfx1250 stage-1 (compact) path.

Reuses bench_mega_moe's setup and replaces its bench: each stage is captured
N times back to back in its own CUDA graph and timed by wall clock, so the
numbers do not depend on torch.profiler (whose device times are unreliable on
this ROCm build). Takes bench_mega_moe's CLI, e.g.

  torchrun --standalone --nproc_per_node=4 bench_mega_stage1.py \
    -q a4w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 --layers 2 -tpr 512 \
    --combine fused --dispatch_wire fp4 --stage1_fused 1 --acc_verify 0 \
    --profile_table 0 --prof_replays 0

STAGE1_REPS (default 16) sets N; STAGE1_STAGES picks a comma-separated subset
of quant,plan,dispatch,stage1,layer.
"""

import os
import sys
import time

import flydsl.expr as fx
import torch

import bench_mega_moe as B
from aiter.ops.quant import dynamic_per_group_scaled_quant

_REPS = int(os.environ.get("STAGE1_REPS", "16"))
_STAGES = os.environ.get("STAGE1_STAGES", "quant,plan,dispatch,stage1,layer").split(",")

_orig_capture = B.DeviceMoEPipeline.capture


def _capture(self, x0):
    self._x0 = x0
    _orig_capture(self, x0)


class _NoProf:
    def key_averages(self):
        return []


def _bench(self, warmup=3, iters=10, prof_replays=0):
    m, dc = self.mega, self.dist_ctx
    x0 = self._x0
    ids, wts = self.routings[0]
    T = x0.shape[0]
    xn = B._rmsnorm(x0)

    def quant():
        dynamic_per_group_scaled_quant(
            m._dispatch_quant_payload[:T],
            xn,
            m._dispatch_quant_scales[:T],
            32,
            shuffle_scale=False,
        )

    def plan():
        m._begin_compact_step(m._compact_recv_bound(T, None))
        m._launch_compact_plan_async(ids, T)
        m._wait_compact_plan()

    def dispatch():
        spec = m._select_dispatch(T)
        m._dispatch_variants[spec](
            m._arena.handle,
            m._dispatch_quant_payload.data_ptr(),
            ids.data_ptr(),
            wts.data_ptr(),
            m._tok_maps[m._compact_slot].data_ptr(),
            m._destination_peer_counter.data_ptr(),
            m._dispatch_barrier.data_ptr(),
            m._total_recv.data_ptr(),
            m._config.rank,
            T,
            fx.Stream(torch.cuda.current_stream()),
        )

    def stage1():
        m._begin_compact_step(m._compact_recv_bound(T, None))
        m._dispatch(xn, wts, ids)

    def layer():
        y = m(
            xn,
            wts,
            ids,
            w1=self.w1_a,
            w2=self.w2_a,
            w1_scale=self.w1_s,
            w2_scale=self.w2_s,
            next_topk_ids=None,
            combine_quant=self.combine_quant,
        )
        return x0 + y

    fns = {
        "quant": quant,
        "plan": plan,
        "dispatch": dispatch,
        "stage1": stage1,
        "layer": layer,
    }

    def time_graph(g):
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        self.comm.barrier()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6 / _REPS)
        self.comm.barrier()
        samples.sort()
        return samples[len(samples) // 2], samples[0]

    self.graph = None
    stage1()
    torch.cuda.synchronize()
    self.comm.barrier()
    for name in _STAGES:
        fn = fns[name]
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        self.comm.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(_REPS):
                fn()
        torch.cuda.synchronize()
        self.comm.barrier()
        med, mn = time_graph(g)
        del g
        med = dc.allreduce_avg_float(med)
        mn = dc.allreduce_avg_float(mn)
        if dc.rank == 0:
            print(
                f"[stage1] tpr={T} {name:9s} median {med:8.2f} us  min {mn:8.2f} us",
                flush=True,
            )
    self._prof = _NoProf()
    return {"min": 1.0, "median": 1.0, "mean": 1.0, "max": 1.0}, 0.0


if __name__ == "__main__":
    B.DeviceMoEPipeline.capture = _capture
    B.DeviceMoEPipeline.bench = _bench
    sys.argv[0] = "bench_mega_moe.py"
    B.main()
