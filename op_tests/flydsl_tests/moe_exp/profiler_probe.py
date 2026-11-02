# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Does the torch profiler record GPU kernels on this node?

profiler_us (the test's own timer) comes from torch profiler device events. If a run
prints ``profiler_us = n/a`` or the test prints ``us = nan``, run this (a GPU process:
under LOCK on a shared node). It profiles 128 copies and 128 bf16 matmuls with
AITER run_perftest's profiler settings and prints how many device events came back.
Healthy: 256 events per trial. 0 = the profiler's kernel tracing is broken on this
node/boot (seen 2026-11-01 on heliosr-1b114-d01-3, MEC 2430, while rocprofv3
--kernel-trace still recorded every kernel); only graph_us is then valid.
The cause there: after warm reboots the GPU's REALTIME counter (kernel completion
timestamps) ran 38.8 s per reboot ahead of KFD's clock counter (which ROCr uses to
convert GPU ticks to host time), so every kernel record fell outside the profiler
window. A power cycle re-aligned them (2026-11-01 23:42 UTC boot: 256 of 256).

usage: python op_tests/flydsl_tests/moe_exp/profiler_probe.py   (no aiter import)
"""

import collections

import torch
import torch.profiler as tpf

x = torch.empty(64 << 20, dtype=torch.float16, device="cuda").normal_()  # 128 MiB
y = torch.empty_like(x)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(5):
    y.copy_(x)
    a @ a
torch.cuda.synchronize()
for trial in range(3):
    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False,
        with_stack=False,
        with_modules=True,
    ) as prof:
        for _ in range(128):
            y.copy_(x)
        for _ in range(128):
            a @ a
        torch.cuda.synchronize()
    cnt = collections.Counter()
    dur = collections.defaultdict(float)
    for e in prof.events():
        if str(e.device_type).endswith("CUDA"):
            cnt[e.name] += 1
            dur[e.name] += e.self_device_time_total
    print(f"trial {trial}: {sum(cnt.values())} device events (expect 256)")
    for n, c in cnt.most_common(4):
        print(f"   {c:4d} x {n[:70]}  avg {dur[n] / c:.2f} us")
