# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Per-device, per-shape block/warp tuning for the cco-LSA dispatch/combine
kernels.

Geometry is picked per device (MI308X / MI300X / MI355X), then by
(world_size, hidden_dim, topk). Devices are told apart by PCI device id (MI300X
and MI308X are both gfx942, differing only in CU count), falling back to arch for
gfx950 — all from KFD sysfs, no torch/HIP dependency. block_num must stay
<= CU count; re-tune per GPU.
"""

import functools
import glob


# ── MI308X (gfx942, 80 CU) — measured 2026-07-08, EP8, from a block x warp sweep.
# Best (block, warp) is token-count dependent, so this is a per-token SCHEDULE of
# (max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp) buckets;
# the op precompiles the distinct (block, warp) variants and picks by token count.
_MI308X_SCHEDULE = (
    (256, 64, 8, 64, 4),
    (2048, 64, 16, 64, 4),
    (None, 64, 16, 80, 4),
)
# Single-shot fallback (schedule ignored) = peak-optimal.
_MI308X_DEFAULT = dict(
    dispatch_block_num=64,
    combine_block_num=80,
    warp_num_per_block=16,
    combine_warp_num_per_block=4,
    schedule=_MI308X_SCHEDULE,
)
# Per-shape -> per-dtype schedules. These were tuned as fp8-dispatch + bf16-combine,
# so they live under "fp8" (the default / fallback for any untuned dtype).
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI308X_TABLE = {
    (8, 7168, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI308X_SCHEDULE},
}

# ── MI325X (gfx942, 304 CU, DID 0x74a5) — measured 2026-07-08, EP8, full 2D block x
# warp sweep (graph, ITERS>=300, tok 8..8192), selected by min LATENCY (us). Key
# lessons vs MI308X: (a) dispatch always wants warp 8 (not 16), block grows with tok
# (64->152->228->304); (b) combine is latency-bound at small/mid tok — on this big
# 304-CU part a SMALL block (64) + warp 4 wins (the 0.5*CU block's grid-barrier cost
# dominates), only bandwidth-bound large tok (>1024) want b~0.5*CU (152) + warp 2;
# at the tiniest tok (8) combine warp 2 edges warp 4. b64 combine beats MI308X at
# every small size. Measured combine us: 8=20 64=38 128=53 256=83 512=145 1024=267
# 2048=497 4096=961 8192=1839; dispatch us: 8=18 64=30 128=47 256=78 512=138 1024=256
# 2048=490 4096=960 8192=1881.
_MI325X_SCHEDULE = (
    (8, 64, 8, 64, 2),  # <=8 tok: tiny, combine warp 2
    (64, 64, 8, 64, 4),  # <=64:   small block both
    (1024, 152, 8, 64, 4),  # <=1024: disp 0.5*CU, comb small-block/warp4 (latency)
    (4096, 228, 8, 152, 2),  # <=4096: comb 0.5*CU/warp2 (bandwidth)
    (None, 304, 8, 152, 2),  # >4096 (peak)
)
_MI325X_DEFAULT = dict(
    dispatch_block_num=304,
    combine_block_num=152,
    warp_num_per_block=8,
    combine_warp_num_per_block=2,
    schedule=_MI325X_SCHEDULE,
)
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI325X_TABLE = {
    (8, 7168, 8): {"fp8": _MI325X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI325X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI325X_SCHEDULE},
}

# ── MI300X (gfx942, 304 CU) — TODO: re-tune. Falls back to CU-scaled default. ──
_MI300X_DEFAULT = None  # None => derive from CU count (see _cu_scaled_default)
_MI300X_TABLE = {}

# ── MI355X (gfx950, wave64) — re-tuned 2026-07-13 for the vec4 combine gather,
# EP8, 8x gfx950 single-node xGMI, 2-pass block x warp sweep (tok 8..8192). vec4
# combine wants a small block (32-48) + warp up to 16; dispatch grows block 96->160.
# wave64 => warp <= 16 (1024-thread block). Geometry is topk-independent.
# bf16 dispatch + bf16 combine:
_MI355X_SCHED_BF16 = (
    (128, 128, 8, 32, 8),
    (256, 96, 8, 64, 8),
    (1024, 96, 8, 32, 16),
    (4096, 160, 8, 32, 16),
    (None, 128, 16, 48, 16),
)
# fp8 dispatch + bf16 combine (combine geometry shared with bf16):
_MI355X_SCHED_FP8 = (
    (128, 128, 8, 32, 8),
    (256, 160, 8, 64, 8),
    (1024, 128, 8, 32, 16),
    (4096, 160, 8, 32, 16),
    (None, 128, 16, 48, 16),
)
# fp4 dispatch + fp4 combine (0.5 B/elem):
_MI355X_SCHED_FP4 = (
    (256, 128, 4, 32, 8),
    (2048, 144, 4, 64, 16),
    (None, 128, 8, 64, 16),
)
_MI355X_DEFAULT = dict(
    dispatch_block_num=128,
    combine_block_num=48,
    warp_num_per_block=16,
    combine_warp_num_per_block=16,
    schedule=_MI355X_SCHED_BF16,
)
_MI355X_TABLE = {
    (8, 7168, 8): {
        "bf16": _MI355X_SCHED_BF16,
        "fp8": _MI355X_SCHED_FP8,
        "fp4": _MI355X_SCHED_FP4,
    },
    (8, 7168, 6): {
        "bf16": _MI355X_SCHED_BF16,
        "fp8": _MI355X_SCHED_FP8,
        "fp4": _MI355X_SCHED_FP4,
    },
}

# ── gfx1250 (256 CU, wave32) — RE-TUNED 2026-07-15 EP4, bf16, with the vec4 combine
# kernel + inner-unroll load-first scheduling, FINE 2-pass block x warp sweep
# (tok 16..16384; comb block 48..192, warp 2..16). Key lessons: (a) dispatch
# unchanged by vec4 — warp grows to 32, block 192; peaks ~287 GB/s @16384.
# (b) combine SHIFTED with vec4 AND is very warp-sensitive at small tok: warp 2 wins
# <=128 (64/2: 64tok 27 vs warp4 21 vs warp16 11 — over-parallelizing the gather+xdb
# barrier collapses it), warp 4 for mid, warp 8 for large; block grows 64->96->128.
# The old pre-vec4 schedule (comb block->192, warp 16) is stale: comb @8192 189->242
# (+28%), mid 512 55->102 (+84%), tiny 64 ~11->27 (2.4x). (c) GUARDRAIL: block_num
# < CU (256); 192 safe ceiling (Phase-2 grid barrier needs co-residence). Measured
# vec4 GB/s (disp/comb): 64=29/27 128=56/47 256=114/74 512=182/102 1024=214/132
# 2048=248/174 4096=264/207 8192=282/242 16384=287/273.
_GFX1250_SCHED_BF16 = (
    (128, 128, 8, 64, 2),  # <=128:  latency-bound; combine warp 2 (fewest warps win)
    (512, 192, 32, 64, 4),  # <=512:  disp peak; comb 64/4
    (1536, 192, 32, 96, 4),  # <=1536: comb block 96
    (4096, 192, 32, 128, 4),  # <=4096: comb block 128
    (None, 192, 32, 128, 8),  # >4096:  comb warp 8 (242/273 GB/s @8192/16384)
)
_GFX1250_DEFAULT = dict(
    dispatch_block_num=192,
    combine_block_num=192,
    warp_num_per_block=32,
    combine_warp_num_per_block=16,
    schedule=_GFX1250_SCHED_BF16,
)
# DeepSeek-V4-Pro shape (hidden 7168, topk 6, 384 experts). RE-VALIDATED 2026-07-15
# EP4 bf16 vec4 at the topk=8 schedule geometries: geometry is topk-independent
# (tracks token count, not topk) — per-size optimum matches topk=8, so reuse the
# same schedule. Measured vec4 GB/s (disp/comb): 16=6/4 256=95/67 512=164/93
# 1024=212/116 2048=252/164 4096=281/200 8192=293/238 16384=300/272.
_GFX1250_SCHED_BF16_T6 = _GFX1250_SCHED_BF16
# EP8 (world_size=8) RE-TUNED 2026-07-13 on gfx1250 CROSS-NODE (2 nodes x 4 GPUs
# over the UALink fabric), bf16, with the vec4 combine-gather kernel, full 2-pass
# block x warp sweep (tok 8..8192). dispatch unchanged by vec4: block 128, warp
# ramps 16->32. combine SHIFTED: block 64 is now uniformly best (128 loses even at
# 8192 with vec4's wider loads), warp ramps 4->8->16. Cross-node fabric caps disp
# ~200 GB/s; geometry is world_size-independent so this also serves single-node EP8.
# Measured vec4 GB/s at scheduled geom (disp/comb): 8=5/3 64=17/23 128=71/47
# 256=127/76 512=162/105 1024=187/135 2048=198/152 4096=200/176 8192=200/200
# (<=64 disp is launch-latency noise). vs pre-vec4 scalar tuned comb 8192=173
# 512=63 256=42 -> +15% / +67% / +80%. topk6 (384 exp) reuses this schedule,
# comb 8192=198 512=99 256=68 (geometry is topk-independent).
_GFX1250_SCHED_BF16_EP8 = (
    (256, 128, 16, 64, 4),  # <=256:  disp warp 16, comb 64/4 (latency-bound)
    (1024, 128, 32, 64, 8),  # <=1024: disp warp 32, comb 64/8
    (None, 128, 32, 64, 16),  # >1024 (peak): disp ~200 / comb ~200 GB/s
)
# bf16-tuned (EP4 + EP8). fp8/fp4 fall back to the bf16 schedule until separately tuned.
_GFX1250_TABLE = {
    (4, 7168, 8): {"bf16": _GFX1250_SCHED_BF16},
    (4, 7168, 6): {"bf16": _GFX1250_SCHED_BF16_T6},  # DeepSeek-V4-Pro
    (8, 7168, 8): {"bf16": _GFX1250_SCHED_BF16_EP8},  # cross-node / single-node EP8
    # V4-Pro topk=6 EP8 cross-node (measured 2026-07-12, full tok 8..8192 sweep):
    # geometry is topk-independent (tracks token count) — per-size optimum matches
    # topk=8, so reuse the same schedule. Measured GB/s (disp/comb): 8=4/3 64=32/16
    # 128=65/25 256=119/39 512=163/56 1024=178/81 2048=199/113 4096=200/145 8192=206/171.
    (8, 7168, 6): {"bf16": _GFX1250_SCHED_BF16_EP8},
}

_DEVICES = {
    "mi308x": (_MI308X_DEFAULT, _MI308X_TABLE),
    "mi325x": (_MI325X_DEFAULT, _MI325X_TABLE),
    "mi300x": (_MI300X_DEFAULT, _MI300X_TABLE),
    "mi355x": (_MI355X_DEFAULT, _MI355X_TABLE),
    "gfx1250": (_GFX1250_DEFAULT, _GFX1250_TABLE),
}


# PCI device IDs (KFD `device_id`) → device table key. gfx942 family only; the
# gfx950 parts (MI350/MI355X) are matched by arch below since their DIDs vary.
_DID_TO_KEY = {
    0x74A1: "mi300x",  # MI300X
    0x74A5: "mi325x",  # MI325X (304-CU gfx942; tuned 2026-07-08)
    0x74A2: "mi308x",  # MI308X (80 CU)
}


@functools.lru_cache(maxsize=1)
def _topology():
    """(cu_count, gfx_target_version, device_id) of the first GPU node from KFD
    sysfs. torch/HIP-free. gfx_target_version is e.g. 90402 (gfx942), 90500
    (gfx950); device_id is the PCI DID (e.g. 0x74a2 = MI308X). Homogeneous host
    assumed. Returns (0, 0, 0) if sysfs is unavailable (no KFD mounted)."""
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            vals = {}
            with open(props) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        vals[parts[0]] = int(parts[1])
            simd = vals.get("simd_count", 0)
            if simd <= 0:  # CPU / non-GPU node
                continue
            spc = vals.get("simd_per_cu", 0) or 1
            return (
                simd // spc,
                vals.get("gfx_target_version", 0),
                vals.get("device_id", 0),
            )
        except Exception:
            continue
    return 0, 0, 0


def _cu_count():
    return _topology()[0]


def _device_key():
    """Map the current GPU to a device table key: exact PCI DID first, then arch
    for gfx950. Returns None if unknown (caller uses a CU-scaled default)."""
    _, gfx, did = _topology()
    key = _DID_TO_KEY.get(did)
    if key is not None:
        return key
    if gfx == 90500:  # gfx950 (MI350 / MI355X), DID varies
        return "mi355x"
    if gfx == 120500:  # gfx1250 (256 CU, wave32), DID varies
        return "gfx1250"
    return None


def _cu_scaled_default():
    """Untuned fallback: ~1 block/CU for combine (<= CU count), many warps for
    dispatch, few warps for combine, no per-token schedule (single-shot). Used
    for devices without a measured table."""
    cu = _cu_count() or 80
    return dict(
        dispatch_block_num=cu,
        combine_block_num=cu,
        warp_num_per_block=16,
        combine_warp_num_per_block=4,
        schedule=None,
    )


def lookup(world_size, hidden_dim, topk, dtype="fp8"):
    """Return {dispatch_block_num, combine_block_num, warp_num_per_block,
    combine_warp_num_per_block, schedule} for the current GPU, shape, and dtype.

    `dtype` (the token / dispatch dtype: "bf16" | "fp8" | "fp4") selects the
    per-dtype schedule, because dtype sets the communication volume (fp4 = 0.5 B,
    fp8 = 1 B, bf16 = 2 B per element) and thus the best block/warp. It falls back
    to the "fp8" schedule, then to the device default, when a dtype isn't tuned.

    `schedule` (or None) is a per-token-count launch plan: a tuple of
    (max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp)
    buckets, ascending; the op precompiles the distinct (block, warp) variants
    and picks a bucket at runtime from cur_rank_num_token. dispatch_block_num /
    warp_num_per_block / combine_block_num / combine_warp_num_per_block are the
    single-shot fallback used when schedule is None."""
    key = _device_key()
    if key is None or key not in _DEVICES:
        return _cu_scaled_default()
    dev_default, dev_table = _DEVICES[key]
    base = dict(dev_default) if dev_default is not None else _cu_scaled_default()
    base.setdefault("schedule", None)
    entry = dev_table.get((world_size, hidden_dim, topk))
    if entry:
        base["schedule"] = entry.get(dtype) or entry.get("fp8") or base["schedule"]
    return base
