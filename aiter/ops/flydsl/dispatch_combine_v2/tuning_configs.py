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
"""Per-device, per-shape block/warp tuning for the cco-LSA dispatch/combine kernels.

Geometry is picked per device, then by (world_size, hidden_dim, topk). Devices are
told apart by PCI device id (gfx942 family) or arch (gfx950/gfx1250) from KFD sysfs
(no torch/HIP dep). block_num must stay <= CU count; re-tune per GPU. A SCHEDULE is a
tuple of (max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp)
buckets; the op precompiles the distinct (block, warp) variants and picks by token count.
"""

import functools
import glob


# ── MI308X (gfx942, 80 CU) — measured 2026-07-08, EP8, block x warp sweep.
_MI308X_SCHEDULE = (
    (256, 64, 8, 64, 4),
    (2048, 64, 16, 64, 4),
    (None, 64, 16, 80, 4),
)
# Single-shot fallback (schedule ignored) = peak-optimal.
_MI308X_DEFAULT = dict(
    dispatch_block_num=64,
    combine_block_num=80,
    dispatch_warp_num_per_block=16,
    combine_warp_num_per_block=4,
    schedule=_MI308X_SCHEDULE,
)
# Per-shape -> per-dtype schedules (tuned fp8-dispatch + bf16-combine, under "fp8").
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI308X_TABLE = {
    (8, 7168, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI308X_SCHEDULE},
}

# ── MI325X (gfx942, 304 CU, DID 0x74a5) — measured 2026-07-08, EP8, 2D block x warp
# sweep by min latency. Dispatch wants warp 8 (block grows with tok); combine is
# latency-bound at small/mid tok (small block 64 + warp 4 wins), bandwidth-bound only
# at large tok (>1024: block ~0.5*CU + warp 2).
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
    dispatch_warp_num_per_block=8,
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

# ── MI355X (gfx950, wave64) — re-tuned 2026-07-13 (vec4 combine gather), EP8 single-node
# xGMI. vec4 combine wants a small block (32-48) + warp up to 16, dispatch block 96->160
# (wave64 => warp <= 16). Geometry is topk-independent.
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
    dispatch_warp_num_per_block=16,
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

# ── gfx1250 (256 CU, wave32) — RE-TUNED 2026-07-15 EP4, bf16, vec4 combine + inner-unroll
# scheduling, 2-pass block x warp sweep (tok 16..16384). Dispatch warp 32 / block 192;
# combine is warp-sensitive at small tok (warp 2 <=128, warp 4 mid, warp 8 large; block
# 64->96->128). GUARDRAIL: block_num < CU (256); 192 ceiling (Phase-2 grid barrier).
_GFX1250_SCHED_BF16 = (
    (128, 128, 8, 64, 2),  # <=128:  latency-bound; combine warp 2 (fewest warps win)
    (512, 192, 32, 64, 4),  # <=512:  disp peak; comb 64/4
    (1536, 192, 32, 96, 4),  # <=1536: comb block 96
    (4096, 192, 32, 128, 4),  # <=4096: comb block 128
    (None, 192, 32, 128, 8),  # >4096:  comb warp 8
)
_GFX1250_DEFAULT = dict(
    dispatch_block_num=192,
    combine_block_num=192,
    dispatch_warp_num_per_block=32,
    combine_warp_num_per_block=16,
    schedule=_GFX1250_SCHED_BF16,
)
# DeepSeek-V4-Pro (hidden 7168, topk 6): geometry is topk-independent, reuse topk=8.
_GFX1250_SCHED_BF16_T6 = _GFX1250_SCHED_BF16
# EP8 RE-TUNED 2026-07-13 cross-node (UALink fabric). Dispatch block 128 (warp 16->32);
# combine block 64 uniformly best (warp 4->8->16). Fabric caps disp ~200 GB/s;
# world_size-independent so this also serves single-node EP8.
_GFX1250_SCHED_BF16_EP8 = (
    (256, 128, 16, 64, 4),  # <=256:  disp warp 16, comb 64/4 (latency-bound)
    (1024, 128, 32, 64, 8),  # <=1024: disp warp 32, comb 64/8
    (None, 128, 32, 64, 16),  # >1024 (peak)
)
# bf16-tuned (EP4 + EP8). fp8/fp4 fall back to the bf16 schedule until separately tuned.
_GFX1250_TABLE = {
    (4, 7168, 8): {"bf16": _GFX1250_SCHED_BF16},
    (4, 7168, 6): {"bf16": _GFX1250_SCHED_BF16_T6},  # DeepSeek-V4-Pro
    (8, 7168, 8): {"bf16": _GFX1250_SCHED_BF16_EP8},  # cross-node / single-node EP8
    (8, 7168, 6): {"bf16": _GFX1250_SCHED_BF16_EP8},  # topk-independent, reuse topk=8
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
    """(cu_count, gfx_target_version, device_id) of the first GPU node from KFD sysfs
    (torch/HIP-free, homogeneous host). Returns (0, 0, 0) if sysfs is unavailable."""
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
    """Untuned fallback (~1 block/CU, single-shot) for devices without a measured table."""
    cu = _cu_count() or 80
    return dict(
        dispatch_block_num=cu,
        combine_block_num=cu,
        dispatch_warp_num_per_block=16,
        combine_warp_num_per_block=4,
        schedule=None,
    )


def lookup(world_size, hidden_dim, topk, dtype="fp8"):
    """Return {dispatch_block_num, combine_block_num, dispatch_warp_num_per_block,
    combine_warp_num_per_block, schedule} for the current GPU, shape, and dtype.

    `dtype` ("bf16" | "fp8" | "fp4") selects the per-dtype schedule (dtype sets the
    communication volume and thus the best geometry); falls back to "fp8" then the
    device default. `schedule` (or None) is the per-token launch plan; the block/warp
    fields are the single-shot fallback used when schedule is None."""
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
