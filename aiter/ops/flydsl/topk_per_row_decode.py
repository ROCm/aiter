# SPDX-License-Identifier: MIT

"""High-level FlyDSL decode TopK-per-row API."""

from __future__ import annotations
import functools
import math
import os
import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode_tiered import (
    BLOCK_THREADS as _TIERED_BLOCK_THREADS,
    LOAD_VEC as _TIERED_LOAD_VEC,
    SCAN_STAGES as _TIERED_SCAN_STAGES,
    create_topk_per_row_decode_tiered_kernel,
    needs_workspace_zero,
    topk_workspace_slots,
)

from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP


##################################

# Independent of K: K only affects the final O(K) index scatter, negligible vs
# the O(L) scan.
_TIERED_MID_MAX = 65536

# Co-resident occupancy of the 1024-thread tiered block on CDNA (waves/CU limited;
# VGPR/LDS have headroom). Same value the deadlock guard assumes. Envelope = CU * this.
_CDNA_OCCUPANCY = 2

# Batch-adaptive occupancy for the grid-width cap (Step 4 Stage 14). The 1024-thread
# block reaches the full occ=2 (512-wg envelope) only for modest grids; beyond ~CU/8
# rows the 512-wg envelope spills into a 2nd wave and occ=1 (256-wg, one true wave)
# is ~1.6x faster at rows>=40 (rocprofv3). occ=2 stays best up to num_rows=32.
_COCAP_OCC2_MAX_ROWS = 32

# Per-arch tunable config for the tiered decode Top-K kernel. All values are chosen
# on the HOST at config time and baked into the JIT kernel as compile-time constants,
# so per-arch branching adds ZERO runtime / in-kernel overhead.
#
#   gfx942 : Robin's validated MI300X tuning — FROZEN (do not change).
#   gfx950 : SILOTIGER-699 tuning target. Starts as a clone of the validated gfx942
#            config — Step 4 Stage 2/3 on real gfx950 showed the gfx942 short_max
#            (base 16384) already beats the previous gfx950 guess (24576/1792/57344),
#            so gfx942 values are the correct, validated starting point; we tune from here.
#   other  : falls back to the gfx942 entry.
#
# Fields:
#   short_max   : (base, slope, cap)  -> short_max = min(cap, base + num_rows*slope)
#   mid_cap     : (rows<=1, rows>1)
#   long_cap    : (rows<=1, rows<8, rows>=8)
#   mid_max     : mid<->long tier boundary
#   scan_stages : load-unroll / ILP (1/2/4/8)
#   blocks_cap  : max workgroups per row (grid-width cap; BLOCK_THREADS=1024 -> 32)
#   bpp         : bits_per_pass override (10/11); None = auto (CU-based)
#   row_proportional_parts : cap participating workgroups by the row's actual coverage
#                            (ceil(row_len / (LOAD_VEC*BLOCK_THREADS))) instead of always
#                            using the full padded-width grid. Cuts cross-CU barrier/merge
#                            latency for short rows. gfx942 frozen (False).
#   batch_coresident_cap  : cap the launch grid WIDTH (blocks_per_row) by the per-row
#                            co-resident budget (CU*occ // num_rows) so the whole batch
#                            fits in one co-resident wave. Without it the fixed 32-wide
#                            grid makes blocks_per_row*num_rows exceed CU*occ once
#                            num_rows>~16, forcing the persistent barrier to serialize
#                            rows into waves (latency ~ number of waves). gfx942 frozen.
_ARCH_CONFIG: dict[str, dict] = {
    "gfx942": {
        "short_max": (16384, 1536, 40960),
        "mid_cap": (32, 8),
        "long_cap": (32, 16, 8),
        "mid_max": _TIERED_MID_MAX,
        "scan_stages": _TIERED_SCAN_STAGES,
        "blocks_cap": 32,
        "bpp": None,
        "row_proportional_parts": False,
        "batch_coresident_cap": False,
        "early_stop": False,
    },
}
# gfx950 tuning starts as a validated clone of gfx942 (Step 4 Stage 2/3). Tune this
# entry only; gfx942 stays frozen and unknown archs fall back to gfx942.
_ARCH_CONFIG["gfx950"] = dict(_ARCH_CONFIG["gfx942"])
# Step 4 Stage 3: the short<->mid crossover knee measured on real gfx950 is ~18432
# (vs gfx942's 16384 and Robin's original gfx950 guess of 24576). Bumping the short_max
# base to the knee keeps rows=1 on the faster short tier ~2k elements longer, recovering
# ~2–2.5µs at L≈20k–26k without regressing elsewhere. Slope/cap stay at the gfx942 values.
_ARCH_CONFIG["gfx950"]["short_max"] = (18432, 1536, 40960)
# Step 4 Stage 6: on real gfx950 the mid/long floor is cross-CU sync latency. The grid is
# sized for the padded buffer width (→32 wgs) even for short rows, so a 32k row spins up 32
# cooperating workgroups when ~8 suffice, paying needless barrier/merge cost. Sizing the
# participating workgroups to the row's coverage recovers ~1–2µs across L≈24k–65k (rows=1),
# bringing FlyDSL to ~parity with HIP-mb at L≤~32k. gfx942 stays frozen (False).
_ARCH_CONFIG["gfx950"]["row_proportional_parts"] = True
# Step 4 Stage 11: the launch grid is (blocks_per_row, num_rows). blocks_per_row is sized
# for the padded buffer width (→32 on gfx950), so the launched workgroup count 32*num_rows
# exceeds the co-resident envelope (CU*occ = 256*2 = 512) once num_rows>16. Because the row
# barrier spins over a non-cooperative launch, the excess rows serialize into sequential
# waves and latency blows up ~linearly with batch (rows=32→2×, 64→4×, 128→8×). Capping the
# grid width to the per-row budget (envelope//num_rows) keeps the whole batch in one wave;
# when the budget collapses the deadlock guard folds rows onto the barrier-free short tier.
# gfx942 stays frozen (False).
_ARCH_CONFIG["gfx950"]["batch_coresident_cap"] = True
# Step 4 Stage 15: FlyDSL ran all 3 radix passes while HIP mb early-stops after 2 (its
# 3rd "pass" is write-only). When the boundary bucket after the 2nd pass is taken whole
# (remaining_len == remaining_k, the common case for high-entropy logits) the final
# cross-CU histogram pass is unnecessary: write those elements directly. Saves one full
# barrier+merge round -> ~20% at rows=1 (now faster than HIP mb) and ~13-16% across batch.
# Data-dependent: when a boundary tie prevents the clean cut it falls back to the full
# last pass (no regression). gfx942 stays frozen (False). Env FLYDSL_TOPK_TIERED_EARLY_STOP.
_ARCH_CONFIG["gfx950"]["early_stop"] = True
# Step 4 Stage 17: the launch grid width (blocks_per_row) is sized for the padded buffer
# (→ up to 32), but the real workers per row is active_parts = min(blocks_per_row, tier_cap,
# row_cover). When blocks_per_row exceeds the longest row's tier cap (e.g. long_cap=8 at
# rows>=8), the extra (blocks_per_row - cap) workgroups per row ALWAYS return immediately —
# "dead" blocks that still consume co-resident scheduler slots. At rows=16-32, L>=120k this
# is up to half the grid (e.g. rows=32 L=256k: grid 16 wide, only 8 active -> 256 dead wgs).
# Trimming blocks_per_row down to the max active_parts removes those dead blocks WITHOUT
# changing any row's active_parts (min() is unaffected), so results/parallelism are identical
# and only wasted scheduling is cut (~8% at the worst dead-heavy cells). This generalises the
# budget<2 grid=1 fold to every batch. gfx942 stays frozen (absent -> False).
_ARCH_CONFIG["gfx950"]["dead_block_trim"] = True
# Step 4 Stage 18: mid-batch coordination cap. In the occ=2 co-resident band the grid
# budget (envelope//num_rows, up to ~21-32) far exceeds the cooperation sweet spot once
# the batch alone already fills the device (num_rows>16). Measured G-sweeps (rows 20-48,
# gfx950) show the optimum collapses well below the budget: fewer cooperating blocks/row
# halve the cross-CU barrier/histogram-merge cost while the batch still saturates the CUs.
# The sweet spot is length-dependent (a 2D rows x L surface), so the cap is a small step
# function on L, keyed to the actual measured optimum, applied only for num_rows>16 in the
# multi-block regime (tiered_short_max < L):
#   * mid-length tier   (short_max < L <= 131072): cap 4  -- opt is ~4 at L 49k/65k/120k;
#     recovers ~4-9us (10-23%) at rows 20/24/32/48.
#   * very-long tier     (L > 131072):             cap 8  -- opt rises to ~6-8 at L=256k;
#     recovers ~2-6us at rows 20/24/32 (rows>=48 already sit below 8 via the budget).
# The 131072 breakpoint is what threads the needle: coverage//4 cannot separate L=120k
# (wants 4) from L=256k (wants ~6-8) since both saturate the 32-wide coverage cap, but the
# absolute-L step can. Only ever REDUCES blocks_per_row (a min), so active_parts, results
# and parallelism stay valid. Deliberately excludes num_rows<=16 and the decode target
# (rows 1-8), which still want the full cooperative width (e.g. rows=16/L=256k wants 8;
# capping to 4 would regress it ~10us).
#
# Rules are (rows_min_exclusive, L_max_inclusive|None, cap), scanned in order; the first
# rule whose num_rows>rows_min AND (L_max is None or L<=L_max) wins. Measured optima:
#   * short-mid (short_max<L<=65536), rows>=64:  cap 1 (single-workgroup / one-block-per-row).
#     Once the batch alone fills the device (rows*1 ~ CU) AND the per-row scan is short
#     enough, cooperation is pure barrier overhead -> collapse to one block/row (HIP-ob
#     shape). Measured L=65536: rows=128 59.6->46.2us (-22%), rows=64 -1.8us; L=49152
#     rows>=48 also win. Gated to L<=65536: at L>=120k a lone workgroup's linear scan
#     explodes (rows=48/L=120k G1=73us vs 46us cooperative), so those stay multi-block.
#   * mid-length (short_max<L<=131072), rows>16:            cap 4
#   * very-long (L>131072), rows in (16,20]:                cap 8  (rows~20 still wants 8)
#   * very-long (L>131072), rows>20:                        cap 6  (opt tightens to ~6;
#     rows=32/L=256k measured 54-56us@G6 vs 58us@G8, rows=24 neutral, no regression)
# gfx942 stays frozen (absent -> disabled). Env FLYDSL_TOPK_TIERED_MIDBATCH_CAP overrides
# the cap value for all matched rules (0 disables).
_ARCH_CONFIG["gfx950"]["midbatch_coord_cap"] = (
    (63, 65536, 1),
    (16, 131072, 4),
    (20, None, 6),
    (16, None, 8),
)


def _arch_config(arch: str) -> dict:
    return _ARCH_CONFIG.get(arch, _ARCH_CONFIG["gfx942"])


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


@functools.cache
def _multi_processor_count(arch: str | None = None) -> int:
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(props.multi_processor_count)
    except Exception:
        fallback_min_cu = {"gfx942": 228, "gfx950": 256}
        return fallback_min_cu.get(arch or get_rocm_arch(), 64)


@functools.cache
def _environ_kernel_config() -> dict:
    cfg = dict(
        scan_stages=_env_int("FLYDSL_TOPK_SCAN_STAGES"),
        # A/B override for the launch grid width (blocks_per_row). 1 -> grid=(1, num_rows),
        # i.e. every row single-workgroup short tier (matches HIP one-block launch shape).
        blocks_per_row=_env_int("FLYDSL_TOPK_TIERED_G"),
        tiered_short_max=_env_int("FLYDSL_TOPK_TIERED_SHORT_MAX"),
        tiered_mid_cap=_env_int("FLYDSL_TOPK_TIERED_MID_CAP"),
        tiered_mid_max=_env_int("FLYDSL_TOPK_TIERED_MID_MAX"),
        tiered_long_cap=_env_int("FLYDSL_TOPK_TIERED_LONG_CAP"),
        bits_per_pass=_env_int("FLYDSL_TOPK_TIERED_BPP"),
        # 0/1 override for the non-finite mask (default on); set 0 to disable.
        mask_non_finite=_env_int("FLYDSL_TOPK_TIERED_MASK_NONFINITE"),
        # 0/1 override for row-proportional active_parts (default per-arch); A/B knob.
        row_proportional_parts=_env_int("FLYDSL_TOPK_TIERED_RPP"),
        # 0/1 override for HIP-mb-style last-pass early-stop (default per-arch); A/B knob.
        early_stop=_env_int("FLYDSL_TOPK_TIERED_EARLY_STOP"),
    )
    return {k: v for k, v in cfg.items() if v is not None}


def _default_kernel_config(
    num_rows: int,
    max_model_len: int,
) -> dict:
    # FLYDSL_TOPK_ARCH forces a specific arch's tuning table on the current HW
    # (A/B knob only; unset -> real device arch, so production is unaffected). Used
    # to measure the gfx942 (untuned) config against the gfx950 config on one box.
    arch = os.environ.get("FLYDSL_TOPK_ARCH") or get_rocm_arch()
    ac = _arch_config(arch)

    # Grid width per row: enough workgroups to cover the row at LOAD_VEC elements
    # per thread, clamped to [2, blocks_cap] (BLOCK_THREADS is fixed at 1024, so the
    # mid/long tiers can use at most 32 workgroups).
    items_per_block = _TIERED_LOAD_VEC * _TIERED_BLOCK_THREADS
    blocks_per_row = min(ac["blocks_cap"], max(2, math.ceil(max_model_len / items_per_block)))

    # bits_per_pass: arch override if set, else 11 (2048-bin LDS histogram) whenever the
    # arch can afford it (the short tier requires 11); gfx942/gfx950 both qualify.
    # Computed before the batch cap because the grid=1 large-batch fold below requires
    # the single-workgroup short tier (bpp==11).
    cu_count = _multi_processor_count(arch)
    bits_per_pass = ac["bpp"]
    if bits_per_pass is None:
        bits_per_pass = (
            11 if cu_count >= 128 or SMEM_CAPACITY_MAP.get(arch, 0) >= 128 * 1024 else 10
        )

    # Max cooperating workgroups per row for the mid/long tiers (real wg count is
    # min(blocks_per_row, cap)). Scales down with batch size: a single long row wants
    # the full wg set, while multi-row batches already fill the device so fewer
    # workgroups/row cut barrier and histogram-merge cost. Computed before the batch
    # cap because the dead-block trim below clamps blocks_per_row to these caps.
    mid_cap_le1, mid_cap_gt1 = ac["mid_cap"]
    tiered_mid_cap_default = mid_cap_le1 if num_rows <= 1 else mid_cap_gt1

    long_le1, long_lt8, long_ge8 = ac["long_cap"]
    if num_rows <= 1:
        tiered_long_cap_default = long_le1
    elif num_rows < 8:
        tiered_long_cap_default = long_lt8
    else:  # num_rows >= 8
        tiered_long_cap_default = long_ge8

    # Batch-aware short vs multi-block crossover (arch-specific base/slope/cap). The
    # multi-block barrier floor grows under CU contention as more rows launch, while
    # the single-workgroup path is flat in batch.
    base, slope, cap = ac["short_max"]
    tiered_short_max = min(cap, base + num_rows * slope)
    tiered_mid_max = ac["mid_max"]

    # Dead-block trim (Step 4 Stage 17, gfx950 only): the grid only needs to be as wide
    # as the max active_parts the LONGEST row will use. Since active_parts is always
    # min(blocks_per_row, tier_cap, row_cover), clamping blocks_per_row to the tier cap
    # of the longest reachable tier removes purely-dead (immediate-return) blocks without
    # changing any row's active_parts. A batch whose longest row is in the long tier may
    # still hold shorter rows in the mid tier, so use max(mid_cap, long_cap) there.
    #
    # Gate: only trim when the FULL padded grid (blocks_cap*num_rows) already fits one
    # co-resident wave (CU*occ). Then the dead blocks fill no otherwise-idle slot and are
    # pure waste -> trimming is a clean win (rows<=16 on gfx950: 8-12% at L>=120k, 0 regress
    # incl. 3 win-flips). Once the full grid spills past one wave (rows>=32), the extra
    # resident blocks instead help hide memory/barrier latency, so trimming below the
    # batch-cap width regresses at mid L (measured rows=32 L=65k/120k: +7-9%); leave those
    # to the batch co-resident cap below. Env FLYDSL_TOPK_TIERED_TRIM (0/1) overrides.
    trim_on = _env_int("FLYDSL_TOPK_TIERED_TRIM")
    if trim_on is None:
        trim_on = 1 if ac.get("dead_block_trim") else 0
    if trim_on:
        if max_model_len <= tiered_short_max:
            # All rows are short-tier (active_parts=1): every block but one is dead, and
            # the barrier-free single-wg tier has no cooperative scan/barrier for the
            # extras to hide latency for -> always collapse to grid=1 (HIP one-block shape),
            # regardless of the wave gate below.
            blocks_per_row = 1
        elif num_rows * ac["blocks_cap"] <= cu_count * _CDNA_OCCUPANCY:
            # Mid/long: only trim when the FULL padded grid (blocks_cap*num_rows) already
            # fits one co-resident wave (CU*occ). Then the dead blocks fill no idle slot
            # and are pure waste -> clean win (rows<=16: 8-12% at L>=120k, 0 regress incl.
            # 3 win-flips). Once the full grid spills past one wave (rows>=32), the extra
            # resident blocks help hide memory/barrier latency, so trimming below the
            # batch-cap width regresses at mid L (rows=32 L=65k/120k: +7-9%) -> leave those
            # to the batch co-resident cap below.
            if max_model_len <= tiered_mid_max:
                max_active_parts = tiered_mid_cap_default
            else:
                max_active_parts = max(tiered_mid_cap_default, tiered_long_cap_default)
            blocks_per_row = max(1, min(blocks_per_row, max_active_parts))

    # Batch co-resident grid-width cap (Step 4 Stage 11): keep blocks_per_row*num_rows within
    # one co-resident wave (envelope = CU*occ) so the persistent barrier does not serialize
    # rows. Env FLYDSL_TOPK_TIERED_BATCH_CAP (0/1) overrides the arch default for A/B.
    force_single_wg = False
    batch_cap_on = _env_int("FLYDSL_TOPK_TIERED_BATCH_CAP")
    if batch_cap_on is None:
        batch_cap_on = 1 if ac.get("batch_coresident_cap") else 0
    if batch_cap_on and num_rows > 1:
        # Batch-adaptive occupancy: occ=2 co-resides for modest grids, but beyond
        # _COCAP_OCC2_MAX_ROWS the 512-wg envelope spills into a 2nd wave, so occ=1
        # (one true wave) is ~1.6x faster (Stage 14). Env FLYDSL_TOPK_TIERED_OCC forces.
        occ = _env_int("FLYDSL_TOPK_TIERED_OCC")
        if not occ:
            occ = _CDNA_OCCUPANCY if num_rows <= _COCAP_OCC2_MAX_ROWS else 1
        envelope = cu_count * occ
        budget = envelope // num_rows
        if budget >= 2:
            blocks_per_row = min(blocks_per_row, budget)
        elif bits_per_pass == 11:
            # budget<2 (Step 4 Stage 16 / Phase 0): even a width-2 cooperative grid
            # can't fit the batch in one co-resident wave. Leaving the padded-width
            # grid launches blocks_per_row*num_rows workgroups, of which
            # (blocks_per_row-1)*num_rows are dead (immediate-return) yet still occupy
            # co-resident slots, forcing the real workers to serialize into many waves
            # -> catastrophic latency collapse (measured 4-25x vs HIP at num_rows>=192).
            # Collapse the launch to grid=(1, num_rows): every row runs the barrier-free
            # single-workgroup short tier, matching HIP's one-block launch shape (no dead
            # blocks, ~1 wave). Strictly better here than the old deadlock-guard fold,
            # which folded short_max but left the wide grid intact (still collapsed).
            blocks_per_row = 1
            force_single_wg = True

    if force_single_wg:
        # grid=(1, num_rows): active_parts is 1 for every row regardless of length, so
        # make the all-short intent explicit. This keeps needs_workspace_zero() False
        # (no multi-block tier runs) and lets the deadlock guard early-out cleanly.
        tiered_short_max = max(tiered_short_max, max_model_len)
        tiered_mid_max = max(tiered_mid_max, tiered_short_max)

    # Mid-batch coordination cap (Step 4 Stage 18, gfx950 only): in the mid-length
    # multi-block tier (tiered_short_max < L <= L_max) a wide batch (num_rows > min_rows)
    # already saturates the CUs, so the co-resident budget over-provisions blocks_per_row;
    # the measured cooperation optimum there is ~cap. Only reduces blocks_per_row (a min),
    # so active_parts, results and parallelism stay valid. Skipped when force_single_wg
    # (short tier) since tiered_short_max was raised to >= max_model_len above. Env
    # FLYDSL_TOPK_TIERED_MIDBATCH_CAP overrides the cap value (0 disables).
    midbatch_cap_cfg = ac.get("midbatch_coord_cap")
    if midbatch_cap_cfg is not None and tiered_short_max < max_model_len:
        mb_cap = None
        for mb_min_rows, mb_L_max, cap in midbatch_cap_cfg:
            if num_rows > mb_min_rows and (mb_L_max is None or max_model_len <= mb_L_max):
                mb_cap = cap
                break
        mb_env = _env_int("FLYDSL_TOPK_TIERED_MIDBATCH_CAP")
        if mb_env is not None and mb_cap is not None:
            mb_cap = mb_env
        if mb_cap:
            blocks_per_row = max(1, min(blocks_per_row, mb_cap))

    return dict(
        blocks_per_row=blocks_per_row,
        bits_per_pass=bits_per_pass,
        tiered=True,
        scan_stages=ac["scan_stages"],
        tiered_short_max=tiered_short_max,
        tiered_mid_cap=tiered_mid_cap_default,
        tiered_mid_max=tiered_mid_max,
        tiered_long_cap=tiered_long_cap_default,
        mask_non_finite=True,
        row_proportional_parts=ac["row_proportional_parts"],
        early_stop=ac.get("early_stop", False),
    )


def _kernel_config(num_rows: int, max_model_len: int) -> dict:
    default_config = _default_kernel_config(num_rows, max_model_len)
    environ_config = _environ_kernel_config()

    kernel_config = {
        **default_config,
        **environ_config,
    }

    bits_per_pass = kernel_config["bits_per_pass"]
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")

    kernel_config = _apply_deadlock_guard(kernel_config, num_rows, max_model_len)
    return kernel_config


def _apply_deadlock_guard(
    kernel_config: dict,
    num_rows: int,
    max_model_len: int,
) -> dict:
    """Clamp the tiered config so the mid/long-tier inter-workgroup barrier cannot
    deadlock. The possibility of deadlock requires both a wide batch (num_rows > ~80-90)
    and long rows (L > ~16-40K). Note: The HIP kernel also redirects to a
    single-workgroup tier for large batch-sizes, but doesn't explicitly call out the
    deadlock risk (see the should_use_mulblocks function).

    The tiered kernels spin on a barrier over a non-cooperative launch, which gives
    no guarantee that a row's participating workgroups are all resident at the same
    time. A workgroup spinning at the barrier holds its slot until every participant
    of its row arrives. A deadlock can potentially happen once the barrier-blocked
    workgroups exceeds the resident capacity. In this case, we cap the active
    workgroups or force the barrier-free short tier (single workgroup/row).

    Example for MI300X:
    The deadlock guard is active around num_rows >= 87 with max_model_len > 32768
    CUs * Occupancy = 304 * 2 = 608
    8 cooperating workgroups, 87*7 >= 608
    """
    if num_rows <= 0:
        return kernel_config

    short_max = kernel_config["tiered_short_max"]
    if max_model_len <= short_max:
        return kernel_config  # all rows short-tier -> barrier-free

    # Worst-case workgroups any single row can put on the barrier, given the tier
    # its length can reach (mid vs long) and the grid width.
    mid_cap = kernel_config["tiered_mid_cap"]
    long_cap = kernel_config["tiered_long_cap"]
    blocks_per_row = kernel_config["blocks_per_row"]
    if max_model_len <= kernel_config["tiered_mid_max"]:
        max_active_workgroups_per_row = min(blocks_per_row, mid_cap)
    else:
        max_active_workgroups_per_row = min(blocks_per_row, max(mid_cap, long_cap))

    is_single_workgroup = max_active_workgroups_per_row <= 1
    if is_single_workgroup:
        return kernel_config  # single-workgroup tier -> barrier-free

    # Co-resident envelope N = num_CU x occupancy. Occupancy is 2 on all CDNA:
    # the 1024-thread block is wave-limited (32 waves/CU / 16), with VGPR/LDS
    # headroom (measured gfx942: VGPR=40, LDS=8.7KB). Re-check if scan_stages or
    # the histogram grows enough to push VGPR>64 / LDS>32KB (would drop occ to 1).
    max_coresident_workgroups = _multi_processor_count() * 2
    is_deadlock_free = (
        num_rows * (max_active_workgroups_per_row - 1) < max_coresident_workgroups
    )
    if is_deadlock_free:
        return kernel_config

    # Largest cap A satisfying num_rows * (A - 1) < N.
    max_safe_active_workgroups = (max_coresident_workgroups - 1) // num_rows + 1
    if max_safe_active_workgroups >= 2:
        kernel_config["tiered_mid_cap"] = min(mid_cap, max_safe_active_workgroups)
        kernel_config["tiered_long_cap"] = min(long_cap, max_safe_active_workgroups)
    else:
        kernel_config["tiered_short_max"] = max_model_len

        kernel_config["tiered_mid_max"] = max(
            kernel_config["tiered_mid_max"], max_model_len
        )
    return kernel_config


def flydsl_top_k_per_row_decode_workspace_size(
    num_rows: int,
    max_model_len: int,
) -> int:
    """
    Number of int32 elements the decode TopK workspace needs for this shape.
    max_model_len = int(logits.shape[1])
    """
    if num_rows <= 0:
        return 0

    kernel_config = _kernel_config(num_rows, max_model_len)
    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    return workspace_slots


@functools.lru_cache(maxsize=16384)
def _compile_launcher(
    top_k: int,
    num_rows: int,
    max_model_len: int,
):
    kernel_config = _kernel_config(num_rows, max_model_len)

    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    workspace_zero = needs_workspace_zero(
        max_model_len,
        top_k,
        kernel_config["tiered_short_max"],
    )
    launcher = create_topk_per_row_decode_tiered_kernel(
        top_k=top_k,
        **kernel_config,
    )
    return launcher, workspace_slots, workspace_zero


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")


def _required_seq_rows(next_n: int, num_rows: int) -> int:
    if num_rows <= 0:
        return 0
    return math.ceil(num_rows / next_n)


def _validate_inputs(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    ordered: bool = False,
    workspace: torch.Tensor | None = None,
):
    _check_cuda_tensor("logits", logits)
    _check_cuda_tensor("seqLens", seqLens)
    _check_cuda_tensor("indices", indices)

    if logits.dtype is not torch.float32:
        raise TypeError(f"logits must be torch.float32, got {logits.dtype}")
    if seqLens.dtype is not torch.int32:
        raise TypeError(f"seqLens must be torch.int32, got {seqLens.dtype}")
    if indices.dtype is not torch.int32:
        raise TypeError(f"indices must be torch.int32, got {indices.dtype}")
    if logits.device != seqLens.device or logits.device != indices.device:
        raise ValueError("logits, seqLens, and indices must be on the same device")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape={tuple(indices.shape)}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if numRows < 0:
        raise ValueError(f"numRows must be non-negative, got {numRows}")
    if numRows > logits.shape[0]:
        raise ValueError(f"numRows={numRows} exceeds logits rows={logits.shape[0]}")
    if numRows > indices.shape[0]:
        raise ValueError(f"numRows={numRows} exceeds indices rows={indices.shape[0]}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if ordered:
        raise ValueError(
            "FlyDSL decode TopK returns unordered set output only; "
            "ordered=True is not supported"
        )
    if indices.shape[1] < k:
        raise ValueError(f"indices second dimension must be at least k={k}")
    if indices.stride(1) != 1:
        raise ValueError("indices must have contiguous per-row storage")
    if stride1 != 1:
        raise NotImplementedError(
            f"FlyDSL decode TopK currently supports stride1 == 1 only, got {stride1}"
        )
    if stride0 != logits.stride(0) or stride1 != logits.stride(1):
        raise ValueError(
            "stride0/stride1 must match logits.stride(); received "
            f"({stride0}, {stride1}) for logits.stride()={logits.stride()}"
        )

    required_seq_rows = _required_seq_rows(next_n, numRows)
    if required_seq_rows > seqLens.numel():
        raise ValueError(
            f"numRows={numRows} with next_n={next_n} requires at least "
            f"{required_seq_rows} seqLens entries, got {seqLens.numel()}"
        )

    if workspace is not None:
        _check_cuda_tensor("workspace", workspace)
        if workspace.dtype is not torch.int32:
            raise TypeError(f"workspace must be torch.int32, got {workspace.dtype}")
        if workspace.device != logits.device:
            raise ValueError("workspace must be on the same device as logits")


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    stream: torch.cuda.Stream | None = None,
    ordered: bool = False,
    workspace: torch.Tensor | None = None,
) -> None:
    if numRows == 0:
        return

    _validate_inputs(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        ordered,
        workspace,
    )

    launcher, workspace_slots, workspace_zero = _compile_launcher(
        k,
        numRows,
        logits.shape[1],
    )

    if workspace is None:
        workspace = torch.empty(
            workspace_slots,
            dtype=torch.int32,
            device=logits.device,
        )
    elif workspace.numel() < workspace_slots:
        raise ValueError(
            f"workspace too small: need >= {workspace_slots} int32 "
            f"elements, got {workspace.numel()} (use "
            f"flydsl_top_k_per_row_decode_workspace_size)"
        )

    if stream is None:
        stream = torch.cuda.current_stream(logits.device)

    if workspace_zero:
        with torch.cuda.stream(stream):
            workspace.zero_()

    with torch.cuda.device(logits.device.index):
        _run_compiled(
            launcher,
            logits,
            int(next_n),
            seqLens,
            indices,
            workspace,
            int(numRows),
            int(stride0),
            int(stride1),
            stream,
        )


def flydsl_top_k_per_row_decode_fast(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    workspace: torch.Tensor,
) -> None:
    """Low-overhead decode Top-K for hot decode loops (host-overhead levers 2+3).

    Same math and same result set as :func:`flydsl_top_k_per_row_decode`, but it
    strips the per-call host overhead that dominates decode (where the kernel is
    only ~18-40us): it skips input validation and the two per-call context
    managers, and it REQUIRES a caller-owned persistent ``workspace`` (no
    per-call ``torch.empty``). On gfx950 this cuts CPU submit time ~54% and
    per-call latency ~25-38% vs the safe path (SILOTIGER-699 measurements).

    Contract (the caller is responsible for all of this — nothing is checked in
    the hot path):

    * ``logits`` float32 CUDA, 2D, row-major (``stride1 == 1``); ``seqLens`` and
      ``indices`` int32 CUDA on the same device; ``indices`` is
      ``(>=numRows, >=k)`` and row-contiguous; ``stride0/stride1`` equal
      ``logits.stride()``. Pass exactly what you would to the safe path.
    * ``workspace`` is int32 CUDA on ``logits``'s device with at least
      :func:`flydsl_top_k_per_row_decode_workspace_size(numRows, logits.shape[1])`
      elements. It may be reused across calls of the SAME shape; the kernel
      re-zeroes the barrier counters itself when required.
    * The current CUDA device and stream must be the ones you want to launch on.
      The memset and the kernel are both issued on ``current_stream()`` so they
      stay ordered; set device/stream ONCE outside the loop::

          ws = torch.empty(
              flydsl_top_k_per_row_decode_workspace_size(nr, L),
              dtype=torch.int32, device=dev)
          with torch.cuda.device(dev), torch.cuda.stream(s):
              for _ in decode_steps:
                  flydsl_top_k_per_row_decode_fast(
                      logits, next_n, seqLens, indices, nr, s0, s1, k, ws)

    Use :func:`flydsl_top_k_per_row_decode` when you need the validated,
    self-contained (allocates + guards device/stream) entry point.
    """
    if numRows == 0:
        return

    launcher, workspace_slots, workspace_zero = _compile_launcher(
        k,
        numRows,
        logits.shape[1],
    )
    if workspace.numel() < workspace_slots:
        raise ValueError(
            f"workspace too small: need >= {workspace_slots} int32 elements, "
            f"got {workspace.numel()} (use "
            f"flydsl_top_k_per_row_decode_workspace_size)"
        )

    # Barrier-counter bootstrap: the cross-CU barrier requires its counters to
    # start at zero, so the memset is mandatory for multi-block configs (skipping
    # it deadlocks the barrier). Issue it on the current stream so it is ordered
    # before the kernel launch below without a stream context manager.
    if workspace_zero:
        workspace.zero_()

    _run_compiled(
        launcher,
        logits,
        int(next_n),
        seqLens,
        indices,
        workspace,
        int(numRows),
        int(stride0),
        int(stride1),
        torch.cuda.current_stream(),
    )


def flydsl_top_k_per_row_decode_unordered(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Benchmark-friendly wrapper for the unordered set-output path."""

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k=k,
        stream=stream,
        ordered=False,
    )
