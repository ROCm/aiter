# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Spec builders and the validated cell table for MiniMax-M3 SP4 QKV (pure Python, no torch).

Field map authority: golden case_config.py field_spec (sp4-attention-fusion-cases-20261009):
  q (0,8192,4) k (8192,512,4) v (8704,512,4) [iq (9216,512,4) ik (9728,128,1)]  -> dense N=9216 F=2304; indexer N=9856 F=2560.
Reuse uses the indexer GEMM fields but drops iq/ik in the exchange (compact F=2304, approved).
"""

from __future__ import annotations

from aiter.gemm_fusion.spec import (
    FusionSpec,
    GemmSpec,
    HeadNormRopeCache,
    SPExchange,
    qkv_fields,
)

MODES = ("dense", "indexer", "reuse")
WORLD = 4
HIDDEN_K = 6144
# Golden case_config.cache_page_size = 16 if has_indexer else 128: dense -> page128; indexer AND reuse (N=9856) -> page16.
KV_PAGE = {"dense": 128, "indexer": 16, "reuse": 16}


def kv_cache_str(mode: str) -> str:
    return f"shuffle_page{KV_PAGE[mode]}_fp8_ptok"


def sp_qkv_spec(mode: str, world: int = WORLD, eps: float = 1e-6) -> FusionSpec:
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "dense":
        fields = qkv_fields(q=8192, kv=512, world=world, kv_shards=world)
        n, drop, idx, num = 9216, (), None, "triton_norm_rope_cache_v1"
    else:
        fields = qkv_fields(
            q=8192, kv=512, world=world, iq=512, ik=128, kv_shards=world, ik_shards=1
        )
        n = 9856
        if mode == "indexer":
            drop, idx, num = (), "fp8_unit_page128", "aiter_fused_qknorm_idxrqknorm_v1"
        else:
            drop, idx, num = ("iq", "ik"), None, "triton_norm_rope_cache_v1"
    return FusionSpec(
        GemmSpec(n=n, k=HIDDEN_K),
        (
            SPExchange(world=world, fields=fields, drop=drop),
            HeadNormRopeCache(
                head_dim=128,
                q_heads=16,
                kv_heads=1,
                rotary_dim=64,
                eps=eps,
                numerics=num,
                kv_cache=kv_cache_str(mode),
                index=idx,
            ),
        ),
    )


def mode_of(spec: FusionSpec) -> str:
    """Recover the mode from a spec (used by the variant to select the compiled geometry)."""
    ex = spec.stages[0]
    names = {f.name for f in ex.fields}
    if "iq" not in names:
        return "dense"
    return "reuse" if "iq" in ex.drop else "indexer"


# (mode, tokens T) -> (vid, params). local M = T / 4. Ledger r165 (small) / r152 (large, filled in S2-2b-iii).
SMALL_CELLS = {
    ("dense", 64): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1},
    ),
    ("dense", 128): (
        "sp_qkv.small.v1",
        {
            "src": "r18",
            "sk": 3,
            "nh": 1,
            "no_credit": 1,
            "b_rowsplit": 1,
            "rx_bypass": 1,
        },
    ),
    ("indexer", 64): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1, "push8": 1},
    ),
    ("indexer", 128): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1},
    ),
    ("reuse", 64): (
        "sp_qkv.small.v1",
        {"src": "r18", "sk": 3, "nh": 1, "no_credit": 1, "reuse_no_idx": 1},
    ),
    ("reuse", 128): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1, "push_halves": 1},
    ),
}

# The kernel source the shipped PROFILE runs for every small cell: r19 (R12: per-launch kernargs, per-comm ERR/STEP page, call_id tags; derived from r18 so it carries
# every r17/r18 shape switch). SMALL_CELLS keeps the per-cell r17/r18 sources of the pre-R12 build: they stay selectable per row (set "src" back) for the A/B.
SMALL_PROFILE_SRC = "r19"


def small_cell_params(mode: str, T: int) -> dict:
    """The profile params of a small cell: the tuned plan params with the kernel source the profile selects."""
    return {**SMALL_CELLS[(mode, T)][1], "src": SMALL_PROFILE_SRC}


# ---- small-M coverage beyond the two tuned cells (T64 = m16, T128 = m32) -----------------------------------------------------------------------------------------------
# R13 P2: the decode buckets below T=64 (ATOM's default capture list pads to T = 4, 8, 16, 32, 48 -> local m = 1, 2, 4, 8, 12) run on r19 with a COPY of their mode's m=16
# params (same build, same owner / windows / context: no new unit, no new window). R13 MP64 (r20, SM_MP 64) serves local m 33..64: an exact row at m=64 (T=256, the default
# captured bucket) and a range rule over 33..64 (any padded T = 4 m in 132..256, e.g. a custom capture list or an eager step). r20 has ONE push shape (the previous-generation shape switches
# do not exist), requires rx_bypass=1 in every mode and B_HEADS=1; reuse keeps reuse_no_idx=0 (untuned). The MP128 slot serves local m 65..128 (an exact row at m=128 = T=512, the
# default captured bucket, and a range rule over 65..128): r22 (tuned MP128, SM_MP 128, 3-slot Rx in ONE 4 MiB view) {nh 1, b_quad 1, gemm_bn128 0, rx3 1} (SMALL_R22_IN_PROFILE),
# which replaced r21 (SM_MP 128, nh=2, its Rx needs an 8 MiB arena view; still selectable, SMALL_R21_IN_PROFILE). The MP256 slot serves local m 129..255 (T 516..1020) with ONE range
# rule per mode and no exact row (the default capture list pads to T = 256, 512 at most): r23 (SM_MP 256, 3-slot Rx in an 8 MiB view) {nh 1, b_quad 1, rx3 1, gemm_bn128 0, pfk 4,
# p_nt 2, push_passtrip 1} (SMALL_R23_IN_PROFILE) plus its bucket row at m=256. So every local m 1..256 fuses on the small plans (pad-M buckets 1, 2, 4, 8, 12, 16, 32, 64, 128, 256); local m
# 257..8192 pads into the large r18 buckets (512 from its floor 386, 768 from 513, 1024 from 769, then every multiple of 256); m 257..385 is native, and m > 8192 is native.
SMALL_P2_M = (1, 2, 4, 8, 12)
# R16 pad-M, the large buckets (match.bucket_for_m): a real local m is padded into the first exact large row >= m (the bucket) only from that row's floor on: the row's own pad_min, else the
# profile field m_pad_min_large (m_pad_min_large()). SHIPPED (LARGE_R18_IN_PROFILE on): the r18 route-B bucket rows carry their own floors, 386 (bucket 512) / 513 (bucket 768) in every mode
# (Evidence's route-B timing: B/C >= 1.0 from m 386 in the 512 bucket and from 513 in the 768 bucket), and the profile field is 386; m 257..385 is native.
# R13 ROLLBACK layout (LARGE_R18_IN_PROFILE off), for the record: Evidence's floors on the r13 profile with the 512 / 768 / 1024 buckets (two windows, worst B/C vs native):
#   bucket 512 : never pays for dense (0.958 even at m=512 exact) / indexer (0.979); reuse only ~1.005 from m=450 up; n=2 is worse  -> NOT in the r13 layout (capability kept below);
#   bucket 768 : dense >= 1.0 from m=700 (1.015), indexer from 600 (1.016), reuse from 513 (1.022)                                  -> the r13 layout ships it with per-mode pad_min 700 / 600 / 513;
#   bucket 1024: every mode >= 1.0 at m=769 (dense 1.025, indexer / reuse ~1.19)                                                      -> shipped, pad_min 769 in both layouts.
# Both layouts carry pad_min on the bucket and 1024 rows, so m_pad_min_large only matters for rows WITHOUT one (from 1280 up): 386 shipped / 769 in the r13 layout.
M_PAD_MIN_LARGE = 769  # the r13 rollback field (specs.m_pad_min_large() returns LARGE_R18_M_PAD_MIN_LARGE = 386 for the shipped r18 layout)


def m_pad_min_large() -> int:
    """The profile field m_pad_min_large of the profile being generated: 386 for the shipped r18 layout, 769 in the r13 rollback layout (gen_profile reads the flag at call time)."""
    return LARGE_R18_M_PAD_MIN_LARGE if LARGE_R18_IN_PROFILE else M_PAD_MIN_LARGE


# The large buckets BELOW the swept grid (p1_sweep.P1_GRID = 1024..8192 step 256): local m -> (n, half_first), same for the three modes. NOT swept: n = 1 / half_first = 0 is the best at
# the smallest swept cells (m=1024 and 1280, every mode, p1_sweep.P1_BEST) and the only legal puts_per_peer at 3 chunks (n must divide the chunk count; n=3 exists only as quarter-first at
# 8 chunks); 2 chunks would also allow n=2, which Evidence measured worse at 512. The CAPABILITY (both buckets) is kept; LARGE_PAD_BUCKETS_IN_PROFILE selects what the profile ships.
LARGE_PAD_BUCKETS = {512: (1, 0), 768: (1, 0)}
LARGE_PAD_BUCKETS_IN_PROFILE = (
    768,
)  # the r13 ROLLBACK set (not shipped while LARGE_R18_IN_PROFILE is on: the r18 layout ships 512 and 768); 512 dropped: it never pays for dense / indexer and only ~1.005 for reuse (see above)
# Per-mode floors (row.pad_min) of the buckets below / at the start of the grid, from the Evidence numbers above. The 512 entries are the capability's floors if it is switched on: dense /
# indexer never pay (floor = 512: exact m only), reuse from 450.
LARGE_BUCKET_PAD_MIN = {
    "dense": {512: 512, 768: 700, 1024: 769},
    "indexer": {512: 512, 768: 600, 1024: 769},
    "reuse": {512: 450, 768: 513, 1024: 769},
}

# R18 route B (src r18; SHIPPED): the large buckets as the THREE-kernel pipeline GEMM (large/r17 FlyDSL dst-map GEMM, writes S with srows = the launch M) + put kernel + r13's
# quiet_ready (large/r18-put, lg_put_launch_v1), receiver as r13. ON (gen_profile reads it at CALL time): the profile ships, per mode, a row at each
# bucket of LARGE_R18_BUCKETS (default both 512 and 768; the plan serves every real m up to its bucket by pad-M: m 386..512 use the 512 bucket, 513..768 the 768 bucket) with src r18
# {n 1, half_first 0, no_gridread 1} (no `tile`: the wave rule r18_gemm.tile_for picks the FlyDSL tile per launch M; a row can pin it with "tile": 477 | 236) and the floors below
# (every mode; from Evidence's timing: route the fused path where B/C >= 1.0 in both windows). The 1024.. grid rows stay r13.
LARGE_R18_IN_PROFILE = True
LARGE_R18_SRC = "r18"
LARGE_R18_BUCKETS = (512, 768)
LARGE_R18_BUCKET_PAD_MIN = {
    512: 386,
    768: 513,
}  # pad_min, all modes (Evidence's sweep: B/C >= 1.02 from m 386 in the 512 bucket, >= 1.03 from 513 in the 768 bucket)
LARGE_R18_M_PAD_MIN_LARGE = 386  # the profile field m_pad_min_large of an r18 profile


def large_pad_buckets_in_profile() -> tuple[int, ...]:
    """The large buckets below the swept grid the profile being generated ships: both 512 and 768 in the shipped r18 route-B layout; (768,) in the r13 rollback layout."""
    if LARGE_R18_IN_PROFILE:
        return tuple(sorted(LARGE_R18_BUCKETS))
    return LARGE_PAD_BUCKETS_IN_PROFILE


def large_bucket_params(m: int) -> dict:
    """The plan params of a below-grid bucket row: r13g's n=1 plan or the r18 route-B plan (the shipped layout; same n / half_first 0 / no_gridread 1)."""
    n, half_first = LARGE_PAD_BUCKETS[m]
    if LARGE_R18_IN_PROFILE and m in LARGE_R18_BUCKETS:
        return {
            "n": n,
            "half_first": half_first,
            "no_gridread": 1,
            "src": LARGE_R18_SRC,
        }
    return {
        "n": n,
        "half_first": half_first,
        "no_gridread": 1,
        "src": large_profile_src(),
    }


def large_bucket_pad_min(mode: str, m: int) -> int:
    """Row pad_min of a bucket (512 / 768 / 1024) of the profile being generated: the r18 route-B floors (512 / 768) or Evidence's per-mode r13 floors (1024)."""
    if LARGE_R18_IN_PROFILE and m in LARGE_R18_BUCKET_PAD_MIN:
        return LARGE_R18_BUCKET_PAD_MIN[m]
    return LARGE_BUCKET_PAD_MIN[mode][m]


SMALL_R20_ROW_M = (64,)
SMALL_R20_RULE_M = (33, 64)  # (m_min, m_max), every m (m_multiple 1)
SMALL_R20_SRC = "r20"
SMALL_R21_ROW_M = (128,)
SMALL_R21_RULE_M = (65, 128)  # (m_min, m_max), every m (m_multiple 1)
SMALL_R21_SRC = "r21"
# MP128 slot (exact rows at m=128 + the rule over 65..128), history and numbers:
#   r21 nh=2 (216 / 240 B WGs, grid 336 / 368) beat native in the fresh GPU58 timing (tick-free, X1 on, two windows): T=260 (m=65) 1.35-1.42x, T=384 (m=96) 1.25-1.30x, T=512 (m=128)
#   1.10-1.16x, all 3 modes; r21 nh=1: 0.95-1.18x (an earlier formal timing had put nh=1 at 0.90-0.97x at T=512, which is why r21 was removed once: nh=1 is not worth selecting).
#   r22 {nh 1, b_quad 1, gemm_bn128 0, rx3 1} (B-quad, BM128 x BN64 GEMM, 3-slot Rx: ONE 4 MiB view, grid 228 / 248 <= 256 resident) is the SHIPPED arm: it passed all 8 GPU gates
#   (outputs byte-identical to r21 nh=2 at 924 / 924, the controls bite, G10 quiet / loud, hang, dead-pass, G8) and measured T=512 1.19-1.23x of native vs r21 nh=2 at 1.09-1.16x
#   (BN128 is a net loss: GEMM 200 -> 360 VGPR + 128 AGPR). The shipped arm also sets p_nt=2 (P stored sc0 sc1 nt): 3.7-3.9 % faster at T512 (58.2 / 59.0 / 58.1 us, B/C 1.23-1.28x),
#   3.7-4.3 % at T384, never slower at T260; outputs byte-exact vs golden in all arms, RAN .text identical. It also sets push_passtrip=1 (the push live-set option; all gates passed:
#   sweep 63 / 63, byte-identical to the previous build, controls, G10 quiet / loud, G8, hang, dead-pass): another 4.3-4.5 % at T512 (55.7 / 56.5 / 55.6 us, B/C 1.28-1.33x) and 1.3-3.4 % at
#   T260 / T384. r21 stays in the code, selectable (src r21 in a row / rule, or SMALL_R21_IN_PROFILE with SMALL_R22_IN_PROFILE off).
# SMALL_R21_IN_PROFILE / SMALL_R22_IN_PROFILE gate the MP128 rows / rule (gen_profile reads them at CALL time; r22 takes the slot when both are on; both off = the r21-off layout,
# m 65..255 native, which tests switch on to keep it exercised).
SMALL_R21_IN_PROFILE = False
SMALL_R21_PROFILE_NH = 2  # B_HEADS of r21's rows / rule WHEN SMALL_R21_IN_PROFILE (small_r21_params(mode, nh) accepts 1 or 2)
SMALL_R22_SRC = "r22"
SMALL_R22_IN_PROFILE = True
SMALL_R22_PROFILE_NH = 1  # 1 = the B-quad design (grid 228 / 248 <= 256 resident); 2 = B-quad with 2 heads per B WG (grid 174 / 188). A row / rule can also select r22 directly + the knobs
# r23 (MP256 v1, R14 card: BM256 GEMM, 2 iterations per RAN WG, 3-slot Rx 7,077,888 / 7,864,320 B = one 8 MiB view, P 28.3 / 30.3 MB) IS in the shipped profile (SMALL_R23_IN_PROFILE
# = True): ONE range rule per mode covers local m 129..255 (T 516..1020; no exact row: the default capture list pads to T = 256, 512 at most) with small_r23_params(mode,
# SMALL_R23_PROFILE_NH): the arm {nh 1, b_quad 1, rx3 1, gemm_bn128 0, pfk 4, p_nt 2, push_passtrip 1} for every mode. GPU58 (the all-passtrip build): every gate passed (ISA 0 scratch, sweep
# 45 / 45, hang, A-8, controls (iii) / (iv), G10 quiet / loud, G8; N-5: caches byte-exact, only 1-ULP q differences in one cell (m=192, pad 1), deterministic); T=1020 C 92.1 / 92.8 /
# 92.3 us = 1.24-1.30x of native, T=516 1.34-1.39x. r23 requires rx3 + b_quad (the header #errors otherwise); gemm_bn128 is optional (default 0: BM256 x BN64 PFK4, the GPU-measured better arm).
SMALL_R23_SRC = "r23"
SMALL_R23_IN_PROFILE = True
SMALL_R23_RULE_M = (
    129,
    255,
)  # (m_min, m_max), every m (m_multiple 1); the plan itself takes local M 1..256
SMALL_R23_PROFILE_NH = 1
# R16 pad-M: ONE exact r23 row per mode at the top of its accepted range (SM_MP 256, local m 1..256, T=1024): the BUCKET that real local m 129..256 are padded into (bucket_for_m).
# The rule over 129..255 keeps serving exact m; the row adds no owner / unit / window (same params as the rule).
SMALL_R23_ROW_M = (256,)
_NATIVE_NOTE_R23_ONLY = "no fused path (T 260..512): native by design in this layout (r23 MP256 serves T 516..1020; no MP128 source is selected)"
_NATIVE_NOTE_R21_ON = "no fused path (T 516..1020): native by design; the small kernels end at m=128 (r21 MP128, nh=2: 1.10-1.42x of native over T 260..512, GPU58 tick-free X1 timing), the large path starts at m=1024"
_NATIVE_NOTE_R22_ON = "no fused path (T 516..1020): native by design; the small kernels end at m=128 (r22 MP128 {nh 1, BN64, p_nt 2, push_passtrip 1}: T512 B/C 1.28-1.33x of native (55.7 / 56.5 / 55.6 us; 1.23-1.28x before passtrip, 1.19-1.23x before p_nt) vs r21 nh=2 at 1.09-1.16x, GPU58 timing, all gates passed), the large path starts at m=1024"
_NATIVE_NOTE_R21_OFF = "no fused path (T 260..1020, incl. the default capture bucket T=512): native by design; the small kernels end at m=64, the large path starts at m=1024 (r21 MP128 is not selected in this layout)"
NATIVE_M_RANGES = ()  # the SHIPPED layout (r22 + r23 on): every local m 1..255 fuses (local m 257..385 is native; above it the large r18 buckets 512 / 768 and the r13 1024.. rows pad)


def native_m_ranges() -> tuple[dict, ...]:
    """The documented native ranges for the current profile definition (inside the small envelope, local m 1..255). Shipped (r22 and r23 on): none, every local m fuses.
    r23 on without an MP128 source: m 65..128. An MP128 source (r21 or r22) without r23: m 129..255. Neither: m 65..255.
    """
    mp128 = SMALL_R21_IN_PROFILE or SMALL_R22_IN_PROFILE
    if SMALL_R23_IN_PROFILE:
        return (
            ()
            if mp128
            else ({"m_min": 65, "m_max": 128, "note": _NATIVE_NOTE_R23_ONLY},)
        )
    if SMALL_R22_IN_PROFILE:
        return ({"m_min": 129, "m_max": 255, "note": _NATIVE_NOTE_R22_ON},)
    if mp128:  # r21 only
        return ({"m_min": 129, "m_max": 255, "note": _NATIVE_NOTE_R21_ON},)
    return ({"m_min": 65, "m_max": 255, "note": _NATIVE_NOTE_R21_OFF},)


def small_r20_params(mode: str) -> dict:
    """Params of the MP64 build of a mode (same for every m of the r20 rows and rule: one owner per mode)."""
    if mode not in MODES:
        raise ValueError(mode)
    return {
        "src": SMALL_R20_SRC,
        "sk": 3,
        "nh": 1,
        "no_credit": 1,
        "rx_bypass": 1,
    }


def small_r21_params(mode: str, nh: int = 1) -> dict:
    """Params of the MP128 build of a mode (same for every m of the r21 rows and rule: one owner per mode). nh = B_HEADS: 1 (default, 432 / 480 B WGs) or 2 (216 / 240,
    the A/B for the occupancy caveat); a profile row / rule selects it through its params.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if nh not in (1, 2):
        raise ValueError(f"r21 supports nh in (1, 2), got {nh}")
    return {
        "src": SMALL_R21_SRC,
        "sk": 3,
        "nh": nh,
        "no_credit": 1,
        "rx_bypass": 1,
    }


def small_r22_params(mode: str, nh: int = 1) -> dict:
    """Params of the r22 (tuned MP128) build of a mode: {src r22, sk 3, nh, no_credit 1, rx_bypass 1} plus the option switches EXPLICITLY at the PROFILE values b_quad 1,
    gemm_bn128 0, rx3 1, p_nt 2, push_passtrip 1: the GPU result puts r22 with BN64 (gemm_bn128=0) as the best MP128 arm (T512 1.19-1.23x vs r21 nh=2 at 1.09-1.16x; BN128 is a net loss), and
    p_nt=2 (the GEMM epilogue stores P as sc0 sc1 nt, values and RAN .text unchanged) is 3.7-3.9 % faster at T512 (58.2 / 59.0 / 58.1 us, B/C 1.23-1.28x), 3.7-4.3 % at T384 and
    never slower at T260, byte-exact vs golden in all arms, and push_passtrip=1 (two one-pass push sub-trips, pk 72 floats) on top is another 4.3-4.5 % at T512 (55.7 / 56.5 / 55.6 us,
    B/C 1.28-1.33x) and 1.3-3.4 % at T260 / T384 (sweep 63 / 63, byte-identical to the previous build, controls, G10 quiet / loud, G8, hang, dead-pass); so the profile builds that arm.
    (The source's own defaults are gemm_bn128 1, p_nt 0 and push_passtrip 0, small_defs' defaults for an r22 row that does not set them: a row / rule overrides through its params.) nh = B_HEADS 1 (default: B-quad, 228 / 248 WGs) or 2.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if nh not in (1, 2):
        raise ValueError(f"r22 supports nh in (1, 2), got {nh}")
    return {
        "src": SMALL_R22_SRC,
        "sk": 3,
        "nh": nh,
        "no_credit": 1,
        "rx_bypass": 1,
        "b_quad": 1,
        "gemm_bn128": 0,
        "rx3": 1,
        "p_nt": 2,
        "push_passtrip": 1,
    }


def small_r23_params(mode: str, nh: int = 1) -> dict:
    """Params of the r23 (MP256) build of a mode, at the PROFILE values (SMALL_R23_IN_PROFILE is True: all GPU gates passed on the all-passtrip build):
    {src r23, sk 3, nh, no_credit 1, rx_bypass 1} plus the options explicit: b_quad 1 and rx3 1 (REQUIRED by the source), gemm_bn128 0 (BM256 x BN64), pfk 4
    (BN64's default: 4 x (32 + 8) KiB = 160 KiB), p_nt 2 (the P store flavour that won on r22) and push_passtrip 1 (the push live-set option), the same for every mode. nh = B_HEADS 1
    (default) or 2. The remaining options (ik_reload_p, ik_seq, push_ws_late, sk 2, ...) are NOT part of the dict: they stay at the source defaults and a row / rule can set them.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if nh not in (1, 2):
        raise ValueError(f"r23 supports nh in (1, 2), got {nh}")
    return {
        "src": SMALL_R23_SRC,
        "sk": 3,
        "nh": nh,
        "no_credit": 1,
        "rx_bypass": 1,
        "b_quad": 1,
        "rx3": 1,
        "gemm_bn128": 0,
        "pfk": 4,
        "p_nt": 2,
        "push_passtrip": 1,
    }


def _mp128_params(mode: str) -> dict | None:
    """The params of the MP128 slot (exact rows at m=128 + the rule over 65..128) of the current profile definition: r22 when SMALL_R22_IN_PROFILE, else r21 when
    SMALL_R21_IN_PROFILE, else None (no MP128 rows / rule: m 65..255 native)."""
    if SMALL_R22_IN_PROFILE:
        return small_r22_params(mode, SMALL_R22_PROFILE_NH)
    if SMALL_R21_IN_PROFILE:
        return small_r21_params(mode, SMALL_R21_PROFILE_NH)
    return None


def small_exact_cells() -> list[tuple[str, int, str, dict]]:
    """Every EXACT small profile row as (mode, local m, vid, params), ordered by mode then m: the two tuned cells (m16, m32), the P2 copies of the m16 params
    (m = 1, 2, 4, 8, 12), the r20 row(s) at m=64 and the r21 row(s) at m=128."""
    out = []
    for mode in MODES:
        vid16, p16 = SMALL_CELLS[(mode, 64)]
        cells = {
            T // 4: (vid, small_cell_params(mode, T))
            for (md, T), (vid, _p) in SMALL_CELLS.items()
            if md == mode
        }
        for m in SMALL_P2_M:
            cells[m] = (vid16, {**p16, "src": SMALL_PROFILE_SRC})
        for m in SMALL_R20_ROW_M:
            cells[m] = (vid16, small_r20_params(mode))
        mp128 = _mp128_params(mode)
        for m in SMALL_R21_ROW_M if mp128 else ():
            cells[m] = (vid16, mp128)
        for m in SMALL_R23_ROW_M if SMALL_R23_IN_PROFILE else ():
            cells[m] = (vid16, small_r23_params(mode, SMALL_R23_PROFILE_NH))
        for m in sorted(cells):
            out.append((mode, m, cells[m][0], cells[m][1]))
    return out


def small_range_rules() -> list[dict]:
    """The small range rules as dicts {mode, m_min, m_max, m_multiple, vid, params, name, note}, per mode (m_multiple 1, disjoint ranges; exact rows win):
    local m 1..16 on r19 with a COPY of the mode's m=16 params, 17..32 on r19 with a COPY of the m=32 params, 33..64 on r20 (MP64), and (only when SMALL_R21_IN_PROFILE / SMALL_R22_IN_PROFILE) 65..128
    on r21 (MP128). So eager decode and custom capture lists at ANY local m <= 64 fuse (<= 128 with r21). The shape switches follow the copy: push8 (an M <= 16 shape) only below 17, b_rowsplit /
    push_halves (M > 16 shapes) only above 16; the generator validates every covered m against the real plan constructor.
    """
    out = []
    for mode in MODES:
        vid, _p = SMALL_CELLS[(mode, 64)]
        ranges = [
            (
                1,
                16,
                small_cell_params(mode, 64),
                "r19_m1-16",
                "copy of the mode's m=16 params (r19); exact rows win at m 1, 2, 4, 8, 12, 16",
            ),
            (
                17,
                32,
                small_cell_params(mode, 128),
                "r19_m17-32",
                "copy of the mode's m=32 params (r19); exact row wins at m=32",
            ),
            (
                SMALL_R20_RULE_M[0],
                SMALL_R20_RULE_M[1],
                small_r20_params(mode),
                f"mp64_m{SMALL_R20_RULE_M[0]}-{SMALL_R20_RULE_M[1]}",
                "R13 MP64 (src r20), any padded T = 4 m in 132..256; m=64 (T=256, the default capture bucket) is an exact row",
            ),
        ]
        mp128 = _mp128_params(mode)
        if mp128:
            ranges.append(
                (
                    SMALL_R21_RULE_M[0],
                    SMALL_R21_RULE_M[1],
                    mp128,
                    f"mp128_m{SMALL_R21_RULE_M[0]}-{SMALL_R21_RULE_M[1]}",
                    (
                        f"R14 MP128 tuned (src r22, nh={mp128['nh']}, 4 MiB Rx view), any padded T = 4 m in 260..512; m=128 (T=512, the default capture bucket) is an exact row"
                        if mp128["src"] == SMALL_R22_SRC
                        else f"R13 MP128 (src r21, nh={mp128['nh']}, 8 MiB Rx view), any padded T = 4 m in 260..512; m=128 (T=512, the default capture bucket) is an exact row"
                    ),
                )
            )
        if SMALL_R23_IN_PROFILE:
            ranges.append(
                (
                    SMALL_R23_RULE_M[0],
                    SMALL_R23_RULE_M[1],
                    small_r23_params(mode, SMALL_R23_PROFILE_NH),
                    f"mp256_m{SMALL_R23_RULE_M[0]}-{SMALL_R23_RULE_M[1]}",
                    f"R14 MP256 (src r23, nh={SMALL_R23_PROFILE_NH}, 8 MiB Rx view, 3 slots), any padded T = 4 m in 516..1020",
                )
            )
        for lo, hi, params, label, note in ranges:
            out.append(
                {
                    "mode": mode,
                    "m_min": lo,
                    "m_max": hi,
                    "m_multiple": 1,
                    "vid": vid,
                    "params": params,
                    "name": f"{mode}_{label}",
                    "note": note,
                }
            )
    return out


# Large cells: (mode, tokens T) -> (vid, params). local M = T/4 in {2048, 4096, 8192}. Ledger r152 + r154 (gcnasm 822c1d1, r11 recipe).
# DEVIATION (recorded): all nine cells use the single LG-QSDMA-r11 code base (NO_GRIDREAD=1 default) with the per-cell plan below,
# instead of the individual r6b/r6x4/r7/r8g candidates of ledger r152 (r11 plans reproduce them; gcnasm 822c1d1 README: all 2-12 us faster, same numerics).
#   plan: n = SDMA puts per peer; half_first 0 = r6x plan, 1 = half-N first group (r7), 2 = quarter-first n3q (r9, dense M=2048 only, n=3).
LARGE_CELLS = {
    ("dense", 8192): (
        "sp_qkv.large.v1",
        {"n": 3, "half_first": 2, "no_gridread": 1},
    ),  # r11 n3q
    ("dense", 16384): (
        "sp_qkv.large.v1",
        {"n": 2, "half_first": 0, "no_gridread": 1},
    ),  # r6b n2
    ("dense", 32768): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6b n4
    ("indexer", 8192): (
        "sp_qkv.large.v1",
        {"n": 2, "half_first": 1, "no_gridread": 1},
    ),  # r7 n2h
    ("indexer", 16384): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
    ("indexer", 32768): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
    ("reuse", 8192): (
        "sp_qkv.large.v1",
        {"n": 2, "half_first": 1, "no_gridread": 1},
    ),  # r8g F0 n2h (dense build, compact)
    ("reuse", 16384): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
    ("reuse", 32768): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
}

LARGE_PROFILE_SRC = "r12"  # the large rows/rules of the shipped profile run the R12 kernels; "r11" stays selectable per row/rule for the A/B
# R16 pad-M: the large rows / rules run src r13 (lg_launch_v3: the kernel gets the valid m_real, the bucket M only selects the plan). Evidence on r13g (Triton 3.8 env, same-window
# ratios): ISA clean (0 spills); G1 byte-identical to r12 at m == M; odd m_real 1025..4351 byte-exact vs native; the m_real disagreement fault is detected as 0x31 with abort / reset /
# recovery working; timing r13g = r12 within +-1 % at the anchors and the pad path beats native at every m_real >= 1024 (B/C 1.07-1.23). False = the r12 layout (A/B, rollback);
# gen_profile reads it at CALL time.
LARGE_R13_IN_PROFILE = True
LARGE_R13_SRC = "r13"


def large_profile_src() -> str:
    """The kernel source of the large rows / rules of the profile being generated: r13 when LARGE_R13_IN_PROFILE, else LARGE_PROFILE_SRC (r12)."""
    return LARGE_R13_SRC if LARGE_R13_IN_PROFILE else LARGE_PROFILE_SRC


def large_cell_params(mode: str, T: int) -> dict:
    """The profile params of a large cell: the tuned plan params plus the kernel source the profile selects."""
    _vid, params = large_cells()[(mode, T)]
    src = LARGE_R19_SRC if (mode, T) in LARGE_R19_CELLS else large_profile_src()
    return {**params, "src": src}


# ROUTE A (local m <= 16384): the extended band on large/r19 (ABI 7). Flag off = the pre-r19 layout (rollback is one flag); the flag is read at
# profile-generation time, so the shipped JSON is generated with it on.
LARGE_R13_M_MAX = 8192
LARGE_R19_IN_PROFILE = False  # SHIPPING DEFAULT: the r19 capability is opt-in; the alternate profile ..._r19.json is generated with the flag temporarily on
LARGE_R19_SRC = "r19"
LARGE_R19_M_MAX = 16384
# exact-row stride of the extended band: 1024 local rows keeps chunks = M/256 a multiple of 4 -> n=4 (and n=1/2) legal everywhere and the GEMM/put
# overlap available; the 256-step grid would leave odd chunks (33, 35, ...) with n=1 only (no overlap). Rows at 9216..16384 step 1024 = 8 per mode.
LARGE_R19_GRID_STRIDE = 1024
# window-budget cells of the extended band (T = 4 * M): merged by large_cells() only while the flag is on, so the rolled-back budget stays 9 large owners.
# Only the LARGEST extended plan needs a budget cell: indexer F=2560 / M=16384 -> 324 MiB; the dense / reuse M=16384 windows are 292 MiB and fit in the same
# shared S / R. One cell keeps the owner model exact: large owners 9 -> 10, max_windows 58 -> 61 (= 2*15 + 3*10 + 1).
LARGE_R19_CELLS = {
    ("indexer", 65536): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),
}
# the tag field of the large launches: r13 / r18 keep the 13-bit m_real + 19-bit call_id tag, r19 repartitions it to 14-bit m_real + 18-bit call_id
LARGE_TAG_BITS = {"r11": 19, "r12": 19, "r13": 19, "r18": 19, "r19": 18}


def large_tag_bits(src: str) -> int:
    """The low call_id tag bits of a large source's launches (the profile's static key check must hold on them, per token count)."""
    return LARGE_TAG_BITS.get(src, 19)


def large_cells() -> dict:
    """(mode, T) -> (vid, params) cells the window budget uses: the legacy nine cells, plus the r19 T=65536 cell while LARGE_R19_IN_PROFILE is on."""
    out = dict(LARGE_CELLS)
    if LARGE_R19_IN_PROFILE:
        out.update(LARGE_R19_CELLS)
    return out


def large_row_src(m: int) -> str:
    """The kernel source of a large exact row: r19 in the extended band while the flag is on, else the layout's large source (r13 / r12)."""
    if LARGE_R19_IN_PROFILE and m > LARGE_R13_M_MAX:
        return LARGE_R19_SRC
    return large_profile_src()


def large_r19_grid_m() -> tuple[int, ...]:
    """The extended-band exact-row M values for the CURRENT stride: 9216..16384 step 1024 (or 8448..16384 step 256 for the Evidence A/B)."""
    return tuple(
        m
        for m in range(LARGE_R13_M_MAX + 256, LARGE_R19_M_MAX + 1, 256)
        if m % LARGE_R19_GRID_STRIDE == 0
    )
