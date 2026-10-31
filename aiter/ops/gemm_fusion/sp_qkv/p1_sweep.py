# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-cell best (n, half_first) of the P1 large/r12 legal-parameter sweep: the data behind the exact large profile rows (gen_profile.build_large_rows).

Provenance (independent Evidence Owner): AAC16 job 8438, mi355-gpu-58, 4 ranks, physical GPU0-3; grid m = 1024..8192 step 256 (29 points) x dense / indexer / reuse = 87
cells, 226 legal parameter points timed; archive aac58-r12-p1-01.tar sha256 15786d49bc0821641e6c99e9f1ff6605bb595713121dcf48f0faca80ee96ae3c; table =
raw/results/r12-p1/result-rank*.json (all four ranks pick the same best) ; every best point re-run against the unmodified golden: byte-exact on q/k/v/k_cache/v_cache/k_scale/
v_scale, zero device ERR (status PASS_SCOPED_BEST_CORRECT in all 87 cells).
Timing kind: C_med = median over 3 repeats of the max-over-ranks mean of 101 whole-graph CUDA-event replays. This is SELECTION evidence only, not the formal re-timing
protocol and not a candidate performance claim.
Legality (r12 host driver lg_comm_driver.cpp:163-169): puts_per_peer n in {1,2,4} with (m/256) % n == 0, except quarter-first (n=3, half_first=2) for dense m=2048 only;
half_first=1 needs n in {2,4}. Compact m-range rules lose up to 4.6 % against this table (half_first at n=4 has no closed form), hence one exact row per cell.
"""

P1_SWEEP_ARCHIVE_SHA256 = (
    "15786d49bc0821641e6c99e9f1ff6605bb595713121dcf48f0faca80ee96ae3c"
)
P1_SWEEP_SOURCE = "records/evidence-owner/stage2-evidence/aac16-58-r12-p1 (AAC16 job 8438, mi355-gpu-58)"
# P1_GRID is defined at the bottom of this file: the measured 29-point grid plus the ROUTE A extension (the 24 measured
# 1024-stride buckets; only the explicit 256-stride A/B scan stays placeholder-filled).

# mode -> {local m: (n, half_first)}
P1_BEST: dict[str, dict[int, tuple[int, int]]] = {
    "dense": {
        1024: (1, 0),
        1280: (1, 0),
        1536: (2, 1),
        1792: (1, 0),
        2048: (3, 2),
        2304: (1, 0),
        2560: (2, 0),
        2816: (1, 0),
        3072: (4, 1),
        3328: (1, 0),
        3584: (2, 0),
        3840: (1, 0),
        4096: (4, 1),
        4352: (1, 0),
        4608: (2, 0),
        4864: (1, 0),
        5120: (4, 0),
        5376: (1, 0),
        5632: (2, 0),
        5888: (1, 0),
        6144: (4, 0),
        6400: (1, 0),
        6656: (2, 0),
        6912: (1, 0),
        7168: (4, 0),
        7424: (1, 0),
        7680: (2, 0),
        7936: (1, 0),
        8192: (4, 0),
    },
    "indexer": {
        1024: (1, 0),
        1280: (1, 0),
        1536: (2, 1),
        1792: (1, 0),
        2048: (2, 1),
        2304: (1, 0),
        2560: (2, 0),
        2816: (1, 0),
        3072: (4, 1),
        3328: (1, 0),
        3584: (2, 0),
        3840: (1, 0),
        4096: (4, 1),
        4352: (1, 0),
        4608: (2, 0),
        4864: (1, 0),
        5120: (4, 0),
        5376: (1, 0),
        5632: (2, 0),
        5888: (1, 0),
        6144: (4, 0),
        6400: (1, 0),
        6656: (2, 0),
        6912: (1, 0),
        7168: (4, 0),
        7424: (1, 0),
        7680: (2, 0),
        7936: (1, 0),
        8192: (4, 1),
    },
    "reuse": {
        1024: (1, 0),
        1280: (1, 0),
        1536: (2, 1),
        1792: (1, 0),
        2048: (2, 1),
        2304: (1, 0),
        2560: (2, 0),
        2816: (1, 0),
        3072: (4, 1),
        3328: (1, 0),
        3584: (2, 0),
        3840: (1, 0),
        4096: (4, 1),
        4352: (1, 0),
        4608: (2, 0),
        4864: (1, 0),
        5120: (4, 0),
        5376: (1, 0),
        5632: (2, 0),
        5888: (1, 0),
        6144: (4, 0),
        6400: (1, 0),
        6656: (2, 0),
        6912: (1, 0),
        7168: (4, 0),
        7424: (1, 0),
        7680: (2, 0),
        7936: (1, 0),
        8192: (4, 0),
    },
}

# ---- ROUTE A extension: local m 9216..16384 (T <= 65536) ---------------------------------------------------------------------------------
# The measured P1 sweep stops at 8192. The SHIPPED extended band is 1024-stride (Lead ruling): 8 bucket M values x 3 modes = 24 rows at
# 9216..16384 step 1024. chunks = M/256 is then always a multiple of 4, so n=4 stays legal and the in-GEMM put/GEMM overlap is preserved;
# every m 8193..9215 pads into the 9216 bucket (band floor 8193), m > 16384 stays native.
# The 256-stride universe below (8448..16384 = 32 M values) is ONLY the explicit Evidence A/B scan; it is never part of the candidate.
# Measured by the pinned Gate-3 sweep (commit 63c2429c; 48/48 points = 8 buckets x 3 modes at the exact bucket M + one pad each): every
# bucket keeps n = 4 / half_first = 0 (chunks = M/256 is always a multiple of 4, so n=4 stays legal everywhere), and the fused/native
# ratio is strongly positive in every row, so no n=2 / n=1 / half_first=1 alternative was needed.
P1_MEASURED_M = tuple(range(1024, 8193, 256))
P1_EXTENDED_M = tuple(range(9216, 16385, 1024))  # 8 values: the shipped 24-row band
P1_EXTENDED_AB_M = tuple(
    range(8448, 16385, 256)
)  # 32 values: the explicit 256-stride A/B scan only
P1_GRID = (
    P1_MEASURED_M + P1_EXTENDED_M
)  # 37 values: the rows the shipped / candidate layout emits
P1_SCAN_M = (
    P1_MEASURED_M + P1_EXTENDED_AB_M
)  # 61 values: build_large_rows scans this and the stride filter selects the layout


def placeholder_best(m: int) -> tuple[int, int]:
    """Legal placeholder (n, half_first) for the explicit 256-stride A/B scan (the shipped band is measured, Gate 3)."""
    chunks = m // 256
    n = 4 if chunks % 4 == 0 else (2 if chunks % 2 == 0 else 1)
    return n, 0


for _cells in P1_BEST.values():
    for _m in P1_EXTENDED_AB_M:
        _cells[_m] = placeholder_best(_m)

# ROUTE A: THE measured Gate-3 winners, one line per bucket row, `(mode, local_m) -> (n, half_first)`; this table was the substitution
# point during wiring and now holds the measured values (all n=4 / half_first=0, which is why no row changed). To re-open the band, put
# unmeasured M values back in P1_NOT_MEASURED. The 256-stride A/B scan is filled by placeholder_best() and is never shipped.
R19_PLACEHOLDER_BEST = {
    ("dense", 9216): (4, 0),
    ("dense", 10240): (4, 0),
    ("dense", 11264): (4, 0),
    ("dense", 12288): (4, 0),
    ("dense", 13312): (4, 0),
    ("dense", 14336): (4, 0),
    ("dense", 15360): (4, 0),
    ("dense", 16384): (4, 0),
    ("indexer", 9216): (4, 0),
    ("indexer", 10240): (4, 0),
    ("indexer", 11264): (4, 0),
    ("indexer", 12288): (4, 0),
    ("indexer", 13312): (4, 0),
    ("indexer", 14336): (4, 0),
    ("indexer", 15360): (4, 0),
    ("indexer", 16384): (4, 0),
    ("reuse", 9216): (4, 0),
    ("reuse", 10240): (4, 0),
    ("reuse", 11264): (4, 0),
    ("reuse", 12288): (4, 0),
    ("reuse", 13312): (4, 0),
    ("reuse", 14336): (4, 0),
    ("reuse", 15360): (4, 0),
    ("reuse", 16384): (4, 0),
}
assert (
    len(R19_PLACEHOLDER_BEST) == 24
)  # 8 buckets x 3 modes: the measured Gate-3 winners (the former substitution point)
for (_mode, _m), _best in R19_PLACEHOLDER_BEST.items():
    P1_BEST[_mode][_m] = _best
P1_NOT_MEASURED = ()  # Gate-3 (pinned 63c2429c, 48/48 points) measured the whole shipped band; the generator writes measured notes
P1_NOT_MEASURED_AB = P1_EXTENDED_AB_M  # the 256-stride A/B scan placeholders
