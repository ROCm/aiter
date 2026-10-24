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
P1_GRID = tuple(range(1024, 8193, 256))

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
