# Round R16: occupancy-neutral forced-TDM r16 (32-column wave-private tiles)

Base: champion `3e3555b3` (kernel sha `1ef645c1…`). No GPU was used; no latency is claimed.

## Change

* `QSA_BF16_R16_TDM_COLS` (new, default `32`, also `64`) only affects the
  **forced** `QSA_BF16_K_LOAD=tdm` r16 kernel. `auto`/`raw` r16 and every r32
  kernel keep 64 columns; their final ISA is **byte-identical to base**.
* r16 TDM c32: workgroup = 16 rows x **32 columns**, grid.x = ceil(columns/32)
  (doubled), per-column K traffic unchanged. Each wave owns a padded
  **32 x 272 B = 8,704 B** tile -> **34,816 B LDS per workgroup** (was 69,632 B).
  Per active request pass a wave issues **4** eight-row gathers (was 8);
  consumer base `(c % 32) * 272 + kgrp * 16`, lo/hi b128 at +0/+32.
* Unchanged discipline: SGPR-only dedup/visibility issue guard, `tensor_wait(0)`
  + workgroup barrier before consumption and a second barrier before overwrite
  (4 passes, outside every per-wave guard), invalid request/page/logical ->
  physical page 0, partial blocks take the raw fallback, unaligned (non-128 B)
  cache -> actual raw r16 c64 specialization.
* Host grid is bound to the launched builder: `launch.cols_per_block`
  (`wmma_grid(rows, columns, rows_per_block, cols_per_block)` validates 32 only
  for r16). Kernel name `..._r16_c32_flydsl_tdm`.
* Shared helpers loop over `len(cols)` / `len(res)` (trace-identical for 4 tiles).

## No-device real ISA (stock FlyDSL 0.3.3.dev886, `--network none`, no `/dev/kfd|dri`)

Results: `/data/blyu/qsa/work/opt-r16-cpu-trace-20261031/results-r16.m3NHui`
(`host_compile_r16_cpu.sh new-full.tar.gz base-full.tar.gz`; base compiled from
`git archive 3e3555b3`).

| kernel | LDS B | VGPR | SGPR | tensor_load | tensorcnt waits | wait(0) | ds_load_b128 | barrier pairs | 272 B pitch | readfirstlane / v_max_i64 / addtid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| r16 raw (auto/raw) | 0 | 105 | 66 | 0 | 0 | 0 | 0 (136 buffer_load_b128) | 0 | - | 0/-/- |
| **r16 c32 TDM (forced, new)** | **34,816** | **97** | 105 | **16** | 16 | 4 | **64** | 8 | yes (0x110) | 0/0/0 |
| r16 c64 TDM (legacy, `=64`) | 69,632 | 193 | 105 | 32 | 32 | 4 | 128 | 8 | yes | 0/0/0 |
| r32 TDM (auto/tdm) | 17,408 | 183 | 79 | 2 | 2 | 1 | 32 | 1 | yes | 0/0/0 |

Descriptor group1 `0x07510000` (bf16, pad_enable, encoded 5/3) present in all TDM kernels.
SHA256 vs base: auto r16 `3634e008…`, auto/tdm r32 `869941f5…`, raw r32 `54128717…`
and legacy tdm64 r16 `0cc338f5…` are all **byte-identical** to the base compile.

## Occupancy rationale (SPG facts; the occupancy bound is an inference)

LDS is allocated in 2 KB blocks out of at most 320 KB per WGP (partition-dependent).
VGPRs are allocated in 16-VGPR blocks out of 64 per SIMD (wave32).

| kernel | LDS-bound WG/WGP (320 KB) | VGPR-bound waves/SIMD | resulting waves/SIMD |
|---|---:|---:|---:|
| raw r16 | unbounded | 9 (105 -> 7 blocks) | 9 |
| c64 TDM | 4 (68 KB) | **4** (193 -> 13 blocks) | **4** |
| c32 TDM | 9 (34 KB) | 9 (97 -> 7 blocks) | **9** |

The legacy kernel was limited by **both** LDS and VGPRs (4 waves/SIMD, 44 % of raw).
c32 matches raw's 9 waves/SIMD on both resources (if LDS gets the full 320 KB). With a
smaller LDS partition (e.g. 128 KB) c32 still allows 3 WG/WGP vs 1 for c64. The hard
per-SIMD wave-slot limit is not given in the SPG.

## CPU gates (all green in the container)

contract, address, Q-LDS, trace16, trace32 (raw), K-TDM **37/37** (28 old + 9 new),
routing **10/10** (7 old + 3 new), forced-tdm c32 + r32 ISA, legacy c64 ISA, routing-ISA
including the base-identity check.
The new K tests cover, for the c32 geometry: every lane/k-step/N-tile byte-exact against
a per-lane global-load reference in all 4 request passes and 5 origins; mixed requests and
separate wave tiles; dedup/visibility/invalid-request issue skipping; invalid physical
pages, invalid requests and logical tails -> page 0; padding canaries and commuting wave
orders; 128 B-aligned gather starts and in-tile, padding-free b128 consumer reads;
64-bank spread; 34,816 B footprint; and every logit stored exactly once for 32- and 64-column
blocks (8 shapes, including invalid requests and tails). These are layout and plan models,
not GPU correctness.

## Expected effect (hypothesis, to be measured)

Forced c64 TDM vs raw: decode_c65536 -16.6 %, decode_c32768 +8 %, draft_c65536 +10 %,
draft_c32768 +16 %. c32 removes the 4-vs-9 waves/SIMD deficit but doubles the per-column-block
overhead: Q fragments (L2-resident), scalar metadata, the 8 barriers per workgroup and the
partial-wave prologue. Expected: decode_c65536 -10 to -20 %, decode_c32768 0 to -10 %,
draft_c65536 0 to -8 %, draft_c32768 -3 to +3 % (the shortest case is the most overhead-sensitive).

## GPU plan

Acquire the cooperative GPU lock. Run a first-touch correctness ladder with `QSA_BF16_K_LOAD=tdm`
(c32, the default) covering full/partial r16, mixed requests, holes, invalid metadata,
visible/column tails, columns 1/31/32/33/65 and a 16 B-but-not-128 B cache. Confirm the dispatch name
`_r16_c32_flydsl_tdm` and 34,816 B LDS. Then run unprofiled A/A and A/B on all four decode/draft
buckets: auto (raw r16) vs `tdm` (c32) vs `tdm` + `QSA_BF16_R16_TDM_COLS=64`, plus held-out reruns.
Change the `auto` default only if c32 wins on every bucket.
