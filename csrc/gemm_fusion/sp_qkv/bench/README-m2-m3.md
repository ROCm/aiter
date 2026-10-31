# R13 microbenches M2 (CU push) and M3 (BM32/BM64/BM128 split-K GEMM): run note for Evidence

Status: **SOURCE ONLY**. Written without a compiler and without a GPU; nothing here was compiled, syntax-checked (bash and hipcc were unavailable in the authoring sandbox) or run.
Design reference: `records/architecture/stage3/R13-card-a-mp64-r2.md` (gates M2, M3 in D3) and R12-design-card-r2.md section R2.10 (E0).
Files: `m2_cu_push.cu`, `m3_gemm_bm.cu`, `build_bench.sh`, `run_bench.sh`. Single-process, plain hipcc binaries (no MORI, no CCO window, no kernel arguments from the RAN driver).

## Build
```
BUILD_DIR=/fresh/dir bash build_bench.sh all        # needs hipcc (gfx950), opus headers at $REPO_ROOT/csrc/include (override OPUS_INC)
```
The script also emits `*.gfx950.s`; gate: `private_segment_fixed_size` must be 0 for every `gemm_bm<...>` instantiation (it fails the build otherwise), and `s_waitcnt` counts in the producer loop should show the counted `vmcnt(MB*newer)` pattern.
Expected first-compile risks (the author could not check): opus `gmem::load/store` with `number<17>` on the host-visible template, `wall_clock64()` in device code, `__builtin_nontemporal_store` on `vector_t<u32,4>`, `static_ford` + captured `t` loop variable in the epilogue lambda (if this does not compile, replace by a manual `[&, t]` capture or hoist the lambda out of the `t` loop).

## M3: BM32 / BM64 / BM128 split-K GEMM and the P byte-compare (`m3_gemm_bm`)
What it is: the r18 GEMM body (`small/r18/device/sk_gemm_fp8.cu`) parameterised by AQMAX (2/4/8 A instructions per producer wave = 32/64/128 rows), PFK (ring depth) and SK; epoch/MORI/SM_STAMP code replaced by plain per-WG `wall_clock64` stamps. The consumer wave keeps the r18 tiled MMA (TM2, EM1, EN4, one 16x16x128 MFMA per 32-row tile and K block); BM64 holds 2 accumulator tiles per consumer wave, BM128 holds 4. Hence every output element is produced by exactly the r18 MFMA sequence and P must be byte-identical.
Reference for the byte compare: the BM32/PFK6 instantiation (same math as r18, `live_wave` generalised to `2*mt + cw < AQ`), launched once per 32-row M tile (A pointer offset 32*t rows, M_t = min(32, M - 32t)). Compared: live rows only, `P64[s][t*32 + r][n]` vs `ref_t[s][r][n]`, memcmp of the FP32 bytes. A and W are random fp8 bytes with NaN encodings masked out; W is NOT in a meaningful layout (both kernels read the same bytes in the same physical order, so equality is what is tested, not a numerical GEMM result).
Variants: `bm32_pfk6` (72 KiB LDS, M<=32), `bm64_pfk4` (64 KiB, M<=64; the card's choice), `bm64_pfk5` (80 KiB, exactly the 2 WG/CU limit; A/B only), `bm128_pfk3` (72 KiB, M<=128). Grid = (N/64, SK), 256 threads, as r18.
Output (TSV): `event_us_med/min` (hipEvent around one launch, includes launch overhead), `stamp_span_us_med` (max WG end - min WG start by `wall_clock64`, 100 MHz; this is the number to compare with the card's "GEMM end <= 12.5 us at m = 64"), `P_byte_equal_vs_r18_ref` (EQUAL / DIFFERENT, `first_diff` = (s*1000+m)*100000+n of the first differing float).
Gates (card D3 item 2): at M=64, N in {9216, 9856}, SK=3: `bm64_pfk4` stamp span <= 12.5 us AND EQUAL. Falsifier: span > 12.5 us => A-from-L2 (57 MB) is binding; try `bm64_pfk5` or `--sk 2`. Also report cold-L2 (`--cold 1`) because the A re-read from L2 is the suspected limiter. M in {33, 48, 63} exercise the dead-row mask (AQ = 3 leaves tile (mt=1, cw=1) idle).
Expected failure modes to read first if the compare says DIFFERENT: (1) fragment row mapping for tile t>0 (`a_row = t*32 + cw*16 + l16`); (2) `wait_newer` count with MB = 4 + AQ (AQ 4 -> MB 8: vmcnt immediates up to 32, below the 63 limit; AQ 8 -> MB 12, newer <= 1 at PFK 3); (3) epilogue row `m = t*32 + u_cm(y)`.

## M2: CU push microbench (`m2_cu_push`)
What it is: GPU 0 pushes to GPUs 1..D (peer access in one process). Same idioms as r18: 16 B per lane SC0|SC1 stores via an opus gmem buffer resource (push16), per-wave `s_waitcnt vmcnt(0)` + barrier, then thread 0 does `fence(release, system)` + `vmcnt(0)` + relaxed system fetch_add (`sys_add_release`) on the destination's flag word, `--rel` times. WG w pushes to destination (w*D)/G (contiguous blocks like the r18 role table); XCC of a WG = wgid%8. The flag chain, polls and compute are not modelled.
Sweep parameters (cartesian product of comma lists): `--bytes` per WG, `--wgs`, `--threads` (256/512), `--rel` (0..3 releases per WG), `--ndst` (1..3), `--mode` (0 SC0|SC1 write-through as r18, 1 plain, 2 non-temporal), `--rounds`.
Output (TSV): `event_us_med` (whole launch), `retire_us_med` (max t1 - min t0: all stores acknowledged + barrier, from WG entry stamps), `release_us_med` (max t2 - max t1: the release tail), `gbps_event`, `flag_ok` (destination flag == WGs-per-destination * releases), `data_ok` (first/last vector of WG 0's region hold the expected pattern).
`run_bench.sh m2` runs the card's questions: (a) bytes x WGs; (b) 24x512 vs 48x256 at equal total bytes; (c) releases per WG (0..3); (d) fixed total bytes with growing WG count, 1 destination vs 3 (falsifier for the link-bound model: retire flat in bytes or falling with WG count => concurrency-bound, not link-bound); (e) store flavour.
Reference sizes for the real push at m = 64 (dense, 18 heads x 128 cols x 64 rows x 2 B = 295,936 B per destination, 3 destinations): ~37 KiB per WG at 24 WGs, ~18.5 KiB per WG at 48 WGs. The card's push-retire forecast is ~5.4 us (link-bound model).
Limitations: plain `hipMalloc` peer memory, not MORI CCO windows (same XGMI store path expected; confirm once with one `--mode 0` row against the r18 stamps), a single sender (the real kernel has all 4 ranks pushing at once: all-to-all link contention is NOT modelled; run 4 concurrent processes with different sender GPUs if that matters), no gap/flag-poll/compute.

## Unverified
Everything: compile, ISA gates, P equality, timings, and the model that MORI window pushes behave like plain peer stores.