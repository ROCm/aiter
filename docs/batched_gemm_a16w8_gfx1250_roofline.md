# batched_gemm_a16w8 on gfx1250: roofline gap analysis

Kernel: `_batched_gemm_a16w8_gfx1250_persistent_kernel` (Gluon), wrapper
`aiter.ops.triton.gemm.batched.batched_gemm_a16w8`. Both live on
`satya/temp_delete` and are not on `main` yet; all experiments ran against
`ee7e304fb`. Scripts, experiment configs and raw results are in
[`examples/batched_gemm_a16w8_gfx1250/`](examples/batched_gemm_a16w8_gfx1250/).

## Shape and roofline

DeepSeek-R1 MLA absorbed BMM (W_UK), as in `test_batched_gemm_a16w8.py`:
B=128 heads, M=1536, N=512, K=128. X is bf16 `(B, M, K)`, W is fp8 e4m3
`(B, N, K)` with a per-tensor scale, Y is bf16.

| Traffic | MB |
|---|---|
| X read | 50.3 |
| W read | 8.4 |
| Y write | 201.3 |
| **Total** | **260.0** |

At the MI450X's 23.35 TB/s that is **11.1 µs**. The output write alone is
8.6 µs. FLOPs are 25.8 G.

The tuned config for this shape is the `M_LEQ_2048` entry of
`BATCHED_GEMM-A16W8-N=512-K=128.json`: BM=256, BN=256, BK=64, 4 warps,
`NUM_BUFFERS=3`, `WG_PER_CU=1`, `STORE_MODE=2` (TDM store). That gives 1536
tiles on 256 workgroups, 6 tiles each, 2 K-steps per tile. The compiled kernel
uses 808 VGPRs and 305 KB of LDS, with occupancy 1 and no spills.

The reported kernel time at full clock is about 31 µs.

## Measurement caveat: the GPUs were throttled

All four GPUs on the measurement machine were voltage-regulator throttled from
driver load onwards: `amdgpu ... WARN: GPU is throttled, expect performance
decrease. VR.` repeats every minute. The evidence is in
`results/machine_state.txt`.

- The shader clock, measured in-kernel with `clock64()` against
  `wall_clock64()`, was 65 MHz on GPU 0. The rated clock is 2400 MHz.
- Normalised by a raw-WMMA reference, the four GPUs ran at 65, 49, 61 and
  56 MHz. GPU 1 is 1.33× slower than GPU 0.
- A plain `torch.zero_()` of 201 MB reached 0.36–0.48 TB/s.
- The kernel itself took 605–790 µs.

Absolute times from this machine are therefore meaningless. The method used
instead:

- **Clock-bound work is measured in cycles.** That covers WMMA, LDS traffic,
  conversions and instruction overhead. Each GPU ran the same experiment set,
  and its times are normalised by that GPU's raw-WMMA reference. Results are
  then projected to 2400 MHz. Ratios agree within ±1% across the four GPUs.
- **Memory latency is not observable here.** Bandwidth per shader cycle is
  similar to full clock: about 7 KB/cycle here against 8–10 at 2400 MHz.
  Memory latency, though, is likely far cheaper in cycles at 60 MHz. Gaps that
  come from exposed latency are therefore inferred, not measured.
- **Timing method.** CUDA-graph replay of 24 launches over rotating cold input
  copies (the tuning harness's method), taking the median of 25 replays.
  Correctness of the variants was checked bit-exact against the unmodified
  kernel.
- **One GPU process per GPU at all times**, enforced by `gpurun.sh`.

## Where the time goes

These are ablations of the kernel, each a constexpr switch in
`batched_gemm_a16w8_variant.py`, projected to 2400 MHz. "×base" is the fraction
of baseline runtime, with the standard deviation across the four GPUs.

| Variant | ×base | µs @ 2400 MHz |
|---|---|---|
| Baseline (tuned config) | 1.00 | 16.2 |
| Compute only: no loads, no stores | 0.64 ± 0.01 | 10.4 |
| Compute only, BK=128 (1 K-step/tile) | 0.54 | 8.8 |
| Raw bf16 WMMA, same instruction count | — | **5.6** |
| No epilogue store | 0.74 ± 0.01 | 12.0 |
| Stage C in LDS, skip global store | 0.84 ± 0.01 | 13.5 |
| Store path only (epilogue + TDM store) | 0.61 | 9.9 |
| Loads only | 0.23 | 3.6 |
| Loop skeleton only | 0.07 | 1.2 |
| Exact epilogue wait (fix 1 below) | 0.97 | 15.7 |
| MLA layout (strided X, `transpose_bm`) | same as dense, all rows | |

How the 31 µs splits:

- **~16.2 µs of on-chip critical path (measured).**
  - The compute stream is 10.4 µs. Of that, 5.6 µs is raw WMMA, so the
    stream runs at **54% of the measured bf16 WMMA rate**.
  - About 5.8 µs of epilogue and store work is not overlapped with compute.
- **About 15 µs of exposed memory latency and bandwidth stalls (inferred:
  31 − 16.2).** This assumes the 31 µs was measured at full clock.

Host side, measured on CPU and therefore real:

| Host cost per `batched_gemm_a16w8(...)` call | µs |
|---|---|
| Total | 82–86 |
| Rebuilding Gluon layout objects (`create_wmma_layouts`, `create_shared_layouts`, `PaddedSharedLayout`, `BlockedLayout`) | 45 |
| Triton JIT dispatch with prebuilt layouts | 22–26 |
| Direct compiled-kernel launch (for comparison) | 8–12 |
| `get_gemm_config` | 0.5 |

`do_bench` and any eager caller pay this cost. CUDA-graph replay does not.

Raw WMMA throughput (registers only, one workgroup per CU): **fp8 16x16x64
delivers 3.8× the FLOP rate of bf16 16x16x32** (`results/wmma_peak.jsonl`).

## Gaps and candidate fixes

### 1. Pipeline drain at every tile boundary

This is inferred to be the largest gap at full clock.

Each tile's epilogue calls `tdm.async_wait(0)`, which compiles to
`s_wait_tensorcnt 0x0` (`results/isa_summary.txt`). That waits for the
previous tile's TDM store, as intended, and also for the next tile's
already-prefetched loads. The epilogue then runs about 1,700 instructions of
scale, convert and LDS staging with no memory traffic in flight.

Several things leave nothing to hide this latency:

- one workgroup per CU (4 waves);
- one C staging buffer;
- a prefetch depth of one tile.

Candidate fixes:

- **Exact wait.** Wait only for the ops that touch the staging buffer:
  `async_wait(EPI_N)` with `EPI_N = 2*k_tiles` TDM ops issued after the
  previous store, falling back to `async_wait(0)` at the tail. Prototyped
  as `EPI_WAIT=2`. It is bit-exact over repeated runs on both layouts and
  0.97–0.98× here. The latency it removes is mostly invisible at 60 MHz.
- **Overlap the epilogue with the next tile's MMA.** A second C buffer does
  not fit at 256×256: it needs 400 KB against the 320 KB LDS cap. This needs
  ping-pong workgroups or a different LDS split.

### 2. Epilogue not overlapped with compute (~5.8 µs, measured)

Removing the epilogue store saves 26% of runtime. The LDS staging step alone
accounts for 10%, and the TDM store issue for 16%.

### 3. Compute stream at 54% of WMMA peak

The stream takes 10.4 µs against 5.6 µs of raw WMMA, where peak means the
measured register-only WMMA rate. Each K-step carries:

- two barriers;
- the LDS operand reads (`ds_load_b128` and `ds_load_2addr_b64`) and their
  `s_wait_dscnt` chain.

On top of that, the 808-VGPR accumulator forces VGPR bank switching: the
kernel's code contains 395 `s_set_vgpr_msb` instructions.

BK=128 (one K-step per tile) cuts the stream to 8.8 µs. However, 256×256×128
plus the C staging buffer needs 340 KiB of LDS against the 320 KiB cap.
Making it fit means smaller pads or restructured staging.

Storing straight from registers instead (`STORE_MODE=1`) frees the LDS but is
1.5–1.6× slower overall, so that route is out.

### 4. Host overhead (~85 µs per call)

Two zero-risk fixes:

- cache the layout objects per (config, dtypes);
- launch through the compiled-kernel handle.

Together they bring the call to about 10 µs. This matters for eager-mode
callers and for every `do_bench` number of this op.

### 5. Ceiling changers (product decisions)

- **fp8 WMMA** runs at 3.8× the bf16 rate and would take compute off the
  critical path. It needs X in fp8, which this kernel avoids by design for
  accuracy.
- **fp8 output** instead of bf16 cuts traffic from 260 to 160 MB, a roofline
  of about 6.8 µs. It applies only if the downstream attention kernel accepts
  fp8 Q.

### Tried and slower on this machine

All compared with the same GPU's baseline (`results/tiles.jsonl`,
`results/fixes.jsonl`):

| Config | vs baseline |
|---|---|
| `STORE_MODE=1` | 1.63× |
| BM=128, BN=256, `WG_PER_CU=2` | 1.45× |
| BM=128 with double-buffered C | 1.32× |
| 8 warps | 1.29× |
| BM=128 | 1.23× |
| BN=512 | 1.09× |

These options add latency hiding, so they are penalised on a throttled
machine. Re-check them at full clock.

## Tuning on a throttled machine

Comparing candidates against a same-machine baseline only cancels a uniform
slowdown. Two things break that here.

1. **The slowdown is not uniform.** Work bound by clock or bandwidth scales
   with the shader clock, but memory latency does not. The knobs that buy
   latency hiding are therefore under-rewarded: `NUM_BUFFERS`, `WG_PER_CU`,
   `STORE_MODE`, and smaller tiles.
2. **The GPUs are unevenly throttled.** GPU 1 is 1.33× slower than GPU 0. The
   tuning harness deals candidates round-robin and keeps the 10 fastest by
   raw time before re-timing them on one GPU. A candidate that lands on
   GPU 1 has to be about 33% faster just to look equal.

The throttle began at 2026-10-01 20:41 UTC. The `dsr oct 1 tuning` and
`base_tuner run` commits were authored at 22:17 and 22:53 UTC. If those sweeps
ran on this machine, re-tune once the throttle is fixed. Until then:

- tune on a single GPU (`--gpu 0`);
- treat a winner that differs from the runner-up only in latency-hiding knobs
  as provisional.

## Next steps

1. Fix the VR throttle. Then re-run `run_replicated.sh` on four GPUs, which
   takes under 30 minutes, to measure gaps 1 and 2 at full clock.
2. Prototype the low-risk changes: the exact epilogue wait (fix 1) and the
   wrapper caching plus compiled launch (fix 4).
3. With LDS headroom from fix 3, revisit double-buffered C and BK=128 with the
   TDM store.
