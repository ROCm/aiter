# gfx1250 gluon GEMM-A16W16 tuning for the DeepSeek-R1 QKV_A projection

The DeepSeek-R1 fused QKV_A projection (bf16, N=2112, K=7168) now has a tuned config for
every M bucket on gfx1250. It is faster at every M bucket except M=256: 1.44× at M=1024,
1.37× at M=4096, 1.27× at M=16384 and 1.22× at M=1. The rest range from 1.02× to 1.12×.
M=256 shows no gain. The change is one file:
`aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16/GEMM-A16W16-N=2112-K=7168.json`.

## What runs

- **The GEMM.** The QKV_A projection maps hidden 7168 to q_lora_rank 1536 + kv_lora_rank 512 +
  qk_rope_head_dim 64 = 2112 outputs.
- **The dispatch path.** gfx1250 has no tuned bf16 CSV entry for this shape. So
  `aiter.tuned_gemm` falls back to libtype `triton`, which calls `gemm_a16w16(x, w, bias=None)`.
  On gfx1250 that wrapper uses the gluon backend. The kernel is
  `_gemm_a16w16_gfx1250_{bandwidth,compute}_bound_kernel`, and it reads the config file above.
- **The previous file.** It was tuned for Kimi-K3 (#5875) and had buckets for M ≤ 16 … 512. Every
  M above 512 fell back to its `any` tile: 128×128×128, 2 buffers, 2 warps, bandwidth_bound. That
  is the 128×128×128 tile that shows up in DSR1 traces.

## Before vs after

Each row compares the config the loader picks before and after, for that M.

| M | before (bucket: tile) | after (bucket: tile) | before µs | after µs | speedup |
|---:|---|---|---:|---:|---:|
| 1 | M_LEQ_16: 16×16×256, 8 buf, 1 w, bandwidth, wpe 1 | M_LEQ_1: 16×8×512, 6 buf, 1 w, compute | 6.60 | 5.39 | 1.22× |
| 4 | M_LEQ_16: 16×16×256, 8 buf, 1 w, bandwidth, wpe 1 | M_LEQ_4: 16×8×512, 6 buf, 1 w, compute | 6.78 | 6.14 | 1.10× |
| 8 | M_LEQ_16: 16×16×256, 8 buf, 1 w, bandwidth, wpe 1 | M_LEQ_8: 16×8×512, 6 buf, 2 w, compute | 7.07 | 6.79 | 1.04× |
| 16 | M_LEQ_16: 16×16×256, 8 buf, 1 w, bandwidth, wpe 1 | M_LEQ_16: 16×16×512, 6 buf, 4 w, compute | 8.30 | 7.92 | 1.05× |
| 32 | M_LEQ_32: 16×32×512, 5 buf, 2 w, bandwidth | M_LEQ_32: 16×16×256, 6 buf, 2 w, bandwidth | 11.04 | 10.13 | 1.09× |
| 64 | M_LEQ_64: 32×16×256, 4 buf, 2 w, bandwidth, wpe 1 | M_LEQ_64: 32×32×256, 6 buf, 4 w, compute | 12.74 | 12.19 | 1.04× (1.04–1.05) |
| 128 | M_LEQ_128: 64×32×512, 3 buf, 4 w, bandwidth | M_LEQ_128: 32×32×256, 3 buf, 4 w, bandwidth | 16.78 | 16.33 | 1.03× (1.02–1.03) |
| 256 | M_LEQ_256: 64×64×256, 4 buf, 4 w, bandwidth, wpe 1 | M_LEQ_256: 64×32×256, 3 buf, 8 w, bandwidth | 21.20 | 21.69 | 0.98× (0.97–1.01) |
| 512 | M_LEQ_512: 64×64×128, 4 buf, 2 w, bandwidth | M_LEQ_512: 64×32×256, 2 buf, 2 w, bandwidth | 28.05 | 25.10 | 1.12× |
| 1024 | any: 128×128×128, 2 buf, 2 w, bandwidth | M_LEQ_1024: 128×128×128, 4 buf, 4 w, compute | 47.44 | 32.97 | 1.44× |
| 2048 | any: 128×128×128, 2 buf, 2 w, bandwidth | M_LEQ_2048: 128×256×128, 3 buf, 4 w, compute | 53.51 | 52.70 | 1.02× (1.01–1.02) |
| 4096 | any: 128×128×128, 2 buf, 2 w, bandwidth | M_LEQ_4096: 256×256×64, 4 buf, 4 w, compute | 111.55 | 81.21 | 1.37× |
| 8192 | any: 128×128×128, 2 buf, 2 w, bandwidth | M_LEQ_8192: 256×256×64, 4 buf, 4 w, compute | 182.30 | 165.20 | 1.10× |
| 16384 | any: 128×128×128, 2 buf, 2 w, bandwidth | any: 256×256×64, 4 buf, 4 w, compute | 378.16 | 297.97 | 1.27× |

How the numbers were measured:

- **Same process, same GPU.** For each M, the before and after configs are timed in one process
  on one GPU, alternating in ABBA order. That way clock and thermal drift hit both configs.
  Most rows run 6 rounds. Rows 64, 128, 256 and 2048 also run 20 rounds on two more GPUs, and
  for those the range across runs is shown in parentheses.
- **Timing method.** Each timing uses the tuning harness's method: CUDA-graph replay of 24
  launches over cold input copies, taking the median of 25 replays.
- **The call.** Inputs are bf16, and the call is DSR1's exactly: `bias=None` on the default gluon
  backend. The before configs keep their `waves_per_eu`.
- **Aggregation.** Each value is the median over GPUs 0–3. The raw data is in `results/ab_*.json`
  under `docs/examples/gemm_a16w16_gfx1250_dsr1_qkv_a/`.
- **The two close cases.** M=256 shows no gain: the after tile is 2% slower in median, which is
  inside the run-to-run spread. M=2048 gains a small but consistent 1–2%.

## How it was tuned

- **Harness and command.** Tuned with the gluon tuning harness in
  `aiter/ops/triton/utils/_triton/tuning/`. This is the refactored harness, which is not on main
  yet. The command was:
  `sweep_configs.py gemm_a16w16 --M 1024 2048 4096 8192 1 4 8 16 32 64 128 256 512 --N 2112 --K 7168 --gpu 0 1 2 3`.
  It sweeps every standard M bound and runs one serial worker per GPU.
- **Search space.** The swept keys come from the gluon `DEFAULT.json`: BLOCK_M/N/K, NUM_BUFFERS,
  num_warps and kernel_type, with `persistent` pinned.
- **Size and run time.** 1,451 configs per M at M ≤ 16, rising to 6,501 at M ≥ 4096. That is
  55,638 configs in total, which took 4.5 h on 4 gfx1250 GPUs.
- **Register-pressure prune.** For this sweep the harness's `gemm_a16w16` `should_skip` rejects
  tiles with more than 1024 fp32 accumulators per lane (wave32). That removed 2,630 configs.
  - Such tiles spill. They compile for minutes until the `--stall` watchdog kills them, and each
    would cost a full stall timeout at every M.
  - The change is in `harness_register_pressure_prune.patch`. It applies to the refactored
    harness, not to main.
- **Install and `any`.** `write_best_configs.py` wrote the file, and `any` was then set to the
  M_LEQ_8192 tile.
  - The installer writes `DEFAULT.json`'s `any` instead. That is 32×64×64 bandwidth_bound, which
    is far slower above M=8192 than the old 128×128×128.
  - With the 256×256×64 tile, M=16384 runs 1.27× faster than before.
- **Validity on main.** Every tuned tile has BLOCK_K ≤ 512.
  - A fix that is not on main yet, "Fix bug in pad interval", caps the TDM pad interval at 256
    dwords (512 bf16).
  - At BLOCK_K ≤ 512 that cap has no effect, so these configs build the same kernels with or
    without the fix. The file is therefore valid on main.
  - The measurements were taken with the fix present.

## Validation

- **Numerics.** For every bucket, plus M=16384 for `any`, the installed config's output matches
  `F.linear` bit for bit (`bias=None`). Both differ from an fp32 reference only by bf16 output
  rounding. See `results/numerics.txt`.
- **Out-of-bounds writes.** `y` was surrounded by guard bands holding a sentinel. Both the before
  and the after configs ran at 39 M values, including off-bucket sizes such as 100, 1025, 2047,
  2049 and 8193.
  - Nothing outside `y` was written, and `y` was exact every time: 78 of 78 checks passed.
  - See `results/guard_test.txt`.

## Findings and caveats

1. **Machine state.** All four GPUs ran at 2400 MHz with FCLK 1900 MHz. There were no
   voltage-regulator throttle messages after the 2026-10-03 boot, unlike during the
   batched_gemm_a16w8 roofline analysis. The four GPUs agreed within 1–2%. See
   `results/machine_state.txt`.
2. **Unreproducible early timings.** Before the sweep, single-config timings of the old configs
   were up to 48% faster at M=16–512 than every later measurement. For example, M=32 measured
   7.6 µs then and 11.0 µs afterwards.
   - The later numbers agree across all four GPUs and across measurement methods, including the
     harness's own baseline. The early ones could not be reproduced.
   - They are left out. All numbers above are interleaved, so drift affects both configs.
3. **GPU page faults in the sweep.** Seven candidates faulted the GPU during the sweep: no-retry
   page faults from vector memory writes.
   - All seven were bandwidth_bound with 1–2 warps, mostly with large tiles. They are listed in
     `results/machine_state.txt`.
   - They point to a bug in the bandwidth_bound kernel for those tiles. None of them is a winner,
     and the harness recorded them as `crashed`.
4. **One unexplained page fault.** One more page fault happened on GPU 2 during an interleaved
   timing run at M=2048, which compares the old 128×128×128 bandwidth tile with the new 128×256×128
   compute tile. It did not reproduce in any of these:
   - four identical reruns;
   - 4 × 50 graph timings per config (`results/stress_M2048.txt`);
   - the guard tests.
5. **Hung compiles.** These candidates took more than 300 s to compile: bandwidth_bound with
   NUM_BUFFERS=1, 1–2 warps and BLOCK_K 256–512.
6. **Errors.** The sweep recorded 3,368 shared-memory out-of-resource errors. The harness's LDS
   estimate counts only the A and B tile buffers. It also recorded 630 compiler failures
   (`PassManager::run failed`).
7. **`waves_per_eu`.** The gluon wrapper reads `waves_per_eu`, but the key is not in the gluon
   `DEFAULT.json`.
   - So the harness neither sweeps it nor writes it.
   - It times the installed baseline with the key removed.
   - The old M_LEQ_16/64/256 buckets had `waves_per_eu: 1`. The before/after table above keeps
     it for the before configs.
8. **Bias.** The harness times `gemm_a16w16` with a bias vector (left uninitialized). DSR1's QKV_A
   has no bias, so every number above uses `bias=None`.

## Reproduce

See `docs/examples/gemm_a16w16_gfx1250_dsr1_qkv_a/README.md`.
