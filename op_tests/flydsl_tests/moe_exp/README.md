# FlyDSL MoE experiments on gfx1250 (`flydsl_moe_exp`)

Shared branch for studying the FlyDSL grouped-MoE MXFP4 kernels of Yan Dai's DeepSeek-R1 prefill work
(`dev/yadai_dsr1_conc6144`) on gfx1250 (MI450). It starts at that branch's head
**2489bae2d** ("Optimize gfx1250 MoE A-preshuffle preparation and prefill scheduling (#6984)", 2026-10-27).
The kernels are unchanged; this folder adds the measurement tools, and later experiments add commits on top.
Everything needed to reproduce a number in this README is in the branch.

## The benchmark

`run_dsr1.sh` (repository root): one DeepSeek-R1 MoE layer at prefill, timed kernel by kernel.

| | |
|---|---|
| Shape | 64 experts, 1536 tokens, top-8 routing, model_dim 7168, inter_dim 2048 |
| Routing | balanced (`AITER_MOE_EXPERT_BALANCE=true`): 1536 x 8 = 12,288 routed rows, 192 per expert |
| Data | MXFP4 activations and weights (E8M0 scale per 32 elements), SiLU, no bias |
| gemm1 | per expert 192 x 4096 x 7168 (gate + up), epilogue SiLU + MXFP4 quant of the 192 x 2048 result |
| gemm2 | per expert 192 x 7168 x 2048, bf16 output per route (gather-reduce sums each token's 8 routes) |
| Config | token bucket 2048 of `aiter/configs/tuned_grouped_fmoe.csv`: tile 256x256x256, 2x2 waves, 4 LDS stages, 1x4 cluster |

How the work maps onto the GPU (one workgroup per WGP; the LDS stages leave no room for a second):

- gemm1 launches a worst-case grid of 112 M tiles x 16 N tiles. M tile *e* is expert *e*'s 192 rows; the other
  48 M tiles have no rows and exit at once. So 1024 workgroups do work, about 4 rounds over the 256 WGPs.
- Each workgroup computes one 256 x 256 output tile of one expert. The 16 workgroups of an expert read the same A
  (192 rows x 7168; the 1x4 cluster multicasts it, the four clusters share it through L2). Each reads its own 256
  columns of the expert's W1 (896 KB), so **no weight byte is read twice**.
- gemm2: 64 experts x 28 N tiles = 1792 working workgroups (about 7 rounds), 8 K tiles each.

Kernels (names from the FlyDSL dump):
`a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg1x2_dsfirst16_regpart_planarlds_interleavelds_rolleddrain`
(gemm1) and the same with `_K2048_e64_cluster1x4_` (gemm2).

## Setup

What the results below used (2026-10-28, node heliosr-1b114-d01-3, one MI450X):

| Component | Version |
|---|---|
| Python | 3.12 |
| ROCm, PyTorch | `rocm[libraries,devel,device-gfx1250]==10.1.0a20260820`, `torch[device-gfx1250]==2.11.0+rocm10.1.0a20260820` from https://rocm.nightlies.amd.com/whl-multi-arch/ |
| FlyDSL | `flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl` in the repository root (what `run_dsr1.sh` installs; `requirements.txt` pins 0.3.4.1, not used here) |
| AITER | this branch, installed without composable_kernel |
| Triton | AITER imports it; we used a source build of triton-lang/triton at 93035c44 (`3.9.0`) |
| Other Python packages | numpy, pandas, pyyaml, einops, psutil, ninja (AITER's JIT) |
| ROCm on the host | `/opt/rocm` (hipcc for AITER's JIT host modules; FlyDSL's `ld.lld`) |

Install AITER and FlyDSL into a separate directory (`--target`), so the venv keeps torch and Triton untouched:

```bash
cd <this repository>
SITE=$PWD/../moe_exp_site
pip install --no-deps --target $SITE ./flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl
AITER_USE_SYSTEM_TRITON=1 ENABLE_CK=0 PREBUILD_KERNELS=0 GPU_ARCHS=gfx1250 PYTHONPATH=$SITE \
    pip install --no-deps --no-build-isolation --target $SITE .
```

The first run JIT-compiles AITER's host modules (`module_aiter_core`, `module_moe_asm`) into `$SITE/aiter/jit/`
(a few minutes, once) and the FlyDSL kernels into the run's cache directory. `ENABLE_CK=0` must also be set at
run time (bench.sh does). FlyDSL 0.3.3.dev882 finds `ld.lld` only through `ROCM_PATH` (bench.sh defaults it to
`/opt/rocm`); without it compilation fails with "lld invocation failed".

## Running

From the repository root:

```bash
SITE=$PWD/../moe_exp_site op_tests/flydsl_tests/moe_exp/bench.sh            # Yan's benchmark, 3 timed rounds
# on a shared node: lock every GPU process and wait for an idle GPU before each
LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5
# interleaved A/B: variant = name:ENV=VAL,ENV=VAL on top of run_dsr1.sh's environment
SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 3 base bth0:AITER_TDM_B_TH=0 c44:AITER_FLYDSL_MXFP4_CLUSTER_M=4
```

Each (variant, round) is one process running `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` with
`run_dsr1.sh`'s environment and arguments, through `bench_driver.py`. Round 0 is a warm-up/compile round and is
not counted. Results go to `moe_exp_runs/<UTC time>/` (or `OUT`): `meta.txt` (node, boot time, versions, MEC/PM
firmware, clock), one log per process, `out/*.pt` (outputs), `dump-<variant>/` (FlyDSL IR, named after the
kernels), and `summary.txt` from `summarize.py`.

## What is measured

- **profiler_us**: the test's own kernel-bench. AITER's `run_perftest(testGraph=False)` launches the gemm1 (then
  gemm2) launch callable 5 times as warm-up, then 128 times back to back with the same inputs, and averages the
  kernel durations from the torch profiler (kernel trace; the first iteration and outliers dropped).
- **graph_us**: `bench_driver.py` captures the same callable 128 times into one CUDA graph, replays it 10 times,
  and times each replay with CUDA events: the median time per launch. No profiler; it includes the gaps between
  back-to-back kernels. Kernel tracing has inflated memory-bound kernel times on some MEC firmware builds, so
  the two timers check each other.
- **Correctness**: the test compares the fused-MoE output with its torch reference (`logits_diff`, gate 0.01)
  and raises otherwise; `summarize.py` also compares every saved output bit for bit with the first variant's
  round-1 output.
- **TB/s**: bytes / time, each byte counted once (`summarize.py`): gemm1 = A + W1 + MXFP4 output, each with its
  E8M0 scales = 1,058.4 MB; gemm2 = A2 + W2 (+ scales) + bf16 output = 688.7 MB. The test's own "Bandwidth"
  counts gemm1's output as bf16 (1,095.3 MB), the grouped-MoE tuner's convention.

Rules we follow for numbers worth comparing: one shape per process; interleave the variants being compared;
median of 3 or more timed rounds; compare only numbers from the same boot; keep `meta.txt` with the results.
Clocks and firmware matter: a node whose perf settings were not re-applied after a reboot can run 5-20% slow.

## Results

### 2026-10-28, heliosr-1b114-d01-3 (MI450X), Yan's benchmark at 2489bae2d

`results/20261028_d01-3_base/` (meta.txt, summary.txt, run_log.txt, one full process log). Command:
`LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5`, 5 timed rounds, one process each.

| Kernel | profiler_us (median, range) | graph_us (median, range) | TB/s (profiler) | TB/s (graph) |
|---|---|---|---|---|
| gemm1 | 83.76 (83.73-84.30) | 80.85 (80.35-81.05) | 12.64 | 13.09 |
| gemm2 | 58.87 (58.40-59.30) | 53.73 (53.62-54.05) | 11.70 | 12.82 |

- TB/s uses the byte model above (gemm1 1,058.4 MB, gemm2 688.7 MB). With the test's convention (gemm1 output
  as bf16, 1,095.4 MB) gemm1 is 13.08 TB/s at 83.76 us; gemm2 is the same in both.
- Correct in every round (`logits_diff=3.4249e-06`, `rel_l2=2.6172e-03`); all 6 outputs bit-identical.
- A run 5 minutes earlier gave 83.43 / 80.70 / 59.33 / 53.68 us (same order as the table).
- graph_us is below profiler_us, by 3.5% for gemm1 and 8.7% for gemm2. Most likely the back-to-back launches
  overlap: the next launch's workgroups start while the previous one's last round drains, and the profiler
  counts that overlap in both kernels' durations. Neither number is the time of an isolated launch.
- Node state: boot 2026-10-28 14:09 UTC, MEC firmware 2560 (not the stock 2430; the node's MEC build changes
  often), PM 02.125.15.00, MAX_CLK 2400 MHz, the node's perf settings not re-applied after that boot. Compare
  other numbers with these only after re-running the base on the same node and boot.

Earlier the same day, same boot and code (previous harness, profiler timer, 3 interleaved rounds), relative to
the base: `AITER_TDM_B_TH=0` (temporal B loads) +10.3% gemm1 / +5.0% gemm2; `AITER_FLYDSL_MXFP4_CLUSTER_M=4`
(4x4 cluster for both GEMMs) +6.3% / +6.8%. Re-run with
`bench.sh 3 base bth0:AITER_TDM_B_TH=0 c44:AITER_FLYDSL_MXFP4_CLUSTER_M=4`.

## Notes

- In this branch `AITER_TDM_B_TH` defaults to 1 (non-temporal B loads) and always overrides the config's
  `tdm_b_th` column (`aiter/ops/flydsl/grouped_gemm_mxfp4.py`, `_select_tdm_b_th`).
- `waves_per_tensor_tdm=4` gives wrong results (about 60% NaN) in this branch: with 2 M waves the A-scale is split
  along K. A fix is on `dev/lixzhang/yadai_dsr1_tdm_scale_waves` (`AITER_FLYDSL_TDM_SCALE_WAVES=2`).
- B (weights) is never shared at this shape, and a 4x4 cluster puts four different experts in a column. The
  kernel multicasts B only between M tiles of the same expert, which needs more than 256 rows per expert
  (more than 2048 tokens at 64 experts, top-8).
