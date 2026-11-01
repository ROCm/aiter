# FlyDSL MoE experiments on gfx1250 (`flydsl_moe_exp`)

Shared branch for studying the FlyDSL grouped-MoE MXFP4 kernels of Yan Dai's DeepSeek-R1 prefill work on gfx1250
(MI450). Since 2026-11-01 it starts at **8ea2265dc**, the head of `dev/yadai_prefill_clean` ("update script and
wheel", 2026-10-29). That is Yan's current line: DSR1 conc6144 gemm1 91 -> 78 us "with boost on", from MMA-group
and DS-read scheduling (`s_wait_alu` removed from the main loop) plus pipeline and prologue tuning.
The kernels are unchanged; this folder adds the measurement tools, and later experiments add commits on top.
Everything needed to reproduce a number in this README is in the branch.

History: until 2026-10-31 this branch sat on `dev/yadai_dsr1_conc6144` @ 2489bae2d (#6984, 2026-10-27) with
`run_dsr1.sh`'s environment. That state is kept as `dev/lixzhang/flydsl_moe_exp_2489bae2d`; the 2026-10-28 results
below were measured there.

## The benchmark

`run_a4w4_conc6144.sh` (repository root): one DeepSeek-R1 MoE layer at prefill, timed kernel by kernel.

| | |
|---|---|
| Shape | 64 experts, 1536 tokens, top-8 routing, model_dim 7168, inter_dim 2048 |
| Routing | balanced (`AITER_MOE_EXPERT_BALANCE=true`): 1536 x 8 = 12,288 routed rows, 192 per expert |
| Data | MXFP4 activations and weights (E8M0 scale per 32 elements), SiLU, no bias |
| quant_a1 | MXFP4 quant of the 1536 x 7168 activations and their scatter into the per-expert, preshuffled A of gemm1 (two kernels) |
| gemm1 | per expert 192 x 4096 x 7168 (gate + up), epilogue SiLU + MXFP4 quant of the 192 x 2048 result |
| gemm2 | per expert 192 x 7168 x 2048, bf16 output per route (gather-reduce sums each token's 8 routes) |
| Config | token bucket 2048 of `aiter/configs/tuned_grouped_fmoe.csv`: tile 256x256x256, 2x2 waves, 4 LDS stages, 1x4 cluster; A layout and B load hint from the CSV |
| Schedule (env) | `AITER_FLYDSL_MMA_FIRST_GROUP=4 AITER_FLYDSL_MMA_GROUP=4 AITER_FLYDSL_DS_FIRST_N=8` (`mg4_dsfirst8` in the kernel names; 2489bae2d ran `mg1x2_dsfirst16`) |

How the work maps onto the GPU (one workgroup per WGP; the LDS stages leave no room for a second):

- gemm1 launches a worst-case grid of 112 M tiles x 16 N tiles. M tile *e* is expert *e*'s 192 rows; the other
  48 M tiles have no rows and exit at once. So 1024 workgroups do work, about 4 rounds over the 256 WGPs.
- Each workgroup computes one 256 x 256 output tile of one expert. The 16 workgroups of an expert read the same A
  (192 rows x 7168; the 1x4 cluster multicasts it, the four clusters share it through L2). Each reads its own 256
  columns of the expert's W1 (896 KB), so **no weight byte is read twice**.
- gemm2: 64 experts x 28 N tiles = 1792 working workgroups (about 7 rounds), 8 K tiles each.

Kernels (names from the FlyDSL dump):
`a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg4_dsfirst8_regpart_planarlds_interleavelds_rolleddrain`
(gemm1) and the same with `_K2048_e64_cluster1x4_` (gemm2); quant_a1 = `moe_apre_quant_fd7168_r8_fp4_pk8_w16` +
`moe_apre_scatter_fd7168_r8_fp4_pk8_w16_tk7168_grid_sw8_async_ballot`.

## Setup

What the results below used (node heliosr-1b114-d01-3, one MI450X):

| Component | Version |
|---|---|
| Python | 3.12 |
| ROCm, PyTorch | `rocm[libraries,devel,device-gfx1250]==10.1.0a20260820`, `torch[device-gfx1250]==2.11.0+rocm10.1.0a20260820` from https://rocm.nightlies.amd.com/whl-multi-arch/ |
| FlyDSL | `flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl` in the repository root (what `run_a4w4_conc6144.sh` requires; `requirements.txt` pins 0.4.1, not used here) |
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
`/opt/rocm`); without it compilation fails with "lld invocation failed". Never import `aiter` with the
repository root on `sys.path` (e.g. `python -c "import aiter"` from the root): the source tree then shadows the
site and JIT-builds into it.

## Running

From the repository root:

```bash
SITE=$PWD/../moe_exp_site op_tests/flydsl_tests/moe_exp/bench.sh            # Yan's benchmark, 3 timed rounds
# on a shared node: lock every GPU process; inside the lock check the GPU is idle before each process
LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5
# interleaved A/B: variant = name:ENV=VAL,ENV=VAL on top of run_a4w4_conc6144.sh's environment
SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 3 base bth0:AITER_TDM_B_TH=0 c44:AITER_FLYDSL_MXFP4_CLUSTER_M=4
```

Each (variant, round) is one process running `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` with
`run_a4w4_conc6144.sh`'s environment and arguments, through `bench_driver.py`. Round 0 is a warm-up/compile round
and is not counted. Results go to `moe_exp_runs/<UTC time>/` (or `OUT`): `meta.txt` (node, boot time, versions,
MEC/PM firmware, clock, power cap), one log per process, `out/*.pt` (outputs), `dump-<variant>/` (FlyDSL IR, named
after the kernels), and `summary.txt` from `summarize.py`. With `IDLE_CHECK=1` a process runs only if, inside the
lock, `rocm-smi` shows 0% GPU use and less than `IDLE_VRAM_MB` (1024) MB of VRAM in use; otherwise the series
stops. The VRAM check catches a loaded but idle model server, which shows 0% use.

## What is measured

The test's kernel-bench times three launch callables, in this order: `quant_a1`, `gemm1`, `gemm2`. Each callable
is the Python launcher bound to one eager run's tensors, so calling it re-launches that stage on the same buffers.

- **profiler_us**: the test's own number. For each callable AITER's `run_perftest(testGraph=False)` makes 5 warm-up
  calls, then 128 calls with no synchronization in between inside the torch profiler, and averages the kernels'
  device durations from the trace (first iteration and outliers outside 1.5 IQR dropped; a two-kernel callable
  like `quant_a1` reports the sum). The callables pass no tensor arguments, so `run_perftest`'s input rotation
  does nothing: every launch reads the same inputs and writes the same outputs. The test prints gemm1 and gemm2;
  `bench_driver.py` prints all three (`[moe-exp] <name>: profiler_us`), from the same values. If it prints
  `n/a` (the test: `us = nan`), the profiler trace had no kernel events; `profiler_probe.py` checks whether the
  node's torch profiler records kernels at all.
- **graph_us**: after the test has timed all three callables, `bench_driver.py` captures each callable 128 times
  into one CUDA graph, replays it 10 times, and times each replay with CUDA events: the median time per launch.
  No profiler; launches run back to back and the number includes the ~1.5 us gaps between kernels. The GPU idles
  1 s before each callable's graph timing (`MOE_EXP_GRAPH_GAP_S`), so one kernel's back-to-back run does not heat
  the GPU for the next kernel.
- **Correctness**: the test compares the fused-MoE output with its torch reference (`logits_diff`, gate 0.01)
  and raises otherwise; `summarize.py` also compares every saved output bit for bit with the first variant's
  round-1 output.
- **TB/s**: bytes / time, each byte counted once (`summarize.py`): gemm1 = A + W1 + MXFP4 output, each with its
  E8M0 scales = 1,058.4 MB; gemm2 = A2 + W2 (+ scales) + bf16 output = 688.7 MB. The test's own "Bandwidth"
  counts gemm1's output as bf16 (1,095.3 MB), the grouped-MoE tuner's convention.

Why the two timers differ. They time different launch regimes, and only one of them traces:
- The Python launch path takes about 190 us per call (the `AITER_LOG_MORE` event lines: ~272 us per synchronized
  gemm1 launch for an 84 us kernel), longer than either GEMM. So in the profiler loop the host probably cannot keep
  the GPU fed and each launch starts on an idle GPU (inferred, not measured). The graph replays run back to back.
- Back to back, a power-limited GPU lowers its clock. On 2026-10-29 at a 2.5 kW socket limit (this node), 2.5 s
  of back-to-back gemm1 ran at about 1.07 GHz against about 2.2 GHz for spaced launches. A kernel that saves
  cycles but spends more energy per launch can be faster on profiler_us and slower on graph_us; 8ea2265dc's gemm1
  did exactly that against 2489bae2d (-6.5% and +4.1%).
- The profiler adds a timestamp step per dispatch. Its cost depends on the MEC firmware (about 4-5 us per launch
  on build 2430; build 882532 inflated memory-bound kernels 1.5-2.3x).
- Back-to-back launches do not overlap: a probe kernel that stamped every workgroup's start and end (2026-10-25)
  never saw a workgroup start before the previous kernel's last one ended. (This README said the opposite until
  2026-11-01.)

Rules we follow for numbers worth comparing: one shape per process; interleave the variants being compared;
median of 3 or more timed rounds; compare only numbers from the same boot; keep `meta.txt` with the results.
Clocks, power limits and firmware matter: a node whose perf settings were not re-applied after a reboot can run
5-20% slow.

## Results

### 2026-11-01, heliosr-1b114-d01-3 (MI450X), at 8ea2265dc

`results/20261101_d01-3_base/` (meta.txt, summary.txt, run_log.txt, one full process log, profiler_probe.txt).
Command: `LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5`, 5 timed rounds, one process
each, 21:15-21:21 UTC.

| Kernel | graph_us (median, range) | TB/s (graph) | profiler_us |
|---|---|---|---|
| quant_a1 | 20.25 (20.11-20.28) | - | n/a |
| gemm1 | 74.76 (74.69-74.95) | 14.16 | n/a |
| gemm2 | 51.69 (51.63-51.74) | 13.32 | n/a |

- **No profiler_us on this boot.** The torch profiler returned no device events for gemm1 and gemm2 in any round
  (the test prints `us = nan`) and a single, implausible event (2.85 ms) for quant_a1 in 2 of 5 rounds.
  `profiler_probe.py` gets 0 of 256 events for plain torch copies and matmuls on the same boot, while rocprofv3
  `--kernel-trace` records every kernel. So the node's torch-profiler path is broken, not this code; on
  2026-10-29 (MEC 2560) the same code gave profiler numbers.
- Correct in every round (`logits_diff=3.4249e-06`, `rel_l2=2.6172e-03`); all 6 outputs bit-identical.
- Node state: boot 2026-11-01 08:28 UTC, MEC firmware 2430 (`gc_12_1_0_mec.bin` sha256 d798bf71...), PM
  02.125.15.00, MAX_CLK 2400 MHz, 2.5 kW socket power limit, the node's perf settings not re-applied after that boot.
- Against 2026-10-29 (same code; MEC 2560, another boot, the kit's first driver): graph_us gemm1 84.83 -> 74.76
  (-11.9%), gemm2 54.18 -> 51.69 (-4.6%), quant_a1 20.73 -> 20.25. Firmware, boot and the graph timer's order
  changed together, so the cause is not known.

### 2026-10-28, heliosr-1b114-d01-3 (MI450X), at 2489bae2d (`run_dsr1.sh`'s environment)

Measured on the branch's earlier base, now `dev/lixzhang/flydsl_moe_exp_2489bae2d`, with the kit's first version.
That version graph-timed each kernel right after its own profiler loop, so gemm2's profiler loop ran right after
gemm1's back-to-back replays. 2489bae2d's gemm2 is insensitive to that (about 59 us with and without the driver);
8ea2265dc's is not (54.2 us cool, 60.5 us after gemm1's replays, 2026-10-29), hence the change.
`results/20261028_d01-3_base/` (meta.txt, summary.txt, run_log.txt, one full process log). Command:
`LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5`, 5 timed rounds, one process each.

| Kernel | profiler_us (median, range) | graph_us (median, range) | TB/s (profiler) | TB/s (graph) |
|---|---|---|---|---|
| gemm1 | 83.76 (83.73-84.30) | 80.85 (80.35-81.05) | 12.64 | 13.09 |
| gemm2 | 58.87 (58.40-59.30) | 53.73 (53.62-54.05) | 11.70 | 12.82 |

- Correct in every round (`logits_diff=3.4249e-06`, `rel_l2=2.6172e-03`); all 6 outputs bit-identical.
- A run 5 minutes earlier gave 83.43 / 80.70 / 59.33 / 53.68 us (same order as the table).
- Node state: boot 2026-10-28 14:09 UTC, MEC firmware 2560 (not the stock 2430), PM 02.125.15.00, MAX_CLK
  2400 MHz, 2.5 kW socket power limit, the node's perf settings not re-applied after that boot.

Earlier the same day, same boot and code (previous harness, profiler timer, 3 interleaved rounds), relative to
the base: `AITER_TDM_B_TH=0` (temporal B loads) +10.3% gemm1 / +5.0% gemm2; `AITER_FLYDSL_MXFP4_CLUSTER_M=4`
(4x4 cluster for both GEMMs) +6.3% / +6.8%.

## Notes

- B load hint: in 8ea2265dc `AITER_TDM_B_TH` overrides the config's `tdm_b_th` column only when set
  (`aiter/ops/flydsl/grouped_gemm_mxfp4.py`, `_select_tdm_b_th`). In 2489bae2d it defaulted to 1 and always won.
- `waves_per_tensor_tdm=4` gives wrong results (about 60% NaN) at 2489bae2d: with 2 M waves the A-scale is split
  along K. A fix is on `dev/lixzhang/yadai_dsr1_tdm_scale_waves` (`AITER_FLYDSL_TDM_SCALE_WAVES=2`). Not rechecked
  at 8ea2265dc.
- B (weights) is never shared at this shape, and a 4x4 cluster puts four different experts in a column. The
  kernel multicasts B only between M tiles of the same expert, which needs more than 256 rows per expert
  (more than 2048 tokens at 64 experts, top-8).
- `quant_a1` at 8ea2265dc (`moe_apre_quant_*_pk8_w16` + `moe_apre_scatter_*`) took 20.7 us graph_us on 2026-10-29,
  against 14.0 us for 2489bae2d's `moe_apre_gather_*` + `moe_apre_quant_*_qks_k512_pkamax` (#6984).
