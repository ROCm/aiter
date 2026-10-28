# FlyDSL AOT Pre-compilation & Tests

This directory holds the **AOT (Ahead-Of-Time) pre-compilation entry points** for
FlyDSL kernels. Each module derives the runtime-reachable jobs from aiter's
tuned CSV configs (and, for MoE, from the deployments in
`moe_deployments.json`)—including ABI/cache variants that share a kernel
name—and compiles them up front so the runtime JIT path hits the cache.

| Module | OpKind | Description |
| --- | --- | --- |
| `moe.py` | `MOE` | Every FlyDSL kernel the tuned fused-MoE / FHMoE / gfx1250 grouped rows and the MoE deployments reach, by replaying `fused_moe` itself (see below) |
| `gemm.py` | `GEMM` | GEMM kernels |
| `conv.py` | `CONV` | conv3d implicit-GEMM kernels, plus the NCDHW->NDHWC pre-transpose each NCDHW caller reaches first |
| `chunk_gdn_h.py` | `CHUNK_GDN_H` | chunk-gdn-h opt (K5) kernels |
| `mega_moe.py` | `MEGA_MOE` | MegaMoE A8W4 and A4W4 bundles of every MegaMoE deployment, by preloading `MegaMoEV2` itself (see below) |
| `deployments.py` | — | Loads `moe_deployments.json`, the MoE models and deployments `moe.py`, `mega_moe.py` and `op_tests/test_moe_ep.py` derive each rank's call from (see below) |
| `fmha_fp8.py` | `FMHA_FP8` | gfx950 FP8 flash-attention forward, per head shape in `DEFAULT_SHAPES` |
| `mxfp4_moe.py` | — | Not part of the AOT build; the MXFP4 tuner's `--fast-scan` precompiles its candidates with it |
| `common.py` | — | Shared job collection, the deadlock-free fork pool, and cache-hit checking logic |
| `dry_run.py` | — | Runs aiter's real host code on CPU (FakeTensors, `COMPILE_ONLY=1`, GPU runtime forbidden) |

### How `moe.py` works

`moe.py` does not restate any MoE launch logic. For every tuned row it builds
what a caller passes `fused_moe` (bf16 activations, routing, and weights
quantized and preshuffled the way `op_tests/test_moe_2stage.py` prepares them)
as FakeTensors and calls the runtime's own implementation (`_fused_moe_impl`,
or `_fhmoe_impl` for shared-expert rows) inside `dry_run`. The runtime looks the
row up, picks and parameterizes its kernels, sizes its buffers and launches its
auxiliary kernels as at inference; every FlyDSL launch on that path is
compiled into the cache instead of run. Kernel or wrapper changes therefore
need no AOT change.

- Each row is replayed at its tuned token count and at the smallest M that
  `get_padded_M` maps to it, because some compile-time parameters follow the
  actual M. Where the two request different FlyDSL kernels, the replay bisects
  the M between them down to every M at which the runtime switches kernels,
  since a row can switch more than once in its bucket (e.g. the gfx1250
  stage-1 quant's K split). These extra replays are reported as `[BISECT]`.
  `--row-tokens-only` replays the tuned count only.
- Caller-side choices a row does not fix are enumerated: expert bias for
  Swiglu + MXFP4 rows only (the GPT-OSS path; on both stages, and on stage 1
  alone, since a stage-2 bias makes the runtime re-select configs whose stage 2
  cannot take one), both gate/up layouts for MXMOE
  stage-1 rows, the default and model SiTUv2 betas, a received token count
  (`num_local_tokens`, as callers behind an EP dispatch pass), and for FHMoE
  rows with MXFP8 routed weights both values of `clamp_shared`, which the
  runtime compiles into stage 1 (MXFP4 rows keep the shared expert clamped).
  Combinations the runtime rejects (`NotImplementedError`) are reported as
  `[SKIP]`.
- The default caller of a row must be served. If the runtime refuses it at
  only some of the row's token counts, an M-dependent runtime limit applies
  (e.g. the FlyDSL route-reduce workspace beyond 32-bit addressing, which
  `op_tests/test_moe_2stage.py` skips as well); those token counts compile
  nothing and are reported as `[REFUSED]`. Refused at every token count, or
  failing with any other error, the row fails the build.
- Rows of the gfx1250 grouped table replay with the gate/up-interleaved
  weights and n32k4 scales gfx1250 callers pass; `_fused_moe_impl` hands them
  to FlyDSL's grouped GEMM. Grouped rows tuned for EP only (`topk` -1, or
  `ep_fused`) are not replayed as rows: the runtime picks them only under a
  caller's expert mask or MegaMoE's stage-2 scatter, and the table records
  neither the global expert count nor the real topk. The EP deployments
  below reach the `topk` -1 rows; MegaMoE's stage-2 scatter (`ep_fused`) is
  not replayed.
- Each row compiles for the arch its `gfx` column names. When `GPU_ARCHS`
  names targets, as `setup.py`'s wheel build has it, only rows of those archs
  are replayed; unset or `native`, rows of every arch are.
- aiter fixes some arch-dependent values at import (e.g. `dtypes.fp8`), so the
  replay runs in one process per arch, started with `GPU_ARCHS` naming that
  arch and `AITER_AOT_REPLAY=1`. Under that flag `get_gfx_runtime()` and
  `dtypes.fp8` answer the `GPU_ARCHS` arch instead of probing a device, so a
  visible GPU of another arch cannot leak in; the GPUs stay hidden all the
  same. `python -m aiter.aot.flydsl.moe` starts those processes side by side
  and splits the worker budget (`AITER_MAX_JOBS`, below) between them by job
  count. Beforehand, it builds aiter's C++ core module and the one
  behind the two-stage host-side query once, for the arch with the most
  two-stage jobs and with the whole budget: the replays run only their host
  code, and not every toolchain builds every arch (ROCm 7.2 cannot build the
  cluster launch gfx1250 builds turn on).
- A worker replays up to 12 jobs of one row in turn, so the kernels they share
  compile once. `AITER_FLYDSL_AOT_TIMEOUT` applies to the worker's whole batch.
- `--check` compiles nothing and fails on any kernel the replay would need but
  the cache lacks; it is the CPU-side counterpart of the step-2 test below.

#### Deployments

A caller behind an expert-parallel dispatch passes `fused_moe` an
`expert_mask`, which brings in launches no tuned row reaches: the gfx950
stage-2 reduction, compiled for the mask's length
(`moe_reduction_..._m1e<len>`), and the gfx1250 route (`moe_g2l_lut`,
`moe_route_g2l_lds`) and bf16 gather-reduce. The mask's length follows from
the deployment, which no tuned row records, so these callers come from the
`fused_moe` deployments of `moe_deployments.json` (see
[MoE deployments](#moe-deployments)). A tensor-parallel caller passes no
mask, but its call also depends on what tuned rows leave out: how it runs the
shared expert (in the routed weights, apart, or through FHMoE). The file
lists those deployments too.

Each deployment replays at the smallest and largest M of every
`get_padded_M` bucket up to `max_tokens`, bisected as for rows, with the
caller's default gate/up layout and the SiTUv2 betas a row would get. An EP
deployment replays with its mask, with and without `num_local_tokens` (MORI
dispatch passes it, the all-gather path does not). A deployment reads the
tuned tables serving reads (`AITER_CONFIG_FMOE`, `AITER_CONFIG_GROUPED_FMOE`,
`AITER_CONFIG_FHMOE`, or the merged defaults), so it needs no tuned row and
picks one up where the table has it. The activation dtype is
pinned to the deployment's quant recipe, as serving pins it: left to
`fused_moe`, gfx950 takes bf16 activations at small M (at every M for SiTUv2),
which expert masking refuses, and gfx1250 fp4 unless `AITER_FORCE_A8W4` is
set. `--csv` or `--deployments` alone replays only what it names; with
neither, both defaults are replayed.

### MoE deployments

`moe_deployments.json` lists the MoE models aiter serves and how each is
deployed; `deployments.py` validates it and derives what each rank hands the
runtime. `moe.py` replays its `fused_moe` deployments, `mega_moe.py` its
`mega_moe` ones, and `op_tests/test_moe_ep.py -t aot_ep_profiles` checks the
EP `fused_moe` ones on a GPU.

A model, under `models`, records its architecture as its `config.json` gives
it:

| Field | Meaning |
| --- | --- |
| `name`, `source` | display name; where the numbers come from |
| `model_dim`, `inter_dim` | `hidden_size`; `moe_intermediate_size`, unsplit |
| `routed_experts`, `top_k` | routed experts; how many each token picks (`num_experts_per_tok`) |
| `shared_experts` | shared experts, 0 if none |
| `activation` | `Silu`, `Situv2` or `Swiglu` |
| `swiglu_limit` | the activation clamp, if the model has one |
| `situ_beta`, `situ_linear_beta` | the SiTUv2 betas: required for `Situv2` models, refused for others |
| `note` | free text (JSON has no comments) |

A deployment, under `fused_moe` or `mega_moe`, names a model and the knobs it
is served with. `quant`, `ep`, `tp` and `redundant_experts` take one value or
a list, and expand to every combination:

| Field | Meaning |
| --- | --- |
| `model`, `gfx`, `cu_num` | the model; the GPU (`fused_moe`: gfx950 or gfx1250, `mega_moe`: gfx950) |
| `quant` | `a8w4` or `a4w4`: MXFP4 (per_1x32) weights with FP8 or FP4 activations |
| `ep` | EP size (`fused_moe`: default 1, no EP), the world size for MegaMoE |
| `tp` | `fused_moe` only: TP size of the experts, default 1. The experts split over TP or over EP, not both |
| `redundant_experts` | EPLB redundant experts (ATOM's `num_redundant_experts`), default 0; EP only |
| `shared_expert_mode` | `fused_moe` only: `separate` (default), `fused` into the rank's experts, or `fhmoe` (FHMoE runs the one shared expert from its own weights; TP only) |
| `max_tokens` | `fused_moe` only: the largest M a rank passes |
| `max_tokens_per_rank` | `mega_moe` only: MegaMoE's `max_tok_per_rank`, one value or a list |
| `source`, `note` | where the deployment comes from; free text |

ATOM and vLLM build a rank's call one way, and `deployments.py` follows it.
With P the routed plus redundant experts, which must split evenly over EP, and
S the shared experts the call runs (0 when `separate`), a `fused_moe` rank
passes:

- `w1.shape[0]` = P / EP + S experts, every routed expert under TP;
- `topk_ids.shape[1]` = top-k, or top-k + S when the call runs shared experts,
  plus, behind an EP dispatch, a sentinel column that masks them on the tokens
  of other ranks (ATOM's `init_aiter_topK_meta_data`);
- behind an EP dispatch, an `expert_mask` of P + S + 1 entries; under TP none;
- an intermediate dim of `inter_dim` / TP, and the model's `hidden_size`;
- with `fhmoe`, `shared_expert_id` = the routed expert count, the shared
  expert's slot.

For example DeepSeek-V4-Pro at EP8 with 64 EPLB redundant experts passes 56
local experts, top-k 6 and a 449-entry mask; at TP8 through FHMoE it passes
385 experts (the shared one at slot 384), top-k 7 and no mask, with an
intermediate dim of 384. A MegaMoE rank holds P / EP experts.

Each deployment gets an id, `{model}/{gfx}/{quant}/ep{N}[-r{redundant}][-fused]`
or `{model}/{gfx}/{quant}/tp{N}[-fused|-fhmoe]`, for example
`deepseek-v4-pro/gfx950/a8w4/ep8-r32`, which
the replays log and `test_moe_ep.py --deployment-id` selects by. Files given
together load together: their models merge, one id per definition, and a file
of your own may name the default file's models without repeating them.
Unknown or missing fields, experts that do not split over EP or an
intermediate dim that does not split over TP, TP and EP together, redundant
experts without EP, fusing a model without shared experts, FHMoE with EP or
with more than one shared expert, or an id listed twice fail the load.

To add a deployment:

1. add its model under `models` if it is new, from the model's `config.json`;
2. add the deployment with its `source`;
3. check what it expands to: `python -m aiter.aot.flydsl.deployments`;
4. build it (`python -m aiter.aot.flydsl.moe --deployments ...` or
   `python -m aiter.aot.flydsl.mega_moe --deployments ...`) and, for an EP
   `fused_moe` deployment, run `op_tests/test_moe_ep.py -t aot_ep_profiles`
   on its GPU (see section 2);
5. to see which of the model's tuned rows still compile only because of the
   tuned tables, build the default deployments alone into an empty cache and
   run `python -m aiter.aot.flydsl.moe --check` against it: each `[FAIL]` is a
   tuned-row job no deployment builds.

### How `mega_moe.py` works

`mega_moe.py` does not restate MegaMoE's kernel selection either. For each rank
of each `mega_moe` deployment, at each of its max tokens per rank, it builds
the runtime's own `MegaMoEV2` inside `dry_run`, with FakeTensor expert
weights, and calls `preload_aot_bundles()`: the method serving uses to load
every Stage1/prepare, Stage2 and combine variant of the deployment's bundle
plan.

- mori shmem, through which `MegaMoEV2` allocates its symmetric buffers and
  exchanges peer pointers, hands out FakeTensors and fake addresses, so no GPU,
  process group or mori install is needed.
- The device reports itself as an MI355X, because the combine launch geometry
  is tuned per GPU model.
- `quantize()` is replayed too: `preload_aot_bundles()` leaves it out, but
  fixed-slot `forward` and the per-stage benchmarks call it.
- Like the MoE replay, it runs in a process of its own, started with
  `GPU_ARCHS=gfx950` and `AITER_AOT_REPLAY=1`; both
  `python -m aiter.aot.flydsl.mega_moe` and `setup.py` start it that way. It
  is skipped when `GPU_ARCHS` names targets without gfx950.

---

## 0. Set up the environment

Run everything inside your Python virtualenv, e.g.:

```bash
source /path/to/venv/bin/activate
```

All commands below assume you run them from the repo root (the top-level `aiter`
directory of your checkout).

---

## 1. Run AOT pre-compilation (compile smoke test)

The most direct "test" is to run each module as a `python -m` entry point and
confirm every kernel compiles. Each module prints `Compiled: N ok, M failed` at
the end and exits 0 when all succeed, 1 on any failure — so it plugs straight
into CI.

```bash
# MoE / Mixed-MoE (default CSVs)
python -m aiter.aot.flydsl.moe

# MoE rows of gfx1250 only, which CI's gfx942;gfx950 wheel build leaves out
GPU_ARCHS=gfx1250 python -m aiter.aot.flydsl.moe

# GEMM
python -m aiter.aot.flydsl.gemm

# conv3d
python -m aiter.aot.flydsl.conv

# chunk-gdn-h
python -m aiter.aot.flydsl.chunk_gdn_h

# MegaMoE bundles of every MegaMoE deployment (all token capacities, all ranks)
python -m aiter.aot.flydsl.mega_moe

# Only some deployments, when building a smaller custom image
python -m aiter.aot.flydsl.mega_moe --experts-per-rank 48
```

The FP8 flash-attention kernel picks its tile, rescale threshold and split-K
factor per call, so a serving process JIT-compiles each new combination on the
request path. `fmha_fp8` enumerates every combination the wrapper's heuristics
can reach for each head shape in its `DEFAULT_SHAPES` table. Add a model by
adding an entry (`num_heads, num_kv_heads, head_dim, head_dim_v, layout`, heads
per rank, layout `varlen_cross` for the chunked-prefill serving path).

```bash
# every configured shape (Kimi-K3 at TP8: 12:12:192:128 -> 92 kernels)
python -m aiter.aot.flydsl.fmha_fp8
# ad hoc shape, without editing DEFAULT_SHAPES
python -m aiter.aot.flydsl.fmha_fp8 --shape 12:12:192:128
```

The gfx1250 MoE build imports aiter, so it needs aiter's C++ core module. An
installed wheel ships it. Built for gfx1250, the module needs a HIP that
declares the cluster launch API, which ROCm 7.2.4's does not; there, build it
once for a CDNA arch first (it has host code only):

```bash
GPU_ARCHS=gfx950 python -c "import aiter"
GPU_ARCHS=gfx1250 python -m aiter.aot.flydsl.moe
```

MegaMoE builds every `mega_moe` deployment of `moe_deployments.json`:
DeepSeek-V4-Pro r0/r32/r64 (`experts_per_rank` 48/52/56) and Kimi-K3 (112).
The expert count is part of the compiled ABI, so one deployment cannot safely
reuse another's bundle. `--mtpr`, `--quant` and `--experts-per-rank` narrow
the deployments down; `--deployments` takes files of your own.

### Common arguments

```bash
# Custom CSV(s) — every module supports --csv and accepts multiple paths
python -m aiter.aot.flydsl.moe --csv /path/to/config1.csv /path/to/config2.csv
python -m aiter.aot.flydsl.chunk_gdn_h --csv /path/to/tuned.csv

# MoE: verify an existing cache on CPU instead of compiling
python -m aiter.aot.flydsl.moe --csv /path/to/config.csv --check

# MoE: only the deployments (default file or your own)
python -m aiter.aot.flydsl.moe --deployments aiter/aot/flydsl/moe_deployments.json

# Validate deployment files and list what they expand to
python -m aiter.aot.flydsl.deployments
```

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `AITER_AOT_IMPORT` | Set to `1` so `import aiter` only loads the lightweight JIT core and skips the full top-level op namespace — faster and avoids heavy import side effects during AOT compilation (this is what `setup.py` sets while pre-compiling). | `0` |
| `FLYDSL_RUNTIME_CACHE_DIR` | Cache directory | `~/.flydsl/cache` |
| `AITER_MAX_JOBS` | Optional AITER-local ceiling for concurrent CPU compilation workers. If it is unset, standalone FlyDSL AOT CLIs adopt a valid positive legacy `MAX_JOBS` with a `FutureWarning`; AITER imports and runtime JIT do not. Sizing and precedence follow the shared policy in the [top-level README](../../../README.md#build-parallelism); invalid values use automatic sizing and non-positive values impose a one-worker ceiling. | auto |
| `AITER_FLYDSL_AOT_TIMEOUT` | Per-kernel wall-clock cap (seconds). A worker stuck *alive* past this is killed (and retried); `0` disables. | `1200` |
| `AITER_FLYDSL_AOT_MAX_RETRIES` | Retries for a worker that **died abnormally** (OOM-kill / segfault / timeout-kill). A clean compile error is never retried. `0` disables. | `2` |
| `AITER_CONFIGS` | Resolves the default CSV lookup path (same as the runtime JIT) | repo built-in |
| `ARCH` / `GPU_ARCHS` | Selects which jobs to build, not what arch a job compiles *for* (that comes from the CSV's `gfx`, with legacy `cu_num` fallback). `conv.py` applies it inside `parse_csv`, and `moe.py` and `mega_moe.py` (`GPU_ARCHS` only) before they start their replays, so both `python -m` and the `setup.py` path (`run_aot`) honour it. `gemm.py` still filters in `main()` only, so `run_aot` builds all of its archs. | auto-detect |

> **About the compile target arch.** The arch each kernel is actually compiled
> for is taken from the CSV's `gfx` column, with `cu_num_to_arch(...)` retained
> as a fallback for legacy rows, and applied internally via `FLYDSL_GPU_ARCH`.
> That internal var is overwritten
> for every job, so setting `ARCH` / `GPU_ARCHS` / `FLYDSL_GPU_ARCH` in your shell
> does **not** change what gets built. To cross-compile, use the target `gfx` and
> `cu_num` values in the CSV. `moe.py` and `mega_moe.py` additionally run each
> target arch in its own process with `GPU_ARCHS` pinned and `AITER_AOT_REPLAY=1`,
> so they build the same kernels on a machine with or without GPUs.

Example:

```bash
AITER_MAX_JOBS=16 python -m aiter.aot.flydsl.moe
```

An explicit `AITER_MAX_JOBS` always takes precedence. Legacy adoption copies
the positive value without changing `MAX_JOBS`; without either setting, worker
selection is fully automatic.

---

## 2. Run the "AOT cache hit" test

Compiling successfully is not enough — you also want to verify that the **runtime
actually hits the AOT cache** (no cache miss). The MoE tests do that by running
cases in FlyDSL run-only mode (`aiter.aot.flydsl.common.run_only_env`,
`FLYDSL_RUNTIME_RUN_ONLY=1`): a kernel missing from the cache raises instead of
compiling, and the case fails.

- `op_tests/test_moe_2stage.py` runs every FlyDSL row of the tuned tables
  (cases with `check_aot_cache=True`).
- `op_tests/test_moe_ep.py -t aot_ep_profiles` runs every EP `fused_moe`
  deployment of this GPU's arch through the expert-parallel `fused_moe` call
  (MXFP4 a8w4 / a4w4, Silu / SiTUv2), at each `-m` token count up to the
  deployment's `max_tokens`. `--deployments` points it at other files,
  `--deployment-id <glob>` picks deployments by id, `--ep-mode fake` routes
  every token locally. It runs first among the test's defaults, before the other
  tests JIT-compile in the same process.

Full flow:

```bash
source /path/to/venv/bin/activate

# (1) First compile the kernels into the cache
python -m aiter.aot.flydsl.moe

# (2) Then run the tests with cache checking.
python op_tests/test_moe_2stage.py
python op_tests/test_moe_ep.py -t aot_ep_profiles -m 16 128 1024 8192
```

> Note: both steps must use the **same** `FLYDSL_RUNTIME_CACHE_DIR` and run on
> (or target) the **same GPU arch**, otherwise step 2 will be treated as a miss
> because the cache dir / arch don't line up. Deployments replay against
> the tuned tables the environment selects, so keep `AITER_CONFIG_FMOE` /
> `AITER_CONFIG_GROUPED_FMOE` the same in both steps too.

---

## 3. Troubleshooting

- **`CSV file not found` / `file not found`**: check the `--csv` (or, for MoE,
  `--deployments`) path, or whether `AITER_CONFIGS` points at a valid config
  directory.
- **Lots of `[FAIL]` prints + exit code 1**: an individual kernel failed to
  compile; stdout has per-kernel diagnostics. The exception message inlines at
  most 10 entries (`_MAX_ERRORS_IN_MSG` in `common.py`), the rest are elided as
  `(... N more)`.
- **Every kernel fails with the same error** (e.g. `'ArithValue' object has no
  attribute 'ir_value'`): this is a **FlyDSL version mismatch**, not a per-kernel
  problem. Check the *imported* FlyDSL:
  ```bash
  python -c "import flydsl, os; print(flydsl.__version__, os.path.dirname(flydsl.__file__))"
  ```
  If the version is older than the project's `FLYDSL_VERSION` (top-level
  `setup.py`), a stale build is winning on `PYTHONPATH`. Either `pip install` the
  matching version *and* drop the shadowing entry from `PYTHONPATH`, or rebuild
  your local FlyDSL checkout (`scripts/build.sh`, after `pip install
  nanobind==2.12.0` if CMake reports it missing) so the on-`PYTHONPATH` build dir
  is refreshed to the right version.
- **Worker OOM / killed (exitcode -9)**: abnormal exits are auto-retried
  (`AITER_FLYDSL_AOT_MAX_RETRIES`) and the default worker count is already
  bounded by the shared CPU and memory policy. If it still happens,
  lower `AITER_MAX_JOBS`. Nested compiler fanout is fixed at one inside each
  AOT worker.
- **A kernel hangs / never finishes**: it is killed once it exceeds
  `AITER_FLYDSL_AOT_TIMEOUT` (default 1200 s) and then retried. Lower the timeout
  to fail faster, or raise it for genuinely slow kernels.
- **`hipModuleLoadData ... hipErrorNoBinaryForGpu` printed but the kernel still
  shows `[OK]`**: expected when AOT-compiling for an arch that is **not** the
  machine's GPU (e.g. building `gfx950` artifacts on a different card). MLIR
  compilation and the cache write succeed; only the *load* step fails, which AOT
  does not need. It is noise, not a failure.
- **Step 2 reports a cache miss**: confirm step 1 actually ran, the cache dir and
  arch match, and the CSV config hasn't changed.
