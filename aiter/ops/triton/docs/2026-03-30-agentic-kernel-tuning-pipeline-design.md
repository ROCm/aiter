# Agentic Triton Kernel Tuning Pipeline — Design Spec

**Date**: 2026-03-30
**Status**: Draft
**Context**: Automate the end-to-end kernel config tuning workflow when upgrading Triton compiler versions on AMD GPUs (MI300X/gfx942, MI350X/MI355X/gfx950, etc.). Replaces the current manual process that is error-prone, context-heavy, and takes days per kernel.

---

## 1. Problem Statement

Upgrading the Triton compiler (e.g., from one commit to another on the same branch, or across major versions) changes code generation, which can regress GEMM kernel performance. The current process requires a human to:

1. Identify all kernels and their shapes
2. Collect baselines on the old Triton
3. Switch to new Triton and validate with existing configs
4. Tune regressed shapes using screen.py
5. Generate new configs, validate, fix regressions
6. Iterate until no tuning regressions remain
7. Commit

This process is manual, takes 3-5 days per kernel, and is error-prone due to context loss (agent forgets steps, uses wrong measurement methodology, corrupts shared configs).

## 2. Goals

- **Fully automated**: human provides a YAML config and approves commits — everything else is autonomous
- **Multi-machine**: distribute work across a pool of GPU machines
- **Reliable**: active health monitoring at every level, no silent failures
- **Observable**: terminal dashboard + notifications for critical decisions
- **Adaptive**: learn config patterns from scout tuning to narrow search space

## 3. Non-Goals

- Modifying kernel source code (only configs)
- Supporting non-GEMM kernels in v1 (but architecture should not preclude it)
- Auto-merging PRs (commit only, human pushes)

---

## 4. Architecture

### 4.1 Agent Hierarchy

```
Human ── triton-upgrade.yaml
              │
      Orchestrator Agent (local WSL)
      ├── Dashboard (terminal UI)
      ├── Notification system
      ├── Machine pool manager
      │
      ├── Kernel Supervisor: a8w8 ──→ Machine 1 (via SSH + docker exec)
      │   ├── Setup Agent
      │   ├── Discovery Agent
      │   ├── Script Creator Agent
      │   ├── Baseline Agent
      │   ├── Tuning Agent
      │   ├── Pattern Analyzer Agent
      │   ├── Config Generator Agent
      │   ├── Validation Agent
      │   └── Regression Fixer Agent
      │
      ├── Kernel Supervisor: afp4wfp4 ──→ Machine 2
      │   └── (same subagent types)
      │
      └── Kernel Supervisor: a16w16 ──→ Machine 1 (reused)
          └── (same subagent types)
```

### 4.2 Execution Model

- **All agents run on the local WSL machine** (your laptop)
- Remote commands execute via `ssh user@host "docker exec container_name bash -c 'command'"`
- No Claude instances run inside containers — avoids auth distribution and TOS issues
- Each kernel supervisor gets one machine allocated by the orchestrator
- Multiple machines can tune different kernels in parallel
- Subagents within a kernel supervisor can parallelize across GPUs on their machine

### 4.3 Health Monitoring (Watchdog Pattern)

Every level of the hierarchy actively monitors what it dispatched:

**Subagents**: when launching any long-running command (screen.py, rocprof, etc.):
- Check within 2 minutes that output is being produced
- Verify incremental progress at configurable intervals (e.g., new `screencase` lines appearing in logs)
- Kill and report failure if no progress after timeout
- Check for stale processes before starting new work (`ps aux`, `rocm-smi --showpids`)

**Kernel Supervisor**: monitors its subagents:
- If a subagent goes silent (no log updates) for longer than expected → kill, diagnose, retry or escalate
- Track wall-clock time per phase, alert if exceeding expected bounds

**Orchestrator**: monitors kernel supervisors:
- If a machine shows no GPU activity for too long → investigate
- If a kernel supervisor hasn't reported progress → ping and escalate
- Dead container detection via `docker ps` health checks

---

## 5. Configuration

Single YAML file read by the orchestrator at startup:

```yaml
# triton-upgrade.yaml

baseline:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: triton_3_4  # or a specific commit hash

target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: alizaidy/gfx950-kernel-fixes-cherry-picked
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main  # or a specific commit hash

machines:
  - host: gpu-machine-1.internal
    user: root
    ssh_key: ~/.ssh/id_gpu_cluster
    gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  - host: gpu-machine-2.internal
    user: root
    ssh_key: ~/.ssh/id_gpu_cluster
    gpus: [0, 1, 2, 3]  # only 4 available

container:
  image: rocm/pytorch:latest
  run_script: ./scripts/create_tuning_container.sh  # optional, agent can use directly or parse
  # If no run_script, agent constructs: docker run -d --device=/dev/kfd --device=/dev/dri ...

gpu:
  arch: gfx950  # auto-detected via rocminfo if omitted. Used for config file prefix (gfx950-GEMM-...)

triton_install:
  # How to install Triton. Agents cd into the repo and run this command.
  command: "pip install -e ."  # or "pip install dist/*.whl" for pre-built wheels

tuning:
  mode: regression_only  # or "full"
  scout_fraction: 0.15   # fraction of shapes to use in scout phase
  thresholds:
    regression_vs_baseline: 5   # % — escalate to human for approval
    regression_vs_untuned: 2    # % — auto-restore old config
  timeouts:
    command_default: 300        # seconds — default per-command timeout
    tuning_per_shape: 1800      # seconds — max time for one shape's tuning
    progress_check: 120         # seconds — check for output within this window
    phase_max: 14400            # seconds — max time for an entire phase

kernels:
  # Optional overrides. If absent, auto-discover all kernels.
  exclude: [a16w16_agnostic]  # skip these
  include: []                  # if non-empty, ONLY tune these
  overrides:
    afp4wfp4_preshuffle:
      m_leq_16_bucket_name: M_LEQ_31  # preshuffling convention
    a16w16:
      extra_block_k: [64]  # bf16 needs BK=64 for large M
```

---

## 6. Orchestrator

### 6.1 Startup

1. Parse and validate `triton-upgrade.yaml`
2. Test SSH connectivity to each machine, verify GPU availability (`rocm-smi`)
3. Auto-discover kernels across all GEMM categories:
   - Scan `aiter/ops/triton/gemm/basic/gemm_*.py` for basic GEMM kernels
   - Scan `aiter/ops/triton/gemm/batched/batched_gemm_*.py` for batched GEMM kernels
   - Scan `aiter/ops/triton/gemm/feed_forward/ff_*.py` for feed-forward fused kernels
   - Scan `aiter/ops/triton/gemm/fused/fused_gemm_*.py` for fused GEMM kernels
   - Match to config files in `aiter/ops/triton/configs/gemm/` (GEMM-*, BATCHED_GEMM-*, FF-*, FUSED-GEMM-*)
   - Match to ut_* scripts in `aiter/ops/triton/utils/_triton/tunning/`
   - Match to bench scripts in `op_tests/op_benchmarks/triton/` (bench_gemm_*, bench_batched_gemm_*, bench_ff_*)
   - Match to test scripts in `op_tests/triton_tests/gemm/`
4. Apply kernel include/exclude overrides from config
5. Launch terminal dashboard
6. Present plan to human: "Found N kernels across M machines. Estimated time: X hours. Proceed?"

### 6.2 Main Loop

1. Pick next unprocessed kernel
2. Allocate a machine from the pool (wait if none available)
3. Dispatch kernel supervisor agent with full context:
   - Kernel name, file paths, shape list
   - Machine info (host, user, SSH key, GPU list)
   - Container config
   - Baseline/target repo+branch info
   - Thresholds and timeouts
   - Tuning mode
4. Mark machine as busy, update dashboard
5. When kernel supervisor completes: record results, release machine
6. If escalation needed: notify human, pause kernel, continue others
7. Repeat until all kernels processed

### 6.3 Shutdown

1. Generate summary report:
   - Per-kernel: geomean speedup vs baseline, regressions count, shapes improved
   - Overall: total shapes tuned, total regressions, machines used, wall time
2. Notify human: "Upgrade complete. Review summary."

---

## 7. Kernel Supervisor

One kernel supervisor per kernel. Owns the full tuning lifecycle for that kernel on its allocated machine.

### 7.0 Phase 0: Environment Setup

**Setup Agent** handles:
1. SSH into machine
2. Create container using provided image/script
3. Clone aiter repo (target branch) inside container
4. Install target Triton inside container
5. Verify: `python -c "import triton; print(triton.__version__)"` — check commit hash matches expected
6. Verify: `git rev-parse HEAD` in aiter — check branch matches expected
7. Clean any stale GPU processes: `rocm-smi --showpids`, kill orphans

### 7.1 Phase 1: Pre-flight Discovery

**Discovery Agent** handles:
1. Scan kernel source file — identify function signatures, tensor dtypes, config lookup mechanism
2. Find all config files for this kernel (suffixed + default fallback)
3. Extract all N,K pairs from config files
4. Extract relevant shapes from `op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json`
5. Identify the fallback shape (typically N=8192, K=8192)
6. Check for existence of:
   - `ut_*.py` tuning script
   - `bench_gemm_*.py` benchmark script
   - `test_gemm_*.py` unit test script
7. Report: shapes list, missing scripts, config file inventory

**Script Creator Agent** (if needed):
1. Read existing ut_*/bench/test scripts as templates
2. Read the kernel source to understand:
   - Function signature and arguments
   - Input tensor shapes, dtypes, and packing conventions
   - Config parameter names and their mapping
3. Generate missing scripts following established patterns
4. Smoke test: run one shape end-to-end to verify the script works
5. If kernel intent is ambiguous (e.g., unclear which function to profile, unusual tensor layout) → **escalate to human**

### 7.2 Phase 2: Baseline Collection

**Setup Agent**: install baseline Triton (overwrite target) inside the same container. Verify commit hash.

**Baseline Agent**:
1. Clean stale GPU processes
2. For each shape, sequentially on a single GPU:
   ```
   rocprof --stats -o <unique_prefix>.csv python bench_gemm_<variant>.py \
       --shape M N K --metric time --layout TN
   ```
3. Parse `<prefix>.stats.csv`: extract main kernel AverageNs + reduce kernel AverageNs (if split-K)
4. Save structured output: `baseline_<kernel>.json` with per-shape main_ns, reduce_ns, total_ns
5. Health check: verify each shape produces a stats file, abort if consecutive failures

**Setup Agent**: reinstall target Triton. Verify commit hash.

### 7.3 Phase 3: Untuned Validation

**Validation Agent**:
1. Same methodology as baseline collection, but on target Triton with existing (old) configs
2. Save as `untuned_<kernel>.json`
3. Can parallelize across GPUs using unique `-o` prefixes per shape (no file collisions)

**Note on parallelization**: Baseline (Phase 2) MUST be sequential on a single GPU because it establishes the reference timing — GPU contention from parallel runs introduces noise that corrupts the reference. Validation (Phase 3, 5) can parallelize because we compare *relative* to the baseline; consistent bias across parallel runs cancels out. However, if validation results show suspicious noise (>10% variance on repeated measurements), fall back to sequential collection.

### 7.4 Phase 4: Tuning

**Decision point** (based on tuning mode):
- **Regression-only mode**: compare untuned vs baseline, identify regressed shapes (> threshold). Only tune those.
- **Full mode**: tune all shapes.

**Scout Phase** (Tuning Agent):
1. Select representative subset (~15% of shapes to tune): one small M, one medium M, one large M per N,K pair
2. Run screen.py with **broad search space** (all BM/BN/BK/stages/nonkdim/ksplit)
3. Health check: verify `screencase` lines appearing in logs within 2 minutes
4. If a shape produces 0 valid configs → diagnose (LDS overflow? AOT cache? batch size too large?)

**Pattern Analysis** (Pattern Analyzer Agent):
1. Analyze scout results: for each config parameter and M range, count how often each value appears in the top-3 configs across scout shapes
2. Load historical data from prior kernel tunings (stored in `~/.tuning_results/history/`) as a weak prior — historical values get 0.25x weight vs scout results
3. For each parameter: keep values that appear in >20% of winning configs (scout-weighted). If a value never appears in any winning config, drop it.
4. Output a **narrowed search space** per M range, e.g.:
   ```
   M <= 16: BM=[4,8,16], BN=[16,32], BK=[512,1024], stages=[2], nonkdim=[16]
   M 32-64: BM=[32,64], BN=[64], BK=[256,512], stages=[2,3], nonkdim=[16]
   M >= 128: BM=[64,128], BN=[128], BK=[256,512], stages=[2,3], nonkdim=[16,32]
   ```
5. Estimate time savings vs broad search
6. Sanity check: narrowed space must be >= 25% of broad space (if too narrow, widen back — may indicate insufficient scout coverage)

**Full Tuning Phase** (Tuning Agent):
1. Tune remaining shapes with narrowed search space
2. Distribute M values across GPUs (one M per GPU)
3. Process N,K pairs one at a time (avoids GPU contention)
4. For M=8192 or other very large shapes: use `SCREEN_MAX_BATCH` env var to reduce batch size
5. Active monitoring: check each GPU's log every `progress_check` seconds
6. Kill and retry (or escalate) if no progress

**Config Generator Agent**:
1. Run `view-screen.py` for each tuned N,K pair to generate JSON config files
2. Apply kernel-specific naming conventions (e.g., M_LEQ_31 for preshuffled afp4wfp4)
3. Copy configs to `aiter/ops/triton/configs/gemm/`
4. For shapes using the default fallback: only update the `any` bucket (M=8192 tuned config)

### 7.5 Phase 5: Validation + Regression Fixing

**Validation Agent**:
1. Collect all shapes (not just tuned ones) with new configs
2. Compare three-way: baseline (3.4) vs untuned (3.6) vs tuned (3.6)
3. Classify each shape using the two thresholds:
   - **Improved**: tuned better than baseline by > `regression_vs_baseline` threshold
   - **Compiler regression**: untuned worse than baseline by > `regression_vs_baseline` threshold, AND tuned ≈ untuned (within `regression_vs_untuned` threshold) — compiler's fault, tuning can't fix
   - **Tuning regression**: tuned worse than untuned by > `regression_vs_untuned` threshold — our fault, auto-restore old config
   - **Tuning regression (severe)**: tuned worse than baseline by > `regression_vs_baseline` threshold AND untuned was fine — escalate to human
   - **Neutral**: all deltas within their respective thresholds

**Regression Fixer Agent**:
1. For each tuning regression where the config actually changed (not measurement noise):
   - If the shape uses a **suffixed config**: restore the old config for the regressed bucket only
   - If the shape uses the **default fallback**: do NOT modify the fallback. Instead, create a new suffixed config for this N,K pair (copy fallback as base, the old config's bucket is already correct)
2. For fallback conflicts (tuning the fallback helps some shapes but regresses others):
   - Promote regressed shapes to their own suffixed config files
3. Revalidate all shapes for any N,K pair whose config was modified
4. Repeat until no tuning regressions above threshold remain (max 3 iterations, then escalate)

### 7.6 Phase 6: Commit

1. Generate per-kernel summary:
   - Geomean speedup vs baseline
   - Count: improved / regressed / neutral (vs baseline and vs untuned)
   - Table of all regressions with before/after timings
2. **Notify human** with summary. Wait for approval.
3. On approval: `git add` config files + any new scripts, commit with detailed message
4. Do NOT push — human decides when to push

---

## 8. Subagent Specifications

### 8.1 Common Requirements (all subagents)

- **Environment verification**: before any operation, verify Triton commit hash and aiter branch match expected values
- **Stale process check**: before using a GPU, check `rocm-smi --showpids` and kill orphans
- **Structured output**: every subagent writes results to a known JSON file that the kernel supervisor reads
- **Timeout enforcement**: every command has a timeout. Long-running commands have progress checks.
- **Error reporting**: on failure, write error details to a structured log. Don't silently continue.

### 8.2 Setup Agent

**Input**: machine info, container config, repo/branch info
**Output**: container ID, verified environment state
**Commands**: ssh, docker run/exec, pip install, git clone/checkout
**Key behavior**: idempotent — can be re-run safely if container already exists

### 8.3 Discovery Agent

**Input**: kernel name, paths to kernel source / configs / model_shapes.json
**Output**: shape list, missing script inventory, config file inventory
**Key behavior**: read-only, never modifies files

### 8.4 Script Creator Agent

**Input**: kernel name, kernel source code, existing scripts as templates, list of what's missing
**Output**: new ut_*/bench/test scripts, smoke test results
**Key behavior**: examines kernel API deeply, follows existing patterns, escalates ambiguity to human
**Templates**: uses existing ut_*.py and bench_gemm_*.py as structural templates

### 8.5 Baseline Agent

**Input**: shape list, bench script path, GPU ID
**Output**: `baseline_<kernel>.json` with per-shape {main_ns, reduce_ns, total_ns}
**Key behavior**: sequential on single GPU, uses `rocprof --stats -o <unique_prefix>.csv`, cleans up after each shape

### 8.6 Tuning Agent

**Input**: shapes to tune, search space (from Pattern Analyzer or broad default), GPU list
**Output**: screen.py log files
**Key behavior**: distributes M values across GPUs, monitors progress, uses SCREEN_MAX_BATCH for large shapes

### 8.7 Pattern Analyzer Agent

**Input**: scout tuning results, historical tuning data (optional)
**Output**: narrowed search space per M range
**Key behavior**: statistical analysis of what config parameters work, historical data weighted lower than scout results

### 8.8 Config Generator Agent

**Input**: screen.py log files, kernel-specific naming conventions
**Output**: JSON config files in the config directory
**Key behavior**: runs view-screen.py, applies M_LEQ naming rules, handles suffixed vs fallback distinction

### 8.9 Validation Agent

**Input**: shape list, baseline data, untuned data, bench script, GPU list
**Output**: three-way comparison table, classified regressions
**Key behavior**: can parallelize across GPUs using unique -o prefixes, sequential per GPU

### 8.10 Regression Fixer Agent

**Input**: regression list, current configs, old configs, config file paths
**Output**: fixed configs, list of promoted fallback shapes
**Key behavior**: never modifies the default fallback — promotes to suffixed instead. Iterates max 3 times then escalates.

---

## 9. Dashboard + Notifications

### 9.1 Terminal Dashboard

Live-updating terminal UI showing:
- Machine pool: hostname, status (idle/busy), current kernel, GPU utilization
- Per-kernel progress: current phase, shapes done/total, elapsed time, ETA
- Recent log lines from active subagents
- Regression summary: count per kernel

### 9.2 Notification System

Alerts for:
- **Approval needed**: commit ready, regression above threshold, kernel skip decision
- **Failure**: subagent crashed, command timed out, container died
- **Milestone**: kernel tuning complete, all kernels done

Delivery: terminal bell/notification initially. Extensible to Slack/email.

### 9.3 Pre-approved Commands

The following actions do not require human approval:
- Installing/switching Triton versions
- Collecting baselines and validation data
- Running tuning (screen.py)
- Generating configs
- Restoring old configs for tuning regressions below threshold
- Creating missing scripts (with smoke test passing)
- Killing stale processes
- Retrying failed subagents (up to 2 retries)

The following require human approval:
- Committing to git
- Pushing to remote
- Skipping a kernel that's persistently failing
- Accepting regressions above the configured threshold
- Script Creator when kernel intent is ambiguous

---

## 10. Data Flow & Artifact Location

All intermediate artifacts live **inside the remote container** at a well-known path: `/workspace/tuning_artifacts/<kernel_name>/`. Subagents read/write artifacts there via SSH + docker exec. The kernel supervisor retrieves final results (summary JSON, config files) to the local machine via `docker cp` / `scp` during Phase 6.

**Local machine** (WSL):
- `triton-upgrade.yaml` — input config
- `~/.tuning_results/<run_id>/` — final reports, committed configs (copied from containers)
- Dashboard state file

**Remote container** (`/workspace/tuning_artifacts/<kernel>/`):
```
baseline.json                    (Phase 2)
untuned.json                     (Phase 3)
scout_results/                   (Phase 4 scout — screen-*.log files)
patterns.json                    (Phase 4 analysis)
tuning_logs/                     (Phase 4 full — screen-*.log files)
configs/                         (Phase 4 config gen — generated JSON files)
tuned.json                       (Phase 5 validation)
regression_report.json           (Phase 5)
summary.json                     (Phase 6 — commit message + tables)
```

### 10.1 Config File Naming Convention

Config files follow the pattern: `<gfx_arch>-GEMM-<VARIANT>[-N=<N>-K=<K>].json`

The mapping from kernel name to VARIANT prefix:

**Basic GEMMs** (`aiter/ops/triton/gemm/basic/`):

| Kernel file | Config VARIANT | Notes |
|-------------|---------------|-------|
| `gemm_a8w8.py` | `A8W8` | |
| `gemm_a8w8.py` (blockscale) | `A8W8_BLOCKSCALE` | Separate config set |
| `gemm_a8w8.py` (blockscale preshuffle) | `A8W8_BLOCKSCALE_PRESHUFFLED` | |
| `gemm_a8w8_per_token_scale.py` | `A8W8_PER_TOKEN_SCALE` | |
| `gemm_a16w16.py` | `A16W16` | |
| `gemm_a16w16_atomic.py` | `A16W16-ATOMIC` | Note: hyphen not underscore |
| `gemm_a16w16_gated.py` | `A16W16_GATED` | |
| `gemm_a16w8_blockscale.py` | `A16W8_BLOCKSCALE` | |
| `gemm_a16wfp4.py` | `A16WFP4` | |
| `gemm_a8wfp4.py` | `A8WFP4` | |
| `gemm_afp4wfp4.py` | `AFP4WFP4` | Non-preshuffled |
| `gemm_afp4wfp4.py` (preshuffle) | `AFP4WFP4_PRESHUFFLED` | |
| `gemm_afp4wfp4_pre_quant_atomic.py` | `AFP4WFP4_PRE_QUANT_ATOMIC` | |

**Batched GEMMs** (`aiter/ops/triton/gemm/batched/`):

| Kernel file | Config VARIANT | Notes |
|-------------|---------------|-------|
| `batched_gemm_a8w8.py` | `BATCHED_GEMM-A8W8` | Has B (batch) dimension |
| `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py` | `BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | Long variant name |
| `batched_gemm_afp4wfp4.py` | `BATCHED_GEMM-AFP4WFP4` | |
| `batched_gemm_afp4wfp4_pre_quant.py` | `BATCHED_GEMM_PREQUANT-AFP4WFP4` | |
| `batched_gemm_a16wfp4.py` | `BATCHED_GEMM-A16WFP4` | |
| `batched_gemm_bf16.py` | `BATCHED_GEMM-A16W16` | |

**Feed-Forward Fused** (`aiter/ops/triton/gemm/feed_forward/`):

| Kernel file | Config VARIANT | Notes |
|-------------|---------------|-------|
| `ff_a16w16.py` | (no config) | May not use autotuned configs |
| `ff_a16w16_fused_gated.py` | `FF-A16W16-fused` | Note: lowercase "fused" |
| `ff_a16w16_fused_ungated.py` | (shares config with gated) | |

**Fused GEMMs** (`aiter/ops/triton/gemm/fused/`):

| Kernel file | Config VARIANT | Notes |
|-------------|---------------|-------|
| `fused_gemm_a8w8_blockscale_a16w16.py` | `FUSED-GEMM-A8W8_BLOCKSCALE-A16W16` | |
| `fused_gemm_a8w8_blockscale_mul_add.py` | (may share config) | |
| `fused_gemm_a8w8_blockscale_split_cat.py` | (may share config) | |
| `fused_gemm_afp4wfp4_a16w16.py` | `FUSED-GEMM-AFP4WFP4-A16W16` | |
| `fused_gemm_afp4wfp4_mul_add.py` | (may share config) | |
| `fused_gemm_afp4wfp4_split_cat.py` | (may share config) | |

**Notes on batched/fused kernels:**
- Batched kernels have a B (batch) dimension in addition to M, N, K. Shape format is (B, N, K) not (M, N, K).
- Some fused kernels only have gfx942 configs — gfx950 configs may need to be created from scratch.
- The Script Creator Agent must handle these different shape formats and config patterns.
- Feed-forward kernels may use different bench scripts (`bench_ff_*.py` vs `bench_gemm_*.py`).

- **Fallback config**: `gfx950-GEMM-A8W8.json` (no N,K suffix) — used when no suffixed config matches
- **Suffixed config**: `gfx950-GEMM-A8W8-N=128-K=2048.json` — used for specific N,K pairs

The Discovery Agent must derive this mapping by inspecting existing config files. The Config Generator Agent uses `view-screen.py` which auto-derives the prefix from the ut_* script filename.

### 10.2 Git Operations

Git operations happen **inside the remote container** where the aiter repo is cloned:
1. Config files are generated/modified in the container's aiter checkout
2. The kernel supervisor runs `git add` and `git commit` inside the container
3. On human approval, the supervisor runs `git push` from the container (or `git format-patch` + transfer to local)
4. Each kernel gets its own commit with a detailed performance summary in the commit message

Machine cleanup on release: the orchestrator runs `docker stop` + `docker rm` when releasing a machine. The container and all artifacts are destroyed. Final results must be copied to local before cleanup.

---

## 11. Error Handling

| Error | Detection | Response |
|-------|-----------|----------|
| SSH connection lost | Command timeout | Retry 3x with backoff, then mark machine as dead, reallocate kernel |
| Container crashed | `docker ps` check | Recreate container, restart from last checkpoint |
| screen.py 0 valid configs | Log monitoring (no `screencase` lines within 2 min) | Kill, diagnose (LDS? AOT cache? batch size?), retry with fixes |
| rocprof stats missing | File existence check after command | Retry once, then skip shape and flag |
| Stale GPU processes | `rocm-smi --showpids` before each operation | Kill orphans automatically |
| Triton commit mismatch | Verify before every subagent operation | Abort subagent, alert supervisor to reinstall |
| Measurement noise | Same config produces different results across runs | Revalidate regressed shapes, filter noise from real regressions |
| Fallback config conflict | Tuning fallback regresses some shapes | Promote regressed shapes to suffixed configs |

---

## 12. Checkpoint & Resume

Each phase writes a completion marker to the artifact directory. On restart (e.g., after container crash), the kernel supervisor checks which phases completed and resumes from the next incomplete phase.

**Checkpoint markers**: `<artifact_dir>/phase_<N>_complete.json` — contains timestamp and summary.

**Phase restartability**:
- Phase 0 (setup): always re-run (idempotent)
- Phase 1 (discovery): always re-run (fast, read-only)
- Phase 2 (baseline): restartable — skip shapes already in `baseline.json`
- Phase 3 (untuned): restartable — skip shapes already in `untuned.json`
- Phase 4 (tuning): restartable — skip N,K pairs with existing `Screen complete` log files
- Phase 5 (validation): restart from scratch (configs may have changed)
- Phase 6 (commit): restart from scratch (human approval is stateless)

**Subagent retries**: on subagent failure, the kernel supervisor retries up to 2 times. Each retry re-runs the subagent from scratch with its original inputs. If the subagent had partial output (e.g., baseline for 50/100 shapes), the retry can read existing output and skip completed items.

---

## 13. Key Tool Reference

### screen.py (tuning)
```bash
python screen.py <M> <N> <K> <GPU_ID> <ut_script.py> \
    --block-size-m-range <BM values...> \
    --block-size-n-range <BN values...> \
    --block-size-k-range <BK values...> \
    --num-stages-range <min> <max> \
    --matrix-instr-nonkdim-range <values...> \
    --num-warps-range <values...> \
    --waves-per-eu-range <values...> \
    --cache-modifier-range <values...> \
    --num-ksplit-range <values...> \
    [--overwrite] [--verbose]
```
Output: `screen-<ut_script>-<M>-<N>-<K>.log` in current directory.
Env: `SCREEN_MAX_BATCH=<N>` to reduce batch size for large shapes (default 100).

### view-screen.py (config generation)
```bash
python view-screen.py <ut_script.py> --n-list <N values...> --k-list <K values...>
```
Output: `<gfx>-GEMM-<VARIANT>-N=<N>-K=<K>.json` in current directory.

### rocprof --stats (benchmarking)
```bash
rocprof --stats -o <unique_prefix>.csv python bench_gemm_<variant>.py \
    --shape <M> <N> <K> --metric time --layout TN [--shuffle]
```
Output: `<prefix>.stats.csv` — parse column 4 (AverageNs) for kernel rows matching the variant name. Use `rocprof` (v1), NOT `rocprofv3` — the latter has different output format.

---

## 14. Future Extensions

- Support non-GEMM kernels (attention, MoE, conv1d)
- Auto-create PRs after commit
- Persistent historical database of tuning results across upgrades
- Slack/email notifications
- Support for multi-node tuning (single kernel across machines)
- Container image caching (pre-built images with Triton installed)
