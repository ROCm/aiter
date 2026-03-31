# Agentic Kernel Tuning Pipeline — E2E Testing Guide

## Status
- **944 tests passing**, 14 xfailed (ScriptCreator mock alignment — not a code bug)
- **9 rounds of code review completed**, 34 issues found and fixed
- **Dry-run tested**: discovers 26 kernels across basic/batched/feed_forward/fused categories
- **Ready for e2e testing on real GPU machines**

## Quick Start

### 1. Run Tests
```bash
cd /app/aiter
python -m pytest aiter/ops/triton/utils/_triton/tunning/tuning_agent/ --tb=short -q
```

### 2. Dry Run (no GPU needed)
```bash
python -m aiter.ops.triton.utils._triton.tunning.tuning_agent \
    --config triton-upgrade-dryrun.yaml \
    --dry-run \
    --repo-root /app/aiter
```

### 3. E2E Test (requires GPU machine with SSH access)

Create a `triton-upgrade.yaml`:
```yaml
baseline:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: /app/triton_3_4          # local path or git URL
  triton_branch: main                    # branch or commit

target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: alizaidy/gfx950-kernel-fixes-cherry-picked
  triton_repo: /tmp/triton-latest        # local path or git URL
  triton_branch: main

machines:
  - host: <your-gpu-machine>
    user: root
    ssh_key: ~/.ssh/id_rsa
    gpus: [0, 1, 2, 3, 4, 5, 6, 7]

container:
  image: rocm/pytorch:latest
  # run_script: ./scripts/create_container.sh  # optional

gpu:
  arch: gfx950

triton_install:
  command: "pip install -e ."

tuning:
  mode: regression_only    # or "full"
  thresholds:
    regression_vs_baseline: 5
    regression_vs_untuned: 2

kernels:
  include: [a8w8]          # start with ONE kernel for e2e test
  exclude: []
```

Then run:
```bash
python -m aiter.ops.triton.utils._triton.tunning.tuning_agent \
    --config triton-upgrade.yaml \
    --repo-root /app/aiter
```

## Architecture

```
CLI (__main__.py)
  → Orchestrator (orchestrator.py) — discovers kernels, allocates machines
    → KernelSupervisor (kernel_supervisor.py) — phases 0-6 per kernel
      → Phase 0: SetupAgent — SSH, create container, clone repos, install Triton
      → Phase 1: DiscoveryAgent — find configs, shapes, scripts
      → Phase 2: BaselineAgent — rocprof --stats on old Triton
      → Phase 3: ValidationAgent — rocprof --stats on new Triton (untuned)
      → Phase 4: TuningAgent + PatternAnalyzer + ConfigGenerator
      → Phase 5: ValidationAgent + RegressionFixer (iterate up to 3x)
      → Phase 6: Human approval → git commit
```

All agents run **locally** and execute commands remotely via `ssh user@host "docker exec container bash -c '...'"`.

## Package Location
```
aiter/ops/triton/utils/_triton/tunning/tuning_agent/
├── __init__.py, __main__.py
├── types.py, config.py, remote.py, machine_pool.py
├── watchdog.py, notifications.py, artifacts.py
├── kernel_supervisor.py, kernel_discovery.py
├── orchestrator.py, dashboard.py
└── subagents/
    ├── base.py, setup_agent.py, discovery_agent.py
    ├── baseline_agent.py, validation_agent.py
    ├── tuning_agent.py, pattern_analyzer_agent.py
    ├── config_generator_agent.py, regression_fixer_agent.py
    └── script_creator_agent.py
```

## Key Design Decisions
- **regression_only mode** (default): only tune shapes that regressed vs baseline
- **Never modify default fallback config** — promote to suffixed config instead
- **Adaptive search**: scout broad → pattern analysis → tune narrow
- **rocprof --stats** (v1, NOT rocprofv3) for benchmarking
- **M_LEQ_31** bucket name for preshuffled afp4wfp4 (not M_LEQ_16)
- **Baseline sequential on 1 GPU**, validation can parallelize across GPUs
- **Human approval required** for git commits and regressions above threshold

## Known Limitations
- Orchestrator processes kernels sequentially (no parallel dispatch across machines yet)
- Dashboard doesn't clear terminal between refreshes
- 14 xfailed tests in ScriptCreator (mock alignment, not implementation bugs)
- ScriptCreator escalates to human when kernel API is complex

## Specs & Plans
- Design: `aiter/ops/triton/docs/2026-03-30-agentic-kernel-tuning-pipeline-design.md`
- Plan 1-4: `aiter/ops/triton/docs/2026-03-30-plan-{1,2,3,4}-*.md`
