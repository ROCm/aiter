# Current implementation evidence

Implementation branch: `integration/mxfp4-staged-main-port`, worktree `/root/workspace/tmp_dev/aiter-mxfp4-integration`, fixed main base `882ea79dd7cc7091f859a5bedac914523ac77bbd`. The original source branch/worktree and existing experiments are preserved. The integration tree contains the current specification and four tickets under `.scratch/mxfp4-staged-main-port/`; only the primary coordinator writes their state and Answers.

## Environment and baseline

[toolchain.json](toolchain.json) records the new dedicated Python environment and package versions. It inherits the host ROCm 7.1 Torch/Triton packages, installs FlyDSL 0.3.4.1 from a retained wheel and adds only tools to this dedicated environment. The original `/opt/venv` and historical cross environment are unchanged. The initial baseline used the preserved original checkout for read-only CK headers. Integration now contains a separate detached CK worktree at fixed-base gitlink `96afae1d54fc79c26857de7e1683b0dde1ff3c6f`; the original CK checkout remains unchanged.

The host Torch 2.8 environment contains Triton 3.4. The repository-supported `AITER_USE_SYSTEM_TRITON=1` override permits imports with a version warning. Controlled lifecycle tests execute no GPU kernels; this warning does not establish numerical or deployment acceptance. Every required MXFP4 GPU path must be verified on the current integration implementation. Any actual unsupported API/path remains an acceptance failure until addressed.

Baseline shared infrastructure checks were run before ticket-01 code changes on documentation-only integration commit `559e83fa7d750204a9164cdf97777c830fa760ff`: 55 tests passed in 19.311 seconds. [baseline-infra.log.txt](baseline-infra.log.txt) retains output. These tests exercise shared infrastructure and simulated process logic; module import builds the CPU enum/core extension and requires ROCm/FlyDSL imports. No GPU kernel tuning or measurement was performed.

```bash
env PATH=/root/workspace/tmp_dev/mxfp4-port-env/bin:$PATH \
  AITER_USE_SYSTEM_TRITON=1 \
  CK_DIR=/root/workspace/tmp_dev/aiter/3rdparty/composable_kernel \
  AITER_JIT_DIR=/root/workspace/tmp_dev/mxfp4-port-notes/baseline-jit \
  python -m unittest op_tests.tuning_tests.test_tuner_infra \
    op_tests.tuning_tests.test_compare_logic \
    op_tests.tuning_tests.test_mp_tuner_logic -v
```

Physical target for all subsequent real tasks: `ROCR_VISIBLE_DEVICES=GPU-5d8a72a332aa124c`, `HIP_VISIBLE_DEVICES=0`, expected PCI `0000:f5:00.0`, gfx950/256 CUs. Monitoring-only exploration confirmed this identity and one idle sample; every real task needs its own fresh idle check and child PCI verification. GPU execution is coordinated sequentially by the primary agent.

Baseline results and environment preparation are not completion evidence for tickets 02–04 or the 27-search calibration. Ticket-specific records will identify implementation commits, command/configuration, measurement conditions, exit status, numerical, exact production and precompiled outcomes.

The same 55 baseline checks also passed with the precise fixed-base CK checkout in 18.210 seconds; [target-ck-baseline-infra.log.txt](target-ck-baseline-infra.log.txt) records this dependency-aligned baseline. Its command uses `CK_DIR=/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel` and `AITER_JIT_DIR=/root/workspace/tmp_dev/mxfp4-port-notes/target-ck-baseline-jit`; all other arguments match the initial baseline above.

Incremental read-only reviews are retained alongside this record. They are scoped snapshots, not the final required two-axis review. Ticket 01's concrete findings were fixed and its final isolated checks passed. Ticket 02 planning/policy/default execution found no outstanding code defect; its supervisor review requested deterministic exit-before-read coverage, subsequently supplied at `6d85dc80a` with the corrected fixture passing. Real numerical, production/AOT and complete-matrix acceptance remain separate gates.
