# Sequential physical-GPU acceptance tasks

`run_gpu_task.py` is an acceptance controller used by the primary coordinator. Every invocation checks monitoring GPU7 UUID `0x5d8a72a332aa124c` and PCI `0000:f5:00.0`, waits for three idle samples, fixes `ROCR_VISIBLE_DEVICES=GPU-5d8a72a332aa124c` and `HIP_VISIBLE_DEVICES=0`, and starts one independent Python task. The child verifies HIP PCI identity plus gfx950/256 CUs before executing its command. The coordinator waits for the entire task to return before launching the next invocation. This script has no normal-task hard deadline.

The artifact directory records idle samples, exact command/environment/implementation commit, child runtime identity, stdout/stderr and process start/exit/return code. Use a fresh artifact path for each task so prior evidence remains available. Environment options are applied before child startup and remain fixed for the invocation. Configuration CSV choice remains the caller's responsibility.

```bash
/root/workspace/tmp_dev/mxfp4-port-env/bin/python \
  docs/mxfp4_staged_main_port/scripts/run_gpu_task.py \
  --artifact docs/mxfp4_staged_main_port/02/gpu/small-staged \
  --cwd /root/workspace/tmp_dev/aiter-mxfp4-02 \
  --set-env AITER_FLYDSL_STAGE2_FP8=0 \
  --set-env AITER_MXFP4_INTERMEDIATE=0 \
  --set-env AITER_MXFP4_TUNE_SWIGLU_LIMIT= \
  -- csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --mxfp4-flydsl --mxfp4-search-mode staged --mp 1 --all \
  -i <one-shape-input.csv> -o <separate-winner.csv>
```

Set the dedicated environment's PATH, exact CK_DIR, task-specific AITER_JIT_DIR and FLYDSL_RUNTIME_CACHE_DIR, and matching flags/limits before real tasks. The script's fixed visibility takes precedence. On cancellation the task's independent process group is stopped, including its spawned workers, and interruption/exit evidence is finalized. GPU use and VRAM/process allocation gate activity; discovery-only entries with unavailable AMD-SMI occupancy use the remaining activity/memory/status evidence.

Five offline behavioral tests passed. They cover identity, activity/allocation versus discovery, unavailable occupancy and cancellation of a harmless CPU descendant. Red and final logs are adjacent. These checks executed with empty GPU visibility and establish controller behavior only; they are not MXFP4 numerical or production/AOT acceptance.

`default_replay.py --csv <canonical-one-row-winner.csv> --result <result.json>`
performs explicit A4W4 public dispatch, checks the exact CSV G1/G2 names and
metadata, observes actual direct kernel arguments, and compares final output
with Torch at the configured default gate. Its synchronized elapsed and real
kernel trace span are companion metrics. They are labelled separately from the
main kernel-duration sum; synthetic GPU annotations do not enter the real
kernel event counts. The trace and data/routing fingerprints are retained.

Run `--action aot` first in a fresh `FLYDSL_RUNTIME_CACHE_DIR` to compile the
selected CSV's default jobs; every job must report compilation success. Use that
same AOT cache in a later independent replay process with
`FLYDSL_RUNTIME_RUN_ONLY=1`. This default small BM16 fixture uses the generated
C++ auxiliary sort (no FlyDSL input-prequant job); non-atomic G2 reduction is
compiled by the AOT stage2 entry. Expanded configurations/auxiliary builders are
validated by their separate acceptance tasks. The controller launches every
real replay/AOT task after its idle gate and waits for exit.
