# Ticket 01 incremental read-only review

Scope: current ticket 01 implementation in `/root/workspace/tmp_dev/aiter-mxfp4-01`, against fixed base `882ea79dd7cc7091f859a5bedac914523ac77bbd` and current `.scratch/mxfp4-staged-main-port/spec.md` / `issues/01-safe-tuning-lifecycle.md`. Followed karpathy-guidelines. No code edits, test launches, GPU operations or tracker edits were performed by this reviewer. This is an incremental review, not the final Standards/Spec review.

## Actionable findings and disposition

| Finding | Concrete failure and evidence | Current disposition |
| --- | --- | --- |
| Duplicate historical shape rows discard the retuned winner | `TunerCommon.update_tunedf` originally updated only the first old row of a duplicate key; both fMoE/MX writers subsequently kept the last row. Old timings 20/21 plus a new timing 10 therefore retained old 21. | Fixed in WIP: safely deduplicate old keys before updates, retaining tag distinctions. New real ordinary and MX lifecycle retune tests cover matching duplicate historical winners. Coordinator verification pending after final changes. |
| Same-file `--all` input without device identity fails | `get_retune_gemm_list` same-file branch filled gfx but indexed absent cu_num; the separate-file branch already supplied both. | Fixed in WIP: supply missing cu_num and preserve gate metadata. The new regression progressed to writing and exposed a related input-only-row persistence defect below. |
| Single-slot prune/full silently move to logical GPU0 | Fixed base direct execution used caller current_device; WIP initially forced GPU0 for identity/preflight/execution. Spec runtime section requires checking the existing scheduling path; only staged explicitly selects the first visible logical GPUs for one slot. | Root adjudicated and WIP fixed: prune/full one slot use current_device without set_device(0); multiple slots retain range(mp_num). Tests now include unsupported unused GPU0 with supported current GPU1, plus unsupported participating current GPU1 with supported unused GPU0. |
| Multi-file legacy CSV identity normalization occurs after key use | `update_config_files` deduplicated with gfx keys before normalization. All-legacy files lacked gfx and raised; legacy + modern filled legacy gfx with 0 before row inference. | Fixed in WIP: normalize each input identity before column filling and deduplication; use identity-only helper so profile/other-tuner columns remain intact. The new regression now validates the merged read and preserves the secondary file byte-for-byte. |

The coordinator's guarded run at `/root/workspace/tmp_dev/mxfp4-port-notes/isolated-01-lifecycle-final.txt` recorded **52 passed / 3 failed** (55 cases). Two failures were incorrect new test expectations that a non-all fMoE invocation copies all secondary-file rows into its primary output. Existing behavior writes the first file and leaves others unchanged; the spec does not authorize expanding this behavior. The WIP test now reads `get_tuned_gemm_list(first:second)` to verify merging and checks primary updates plus unchanged secondary bytes.

The remaining observed failure is a continuation of the same-file missing-identity case: the writer reloads an input-only CSV; normalization supplies cu_num=0/us=0, leaving a fake historical row alongside the real CU256 winner. Root and implementer have the evidence and proposed correction: input-only rows are not valid historical winners; retain old finite positive timed rows while excluding untuned pseudo-rows at the MX persistence boundary. At report time this last correction and the next guarded run remain pending. Do not mark ticket 01 accepted from the partial result.

## Controlled-test isolation review

Initial parent-instance measurement mocks did not propagate through spawn. Current test isolation is statically suitable for a fresh coordinator-run command with `ROCR_VISIBLE_DEVICES=''`, `HIP_VISIBLE_DEVICES=''` and `CUDA_VISIBLE_DEVICES=''`:

- MX autouse guards prohibit real BaseProcess.start, real shape supervision, class preparation/validation/e2e paths, ordinary data producers, performance timing, CUDA initialization/events/graphs and tensor allocation. Multi-slot tests provide synchronous fake supervisors and test-local fake measurements.
- Ordinary `prohibit_gpu_work()` is entered by both the parent autouse fixture and `controlled_environment()` inside each runpy command child. The subprocess explicitly receives empty visibility. Both canonical and module-bound measurement paths, workers, process starts, data producers, timing and allocation are guarded. `pytest.fail` escapes the tuner's catch(Exception) path.
- `_controlled_tuner_import.py` imports the real tuner/toolchain. Only import-time rocminfo/hipinfo subprocess calls return synthetic gfx950/CU256 metadata; other compilation subprocesses pass through. The helper forces CPU default device and clears cached discovery afterward. It does not patch runtime properties, current-device selection, identity or preflight. Those are still exercised against per-case simulated properties.

This is a static isolation judgment and applies to these controlled lifecycle files, not a claim of GPU numerical/production/AOT coverage. The early uncontrolled execution incident and incomplete raw-log retention remain explicitly documented in `docs/mxfp4_staged_main_port/01/incident.md` and must not be counted as acceptance.

No additional concrete in-scope issue was found in this incremental pass. Final branch review remains required after all four tickets.
