# Curated pytest manifests

Each `.txt` file here is a manifest for `.github/scripts/run_pytest_manifest.py`:
one pytest selector per line, optionally followed by the exact number of nodes
it must collect. The runner runs each test file in its own pytest process and
fails unless every selector collects its count and every node passes setup,
call, and teardown with no skip, xfail, or xpass. See the runner's docstring
for the full contract and its artifacts.

```bash
python3 .github/scripts/run_pytest_manifest.py \
  --manifest .github/test-manifests/mi450-gfx1250-p0.txt \
  --require-gfx gfx1250 \
  --artifact-dir test-reports/mi450-gfx1250/p0
```

`--collect-only` checks collection without running tests. `--expect-failure`
inverts the per-node rule for a manifest of known bugs. The selftest
(`.github/scripts/run_pytest_manifest_selftest.py`) needs no GPU.

When a count stops matching, review what upstream added, removed, or changed
in that test before editing the count. Never edit a count, or drop a node, to
turn a run green.

## MI450 (gfx1250) manifests

| Manifest | Nodes | Contents |
| --- | ---: | --- |
| `mi450-gfx1250-p0.txt` | 23 | A0 and architecture guards, gfx1250 code-object loading and dispatch, A16W16 contracts, MoE routing, DSV4 fused QK norm/RoPE |
| `mi450-gfx1250-p1.txt` | 260 | Triton/Gluon MLA prefill and decode, sparse paged-attention prefill and decode, MXFP4 GEMM |
| `mi450-gfx1250-known-failures.txt` | 11 | Filed gfx1250 bugs (#7536, #7777, #7778, #7779), run with `--expect-failure` so a fix is noticed |

`.github/workflows/mi450-test.yaml` runs all three nightly on a one-GPU MI450
runner once the `AITER_MI450_RUNNER` and `AITER_MI450_IMAGE` repository
variables are set.

### Validation

On 2026-11-11 the workflow's steps ran on one MI455X A0 (`gfx1250`, PCI
revision `0x00`) at ROCm/aiter `1350defcf`, from a fresh clone, with the ROCm
10.2 image
`rocm/ufb-private:pytorch-mi450-release-2.11-py3.12-rocm10.2.0a20260922-bkc.20260929`
(Torch 2.13) plus a pretranslation layer, and Triton
`3.8.0+amd.rocm7.2.0.git7cb7b059` from `install_triton.sh`: P0 23/23 passed
(179 s), P1 260/260 passed (420 s), and all 11 known failures still failed as
filed (28 s). Install took under five minutes. In two earlier runs P1 and the
known failures gave the same result; those runs are how `test_mla_flydsl`
(#7781) and the grouped GEMM tests (#7536) left P0.

The same selections have also run as a private nightly on that machine since
2026-11-05. P1 passed 277/277 in three consecutive first-attempt runs; 260 of
those nodes are here, and the other 17 need #7780.

### Runtime requirements

The ROCm 10 MI450 images need three things the MI300/MI350 lanes do not:

- **B0-to-A0 pretranslation in the image.** ROCm 10 translates gfx1250 B0
  code objects for A0 hardware when they load and keeps the result only for the
  process, so a fresh container spends over ten minutes translating Torch and
  hipBLASLt before its first GEMM. Build the image with ROCm's
  `rocjitsu-pretranslate` (`rj_pretranslate`) over the SDK packages. The
  runtime hook ignores a translation store that is a symlink, group-writable,
  or owned by another non-root user, and the `_rocm_sdk_core` and
  `_rocm_sdk_devel` prefixes each read their own store (a hard-linked copy
  works).
- **AITER's Triton.** The image's own Triton (`3.8.0+git669b31ac`) lacks Gluon
  APIs that AITER uses on gfx1250 (`gl.amd.cdna5.scaled_downcast`); 55 MXFP4
  cases fail to compile with it. `install_triton.sh` installs the pinned build.
- **`ROCM_PATH` on `_rocm_sdk_devel`.** The image points `ROCM_PATH` at
  `_rocm_sdk_core`, a runtime-only prefix: JIT builds there find no rocPRIM,
  and FlyDSL finds no `ld.lld` or device bitcode (ROCm/FlyDSL#946). The
  workflow points `ROCM_PATH`, `ROCM_HOME`, `HIP_PATH`, and `ROCM_SOURCE_DIR`
  at the complete `_rocm_sdk_devel` tree.

Runner cautions:

- Some provisioning loads amdgpu with `gpu_recovery=0`. A kernel that never
  finishes then holds the GPU until the host is power-cycled; `amd-smi reset`
  and a reboot hang. That is why `test_unified_attention.py` (#7776) is not
  selected.
- A GPU page fault aborts the whole pytest process. `test_mla_flydsl` does
  this intermittently (#7781), so it is not selected; the runner's
  one-process-per-file isolation keeps such a crash from hiding other files'
  results.
