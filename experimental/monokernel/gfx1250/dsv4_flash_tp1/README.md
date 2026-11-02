# DeepSeek V4 Flash TP1 gfx1250 monokernel

Single GPU. Selected C1/C2/C4/C8 implementations fuse attention and MoE per decode layer across 43 layers; prefill remains native. C16 has full-layer experimental implementations and isolated native-design projection experiments, but no model-wide serving qualification in this snapshot. Native-model accuracy and mixed-batch text parity remain unqualified; this is opt-in research code.

## Selected entry points

| Scope | Source / runner | Status |
|---|---|---|
| C1–C8 full layer | `work/integrated-c8/model_layer_batch.py`, `runtime_batch.py` | Selected serving implementation |
| C1–C8 recipe and E2E results | `work/rollout-c1248/run-selected.sh`, `REPORT.md`, `E2E.json`, `VALIDATION.md` | Latest reported serving comparison |
| C16 full layer | `work/c16-native-projections/model_layer_batch.py`, `advantage_set.env`, `run.sh` | Selected full-layer microbenchmark implementation; attention + MoE |
| C16 latest native parity | `work/c16-native-parity/run_native_parity.sh`, `REPORT.md` | Isolated projection experiments; single-launch regressions not promoted |
| Earlier full layers / MoE | `work/full-layer-c1/`, `full-layer-native/`, `batch-full/`, `ffn-c1/`, `ffn-c64/` | Retained development stages |
| Region optimizations | `work/qkv-*/`, `mhc-*/`, `attention-*/`, `c8-*/`, `latency-overlap/` | Ablations, variants and reference comparisons |
| Original hybrid / FP4 MoE prototypes | `work/flash/` | Historical prototypes, including rejected variants |

## Commands inside the recorded image

```bash
# CPU only, from this branch's experiment directory:
python3 verify_sources.py
# Server: choose baseline OR fused; use one server lifetime per configuration.
HIP_VISIBLE_DEVICES=0 bash /work/rollout-c1248/run-selected.sh fused
# Separate shell: runs two clients, reports stored by the script.
HTTP_PORT=8000 bash /work/rollout-c1248/run-selected.sh client my-fused 8
# C16 full-layer synthetic microbenchmarks:
HIP_VISIBLE_DEVICES=0 bash /work/c16-native-projections/run.sh swa
HIP_VISIBLE_DEVICES=1 bash /work/c16-native-projections/run.sh hca
HIP_VISIBLE_DEVICES=2 bash /work/c16-native-projections/run.sh csa
# Latest isolated native-parity experiment:
HIP_VISIBLE_DEVICES=3 bash /work/c16-native-parity/run_native_parity.sh advantage q 5
```

The scripts above are GPU commands except `verify_sources.py`; run only on reserved devices. C16 microbenchmark correctness is not a serving-quality claim. Follow each experiment's own environment file: polling modes differ across generations and must not be interchanged.

These branches preserve the kernels and supporting experiments from the October 21, 2026 Central-time handoff. They do not enable experimental dispatch in AITER's installed public API. Run the recorded implementations in their validated runtime before porting them to a newer AITER/Triton version.

## Source and coverage

- `work/`: authored kernels, variants, ablations, helpers, benchmark/validation scripts, configuration and compact result files. Directory names retain their original `/work` layout, including frozen previous versions and rejected experiments.
- `runtime/`: matching AITER mHC implementation/public wrapper and ATOM hybrid installer, dispatch, configs and tests.
- `KERNEL_INDEX.md` / `KERNEL_INDEX.json`: every retained JIT symbol, with source locations.
- `SOURCE_MANIFEST.json`: original archive member and SHA256 for every retained source/artifact. Source files are copied without arithmetic or synchronization changes.
- `verify_sources.py`: CPU-only source hashes, Python syntax, and shell syntax verification.

Machine-generated compiler caches, binaries, tensor dumps, bulky per-request results, model weights, unrelated upstream checkouts and internal session notes are omitted. The full evidence remains in the image and original archives. Historical reports retain their original statements; in particular, "no image/Git push" describes the experiment at the time of that report, not the current handoff.

## Reproduction environment

Image: `rocm/fw-bringup:dsv4_tp4_mk_expts`

Immutable image digest: `sha256:d92055c8341c1b766fffa30101c4297083dc92d1948082bfd37203ea80c80e19`.

Recorded runtime: ATOM `f19778df4054e94d64d6f6bf6dd68f5681bd5945` plus saved local integration; AITER `7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe` plus saved local changes. This branch is based on AITER main `b6c376cfa65deba7a1a26656168f13fd3ba402c3`, but GPU execution against that newer main has not been qualified.

Inside a disposable container from that image, copy this directory's `work/.` over `/work/` to use the branch sources while preserving archived fixtures. Alternatively, bind this `work` directory to `/work` for source-only runs; tests that require saved tensors need the image's fixtures. Mount model checkpoints under `/data`. Historical scripts may contain the original container/host paths; the commands below run directly inside the container.

Do not overwrite the image's AITER installation with current main and assume old results still apply. The `runtime/` copies document the modified operator dependencies; they are not an unconditional installer.

For serving comparisons: start each server once, run the client twice, and report only the second run. No GPU work or new performance measurement was performed while publishing these branches.
