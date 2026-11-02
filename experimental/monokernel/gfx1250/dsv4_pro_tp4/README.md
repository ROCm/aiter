# DeepSeek V4 Pro TP4 gfx1250 hybrid kernels

Four GPUs, tensor parallelism 4, EP off. This is regional hybrid fusion; MoE remains native. It is not a full attention-and-MoE layer monokernel.

## Entry points and experiments

| Scope | Sources | Status |
|---|---|---|
| mHC post/pre + RMSNorm + FP8 quant epilogue | `work/candidate/mhc_kernel.py`, `mhc_wrapper.py`; matching `runtime/app/aiter/aiter/ops/triton/` copies | Core hybrid implementation |
| Projection dispatch | `work/candidate/dsv4_hybrid.py` | Historical experiment; split-K variant has a documented correctness failure |
| Corrected ATOM hybrid integration | `runtime/opt/atom-gfx1250-integration/atom/models/deepseek_v4_mono/` | Pro uses unsplit projection; see config and `linear.py` |
| L2 prefetch and synchronization probes | `work/round2/` | Semaphore, cache-control, and collective experiments; not all were selected |
| E2E rerun evidence | `work/tp4-c16-rerun-20261019/` | Historical second-client rerun |
| Integration evidence | `work/gfx1250-integration/REPORT.txt`, `LEGACY_TP4_CORRECTNESS_UPDATE.txt` | Read before using old timings |

The original split-K QKV path differed on a real decode input and was superseded by the corrected unsplit integration. Do not treat historical split-K timings as measurements of the corrected implementation. Native MoE is retained throughout this branch.

## Commands inside the recorded image

```bash
# CPU only, from this branch's experiment directory:
python3 verify_sources.py
# Corrected regional integration already installed in the image:
bash /opt/atom-gfx1250-integration/scripts/serve_dsv4_mono_gfx1250.sh pro
```

For the historical launcher and measurements, consult `work/gfx1250-integration/REPRODUCE.txt` and `work/REPORT.txt`. The separate `runtime/` subtree records the matching sources; it is not part of AITER's default package imports. Pro model checkpoints must be available at `/data/DeepSeek-V4-Pro`.

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
