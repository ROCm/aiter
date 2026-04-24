# Opus GEMM (C++ side)

The user-facing documentation for the opus a16w16 GEMM lives at
[**aiter/ops/opus/README.md**](../../aiter/ops/opus/README.md). It
covers Quick Start, dispatch architecture, tuning workflow, env vars,
testing, internals, and troubleshooting.

This directory holds the C++ / JIT build inputs only:

| File | Role |
|---|---|
| `opus_gemm.cu` | Runtime dispatcher (lookup + heuristic) and pybind entries for `opus_gemm()` / `opus_gemm_a16w16_tune()` |
| `opus_gemm_common.py` | Kernel instance metadata — all kids (a16w16 split-barrier, flatmm, flatmm_splitk) live here |
| `gen_instances.py` | JIT codegen driver; `--tune_file` bakes the tuned CSV into `opus_gemm_lookup.h` |
| `opus_gemm_tune.py` | Offline tuner CLI (see `aiter/ops/opus/README.md` §3 for usage) |
| `include/pipeline/*.cuh` | Kernel source (a16w16, flatmm, flatmm_splitk pipelines + traits) |
| `include/opus_gemm*.{h,cuh}` | Shared C++ headers |
