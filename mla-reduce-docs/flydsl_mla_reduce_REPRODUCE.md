# Reproducing the FlyDSL MLA decode-reduce benchmarks

How to set up the dev container, run the correctness matrix, and reproduce the
bandwidth numbers for the FlyDSL MLA decode-reduce kernel
(`aiter/ops/flydsl/kernels/mla_reduce.py`), a native port of the HIP
`kn_mla_reduce_v1` stage-2 split-KV decode reduce.

The bar is **HBM parity** with the HIP kernel, not a speedup — the reduction is
bandwidth-bound at the byte floor. Published numbers live in
`MLA-reduce-flydsl-benchmark-report.md` (FlyDSL) and
`MLA-reduce-flydsl-vs-hip-comparison.md` (head-to-head). HIP-only baseline:
`MLA-reduce-HIP-kernel-benchmark-report.md`.

**Last updated:** 2026-06-24 (post multi-token decode fix; partial inputs model
stage-1 fp32 outputs).

---

## 1. Hardware / software environment

Numbers in this doc were produced on:

| Component | Version / value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, CDNA3, **304 CUs**, warp 64), HBM3 ~5.3 TB/s peak |
| Host CPU / OS | Xeon Platinum 8480C / Ubuntu 22.04.5 LTS |
| Container image | `rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_027c46b_aiter_eee0abe_qwen3_5_20260529` |
| ROCm | 7.2.0 |
| Python | 3.10.12 |
| PyTorch | 2.9.1+rocm7.2.0 (HIP 7.2.26015) |
| **flydsl** | **0.2.0** |

The kernel is **gfx942-specific**. `flydsl` is not importable on a bare host
interpreter — all build/run happens inside the ROCm container below.

### Input model (stage-1 partials)

The reduce kernel always reads **fp32** partial `O` and `LSE` — the same buffers
the MLA stage-1 ASM kernel writes. Upstream Q/KV may be bf16 or fp8; stage-1 still
emits fp32 partials, so standalone reduce bandwidth is the same regardless of
upstream dtype (only the small final-`O` write scales with output dtype).

- **Standalone sweeps** (`test_flydsl_mla_reduce.py --bench`): synthetic fp32 fills
  in the `get_mla_metadata_v1` CSR layout (`build_inputs`).
- **End-to-end** (`test_mla_sparse.py` + `AITER_MLA_REDUCE_FLYDSL=1`): real
  stage-1 ASM → fp32 partials → reduce; covered by `stress_flydsl_mla_reduce.sh`.

---

## 2. Container setup

The repo is bind-mounted into a long-lived container off the project's own ROCm
image (no host torch / `libstdc++` requirement this way):

```bash
docker run -d --name mla_reduce_bench \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined --ipc=host \
  -v /home/anguyenh/aiter:/aiter -w /aiter \
  rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_027c46b_aiter_eee0abe_qwen3_5_20260529 \
  sleep infinity
```

If the container already exists, skip `docker run` and reuse it.

### Required environment variables

Set these for every command run inside the container:

| Var | Value | Why |
|---|---|---|
| `PYTHONPATH` | `/aiter` | import the bind-mounted repo |
| `AITER_JIT_DIR` | `/tmp/aiter_jit` | fresh JIT build/output dir (avoids stale prebuilt `.so`s) |
| `AITER_USE_SYSTEM_TRITON` | `1` | downgrades the triton-gluon version gate (≥3.6.0) to a warning — unrelated to this kernel |

> The FlyDSL test imports `aiter.ops.flydsl.kernels.mla_reduce` **directly**,
> bypassing aiter's flydsl `__init__` version gate, so it runs even though the
> import chain prints warnings about gfx950-only modules and the triton version.
> Those warnings are benign on gfx942.

### Shell helper (use for all steps below)

Define once on the **host**; all subsequent commands assume it:

```bash
RUN='docker exec \
  -e PYTHONPATH=/aiter \
  -e AITER_JIT_DIR=/tmp/aiter_jit \
  -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench'
```

### Verify the environment

```bash
$RUN bash -lc '
  python3 -c "import torch, flydsl; \
    print(\"torch\", torch.__version__); \
    print(\"flydsl\", flydsl.__version__); \
    print(\"device\", torch.cuda.get_device_name(0)); \
    print(\"CUs\", torch.cuda.get_device_properties(0).multi_processor_count)"'
# expect: torch 2.9.1+rocm7.2..., flydsl 0.2.0, AMD Instinct MI300X, CUs 304
```

First run of a given kernel tier JIT-compiles FlyDSL (~few seconds) and/or HIP
(`module_mla_reduce` ~193 s on a fresh `AITER_JIT_DIR`). Benchmark numbers come
from the steady-state loop after compilation.

---

## 3. Correctness

Reference: **HIP `kn_mla_reduce_v1`** on the same input buffers (direct
kernel-vs-kernel). A vectorized fp64 torch online-softmax (`torch_ref`) is kept
for debugging.

### 3.1 Full matrix (156 cases)

3 shapes × {bf16, fp16} × 8 split counts × `M = max_seqlen_q ∈ {1,2,4}` plus
12 degenerate empty-tile guard cases (qlen>1 regression):

```bash
$RUN python3 op_tests/test_flydsl_mla_reduce.py --matrix
```

> **bf16 tolerance:** both kernels accumulate in fp32 and round to bf16
> independently, so near-equal fp32 values can land on adjacent bf16 codes
> (2⁻⁵ ≈ 3.1e-2). LSE (fp32) still matches to ~1e-7.

### 3.2 Degenerate guard only (12 cases)

Fast tripwire for the empty-tile guard (`n_splits = 0`, garbage `reduce_final_map`):

```bash
$RUN python3 op_tests/test_flydsl_mla_reduce.py --degenerate
```

Expected: all `[PASS]`, exit 0.

### 3.3 Single config spot-check

```bash
# well-formed tile, LSE check
$RUN python3 op_tests/test_flydsl_mla_reduce.py \
  --H 128 --Dv 512 --splits 8 --tiles 4 --lse

# multi-token row layout (M > 1; forward-correct, not production metadata shape)
$RUN python3 op_tests/test_flydsl_mla_reduce.py \
  --H 128 --Dv 512 --splits 8 --tiles 4 --M 2 --lse
```

---

## 4. Bandwidth benchmark (FlyDSL standalone)

Harness: `op_tests/test_flydsl_mla_reduce.py --bench`, 25 warmup / 100 timed iters.

### Kernel-only timing

`--bench` reports **kernel-only GPU device time**, not wall clock:

- **`kernel=`** — aiter's `run_perftest` (`self_device_time_total`, IQR-filtered)
- **`graph=`** — CUDA-graph `cuda.Event` replay cross-check

Both exclude per-call Python host overhead (~230 µs DLPack/launcher in the old
wall-clock harness). BW uses the traffic model:
`T·S·H·Dv·4` (partial-O) + LSE reads + final-O/LSE writes.

Each line prints: `kernel=…us graph=…us BW=…GB/s (…% of 5.3TB/s)`.

### 4.1 Headline saturating points

Matches `MLA-reduce-flydsl-benchmark-report.md` §3.2 / §3.3:

```bash
$RUN bash -lc '
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits  8 --tiles 4096 --bench 2>/dev/null | grep bench
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits 16 --tiles 2048 --bench 2>/dev/null | grep bench'
```

### 4.2 Split-count sweep (tiles=256, H=128, Dv=512, bf16)

Reproduces `MLA-reduce-flydsl-benchmark-report.md` §3.1 and the comparison
report §3 split table (FlyDSL column):

```bash
$RUN bash -lc '
  for S in 2 3 4 8 16 32 64 128; do
    python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --tiles 256 --splits $S --bench 2>/dev/null | grep bench
  done
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --tiles 64 --splits 256 --bench 2>/dev/null | grep bench'
```

### 4.3 Batch (tile) sweep (splits=8, H=128, Dv=512, bf16)

Reproduces `MLA-reduce-flydsl-benchmark-report.md` §3.2:

```bash
$RUN bash -lc '
  for T in 1 4 16 64 128 256 512 1024 2048 4096; do
    python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --tiles $T --splits 8 --bench 2>/dev/null | grep bench
  done'
```

### 4.4 Other shapes / output dtypes

```bash
$RUN bash -lc '
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --tiles 4096 --splits 8 --dtype fp16 --bench 2>/dev/null | grep bench
  python3 op_tests/test_flydsl_mla_reduce.py --H  16 --Dv 512 --tiles 4096 --splits 8 --bench 2>/dev/null | grep bench
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 128 --tiles 8192 --splits 8 --bench 2>/dev/null | grep bench'
```

Partials remain fp32 (stage-1 layout); bf16 vs fp16 differs only in the final-O write.

---

## 5. Cross-check against the HIP kernel

Apples-to-apples comparison uses the same container and traffic model.
`bench_mla_reduce_standalone.py` times HIP `kn_mla_reduce_v1` with the same
`run_perftest` + CUDA-graph methodology.

### 5.1 HIP correctness + single point

```bash
# one-time on fresh AITER_JIT_DIR: module_aiter_core ~20 s, module_mla_reduce ~193 s
$RUN python3 op_tests/bench_mla_reduce_standalone.py \
  --H 128 --Dv 512 --tiles 256 --splits 8 --check
```

### 5.2 HIP split-count sweep (comparison report §3)

```bash
$RUN bash -lc '
  for S in 2 3 4 8 16 32 64 128; do
    python3 op_tests/bench_mla_reduce_standalone.py --H 128 --Dv 512 --tiles 256 --splits $S
  done
  python3 op_tests/bench_mla_reduce_standalone.py --H 128 --Dv 512 --tiles 64 --splits 256'
```

Each line prints `kernel = … us (graph … us) | … | achieved BW = … GB/s`.

### 5.3 HIP tile sweep (comparison report §4)

```bash
$RUN bash -lc '
  for T in 1 4 16 64 128 256 512 1024; do
    python3 op_tests/bench_mla_reduce_standalone.py --H 128 --Dv 512 --tiles $T --splits 8
  done'
```

### 5.4 Optional: HIP fp8 final-output dtype

Stage-1 partials are still fp32; `--dtype fp8` changes only the final-O write size
in the traffic model:

```bash
$RUN python3 op_tests/bench_mla_reduce_standalone.py \
  --H 128 --Dv 512 --tiles 4096 --splits 8 --dtype fp8
```

---

## 6. End-to-end regression (real stage-1 input, bf16/fp8, qlen>1)

Production path: sparse MLA decode runs stage-1 ASM, then reduce. With
`AITER_MLA_REDUCE_FLYDSL=1` the reduce slot uses the FlyDSL port.

### 6.1 Full stress matrix (72 runs, pass/fail gate)

From the repo root on the **host** (script `docker exec`s into the container):

```bash
bash mla-reduce-docs/stress_flydsl_mla_reduce.sh
```

Expected: every config **`0 fault`**, exit 0. Default matrix:

| config | repeats |
|---|---|
| bf16 qlen=2 | 20 |
| fp8 qlen=2 | 20 |
| bf16 qlen=3 / qlen=4 | 8 each |
| bf16 qlen=2 batch=4 / ctx=256 | 8 each |

Quicker smoke (override repeats):

```bash
REPEATS=3 bash mla-reduce-docs/stress_flydsl_mla_reduce.sh
```

### 6.2 Single e2e spot-check (manual)

```bash
$RUN bash -lc '
  AITER_MLA_REDUCE_FLYDSL=1 \
  python3 op_tests/test_mla_sparse.py -n16,2 -b 1 -c 21 -k 512 -d bf16 -kvd bf16'
# expect: clean run, decode:err = 0

$RUN bash -lc '
  AITER_MLA_REDUCE_FLYDSL=1 \
  python3 op_tests/test_mla_sparse.py -n16,2 -b 1 -c 21 -k 512 -d fp8 -kvd fp8'
# expect: clean run; decode:err ~0.36 (fp8 quant delta vs bf16 golden, not a reduce bug)
```

Reduce-only device time is **not** separated in the e2e harness (stage-1
dominates total latency at this shape). Standalone §4 numbers are the authoritative
reduce kernel BW.

Details: `MLA-reduce-flydsl-multitoken-decode-fix-report.md`.

---

## 7. Quick reference — artifacts

| File | Purpose |
|---|---|
| `aiter/ops/flydsl/kernels/mla_reduce.py` | FlyDSL kernel (`compile_mla_reduce`, `select_tier`, empty-tile guard) |
| `aiter/ops/flydsl/mla_reduce_kernels.py` | Production wrapper (`flydsl_mla_reduce_v1`; opt-in via `AITER_MLA_REDUCE_FLYDSL=1`) |
| `op_tests/test_flydsl_mla_reduce.py` | Correctness matrix (`--matrix`, `--degenerate`, `--M`) + `--bench` |
| `op_tests/bench_mla_reduce_standalone.py` | HIP `kn_mla_reduce_v1` bench (baseline / comparison) |
| `op_tests/prof_mla_reduce.py` | Fixed-launch driver for rocprofv3 |
| `op_tests/test_mla_sparse.py` | End-to-end sparse MLA decode (stage-1 + reduce) |
| `mla-reduce-docs/stress_flydsl_mla_reduce.sh` | E2e multi-token regression gate (bf16/fp8, qlen 2–4) |
| `mla-reduce-docs/MLA-reduce-flydsl-benchmark-report.md` | Published FlyDSL BW tables |
| `mla-reduce-docs/MLA-reduce-flydsl-vs-hip-comparison.md` | Head-to-head FlyDSL vs HIP |
| `mla-reduce-docs/MLA-reduce-flydsl-multitoken-decode-fix-report.md` | qlen>1 fix + regression design |
