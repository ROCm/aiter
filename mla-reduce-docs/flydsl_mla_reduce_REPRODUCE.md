# Reproducing the FlyDSL MLA decode-reduce benchmarks

How to set up the dev container, run the correctness matrix, and reproduce the
bandwidth numbers for the FlyDSL MLA decode-reduce kernel
(`aiter/ops/flydsl/kernels/mla_reduce.py`), a native port of the HIP
`kn_mla_reduce_v1` stage-2 split-KV decode reduce.

The bar is **HBM parity** with the HIP kernel, not a speedup — the reduction is
bandwidth-bound at the byte floor. See `MLA-reduce-HIP-kernel-benchmark-report.md`
for the HIP baseline this is measured against.

---

## 1. Hardware / software environment

Numbers in this doc were produced on:

| Component | Version / value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, CDNA3, **256 CUs**, warp 64), HBM3 ~5.3 TB/s peak |
| Host CPU / OS | Xeon Platinum 8480C / Ubuntu 22.04.5 LTS |
| Container image | `rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_027c46b_aiter_eee0abe_qwen3_5_20260529` |
| ROCm | 7.2.0 |
| Python | 3.10.12 |
| PyTorch | 2.9.1+rocm7.2.0 (HIP 7.2.26015) |
| **flydsl** | **0.2.0** (matches the repo's `[FLYDSL] bump version to 0.2.0` HEAD) |

The kernel is **gfx942-specific**. `flydsl` is not importable on a bare host
interpreter — all build/run happens inside the ROCm container below.

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

flydsl 0.2.0 installed

### Required environment variables

Set these for every command run inside the container:

| Var | Value | Why |
|---|---|---|
| `PYTHONPATH` | `/aiter` | import the bind-mounted repo |
| `AITER_JIT_DIR` | `/tmp/aiter_jit` | fresh JIT build/output dir (avoids stale prebuilt `.so`s) |
| `AITER_USE_SYSTEM_TRITON` | `1` | downgrades the triton-gluon version gate (≥3.6.0) to a warning — unrelated to this kernel |

> The FlyDSL test imports `aiter.ops.flydsl.kernels.mla_reduce` and
> `aiter.ops.flydsl.moe_kernels` **directly**, bypassing aiter's flydsl
> `__init__` version gate, so it runs even though the import chain prints
> warnings about gfx950-only modules and the triton version. Those warnings are
> benign on gfx942.

### Verify the environment

```bash
docker exec mla_reduce_bench bash -lc '
  python3 -c "import torch, flydsl; \
    print(\"torch\", torch.__version__); \
    print(\"flydsl\", flydsl.__version__); \
    print(\"device\", torch.cuda.get_device_name(0)); \
    print(\"CUs\", torch.cuda.get_device_properties(0).multi_processor_count)"'
# expect: torch 2.9.1+rocm7.2..., flydsl 0.2.0, AMD Instinct MI300X, CUs 304
```

---

## 3. Correctness

Full sweep — 3 shapes × {bf16, fp16} × 8 split counts (spans all four tiers
simple/m64/m256/mlds):

```bash
docker exec \
  -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench \
  python3 op_tests/test_flydsl_mla_reduce.py --matrix
```

Expected: **48 passed, 0 failed**. Per-config output max_abs_err ≤ 3.12e-2 (bf16,
one bf16 ULP) / 1.95e-3 (fp16); LSE max_abs_err ≤ 9.5e-7.

Single config (with LSE check):

```bash
docker exec -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench \
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits 8 --tiles 4 --lse
```

The reference is the **HIP `kn_mla_reduce_v1` kernel** (`hip_ref` in the test), run on
the same input buffers — a direct kernel-vs-kernel equivalence check. A vectorized fp64
torch online-softmax (`torch_ref`) is also kept in the test for debugging.

> **Why the bf16 tolerance is one ULP:** both kernels accumulate in fp32 and round the
> final output to bf16 independently, so near-equal fp32 results can land on adjacent bf16
> codes — a 2⁻⁵ ≈ 3.1e-2 difference that is rounding, not a behavioral gap. LSE (fp32) still
> matches to ~1e-7. The tolerance is set to one ULP of the output dtype.

---

## 4. Bandwidth benchmark

The first invocation of a given `(shape, tier)` JIT-compiles the kernel (a few
seconds); benchmark numbers come from the steady-state timed loop (25 warmup /
100 iters, CUDA events).

### Kernel-only timing

`--bench` reports **kernel-only GPU device time**, not wall clock. The headline
`kernel=` number is aiter's profiler-based `run_perftest`
(`aiter/test_common.py` → `self_device_time_total`, IQR-filtered); the `graph=`
number is an independent CUDA-graph `cuda.Event` replay. Both exclude the per-call
Python host overhead (the standalone driver's ~230 µs DLPack/launcher cost), so
BW is meaningful at every work size — there is no host-floor caveat to work around.

### Saturating batch (HBM parity — the headline number)

```bash
docker exec -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench bash -lc '
    python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits 16 --tiles 2048 --bench 2>/dev/null | grep bench
    python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits  8 --tiles 4096 --bench 2>/dev/null | grep bench'
```

Reference result (this machine):

| shape | splits | tiles | path | kernel | graph | achieved BW |
|---|---|---|---|---|---|---|
| H=128,Dv=512 | 16 | 2048 | massive | ~2414 µs | ~2438 µs | **3677 GB/s (69%)** |
| H=128,Dv=512 | 8 | 4096 | massive | ~2448 µs | ~2496 µs | **3736 GB/s (70%)** |

This matches/slightly exceeds the HIP baseline in the same container at the
comparable point (HIP T2048/S16 ≈ 3422 GB/s, device time). The HIP report's
split-sweep band is 3.10–3.86 TB/s; FlyDSL lands in/above that band → **HBM parity
confirmed**.

### Low-work configs now read the true kernel BW

With device-time measurement, small batches are only **latency-bound at very low
tile counts** (1–4 tiles = 128–512 work items, far below the machine's WG slots).
From tiles≈16 up the kernel already shows high BW. The simple path (splits 2–3)
reaches 87–91%. There is no longer a flat ~230 µs harness floor to discount, so
every row reports the kernel, not the driver.

---

## 5. Cross-check against the HIP kernel (optional)

To reproduce the HIP baseline in the same container (the apples-to-apples
comparison), the HIP path needs its JIT module built and the flydsl import gate
relaxed (the HIP test imports `aiter` top-level, unlike the FlyDSL test):

```bash
# inside the container, one-time per fresh AITER_JIT_DIR:
#   module_aiter_core ~20 s, module_mla_reduce ~193 s to build
docker exec -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench \
  python3 op_tests/bench_mla_reduce_standalone.py --H 128 --Dv 512 --tiles 2048 --splits 16 --check
```

See `MLA-reduce-HIP-kernel-benchmark-report.md` for the full HIP split-sweep and
batch-sweep tables and the rocprofv3 profile (byte-floor confirmation).

---

## 6. Quick reference — what each artifact is

| File | Purpose |
|---|---|
| `aiter/ops/flydsl/kernels/mla_reduce.py` | the FlyDSL kernel (`compile_mla_reduce`, `select_tier`) |
| `op_tests/test_flydsl_mla_reduce.py` | FlyDSL correctness matrix + `--bench` |
| `op_tests/bench_mla_reduce_standalone.py` | HIP `kn_mla_reduce_v1` bench (baseline) |
| `op_tests/prof_mla_reduce.py` | fixed-launch driver for rocprofv3 |
