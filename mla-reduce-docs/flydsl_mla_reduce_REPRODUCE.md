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

Expected: **48 passed, 0 failed**. Per-config output max_abs_err ≤ 3.9e-3 (bf16) /
4.9e-4 (fp16); LSE max_abs_err ≤ 9.5e-7 (exact, matching the HIP kernel).

Single config (with LSE check):

```bash
docker exec -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench \
  python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits 8 --tiles 4 --lse
```

The reference is a vectorized fp64 torch online-softmax (`torch_ref` in the test).

> **Note on a large-batch reference artifact:** at very large tile counts with the
> default random LSE spread, bf16 output error can spike (e.g. ~7e0). This is a
> harness/bf16 artifact, **not a kernel bug** — the HIP kernel produces the
> identical error on the same input (near-degenerate softmax in bf16). The
> `--matrix` sweep uses `tiles=4`, which avoids this.

---

## 4. Bandwidth benchmark

The first invocation of a given `(shape, tier)` JIT-compiles the kernel (a few
seconds); benchmark numbers come from the steady-state timed loop (25 warmup /
100 iters, CUDA events).

### Saturating batch (HBM parity — the headline number)

Small batches are **launch-overhead / latency bound** and will report low BW
(the kernel time is dwarfed by per-call host work). To measure achieved HBM
bandwidth, use a large tile count so kernel time dominates:

```bash
docker exec -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit -e AITER_USE_SYSTEM_TRITON=1 \
  -w /aiter mla_reduce_bench bash -lc '
    python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits 16 --tiles 2048 --bench 2>/dev/null | grep bench
    python3 op_tests/test_flydsl_mla_reduce.py --H 128 --Dv 512 --splits  8 --tiles 4096 --bench 2>/dev/null | grep bench'
```

Reference result (this machine):

| shape | splits | tiles | path | latency | achieved BW |
|---|---|---|---|---|---|
| H=128,Dv=512 | 16 | 2048 | massive | ~2517 µs | **3527 GB/s (67%)** |
| H=128,Dv=512 | 8 | 4096 | massive | ~2492 µs | **3670 GB/s (69%)** |

This matches/slightly exceeds the HIP baseline in the same container at the
comparable point (HIP T2048/S16 ≈ 3334 GB/s). The HIP report's split-sweep band
is 3.24–3.51 TB/s (61–74%); FlyDSL lands in/above that band → **HBM parity
confirmed**.

### Why small configs report low BW

The traffic model and BW math live in the test (`--bench`). At small tiles the
denominator (kernel time) is tiny while fixed per-call Python launch overhead
dominates, so reported BW is misleadingly low — this is a measurement artifact of
the standalone harness, not the kernel. Always benchmark at saturating batch
(tiles ≳ 2048) for the BW figure.

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
