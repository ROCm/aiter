# `kn_mla_reduce_v1` — compile / benchmark / profile report

**Target:** AMD Instinct MI300X (gfx942, CDNA3, 304 CUs, warp 64), ROCm 7.2.
**Kernel:** `kn_mla_reduce_v1` / `kn_mla_reduce_v1_ps`, source `aiter/csrc/kernels/mla/reduce.cu`
(JIT module `module_mla_reduce`). Stage-2 epilogue of split-KV MLA decode: LSE-weighted
online-softmax combine of per-split partials. **Pure reduction — no MFMA, HBM-bandwidth bound.**
See `MLA-reduce-HIP-kernel-dissection.md` for the design dissection this report validates.

---

## 1. Build / environment setup

The host source is torch-bound (`module_mla_reduce`, two `.cu` files via the aiter JIT system).
Two environment obstacles and their fixes:

1. **No torch on the host interpreter**, and the host-prebuilt `.so`s require a newer
   `libstdc++` (GLIBCXX_3.4.31) than any local image provides. → Ran inside a fresh container
   off the repo's own ROCm image with the source bind-mounted:
   ```
   docker run -d --name mla_reduce_bench --device=/dev/kfd --device=/dev/dri \
     --group-add video --security-opt seccomp=unconfined --ipc=host \
     -v /home/anguyenh/aiter:/aiter -w /aiter \
     rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_027c46b_aiter_eee0abe_qwen3_5_20260529 \
     sleep infinity
   ```
2. **Stale prebuilt modules + unrelated import gates.** Redirected the JIT to a clean dir so
   everything rebuilds fresh, and cleared two import-chain version gates unrelated to this kernel:
   ```
   AITER_JIT_DIR=/tmp/aiter_jit         # fresh JIT build/output dir
   AITER_USE_SYSTEM_TRITON=1            # downgrade triton-gluon gate to a warning
   pip uninstall -y flydsl              # container's flydsl 0.1.2 < required 0.1.8
   PYTHONPATH=/aiter
   ```

Build time: `module_aiter_core` ~20 s, `module_mla_reduce` ~193 s. First run also needs these
env vars set.

**Correctness:** validated against a vectorized torch reference (reshape `[tiles, splits, H, Dv]`,
online softmax). Max abs err **0.031** (pure bf16 rounding), LSE exact (9.5e-7). Supported
`(H, Dv)` combos confirmed: **(128,512), (16,512), (128,128)** — other shapes raise
"doesn't support the specified settings".

Artifacts (in `aiter/op_tests/`):
- `bench_mla_reduce_standalone.py` — sweepable benchmark + correctness check
- `prof_mla_reduce.py` — fixed-launch driver for rocprofv3

### Reproducing this report

All commands run **inside the container** with the four env vars set. Define a helper once:

```bash
# host shell
RUN='docker exec -e PYTHONPATH=/aiter -e AITER_JIT_DIR=/tmp/aiter_jit \
  -e AITER_USE_SYSTEM_TRITON=1 -w /aiter mla_reduce_bench'
```

**Correctness** (the 0.031 / 9.5e-7 numbers above):
```bash
$RUN python3 op_tests/bench_mla_reduce_standalone.py --H 128 --Dv 512 --tiles 256 --splits 8 --check
```

**§2 split-count sweep** (Table, tiles=256; last row uses tiles=64):
```bash
for S in 2 3 4 8 16 32 64 128; do
  $RUN python3 op_tests/bench_mla_reduce_standalone.py --tiles 256 --splits $S
done
$RUN python3 op_tests/bench_mla_reduce_standalone.py --tiles 64 --splits 256
```

**§2 batch (tile) sweep** (Table, splits=8):
```bash
for T in 1 4 16 64 128 256 512 1024; do
  $RUN python3 op_tests/bench_mla_reduce_standalone.py --tiles $T --splits 8
done
```

Each line prints latency, traffic (MB), achieved BW, and % of 5.3 TB/s peak. Other shapes:
add `--H 16 --Dv 512` or `--H 128 --Dv 128`; fp16 with `--dtype fp16`.

**§3 profile** (rocprofv3, persistent path, tiles=256/splits=8). `prof_mla_reduce.py` reads
`PROF_TILES`/`PROF_SPLITS`/`PROF_ITERS` from the env and does a fixed warmup + timed launch
loop. Derived metrics overflow the HW counter slots, so collect **raw counters one group per
pass** and compute ratios offline:

```bash
# one counter (or a small group) per invocation; repeat per metric, then combine offline
$RUN bash -lc 'cd /aiter && PROF_TILES=256 PROF_SPLITS=8 PROF_ITERS=20 \
  rocprofv3 --pmc FETCH_SIZE WRITE_SIZE -- python3 op_tests/prof_mla_reduce.py'
$RUN bash -lc 'cd /aiter && PROF_TILES=256 PROF_SPLITS=8 PROF_ITERS=20 \
  rocprofv3 --pmc SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_CVT -- python3 op_tests/prof_mla_reduce.py'
$RUN bash -lc 'cd /aiter && PROF_TILES=256 PROF_SPLITS=8 PROF_ITERS=20 \
  rocprofv3 --pmc TCC_HIT TCC_MISS SQ_LDS_BANK_CONFLICT -- python3 op_tests/prof_mla_reduce.py'
```

Occupancy / VGPR-SGPR / LDS come from the same passes' kernel-dispatch records. Roofline
floors (HBM read 538 MB, write 34 MB) are the traffic model from the bench output at the same
tiles/splits — compare against measured `FETCH_SIZE`/`WRITE_SIZE` for the 0.99 / 1.01 ratios.

> `rocprofv3` results land in a per-run output dir (CSV/JSON); the persistent kernel
> (`kn_mla_reduce_v1_ps`) is the one dispatched at this work size — confirm the kernel name in
> the dispatch record.

---

## 2. Benchmark results

Shape: **H=128, Dv=512, bf16 output** (GLM/DeepSeek decode shape). 25 warmup / 100 timed iters.
Latency is **kernel-only device time** from aiter's `run_perftest` (`self_device_time_total`);
the `CUDA graph` column is a CUDA-graph `cuda.Event` replay cross-check. Both exclude host overhead.

### Split-count sweep (tiles=256)

| splits | path | kernel | CUDA graph | achieved BW | % of 5.3 TB/s |
|---|---|---|---|---|---|
| 2 | simple | 49.7 µs | 40.1 µs | 3.38 TB/s | 64% |
| 3 | simple | 66.1 µs | 53.8 µs | 3.56 TB/s | 67% |
| 4 | massive | 78.5 µs | 77.6 µs | 3.86 TB/s | 73% |
| 8 | massive | 176.7 µs | 168.7 µs | 3.24 TB/s | 61% |
| 16 | massive | 357.6 µs | 353.6 µs | 3.10 TB/s | 59% |
| 32 | massive | 661.2 µs | 666.9 µs | 3.31 TB/s | 62% |
| 64 | massive | 1255.7 µs | 1235.4 µs | 3.45 TB/s | 65% |
| 128 | massive | 2501.0 µs | 2498.5 µs | 3.46 TB/s | 65% |
| 256 | massive (tiles=64) | 1167.9 µs | 1169.8 µs | 3.69 TB/s | 70% |

The simple↔massive switch at splits=4 (`kMassiveThreshold`) is visible. BW dips in the
mid-split range (8–16) then recovers as the 2-way software pipeline amortizes better at high
split counts.

### Batch (tile) sweep (splits=8)

| tiles | work items | kernel | graph | achieved BW |
|---|---|---|---|---|
| 1 | 128 | 4.7 µs | 3.8 µs | 474 GB/s |
| 4 | 512 | 4.5 µs | 3.9 µs | 1.98 TB/s |
| 16 | 2048 | 8.2 µs | 6.9 µs | **4.39 TB/s (83%)** |
| 64 | 8192 | 45.7 µs | 33.5 µs | 3.13 TB/s |
| 128 | 16384 | 74.1 µs | 64.7 µs | 3.86 TB/s |
| 256 | 32768 | 175.8 µs | 169.0 µs | 3.25 TB/s |
| 512 | 65536 | 337.2 µs | 339.5 µs | 3.39 TB/s |
| 1024 | 131072 | 683.2 µs | 682.4 µs | 3.35 TB/s |

Small batches are **latency-bound** (tiles=1 → 128 work items vs 4864 WG slots, only 5% machine
fill). With host overhead removed, the kernel reaches peak by tiles≈16 — **4.39 TB/s (83%)**,
a small-footprint cache effect — then settles to the streaming **3.2–3.9 TB/s** band as the
working set exceeds cache.

---

## 3. Profile (rocprofv3)

Config: tiles=256, splits=8, persistent path (`kn_mla_reduce_v1_ps`). Derived metrics overflow
the HW counter slots even in pairs ("Request exceeds the capabilities of the hardware"), so raw
`SQ_*`/`TCC_*` counters were collected one-per-pass and derived ratios computed offline.

### Launch / occupancy

| metric | value | note |
|---|---|---|
| kernel | `_ps` (persistent) | grid-stride over work items |
| grid | 4864 WGs | = 304 CU · 8 occ · 2 |
| block | 128 threads (2 waves) | |
| VGPR / SGPR | 52 / 80 | scratch=0 |
| LDS | dynamic, ~few KB | LDS_Block_Size=0 (dynamic alloc) |
| OccupancyPercent | 88% | |
| MeanOccupancyPerActiveCU | 7.8 | ≈ `__launch_bounds__(128,8)` target — 8-way occupancy is real |

### Bound: memory, decisively

| metric | measured | model floor | ratio |
|---|---|---|---|
| HBM read (FETCH_SIZE) | 533 MB | 538 MB | **0.99** |
| HBM write (WRITE_SIZE) | 34 MB | 34 MB | **1.01** |

The kernel reads each partial-O **exactly once** and writes the output once — **at the byte floor**
for a reduction. Roofline:

- Arithmetic intensity = **0.23 flop/byte** (130 MFLOP / 567 MB); fp32 ridge ≈30 → ~130× left of
  the knee.
- Compute time ≈ **0.8 µs** vs memory time ≈ **107 µs** → ~99% of time is HBM traffic.

### VALU / MFMA

| metric | value | interpretation |
|---|---|---|
| MfmaUtil / SQ_INSTS_MFMA | 0 / 0 | confirmed no MFMA (pure reduction) |
| VALUBusy | 7.9% | VALU mostly idle, stalled on HBM |
| VALUUtilization | 91.7% | when a VALU op issues, ~59/64 lanes active — good packing |
| SQ_INSTS_VALU_FMA_F32 | 1.02M wave-insts | the weighted accumulate |
| SQ_INSTS_VALU_CVT | 65.5K | bf16 output cast |
| SALU/VALU instr ratio | 0.68 | heavy scalar bookkeeping (readfirstlane'd bounds, online-softmax scalars) — as dissection predicted |

### LDS

| metric | value | interpretation |
|---|---|---|
| SQ_LDS_BANK_CONFLICT | **0** | no serialization |
| LdsUtil | 3.3% | minimal — only gather-map + lse_scale arrays |
| SQ_INSTS_LDS | 1.08M | staging + broadcast |
| MemUnitStalled | 0.7% | |

LDS is deliberately tiny by design (holds only metadata, never the O data) to preserve occupancy-8.

### L2

L2 hit rate = **10.5%** (TCC_HIT 534K / MISS 4.54M). Expected: a pure streaming reduction with no
cross-block reuse. Chiplet-remap / L2-reuse levers do **not** apply here.

---

## 4. Verdict

The kernel behaves exactly as the dissection's TL;DR claims:

- **HBM-bandwidth-bound at the byte floor** (read/write ratios 0.99/1.01 vs model).
- **No MFMA, no LDS bank conflicts**, occupancy-8 sustained.
- Online-softmax scalar work is hidden behind loads (VALUBusy 8%, VALUUtil 92%).
- Real-world BW utilization: **~60→83% of peak** depending on batch (best ≈4.39 TB/s at
  tiles=16, a small-footprint cache effect). The gap to peak in the streaming regime is
  read latency the 2-way software pipeline only partly hides at high split counts.

### Implications for the FlyDSL port

- The byte floor is already hit → **expect HBM parity, not speedup**. Confirm with rocprofv3
  before tuning.
- The only lever with headroom is the ~60%→83% BW utilization band. The most promising knob is
  **deeper prefetch (`prefetch_depth > 2`)** at high split counts, where the current 2-way pipeline
  under-hides latency.
- Levers that do **not** apply: chiplet/XCD remap, L2-reuse swizzle (no reuse), MFMA atom tuning
  (no MFMA), LDS swizzle (no bank conflicts).
