# jagged_dense_bmm (jdbba) — FlyDSL optimization report

End-to-end results and per-experiment reproduction for the FlyDSL port and
optimization of HSTU `jagged_dense_bmm_broadcast_add` on AMD CDNA4 (MI355X /
gfx950). Companion to `jagged_dense_bmm_benchmark_repro.md` (which documents the
toy-shape prototype benchmark in detail); this document is the consolidated
results + the exact commands to reproduce every experiment.

## TL;DR

- The **generalized + optimized** FlyDSL kernel
  (`aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py`) **beats upstream
  Meta/HSTU Triton on all four production headline shapes by 1.13–1.31×**
  (device time, MI355X).
- The two **winning performance levers** were: (1) an **LDS C-shuffle epilogue**
  that turns 64 scalar `buffer_store_short` per thread into 8 wide
  `buffer_store_dwordx4` (the MFMA accumulator is M-major per lane, so a direct
  vectorized store is impossible — route C through LDS to transpose), worth
  **+6–17%**; and (2) a **shape-dependent `BLOCK_K`** (128 for reduction K≤256,
  else 64), worth **~+4%** on the K=256 cells.
- One **correctness fix** was load-bearing at scale: **i64 offset math** — the
  row-base offset `seq_start*K` overflows i32 on `B1024_D512` (7.86M·512 ≈ 4.0e9
  > 2³¹) and GPU-faults without it.
- **Method lesson:** CUDA-event **wall-clock lied by ~10×** on the small shapes
  (it is ~90% host launch overhead). Every conclusion here is from **rocprofv3
  per-kernel device time**. Two separate levers looked identical in wall-clock
  while device time moved 30%.

---

## 1. The op and the shape mapping

Per group `b` over its packed row slice `[s,e) = [seq_offsets[b], seq_offsets[b+1])`:

```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
  (M_b × N)     (M_b × K)      (K × N)     (1 × N broadcast)
```

bf16 in/out, fp32 accumulate. Dense is host pre-transposed to a tall
`(B_groups·N, K)` matrix and consumed plain (no preshuffle). `seq_offsets` is
device-resident int32.

**HSTU `(B,D,K,N)` bench naming ≠ upstream `(B,K,N)`.** Mapping to standard GEMM:

| Bench | Meaning | GEMM dim |
|---|---|---|
| `B` | number of groups | batch/group count |
| `N` | `max_seq_len` envelope (M_i ≤ N) | grid M-axis |
| `D` | jagged width | **reduction K** |
| `K` | output channels | **output N** |

So `B1024_D512_K512_N16384` runs with reduction K=512, output N=512,
max_seq_len=16384, 1024 groups.

### Headline shapes (uniform M_i = 7680, a tile-multiple near the deployment mean L/B≈7800)

| Shape | B (groups) | D = reduction K | K_bench = output N | regime |
|---|---|---|---|---|
| B1024_D256_K256_N16384 | 1024 | 256 | 256 | train, small hidden |
| B1024_D512_K512_N16384 | 1024 | 512 | 512 | train, large hidden |
| B120_D256_K256_N16384 | 120 | 256 | 256 | inference, small |
| B120_D512_K512_N16384 | 120 | 512 | 512 | inference, large |

---

## 2. Final results (device time, MI355X / gfx950, M_i = 7680)

rocprofv3 `--kernel-trace`, median of the per-kernel dispatch durations:

| shape | FlyDSL gen | Triton | speedup (tri/fly) |
|---|---|---|---|
| B120_D256_K256_N16384 | **312 µs** | 352 µs | **1.13×** |
| B120_D512_K512_N16384 | **841 µs** | 1060 µs | **1.26×** |
| B1024_D256_K256_N16384 | **2294 µs** | 2994 µs | **1.31×** |
| B1024_D512_K512_N16384 | **6843 µs** | 8828 µs | **1.29×** |

All cells `cos = 1.0000` vs the torch eager reference, including partial-tile,
empty-group, and skewed edge cases.

> Microsecond numbers drift run-to-run with GPU clock; the **per-shape ordering
> and the >1× speedup are the stable, reproducible signal.**

---

## 3. Measurement methodology (read before reproducing anything)

**Do not trust CUDA-event wall-clock at these shapes.** On the small toy shapes
the wall-clock is ~90% fixed host launch/dispatch overhead (~70 µs) and reads
*flat* across problem size — it masked a real C-store waterfall and a 30% tile
regression. The only number to optimize against is **per-kernel device time from
rocprofv3**.

The repo provides a ready-made device-time harness:

- `op_tests/flydsl_tests/bench_headline_worker.py` — args
  `<flydsl|triton> <B> <D> <Kout> <Mi>`; builds inputs for one headline cell and
  fires warmup + 30 launches of one implementation, nothing else.
- `op_tests/flydsl_tests/read_us2.py` — args `<rocprof_outdir> <name_substr>`;
  joins the rocprofv3 sqlite `kernel_dispatch`×`kernel_symbol` tables, filters by
  kernel name, prints the median dispatch duration in µs.

There is **no pure-Python substitute**: CUDA-graph capture of the FlyDSL launch
path produces an empty graph (replay yields zeros), and batched CUDA-event timing
is still dispatch-starved.

All commands run inside the `anguyenh-dev` devcontainer (torch/triton/flydsl live
in the container venv, not the bare host).

---

## 4. Per-experiment reproduction

Every block below is copy-pasteable. `FLYDSL_RUNTIME_ENABLE_CACHE=1` is fine for
timing; use `=0` when you change kernel source and want a guaranteed recompile.

### 4.1 Correctness (all shapes + edges)

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
cat > /tmp/correct.py << "PY"
import torch, importlib
import flydsl.compiler as flyc
m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_gen")
def run(B, N, K, rc=None, tag=""):
    rc = rc or [7680]*B
    Bg = len(rc)
    so = torch.zeros(Bg+1, dtype=torch.int32)
    for i in range(Bg): so[i+1] = so[i] + rc[i]
    L = int(so[-1]); msl = max(rc) if rc else 0
    if L == 0: return
    jag = torch.randn(L+128, K, dtype=torch.bfloat16).cuda()[:L].contiguous()
    dense = torch.randn(Bg, K, N, dtype=torch.bfloat16).cuda()
    bias = torch.randn(Bg, N, dtype=torch.bfloat16).cuda()
    dt = dense.transpose(1,2).reshape(Bg*N, K).contiguous()
    bf = bias.reshape(Bg*N).contiguous()
    out = torch.zeros(L+128, N, dtype=torch.bfloat16).cuda(); sod = so.cuda()
    tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, Bg, msl, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    chk = [0, Bg//2, Bg-1] if Bg > 2 else list(range(Bg))
    bad = 0
    for b in chk:
        s = int(so[b]); e = int(so[b+1])
        if e <= s: continue
        exp = (jag[s:e].float() @ dense[b].float() + bias[b].float()[None,:]).to(torch.bfloat16)
        cos = torch.nn.functional.cosine_similarity(exp.float().flatten(), out[s:e].float().flatten(), dim=0).item()
        if cos < 0.999: bad += 1
    print(tag.ljust(26), "PASS" if bad == 0 else "FAIL %d" % bad)
run(120, 256, 256, tag="B120_D256_K256")
run(120, 512, 512, tag="B120_D512_K512")
run(1024, 256, 256, tag="B1024_D256_K256")
run(1024, 512, 512, tag="B1024_D512_K512")
run(2, 256, 256, [200, 100], "partial[200,100]")
run(3, 256, 256, [128, 0, 128], "empty[128,0,128]")
PY
PYTHONPATH=/home/anguyenh/aiter:$PYTHONPATH FLYDSL_RUNTIME_ENABLE_CACHE=0 python /tmp/correct.py
'
```

Expected: all six lines `PASS`.

### 4.2 Device-time vs Triton (one shape)

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
  for IMPL in flydsl triton; do
    rm -rf /tmp/h_$IMPL
    PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders:$PYTHONPATH \
      rocprofv3 --kernel-trace -d /tmp/h_$IMPL -o trace -- \
      python op_tests/flydsl_tests/bench_headline_worker.py $IMPL 120 512 512 7680 >/dev/null 2>&1
  done
  echo -n "FlyDSL  "; python op_tests/flydsl_tests/read_us2.py /tmp/h_flydsl jdbba
  echo -n "Triton  "; python op_tests/flydsl_tests/read_us2.py /tmp/h_triton jagged_dense_bmm_broadcast_add
'
```

Swap the `120 512 512` for the other cells: `120 256 256`, `1024 256 256`,
`1024 512 512`. (Args are `<B> <D> <Kout>`; D=reduction K, Kout=output N.)

### 4.3 Full headline sweep (all 4 cells, both impls)

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
for C in 120:256:256 120:512:512 1024:256:256 1024:512:512; do
  B=${C%%:*}; R=${C#*:}; D=${R%%:*}; KO=${R##*:}
  for IMPL in flydsl triton; do
    O=/tmp/sw_${IMPL}_${B}_${D}; rm -rf $O
    PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders:$PYTHONPATH \
      rocprofv3 --kernel-trace -d $O -o trace -- \
      python op_tests/flydsl_tests/bench_headline_worker.py $IMPL $B $D $KO 7680 >/dev/null 2>&1
  done
  SUB=jdbba; F=$(python op_tests/flydsl_tests/read_us2.py /tmp/sw_flydsl_${B}_${D} jdbba)
  T=$(python op_tests/flydsl_tests/read_us2.py /tmp/sw_triton_${B}_${D} jagged_dense_bmm_broadcast_add)
  echo "B${B}_D${D}_K${KO}_N16384  FlyDSL ${F}us  Triton ${T}us"
done
'
```

### 4.4 Static ISA probe (prove the store pattern / VGPR / occupancy)

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
  rm -rf /tmp/isa
  PYTHONPATH=/home/anguyenh/aiter:$PYTHONPATH FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
    python op_tests/flydsl_tests/bench_headline_worker.py flydsl 120 512 512 7680 >/dev/null 2>&1
  F=$(find /tmp/isa -name 22_final_isa.s | head -1)
  echo "=== resource header ==="
  grep -iE "vgpr_count|agpr_count|sgpr_count|group_segment_fixed_size" $F | head
  echo "=== store/load histogram ==="
  grep -oE "buffer_store_short|buffer_store_dwordx[0-9]|buffer_load_dwordx[0-9]|ds_write_b128|ds_read_b128|s_barrier" $F | sort | uniq -c | sort -rn
'
```

After the C-shuffle epilogue you should see `buffer_store_dwordx4` (wide stores),
**not** `buffer_store_short` (scalar). That ISA delta is the proof the memory
pattern changed.

---

## 5. The optimization campaign (what was tried, in order)

Device time is the metric throughout. Each lever was changed in isolation on a
clone, verified for correctness (cos=1.0), then measured with rocprofv3; winners
were re-verified in a single controlled interleaved sweep before promotion
(GPU-clock drift between separate runs is real).

### 5.1 Prerequisite — generalize N/K

The original prototype hardcoded `N == K == 128` as compile-time constants and
**could not run the production shapes at all**. Generalization makes N (output)
and K (reduction) runtime-parametric via a memoized factory (`_build_launcher`)
that bakes them as closure constants per shape — mirrors
`splitk_hgemm.compile_hgemm_kernel`. Public entry derives N, K from the tall
dense matrix shape.

### 5.2 Correctness fix — i64 offset math (load-bearing at scale)

`a_row_off = seq_start * K` was computed in i32. With `seq_start` reaching ~L
(millions of rows), `B1024_D512` hits `7.86M·512 = 4.0e9 > 2³¹` and the kernel
**GPU-faults**. Only that one cell overflows; the other three stay under 2³¹
(which is why they ran). Fix: cast to i64 *before* the stride multiply.

### 5.3 WIN — epilogue store vectorization (LDS C-shuffle), +6–17%

The MFMA C accumulator is **M-major per lane**: a lane's contiguous fragment
elements map to different output *rows* (stride N in global), so a vectorized
N-contiguous store straight from the fragment is impossible — the baseline
emitted **64 scalar `buffer_store_short` per thread**, and a store-deletion
diagnostic showed the store alone cost **38–62% of runtime** at these
memory-bound shapes. Fix: route C through LDS to transpose the layout — write the
bf16+bias fragment to a row-major shared C tile in its natural MFMA layout
(reusing the A-staging LDS, no extra smem), barrier, then re-read N-contiguous (8
bf16/thread) and store `buffer_store_dwordx4`. **64 narrow stores → 8 wide
stores.** This is the cdna-kernel-opt method's "epilogue store vectorization is
often the single largest win for a kernel that already has a good main loop."

### 5.4 WIN — shape-dependent BLOCK_K, ~+4% on K=256

`block_k = 128 if K <= 256 else 64`. For reduction K=256 a 2-iter K-loop
(BLOCK_K=128) has fewer barriers and wins; for K=512 the deeper BLOCK_K=64
pipeline keeps occupancy. **`BLOCK_K=256` is unsafe** — the 2-stage
double-buffer epilogue silently mis-accumulates a single K-tile (cosine masked a
123% relative error).

### 5.5 DEAD ENDS (verified, do not repeat)

| Lever | Why it failed at the real shapes |
|---|---|
| `BLOCK_M = 64 / 256`, `BLOCK_N = 256` | VGPR blowup → occupancy collapses to 1 wave/SIMD. Default 128/128 is the sweet spot even when compute-bound. |
| `STAGES_A = 3` | No-op — the ping-pong only uses 2 buffers; the 3rd is allocated, never read. Real 3-stage needs a `run_pipeline_stage` restructure. |
| Dense (B) staged through LDS | Regresses — B is already maximally coalesced (`buffer_load_dwordx4`); intra-block B-reuse is low. The real B reuse is cross-block (L2 / chiplet, a different lever). |
| Packed / persistent grid | Irrelevant for the headline shapes — uniform `M_i = max_seq_len` ⇒ zero tail-tile waste already. (Only helps the skewed deployment distribution.) |

### 5.6 Toy-shape detour (and the measurement lesson)

Before the production shapes were known, levers were swept on N=K=128, L≤32k toy
shapes. Those results **did not transfer** — at K=128 the kernel is launch-bound,
not compute-bound, and the CUDA-event wall-clock was ~90% host overhead. The
`seq_start/seq_end` `readfirstlane` scalarization (kills a divergent-C-descriptor
store waterfall — 257 `v_readfirstlane_b32` + a 64-wide exec-mask loop collapse
to ~0) was the one toy-shape finding that carried forward; it is part of the
prototype baseline. Full detail in `jagged_dense_bmm_benchmark_repro.md` §1–7.

---

## 6. Next opportunities

- **Cross-block Dense[b] L2 reuse via a chiplet/XCD grid remap** — the largest
  untried lever for the B1024 cells (each Dense[b] N-tile is re-read by all ~60
  M-tile blocks of its group). See the `chiplet-xcd-remap` method.
- **Store-side residual** — the C-shuffle's bf16 LDS *write* is the next
  bottleneck (the readback and global store are already wide).
- **Skew-tolerant grid** — a persistent/sorted-grid variant for the non-uniform
  deployment `M_i ~ 0.95·Uniform(1, N)` distribution (the uniform headline grid
  has zero tail waste but the skewed distribution does not).

---

## 7. Files

| File | Role |
|---|---|
| `aiter/ops/flydsl/kernels/jagged_dense_bmm.py` | N=K=128 prototype (validated reference) |
| `aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py` | generalized + optimized production kernel |
| `op_tests/flydsl_tests/bench_headline_worker.py` | per-shape device-time worker (rocprofv3) |
| `op_tests/flydsl_tests/read_us2.py` | rocprofv3 sqlite → median µs parser |
| `op_tests/flydsl_tests/bench_jagged_dense_bmm.py` | toy-shape correctness + wall-clock / `--device-time` bench |
| `jagged_dense_bmm_benchmark_repro.md` | toy-shape benchmark detail + optimization log |
