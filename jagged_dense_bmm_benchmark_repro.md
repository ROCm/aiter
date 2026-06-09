# Reproducing the FlyDSL jdbba vs Triton benchmark

Step-by-step guide to reproduce the `jagged_dense_bmm_broadcast_add` (jdbba)
correctness + performance numbers comparing the FlyDSL prototype against the
upstream Meta/HSTU Triton kernel.

What it measures, per group `b` over its packed row slice `[s, e)`:

```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
  (M_b x N)     (M_b x K)      (K x N)     (1 x N broadcast)
```

bf16 in/out, fp32 accumulate, `N = K = 128`.

---

## 1. Prerequisites

Everything runs inside the **`anguyenh-dev`** devcontainer — `torch`, `triton`,
and `flydsl` live in the container venv, NOT on the bare host. Do not run these
with the host Python.

Confirm the container is up and the three deps import:

```bash
docker ps --format '{{.Names}}' | grep anguyenh-dev

docker exec anguyenh-dev bash -c '
  python -c "import torch;  print(\"torch \", torch.__version__)"
  python -c "import triton; print(\"triton\", triton.__version__)"
  python -c "import flydsl; print(\"flydsl\", flydsl.__file__)"
'
```

Expected (versions may differ slightly):

```
torch  <2.x>
triton 3.6.0
flydsl /venv/lib/python3.12/site-packages/flydsl/__init__.py
```

You also need a GPU visible to the container (the benchmark calls `.cuda()`).

---

## 2. Files involved

| File | Role |
|---|---|
| `aiter/ops/flydsl/kernels/jagged_dense_bmm.py` | FlyDSL kernel + `@flyc.jit` launcher |
| `op_tests/flydsl_tests/bench_jagged_dense_bmm.py` | correctness check + timing harness |
| `~/generative-recommenders/.../triton/triton_jagged.py` | upstream Triton reference |

---

## 3. Run the benchmark

Two repos must be on `PYTHONPATH`: `aiter` (for the kernel package import) and
`generative-recommenders` (for the Triton reference).

The benchmark reports **two different timings** — read §4.0 before trusting any
number.

### 3a. End-to-end wall-clock (default)

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
  PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders:$PYTHONPATH \
  FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python op_tests/flydsl_tests/bench_jagged_dense_bmm.py
'
```

Validates correctness (cos vs torch eager) and reports per-call wall-clock. This
is **end-to-end latency** and is ~90% host launch overhead — see the caveat.

### 3b. True per-kernel GPU time (recommended for optimization)

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
  PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders:$PYTHONPATH \
  python op_tests/flydsl_tests/bench_jagged_dense_bmm.py --device-time
'
```

`--device-time` re-runs each shape under `rocprofv3 --kernel-trace` (worker
subprocess per shape/impl) and reads the per-kernel device durations out of the
resulting sqlite db. Requires `rocprofv3` on PATH. Slower to run (it spawns ~14
profiled subprocesses) but it is the only number worth optimizing against.

---

## 4. Expected output

### 4.0 Which number to trust (read this first)

At these shapes (K=128, tiny GEMMs) the default **wall-clock is ~90% fixed host
launch/dispatch overhead (~70 us) and only ~10% GPU work.** Consequences:

- The wall-clock is **flat across L** — L=512 and L=32768 both read ~0.077 ms —
  because it is measuring dispatch, not compute. This is *not* the kernel being
  launch-bound on the device; it is the *host* call path.
- Cross-kernel comparisons in wall-clock mode are **unreliable**. During
  optimization, two changes (a C-store scalarization, and a `BLOCK_M` sweep)
  both left the wall-clock unchanged while `--device-time` showed the scalarize
  was neutral-to-good and the `BLOCK_M` bump was a **30% device regression**.
- There is no pure-Python substitute for `--device-time`: CUDA-graph capture of
  the FlyDSL launch path produces an empty graph (replay yields zeros), and
  batched CUDA-event timing is still dispatch-starved.

**Use `--device-time` for any kernel-level conclusion.** Use the wall-clock only
as the end-to-end "one call in a Python loop" number.

### 4.1 End-to-end wall-clock (default mode)

```
FlyDSL jdbba vs Triton  END-TO-END wall-clock  (N=128, K=128, bf16)
NOTE: ~90% of this is host launch overhead; use --device-time for GPU time.

[exact-multiple] L=  512 ... FlyDSL ~0.076 ms (cos=1.0000)  Triton ~0.026 ms (cos=1.0000)  speedup(tri/fly)~0.35x
... (all cases ~0.075-0.081 ms FlyDSL / ~0.026-0.034 ms Triton; flat across L) ...
```

The `~0.35x` here is a **launch-path artifact, not kernel speed** (Triton's
wrapper amortizes dispatch differently). Do not cite it as a kernel comparison.

### 4.2 True device time (`--device-time`, rocprofv3)

```
FlyDSL jdbba vs Triton  DEVICE TIME (rocprofv3)  (N=128, K=128, bf16)

[exact-multiple] L=  512  FlyDSL   5.08 us  Triton   5.36 us  speedup(tri/fly)=1.06x
[partial-bottom] L=  300  FlyDSL  12.12 us  Triton   5.20 us  speedup(tri/fly)=0.43x
[empty-group   ] L=  256  FlyDSL   5.64 us  Triton   5.00 us  speedup(tri/fly)=0.89x
[skewed        ] L=  688  FlyDSL   5.60 us  Triton   5.36 us  speedup(tri/fly)=0.96x
[uniform-32x256] L= 8192  FlyDSL   6.20 us  Triton   5.24 us  speedup(tri/fly)=0.85x
[uniform-64x512] L=32768  FlyDSL  19.08 us  Triton   8.48 us  speedup(tri/fly)=0.44x
[ragged-128grp ] L=32672  FlyDSL  10.40 us  Triton  14.00 us  speedup(tri/fly)=1.35x
```

### How to read it

- **`cos=1.0000`** (wall-clock mode) on every case = numerically correct against
  the torch eager reference. The small cases exercise the jagged edge logic:
  exact tile multiples, a partial bottom tile, an empty group (`M_b=0`), and a
  skewed mix where short groups trigger the runtime early-exit.
- **Device time is the real story.** The prototype is **competitive-to-faster on
  small and ragged shapes** (`exact-multiple` 1.06x, `ragged-128grp` 1.35x) but
  **~2x slower on the large uniform shape** (`uniform-64x512` 0.44x) and on
  `partial-bottom` (0.43x). Those two are the optimization targets (see §7).
- Microsecond numbers vary run-to-run and by GPU; the **cos values and the
  per-shape ordering** are the stable signal.

---

## 5. Customizing the shapes

Edit the `CASES` dict near the bottom of
`op_tests/flydsl_tests/bench_jagged_dense_bmm.py` (label -> per-group row
counts); both timing modes iterate it:

```python
CASES = {
    "my-shape": [128, 64, 512, 0, 300],
    ...
}
```

`N` and `K` are fixed at 128 by the prototype kernel constants in
`aiter/ops/flydsl/kernels/jagged_dense_bmm.py` (`N`, `K`, `BLOCK_M/N/K`); change
them there if you need a different problem size (the kernel currently assumes
`N == BLOCK_N == 128` and `K == 128`).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named 'generative_recommenders'` | `generative-recommenders` not on `PYTHONPATH` — include `/home/anguyenh/generative-recommenders` (step 3). |
| `WARNING: upstream Triton kernel unavailable: ...` | Triton import failed; the FlyDSL column still runs, Triton column shows `n/a`. Check the printed exception. |
| `ImportError: cannot import name 'MxScaleRoundMode'` | Stale compiled `module_aiter_core.so`. Force a clean rebuild: `rm -rf aiter/jit/build/module_aiter_core aiter/jit/module_aiter_core.so && python -c "import aiter"`. |
| Stale / wrong FlyDSL timings after editing the kernel | Keep `FLYDSL_RUNTIME_ENABLE_CACHE=0` so the JIT recompiles. |
| `No module named pytest` | This is a plain `python` script, not a pytest module — run it with `python`, not `pytest`. |
| `--device-time` prints all `nan` | `rocprofv3` not on PATH, or the sqlite schema differs. Run `rocprofv3 --kernel-trace -d /tmp/x -- python ...` by hand and inspect `trace_results.db`. |

---

## 7. Optimization log (what was tried)

Device time (§4.2) is the metric throughout. Method: static ISA probe
(`FLYDSL_DUMP_IR=1` → `22_final_isa.s`) + `--device-time`, one lever at a time.

### Applied — kept

- **`seq_start`/`seq_end` scalarization** (`fx.rocdl.readfirstlane`, kernel
  ~line 91). The per-group offsets came out of `buffer_load` typed per-lane
  (VGPR), which made the bounded C buffer descriptor **divergent**. That forced
  the epilogue store into a per-lane `readfirstlane`/`exec`-mask **waterfall
  loop** — the ISA showed 257 `v_readfirstlane_b32` and 64×
  `{cmp_eq_u64, and_saveexec, xor, cbranch_execnz}` for only 128 MFMA. Scalarizing
  collapsed that cluster to ~0 and dropped VGPR 152→146. Correct on all cases.
  Keep it — it is a prerequisite once the kernel is no longer launch-bound.

### Tried — DEAD END, do not repeat

- **`BLOCK_M = 256`** (was a "v2" fork, now deleted). Intent: raise
  work-per-block / halve the launched M-tile count. Result: **device-time
  regression** — `uniform-64x512` went 7.2 µs → 9.4 µs (~+30%). Cause: VGPR
  blew up 146 → **276** (+20 AGPR), dropping occupancy to ~1 wave/SIMD. Tile
  reuse cannot pay for the occupancy loss at K=128. This is the textbook
  "doubling the tile drops occupancy below 2 WG/CU" failure. **Future tile
  tuning should go the other way: smaller / lower-register-pressure blocks, not
  bigger.** The wall-clock benchmark hid this entirely (it read flat), which is
  why §4.0 insists on `--device-time`.

---

## 8. Generalized kernel + REAL target shapes (the actual deliverable)

Everything in §1–7 used the N=K=128 prototype (`jagged_dense_bmm.py`) on tiny
launch-bound shapes. The production target shapes are far larger and the prototype
**cannot run them** (N, K were hardcoded constants). The generalized kernel
`aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py` makes N (output) and K
(reduction) runtime-parametric via a memoized factory (`_build_launcher`) and is
the kernel to use going forward. The prototype is kept only as the validated
N=K=128 reference.

### 8.1 HSTU (B,D,K,N) naming → GEMM dims

The HSTU bench names shapes `(B,D,K,N)`, NOT the upstream `(B,K,N)`:

| Bench | Meaning | GEMM dim |
|---|---|---|
| `B` | number of groups | batch/group count |
| `N` | `max_seq_len` envelope (M_i ≤ N) | grid M-axis |
| `D` | jagged width | **reduction K** |
| `K` | output channels | **output N** |

So `B1024_D512_K512_N16384` = 1024 groups, reduction K=512, output N=512,
max_seq_len=16384. Per group: `(M_i × D) · (D × K_bench) → (M_i × K_bench)`.

### 8.2 Headline benchmark (device time, MI355X, M_i=7680)

Harness: `op_tests/flydsl_tests/bench_headline_worker.py` (args
`<flydsl|triton> B D Kout Mi`, warmup+30 launches) + `read_us2.py`
(`python read_us2.py <rocprof_outdir> <substr> p10` → 10th-percentile µs). Note
the kernel-name substring differs by provider: `jdbba` for FlyDSL,
`jagged_dense_bmm_broadcast_add` for the Triton kernel. Example:

```bash
docker exec -w /home/anguyenh/aiter anguyenh-dev bash -c '
  rm -rf /tmp/h
  PYTHONPATH=/home/anguyenh/aiter:$PYTHONPATH \
    rocprofv3 --kernel-trace -d /tmp/h -o trace -- \
    python op_tests/flydsl_tests/bench_headline_worker.py flydsl 120 512 512 7680
  python op_tests/flydsl_tests/read_us2.py /tmp/h jdbba p10
'
```

Final results (current FlyDSL gen = 16x16x32 atom + XCD remap + bounded-A fix, vs
upstream Triton). **Authoritative two-method comparison is in
`jagged_dense_bmm_optimization_report.md` §2.** Summary (autotune inflation
excluded — Triton best-config only):

```
                         rocprofv3 hot-L2 (us)        do_bench cold-L2 (ms)
shape                  FlyDSL_p10  Triton_min     FlyDSL    Triton    verdict
B120_D256_K256_N16384     267        256          0.2748    0.2770    ~tie / +0.8%
B120_D512_K512_N16384     730        752          0.7377    0.7764    +3-5% FlyDSL
B1024_D256_K256_N16384   2091       2085          2.1986    2.1319    tie / -3% (L2-warmth dependent)
B1024_D512_K512_N16384   5960       6326          6.0422    6.4039    +5.6-6.1% FlyDSL
```

**Verdict: FlyDSL wins 3 of 4** — ~5–6% ahead on both D=512 cells (robust under
both methods), a hair ahead on B120_D256, tie-to-slightly-behind on B1024_D256
depending on L2 warmth. The 16x16x32 atom + XCD remap moved it from the earlier
parity baseline to ahead on D=512.

> **Use `p10` for `jdbba` (FlyDSL) and `min` for `jagged_dense_bmm_broadcast_add`
> (Triton) — NEVER median.** Triton autotunes (e.g. B1024_D512: 849 dispatches,
> spread 6307–18792 µs); its median (8789) is trial-inflated and is NOT deployed.
> An earlier version of this table compared medians and wrongly claimed FlyDSL
> "beats Triton 1.13–1.31×" — a median-vs-autotune-min artifact. `do_bench`
> (autotuner settled) and rocprofv3-min agree and are the honest numbers.
> **Cold vs hot L2:** do_bench flushes the L2 each rep, which erodes the XCD
> remap's `Dense[b]`-reuse win — that's why B1024_D256 reads tie (hot) vs −3%
> (cold). Repeated/back-to-back deployment (resident weights) tracks the hot-L2
> figure. See report §2 and method-lessons #3/#4 (incl. the corrected skew result).

### 8.3 What was applied to the generalized kernel (device-time verified)

- **i64 offset math (correctness, CRITICAL).** `a_row_off = Int64(seq_start)*Int64(K)`
  etc. Without it `B1024_D512` GPU-faults: `seq_start*K = 7.86M*512 = 4.0e9 > 2³¹`.
  Only that cell overflows i32; the others stay under 2³¹ (which is why they ran).
- **Epilogue store vectorization (the big win, +6–17%).** The MFMA C accumulator
  is M-major per lane, so the original store emitted 64 scalar `buffer_store_short`
  per thread (the store alone cost 38–62% of runtime). Fix = an LDS C-shuffle
  epilogue: write the bf16+bias fragment to a row-major shared C tile (reuses the
  A-staging LDS, no extra smem), barrier, re-read N-contiguous (8 bf16/thread),
  store `buffer_store_dwordx4`. 64 narrow stores → 8 wide stores.
- **Shape-dependent `BLOCK_K` (+~4% on K=256).** `block_k = 128 if K<=256 else 64`
  in the public entry. K=256 prefers a 2-iter K-loop (fewer barriers); K=512 keeps
  the deeper BLOCK_K=64 pipeline for occupancy. `BLOCK_K=256` is UNSAFE — the
  2-stage double-buffer epilogue silently mis-accumulates a single K-tile.
- **Chiplet (XCD) block-ID remap (+~5% on D=512, gap-closer vs Triton).**
  HipKittens Algorithm 1: a launch-time block-ID remap (`_xcd_remap`) that
  inverts the hardware's round-robin XCD assignment so a group's M-tiles (which
  all share that group's `Dense[b]` weight) co-locate on one XCD's private 4 MB
  L2. Diagnosis that motivated it: B1024_D512 was memory-bound with **L2 hit
  49.3%** and **DRAM read traffic ≈ 44× the model minimum** — `Dense[b]` was
  being re-fetched from HBM by its scattered M-tiles. After remap: **L2 hit
  49→76%, DRAM read requests −67%, device time −5.3%** (B1024_D512). Two knobs:
  `W` (window height, =8, flat optimum) and `C` (XCD chunk size), which is
  **weight-size dependent** — small reduction K (≤256, small `Dense[b]` fits one
  XCD's L2 in a small chunk) wants `C=32` and *regresses* with large C (the
  L2-greedy / LLC-starvation trap); larger K wants `C=60`. Verified `cos=1.0`
  and win-or-neutral on all four headline shapes. Occupancy probe confirmed this
  was the right lever over ping-pong / multi-wave interleave (see §8.5).

### 8.4 DEAD ENDS at real shapes (do not repeat)

- **BLOCK_M=64 / 256, BLOCK_N=256** — all regress. VGPR blowup drops occupancy to
  1 wave/SIMD; default `BLOCK_M=BLOCK_N=128` is the sweet spot even when
  compute-bound.
- **STAGES_A=3** — no-op. The ping-pong only uses 2 buffers; the 3rd is allocated
  but never read. Real 3-stage prefetch needs a `run_pipeline_stage` restructure.
- **Dense (B) staged through LDS** — regresses. B is already maximally coalesced
  (`buffer_load_dwordx4`) and intra-block B-reuse is low. The real B reuse is
  cross-block (an L2/chiplet-scheduling concern — different lever, see
  `chiplet-xcd-remap`).
- **Packed / persistent grid** — irrelevant for the headline shapes: uniform
  `M_i = max_seq_len` ⇒ zero tail-tile waste already. (Would only help the skewed
  deployment distribution `M_i ~ 0.95·Uniform(1,N)`.)

### 8.5 Next opportunities

**Bound:** the kernel is **memory-bandwidth-bound** at these shapes. Roofline:
all four cells sit below the MI355X ridge (AI 126–248 FLOP/B < 312), at 32–48% of
peak HBM. Occupancy probe (B1024_D512): 84 VGPR, 32 KB LDS/block → resource
ceiling 5 waves/SIMD, **achieved 2.8 waves/SIMD**; `MemUnitStalled` 23%,
`VALUBusy` 19% (ALU idle, not compute-bound).

**Ruled out by the probe:**
- **Ping-pong / 8-wave / 4-wave interleave** — SKIP. The kernel is occupancy-
  resource-limited (LDS+VGPR), not latency-bound; more waves can't fit (32 KB
  LDS already caps at ~5 blk/CU) and can't fix a 49% L2 miss. `STAGES_A=2`
  already provides the 2-stage A pipeline. Both roofline and the stall counters
  agree latency is not the wall — traffic is.
- **Different MFMA atom (16×16×32, 32×32×16)** — low value; compute is only
  ~25% of peak, so a faster MMA can't move a memory-bound kernel.

**Done since:** the XCD remap (§8.3) — the lever this section previously flagged
as "largest untried" — landed at +5% on D=512.

**Still open:**
- **Per-shape `C` autotune** — current defaults (`C=32` for K≤256, `C=60` for
  K>256, `W=8`) are from a 5-point sweep; a finer sweep per deployment shape may
  add a little.
- **Raise arithmetic intensity above the 312 ridge** — larger `BLOCK_M` reuses
  each weight load more, but hits the occupancy cliff at `BLOCK_M=256` (§8.4).
  A 2× over Triton is not reachable by scheduling alone at bf16; it would need a
  byte-cutting change (e.g. fp8 weights), which changes the numerics contract.
- **Skew-tolerant persistent/sorted grid** — follow-up for the non-uniform
  deployment `M_i` distribution (irrelevant for the uniform headline shapes).
- **C-shuffle LDS write** — the bf16 LDS store in the epilogue is the remaining
  store-side cost (the readback + global store are already wide).
