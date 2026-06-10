# jagged_dense_bmm (jdbba) — MI300X / gfx942 experiment plan

A neutral, re-runnable plan to optimize the FlyDSL `jagged_dense_bmm_broadcast_add`
kernel on **AMD Instinct MI300X (gfx942 / CDNA3)**.

This plan deliberately carries **no MI355X/gfx950 verdicts**. The prior campaign was
run on gfx950 and several of its conclusions are hardware-specific (different MMA
atom availability, different XCD count, different LDS budget, different occupancy
limits). Each lever below is listed as a **hypothesis to test on gfx942**, not a
known win or loss. Record the gfx942 result independently and only then conclude.

---

## 0. The op

For each group `b` over its packed row slice `[s, e) = [seq_offsets[b], seq_offsets[b+1])`:

```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
  (M_b x N)     (M_b x K)      (K x N)     (1 x N broadcast)
```

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `A` jagged | `(L, K)`, `L = ΣM_b` | bf16 | packed, row-major |
| `B` dense | `(B_groups·N, K)` | bf16 | host pre-transposed (tall) |
| `BIAS` | `(B_groups·N,)` | bf16 | broadcast over rows |
| `C` out | `(L, N)` | bf16 | packed; fp32 accumulate |
| `SEQ_OFFSETS` | `(B_groups+1,)` | int32 | device-resident prefix sum |

HSTU bench naming `(B, D, K)` → GEMM dims: **reduction K = bench D**, **output N =
bench K**, **M-envelope = max_seq_len (Mi)**.

---

## 1. Shapes of interest (same as the original campaign)

Headline shapes `(B groups, D=reduction K, Kout=output N)`, `max_seq_len (Mi) = 7680`:

| shape | B | D (reduction) | Kout (output) |
|---|---|---|---|
| B120_D256  | 120  | 256 | 256 |
| B120_D512  | 120  | 512 | 512 |
| B1024_D256 | 1024 | 256 | 256 |
| B1024_D512 | 1024 | 512 | 512 |

Two regimes, both must be measured:
- **UNIFORM** — `M_i = max_seq_len` for every group (controlled baseline, zero
  tail-tile waste).
- **SKEW** — `M_i = max_seq_len · U(0,1)^4`, ~20–30% empty groups, plus one full and
  one near-full group (the deployment distribution). The dispatch picks the kernel
  variant per regime via the `uniform_seqlen` flag.

---

## 2. Hardware deltas to keep in mind on gfx942 (CDNA3)

These are *facts about the hardware*, not results. They tell you which levers are
even applicable and which constants must change before any tuning.

| Aspect | gfx942 / MI300X (CDNA3) | (gfx950 / MI355X, for contrast) |
|---|---|---|
| Wave size | 64 | 64 |
| bf16 MFMA atom | **16×16×16** (no 16×16×32) | 16×16×32 available |
| XCDs per GPU | **varies by SKU (e.g. MI300X = 8)** — re-confirm with `rocminfo` | 8 |
| LDS / CU | **64 KB** | 160 KB |
| FP4 / MFMA-scale | no | yes |

Consequences to verify, not assume:
- The MFMA-atom lever is **different code** on gfx942 (the 16×16×16 path is the only
  option; there is no 32-K atom to switch to). Whatever atom logic exists must select
  16×16×16 for gfx942.
- The XCD chiplet remap depends on the XCD count and on per-CU L2 behavior; both
  differ. Its sign and magnitude **must be re-measured** on gfx942 in both regimes.
- The 64 KB LDS budget is **2.5× smaller** than gfx950. Any lever that grows LDS
  (deeper pipeline stages, wider C-shuffle tiles, larger BLOCK_M/BLOCK_K staging)
  has a much tighter ceiling here — re-check `smem_bytes` against 64 KB.

---

## 3. Methodology (fixed; do not vary between levers)

1. **Always run inside the devcontainer.** torch/triton/aiter are not on the bare
   host. Confirm `rocminfo | grep gfx` reports `gfx942` before trusting any number.
2. **Correctness gate first.** Every lever must hold `cos ≥ 0.999` vs the torch
   eager reference (`Out[s:e] = (A[s:e].float() @ Dense[b].float() + Bias[b]).bf16()`)
   on all 4 shapes AND on edge cases (empty group `M_b=0`, partial bottom tile
   `M_b % BLOCK_M ≠ 0`, one long + many short groups). Speed is meaningless until
   correctness passes.
3. **Two timing tools, stated explicitly per result:**
   - `triton.testing.do_bench` — CUDA-event wall-clock, **L2 flushed each rep**
     (cold-L2). This is the deployment-representative number; it is the headline.
   - `rocprofv3 --kernel-trace` + p10 (NOT median) — per-kernel **device** time,
     runs **hot-L2**. Use for isolating a single kernel and for L2-reuse-sensitive
     levers. For autotuning providers (Triton) always use **p10/min**, never median
     (median is polluted by autotune trial dispatches).
   - **Cold vs hot L2 can flip the verdict** on L2-reuse levers (e.g. any weight-
     caching remap). Quote both and say which the deployment resembles.
4. **One lever at a time, on a clone**, verified, measured, then promoted only after
   a controlled interleaved re-measurement. Never stack two unverified levers.
5. **Record both regimes** (uniform + skew) for every lever — several levers help
   one regime and hurt the other.

---

## 4. Baseline (established — the official gfx942 reference)

The current production kernel was brought up on gfx942 and the reference table
recorded. Run env: the **`jdbba-flydsl`** docker container, with the host `~/aiter`
checkout bind-mounted at **`/workspaces/meta/aiter`** (genrec already on the
container PYTHONPATH).

```bash
# confirm arch first
docker exec jdbba-flydsl bash -c 'rocminfo | grep -m1 gfx'   # expect gfx942

# both regimes (uniform + skew), flydsl vs triton, with correctness, out-of-box.
# The dispatch JSON is arch-keyed (arch-keyed-v1): the loader auto-selects the
# gfx942 section, so NO env override is needed and it will NOT core-dump on the
# gfx950 use_mfma_k32=true atom.
docker exec -e PYTHONPATH=/workspaces/meta/aiter \
  -w /workspaces/meta/aiter jdbba-flydsl \
  python3 op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py --regime both --metric time -test
```

> If `import aiter` fails with `cannot import name 'MxScaleRoundMode'`, the
> prebuilt core module is stale — wipe the build dir and reimport:
> `docker exec -w /workspaces/meta/aiter jdbba-flydsl bash -c 'rm -rf aiter/jit/build/module_aiter_core aiter/jit/module_aiter_core.so && python3 -c "import aiter"'`
> (a plain `AITER_REBUILD=1` reuses the ccache and does NOT fix it).

Capture per-shape per-regime FlyDSL ms, Triton ms, ratio, and cos. This is the
gfx942 starting point — do **not** compare it to any gfx950 number. The baseline
comes from the committed `by_arch.gfx942` section (empty winners → D-bucketed
heuristic, `use_mfma_k32=false`), which is the reproducible pre-tuning
configuration; record the measured numbers in your own results table (§8), not
here.

Also establish a **bound analysis for gfx942**: compute the HBM-traffic floor
(A read-once + C write-once + B re-reads) at MI300X's HBM bandwidth, and the
occupancy ceiling (VGPR + 64 KB LDS limited). This tells you whether the kernel is
memory-bound or compute/occupancy-bound on *this* hardware, which decides which
levers can possibly help.

---

## 5. Levers to evaluate (each a hypothesis, no prejudged outcome)

For each: implement on a clone, gate correctness, measure both regimes both timing
tools, then decide. The "what it does" is mechanism; the "test" is the question to
answer on gfx942.

| # | Lever | What it does | Test on gfx942 |
|---|---|---|---|
| 1 | **MMA atom** | bf16 MFMA tile size. gfx942 has only 16×16×16. | Confirm the 16×16×16 path compiles and is correct; this is the floor, not a choice. |
| 2 | **BLOCK_K** | reduction-tile depth (e.g. 128 for small K, 64 for large). | Sweep per shape; find the gfx942 K-loop depth that balances barriers vs occupancy. |
| 3 | **BLOCK_M / BLOCK_N** | output tile size. | Sweep; watch the 64 KB LDS ceiling (smaller than gfx950 → may force smaller tiles). |
| 4 | **Warp layout (m_warps / n_warps)** | distribution of the 4 warps over M/N. | Sweep warp counts and tile_n; the optimum is arch-specific. |
| 5 | **Pipeline depth (STAGES)** | software-pipeline buffers over the K-loop. | Test 2 vs 3+; depends on whether gfx942 is latency- or bandwidth-bound AND on the 64 KB LDS limit. |
| 6 | **XCD chiplet block-ID remap** | clusters a group's M-tiles onto one XCD for `Dense[b]` L2 reuse (knobs C, W). | Measure sign+magnitude in BOTH regimes; re-derive the C/W defaults for gfx942's XCD count. Do not assume the gfx950 gate. |
| 7 | **Epilogue C store** | how fp32 accumulators reach global C (scalar vs LDS-shuffle wide store). | Verify the wide-store epilogue is correct + a win on gfx942's store path. |
| 8 | **Persistent problem-visitor scheduler** | on-device CUM prefix + occupied-tile-only traversal (skew only). | Compare vs static-grid + early-exit on skew; decide per shape whether it earns its place on gfx942. |
| 9 | **A staging path** | global→LDS→reg vs global→reg, direct-to-LDS, async copy. | Test which staging the gfx942 memory pipeline prefers. |
| 10 | **waves_per_eu** | occupancy hint (`rocdl.waves_per_eu`). | Sweep 1–4; watch for register spills (VGPR pressure differs from gfx950). |
| 11 | **i64 offset math** | row-base offset in 64-bit before stride multiply. | **Correctness-critical, keep.** `seq_start·K` overflows i32 at large L (e.g. B1024_D512). Verify it is present and lowers on gfx942. |

Lever #11 is a correctness invariant (keep it). All others are open questions on
gfx942.

---

## 6. The two design decisions that MUST be re-derived per regime on gfx942

The dispatch (`jagged_dense_bmm_dispatch_v2.py`) makes two regime-dependent choices.
On gfx942 both must be re-measured from scratch; do not import the gfx950 settings.

1. **XCD remap on/off per regime** (lever #6). The remap trades `Dense[b]` L2 reuse
   against chiplet load balance. Both sides of that trade depend on XCD count and L2
   behavior, which differ on gfx942. Measure remap-ON vs remap-OFF for uniform and
   for skew, on all 4 shapes, with do_bench (cold-L2). Set the `uniform_seqlen`-gated
   default from *that* data.
2. **Persistent vs static under skew** (lever #8). Measure the persistent kernel vs
   the static-grid kernel (with whatever remap setting #1 chose) on each skew shape.
   Route to persistent only where it actually wins on gfx942.

A useful sanity check carried over from the prior campaign (methodology, not a
result): **read the competing Triton kernel's source** before attributing its
advantage to a "scheduler trick." If Triton uses the same static-grid + early-exit
algorithm, the gap is autotuning of tile/warp/stage params, not algorithm — which
redirects the search toward levers #2–#5/#10.

---

## 7. Files (production, hardware-neutral — already in the tree)

| File | Role |
|---|---|
| `aiter/ops/flydsl/kernels/jagged_dense_bmm.py` | N=K=128 prototype (validated reference) |
| `aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py` | generalized production kernel (per-shape factory; remap + atom + tiling knobs) |
| `aiter/ops/flydsl/kernels/jagged_dense_bmm_persist_dev.py` | persistent on-device problem-visitor kernel (skew candidate) |
| `aiter/ops/flydsl/jagged_dense_bmm_dispatch_v2.py` | production dispatch (arch-keyed JSON loader, regime gate + per-shape config) |
| `aiter/ops/flydsl/jagged_dense_bmm_dispatch_v2.json` | arch-keyed winners table (`by_arch.gfx942` = official baseline / `by_arch.gfx950`); loader auto-selects by detected arch |
| `op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py` | canonical perf bench (flydsl vs triton, headline shapes, `--regime uniform/skew/both`) |
| `op_tests/flydsl_tests/test_jdbba_dispatch_v2.py` | dispatch correctness test |
| `jagged_dense_bmm_broadcast_add_dev_journal.md` | methodology / running log |
| `jagged_dense_bmm_broadcast_add_flydsl_plan.md` | original FlyDSL port plan |
| `jagged_dense_bmm_triton_kernel_walkthrough.md` | Triton reference kernel walkthrough |

The per-lever experiment clones from the gfx950 campaign were archived in git
history (commit "archive MI355X experiment campaign") and removed from the working
tree; recover individual clones from there if a gfx942 lever wants the same scaffold.

---

## 8. Deliverable

A gfx942 results document (mirroring the structure the gfx950 report had, but with
MI300X numbers): baseline table, per-lever both-regime measurements, the two
re-derived regime gates (§6), and a final FlyDSL-vs-Triton standing for the 4
headline shapes in both regimes. Build it from measured gfx942 data only.
