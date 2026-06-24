# FlyDSL MLA decode reduce — multi-token decode fix (`decode_qlen > 1`)

**Status:** Implemented and verified on GPU (gfx942 / MI300X), 2026-06-23.
**Scope:** opt-in FlyDSL reduce path (`AITER_MLA_REDUCE_FLYDSL=1`) only. The default
HIP path (`aiter.mla_reduce_v1`) is untouched.

---

## 1. Problem

The opt-in FlyDSL port of the MLA decode reduce crashed with an **intermittent GPU
illegal memory access** whenever `decode_qlen > 1` (multi-token / MTP decode). The
default HIP kernel (`kn_mla_reduce_v1`) ran the exact same shapes and metadata clean,
so this was a defect in the FlyDSL port, not in the metadata or the HIP reduce.

Originally root-caused statically (see
`MLA-reduce-HIP-dispatch-stale-so-investigation.md` §3) to a hardcoded
`NTG = 1` / `max_seqlen_q` row-layout mismatch. On-GPU verification proved that
hypothesis **wrong for the production path** — the true cause was different (§3).

---

## 2. What was implemented

Two layers of change. The first (NTG / grid-y) was the planned approach; the second
(empty-tile guard) is the change that actually fixes the production fault.

### 2.1 Grid-y = `max_seqlen_q` + NTG-strided seq loop (planned approach)

Mirrors the HIP kernel's q-position parallelization
(`csrc/kernels/mla/reduce.cu:496`, grid-y from `max_seqlen_q`):

- **Kernel** (`aiter/ops/flydsl/kernels/mla_reduce.py`): replaced the single-seq
  `if valid_seq` block with a raw `scf.ForOp` strided loop
  `for seq in range(q_start + blockIdx.y, q_end, ntg)`, where `ntg = fx.grid_dim.y`
  is read at runtime. Built as a raw builder loop (not the AST `for`-rewriter) to
  avoid auto iter-args inference on the store-only body.
- **Launcher**: added a `max_seqlen_q` parameter; replaced `idx_ntg = fx.Index(1)`
  with grid-y = `max_seqlen_q`.
- **Wrapper** (`aiter/ops/flydsl/mla_reduce_kernels.py`): forwards `max_seqlen_q`
  into the launch; fixed the split-count derivation from
  `reduce_partial_map.numel() // num_reduce_tile` to the CSR-faithful
  `reduce_indptr[1] - reduce_indptr[0]`.
- **Tests** (`op_tests/test_flydsl_mla_reduce.py`): parameterized by
  `M = max_seqlen_q ∈ {1, 2, 4}`; added an `--M` CLI flag.

**Outcome of this layer alone:** the 144-case standalone matrix passed, but the
real end-to-end qlen=2 path *still faulted intermittently*. On the production path
`max_seqlen_q` is actually `1`, so grid-y stayed `1` and these changes were inert
here. They are retained as harmless and forward-correct (if a future metadata
revision widens `max_seqlen_q`), but they are **not** the fix.

### 2.2 Empty-tile guard (the actual fix)

`aiter/ops/flydsl/kernels/mla_reduce.py` (around line 239):

```python
# Skip empty/degenerate tiles (n_splits <= 1). The real decode metadata
# can hand us reduce_indptr rows of width 0 (no partials for this tile)
# whose reduce_final_map q-range is uninitialized garbage; HIP guards this
# with `reduce_tile_start == last` and `num_splits > 1` (reduce.cu:691,743).
# Clamp the loop's upper bound to its lower bound so a no-work tile runs
# zero iterations and never dereferences that garbage q-range / stores OOB.
has_work = arith.cmpi(arith.CmpIPredicate.sgt, n_splits, fx.Int32(1))
ub_seq = has_work.select(q_end, seq0)
```

and the loop upper bound uses `ub_seq` instead of `q_end`:

```python
lb = arith.index_cast(ir.IndexType.get(), _to_raw(seq0))
ub = arith.index_cast(ir.IndexType.get(), _to_raw(ub_seq))   # was q_end
st = arith.index_cast(ir.IndexType.get(), _to_raw(ntg))
```

When `n_splits ≤ 1`, `ub_seq == seq0 == lb`, so the seq-loop runs **zero
iterations** — the tile reads no partials, never dereferences the uninitialized
`reduce_final_map` q-range, and never stores. This mirrors HIP's two skip guards
exactly.

---

## 3. Root cause (confirmed by on-GPU arg dump)

A runtime dump of the real reduce arguments for `decode_qlen = 2`
(`-n16,2 -b 1 -c 21 -k 512`) showed:

| Arg | Value |
|---|---|
| `max_seqlen_q` | **1** (not 2 — multi-token is encoded as more tiles) |
| `num_reduce_tile` | 2 |
| `reduce_indptr` | **`[0, 0, 0]`** → every tile `n_splits = 0` (no partials this call) |
| `reduce_final_map` | **uninitialized garbage** (e.g. `[[1068941271, 1053081206], …]`) |

The FlyDSL kernel unconditionally read `reduce_final_map[tile]` → garbage
`q_start / q_end` → the seq-loop walked a garbage range and stored out of bounds.
It was **intermittent** because it depended on the garbage value: it passed when the
garbage happened to give `q_start ≥ q_end` (empty range), and faulted otherwise.
`AMD_SERIALIZE_KERNEL=3` + `HIP_LAUNCH_BLOCKING=1` masked it (changed timing/contents),
which is the classic signature that pointed away from the false NTG hypothesis.

HIP never hits this because it skips such tiles via
`reduce_tile_start == last_reduce_tile → return false` and `else if(num_splits > 1)`
(`csrc/kernels/mla/reduce.cu:691,743`).

**Why the standalone matrix missed it:** `op_tests/test_flydsl_mla_reduce.py` always
builds *well-formed* tiles (`n_splits ≥ 2`, valid `reduce_final_map`). It never
constructs the degenerate `n_splits = 0` / garbage-`reduce_final_map` tile that the
real `get_mla_metadata_v1` emits — and that degenerate tile is the sole trigger.

---

## 4. Verification (all clean, no faults)

Run inside the `mla_reduce_bench` container, `AITER_JIT_DIR=/tmp/aiter_jit`, cwd `/aiter`,
after clearing the reduce JIT cache to force a recompile.

| Config | FlyDSL | Result |
|---|---|---|
| bf16 qlen=2 | on | clean **3/3 repeats**, `decode:err = 0` (was exit-134 fault) |
| fp8 qlen=2 | on | clean, `decode:err ≈ 0.36` — fp8 quantization delta, not a reduce error (qlen=1 baseline 0.388) |
| bf16 qlen=1 | on | clean, regression-free |
| fp8 qlen=1 | on | clean, regression-free |
| bf16 qlen=2 | off (`FLYDSL=0`) | default HIP path unaffected, `decode:err = 0` |
| standalone matrix (`--matrix`, M∈{1,2,4} + degenerate) | — | **156 passed, 0 failed** |

### Stress runs (committed script, 2026-06-23)

72 e2e runs via `mla-reduce-docs/stress_flydsl_mla_reduce.sh`, all clean:

| Config | Repeats | Result |
|---|---|---|
| bf16 qlen=2 | 20 | 20 clean, 0 fault |
| fp8 qlen=2 | 20 | 20 clean, 0 fault |
| bf16 qlen=3 | 8 | 8 clean, 0 fault |
| bf16 qlen=4 | 8 | 8 clean, 0 fault |
| bf16 qlen=2 batch=4 | 8 | 8 clean, 0 fault |
| bf16 qlen=2 ctx=256 | 8 | 8 clean, 0 fault |

Representative result rows (preserved from the run logs that were since cleaned up):

```
# bf16 qlen=2, FLYDSL=1 (FIXED — was a fault):
| ctx 21 | bs 1 | nhead 16 | … | bf16 | bf16 | qlen 2 | decode:err 0 | 18.29us |

# fp8 qlen=2, FLYDSL=1 (FIXED):
| ctx 21 | bs 1 | nhead 16 | … | fp8 | fp8 | qlen 2 | decode:err 0.361572 | 16.30us |

# standalone matrix (--matrix):
=== 156 passed, 0 failed ===
```

### Reproduce

```bash
# E2e stress (72 runs, exits non-zero on any fault):
bash mla-reduce-docs/stress_flydsl_mla_reduce.sh

# Single e2e spot check:
docker exec -e AITER_MLA_REDUCE_FLYDSL=1 -e AITER_JIT_DIR=/tmp/aiter_jit -w /aiter \
  mla_reduce_bench python3 op_tests/test_mla_sparse.py \
  -n16,2 -b 1 -c 21 -k 512 -d bf16 -kvd bf16        # expect decode:err 0, no fault

# Standalone matrix + degenerate tripwire:
docker exec -e AITER_JIT_DIR=/tmp/aiter_jit -w /aiter \
  mla_reduce_bench python3 op_tests/test_flydsl_mla_reduce.py --matrix   # 156 passed
```

(`-n16,2` = nhead 16, decode_qlen 2; `-n16,1` = decode_qlen 1.)

---

## 5. Files changed

| File | Change |
|---|---|
| `aiter/ops/flydsl/kernels/mla_reduce.py` | empty-tile guard (`has_work`/`ub_seq`); NTG-strided seq loop; `max_seqlen_q` launcher arg + grid-y |
| `aiter/ops/flydsl/mla_reduce_kernels.py` | forward `max_seqlen_q`; fix split count to `reduce_indptr[1]-[0]` |
| `op_tests/test_flydsl_mla_reduce.py` | `M = max_seqlen_q ∈ {1,2,4}` coverage; `--M` flag |
| `csrc/kernels/mla/reduce.cu` | reference only (HIP skip guards at :691,743) — no edit |

## 6. Regression coverage (committed)

Two complementary regressions lock in the guard:

- **End-to-end stress script** — `mla-reduce-docs/stress_flydsl_mla_reduce.sh`. Reruns the
  real production path (`test_mla_sparse.py`, FlyDSL on) many times per config and counts
  GPU faults; exits non-zero on any fault. Covers the proven matrix (bf16/fp8 qlen=2 ×20,
  qlen 3/4, batch=4, ctx=256 — 52 runs, all clean). This is the production-path proof, made
  rerunnable. Run: `bash mla-reduce-docs/stress_flydsl_mla_reduce.sh`.
- **Standalone synthetic tripwire** — `op_tests/test_flydsl_mla_reduce.py --degenerate`
  (also folded into `--matrix`). Builds the exact degenerate tile (`reduce_indptr = [0,0,0]`
  + garbage `reduce_final_map`), runs the reduce, and asserts no fault and no write (a
  sentinel survives because the guarded seq-loop runs zero iterations). Fast, no stage-1
  attention; directly pins the guard line. The well-formed matrix cannot exercise the guard,
  so this is its only unit coverage.

**Negative control:** temporarily reverting the guard (`ub = q_end`) makes both the
`--degenerate` unit test and the stress script fault — confirming they actually exercise the
fix rather than passing vacuously.
