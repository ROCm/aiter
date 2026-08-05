# SILOTIGER-667 Handover Document

*Prepared 2026-07-22. Author: bartgips.*

---

## Getting started (for the new developer)

```bash
# 1. Pull the working branch (already pushed to origin)
git -C ~/code/aiter fetch origin
git -C ~/code/aiter checkout users/bartgips/silotiger-667-warp-decode

# 2. Activate the FlyDSL virtualenv (already built on this machine)
source /home/AMD/bartgips/code/FlyDSL/.venv/bin/activate

# 3. Run a quick correctness check
cd ~/code/aiter
PYTHONPATH=. FLYDSL_RUNTIME_ENABLE_CACHE=0 python -m pytest \
  op_tests/flydsl_tests/test_flydsl_moe_warp_decode.py \
  -k "f32path or (fp8w_dot2 and qwen3next) or fp4w_dot2" --tb=short -q

# 4. Run the benchmark harness vs CK
cd ~/code/aiter/tickets/667/harness
PYTHONPATH=~/code/aiter:$PYTHONPATH FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python bench_flydsl_wd.py --shapes deepseek-v3 minimax --batches 1 4 8
python compare.py --flydsl flydsl.txt --ck ck.txt

# 5. Build the CK reference bench if needed (binary already exists)
#    bash build_ck_bench.sh
#    CK bench binary: /home/AMD/bartgips/code/rocm-libraries-wdec/bench_ck_warp_decode
```

**Note on FlyDSL version:** The aiter `__init__.py` requires FlyDSL ≥ 0.2.4 but the
installed venv has 0.1.9. This causes an `AttributeError` when importing the full aiter
package. The test file and wrapper handle this gracefully by catching the exception and
loading kernels directly from file path — tests still pass. This will resolve when FlyDSL
is rebuilt (`bash ~/code/FlyDSL/scripts/build.sh`).

**Note on running the full test suite:** The full suite takes ~5–6 minutes and may be
killed by system resource limits. Use targeted `-k` filters as shown above.

---

## Repository layout

All code lives in the **aiter** repo, branch `users/bartgips/silotiger-667-warp-decode`.

| What | Path in aiter repo |
|------|-------------------|
| Kernel implementations | `aiter/ops/flydsl/kernels/moe_warp_decode.py` |
| aiter integration wrappers | `aiter/ops/flydsl/warp_decode_moe.py` |
| Tests | `op_tests/flydsl_tests/test_flydsl_moe_warp_decode.py` |
| Bench harness (timing + comparison) | `tickets/667/harness/bench_flydsl_wd.py` |
| CK comparison script | `tickets/667/harness/compare.py` |
| gfx950 benchmark results (2026-07-22) | `tickets/667/harness/flydsl_gfx950_20260722.txt` |
| gfx942 benchmark results | `tickets/667/harness/flydsl_gfx942.txt` |
| CK timing data | `tickets/667/harness/ck.txt` |
| CK bench binary | `/home/AMD/bartgips/code/rocm-libraries-wdec/bench_ck_warp_decode` |
| CK source | `/home/AMD/bartgips/code/rocm-libraries-wdec/projects/composablekernel/include/ck_tile/ops/warp_decode/kernel/` |

---

## The problem and approach

**Goal:** Warp-decode MoE MLP kernels in FlyDSL targeting AMD MI355X (gfx950), optimised
for very small batch sizes (B = 1–4 tokens). At B=1 an MFMA 16×16 tile is 93% empty;
warp-decode avoids this by assigning **one GPU wavefront (64 lanes) to one output scalar**,
using scalar dot products and a butterfly-shuffle reduction instead of matrix tiles.

**CK reference:** branch `users/samremes/ck/warp-decode`, commit `62e30c9098`.

### Two kernel stages

**Stage 1 — gate_up** (`compile_wd_moe_gate_up`):
- Grid: `B × TOPK × INTER` blocks (one per output neuron)
- Each wave splits HIDDEN across 64 lanes, computes `silu(x @ W_gate) * (x @ W_up)`,
  butterfly-reduces, lane 0 writes one BF16 value to `inter_out[B*TOPK, INTER]`

**Stage 2 — down_reduce** (`compile_wd_moe_down_reduce`):
- Grid: `B × (HIDDEN / h_per_warp)` blocks
- Each wave splits INTER across 64 lanes, iterates TOPK expert slots accumulating
  router-weighted dot products, butterfly-reduces, lane 0 atomicAdds to `y_out[B, HIDDEN]`

**Key instructions (gfx950):**
- `v_dot2_f32_bf16` — 2 MACs/lane/cycle
- `v_cvt_scalef32_pk_bf16_fp8` / `v_cvt_scalef32_pk_bf16_fp4` — weight conversion

---

## What was built

### gate_up variants

| `w_dtype` | Compute path | Hardware | Notes |
|-----------|-------------|----------|-------|
| `"bf16"` + `use_dot2=False` | f32 scalar | gfx942+gfx950 | Correctness baseline |
| `"bf16"` + `use_dot2=True` | Dot2 software prefetch | gfx950 | 14 ForOp iter_args; 1.93× on qwen B=8 |
| `"fp8"` | Batched dot2 | gfx950 | Per-tensor scale |
| `"fp4"` | Batched dot2 | gfx950 | Per-tensor E2M1 scale; `v_cvt_scalef32_pk_bf16_fp4` |

Also built but **not recommended**: `compile_wd_moe_gate_up_splitk` + `compile_wd_moe_gate_finalize`
(two-phase split-K) — benchmarked at 1.7× slower due to launch overhead.

### down_reduce variants

| `w_dtype` | `h_per_warp` | Other parameters | Hardware |
|-----------|-------------|-----------------|----------|
| `"bf16"` | 1 or 2 | f32 + **batched** dot2 | gfx942+gfx950 |
| `"fp8"` | 1 or 2 | **batched** dot2, per-tensor scale | gfx950 |
| `"fp4"` | 2 | **MXFP4**, block_k=32, e8m0 per-block scales | gfx950 |
| any | any | `k_batch` split-K | arch-agnostic |
| `"bf16"` f32 only | any | `n_waves` LDS cooperative x-load | gfx942+gfx950 |

H2 (`h_per_warp=2`) computes two adjacent output channels per wave, reusing the same
activation loads for both weight rows — doubles effective weight bandwidth per activation read.

### aiter integration

`aiter/ops/flydsl/warp_decode_moe.py` provides two arch-aware wrappers exported from
`aiter/ops/flydsl/__init__.py`:
- `flydsl_wd_moe_gate_up(x, w_gate, w_up, router_ids, B, topk, inter, hidden, experts, ...)` → `inter_out`
- `flydsl_wd_moe_down_reduce(inter_out, w_down, w_scale, router_ids, router_wts, ...)` → `y_out`

Both auto-detect gfx950 vs gfx942 and pick optimal dtype/path.

---

## Performance results (gfx950, 2026-07-22)

### gate_up FP8 vs CK `gate_bf16_d2`

| shape | B | FlyDSL (ms) | CK (ms) | Gap |
|-------|---|-------------|---------|-----|
| deepseek-v3 | 8 | 0.578 | 0.363 | **1.59×** |
| minimax | 8 | 0.195 | 0.116 | **1.68×** |
| qwen3next | 8 | 0.086 | 0.030 | **2.88×** |
| qwen3next | 1 | 0.091 | 0.006 | **~15×** (occupancy) |

### down_reduce H2 FP8 vs CK `down_h2_d2`

| shape | B | FlyDSL (ms) | CK (ms) | Current gap | Before batched fix |
|-------|---|-------------|---------|------------|-------------------|
| deepseek-v3 | 8 | 0.305 | 0.224 | **1.36×** | ~~2.83×~~ |
| minimax | 8 | 0.108 | 0.076 | **1.42×** | ~~3.06×~~ |
| qwen3next | 8 | 0.099 | 0.020 | **5.0×** | (occupancy) |

### down_reduce H2 FP4 vs CK `down_fp4_h2`

| shape | B | FlyDSL FP4 | FlyDSL FP8 | CK FP4 | Gap vs CK |
|-------|---|-----------|-----------|--------|-----------|
| deepseek-v3 | 8 | 0.262 ms | 0.305 ms | 0.124 ms | **2.11×** |
| minimax | 8 | 0.100 ms | 0.108 ms | 0.036 ms | **2.80×** |

FP4 is ~1.2× faster than FP8 at large B — right direction — but still 2.1× slower than CK.

---

## Key investigations and outcomes

### ✅ Load batching (biggest single win)

**Problem:** `_dot2_dep` embedded `s_waitcnt vmcnt(0)` + `s_nop 2` per dot2 call.
With 8 dot2s per k-step, 8 full stalls serialised all loads. HBM utilisation: 44%.

**Fix:** Issue ALL loads for a k-step → one `vmcnt0` → 4+ independent `_dot2_batched`
calls starting from `zero_f32` (not a chain) → one `s_nop 2` drain. Reduce partial sums
then add to running accumulator.

**Result:** gate_up 44%→65% HBM. Down_reduce: deepseek B=8 gap 2.83×→1.36×; minimax 3.06×→1.42×.

**Commits:** `4b647d23e` (gate_up), `4fe34cce3` + `e7d18ad5c` (down_reduce)

### ✅ BF16 software prefetch for gate_up

**Problem:** Even with batched loads, idle time between k-steps.

**Fix:** Prologue-loop-epilogue structure. The for-loop body issues step k+1's loads
while computing step k. Loaded VGPRs carried as ForOp iter_args (14 total).

**Result:** qwen3next B=8: 0.124→0.064 ms (1.93×). No benefit for deepseek (already saturated).

**Why not FP8:** FP8 prefetch attempted; the same FP8 word SSA is used for both sel=0 and
sel=1 conversions. LLVM CSEs duplicate loads, so structural aliasing cannot be avoided.
Kept FP8 batched.

**Commit:** `81e004f29`

### ✅ LLVM register aliasing fix for H2 dot2

**Problem:** H2 down_reduce (h_per_warp=2) produced wrong results for the h=1 output
channel at n_k_steps ≥ 4 (deepseek). ISA dump showed LLVM assigned the h=0 dot2 output
to x_word's VGPR, corrupting h=1's computation.

**Root cause:** x_word SSA was shared between h=0 and h=1 dot2 calls. With `=v`
(non-early-clobber) output constraint, LLVM could alias the h=0 output with x_word.

**Fix:** (1) Use inner `scf.ForOp` for k_step loop (reduces register pressure); (2) load
x_word separately per h so each SSA value is used only once — LLVM can alias the dead h=0
x_word register for h=0's dot2 output without corrupting h=1.

**Note:** Early-clobber `=&v` was tried but LLVM then reused byte-offset VGPRs — reverted.

**Commit:** `e7897ef90`

### ✅ MXFP4 down_reduce

**What:** `w_dtype="fp4"` for down_reduce with MXFP4 block-scale format.

- `block_k=32`: one e8m0 uint8 scale per 32 weight elements per output channel
- Scale tensor: `[E*HIDDEN, INTER//32]` uint8, separate kernel arg
- e8m0 → f32: `arith.bitcast(f32, arith.shli(arith.extui(i32, scale_byte), 23))`
- Scale index per lane: `K * 16 + lane // 4` (for k_vector=8, block_k=32)
- 4 independent dot2 accumulators + one `s_nop 2` drain (mirrors CK's drain4 pattern)

**Result:** 100% correct. FP4 is 1.2× faster than FP8 at B=8. Still 2.1× slower than CK
(same structural vmcnt limitation as gate_up).

**Commit:** `571e4ca7a`

### ✅ LDS inter_states caching for down_reduce (f32 path)

`n_waves` parameter: all n_waves×64 threads cooperatively load the activation row into LDS,
then each wave reads from LDS. Gives 10-17% improvement for deepseek/minimax at B≥2.

**Limitation:** Only integrated for the f32 scalar path. Dot2 path needs separate lgkmcnt
management (deferred). The `_use_lds` flag is `n_waves > 1 and not use_dot2`.

**Commit:** `0a572dd68`

### ⚠️ gate_up n_waves / LDS x-caching (REVERTED)

**Tried:** Same n_waves cooperative x-load for gate_up. Produces ~25-47% accuracy
on both BF16 f32 and FP8 paths. Root cause unresolved.

**DO NOT retry** without debugging the LDS write/read ordering. The `n_waves` parameter
remains in the signature for future use but `_use_lds_gu` evaluates to False.

### ⚠️ NPerWarp=2 gate_up (STASHED)

**Tried:** Compute 2 adjacent neurons per wave with 4 independent dot2 chains, sharing x loads.

- BF16 f32 path: works correctly.
- FP8 path: 75-91% accuracy. With 16 `has_side_effects=False` dot2/conversion ops per
  ForOp body, LLVM can reorder them such that `s_nop 2` no longer covers all dot2
  write→read hazards. Pre-conversion and early-clobber were tried; same issue persists.

**Status:** `git stash` in aiter repo. Retrievable via `git stash pop`.

### ✅ CK source study (key insights)

From `warp_decode_numeric.hpp` and `warp_decode_gate_up_kernel.hpp`:

1. **Why CK doesn't need explicit vmcnt:** CK uses `load_tile()` hardware intrinsics;
   LLVM's `SIInsertWaitcnts` pass auto-inserts `s_waitcnt`. Our inline asm bypasses this.

2. **Tied accumulation (`"0"(dot)` constraint):** Output and accumulator share the same
   physical register. Prevents LLVM from aliasing output with any other live operand.
   We implemented `_dot2_acc_nonop` — useful helper but not needed for the final fixes.

3. **4-accumulator drain4:** CK's FP4 down uses 4 independent dot2 chains + single drain.

---

## Remaining gaps and how to close them

### gate_up 1.6–1.7× gap (large shapes)

**Root cause:** CK uses `load_tile()` intrinsics; LLVM auto-schedules vmcnt. Our inline
asm approach requires explicit `_vmcnt_n()` stalls. Without FlyDSL gaining native tile
load support, this gap is structural.

**Viable options:**
- **FP8 software prefetch:** BF16 version works (1.93×). FP8 fails due to shared FP8 SSA
  across sel=0/1 conversions. Potential fix: restructure the inner ForOp to avoid the shared
  SSA without CSE — unexplored.
- **NPerWarp=2 FP8:** See above. Possibly fixable by splitting into two independent ForOps
  (one per neuron), losing x-sharing but avoiding the reordering issue.

### qwen3next small-grid gap (~3–15× at B=1)

**Root cause:** Grid = 5120 blocks / 256 CUs = 20 waves/CU. Pure occupancy starvation.

**Viable options:**
- Fix gate_up `n_waves` LDS bug (unknown root cause; try explicit lgkmcnt after each
  ds_write and before each ds_read)
- XCD swizzle: neutral on large deepseek, untested on small-grid qwen specifically

### down_reduce FP4 2.1× gap

Same vmcnt pipelining constraint as gate_up. Viable: prefetch-style overlap of scale loads
with computation; LDS for scales if they fit.

### dot2 path + LDS for down_reduce

Integrate `n_waves` LDS into the dot2 path by adding explicit lgkmcnt after ds_reads
before the dot2 instruction. Pattern:
```
ds_read x_word         → lgkmcnt pending
buffer_load w_word     → vmcnt pending
s_waitcnt vmcnt(0)     → wait for w_word
_lgkmcnt0()            → wait for x_word  (or combined: s_waitcnt vmcnt(0) lgkmcnt(0))
_dot2_batched(acc, x_word, w_bf16)
```

---

## Critical API notes

### down_reduce launch signature (updated 2026-07-22)

The kernel now requires `arg_w_scale` as the **4th argument, before `router_ids`**:

```python
exe(
    _ptr(y_out),           # [B, HIDDEN] f32, zero-initialised before launch
    _ptr(inter_states),    # [B*TOPK, INTER] bf16
    _ptr(w_down),          # weights (bf16/fp8/fp4)
    _ptr(w_scale),         # [E*HIDDEN, INTER//32] uint8  (FP4: e8m0 per-block scales)
                           #                              (BF16/FP8: pass any 1-byte dummy buffer)
    _ptr(router_ids),      # [B*TOPK] i32
    _ptr(router_wts),      # [B*TOPK] f32
    B, topk, inter, hidden, experts,
    w_scale_pt,            # float: FP8 per-tensor scale (use 1.0 for BF16 and FP4)
    stream,
)
```

Use `_dummy_scale_ptr()` (see test file) for non-FP4 paths:
```python
_DUMMY_SCALE_BUF = torch.zeros(1, dtype=torch.uint8, device="cuda")
def _dummy_scale_ptr(): return flyc.from_c_void_p(fx.Uint8, _DUMMY_SCALE_BUF.data_ptr())
```

---

## Commit history (key milestones on this branch)

| Commit | Description |
|--------|-------------|
| `4b647d23e` | Batched loads for gate_up: 44%→65% HBM utilisation |
| `81e004f29` | BF16 dot2 software-prefetch for gate_up (1.93× on qwen B=8) |
| `f037b6a08` | H2 layout for down_reduce (2 outputs/wave) |
| `4f0dc7d49` | FP4 gate_up via `v_cvt_scalef32_pk_bf16_fp4` (per-tensor scale) |
| `bba8e30a1` | FP8 weights for down_reduce |
| `e7897ef90` | **Fix H2 dot2 LLVM register aliasing** (26% wrong → 100% correct) |
| `0a572dd68` | LDS inter_states caching for down_reduce (n_waves, f32 path) |
| `605231514` | aiter integration wrapper (`warp_decode_moe.py`) |
| `7b4472f33` | `_dot2_acc_nonop` tied-accumulation helper (CK pattern) |
| `4fe34cce3` | **Batched independent dot2 for down_reduce** (biggest win) |
| `e7d18ad5c` | Fix H2 aliasing in batched down_reduce + FP8 pre-conversion fix |
| `571e4ca7a` | **MXFP4 down_reduce** (per-block e8m0 scales, block_k=32) |

---

## Further reading

- CK warp-decode design doc: `/rocm-libraries-wdec/projects/composablekernel/include/ck_tile/ops/warp_decode/WARP_DECODE_MOE_KERNELS.md`
- FlyDSL kernel authoring guide: `~/code/FlyDSL/docs/kernel_authoring_guide.md`
- Persistent session notes: `~/.claude/projects/-home-AMD-bartgips-code-FlyDSL/memory/project_667_harness.md`
