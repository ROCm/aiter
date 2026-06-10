# FlyDSL Implementation Plan — `jagged_dense_bmm_broadcast_add` (jdbba)

_A concrete, source-grounded plan to build the AMD FlyDSL forward kernel for Meta's HSTU
`jagged_dense_bmm_broadcast_add`. Forward-only, BF16 in / FP32 accumulate / BF16 out,
broadcast bias, MI300X (gfx942) first._

This plan is written against the **actual** FlyDSL source in this tree:

| What | Where |
|---|---|
| Generic HGEMM we fork | `aiter/ops/flydsl/kernels/splitk_hgemm.py` → `compile_hgemm_kernel` (line 108) |
| Varlen device pattern we copy | `aiter/ops/flydsl/kernels/chunk_gated_delta_h.py` (lines 337–347, grid at 1033) |
| Unified config surface | `aiter/ops/flydsl/kernels/hgemm_dispatch.py` |
| Tensor wrapper API | `aiter/ops/flydsl/kernels/tensor_shim.py` → `GTensor` (line 284) |
| Reference semantics | `~/generative-recommenders/.../triton_jagged.py::jagged_dense_bmm_broadcast_add_kernel` (line 294) |

Companion docs: `jagged_dense_bmm_broadcast_add_dev_journal.md` (current source of truth),
`jagged_dense_bmm_triton_kernel_walkthrough.md` (the reference kernel explained),
`jagged_dense_bmm_broadcast_add_sami_plan.md` (original design doc).

---

## 0. TL;DR — the whole plan in one breath

The generic FlyDSL `compile_hgemm_kernel` is **already a dynamic-M, has-bias, broadcast-bias
GEMM** (`C[m,n] = A[m,k] @ B[n,k]ᵀ + bias[n]`). We fork it into `compile_jdbba_kernel` and add
**exactly three things**, all cheap per-block scalar/pointer ops — the same three the dev
journal calls "new vs a normal GEMM":

1. **A group axis on the grid** (`block_idx.z = b`).
2. **Device-side group resolution** — read `seq_offsets[b]`/`[b+1]`, compute `M_b`, early-exit
   tail tiles.
3. **Per-group rebasing** — offset A/Out rows by `seq_start`, offset B/bias by group `b`.

Everything else (LDS staging, swizzle, MFMA, pipelining, masked store, bias add) is **reused
unchanged**. We pre-transpose `Dense (B,K,N) → (B,N,K)` once on the host so the existing
`A @ Bᵀ` mainloop is correct with zero kernel changes to the B-load path.

```
  Generic hgemm:   C[m,n]      = A[m,k]      @ B[n,k]ᵀ      + bias[n]
                    │ one M      │ one A       │ one B         │ one bias
                    ▼            ▼             ▼               ▼
  jdbba (per b):   Out[s:e,n]  = Jag[s:e,k]  @ Dense[b][n,k]ᵀ + Bias[b][n]
                    └ M_b from   └ rebased by  └ selected by    └ selected by
                      seq_offsets  seq_start     group b          group b
```

---

## 1. What we are computing (the contract)

For each group `b ∈ [0, B)`, over its row slice `[s, e) = [seq_offsets[b], seq_offsets[b+1])`:

```
  Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
                (M_b × N)        (M_b × K)  (K × N)   (1 × N broadcast)
```

| Tensor | Role | Shape (logical) | Layout | Dtype |
|---|---|---|---|---|
| `jagged` (A) | packed input | `(L, K)`, `L = Σ M_b` | row-major | bf16 |
| `dense` (B) | per-group weight | `(B, K, N)` | row-major | bf16 |
| `bias` | per-group bias | `(B, N)` broadcast over rows | row-major | bf16 |
| `out` | packed output | `(L, N)` | row-major | bf16 |
| `seq_offsets` | group boundaries | `(B+1,)` prefix sum | device | int32/int64 |

**The one hard constraint:** `seq_offsets` is **device-resident**. The host does not know any
`M_b` at launch, so group→row resolution happens **on the GPU** (mirrors Triton).

> ⚠️ **Benchmark naming clash (do not get burned):** HSTU bench labels shapes `(B,D,K,N)`
> where bench `D` = reduction `K`, bench `K` = output `N`, bench `N` = `max_seq_len`. This plan
> uses **standard GEMM names** (`K` = reduction, `N` = output width), matching the kernel
> source. See dev journal §5.

---

## 2. Why the generic hgemm is a near-perfect base

Reading `compile_hgemm_kernel` (splitk_hgemm.py) line by line, these features already exist and
map 1:1 onto jdbba:

| jdbba needs… | Generic hgemm already has it | Line |
|---|---|---|
| Dynamic M (unknown at compile) | `m: fx.Int32` runtime arg; `A_`/`C_` shaped `(-1, …)` | 249, 257, 259 |
| Don't read/write past real rows | `row_idx < m` boundary `select` on load **and** store | 444–448, 1014 |
| FP32 accumulation | `c_frags` are `f32` MFMA accumulators | 255, 307 |
| Broadcast bias add `(N,)` | `HAS_BIAS` epilogue `vec = vec + bias_vec` | 1024–1031 |
| BF16 cast on store | `val.truncf(dtype_)` into `cs_`, then `C_.vec_store` | 954, 1029 |
| `A @ Bᵀ` with B as `(N,K)` | `B_ = GTensor(B, shape=(n,k))`; `ldg_matrix_b((n_idx,k_idx))` | 258, 582 |
| Tile/grid fold | `grid=(bm*N_BLOCKS, SPLIT_K, 1)`; `pid//N_BLOCKS`, `pid%N_BLOCKS` | 1062, 287–291 |

So the generic kernel is literally **a single jdbba group** already, *if* we (a) hand it the
right `m`, (b) point it at the right A/B/bias/out slices, and (c) replicate it across groups.

What it is missing for jdbba:
- No notion of a **group** (no `seq_offsets`, no `b` axis).
- A/B/bias/out are addressed from tensor base 0 — no **per-group rebasing**.
- B is expected as `(N,K)`; Meta gives `(K,N)`.

Everything missing is in §3's three changes.

---

## 3. The kernel: `compile_jdbba_kernel` (fork of `compile_hgemm_kernel`)

New file: `aiter/ops/flydsl/kernels/jdbba_hgemm.py`. Start as a **verbatim copy** of
`splitk_hgemm.py`, then apply the diffs below. Keep `SPLIT_K=1` (dev journal §5: abundant tiles
⇒ split-K is counter-productive), so we can **delete** the `IS_SPLIT_K` / `zero_c` /
`split_k_barrier` / semaphore machinery to keep the fork lean (or just leave `SPLIT_K=1` and
let those branches stay dead — recommended for the first cut to minimize diff risk).

### 3.0 New signature

```python
@flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
def jdbba_kernel(
    C: fx.Pointer,            # out  (L, N)
    A: fx.Pointer,            # jagged (L, K)
    B: fx.Pointer,            # dense PRE-TRANSPOSED to (B_groups, N, K)
    BIAS: fx.Pointer,         # (B_groups, N)
    SEQ_OFFSETS: fx.Pointer,  # (B_groups+1,) int32   <-- NEW
    n_groups: fx.Int32,       # B_groups (for grid bookkeeping only)  <-- NEW
):
    ...
```

Note: `m` is **gone** from the signature — it is now computed per group from `SEQ_OFFSETS`.

### 3.1 Change #1 — group axis on the grid

The generic kernel folds (M-tile, N-tile) into `block_idx.x` and uses `block_idx.y` for
split-K. We add the group on **`block_idx.z`** and size the M extent by `max_seq_len`
(exactly Triton's grid; walkthrough §3):

```python
# grid = (cdiv(max_seq_len, BLOCK_M) * N_BLOCKS,  1,  B_groups)
off_b      = fx.Int32(fx.block_idx.z)                      # which group   <-- NEW
pid_mn     = fx.block_idx.x
block_m_idx = pid_mn // N_BLOCKS                            # row-tile within group
block_n_idx = pid_mn %  N_BLOCKS                            # col-tile
```

`N_BLOCKS = n // TILE_N` stays a compile-time constant (N is static across groups).

### 3.2 Change #2 — device group resolution + early-exit

Copy the varlen prologue from `chunk_gated_delta_h.py:337` and add the early-exit from the
Triton kernel (walkthrough §4 step A):

```python
SEQ_ = GTensor(SEQ_OFFSETS, dtype=T.i32, shape=(-1,))
seq_start = SEQ_[fx.Index(off_b)]                          # int32
seq_end   = SEQ_[fx.Index(off_b) + fx.Index(1)]
M_b       = seq_end - seq_start                            # this group's row count

start_m = block_m_idx * BLOCK_M
# EARLY EXIT: this row-tile fell off the end of a short group → nothing to do.
if (start_m >= M_b).ir_value():                            # dynamic-if, see §6 risk
    return
```

`M_b` now plays the role the old runtime `m` played: it is the boundary for every
`row_idx < M_b` mask the generic kernel already emits. **Rename `m` → `M_b` everywhere** in the
copied body (load mask at 444–448, async load mask at 501–505, store boundary at 1014).

> int64 subtlety (dev journal §3): read `seq_offsets` as int32, but compute the **row-base byte
> offset in int64** (`L` can exceed 2³¹). Concretely, build `row_base_i64 =
> i64(seq_start) + i64(local_row)` before multiplying by the row stride. Mirrors Triton's
> `.to(tl.int64)`.

### 3.3 Change #3 — per-group rebasing of A / Out / B / bias

The generic kernel addresses every tensor from base 0. For jdbba each group reads/writes a
different slice. We keep **one buffer resource per tensor** (covering the whole tensor) and just
add a per-group offset to the element indices — `GTensor.rsrc` already spans the full tensor, so
the buffer-load offset is all that needs to move. Two flavors:

**A (jagged) and Out — rebase by ROW (`seq_start`):** use *absolute* jagged rows. The generic
code computes `row_idx = m_offset + m_local_idx`. Change `m_offset` so it includes the group's
row base:

```python
m_offset = fx.Index(seq_start) + fx.Index(block_m_idx * BLOCK_M)   # was: block_m_idx*BLOCK_M
```

With this single change, every existing `A_.vec_load((row_idx, col))` and
`C_.vec_store((row_idx, …))` automatically targets the right rows, because `A_`/`C_` are shaped
`(-1, k)` / `(-1, n)` over the **whole** packed tensor. The boundary `row_idx < (seq_start +
M_b) == seq_end` — so compare against `seq_end`, not `M_b`, on the **global** row index:

```python
# boundary becomes: absolute row < seq_end
safe = arith.cmpi(ult, row_idx, fx.Index(seq_end))
```

(Equivalently keep `m_offset = block_m_idx*BLOCK_M`, compare `< M_b`, and add `seq_start*stride`
as a separate base. Either is fine; the absolute-row form reuses the most existing code.)

**B (dense) and bias — rebase by GROUP (`b`):** add a flat group offset to the B/bias indices.
B is `(B_groups, N, K)` after pre-transpose, so group `b` starts at element `b*N*K`:

```python
b_elem_base    = fx.Index(off_b) * fx.Index(n * k)     # into (B,N,K) flat
bias_elem_base = fx.Index(off_b) * fx.Index(n)         # into (B,N)   flat
```

Then in `ldg_matrix_b` / `ldg_sts_b_async_one`, add `b_elem_base` to the computed B linear
offset (one extra add), and in the epilogue bias load add `bias_elem_base`:

```python
# was: bias_vec = BIAS_.vec_load((n_offset + n_local_idx,), LDG_VEC_SIZE)
bias_vec = BIAS_.vec_load((bias_elem_base + n_offset + n_local_idx,), LDG_VEC_SIZE)
```

Because `B` is now `(B,N,K)` we can either (a) keep `B_ = GTensor(B, shape=(n,k))` and add
`b_elem_base` to the linear offset, or (b) declare `B_ = GTensor(B, shape=(-1, k))` and index
rows `b*N + n`. Option (b) is the cleaner mental model (B as a tall `(B*N, K)` matrix) and keeps
the `(n_idx, k_idx)` load shape identical with `n_idx → b*N + n_idx`. **Prefer (b).**

### 3.4 What stays 100% unchanged

LDS allocation & swizzle (`swizzle_xor16`), the `B_TO_LDS` async-copy path, the `STAGES`
pipeline (`hot_loop_scheduler`, `ldmatrix_compute_tile_streaming`,
`async_copy_ldmatrix_compute_tile_streaming`), the MFMA atoms (`WmmaHalf_*`), the FP32→BF16
`truncf`, and the masked store loop (1009–1032). **We touch only offsets and the prologue.**

### 3.5 Bias-precision note (small, worth tracking)

Triton adds bias in **FP32** (accumulator domain). The generic hgemm epilogue adds it in
**bf16** (`vec` comes from `cs_`, a bf16 LDS staging buffer, line 1019). For broadcast bias on
HSTU shapes this is almost certainly fine, but it is a **known numeric difference**. If
validation (§7) shows bias-dominated error, promote the bias add to fp32: load `bias_vec`,
`extf` to f32, add to an f32 view of `vec` before the final `truncf`. Track as a follow-up, not
a blocker.

---

## 4. Host side — pre-transpose, launcher, torch op

### 4.1 The Dense pre-transpose (do it once, off the hot path)

The generic mainloop computes `A @ Bᵀ` with B stored `(N,K)`. Meta's `dense` is `(B,K,N)`.
Transpose to `(B,N,K)`:

```python
dense_t = dense.transpose(1, 2).contiguous()   # (B,K,N) -> (B,N,K)
```

Dev journal §4: weights are static, so this is a cheap one-time cost (or fold into the model's
weight-prep). Do **not** transpose per call in production — cache it.

### 4.2 `@flyc.jit` launcher

Mirror `launch_hgemm_kernel` (splitk_hgemm.py:1035) and `launch_gdn_h`
(chunk_gated_delta_h.py:992). Grid carries the group on z:

```python
@flyc.jit
def launch_jdbba(C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=fx.Stream(None)):
    allocator.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        allocator.finalize()
    bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    jdbba_kernel(C, A, B, BIAS, SEQ_OFFSETS, n_groups).launch(
        grid=(bm * N_BLOCKS, 1, n_groups),     # (M-tiles*N-tiles, 1, B_groups)
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
    )
```

No `semaphore`/`signal` needed (SPLIT_K=1).

### 4.3 Torch-facing op + cache

There is **no** torch caller of `compile_hgemm_kernel` in-tree yet (only `hgemm_dispatch.py`
re-exports the compiler), so we build the host op fresh. Mirror the layering note at
chunk_gated_delta_h.py:1042 — keep `torch` out of the kernel module; put the torch wrapper in a
sibling, e.g. `aiter/ops/flydsl/jagged_dense_bmm.py`:

```python
@functools.lru_cache(maxsize=...)
def _build(dtype, n, k, **cfg):
    return compile_jdbba_kernel(dtype, n, k, **cfg)

def jagged_dense_bmm_broadcast_add(jagged, dense_t, bias, seq_offsets, max_seq_len):
    L, K = jagged.shape
    Bg, N, _ = dense_t.shape            # dense_t already (B, N, K)
    out = torch.empty((L, N), dtype=jagged.dtype, device=jagged.device)
    launch = _build("bf16", N, K, **pick_config(N, K, max_seq_len))
    launch(out, jagged, dense_t, bias, seq_offsets, Bg, max_seq_len,
           stream=current_stream())
    return out
```

Register it through aiter's normal flydsl op path (follow how `gdr_decode` /
`linear_attention_prefill_kernels` expose theirs — see the layering comment cited above).

---

## 5. Config & tuning surface

Reuse the generic kernel's compile knobs; jdbba inherits the autotune space, just re-centered
for the HSTU short-K regime (dev journal §5–§6: `K∈{256,512}`, `N∈{256,512}`, many M-tiles,
overhead-bound).

| Knob | Generic default | jdbba starting point | Why |
|---|---|---|---|
| `TILE_M` | 128 | **128** (try 256) | many rows per group (`M_b≈7800`); fat M-tiles amortize fixed cost |
| `TILE_N` | 128 | **= N** (256 or 512) if LDS allows, else 128 | journal §6 lever 1: cover whole N → fewer epilogues |
| `TILE_K` | 64 | **64** | K is only 256–512 ⇒ 4–8 K-steps; bigger BLOCK_K kills steady state |
| `SPLIT_K` | 1 | **1** (fixed) | abundant tiles ⇒ split-K counterproductive (journal §5) |
| `STAGES` | 2 | **2**, sweep 3 | short K-loop limits prefetch depth |
| `B_TO_LDS` | False | **True** | needed for the async pipeline + future B-stationary lever |
| `BLOCK_M/N_WARPS` | 2×2 | sweep 2×2, 2×4 | from generic autotune list |

Constraints to respect from the compiler asserts (splitk_hgemm.py:123–136, 182): `n % TILE_N ==
0`, `k % (SPLIT_K*BLOCK_K) == 0`, `BLOCK_K_LOOPS >= STAGES`. For `K=256, TILE_K=64` →
`BLOCK_K_LOOPS=4 ≥ STAGES`, so `STAGES≤4`. For `K=256`, `STAGES=3` is the realistic ceiling.

**Autotune key** (matches Triton's, walkthrough §7): `(bucket(max_seq_len), N, K)`. Cache built
launchers by this key.

---

## 6. Risks & open implementation questions

| Risk | Detail | Mitigation |
|---|---|---|
| **Dynamic `if return` in FlyDSL** | The early-exit is a *runtime* branch on `M_b`. FlyDSL's AST rewriter is picky about dynamic `if` (see the `ReplaceIfWithDispatch` notes at chunk_gated_delta_h.py:768). A bare `return` inside a dynamic `if` may not lower cleanly. | Two fallbacks: (a) **predicate the whole body** — wrap work in `scf.IfOp(start_m < M_b)`; (b) **clamp instead of exit** — `safe_row = (row<seq_end).select(row, 0)` already prevents OOB, so a fully-launched tile that does redundant masked work is *correct but wasteful*. On the uniform bench (`M_i` a tile multiple) there are **no** tail tiles, so the early-exit is a perf-only optimization — ship correctness first (clamp), add exit second. |
| **int64 row base** | `L` can exceed 2³¹; element offsets must be i64. | Build row-base offset in i64 before stride multiply (§3.2). Verify with a large-`L` test (§7). |
| **B as one big buffer resource** | Indexing `(b*N + n, k)` over a `(-1,k)` GTensor must stay within `create_buffer_resource` max-size. | `GTensor` uses `max_size=True` (tensor_shim.py:302) → whole-tensor resource, fine. Confirm offsets are i32-safe or promote (B*N*K for `B=1024,N=512,K=512` ≈ 2.7e8 elems < 2³¹, OK; but `*DTYPE_BYTES` in byte offsets can overflow i32 — keep byte offsets i64). |
| **Bias precision** | bf16 vs Triton's fp32 add (§3.5). | Validate; promote to fp32 only if needed. |
| **Empty groups (`M_b=0`)** | `start_m >= 0 == M_b` → every tile must exit. | Covered by the early-exit / clamp; add an explicit `M_b==0` test (§7). |
| **`TILE_N = N` LDS pressure** | Large N (512) with `B_TO_LDS` may exceed `SMEM_CAPACITY_MAP`. | Asserted in-compiler (line 212); fall back to `TILE_N=128` if it trips. |

---

## 7. Validation plan

Run **inside the docker container** (torch/triton/aiter not on bare host — global CLAUDE.md).

1. **Reference:** PyTorch eager loop
   `for b: out[s:e] = jagged[s:e] @ dense[b] + bias[b]` (note: un-transposed `dense (B,K,N)`),
   plus Meta's Triton kernel on identical inputs. BF16 in, FP32 accum.
2. **Metrics:** mean signed error + cosine similarity (not just `allclose`) to catch the
   systematic bias-precision drift flagged in §3.5.
3. **Shape coverage** (dev journal §8):
   - the 4 bench shapes (`B1024_D256_K256`, `B1024_D512_K512`, `B120_D256_K256`,
     `B120_D512_K512`, all `max_seq_len=16384`);
   - **empty groups** (`M_b=0`), **skewed** (one long + many short), `M_b` **not** a tile
     multiple, `max_seq_len ≫ mean`.
4. **`seq_offsets` dtype:** int32 and int64; verify large-`L` i64 row math.
5. **Correctness-before-exit:** first land the clamp path (§6), confirm bit-for-bit vs eager,
   then add early-exit and re-confirm (it must not change results, only speed).
6. **Build:** FlyDSL JIT (`AITER_REBUILD=1` after changes); confirm kernel name string
   (`KERNEL_NAME`) shows the jdbba variant in the JIT cache.

---

## 8. Milestones (suggested order)

1. **M0 — Fork & compile.** Copy `splitk_hgemm.py` → `jdbba_hgemm.py`, add signature/grid/group
   args, `SPLIT_K=1`, get it to **compile** (no behavior change yet: single group, `seq_start=0`,
   `M_b=L`). Sanity: reproduces a plain HGEMM.
2. **M1 — Group resolution (clamp path).** Add `seq_offsets` read, per-group rebasing (§3.2–3.3),
   `B`-as-`(B,N,K)`. **No early-exit yet.** Validate correctness vs eager on all shapes (§7).
3. **M2 — Early-exit.** Add the `start_m >= M_b` tail-tile skip; re-validate (results unchanged,
   short-group shapes faster).
4. **M3 — Tune.** Sweep §5 config space per shape; build the autotune cache keyed on
   `(bucket(max_seq_len), N, K)`. A/B against Triton (target: match, then beat).
5. **M4 — Profile & amortize.** `rocprofv3` MFMA-active vs total cycles (journal §6). If
   overhead-bound, pursue `TILE_N=N`, then **B-stationary** (port `small_m_hgemm`'s
   `PERSISTENT_N_TILES` idea to the **M** axis so one `Dense[b]` load feeds several M-tiles —
   journal §6 lever 2). Honest ceiling: recover the ~20–40% overhead fraction, not a peak
   multiplier.

Then hand off to the CKTile production path (dev journal §4); the §1–§3 recipe is identical, so
FlyDSL de-risks it.

---

## 9. Quick reference — the three changes, as a diff sketch

```
  compile_hgemm_kernel  ──────────────────►  compile_jdbba_kernel
  ─────────────────────────────────────────────────────────────────
  signature:  (C,A,B,BIAS, m, semaphore,signal)
           →  (C,A,B,BIAS, SEQ_OFFSETS, n_groups)          # m removed, seq added

  grid:       (bm*N_BLOCKS, SPLIT_K, 1)
           →  (bm*N_BLOCKS, 1, n_groups)                   # +group on z

  prologue:   block_m_idx,block_n_idx = pid//NB, pid%NB
           +  off_b = block_idx.z                          # NEW
           +  seq_start,seq_end = SEQ_[off_b],SEQ_[off_b+1]# NEW
           +  M_b = seq_end-seq_start                      # NEW (replaces runtime m)
           +  if start_m >= M_b: return                    # NEW early-exit (M2)

  A/Out:      m_offset = block_m_idx*BLOCK_M
           →  m_offset = seq_start + block_m_idx*BLOCK_M   # row rebase (abs rows)
              boundary: row < m  →  row < seq_end          # i64 row base

  B/bias:     B_=(n,k);  bias @ (n_offset+...)
           →  B_=(-1,k), row = off_b*N + n;  bias @ (off_b*N + ...)   # group rebase

  unchanged:  LDS/swizzle, STAGES pipeline, MFMA, truncf, masked store
```

---

## Worklog

- **2026-06-05** — Plan created. Grounded in real FlyDSL source (`splitk_hgemm.py`,
  `chunk_gated_delta_h.py`, `tensor_shim.py`) + the Triton reference. Status: design complete,
  no code yet. Next: M0 (fork & compile) inside the docker container.
