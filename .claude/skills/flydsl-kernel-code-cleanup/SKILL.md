---
name: flydsl-kernel-code-cleanup
description: >
  Modernize FlyDSL kernels: replace raw MLIR dialects (arith, scf, vector, llvm,
  memref, math), ArithValue, redundant fx.* wrapping, fx.Index, buffer_ops,
  SmemPtr/SmemAllocator, per-tile *_atom_call, and raw rocdl.mfma_* with the
  current fx.* surface (fx types, Python control flow, make_buffer_tensor,
  SharedAllocator, fx.copy/fx.gemm, local @flyc.jit if/else). Also trims comments
  and dead code and applies the _run_compiled fast launch path. Use when
  reviewing, cleaning, or migrating existing kernels.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# FlyDSL Kernel Code Cleanup

Maps legacy/deprecated kernel constructs to the current `fx.*` surface. Companion
to `flydsl-kernel-authoring` (API reference) and `flydsl-tile-programming`
(authoring wizard).

**Golden rule:** in `@flyc.kernel` / `@flyc.jit` bodies, use `fx.*` and Python
operators first. Drop to a raw dialect only at a hard boundary with no wrapper,
and localize it.

## Cautions

- **Surgical, behavior-preserving.** Migration is a refactor: minimal diffs, match
  local style.
- **Don't mass-rewrite heavily-legacy kernels** unless asked (e.g.
  `flash_attn_gfx950.py` uses `_scf.IfOp`/`_raw` pervasively). Clean what the task
  touches.
- **Verify.** Offset/type/SSA changes can shift results — run the kernel test
  before/after; clear cache with `FLYDSL_RUNTIME_ENABLE_CACHE=0` if unsure.
- **`expr/` stays target-neutral:** no `rocdl`/`llvm`/buffer imports in
  `python/flydsl/expr/` top-level (guarded by `test_expr_optional_rocdl.py`).

---

## 1. `ArithValue` and index helpers (deprecated in `expr/arith.py`)

| Deprecated | Replacement |
|---|---|
| `ArithValue(x)` (wrap for operators) | `fx.Int32/Int64/Float32/Vector` — already overload `+ - * / % << >> == < >` |
| `arith.unwrap(v)` / `arith._to_raw(v)` | `v.ir_value()`, only where a raw `ir.Value` is needed |
| `arith.index(n)` / `arith.index_cast(T.index, v)` / `fx.Index(n)` | `fx.Int64(...)` (or `fx.Int32(...)`) |

`fx.Index` maps to MLIR `index` — platform-defined width, ambiguous, and forces
implicit casts. Prefer explicit-width `fx.Int64`/`fx.Int32`; pick the width on
purpose (don't widen counters that must stay `i32`).

```python
# Before
acc  = ArithValue(val) + peer
lane = ArithValue(tid) % fx.Index(64)
cond = arith.unwrap(idx >= limit)
off  = arith.index_cast(T.index, x)
# After
acc  = val + peer                    # val already fx.Float32 / fx.Vector
lane = tid % fx.Int64(64)
cond = (idx >= limit).ir_value()     # only if a raw scf.IfOp needs it
off  = fx.Int64(x)
```

If an operand is a raw `ir.Value`, wrap it once at the source (`fx.Float32(v)`),
not with `ArithValue` per use. Keep an explicit `arith.*FOp` only for non-default
fastmath.

### 1b. Drop redundant `fx.*` wraps

Wrap only to *introduce* a type (Python literal / raw `ir.Value`) or *change* one.
Re-wrapping an already-typed value is noise; double-wrapping is dead.

```python
# Before
for i in range_constexpr(fx.Int32(N)):
    off = fx.Int64(fx.Int64(base) + fx.Int64(4))
tile = fx.make_layout(fx.Int32(BLOCK), fx.Int32(1))
idx  = fx.Int32(tx)                  # tx already fx.Int32
# After
for i in range_constexpr(N):
    off = base + fx.Int64(4)
tile = fx.make_layout(BLOCK, 1)      # builders take Python ints
idx  = tx
```

- Compile-time shapes/strides/bounds (`make_layout`, `make_shape`,
  `range_constexpr`, `Constexpr`) take plain Python ints.
- Wrap a runtime value once, at first typed use.
- A real cast (`fx.Int64(i32)` widen, `fx.Int32(index)` narrow) is not redundant —
  it replaces `arith.index_cast`.

---

## 2. `buffer_ops` → `make_buffer_tensor` + copy atoms

`create_buffer_resource` + manual offsets is legacy. Build a buffer-resource view
with `fx.rocdl.make_buffer_tensor()`, then use layout ops + `fx.copy_atom_call`;
the OOB-checked V# descriptor is built for you.

```python
# Before (manual offsets — see PA //4 offset bugs)
rsrc = buffer_ops.create_buffer_resource(A, max_size=True)
data = buffer_ops.buffer_load(rsrc, row * K + k, vec_width=4, dtype=fx.Float32)
buffer_ops.buffer_store(data, rsrc, row * N + col)
# After
bufA = fx.rocdl.make_buffer_tensor(A)
tA   = fx.make_view(fx.get_iter(bufA), fx.make_layout((M, K), (K, 1)))
copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
fx.copy_atom_call(copy, fx.slice(tA, (None, tid)), rA)   # after partitioning tA
```

- `make_buffer_tensor(tensor, max_size=True)` mirrors `create_buffer_resource`;
  pass `num_records_bytes=` for a const byte count, or `max_size=False` to derive
  from the layout.
- gfx1250 TDM uses a different atom — `fx.rocdl.make_tdm_atom` (raw VA, not a
  buffer resource).
- A scalar-base + per-thread-offset load with no layout form may stay on
  `buffer_ops` — note it. `buffer_load/store` `offset` is in **elements** (×
  `sizeof(dtype)` internally) — a classic bug.

---

## 3. Raw upstream dialects → `fx.*` and Python

### `arith`
| Raw | Preferred |
|---|---|
| `arith.constant(42, index=True)` | `fx.Int64(42)` |
| `arith.mulf/addf(a,b)` | `a * b` / `a + b` |
| `arith.trunc_f(ty, v)` / `ext_f` | `v.to(fx.BFloat16)` |
| `arith.index_cast(T.i32, v)` | `fx.Int32(v)` |
| `arith.select(cond, t, f)` | `cond.select(t, f)` |
| `arith.cmpi(slt, a, b)` | `a < b` |
| `arith.maxnumf(a,b)` | `a.maximumf(b)` |

Keep `arith.cmpf` / explicit `*FOp` only where no operator exists or fastmath is
needed.

### `scf`
| Raw | Preferred |
|---|---|
| `scf.ForOp` | `range_constexpr(N)` (unrolled) or `range(lo, hi, step, init=[...])` (runtime, loop-carried) |
| `scf.IfOp(_raw(cond))` | Python `if cond:` (runtime) / `if const_expr(flag):` (compile-time) |

Runtime bounds must be typed (`fx.Int64`) or the rewriter unrolls and drops
`init=`. See §5 for branches the rewriter can't express.

### `vector`
| Raw | Preferred |
|---|---|
| `vector.extract(v, static_position=[i])` | `fx.Vector(v)[i]` |
| `vector.bitcast(ty, v)` | `fx.Vector(v).bitcast(fx.Float32)` |
| `vector.splat` / const vector | `fx.Vector.filled(width, val, fx.Float32)` |
| build from scalars | `fx.Vector.from_elements(...)` |
| reg-memref load/store | `fx.memref_load_vec(r)` / `fx.memref_store_vec(v, r)` |

### `llvm` / `memref` / `math`
- `llvm.*` ptr math / load/store / const → layout views (`fx.make_view`,
  `fx.get_iter`), `fx.Array` + `SharedAllocator`, `fx` constants. Real intrinsics
  go in `expr/rocdl/inline_asm.py` or `rocdl` wrappers.
- `memref.*` → layout tensors/views + copy atoms.
- `math.*` → `fx` math helpers (`expr/math.py`); keep `math_dialect.fma` etc. only
  where no wrapper exists.

---

## 4. `SmemAllocator` / `SmemPtr` → `SharedAllocator`

Legacy LDS path uses a manual base pointer, byte offsets, and `finalize()`. New
kernels declare an `@fx.struct` of `fx.Array` fields and allocate via
`fx.SharedAllocator` — the compiler sizes the LDS global; **no finalize**.

```python
# Before
allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
base = allocator.get_base()
smem_a = SmemPtr(base, 0, dtype_, shape=(BLOCK_M * BLOCK_K,))
smem_b = SmemPtr(base, a_bytes, dtype_, shape=(BLOCK_K * BLOCK_N,))
allocator.finalize()
# After
@fx.struct
class SharedStorage:
    a: fx.Array[fx.Float16, BLOCK_M * BLOCK_K]
    b: fx.Array[fx.Float16, BLOCK_K * BLOCK_N]

lds   = fx.SharedAllocator().allocate(SharedStorage).peek()
lds_a = lds.a.view(fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
lds_b = lds.b.view(fx.make_layout((BLOCK_K, BLOCK_N), (BLOCK_N, 1)))
```

- Default `static=True` leaves `launch(smem=...)` unset; only `static=False`
  auto-infers `smem` from `allocated_bytes`.
- `SmemPtr.get()` caches its view — reusing it in an epilogue after a `scf.for`
  causes a dominance error. `SharedAllocator` avoids this (view taken per use); for
  legacy code, clear `ptr._view_cache = None`.
- Structural change — migrate a kernel's whole LDS at once and re-run its test.

---

## 5. Runtime `if/else` with side effects → local `@flyc.jit`

A plain `if runtime_cond:` works for simple guarded stores but breaks when a
branch defines values used later, carries loop state, or has `return`/`yield`. The
legacy fix is `scf.IfOp(_raw(cond))`; the current idiom is branch helpers wrapped
in a local `@flyc.jit`.

```python
# Before
with _if_then(_scf.IfOp(_raw(ArithValue(q_start < seqlen_q)))):
    ...
# After
def then_path(): ...
def else_path(): ...

@flyc.jit
def dispatch():
    if q_start < seqlen_q:      # typed fx compare → scf.if
        then_path()
    else:
        else_path()

dispatch()
```

- A bare `if cond:` is fine for a simple guarded side effect — no helper needed.
- `const_expr(flag)` for compile-time branches; never wrap runtime SSA
  (`gpu.thread_id`, `lane`) in `const_expr`.
- Keep manual `scf.IfOp` only where the rewriter can't express it; localize it.

---

## 6. Raw `rocdl.mfma_*` → MMA atom + `fx.gemm`

Raw intrinsics hardcode fragment types, the `[a, b, c, 0, 0, 0]` tuple, and the
instruction. Build an atom and issue it; fragment layouts/packing are handled and
the atom is arch-dispatched (MFMA on CDNA3/4, WMMA on gfx11/gfx1250).

```python
# Before
c_frag = rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0])
# After
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))   # → f32 acc
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)                    # d, a, b, c
fx.mma_atom_call(mma, frag_C, frag_A, frag_B, frag_C)           # single tile
```

- `fx.rocdl.MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None)` picks the intrinsic from
  shape+dtype. Scaled: `fx.rocdl.cdna4.MFMA_Scale`; gfx1250/gfx11:
  `fx.rocdl.WMMA` / `WMMAScale`.
- Build fragments with `fx.make_fragment_like` / `make_fragment_{A,B,C}`, not raw
  `T.vec(...)`.
- Order is **d, a, b, c** (accumulator first).
- Structural — convert the whole MMA loop and diff numerics. Keep a raw call only
  for an instruction the builders don't expose.

---

## 7. Per-tile `*_atom_call` → `fx.copy` / `fx.gemm`

`copy_atom_call` / `mma_atom_call` issue a single atom instance. `fx.copy` /
`fx.gemm` iterate the atom over a tiled/partitioned layout and take atom state as
kwargs — no hand-written loop or `atom_set_value`.

```python
# Before
for k in range_constexpr(K_TILES):
    fx.copy_atom_call(copy_atom, part_src[k], frag[k])
for k in range_constexpr(K_TILES):
    fx.mma_atom_call(mma, frag_C, frag_A[k], frag_B[k], frag_C)
# After
fx.copy(copy_atom, part_src, frag)
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C, scale_a=sa, scale_b=sb)   # atom state as kwargs
```

- `fx.copy` for partitioned tensors (`partition_S`/`partition_D` / tiled divide);
  `fx.gemm` for the MMA loop (accumulator-first order).
- Keep `*_atom_call` for a genuine single atom instance — that's the lower-level
  primitive, not legacy.

---

## 8. Trim comments and dead code

Cut low-value comments and dead code; a net LOC drop with unchanged behavior is
the signal a cleanup landed (the migrations above already collapse verbose code).

**Remove:** comments that restate code; commented-out / dead blocks; per-line step
narration; ASCII banners (keep one concise header per section); stale comments that
contradict the code; unused locals/imports/helpers you made redundant; runs of 2+
blank lines.

**Keep:** the *why* — non-obvious layout/stride math, swizzle rationale, ABI
quirks, offset-unit gotchas, invariants, spec/ISA references.

- Comment cleanup must not touch code — do it in a separate commit.
- When unsure about a *why* comment, keep it. Leave pre-existing dead code (mention
  it); only remove dead code you introduced.

---

## 9. Cut launch overhead with `_run_compiled`

Calling a `@flyc.jit` wrapper directly re-runs per-call dispatch (DLPack, arg
marshalling, cache lookup). On hot paths use `_run_compiled`
(`kernels/common/tensor_shim.py`): compile once, cache the `CompiledFunction`,
fast-dispatch after.

```python
from kernels.common.tensor_shim import _run_compiled

compiled = compile_my_kernel(...)          # {"launch": <exe>, ...}
_run_compiled(compiled["launch"], out.data_ptr(), a.data_ptr(), b.data_ptr(),
              a.stride(0), M, N, K, stream)

def _run_compiled(exe, *args):             # in-tree
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args); exe._cf = cf
    else:
        cf(*args)
```

- Pass flat scalars/pointers (`data_ptr()`, `stride(i)`, sizes, `stream`) — it
  bypasses DLPack. See `pa_decode_fp8.py`, `hgemm_splitk.py`.
- Reuse the shim; don't add a second copy.
- Worth it for small kernels in tight loops, not cold one-shot launches. Arg
  order/types must match the compiled signature — verify.

---

## 10. Procedure

1. **Find** legacy usage:
   ```bash
   grep -nE "ArithValue|_to_raw|arith\.(unwrap|index|index_cast)|fx\.Index\(" <file>
   grep -nE "buffer_ops\.(create_buffer_resource|buffer_load|buffer_store)" <file>
   grep -nE "_mlir\.dialects import.*(arith|scf|vector|llvm|memref|math)" <file>
   grep -nE "\b(scf\.(For|If)Op|vector\.(extract|bitcast|splat)|llvm\.(load|store|mlir))" <file>
   grep -nE "SmemPtr|SmemAllocator|\.finalize\(\)" <file>
   grep -nE "fx\.(Int32|Int64|Float32)\(fx\.(Int32|Int64|Float32)\(" <file>
   grep -nE "rocdl\.mfma_|\bmfma_(f32|i32)_|copy_atom_call|mma_atom_call" <file>
   ```
2. **Triage:** do mechanical swaps (operators, casts, `vector.extract/bitcast`)
   first; structural ones (control flow, `buffer_ops` offsets, MMA loops) next.
3. **Migrate in small commits**, one family at a time, matching local style.
4. **Verify:**
   ```bash
   bash scripts/check_python_style.sh --fix
   FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest tests/kernels/test_<kernel>.py -v
   ```
   Diff numerics for offset-sensitive buffer changes.
5. **Check `git diff --stat`** shows a net line reduction; growth is a red flag.
6. **Report** anything left legacy on purpose (pervasive `_scf.IfOp`, a
   scalar-base load with no layout form).

---

## Quick reference

| Legacy | Current |
|---|---|
| `ArithValue(x) + y` | `x + y` (typed `fx`) |
| `arith.unwrap(v)` / `_to_raw(v)` | `v.ir_value()` (boundary only) |
| `fx.Index(n)` / `arith.index` / `arith.index_cast` | explicit `fx.Int64/Int32(...)` |
| `arith.mulf/addf/trunc_f/select` | `*`, `+`, `.to(ty)`, `.select(...)` |
| `vector.extract/bitcast/splat` | `fx.Vector(v)[i]` / `.bitcast(ty)` / `.filled(...)` |
| `scf.ForOp` / `scf.IfOp` | `range_constexpr` / `range(..., init=)` / Python `if` / `const_expr` |
| `buffer_ops.*` + offsets | `fx.rocdl.make_buffer_tensor` + layout + `fx.copy_atom_call` |
| raw `llvm`/`memref` access | `fx.make_view` / `fx.get_iter` / `SharedAllocator` |
| `SmemAllocator`/`SmemPtr` + `finalize()` | `@fx.struct` + `fx.SharedAllocator().allocate(...).peek().view(...)` |
| `scf.IfOp(_raw(cond))` branch w/ outputs | branch helpers + local `@flyc.jit` |
| `fx.Int32(fx.Int32(x))` / wrapping const ints | plain Python int; wrap once |
| `rocdl.mfma_*` raw intrinsic | `fx.make_mma_atom(fx.rocdl.MFMA(...))` + `fx.gemm` |
| per-tile `*_atom_call` loop | `fx.copy` / `fx.gemm` (state as kwargs) |
| restated/dead/stale comments, blank runs | delete; keep *why*; aim for net LOC drop |
| per-call `@flyc.jit` on a hot path | `_run_compiled(exe, *args)` fast dispatch |
