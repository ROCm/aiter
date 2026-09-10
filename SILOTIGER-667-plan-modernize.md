# SILOTIGER-667 — FlyDSL surface modernization

Bring the warp-decode MoE kernels onto the current FlyDSL public surface from
**flydsl-kernel-authoring** (and the **flydsl-kernel-code-cleanup** companion it
points at). This is a *style / API* track, not a new numeric path: keep packed
`v_dot2_f32_bf16` math, G7 drain, i64 expert-base addressing, and existing
dispatch.

**In scope:** `aiter/ops/flydsl/kernels/warp_decode_moe.py`,
`aiter/ops/flydsl/warp_decode_moe.py`. Tests only as a gate, not as a rewrite.

**Out of scope:** layout-algebra rewrite of packed FP8/FP4 weight dwords;
MFMA/`fx.gemm`; LDS/`SharedAllocator` (this kernel has no LDS); ticket benches
and CK harness.

Work the subtasks **in order**. Later items assume earlier ones have landed.
Each subtask is surgical and behavior-preserving: verify correctness (and a
spot-check of median kernel time) before starting the next.

## Locked decisions

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 1**
  (`HIP_VISIBLE_DEVICES=1`).
- **Surgical, behavior-preserving.** Same numerics (cos ≥ 0.999 vs existing
  cases in `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`). Do not
  change kVector defaults, `dot2_acc`, `kh_per_warp`, split-K policy, or scale
  folding.
- **Keep ISA-level helpers.** `v_dot2_f32_bf16` inline asm, `s_nop 2` / G7
  drain, `cvt_scalef32_pk_bf16_{fp8,fp4}`, and E8M0 `shl 23` stay; there is no
  `fx.gemm` atom for `v_dot2`.
- **Packed weight dwords stay on `buffer_ops` until subtask 5.** Offsets remain
  in **elements** of the load dtype (`i32` / `f32` / `i8`).
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale HSACO
  cannot mask a bad rewrite.
- **Gate.**
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py -v
  ```
  Spot-check one FP8 and one MXFP4 decode shape on the existing 667 bench after
  any change that touches the hot loop or wait/reduce path (scheduler-sensitive).

## Subtasks

### 1. Mechanical `fx.*` surface (thread ids, reduce, vectors)

Replace legacy spellings that the authoring skill already maps 1:1.

- `fx.thread_idx.x` / `fx.block_idx.x` → `gpu.thread_id("x")` / `gpu.block_id("x")`
  (import `gpu` from `flydsl.expr`).
- `wave_reduce_add_f32`: `gpu.ShuffleOp` + `arith.AddFOp` → typed
  `fx.Float32` + `.shuffle_xor(fx.Int32(sh), fx.Int32(64))` then `+`. XOR
  butterfly order may stay `(1, 2, 4, 8, 16, 32)` (equivalent to the skill’s
  high-to-low list).
- `vector.extract` in `load_i32_words` and `dot2_f32_bf16_scalar` →
  `fx.Vector(v)[i]`.
- Prefer `fx.Vector(...).bitcast(...)` / `fx.Int32(...)` over raw
  `arith.ExtFOp` / `ExtUIOp` / `ShLIOp` / `ConstantOp` and `llvm.bitcast` where
  a vector or typed numeric already exists. Leave `llvm.inline_asm` for
  `v_dot2`.

**Done when:** no `thread_idx`/`block_idx`/`vector.extract` in the kernel file;
reduce uses `shuffle_xor`; existing op_test still passes.

### 2. Reuse `tensor_shim._run_compiled`

Delete the local `_run` in `aiter/ops/flydsl/warp_decode_moe.py`. Import
`_run_compiled` next to the existing `ptr_arg` import from
`aiter.ops.flydsl.kernels.tensor_shim`. Call sites become
`_run_compiled(launcher, *args)` (or the shim’s documented `*args` form) —
do **not** add a second copy.

**Done when:** grep shows a single `_run_compiled` definition (in
`tensor_shim.py` only); all warp-decode launches go through it.

### 3. `atomic_add_f32` without hardcoded LLVM address space

Replace `llvm.IntToPtrOp` on `!llvm.ptr<1>` with `fx.to_llvm_ptr` /
`ptr.llvm_ptr` so the backend resolves the global address space. Keep
`AtomicRMWOp(fadd, syncscope="agent")` if there is still no typed wrapper;
localize that remaining dialect call.

**Done when:** no hardcoded `<1>` / `IntToPtrOp` in this file; split-K
`k_batch > 1` down path still matches the non-split path (same op_test cases).

### 4. Single definition path for `const_expr` if/else

The frontend restriction: do not define values inside `if/else` and use them
after the branch. Specialize or flatten:

- `build_gate_up_fp8_module`: `block2d` vs pertensor/pertoken both define
  `gate_acc` / `up_acc` then silu uses them.
- Nested `const_expr(use_i64_base)` that defines `*_rsrc` / `w_word_base`
  used in the K loop (gate_up FP8/FP8-act, down FP8).
- `const_expr(split_k)` is side-effect-only (atomic vs store) — either leave
  it or split into two epilogue helpers; no SSA live-out.

Prefer two builders or a local `@flyc.jit` dispatch over `scf.IfOp`. Runtime
`if lane == 0:` stores stay as-is.

**Done when:** no kernel uses a value first assigned only inside a
`const_expr` if/else arm; op_test covers both `block2d` and i64-base (large
`E*I*H`) paths.

### 5. Buffer views + layouts for *unpacked* tensors only

Only after 1–4. Move **unpacked** tensors onto `fx.rocdl.make_buffer_tensor` +
`fx.make_view` + `fx.copy` (or a documented `buffer_ops` exception with a
comment):

- BF16 activations / intermediate / outputs
- f32 scales (pertensor / pertoken / block2d)
- `router_ids` (i32), `router_wts` (f32)

**Keep** packed i32 weight/activation word loads (`load_i32_words`, FP8/FP4
dwords) and `_ptr_rsrc_off` (K3 i64 expert base) on `buffer_ops` until there
is an i64-base `make_buffer_tensor` equivalent. Do not invent a fake TV layout
for a mandatory packed-dword swizzle.

**Done when:** unpacked loads/stores go through buffer-resource views; packed
dword path is explicitly commented as the leftover legacy exception; op_test +
one FP8 and one MXFP4 bench spot-check are unchanged within noise.

## Non-goals (do not pull into this plan)

- Folding FP8 block2d scale into the convert (that is the separate TODO
  track).
- Changing GPU from the locked `HIP_VISIBLE_DEVICES=1`.
- Mass-comment cleanup as its own commit unless a subtask’s diff is unreadable
  without it.
