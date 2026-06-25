# Plan: full layout-API conversion of v2 (copy-atom NT config + scaled MMA)

Goal: convert the TWO remaining raw pieces of mxfp4_gemm1_v2.py onto the layout API,
by extending FlyDSL where needed. Branch: moe_layout_api (worktree moe2stage-flyc).

## Build loop (de-risked this session)
- FlyDSL = C++/MLIR; LLVM/MLIR PREBUILT at /root/llvm-project/mlir_install.
- Incremental rebuild ~30s: `cd /root/FlyDSL/build-fly && ninja`. Output ->
  build-fly/python_packages/flydsl. Needs nanobind (installed) + cmake arg
  -Dnanobind_DIR=$(python -c "import nanobind;print(nanobind.cmake_dir())").
- Use rebuilt lib via PYTHONPATH=/root/FlyDSL/build-fly/python_packages before
  /opt/venv/bin/python. Confirmed: current FlyDSL source imports + v2 numeric passes.
- Final install (once both land): copy build-fly/python_packages/flydsl/_mlir/_mlir_libs/*.so
  over /opt/venv/.../flydsl (and any changed .py) so default runs pick it up.

## Task 1 - copy-atom cache/NT config (NT B-load via layout API)
C++ (FlyDSL):
1. include/.../FlyROCDL/IR/CopyAtom.td: add `int32_t cacheModifier` (default 0) to
   FlyROCDL_CopyOpBufferCopy (+ keep a 1-arg builder so existing `.get(bitSize)`
   callers still work). Update assemblyFormat.
2. lib/.../FlyROCDL/CDNA3/CopyAtom.cpp emitAtomCallSSA: replace the hardcoded
   `zero` aux operand (lines ~104 load, ~121 store) with a const from getCacheModifier().
3. python/flydsl/expr/rocdl/universal.py: BufferCopy128b/64b/32b/16b/8b accept
   optional cache_modifier=0 -> CopyOpCDNA3BufferCopyType.get(bits, cache_modifier).
   Update the .pyi.
aiter (v2 kernel):
4. issue_b_load_j NT path: drop raw buffer_load; build a second copy atom
   `_b_copy_atom_nt = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=2), 32)`
   and use the SAME layout-API preshuffle view + fx.copy as the cached path, only
   with the nt atom. Removes the raw fallback entirely; both paths become layout-API.
Gate: byte-exact vs v1 for use_nt=True at M=2/256/8192; ISA shows `nt` flag on B loads.

## Task 2 - scaled MMA via layout API (replace raw mfma_scale)
Infra already EXISTS: MmaOpCDNA4_MFMAScaleType has scale_a/scale_b atom state
(set via set_value) + opsel_a/opsel_b params + working emitAtomCallSSA; frontend
exposes fx.make_mma_atom + fx.mma_atom_call + fx.rocdl.cdna4.MFMA_Scale.
Approach A (minimal, try first): in mfma_cluster, replace each
  rocdl.mfma_scale_f32_16x16x128_f8f6f4(ty,[a,b,c,4,4,opselA,sa,opselB,sb])
with
  atom = MFMA_Scale(16,16,128,Float4E2M1FN, opsel_a=opselA, opsel_b=opselB)
  acc  = fx.mma_atom_call(atom.set_value({"scale_a":sa,"scale_b":sb}), d,a,b,c)
operating on the SAME fragments already in hand (A vec4 from ds_read, B from the
layout-API frags, C accumulator). Pre-build the (opselA,opselB) atom variants
(<=16) once. Verify mma_atom_call accepts these fragment/value types; adapt
fragment wrapping if it wants memref-backed frags.
Approach B (fallback, only if A can't thread per-call scale/opsel): extend the
fly.gemm / mma lowering in C++ to accept per-k scale; heavier.
Gate: cos == v1 (mean_row_cos>0.85, e2e>0.99) AND byte-exact vs v1 across
M=64/256/8192, both use_nt; perf parity (±2%).

## Order & validation
1. Task 1 C++ + rebuild + rewire + validate (smaller, proves the build/rewire loop).
2. Task 2 Approach A + validate. Fall back to B only if needed.
3. Full suite both backends; perf sweep at tuned BM32 points; update memory.
4. Decide whether to vendor the FlyDSL .so change or gate behind the rebuilt lib.

## Risks
- FlyDSL source HEAD (30bf36a7) != installed 0.2.2; building from source is the
  way (already validated compatible). Shipping requires the rebuilt .so.
- mma_atom_call fragment-type expectations (SSA vec vs memref frag) - may need a
  small wrapping shim; resolved during Task 2.
- per-(opselA,opselB) atom proliferation - bounded (<=16), built once.
