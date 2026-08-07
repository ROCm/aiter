# Bug: `fx.mma_atom_call` does not forward `modC=0` to ROCDL WMMA intrinsic

## Summary

When using `fx.make_mma_atom(fx.rocdl.WMMA(...))` + `fx.mma_atom_call()` to emit a
`v_wmma_f32_16x16x32_bf16` instruction on gfx1250, the generated ROCDL op does not
carry `modC=0`. This causes a **numerical precision regression** compared to calling
`rocdl_dialect.wmma_f32_16x16x32_bf16(..., modC=0, ...)` directly.

## Root Cause

The lowering path `fly.mma_atom_call` -> `rocdl.wmma_f32_16x16x32_bf16` does not
propagate `modC` (the i16 C-operand modifier attribute). The raw ROCDL op treats
`modC=None` (attribute absent) differently from `modC=0` (attribute present with
value 0), likely resulting in a different ISA encoding for the C operand modifier
field of `v_wmma_f32_16x16x32_bf16`.

### Call chain:
```
fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
  -> MmaOpGFX1250_WMMAType.get(16, 16, 32, bf16, bf16, f32, sign_a=False, sign_b=False, clamp=False)
     *** no modC / reuseA / reuseB parameters ***

fx.mma_atom_call(atom, c, a, b, c)
  -> fly.mma_atom_call MLIR op
  -> (lowering) rocdl.wmma_f32_16x16x32_bf16(res, a, b, c)
     *** modC attribute not set (None, not 0) ***
     *** reuseA, reuseB attributes not set ***
```

### Working direct call (no precision issue):
```python
rocdl_dialect.wmma_f32_16x16x32_bf16(
    T.vec(8, T.f32), src_a, src_b, acc,
    signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
)
```

## Impact

FMHA (Flash Attention) kernel on gfx1250 shows precision regression when
`Atom.wmma_init` / `Atom.wmma_accum` are switched from the direct ROCDL intrinsic
to `fx.mma_atom_call`. The regression is visible as `checkAllclose` failures
with `atol=0.01, rtol=0.01` on output tensors.

## Suggested Fix

### Option A: Forward `modC` through `MmaOpGFX1250_WMMAType`

Add `mod_c`, `reuse_a`, `reuse_b` parameters to `WMMA()` (non-Scale path):

```python
# In flydsl/expr/rocdl/universal.py, WMMA function:
def WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None, *, mod_c=0, reuse_a=False, reuse_b=False, **kwargs):
    ...
    return MmaOpGFX1250_WMMAType.get(
        m, n, k, ty_ab, ty_ab, ty_acc,
        sign_a=..., sign_b=..., clamp=...,
        mod_c=mod_c, reuse_a=reuse_a, reuse_b=reuse_b,  # <-- add these
    )
```

### Option B: Set defaults in lowering

In the `fly.mma_atom_call` -> `rocdl.wmma_*` lowering pass, always emit
`modC=0, reuseA=False, reuseB=False` when the attributes are absent:

```cpp
// In the lowering pass for fly.mma_atom_call:
auto modC = op->getAttrOfType<IntegerAttr>("modC");
if (!modC) modC = rewriter.getI16IntegerAttr(0);  // default to 0
```

## Files involved

- `flydsl/expr/rocdl/universal.py` — `WMMA()` function (L83-126)
- `flydsl/_mlir/_mlir_libs/_mlirDialectsFlyROCDL.pyi` — `MmaOpGFX1250_WMMAType.get()` stub
- The C++ lowering pass for `fly.mma_atom_call` -> `rocdl.wmma_*`
