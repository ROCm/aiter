## Problem

Commit #4029 (`769790445`) refactored the DSv4 rotate-quant path from inplace kernels to out-of-place kernels. During that refactor, all `dim=1024` launch sites were switched to `vec_size=16`.

That tile size is only valid on **wave64** targets where `16 × 64 = 1024`. On **wave32** targets such as gfx1250, the constraint `vec_size × warp_size % dim == 0` fails because `16 × 32 = 512`, which causes:

- JIT compilation failure for `module_dsv4_rotate_quant` (static assert / host-side `AITER_CHECK`)
- Incorrect Hadamard tile geometry when the module does build

This regressed the wave32 fix originally landed in #3528 (`cd744a802`).

Wave64 platforms (gfx950 / MI355X, gfx942, etc.) are unaffected by the bug itself because `16 × 64 = 1024` already satisfies the constraint.

## Solution

Dispatch `dim=1024` kernel tile sizes at runtime based on `WARP_SIZE`:

| Wave size | vec_size | logical_warp | Product |
|-----------|----------|--------------|---------|
| 32 (gfx1250) | 32 | 32 | 1024 |
| 64 (CDNA) | 16 | 64 | 1024 |

Changes:

1. Add `logical_warp_size` template parameter to `hadamard_rotate_activation_fp4quant_kernel` so `m_block` is computed from the selected tile geometry instead of always using `get_warp_size()` at compile time with a mismatched `vec_size`.
2. Introduce `ROTATE_ACTIVATION_FP4QUANT_KERNEL_IMPL_WARP` for explicit `(vec_size, logical_warp_size)` instantiation.
3. Apply the same wave32/wave64 dispatch to all `dim=1024` entry points:
   - `rotate_activation_fp4quant` / `rotate_activation` (`hadamard_*`)
   - `rope_rotate_activation_fp4quant` / `rope_rotate_activation` (`rope_hadamard_*`)
   - `rope_rotate_activation_fp8quant`
   - `rmsnorm_rope_rotate_activation_fp4quant_kvcache` (`norm_rope_hadamard_*`)

Each kernel template takes an explicit `logical_warp_size` so both `(vec=32, warp=32)` and `(vec=16, warp=64)` instantiations compile on gfx1250 JIT builds without affecting the launched configuration at runtime.

## Test plan

- [ ] `python op_tests/test_dsv4_rotate_quant.py --dim 1024` on gfx1250
- [ ] `python op_tests/test_dsv4_rotate_quant.py --dim 1024 --rope` on gfx1250
- [ ] `python op_tests/test_dsv4_rotate_quant.py --dim 1024 --fp8` on gfx1250
- [ ] `python op_tests/test_dsv4_rotate_quant.py --dim 1024 --norm-cache` on gfx1250
- [ ] Regression on gfx950 / wave64: `--dim 128,512,1024` (behavior unchanged)
