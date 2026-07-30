# V2 GEMM Turbo SPART Optimization Design

## Goal

Add an independently switchable SPART arithmetic and coordinate-propagation optimization to `feat/v2-gemm-turbo` without importing the previous branch's B-split implementation or overwriting this branch's newer GEMM2 structure.

The optimized path must preserve the current `GemmSpatiallyLocalTilePartitioner` output order while:

1. using unsigned arithmetic for contractually non-negative mapping values; and
2. passing the already-computed `(m_block_idx, n_block_idx)` directly into the GEMM2 body instead of flattening and re-dividing them.

## Current Branch Context

This branch differs materially from the earlier implementation branch:

- GEMM2 has `g2_kstages` with one-stage and two-stage B pipelines;
- tile support includes BN64/BN128/BN256 and BK128/BK256;
- fragment and scale creation has been refactored into helpers;
- `MXFP4_G2_BF16_LDS` defaults to disabled;
- the previous focused dispatcher unit-test file is absent.

The SPART mapping itself still uses signed floor division, and the non-persistent SPART path still computes `(m,n)`, flattens to `unit_bx`, then lets `gemm2_body_v2` recover `(m,n)` with another division.

## Scope

Modify only:

- `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py`
- `aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py`
- one new focused pytest file under `op_tests/flydsl_tests/`

Do not modify:

- `g2_kstages`, B loading, B fragment carry, or MFMA scheduling;
- BN64/BK128 behavior;
- naive or persistent tile mapping;
- grid dimensions or launch bounds;
- LDS allocation/swizzle;
- atomic/reduce epilogues;
- tuned CSV files;
- `3rdparty/composable_kernel` or any unrelated untracked file.

## Permanent Control and JIT Identity

Add a permanent environment/compile-time control:

```text
MXFP4_G2_SPART_OPT=0|1
```

Requirements:

- `compile_gemm2_a4w4_port` accepts an explicit `g2_spart_opt` argument; explicit argument wins over the environment;
- `get_g2` reads the environment value;
- the normalized Boolean enters the JIT cache key immediately after `g2_spart`;
- active kernels receive a `_spartopt` name tag after the existing `_spart<group>x<m01>` tag;
- `get_g2` forwards the normalized value to the compile function;
- the control remains permanently available regardless of its final default.

Normalize the active value as:

```python
bool(g2_spart_opt) and g2_spart > 0 and not persist
```

This prevents redundant variants for naive and persistent paths.

## Unsigned SPART Arithmetic

Add local helpers following the existing GEMM1 convention:

```python
def _udiv_i32(a, c):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(c))


def _umod_i32(a, c):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(c))
```

Extend `_spart_output_tile_index` with a compile-time `use_unsigned` choice.

On the optimized path:

- use unsigned division for `group_size`, `group_id_y`, `idx_M0`, `M0_tmp`, `idx_M00`, and `N_out`;
- use unsigned remainder only for the power-of-two-friendly `group_id_x`, `M0_mod`, and `idx_M01` cases;
- continue deriving `idx_N0` and `loc_mod` from the quotient to avoid a second general division.

On the switch-off path, preserve the current quotient/subtract expressions exactly.

All unsigned operands have non-negative contracts: block IDs, tile counts, group IDs, `M0/N0`, `GroupNum`, and `M01`. The existing 32-bit launch/index bounds remain authoritative.

## Direct Coordinate Propagation

Add a small resolver in `mxmoe_gemm_v2.py`:

```python
def _resolve_tile_coords(
    bx_i32,
    num_n_blocks,
    precomputed_m_block_idx=None,
    precomputed_n_block_idx=None,
):
    if precomputed_m_block_idx is not None:
        assert precomputed_n_block_idx is not None
        return precomputed_m_block_idx, precomputed_n_block_idx
    assert precomputed_n_block_idx is None
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    return m_block_idx, n_block_idx
```

Add the optional precomputed values to `gemm2_body_v2` after the runtime pad arguments and before keyword-only compile constants.

In the non-persistent SPART dispatcher path:

- compute `(m_block_idx, n_block_idx)` with `use_unsigned=g2_spart_opt`;
- when enabled, preload A with the resolved M coordinate and call the GEMM body with the two precomputed coordinates;
- do not construct `unit_bx` on the enabled path;
- when disabled, retain the existing flatten-to-`unit_bx` call.

The optimized call may pass a constant dummy flat ID; `_resolve_tile_coords` must not read or divide it when precomputed coordinates are supplied.

No `g2_spart_opt` parameter is needed inside `gemm2_body_v2`: the presence of precomputed coordinates expresses the body behavior, avoiding redundant specialization plumbing.

## Test-First Coverage

Create a focused pytest module that is CPU/import safe and follows the existing `op_tests/flydsl_tests` unit-test style.

Before production changes, add tests that fail because the feature is absent.

The tests must cover:

1. control plumbing and cache identity:
   - `SPART_OPT=0/1` produces distinct compile calls when SPART is active;
   - compile kwargs receive the normalized value;
   - `g2_kstages` and all newer branch parameters remain intact;
2. unsigned helpers:
   - both operands pass through `fx.Uint32` and the result through `fx.Int32`;
3. full mapping equivalence:
   - call the real `_spart_output_tile_index` in signed and unsigned modes;
   - compare the complete ordered output sequence;
   - assert range, uniqueness, count, and full Cartesian `(m,n)` coverage;
   - cover production-like, uneven, divisible, and non-power-of-two configurations;
4. coordinate resolution:
   - precomputed values bypass flat-ID division;
   - fallback returns the current mapping;
   - supplying only one precomputed coordinate raises an assertion.

## GPU and Compiler Validation

Compare two variants on the current branch with all other controls unchanged:

| `MXFP4_G2_SPART_OPT` | Path |
|---:|---|
| 0 | current signed + flatten/redivide path |
| 1 | unsigned + direct-coordinate path |

For each variant:

1. run a fixed-seed model-level correctness case and require:
   - return code 0;
   - no NaN;
   - no accuracy-threshold warning;
   - no excessive-logits warning;
2. generate a fresh cache-disabled FlyDSL dump and verify:
   - switch-off retains signed mapping and body coordinate redivision;
   - switch-on contains `divui/remui`, direct coordinates, and no flatten/redivide;
   - grid, MFMA count, LDS, spill counts, and output path are unchanged;
3. report final ISA instruction counts and VGPR/SGPR/LDS/private/spill metadata.

## Performance and Default Policy

Run five independent, interleaved stage2 measurements in `0 -> 1` order. Keep all raw samples and do not rerun outliers.

The environment switch is permanent. Select the final default as follows:

- if the enabled five-run median is no slower than the disabled median, default to `1`;
- otherwise default to `0`.

Report the result as an observed five-run median difference, not a statistically established speedup when sample ranges overlap.

## Acceptance Criteria

The work is complete when:

- focused tests pass and demonstrate a verified RED-to-GREEN cycle;
- Black, Ruff, `py_compile`, and `git diff --check` pass;
- signed and unsigned SPART mappings match exactly for all test configurations;
- both GPU variants pass the model-level numerical gate;
- IR/ISA proves direct coordinates and the targeted arithmetic reduction;
- no spill, private-memory, MFMA, grid, or LDS regression appears;
- five-run performance results and the selected default are reported;
- the composable-kernel gitlink and all unrelated files remain untouched.
