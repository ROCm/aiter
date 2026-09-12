# Semantic preservation for FlyDSL cleanup

Read this when extracting shared helpers or replacing low-level operations in
existing kernels. These are checks for equivalence, not a request to expand the
task to other backends, architectures or dependencies.

## Establish the implementation being tested

- Record the base and final commits, pinned FlyDSL version, imported module
  paths, GPU target and relevant dispatch flags. Keep the baseline immutable in
  a separate checkout. Avoid putting a newer sibling FlyDSL checkout on
  `PYTHONPATH` when validating aiter's pinned release.
- If the baseline fails to compile, retain that result. A temporary compatibility
  patch can isolate the refactor, but label it separately from the real baseline.
  An upstream fix should be compared against the actual updated base after merge.
- Follow the public entry point to the selected kernel and specialization.
  Benchmark labels can be historical: a label containing `ck` does not prove
  that the dispatcher avoided FlyDSL. Direct helper tests alone do not cover
  quantization, weight shuffling, routing and launch arguments together.

## Share helpers without changing arithmetic

| Family | What must remain equivalent |
|---|---|
| Sigmoid/tanh/SiLU/Swiglu/SiTU | Constant rounding, multiplication order, clamp placement, NaN and signed-zero behavior |
| Batch activation | Grouping of exp2 and reciprocal operations, accumulator live ranges and instruction scheduling |
| Integer min/max | Signed or unsigned interpretation, including packed `u16` values with the high bit set |
| Float min/max | `maximumf`/`minimumf` NaN propagation versus `maxnumf`/`minnumf`; do not substitute solely by name |
| Ceildiv | Host integer behavior versus fixed-width wrapping arithmetic, signedness and supported denominator domain |
| Bit packing | Logical versus arithmetic shifts, zero versus sign extension, source and destination widths |

Use `act.py` for shared activation components and `LOG2E`. A sign-restored tanh
and `2 * sigmoid(2 * x) - 1` can round differently; retain distinct variants
while sharing their common operations. Folding an activation coefficient into
`LOG2E` may also change where rounding happens. Avoid interleaving reciprocal
operations into a previously staged batch without checking code generation.

Prefer `fx.ceildiv` for its matching typed division operation. The shared
`kernels_common.ceildiv` expression `(n + d - 1) // d` instead preserves the
caller's fixed-width additions when used with DSL integers. Those formulas can
differ on overflow and produce different instructions. Keep an existing wrapping
contract or prove its bounds before changing it; document the supported domain.

For packed data, signed-to-wide conversion can fill the upper word with ones.
Reinterpret or extend as unsigned when the value represents bits, and verify
every affected byte position. Inspect the original shift operation: legacy
`ArithValue` on a signless integer can use logical right shift, whereas a signed
`fx.Int32` uses arithmetic right shift. Choose `fx.Uint32` when preserving that
logical operation. Treat a discovered correctness fix as a behavior change with
its own oracle, not as an unchanged-ISA refactor.

## Preserve memory contracts

- Track byte and element offsets through pointer extraction, GEPs, views and
  buffer descriptors. `buffer_load/store` element offsets are scaled by dtype;
  packed formats and sub-dword loads need explicit unit checks.
- Preserve alignment and its provenance. Replacing a byte GEP with a typed
  element GEP can change alignment inference and vectorization even when the
  numeric addresses match. In a byte-addressed epilog, preserve that arithmetic
  (for example through an `Int8` pointer) before reinterpreting the access type.
- Typed pointer access is suitable only if it preserves address space, cache
  policy, volatile behavior, alias metadata and atomic scope/ordering. Keep a
  local LLVM boundary when the pinned API cannot express those properties.
- Legacy TDM memref proxies and arbitrary-width integer offsets may require raw
  interfaces. Confirm actual callers before removing them; moving their imports
  into a generic facade does not remove the dependency.
- A shared helper may serve excluded multi-GPU callers. Inspect those contracts
  without broadening the requested migration into their implementation.

## Evidence for no regression

Choose checks for the operations and dispatch branches changed; do not rerun an
unrelated matrix merely to increase the count.

1. **Numerics and memory:** run representative existing public-entry tests on
   both trees with the same inputs. Include applicable tails, alternate epilogs,
   shuffling/routing and output canaries. For arithmetic/packing changes, add a
   bounded probe with special values or exact integer references where ordinary
   floating tolerance would hide a defect.
2. **ISA and resources:** disable the runtime cache with
   `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Capture each specialization in a fresh dump
   directory; dumps keyed only by kernel name overwrite earlier shapes. Compare
   every expected final ISA artifact and target, not just the last file or the
   common subset. Missing artifacts are inconclusive. If ISA changes, inspect
   VGPR/SGPR, spills, scratch, static and dispatch-time LDS as applicable.
3. **Performance:** for changed hot paths, alternate before/after measurements on
   the same idle GPU using the same warm graph and launch parameters. Report
   medians and variation. Identical resource counts do not prove equal latency,
   and small timing changes with identical code do not establish a speedup.
4. **Test integrity:** some scripts print failed comparisons but return zero.
   Inspect the actual assertions and summary, and separate baseline failures
   from introduced ones. Do not relax tolerances, disable checks or remove useful
   regression cases to turn a result green.
5. **Delivery:** review the final merge-base diff and helper callers. After new
   fixes or an upstream merge, validate affected paths at the resulting head.
   Attribute CI failures from logs and baseline evidence; do not infer the
   backend from a test name. State architecture guards and compile-only coverage
   explicitly rather than claiming all operators passed.

The cleanup inventory should identify remaining raw interfaces and duplicate
candidates by file, semantic reason and applicable validation. It is evidence
for a particular tree and scope, not a permanent allowlist that prevents future
migration when an equivalent public API becomes available.
