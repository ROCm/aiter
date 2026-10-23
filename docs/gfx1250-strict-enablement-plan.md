# gfx1250-strict enablement plan

Tracking: AITERKER-209, ROCM-31030

Target date: November 7, 2026

## Goal

Treat `gfx1250-strict` (MI450 A0) as a compiler and code-object identity that is
separate from `gfx1250` (MI450 B0). AITER must never silently truncate the
strict name, share an incompatible JIT artifact, or load a B0-only prebuilt
kernel on A0.

The strict compiler defines `__gfx1250_strict__` and does **not** define
`__gfx1250__` (ROCm/rocm-systems#12068; confirmed by ROCm 10.1 HIP tests). A
source-level `#if defined(__gfx1250__)` device guard therefore compiles **away**
on a strict agent, leaving an empty kernel — the silent failure mode behind the
upstream `TDM_Basic_load_2d` case (empty kernel vs. a zero-filled output, no
runtime error). Every `__gfx1250__` device guard that must also cover A0 has to
become `defined(__gfx1250__) || defined(__gfx1250_strict__)`.

This change must stay backward-compatible with the shipping ROCm (7.x), whose
compiler knows only `__gfx1250__` and has no strict target at all: the
`|| defined(__gfx1250_strict__)` arm is simply never taken there, so the old
build is unaffected. On top of the guards, runtime dispatch, caches, tuned
configs, and prebuilt binaries still need an explicit compatibility decision.

## Workstreams

### 1. Architecture identity and safety — AITER Infra

- Preserve the complete `gfx1250-strict` name from `rocminfo`, `hipinfo`, and
  Torch device properties.
- Accept the target in offline and JIT builds and keep it distinct in cache and
  code-object coverage checks.
- Give strict its own tuned-config key instead of implicitly consuming B0
  entries.
- Reject shipped `hsa/gfx1250` ASM on strict/A0. Keep the legacy ASIC-revision
  guard while older runtime stacks can still report A0 as plain `gfx1250`.
- Add CPU-only unit coverage for parsing, build selection, and incompatible
  code-object rejection.

Exit criteria: `GPU_ARCHS=gfx1250-strict` reaches hipcc unchanged; strict and
regular gfx1250 artifacts are never accepted as substitutes for each other.

### 2. Kernel inventory and compile validation — component owners

- Build every HIP, CK, Opus, FlyDSL, and Triton/JIT module selected for strict.
- Classify each gfx1250 path as A0-compatible, B0-only, or unsupported pending
  owner work.
- Rebuild `csrc/opus_gemm/gen_co/gfx1250-strict` with the strict-capable LLVM
  toolchain only after the Opus owner confirms that each kernel is A0-safe.
- Do not copy or alias `hsa/gfx1250` into a strict directory; the current ASM
  inventory is B0-only.

Exit criteria: the build matrix has an explicit result and owner for every
selected module; unsupported paths fail clearly or dispatch to a validated
fallback.

### 3. Runtime and tuning validation — kernel/performance owners

- Run representative correctness suites on MI450 A0 hardware.
- Generate strict-specific tuning rows where B0 tuning is not demonstrated to
  be valid. Reuse should be explicit and evidence-based, not a filename alias.
- Compare fallback coverage and performance against the former hotswap path.
- Verify that a strict code object is rejected on B0 and a regular gfx1250 code
  object is rejected on A0.

Exit criteria: required model/operator coverage passes on A0 and known gaps are
documented with owner, fallback, and delivery date.

## Schedule

- October 21–23: open the infra Draft PR and run CPU-only tests.
- October 24–28: validate with a strict-capable compiler; publish the kernel
  inventory and assign component owners.
- October 29–November 3: A0 functional validation, Opus/FlyDSL/HIP fixes, and
  strict tuning decisions.
- November 4–6: regression, packaging, negative compatibility tests, and release
  sign-off.
- November 7: delivery target.

## Dependencies and risks

- A ROCm toolchain containing the `gfx1250-strict` compiler target.
- Runtime enumeration that reports the complete strict target name.
- MI450 A0 hardware and sufficient test capacity.
- Component-owner confirmation for B0-specific instructions and performance
  assumptions, especially precompiled Opus code objects.
- Downstream RCCL and math-library builds that use the same target identity.

The infra PR can merge before hardware validation because it is conservative:
it separates identities and blocks known-incompatible prebuilt binaries. Kernel
enablement remains gated on component validation.
