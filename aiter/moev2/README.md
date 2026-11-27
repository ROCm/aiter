# aiter/moev2

## Status / ownership

`aiter.moev2` is under active development (the MoE v2 migration) and no path has
graduated. Owners: @qli88. Please coordinate with them before changing:

- any file under `aiter/moev2/`. The legacy MoE path runs helpers that live here:
  `policy.py`, `shapes.py`, `segments/defaults.py`, `segments/sort.py`, `ck_moe_stage1`
  and `_require_shuffled_gfx950_fp8` in `segments/ck.py`, `asm_stage1` in
  `segments/asm.py`, `execute/metadata.py`, `execute/output.py` (moved from
  `aiter/fused_moe.py`, which re-exports them) and the table helpers in
  `dispatch/tuned.py`. `aiter.fused_moe` also re-exports `routing/topk.py` and
  `reference/torch_moe.py` for its callers and tests. Changes must keep legacy behavior;
- the hook sites in `aiter/fused_moe.py` (comments starting `NOTE: aiter.moev2`);
- legacy code that `aiter.moev2` copies instead of calling, where a fix must be mirrored:
  `_fused_moe_impl` and `fused_moe_2stages` (`execute/executor.py`, `api.dispatch_key`),
  `get_2stage_cfgs` (`dispatch/resolver.py`, `dispatch/ranking.py`), and the arguments
  `fused_moe` passes to `fused_moe_` (`api.op_call`).

Checks after a change on either side (`tests/moe_dispatch/`):

    python tests/moe_dispatch/snapshot.py --check        # legacy dispatch unchanged
    python tests/moe_dispatch/snapshot.py --update       # after an intended change; paste its output into the PR
    python tests/moe_dispatch/differential.py --check    # v2 plans equal legacy's (CPU)
    AITER_MOEV2_GPU_DIFFERENTIAL=1 python -m pytest tests/test_moev2_differential.py  # outputs (GPU)
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt --compare                   # GPU suites, off vs v2

## Overview

MoE v2 rebuilds `fused_moe` dispatch and execution from small parts: a problem
description, segments (one per kernel family) and a plan that chains them. It lives
next to the legacy code (`aiter/fused_moe.py`, `aiter/fhmoe.py`,
`aiter/fused_moe_bf16_asm.py`) and takes over one group of calls (a *path*) at a time.

- `AITER_MOEV2` selects the paths: unset runs the graduated paths, `off` none, `on`
  every registered path, and `a8w8_pt,a4w4` exactly the listed ones. No path is
  graduated, so by default every call runs the legacy code.
- `aiter/moev2` never imports the legacy MoE modules; the legacy modules may import from
  it (`tests/test_moev2_imports.py` enforces this).
- Helpers that both sides need are moved here verbatim and imported back into
  `aiter/fused_moe.py` under their old names, so existing imports keep working.

Status: one path, `a8w8_pt` (per-tensor fp8), is registered and not graduated.

## How a call flows

1. The legacy `fused_moe` (entry `api`) or `fused_moe_` (entry `op`) asks
   `paths.route_to_v2` whether an enabled path claims the call.
2. The path builds a `MoeProblem` from the call and resolves it to a plan: the tuned row
   if one applies (checked as the legacy code checks it), else the default ranking. If
   the row needs a kernel family the path has not ported, the path returns `DECLINED` and
   the legacy code runs the call.
3. The executor runs the plan's segments in order.

Lifetime: *permanent* files stay after the migration. *To be removed* files exist only
for the migration and are deleted when every path has graduated and the legacy code
is retired (legacy `fused_moe.py` and `fhmoe.py` become re-export shims over
`aiter.moev2`).

## Package files

| File | Lifetime | Purpose |
|---|---|---|
| `__init__.py` | permanent | Package docstring: coexistence with the legacy code and the import rule. |
| `envs.py` | permanent; the `AITER_MOEV2` accessor is removed | `AITER_MOEV2` parsing, read once and cached (tests call `reload()`). |
| `paths.py` | to be removed | Path registry: `route_to_v2(call, entry)` picks the path for a call, `DECLINED` hands it back to legacy. Graduation can be scoped to archs. Logs the selection once; with `AITER_MOE_EXPLAIN=1`, why an active path does not claim a call. |
| `api.py` | permanent; `op_call` and `dispatch_key` are removed | `aiter.moev2.fused_moe` with the legacy signature (always v2; raises when no path takes the call; like legacy, clears a registered presort when the call returns or raises); `build_problem` and `dispatch_key` (a call to its `MoeProblem` and legacy key); `op_call` (the `fused_moe_` arguments legacy passes). |
| `problem.py` | permanent | `MoeProblem`: what one call must compute (formats, shapes, flags, never tensors). Round-trips the legacy `get_2stage_cfgs` key, so v2 can be compared with legacy key by key. |
| `hooks.py` | permanent; `legacy_kernel_bench` is removed | Context managers for tests and benchmarks (kernel bench callable, sort implementation) that replace assigning legacy module globals. `legacy_kernel_bench` forwards legacy's `kernel_bench_callable` global to v2 runs. |
| `policy.py` | permanent | Activation quantization policy, moved from `fused_moe.py`: `resolve_activation_dtype`, `quant_remap`, `_normalize_mxfp4_activation_params`. |
| `shapes.py` | permanent | Shape helpers, moved from `fused_moe.py`: `get_inter_dim`, `nextPow2`, `get_padded_M`, `_is_row_major_2d`. |
| `routing/topk.py` | permanent | Top-k routing, moved from `fused_moe.py`: `fused_topk`. |
| `reference/torch_moe.py` | permanent | PyTorch reference MoE that tests check backends against, moved from `fused_moe.py`: `torch_moe`, `torch_moe_stage1/2`, the activations and the per-1x32 dequant helpers. |
| `dispatch/ops.py` | permanent | The MoE op chain that every plan covers: sort, activation quant, GEMM1, activation, intermediate quant, GEMM2, reduce. |
| `dispatch/buf.py` | permanent | Boundary formats between adjacent segments. |
| `dispatch/plan.py` | permanent | A plan: the segments that cover the op chain, with their parameters. |
| `dispatch/kernel_names.py` | permanent | One parser for every kernel-name family in tuned tables; `parse(n).format() == n` for every shipped name. |
| `dispatch/tuned.py` | permanent | Tuned-table reader (shared with legacy `get_2stage_cfgs`) and `lookup` with the legacy fallbacks; `@cached` and `clear_cache()`, which reset every v2 cache together. |
| `dispatch/ranking.py` | permanent | The default ranking table used when no tuned row applies, with static checks that it is unambiguous and complete. |
| `dispatch/resolver.py` | permanent | Resolves a `MoeProblem` to a plan in the legacy order, collecting the legacy log lines and raising where legacy raises (unshuffled fp8 CK weights on gfx950); declines rows for families no segment owns and post-reduction rows. |
| `dispatch/overrides.py` | permanent | Debug overrides of dispatch decisions (`AITER_BYPASS_TUNE_CONFIG`, `AITER_USE_NT`, `AITER_ONLINE_TUNE`, ...), read where legacy reads them. |
| `dispatch/profile.py` | permanent | Measured runners-up per tuned key; empty for now. |
| `segments/base.py` | permanent | The `Segment` interface. A segment declares what it can run (`Cap`); anything a problem has that no `Cap` declares rejects the segment. |
| `segments/sort.py` | permanent | Token sorting: `moe_sorting` and its helpers, moved from `fused_moe.py`, which imports them (Opus, CK or FlyDSL backend; local topk ids, the adaptive auxiliary sort, FLAT's unsorted input), and segment `sort`, which runs the plain sort through it. |
| `segments/quant.py` | permanent | Activation quantization before GEMM1 and between the GEMMs. |
| `segments/ck.py` | permanent | CK 2-stage GEMM kernels; `ck_moe_stage1` and `_require_shuffled_gfx950_fp8` moved from `fused_moe.py`. |
| `segments/asm.py` | permanent | ASM stage-1 kernels; `asm_stage1` moved from `fused_moe.py`. |
| `segments/whole_graph.py` | permanent | Whole-graph implementations (`impl__<name>__<config>`) from `aiter.fused_moe_registry`. |
| `segments/defaults.py` | permanent | Default kernel parameters (`BLOCK_SIZE_M`, `get_block_size_M`, `get_ksplit`, `use_nt`), moved from `fused_moe.py`. |
| `execute/executor.py` | permanent | Runs a plan on one call, statement for statement as the legacy `_fused_moe_impl` / `fused_moe_2stages`. |
| `execute/metadata.py` | permanent | `MOEMetadata` and `AsmMoeMode`, moved from `fused_moe.py`. |
| `execute/output.py` | permanent | The caller's `output=` buffer: validation and the final copy, moved from `fused_moe.py`. |
| `migration/__init__.py` | to be removed | The migration paths, one module per path. |
| `migration/a8w8_pt.py` | to be removed | Path `a8w8_pt`: which calls it claims, which it declines (including a call whose second lookup after the sort, which only logs, would raise or be declined), and its registration. |
| `tuning/__init__.py` | permanent | Empty subpackage; a later step adds the tuner. |

## Tests

| File | Lifetime | Purpose |
|---|---|---|
| `tests/test_moev2_imports.py` | permanent | `aiter/moev2` imports no legacy MoE module. |
| `tests/test_moev2_switch.py` | to be removed | `AITER_MOEV2` parsing and routing; the v2 entry point keeps the legacy signature. |
| `tests/test_moev2_hooks.py` | permanent | The hooks restore the previous state. |
| `tests/test_moev2_differential.py` | to be removed | Runs the differential harness: dispatch on the CPU, outputs on the GPU (with `AITER_MOEV2_GPU_DIFFERENTIAL=1`; CI runs the rest). |
| `tests/test_moev2_dispatch.py` | permanent; the tests against legacy are removed | Unit tests of the problem, plans and capabilities, kernel names, resolver and overrides, the executor, and `a8w8_pt`; the snapshot's feature variants are valid legacy calls. |
| `tests/test_moev2_ranking.py` | permanent | Unit tests of the ranking table and its checks. |
| `tests/test_moe_dispatch_snapshot.py` | permanent | Legacy dispatch matches the golden snapshot. |
| `tests/moe_dispatch/keys.py` | permanent; the feature variants are removed | Dispatch keys per simulated arch: every tuned row plus a grid of untuned keys. For the quant types v2 paths take (`FEATURE_QUANT_TYPES`), feature variants: the same call with one more `get_2stage_cfgs` feature (`num_local_tokens`, activation scales, bias, padding, ...), so the snapshot names the key whose decision changed. |
| `tests/moe_dispatch/arch_sim.py` | permanent | Runs dispatch as if on a given GPU arch, in a process without a GPU. |
| `tests/moe_dispatch/snapshot.py`, `record.py` | permanent | Generate, check or update the legacy dispatch snapshot (`golden/*.jsonl.gz`), read from frozen tuned tables (`golden/inputs/`). |
| `tests/moe_dispatch/differential.py` | to be removed | Requires v2 to make the same dispatch choice as legacy for every key a path claims. |
| `tests/moe_dispatch/key_checks.py` | to be removed | Per key: the problem/legacy key round trip, and the tuned lookup returns legacy's row. |
| `tests/moe_dispatch/ranking_checks.py` | permanent | Every rankable snapshot key lands in one ranking entry and region, and every region is reached. |
| `tests/moe_dispatch/gpu_differential.py` | to be removed | On a GPU, a path launches the same kernels on the same inputs as legacy and gives the same output. |
| `tests/moe_dispatch/inventory.py` | to be removed | Every name used from the legacy MoE modules, and every monkeypatch of them. |
| `tests/moe_dispatch/gpu_suites.py` | to be removed | The GPU test suites each path must pass with `AITER_MOEV2=<path>`; `--compare` runs them with `off` and with the path. |
| `tests/moe_dispatch/perf_baseline.py` | permanent | End-to-end MoE performance baseline (`golden/perf_gfx950.json`). |
