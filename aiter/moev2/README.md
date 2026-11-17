# aiter/moev2

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

Status: scaffolding only (the switch, an empty path registry, the moved helpers and the
tests that lock in legacy behavior). No path is registered yet.

Lifetime: *permanent* files stay after the migration. *To be removed* files exist only
for the migration and are deleted when every path has graduated and the legacy code
is retired (legacy `fused_moe.py` and `fhmoe.py` become re-export shims over
`aiter.moev2`).

## Package files

| File | Lifetime | Purpose |
|---|---|---|
| `__init__.py` | permanent | Package docstring: coexistence with the legacy code and the import rule. |
| `envs.py` | permanent; the `AITER_MOEV2` accessor is removed | `AITER_MOEV2` parsing, read once and cached (tests call `reload()`). |
| `paths.py` | to be removed | Path registry. The legacy `fused_moe` asks `route_to_v2(call)` which path takes a call; with no path active this is one cached boolean check. |
| `api.py` | permanent | `aiter.moev2.fused_moe`, with the legacy signature. It always runs v2 and raises when no path takes the call. |
| `hooks.py` | permanent | Context managers for tests and benchmarks (kernel bench callable, sort implementation) that replace assigning legacy module globals. |
| `policy.py` | permanent | Activation quantization policy, moved from `fused_moe.py`: `resolve_activation_dtype`, `quant_remap`. |
| `shapes.py` | permanent | Shape helpers, moved from `fused_moe.py`: `get_inter_dim`. |
| `routing/topk.py` | permanent | Top-k routing, moved from `fused_moe.py`: `fused_topk`. |
| `reference/torch_moe.py` | permanent | PyTorch reference MoE that tests check backends against, moved from `fused_moe.py`: `torch_moe`, `torch_moe_stage1/2`, the activations and the per-1x32 dequant helpers. |
| `dispatch/tuned.py` | permanent | The `@cached` decorator and `clear_cache()`: every v2 cache that depends on tuned tables, environment variables or the arch is reset together. |
| `dispatch/__init__.py`, `execute/__init__.py`, `segments/__init__.py`, `tuning/__init__.py` | permanent | Empty subpackages; later steps add the resolver, executor, kernel segments and tuner. |

## Tests

| File | Lifetime | Purpose |
|---|---|---|
| `tests/test_moev2_imports.py` | permanent | `aiter/moev2` imports no legacy MoE module. |
| `tests/test_moev2_switch.py` | to be removed | `AITER_MOEV2` parsing and routing; the v2 entry point keeps the legacy signature. |
| `tests/test_moev2_hooks.py` | permanent | The hooks restore the previous state. |
| `tests/test_moev2_differential.py` | to be removed | Runs the differential harness: dispatch on the CPU, outputs on the GPU. A no-op until a path registers. |
| `tests/test_moe_dispatch_snapshot.py` | permanent | Legacy dispatch matches the golden snapshot. |
| `tests/moe_dispatch/keys.py` | permanent | Dispatch keys per simulated arch: every tuned row plus a grid of untuned keys. |
| `tests/moe_dispatch/arch_sim.py` | permanent | Runs dispatch as if on a given GPU arch, in a process without a GPU. |
| `tests/moe_dispatch/snapshot.py`, `record.py` | permanent | Generate, check or update the legacy dispatch snapshot (`golden/*.jsonl.gz`), read from frozen tuned tables (`golden/inputs/`). |
| `tests/moe_dispatch/differential.py` | to be removed | Requires v2 to make the same dispatch choice as legacy for every key a path claims. |
| `tests/moe_dispatch/inventory.py` | to be removed | Every name used from the legacy MoE modules, and every monkeypatch of them. |
| `tests/moe_dispatch/gpu_suites.py` | to be removed | The GPU test suites each path must pass with `AITER_MOEV2=<path>`. |
| `tests/moe_dispatch/perf_baseline.py` | permanent | End-to-end MoE performance baseline (`golden/perf_gfx950.json`). |
