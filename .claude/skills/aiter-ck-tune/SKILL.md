---
name: aiter-ck-tune
description: >
 Tune an existing Composable-Kernel (CK-tile) op in AITER — typically a
 GEMM variant. Thin hook around the per-op READMEs (e.g.
 `csrc/ck_gemm_a8w8/README.md`) which are the authoritative tuning guides.
 Use this skill when the user says "tune a8w8 GEMM", "add a new GEMM
 shape", "retune for gfx950", or "my shape falls back to a generic
 kernel". For MoE tuning specifically see `aiter-moe-tuning`.
allowed-tools: Bash Read Edit Grep Glob
---

# Tune a CK Kernel in AITER

Authoritative per-op READMEs live next to the tuner script, e.g.:

- `csrc/ck_gemm_a8w8/README.md` (A8W8 GEMM)
- `csrc/ck_gemm_a8w8_bpreshuffle/` (A8W8 B-preshuffled)
- `csrc/ck_gemm_a8w8_blockscale/`
- `csrc/ck_gemm_moe_2stages_codegen/README.md` (MoE; use `aiter-moe-tuning`)

Always read the relevant README first — columns and flags change over time.

## The universal tuning loop

Every CK op that supports tuning follows the same 4-step pattern:

1. **Add the shape** to `aiter/configs/<op>_untuned_gemm.csv` (or the op's
   untuned CSV). Columns differ per op; copy an existing row as a template.
2. **Run the tuner**: `python3 csrc/ck_<op>/<op>_tune.py -i <untuned.csv> -o <tuned.csv>`
   on a machine with the target GPU. It JIT-builds `module_<op>_tune`, sweeps
   instances, benchmarks, appends winners to `aiter/configs/<op>_tuned_gemm.csv`.
3. **Force runtime to pick up the new row**: `AITER_REBUILD=1` on next import,
   and delete any merged cache under `/tmp/aiter_configs/` (Linux) or
   `$env:TEMP\aiter_configs` (Windows).
4. **Validate**: `AITER_LOG_TUNED_CONFIG=1 python3 op_tests/test_<op>.py ...`
   prints the kernel id that was selected — confirm it's your new row.

## Finding the tuner for your op

```bash
ls csrc/ | grep '^ck_gemm'
ls csrc/ck_gemm_a8w8/*_tune.py
```

The tuner scripts follow the naming `csrc/ck_<op>/<op>_tune.py`. Their
`--help` is the source of truth for flag names.

## Config resolution at runtime

`aiter/jit/core.py` defines env vars of the form `AITER_CONFIG_GEMM_A8W8`
(and similar) that override the default CSV path. AITER also *merges*
per-model CSVs from `aiter/configs/model_configs/<model>/` with the main
CSV into `/tmp/aiter_configs/<csv_name>` (or the Windows temp dir). Stale
merged files are the #1 source of "I tuned this but it doesn't take effect".

## Arch matters

Tuned rows are keyed by `cu_num` (which encodes the GPU SKU / arch).
Re-tune on every target arch; rows from gfx942 are not portable to gfx950.

## Recurring footguns

| Symptom | Cause / fix |
|---------|-------------|
| Tuner finishes in seconds | Row already in tuned CSV. Delete that row or use `--force` / `-k` depending on script. |
| Tuned row ignored at runtime | Stale merge cache at `/tmp/aiter_configs/`. Delete it, set `AITER_REBUILD=1`. |
| Tuner can't find a compatible instance | The instance list doesn't cover your `(dtype, q_type, ...)`. Regenerate with `gen_instances.py` in the same folder, or extend the instance list. |
| HIP OOM mid-tune | Lower `MAX_JOBS`, tune one shape at a time, or pass `-m <N>` to limit parallel instance builds. |
| Correctness regression after tune | The chosen instance is numerically weak for that shape. Re-tune with stricter tolerance, or blacklist the bad instance id in the tuner. |
| Wrong kernel picked at runtime | Column mismatch — `q_dtype_w`, `cu_num`, dtype string must match call-site exactly. Case-sensitive. |

## When to escalate
- Instance list doesn't cover your `(dtype, q_type)` pair → check
  `csrc/ck_<op>/gen_instances.py` and regenerate, or add a new CK template.
- MoE-specific workflow → `aiter-moe-tuning`.
- JIT rebuild fails after tuning → `aiter-jit-debug`.
