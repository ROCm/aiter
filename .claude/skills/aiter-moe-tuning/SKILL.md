---
name: aiter-moe-tuning
description: >
 Tune AITER's fused Mixture-of-Experts (MoE) op. Covers the
 `aiter/configs/tuned_fmoe.csv` + `aiter/configs/untuned_fmoe.csv` flow,
 per-model configs under `aiter/configs/model_configs/`, running
 `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` (or the current
 MoE tuner), regenerating CK instances, rebuilding the JIT module, and
 validating with `op_tests/test_moe.py`.
 Use this skill when a user asks to "tune MoE", "add a new MoE shape",
 "tune Mixtral / DeepSeek / Llama-4 MoE", or reports a
 `[AITER] NOTICE: no tuned config for fmoe` log line.
allowed-tools: Bash Read Edit Grep Glob Write
---

# Tune AITER Fused MoE

Fused MoE in AITER is dispatched through a CK-tile two-stage GEMM
(codegen lives in `csrc/ck_gemm_moe_2stages_codegen/` and the CK-tile
variant in `csrc/ck_tile_gemm_moe_2stages/`). At runtime the op reads
`aiter/configs/tuned_fmoe.csv` and picks a per-shape kernel instance;
if the shape isn't present, it falls back to a generic kernel and
logs a notice.

## Step 0: Confirm which MoE we're tuning

AITER has several MoE variants. Ask / verify which one the user hits:

| Python entry point | Config file | Tuner |
|--------------------|-------------|-------|
| `aiter.fused_moe` (CK 2-stage) | `aiter/configs/tuned_fmoe.csv` | `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` |
| `aiter.ops.triton.moe.*` | `aiter/configs/*fused_moe*.csv` (Triton) | scripts under `aiter/ops/triton/moe/` |
| ASM MoE (`asm_moe`) | ASM-codegen'd, not CSV-tuned | n/a |

The rest of this skill focuses on **CK 2-stage fused MoE**, which is the
most common path. Adapt the file names if the user is on Triton MoE.

```bash
rg -n "tuned_fmoe|untuned_fmoe|fused_moe_tune" aiter/ csrc/
ls aiter/configs | grep -i fmoe
```

## Step 1: Read the current CSVs

```bash
head -n 5 aiter/configs/untuned_fmoe.csv
head -n 5 aiter/configs/tuned_fmoe.csv
```

Column layout (verify with `head -n 1` of the current file — it may evolve):

```
token,model_dim,inter_dim,expert,topk,dtype,q_type,use_g1u1,doweight_stage1
```

`tuned_fmoe.csv` extends that with the chosen kernel config id (e.g. a
CK `MoeKernel` tag or BLOCK_M / BLOCK_N / BLOCK_K / KSPLIT).

Per-model overrides live in `aiter/configs/model_configs/<model>/`, e.g.
`model_configs/mixtral-8x7b/tuned_fmoe.csv`. If the user targets a specific
model, tune there instead of the top-level CSV.

## Step 2: Add the new shape(s) to `untuned_fmoe.csv`

Compute the shape tuple from the model config:

- `token`: the batch × seq of tokens you want optimized for (common: 1, 32, 64, 128, 256, 512, 1024, 2048, 4096).
- `model_dim`: hidden size (e.g. 4096 for Llama-3-8B, 7168 for DeepSeek-V3).
- `inter_dim`: MoE FFN intermediate dim **per-expert**.
- `expert`: total #experts (e.g. 8, 64, 256).
- `topk`: routed experts per token (usually 2 or 8).
- `dtype`: `bf16` / `fp16`.
- `q_type`: `No` (unquantized), `fp8`, `int8`, `int4` — match the runtime call.
- `use_g1u1`: `1` if SwiGLU-style gated (most models), else `0`.
- `doweight_stage1`: `1` or `0` — match how the Python side calls MoE.

Append one line per shape. **Don't delete existing lines** — they're used by
other users and CI. If your shape already exists, skip.

```bash
# Example: add Llama-4-scout-like MoE shapes
cat >> aiter/configs/untuned_fmoe.csv <<'EOF'
128,4096,8192,16,1,bf16,No,1,0
512,4096,8192,16,1,bf16,No,1,0
2048,4096,8192,16,1,bf16,No,1,0
EOF
```

## Step 3: Run the tuner

```bash
export GPU_ARCHS=gfx942                 # or gfx950 / gfx90a — your target
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
    --untuned_file aiter/configs/untuned_fmoe.csv \
    --tuned_file  aiter/configs/tuned_fmoe.csv \
    --arch $GPU_ARCHS
```

Flags vary per script version — always run `--help` first:

```bash
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --help
```

The tuner:

1. Parses `untuned_fmoe.csv`, skipping rows already present in `tuned_fmoe.csv`.
2. Enumerates CK `MoeKernel` instances compatible with `(dtype, q_type, use_g1u1)`.
3. Builds each instance (this can take a while; parallelize with `MAX_JOBS`).
4. Benchmarks on the live GPU and picks the fastest.
5. Appends a line to `tuned_fmoe.csv`.

Typical runtime: 5–30 min per shape depending on #instances.

Speed up:

```bash
export MAX_JOBS=$(nproc)        # parallel compile
# For quick sanity runs, many AITER tuners support --fast or --subset
```

## Step 4: Regenerate CK instance files (only if needed)

If the tuner selected a kernel config that wasn't in the pre-generated
instance list, you need to (re)run the MoE instance generator. Typical
pattern:

```bash
python csrc/ck_gemm_moe_2stages_codegen/gen_instances.py --arch $GPU_ARCHS
# The CK-tile variant has its own generator:
#   csrc/ck_tile_gemm_moe_2stages/gen_instances.py
```

Check `aiter/jit/optCompilerConfig.json` for the `module_moe_ck2stages*`
entry to see its `blob_gen_cmd` — that is the authoritative generator
command for CK instance files.

## Step 5: Rebuild the JIT module

```bash
export AITER_REBUILD=1            # force recompile
python -c "import aiter; aiter.fused_moe; print('ok')"
```

If `PREBUILD_KERNELS=1` was used at install time, do a proper reinstall:

```bash
PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCHS pip install -e . --no-build-isolation
```

## Step 6: Validate correctness

```bash
pytest -q op_tests/test_moe.py::test_fmoe -k "bf16 and 128"
# or the specific parametrization matching your shape
```

Correctness first, then perf:

```bash
# Triton-backed MoE benchmark (adjust flags via --help):
python op_tests/op_benchmarks/triton/bench_moe.py --help
# For the CK-tile 2-stage path, reuse op_tests/test_moe.py with --benchmark or
# write a small perftest driver — see the `benchmark-aiter-op` skill.
```

Expect the new shape's TFLOPS to be ≥ the previous fallback by a clear
margin (often 1.2–3×). If not, the tuner probably hit a bad search space —
re-check the `use_g1u1` / `q_type` / `doweight_stage1` columns.

## Step 7: Per-model config (optional)

For model-specific shapes, put the tuned rows under:

```
aiter/configs/model_configs/<model_name>/tuned_fmoe.csv
```

AITER will prefer the model-specific CSV when
`AITER_MODEL_CONFIG=<model_name>` is set (grep `model_configs` in
`aiter/jit/core.py` / `aiter/ops/` to confirm the exact env variable name
in the current checkout):

```bash
rg -n "model_configs" aiter/
```

## Step 8: Commit

```bash
git add aiter/configs/untuned_fmoe.csv \
        aiter/configs/tuned_fmoe.csv \
        aiter/configs/model_configs/  # if touched
git commit -m "tune: fused MoE for <model / shape-set> on <arch>"
```

**Do not commit** the `aiter/jit/build/` directory.

## Common pitfalls

| Symptom | Cause / fix |
|---------|-------------|
| `[AITER] NOTICE: no tuned config for fmoe ...` at runtime | Your shape isn't in `tuned_fmoe.csv`. Follow steps 2–5. |
| Tuner says `no compatible MoeKernel for (dtype=..., q_type=..., use_g1u1=...)` | The instance list doesn't cover that combination yet. Run the instance generator (Step 4) with a wider scope, or extend the kernel list in `csrc/ck_gemm_moe_2stages_codegen/` and regenerate. |
| Tuning finishes in seconds | The row was already in `tuned_fmoe.csv`; the tuner skipped it. Delete that row and re-run, or use `--force`. |
| Correctness regression after tuning | The chosen instance has numerical issues at that shape. Re-run the tuner with stricter tolerance or blacklist the bad instance in the tuner script. |
| `HIP OOM` during tuning | Lower `--batch` / `--token` to the sweep, or tune one shape at a time. |
| Tuned row ignored at runtime | Shape mismatch — make sure `token` is the *actual* token count you hit at runtime, and `q_type` / `use_g1u1` / `doweight_stage1` columns match the Python call-site exactly. Case-sensitive. |
| Tuner hangs compiling | `MAX_JOBS` may be too high. Lower to `nproc/2` and retry; CK instance compiles are memory-hungry. |

## References
- `csrc/ck_gemm_moe_2stages_codegen/README.md`
- `aiter/fused_moe.py` — Python entry point and dispatch
- `aiter/fused_moe_dp_shared_expert.py` — DP + shared-expert variant
- `aiter/configs/untuned_fmoe.csv`, `aiter/configs/tuned_fmoe.csv`
- `op_tests/test_moe.py` — reference tests
