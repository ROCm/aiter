# GEMM tuning

One script tunes Triton and Gluon GEMMs through their normal config lookup.
`gemm_cases.py` builds inputs and returns a callable for each GEMM.
`tune_gemm.py` temporarily answers its `get_gemm_config()` calls with each
candidate, then benchmarks the callable with `triton.testing.do_bench`.

## Run it

Use an AITER source checkout with its test dependencies, Triton, ROCm and a
supported AMD GPU. Run on each target GPU: gfx942, gfx950, gfx1250, or a future
architecture with a working kernel and `DEFAULT.json`.

```bash
cd aiter/ops/triton/utils/_triton/tuning
python3 tune_gemm.py --list
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 --backend gluon
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py batched_gemm_bf16 M=32 N=128 K=512 B=4
```

`--list` works without a GPU. `--backend` selects a backend exposed by the
wrapper; omitting it uses the wrapper's default. See [COVERAGE.md](COVERAGE.md)
for available cases, previously missing harnesses, and hardware restrictions.

## The three rules

1. Benchmark the config the wrapper currently resolves, including its
   `DEFAULT.json` fallback. Keep its outputs as the reference when it works.
2. Try every candidate. Check finite outputs and compare with the reference
   before timing. Log a failed config and its full traceback to
   `errors-<op>-<shape>.txt`, then continue. A fatal GPU fault stops the run.
3. Save the fastest working config if it beats the current one, or if the
   lookup reports no tuned config for that shape's M bucket. Otherwise leave
   the file alone. A working default can be the winner.

The benchmark times the whole callable, including reductions, output resets,
and wrapper work. Inputs are created once before timing. If the current config
fails, candidates can still be timed after finite-output checks; run the
kernel's independent correctness tests before trusting those results.

## Search values and output

The keys come from the target arch/backend's required `DEFAULT.json`. Common
launch keys use shared ranges; other keys use values already in the family's
JSON files. A case's `space` can constrain those values. Override them with:

```bash
python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 \
    --space BLOCK_SIZE_M=16,32 BLOCK_SIZE_K=64,128 num_warps=4,8
```

Search keys absent from that backend's `DEFAULT.json` are reported and skipped.

The observed lookup supplies the family, backend, shape, filename and M
bucket. Results go under `configs/<arch>/<backend>/gemm/<family>/`; existing
buckets are preserved. Run jobs that write the same JSON file sequentially.
Modes sharing a lookup also share tuned results. Restart consumers after
installing configs because the loader caches them.

## Add a GEMM

1. Use `get_gemm_config()` in the wrapper and provide a valid `DEFAULT.json`
   for every supported arch/backend. The author owns the config keys: they
   must match what that implementation reads; no common schema is required.
2. Add a `@gemm_case()` function named for the wrapper or variant. Reuse its
   test input generator and return a callable that calls the wrapper without
   `config=`, returns all outputs, and resets any accumulators each time.
   Keep imports inside the case and forward an exposed `backend` argument.
3. Add constrained values with `@gemm_case(space={...})` when needed. Update
   this README, [COVERAGE.md](COVERAGE.md), the [Triton README](../../../README.md)
   and [Copilot instructions](../../../../../../.github/instructions/aiter-ops-triton.instructions.md)
   if the author needs to do something new. Follow the [config rules](../../../configs/CLAUDE.md).

CPU checks: `python3 -m unittest discover -s op_tests/triton_tests/gemm -p test_tune_gemm.py`
from the repo root. Run the relevant GEMM correctness tests on the target GPU.
