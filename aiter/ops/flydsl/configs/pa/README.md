# FlyDSL paged-attention tuning results

`prepare_pa_decode_plan` in `aiter.ops.flydsl.pa_decode_tuning` stores and loads
native FlyDSL autotune results in the directory for the GPU running the query:

```text
aiter/ops/flydsl/configs/pa/<arch>/_pa_decode_autotuner.json
```

For example, MI300/MI308 results use `gfx942/` and MI350/MI355 results use
`gfx950/`. FlyDSL creates the directory and JSON file when tuning saves a result.
Preparing a plan without a cached result uses the `2*CU` default and does not
create a file.

Run the standalone tuning script from the repository root or an installed AITER
package. It constructs the requested shape, benchmarks native candidates, and
saves the winner without requiring an autotune environment variable:

```bash
python -m aiter.ops.flydsl.pa_decode_tuning \
    --batch-size 32 --num-query-heads 16 --num-kv-heads 1 --head-dim 128 \
    --query-length 4 --context-length 16384 --block-size 128 \
    --kv-dtype fp8 --scale-mode per-token --trans-v 1 \
    --output-dir /tmp/pa-configs
```

This saves `/tmp/pa-configs/<arch>/_pa_decode_autotuner.json`. Omit `--output-dir`
to use the package directory shown above, or the environment override described
below. Each invocation searches the requested shape and retains cached results
for other shapes. `--context-length` accepts one length for all batch rows or one
length per row; lengths include query tokens. Use `--kv-dtype bf16` for unscaled
BF16 K/V and `--help` for all shape, layout, window, sink, and device options.

Python callers can import `prepare_pa_decode_plan` from
`aiter.ops.flydsl.pa_decode_tuning` (or `aiter.ops.flydsl`) and pass their actual
tensors with `tune=True, config_dir="/tmp/pa-configs"` before inference or graph
capture. Subsequent calls with the same `config_dir` use saved results by default.
`FLYDSL_AUTOTUNE=1` remains available to force searches through the Python API.
FlyDSL owns the JSON format, candidate benchmarking, selection, and cache
persistence. Cache keys retain attention geometry, execution modes, CU count,
PA source contents, and native toolchain/environment/device fingerprints.

To use a custom cache directory, set `FLYDSL_AUTOTUNE_CACHE_DIR` before importing
the PA tuning module. This overrides the full cache directory, so the file is
saved as `<FLYDSL_AUTOTUNE_CACHE_DIR>/_pa_decode_autotuner.json`. Use separate
directories for parallel tuning processes. Restart inference processes after
updating results on disk, because native tuners keep loaded configurations in memory.
