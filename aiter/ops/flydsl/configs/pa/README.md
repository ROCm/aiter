# FlyDSL paged-attention tuning results

`prepare_pa_decode_plan` stores and loads native FlyDSL autotune results in the
directory for the GPU running the query:

```text
aiter/ops/flydsl/configs/pa/<arch>/_pa_decode_autotuner.json
```

For example, MI300/MI308 results use `gfx942/` and MI350/MI355 results use
`gfx950/`. FlyDSL creates the directory and JSON file when tuning saves a result.
Preparing a plan without a cached result uses the `2*CU` default and does not
create a file.

Set `FLYDSL_AUTOTUNE=1` to tune with the actual caller tensors before inference
or graph capture, then disable it to reuse the saved configuration. FlyDSL owns
the JSON format, candidate benchmarking, selection, and cache persistence. Cache
keys retain attention geometry, execution modes, CU count, PA source contents,
and native toolchain/environment/device fingerprints.

To use a custom cache directory, set `FLYDSL_AUTOTUNE_CACHE_DIR` before importing
the PA module. This overrides the full cache directory, so the file is saved as
`<FLYDSL_AUTOTUNE_CACHE_DIR>/_pa_decode_autotuner.json`. Use separate directories
for parallel tuning processes. Restart inference processes after updating results
on disk, because native tuners keep loaded configurations in memory.
