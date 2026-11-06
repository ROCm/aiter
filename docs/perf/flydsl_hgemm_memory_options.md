# FlyDSL HGEMM: K-stagger and B non-temporal loads

## Scope

The AKLR small-M BF16 GEMMs expose a gap that is not attributable to K-stagger alone. This change adds two independent, default-off tuning options to the existing gfx950 FT/HTI kernel; it does not copy AKLR kernels, change precision, change the reduction protocol, or enable a new dispatch default.

- `k_stagger`: rotate global K tile loads by `(block_idx.x * k_stagger) % partition_tiles`, wrapping within each actual split-K partition, including a shorter final partition. LDS ring indices do not rotate.
- `b_non_temporal`: use the raw **async** LDS load intrinsic with the non-temporal bit on B only. The existing asyncmark groups and waits still track completion. A stays cached.
- Names persist optional `_stgN` and `_bnt` fields. Old names keep their spelling and parse to off.
- The tuner appends combinations for small-M, one-M-tile candidates; original candidate order is retained. Existing tuned CSVs are not changed; use explicit options or re-tune to enable them.

## Measurement

Base AITER: `e1763addecdb2ea25be0c9d0af2a96fb3f62cf80`. AKLR: `551613c81ea1f232cadb72e6f4647f96152e67d4`.
GPU reports gfx950, 256 CUs; Torch 2.10.0+ROCm 7.2.4; FlyDSL **0.4.2**. No clocks were changed. These are device/graph timings, not application end-to-end timings.

The no-profiler benchmark cycles through at least 512 MiB of distinct weights. Inputs/outputs are preallocated; compilation and reference checks are outside the graph. Variant order is shuffled reproducibly per round. This streaming proxy is **not** AKLR’s per-call 1-GiB flush mode.
Raw samples and exact tile/stage/wave configs are in `flydsl_hgemm_memory_options.json`.

## Proving K-stagger alone before integration

These two cases use the existing upstream tuned tile, changing only the K order. Nine readings per arm, with an independent second run:

| M,N,K | Unmodified upstream (us) | Stagger 1 (us) | Speedup | Second run upstream / stagger (us) |
|---|---:|---:|---:|---:|
|16,18432,1536|12.172|11.686|1.042x|12.180 / 11.682|
|32,18432,1536|12.834|12.395|1.035x|12.850 / 12.399|

Default-off measured 12.168 and 12.811 us in the first run. Exploratory 0.3.4.1 compiler results are not used for this table.

## B cache policy ablation and remaining AKLR gap

Five readings per arm. Within each row all AITER arms use the same tile; these tiles are NOT necessarily the current dispatch winner. AKLR uses its own configuration, and its full implementation is timed with the same rotation.

| M,N,K | Same tile off (us) | Stagger 3 only | B-NT only | B-NT + stagger 3 | AKLR (us) | Best candidate / AKLR |
|---|---:|---:|---:|---:|---:|---:|
|16,18432,1536|12.466|12.131|11.190|11.025|9.795|1.126x|
|32,18432,1536|12.966|12.711|11.808|11.728|10.026|1.170x|
|64,18432,1536|14.354|14.075|13.221|13.277|11.493|1.150x|
|112,18432,1536|19.913|19.510|18.893|18.921|13.645|1.385x|
|16,20480,7168|49.242|51.130|43.339|45.109|42.439|1.021x|
|32,20480,7168|50.423|52.382|45.363|47.458|43.428|1.045x|
|64,20480,7168|55.188|56.653|50.260|52.227|48.618|1.034x|
|112,20480,7168|77.886|79.929|72.921|74.294|56.164|1.298x|

**Not a full gap closure.** B-NT gets the 20480x7168 M=16/32/64 cases within about 2–5% of AKLR in this run, with stagger disabled. M=112 remains around 30% slower; 18432x1536 retains a larger gap. Neither switch is a universal win. The 6144x16384 M=112 case regresses from 72.623 to 94.824 us with stagger 1 on the tested tile (see raw data). Retuning, not a global switch, is required.

An experiment decoupling the stored N band (72/80 columns) from a 96-column compute tile was slower and is not included. AKLR also changes operand ordering, pipeline scheduling, geometry and memory mapping; this PR makes no hardware-counter-backed claim about which mechanism explains the residual gap.

## Reproduce

```bash
export PYTHONPATH=$PWD
python op_tests/op_benchmarks/flydsl/bench_hgemm_k_stagger.py \
  --case 0 1 --repeat 9 --json stagger.json
python op_tests/op_benchmarks/flydsl/bench_hgemm_k_stagger.py \
  --case 8 9 10 11 12 13 14 15 --repeat 5 --b-non-temporal \
  --tuned --aklr-root /path/to/AKLR --json memory-options.json
pytest -q op_tests/flydsl_tests/test_hgemm_k_stagger.py
python -m unittest op_tests.tuning_tests.test_a16w16_tuned_config_lookup -v
```

`--tuned` adds the actual AITER dispatch as an independent arm (it can select another backend). `--rotate-mib 0` measures hot replay. `--baseline-kernel /path/to/original.py` adds the unmodified gfx950 source, extracted from the base commit.

## Correctness

253 tests passed on gfx950 / FlyDSL 0.4.2. Includes exact binary-fraction probes, random FP32 output against FP64 reference, input/output guards, all four low-level layouts, FT/HTI, minimum pipeline depth, odd/non-power-of-two tile counts, split-K/slice-K, short final partitions, bias, BF16/FP16/FP32, changed-input graph replay, names/invalid options and tuner augmentation. Other GPUs are untested; nonzero options are rejected on gfx1250 and skinny.

## AKLR per-call cold-cache protocol

Seven readings per arm using AKLR `bench_cudagraph(rep=2)` on the same FlyDSL 0.4.2 environment. This mode reads 1 GiB before **every** call and uses Kineto kernel durations excluding the flush. Unlike the no-profiler streaming proxy above, these timings are instrumented. All candidate arms passed the same 1e-2 absolute/relative reference check; the pre-existing AITER split/slice-K dispatch arm was checked with its BF16-intermediate tolerance.

| M,N,K | Current AITER dispatch (us) | Best tested candidate (us) | NT / stagger | AKLR (us) | Candidate / AKLR |
|---|---:|---:|---|---:|---:|
|16,18432,1536|12.124|11.428|nt1_stg3|10.117|1.130x|
|32,18432,1536|13.113|12.137|nt1_stg3|10.231|1.186x|
|64,18432,1536|14.921|13.519|nt1_stg3|11.619|1.164x|
|112,18432,1536|18.353|19.803|nt1_stg3|13.697|1.446x|
|16,20480,7168|48.827|44.217|nt1_stg0|43.515|1.016x|
|32,20480,7168|47.495|46.107|nt1_stg0|44.355|1.040x|
|64,20480,7168|60.155|51.017|nt1_stg0|49.065|1.040x|
|112,20480,7168|76.117|76.611|nt1_stg3|55.272|1.386x|

The M=112 candidates do not close the gap and can lose against the current dispatch. They are benchmark counterexamples, **not** replacement tuned rows. No claim of full gap closure or automatic application speedup is made.

The existing `python op_tests/test_flydsl_hgemm.py` regression also completed, including 1000 concurrent-copy launches with zero bad launches, skinny rejection tests and the >2-GiB skinny weight case. The four existing config-lookup unit tests passed.
