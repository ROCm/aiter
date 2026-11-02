# Integrated C8 full-layer monokernel — 2026-10-21

Flash single GPU (TP1, no EP). One fused attention+MoE kernel per eligible decode layer; all43layers. Prefill stays native.

## End-to-end performance

1k input/1k output, concurrency8,64requests. Server started once per configuration; client run twice; only second result below. Median TTFT/TPOT. Total tok/s includes input+output.

|Configuration|TTFT ms|TPOT ms|Total tok/s|Output tok/s|
|---|---:|---:|---:|---:|
|native|288.871|13.681|1146.987|573.494|
|previous|313.123|15.782|996.336|498.168|
|optimized|319.521|12.301|1271.711|635.855|

## Integrated changes

- local QKV input quant and early bounded QB prefetch; split16 retains B3
- direct paged SWA/HCA/CSA attention; row-interleaved8splits; no compact-cache staging
- native-derived fused attention mHCpost+FFNpre/RMSNorm with packed weights, explicit FMAs, dedicated64producer barrier, warp specialization, adapted coefficient/residual layouts
- parallel routing metadata
- routedMoE BN256 BK512 two buffers, active-expert scheduling and coherent polling
- HCA four-way pooling plus compressor subgroup barrier for ratio4

## Integration adjustments

- HCA compressor subgroup barriers timed out in full-kernel integration; use all256 barriers before/after parallel pooling. Preserve ratio4 subgroup and mHC64producer barrier.
- Preserve negative BatchIds on padding, destination bounds, 8-field production plans, FP32 cross-layer residual statistic inputs.

## Validation

Resources: every reported serving layer/bucket has zero spills; inspect compiled table in RESULTS.json. Synthetic stage checks, padded requests, and320 full-layer graph replays each for HCA/CSA are saved alongside. Recurrent state exact in tested padded cases. HCA compressed cache can differ by up to0.00048828125 (relative RMS1.93e-5) due to pooling reduction order.

GSM8K32 examples,3-shot,max512 output:
- native: {"gsm8k": {"name": "gsm8k", "alias": "gsm8k", "sample_len": 32, "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.03125, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.03125}}
- previous: {"gsm8k": {"name": "gsm8k", "alias": "gsm8k", "sample_len": 32, "exact_match,strict-match": 0.9375, "exact_match_stderr,strict-match": 0.04347552147751577, "exact_match,flexible-extract": 0.9375, "exact_match_stderr,flexible-extract": 0.04347552147751577}}
- optimized: {"gsm8k": {"name": "gsm8k", "alias": "gsm8k", "sample_len": 32, "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.03125, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.03125}}

## Limitations

- Attention split reduction and HCA pooling order differ from previous/native; not bitwise equivalent overall.
- Synthetic full-layer tests can amplify small attention differences through quantization/routing; saved errors retained, no blanket exactness claim.
- GSM8K is a32-example3shot smoke test, max512 output tokens, not full accuracy qualification.
- No ProTP4 run or image/branch push in this integration turn.

## Reproduce

`install.py` updates only the existing ATOM adapter directory selection; default remains /work/batch-full. `run.sh fused` selects /work/integrated-c8 through ATOM_DSV4_BATCH_FULL_DIR. `run.sh baseline` disables fusion via the existing baseline script. Set HIP_VISIBLE_DEVICES, HTTP_PORT, PORT, CONCURRENCY=8, SERVER_LOG explicitly. Model /data/DeepSeek-V4-Flash; host directory mounted /work. Run `/work/concurrency-rollout/clients.sh TAG HTTP_PORT 8` after health ready; it runs both clients and saves both.

Do not rerun integrate.py or prepare_tests.py blindly: they are initial generation records and precede the HCA full-grid synchronization fix. Final executable source is model_layer_batch.py, runtime_batch.py, mhc_integrated.py, and imported modules. Optional ATOM_DSV4_OPT_MHC/OPT_COMP switches retain diagnostic comparisons. All enabled for the optimized result.

The archive contains selected source dependencies, fixture, code patches, logs, compiled ISA, raw client outputs, and accuracy records. Matching ROCm/Triton/AITER image and the model under /data are still required.
