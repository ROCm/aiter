# GLM-5.3 IQ2R MoE on gfx950

`aiter/iq2r_glm53.py` runs the GLM-5.3 routed experts (256 routed + the
shared expert fused as expert 256, top-9, hidden 6144) from 2-bit IQ2R
weights on MI355X, at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256).

## Layout

`iq2r_glm53_pack` converts the generic per-expert IQ2R records into one
packed stack per layer: the gate/up rows are interleaved in quads and the
codebook signs are folded into the records. The packed stack is the only
copy kept on the GPU. A checkpoint stored in this layout can be sliced to
any supported TP rank with contiguous slices (`iq2r_glm53_slice_gate`, and
the generic `iq2r_slice_*` helpers for down and aux).

## Dispatch

`iq2r_glm53_moe_out` looks up the launch choice for the token count in
`aiter/configs/iq2r_glm53_tuned.csv` (keyed on gfx, cu_num, token,
model_dim, inter_dim, expert, topk). Token counts without a row use
`_default_config`. `AITER_LOG_TUNED_CONFIG=1` logs every hit. Each row
names:

- the gate kernel (`decode`, `nobarrier`, `prefill`) and its grid;
- the down kernel (`route9`, `single`, `packed`, `ordered`, `prefill`) and its grid;
- the number of prefill down chunks.

`AITER_CONFIG_IQ2R_GLM53` points the lookup at a different CSV. To tune
new shapes, add them to `iq2r_glm53_untuned.csv` and run

    python3 csrc/kernels/iq2r/iq2r_glm53_tune.py \
        -i aiter/configs/iq2r_glm53_untuned.csv \
        -o aiter/configs/iq2r_glm53_tuned.csv

## Results

ATOM serving on MI355X, `benchmark_serving`, OSL 1024, 10 prompts per
concurrency, FP8 KV cache, CUDA graphs, no IQ2R environment variables.
Both arms quantize the non-expert layers online to PTPC FP8. MXFP4 is
stock ATOM with the routed experts quantized online to MXFP4. IQ2R loads
the packed IQ2R experts. Output tokens/s:

TP4, ISL 1024 (IQ2R peak weights 58.6 GB/GPU, 198.7k KV blocks):

| C | IQ2R | TPOT ms | MXFP4 | IQ2R vs MXFP4 |
|---|---|---|---|---|
| 1 | 87.8 | 11.30 | 82.6 | +6.3% |
| 2 | 166.1 | 11.64 | 164.1 | +1.3% |
| 4 | 331.5 | 11.80 | 318.1 | +4.2% |
| 8 | 569.8 | 13.59 | 565.0 | +0.9% |
| 16 | 1006.1 | 15.43 | 943.9 | +6.6% |
| 32 | 1599.3 | 19.08 | 1441.5 | +10.9% |
| 64 | 2472.6 | 24.85 | 2219.6 | +11.4% |
| 128 | 3656.0 | 33.58 | 3321.6 | +10.1% |
| 256 | 4990.5 | 49.23 | 4732.0 | +5.5% |

TP4, ISL 8192:

| C | IQ2R | TPOT ms | MXFP4 | IQ2R vs MXFP4 |
|---|---|---|---|---|
| 1 | 80.0 | 12.05 | 76.2 | +5.1% |
| 2 | 155.4 | 12.05 | 154.4 | +0.7% |
| 4 | 288.6 | 13.09 | 280.4 | +2.9% |
| 8 | 457.8 | 16.44 | 465.6 | -1.7% |
| 16 | 705.7 | 21.45 | 692.0 | +2.0% |
| 32 | 971.1 | 30.97 | 936.0 | +3.7% |
| 64 | 1248.2 | 48.50 | 1221.9 | +2.2% |
| 128 | 1514.2 | 80.03 | 1514.9 | -0.0% |
| 256 | 1718.6 | 141.16 | 1762.6 | -2.5% |

TP8, ISL 1024 (IQ2R peak weights 32.7 GB/GPU, 230.5k KV blocks):

| C | IQ2R | TPOT ms | MXFP4 | IQ2R vs MXFP4 |
|---|---|---|---|---|
| 1 | 84.8 | 11.71 | 79.5 | +6.6% |
| 2 | 162.4 | 11.94 | 151.8 | +7.0% |
| 4 | 329.2 | 11.90 | 325.4 | +1.1% |
| 8 | 603.0 | 12.75 | 592.7 | +1.7% |
| 16 | 1109.2 | 13.98 | 1080.4 | +2.7% |
| 32 | 1826.0 | 16.57 | 1683.1 | +8.5% |
| 64 | 2894.4 | 21.15 | 2640.9 | +9.6% |
| 128 | 4532.0 | 26.98 | 4089.4 | +10.8% |
| 256 | 6330.3 | 38.79 | 5808.4 | +9.0% |

The tuner times each candidate by CUDA graph replay, as serving runs it;
eager timing adds host overhead that hides the best small-M launches. The
decode kernels accept up to 1024 tokens so the CSV can choose them for MTP
verify batches, which are (1 + speculative tokens) x concurrency.

## Tests

`op_tests/test_iq2r_glm53.py` checks the packed path against the generic
`iq2r_fused_moe_out` for M = 1..3000 at both TP widths, the chunked long
prefill, that packing commutes with TP slicing, and the tuned CSV lookup.
