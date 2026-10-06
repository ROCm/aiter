# GLM-5.3 IQ2R MoE on gfx950

`aiter/iq2r_glm53.py` runs the GLM-5.3 routed experts (256 routed + the
shared expert fused as expert 256, top-9, hidden 6144) from 2-bit IQ2R
weights on MI355X, at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256).

IQ2R was developed by the MK1 team at AMD. Its learned codebooks are seeded
from the IQ2_XXS grid of ggml/llama.cpp (MIT).

## Layout

`iq2r_glm53_pack` converts the generic per-expert IQ2R records into one
packed stack per layer: the gate/up rows are interleaved in quads and the
codebook signs are folded into the records. The packed stack is the only
copy kept on the GPU. A checkpoint stored in this layout can be sliced to
any supported TP rank with contiguous slices (`iq2r_glm53_slice_gate`, and
the generic `iq2r_slice_*` helpers for down and aux).

## Building the checkpoint

Two tools turn the block-FP8 GLM-5.3 checkpoint into the packed checkpoint
ATOM serves.

1. `aiter.iq2r_glm5_calibrate` runs the FP8 model in Hugging Face
   Transformers (5.16 or newer, plus `kernels` for the FP8 matmul) over a
   text corpus, prefill only. For every routed and shared expert projection
   it records the importance of each input channel k,
   `sum_t r_t^2 x_t[k]^2 / sum_t r_t^2`, where x_t is the projection input
   and r_t the token's routing weight (1 for the shared expert). The
   published checkpoint used 32,768 tokens of the WikiText-2 train split:

       python3 - <<'EOF'
       import json
       from datasets import load_dataset
       rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
       texts = [row["text"].strip() for row in rows if row["text"].strip()]
       with open("wikitext2-train.jsonl", "w") as f:
           f.writelines(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts)
       EOF
       # 23,767 rows, sha256 df1912f6b3c51b485ff7181cdde30f9ccd303c02031fea262179071ee1cbc1d4
       python3 -m aiter.iq2r_glm5_calibrate --model-dir $FP8 \
           --text-file wikitext2-train.jsonl --output calibration.pt \
           --device-map balanced

   The defaults (64 sequences of 512 tokens) are that recipe. Spread over 4
   MI355X GPUs, loading takes about 30 minutes and the capture about 10. An
   expert that no corpus token reaches gets one forced pass over its layer's
   inputs, which the artifact records; any still unobserved is an error. On
   a host without Hugging Face Hub access, `--fp8-kernel-dir` loads a local
   build of `kernels-community/finegrained-fp8`.
2. `aiter.iq2r_glm5_compile` writes the packed checkpoint
   (`iq2r_layout` = `glm53-packed-v1`). For each MoE layer (3-77) it encodes
   the 256 routed experts and the shared expert, fused as expert 256, on the
   GPU, weighting every error term by the calibrated importance. It packs the
   layer with `iq2r_glm53_pack` and writes
   `iq2r-layer-NNNN-{gate-up,down}.safetensors`. The layers can be split
   across GPUs:

       SLICES=(3-12 13-22 23-32 33-42 43-51 52-60 61-69 70-77)
       for i in 0 1 2 3 4 5 6 7; do
         python3 -m aiter.iq2r_glm5_compile --model-dir $FP8 \
             --output-dir $OUT --calibration-cache calibration.pt \
             --device cuda:$i --layers ${SLICES[$i]} --resume &
       done; wait
       python3 -m aiter.iq2r_glm5_compile --model-dir $FP8 \
           --output-dir $OUT --calibration-cache calibration.pt --resume

   The last run covers every layer. It validates the existing layer files
   against the calibration file instead of re-encoding them, copies the
   remaining FP8 tensors into new shards and writes the index, config and
   tokenizer. The config lists one `iq2r_modules` pattern per compiled layer,
   so the experts of the MTP layer (78) stay FP8.

The result is about 235 GB and loads at any supported TP width with no
load-time relayout.

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

ATOM serving on MI355X, `benchmark_serving`, OSL 1024, 10 × C prompts at
concurrency C, FP8 KV cache, CUDA graphs, no IQ2R environment variables.
Both arms quantize the non-expert layers online to PTPC FP8. MXFP4 is
stock ATOM with the routed experts quantized online to MXFP4. IQ2R loads
the packed IQ2R experts, which take 58.6 GB of weights per GPU at TP4 and
32.7 GB at TP8. Output tokens/s.

The MXFP4 runs used an older upstream ATOM base and were not repeated, so
part of each difference may come from upstream changes.

TP4, ISL 1024:

| C | IQ2R | TPOT ms | MXFP4 | IQ2R vs MXFP4 |
|---|---|---|---|---|
| 1 | 92.0 | 10.79 | 82.6 | +11.4% |
| 2 | 176.0 | 11.01 | 164.1 | +7.3% |
| 4 | 352.5 | 11.10 | 318.1 | +10.8% |
| 8 | 621.3 | 12.47 | 565.0 | +10.0% |
| 16 | 1106.5 | 14.04 | 943.9 | +17.2% |
| 32 | 1741.4 | 17.55 | 1441.5 | +20.8% |
| 64 | 2650.0 | 23.25 | 2219.6 | +19.4% |
| 128 | 3874.3 | 31.77 | 3321.6 | +16.6% |
| 256 | 5143.5 | 47.94 | 4732.0 | +8.7% |

TP8, ISL 1024:

| C | IQ2R | TPOT ms | MXFP4 | IQ2R vs MXFP4 |
|---|---|---|---|---|
| 1 | 90.2 | 11.00 | 79.5 | +13.5% |
| 2 | 173.6 | 11.18 | 151.8 | +14.4% |
| 4 | 353.5 | 11.07 | 325.4 | +8.6% |
| 8 | 649.6 | 11.86 | 592.7 | +9.6% |
| 16 | 1200.6 | 12.93 | 1080.4 | +11.1% |
| 32 | 1983.6 | 15.29 | 1683.1 | +17.9% |
| 64 | 3162.4 | 19.45 | 2640.9 | +19.7% |
| 128 | 4790.6 | 25.66 | 4089.4 | +17.1% |
| 256 | 6637.5 | 37.13 | 5808.4 | +14.3% |

TP4, ISL 8192, measured before the current decode configs (the ISL 1024
IQ2R numbers rose 3-10% with them):

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

The tuner times each candidate by CUDA graph replay, as serving runs it;
eager timing adds host overhead that hides the best small-M launches. The
decode kernels accept up to 1024 tokens so the CSV can choose them for MTP
verify batches, which are (1 + speculative tokens) x concurrency.

## Tests

`op_tests/test_iq2r_glm53.py` checks the packed path against a dense Torch
reference (device-materialized weights, MXFP8 activations, BF16 SiLU·up) for
M = 1..3000 at both TP widths, the chunked long prefill, that packing commutes
with TP slicing, and the tuned CSV lookup. `op_tests/test_iq2r_hip.py` checks
the device encoder and materializer bit-exactly against the host reference.
`op_tests/test_iq2r_glm5_calibrate.py` and `test_iq2r_glm5_compile.py`
cover the two checkpoint tools. The calibration test runs the capture on a
tiny random GLM MoE DSA model on the CPU and skips without Transformers. The
compile test checks the written shards, index and config, and runs a split
build with random records in place of the encoder.
