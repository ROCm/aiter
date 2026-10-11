# GLM-5.3 IQ2R MoE on gfx950

`aiter/iq2r_glm53.py` runs the GLM-5.3 routed experts (256 routed + the
shared expert fused as expert 256, top-9, hidden 6144) from 2-bit IQ2R
weights on MI355X, at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256).

IQ2R was developed by the MK1 team at AMD. Its learned codebooks are seeded
from the IQ2_XXS grid of ggml/llama.cpp (MIT).

## Layout

The compiler interleaves the gate and up rows of each expert
(gate0, up0, gate1, up1, ...) before encoding. `iq2r_glm53_pack` then turns
the per-expert IQ2R records into one packed stack per layer: it folds the
codebook signs into the records and regroups the gate/up records into quads
of four 16-column blocks. The packed stack is the only copy kept on the
GPU. A checkpoint stored in this layout can be sliced to any supported TP
rank with contiguous slices (`iq2r_glm53_slice_gate`, and the generic
`iq2r_slice_*` helpers for down and aux).

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

   The last run covers every layer. Instead of re-encoding the existing layer
   files it checks their tensor keys, shapes, dtypes, format metadata and
   quality string. It then copies the remaining FP8 tensors into new shards
   and writes the index, config and tokenizer. The config lists
   `iq2r_modules` patterns for the routed and shared experts of each compiled
   layer, so the experts of the MTP layer (78) stay FP8.

The result is about 235 GB and loads at any supported TP width with no
load-time relayout.

## Dispatch

`iq2r_glm53_moe_out` looks up the launch choice for the token count in
`aiter/configs/iq2r_glm53_tuned.csv` (keyed on gfx, cu_num, token and
inter_dim). Token counts without a row use `_default_config`.
`AITER_LOG_TUNED_CONFIG=1` logs the choice once per shape. Each row names:

- the gate kernel (`decode`, `nobarrier`, `prefill`) and its grid;
- the down kernel (`route9`, `single`, `packed`, `ordered`, `prefill`) and its grid;
- the number of prefill down chunks.

`AITER_CONFIG_IQ2R_GLM53` points the lookup at a different CSV. To tune
new shapes, add them to `aiter/configs/iq2r_glm53_untuned.csv` and run
`python3 csrc/kernels/iq2r/iq2r_glm53_tune.py`, which writes
`iq2r_glm53_tuned.csv`. The tuner times each candidate by CUDA graph
replay, as serving runs it; eager timing adds host overhead that hides the
best small-M launches. The decode kernels accept up to 1024 tokens so the
CSV can choose them for MTP verify batches, which are
(1 + speculative tokens) x concurrency.

## Results

ATOM serving on MI355X, `benchmark_serving`, ISL 1024, OSL 1024, 10 × C
prompts at concurrency C = 1..256, FP8 KV cache, CUDA graphs, no IQ2R
environment variables. Both arms quantize the non-expert layers online to
PTPC FP8. MXFP4 is stock ATOM with the routed experts quantized online to
MXFP4. IQ2R loads the packed IQ2R experts, about 52.0 GB per GPU at TP4 and
26.1 GB at TP8.

IQ2R output throughput is 7-21% above MXFP4 at TP4 (C = 1: 92.0 vs
82.6 tokens/s; C = 256: 5143.5 vs 4732.0) and 9-20% above at TP8 (C = 1:
90.2 vs 79.5; C = 256: 6637.5 vs 5808.4). The MXFP4 runs used an older
upstream ATOM base, so part of each difference may come from upstream
changes.

## Tests

`op_tests/test_iq2r_glm53.py` checks the packed MoE against a dense Torch
reference at both TP widths. The other `op_tests/test_iq2r_*.py` files cover
the format, the host and device encoders and the two checkpoint tools; the
calibration test skips without Transformers.
