# MiniMax-M3 monokernel: results log

All runs: MI355X GPUs 4-7, TP4, container ix_dsv41 (vllm/vllm-openai-rocm:nightly-rocm100-5555a2d8f, vLLM 0.31.1rc1.dev376),
checkpoint /data/models/MiniMax-M3-MXFP4, scripts in /home/vigokhal/m3_bench (serve.sh, bench_1k1k.sh).
Serving: block 128, FP8 KV, TRITON_ATTN, aiter MoE + fused shared expert, shuffled KV layout,
use_index_cache / index_topk_freq 4, no speculative decoding (EAGLE3 draft not on host), max-num-seqs 2.
Bench: vllm bench serve, random 1024 in / 1024 out, --ignore-eos, concurrency 1, 32 prompts, 2 warmups, seed 42.

| date | backend | TPOT mean (ms) | out tok/s | TTFT mean (ms) | notes |
|---|---|---|---|---|---|
| 2026-10-15 | vLLM default (no mono) | 5.21 | 188.71 | 95.0 | results/vllm_off_*.json |
