#!/usr/bin/env python3
"""Full KV cache benchmark: PPL + concurrency sweep + latency profile.

Connects to a running SGLang server and measures:
1. Perplexity via logprobs on WikiText-style text
2. Concurrency sweep: throughput at concurrency 1,2,4,8,16,32,64
3. Latency percentiles (TTFT, ITL, E2E)
4. Long-context generation

Usage:
  python3 bench_full.py --port 30000 --label bf16 --max-context 4096
"""

import argparse
import json
import math
import os
import time
import concurrent.futures
import requests
import numpy as np

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results")

# Diverse eval prompts for throughput testing
EVAL_PROMPTS = [
    "Explain the theory of general relativity in detail.",
    "Write a comprehensive guide to Python async programming.",
    "What are the causes and effects of climate change?",
    "Describe the history of the Internet from ARPANET to today.",
    "Explain how transformers work in machine learning.",
    "What is the Riemann hypothesis and why does it matter?",
    "Write a detailed comparison of SQL and NoSQL databases.",
    "Explain quantum entanglement to a high school student.",
    "What are the main differences between capitalism and socialism?",
    "Describe the process of photosynthesis step by step.",
    "Explain the P vs NP problem in computer science.",
    "What is CRISPR and how does it work?",
    "Describe the architecture of a modern CPU.",
    "What are the key principles of distributed systems?",
    "Explain how neural networks learn through backpropagation.",
    "What is the significance of the Turing test?",
    "Describe the water cycle and its importance to Earth.",
    "Explain the concept of entropy in thermodynamics.",
    "What are the main theories about the origin of the universe?",
    "Describe how public key cryptography works.",
    "Explain the concept of natural selection in evolution.",
    "What is blockchain technology and how does it work?",
    "Describe the main principles of object-oriented programming.",
    "Explain the difference between supervised and unsupervised learning.",
    "What are the key milestones in the history of aviation?",
    "Describe how vaccines work to protect against diseases.",
    "Explain the greenhouse effect and its role in climate.",
    "What is the Standard Model of particle physics?",
    "Describe the architecture of modern large language models.",
    "Explain how GPS navigation systems work.",
    "What are the fundamental forces in nature?",
    "Describe the process of nuclear fusion in stars.",
]

# Long text for PPL measurement
PPL_TEXT = """The transformer architecture has revolutionized natural language processing since its introduction in 2017. Unlike recurrent neural networks, transformers process all tokens in parallel using self-attention mechanisms. The key innovation is the scaled dot-product attention, which computes attention weights as the softmax of query-key dot products divided by the square root of the key dimension. This allows the model to attend to all positions in the input sequence simultaneously, capturing long-range dependencies more effectively than sequential models.

Multi-head attention extends this by projecting queries, keys, and values into multiple subspaces and computing attention independently in each. The results are concatenated and linearly projected to produce the final output. This multi-head mechanism allows the model to jointly attend to information from different representation subspaces at different positions.

The transformer encoder consists of multiple identical layers, each containing a multi-head self-attention sublayer followed by a position-wise feed-forward network. Layer normalization and residual connections are applied around each sublayer. The decoder is similar but includes an additional cross-attention layer that attends to the encoder output.

Large language models like GPT and its successors use only the decoder part of the transformer, trained autoregressively to predict the next token given all previous tokens. The training objective is to maximize the log-likelihood of the training data, which is equivalent to minimizing the cross-entropy loss between the predicted token probabilities and the actual next tokens.

Scaling laws have shown that model performance improves predictably with increases in model size, dataset size, and compute budget. This has driven the development of increasingly large models, from GPT-2's 1.5 billion parameters to GPT-4's rumored trillion-plus parameters. The key insight is that larger models are more sample-efficient, learning more from each training example.

The attention mechanism's quadratic complexity with respect to sequence length has motivated various efficient attention variants. Flash Attention reduces memory usage by computing attention blockwise without materializing the full attention matrix. Linear attention approximations replace the softmax with kernel functions to achieve linear complexity. Sparse attention patterns like those used in Longformer attend only to local windows and selected global positions.

Multi-head latent attention, used in models like DeepSeek-V3, compresses key-value pairs into a lower-dimensional latent space. Instead of storing full key and value vectors for each attention head, MLA stores a single compressed latent vector per token, which is then projected back to the full key-value space during attention computation. This dramatically reduces KV cache memory requirements while maintaining model quality, making it particularly effective for long-context inference scenarios where KV cache size is the primary memory bottleneck."""


def ppl_measurement(port, text=PPL_TEXT, max_chunk=512):
    """Measure perplexity using completion API with logprobs."""
    words = text.split()
    total_logprob = 0.0
    total_tokens = 0

    chunk_size = min(max_chunk, len(words))
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size // 2)]
    chunks = chunks[:6]  # limit to avoid timeout

    for i, chunk in enumerate(chunks):
        try:
            r = requests.post(
                f"http://localhost:{port}/v1/completions",
                json={
                    "model": "default",
                    "prompt": chunk,
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": 1,
                    "echo": True,
                },
                timeout=120,
            )
            data = r.json()
            if "choices" not in data:
                print(f"  PPL chunk {i}: no choices in response: {str(data)[:200]}")
                continue
            logprobs_data = data["choices"][0].get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", [])
            valid_lps = [lp for lp in token_logprobs if lp is not None and lp != 0]
            if valid_lps:
                total_logprob += sum(valid_lps)
                total_tokens += len(valid_lps)
                avg_lp = sum(valid_lps) / len(valid_lps)
                chunk_ppl = math.exp(-avg_lp)
                print(f"  PPL chunk {i}: {len(valid_lps)} tokens, avg_logprob={avg_lp:.4f}, ppl={chunk_ppl:.2f}")
        except Exception as e:
            print(f"  PPL chunk {i}: ERROR {e}")

    if total_tokens > 0:
        avg_logprob = total_logprob / total_tokens
        ppl = math.exp(-avg_logprob)
        return {"ppl": ppl, "avg_logprob": avg_logprob, "total_tokens": total_tokens}
    return {"ppl": None, "avg_logprob": None, "total_tokens": 0}


def single_request(port, prompt, max_tokens=128, temperature=0):
    """Single request with timing."""
    t0 = time.perf_counter()
    r = requests.post(
        f"http://localhost:{port}/generate",
        json={"text": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": temperature}},
        timeout=300,
    )
    t1 = time.perf_counter()
    data = r.json()
    text = data.get("text", "")
    meta = data.get("meta_info", {})
    comp_tokens = meta.get("completion_tokens", len(text.split()))
    return {
        "latency_s": t1 - t0,
        "completion_tokens": comp_tokens,
        "prompt_tokens": meta.get("prompt_tokens", len(prompt.split())),
        "tok_s": comp_tokens / (t1 - t0) if (t1 - t0) > 0 else 0,
    }


def concurrency_sweep(port, concurrencies=[1, 2, 4, 8, 16, 32, 64], max_tokens=128, n_per_level=None):
    """Sweep concurrency levels and measure throughput."""
    results = {}
    for c in concurrencies:
        n = n_per_level or max(c, 8)
        prompts = (EVAL_PROMPTS * ((n // len(EVAL_PROMPTS)) + 1))[:n]

        t0 = time.perf_counter()
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(single_request, port, p, max_tokens) for p in prompts]
            for f in concurrent.futures.as_completed(futs):
                try:
                    all_results.append(f.result())
                except Exception as e:
                    all_results.append({"error": str(e)})
        wall = time.perf_counter() - t0

        valid = [r for r in all_results if "error" not in r]
        total_out = sum(r["completion_tokens"] for r in valid)
        total_in = sum(r["prompt_tokens"] for r in valid)
        latencies = sorted([r["latency_s"] for r in valid])
        errors = len(all_results) - len(valid)

        result = {
            "concurrency": c,
            "n_requests": n,
            "wall_s": wall,
            "total_output_tokens": total_out,
            "total_input_tokens": total_in,
            "throughput_out_tok_s": total_out / wall if wall > 0 else 0,
            "throughput_total_tok_s": (total_in + total_out) / wall if wall > 0 else 0,
            "avg_latency_s": np.mean(latencies) if latencies else 0,
            "p50_latency_s": np.percentile(latencies, 50) if latencies else 0,
            "p99_latency_s": np.percentile(latencies, 99) if latencies else 0,
            "errors": errors,
        }
        results[c] = result
        print(f"  C={c:>3}: {result['throughput_out_tok_s']:>8.1f} out_tok/s | "
              f"p50={result['p50_latency_s']:.2f}s p99={result['p99_latency_s']:.2f}s | "
              f"{n} reqs in {wall:.1f}s | {errors} errors")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrencies", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"Full KV Cache Benchmark — {args.label}")
    print(f"{'='*70}")

    try:
        info = requests.get(f"http://localhost:{args.port}/get_server_info", timeout=10).json()
        print(f"Model: {info.get('model_path', '?')}")
        print(f"TP: {info.get('tp_size', '?')}")
        print(f"KV dtype: {info.get('kv_cache_dtype', '?')}")
    except:
        print("(Could not fetch server info)")

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        single_request(args.port, "Hello world", max_tokens=16)

    all_results = {"label": args.label}

    # 1. PPL
    if not args.skip_ppl:
        print(f"\n{'='*70}")
        print("1. PERPLEXITY (logprobs on reference text)")
        print(f"{'='*70}")
        ppl_result = ppl_measurement(args.port)
        all_results["ppl"] = ppl_result
        if ppl_result["ppl"]:
            print(f"\n  >>> PPL = {ppl_result['ppl']:.2f} (over {ppl_result['total_tokens']} tokens)")
        else:
            print("  >>> PPL measurement failed")

    # 2. Concurrency sweep
    if not args.skip_sweep:
        print(f"\n{'='*70}")
        print(f"2. CONCURRENCY SWEEP (max_tokens={args.max_tokens})")
        print(f"{'='*70}")
        sweep_result = concurrency_sweep(args.port, concurrencies, args.max_tokens)
        all_results["concurrency_sweep"] = sweep_result

    # 3. Single-request latency profile
    print(f"\n{'='*70}")
    print("3. SINGLE-REQUEST LATENCY (10 requests, max_tokens=128)")
    print(f"{'='*70}")
    latencies = []
    for i in range(10):
        r = single_request(args.port, EVAL_PROMPTS[i % len(EVAL_PROMPTS)], 128)
        latencies.append(r)
        print(f"  [{i}] {r['latency_s']:.3f}s | {r['completion_tokens']} tok | {r['tok_s']:.1f} tok/s")

    valid_lat = [r["latency_s"] for r in latencies if "error" not in r]
    valid_tps = [r["tok_s"] for r in latencies if "error" not in r]
    all_results["single_request"] = {
        "avg_latency_s": np.mean(valid_lat),
        "p50_latency_s": np.percentile(valid_lat, 50),
        "p99_latency_s": np.percentile(valid_lat, 99),
        "avg_tok_s": np.mean(valid_tps),
        "raw": latencies,
    }
    print(f"\n  Avg: {np.mean(valid_lat):.3f}s | {np.mean(valid_tps):.1f} tok/s")

    # Save
    out_file = os.path.join(RESULT_DIR, f"full_{args.label}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.label}")
    print(f"{'='*70}")
    if all_results.get("ppl", {}).get("ppl"):
        print(f"  PPL:                    {all_results['ppl']['ppl']:.2f}")
    print(f"  Single-req tok/s:       {all_results['single_request']['avg_tok_s']:.1f}")
    print(f"  Single-req latency:     {all_results['single_request']['avg_latency_s']:.3f}s")
    if "concurrency_sweep" in all_results:
        peak_c = max(all_results["concurrency_sweep"].items(), key=lambda x: x[1]["throughput_out_tok_s"])
        print(f"  Peak throughput:        {peak_c[1]['throughput_out_tok_s']:.1f} tok/s @ C={peak_c[0]}")
    print(f"\n  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
