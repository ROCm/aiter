# aiter
![image](https://github.com/user-attachments/assets/9457804f-77cd-44b0-a088-992e4b9971c6)

AITER is AMD's centralized repository that support various of high performance AI operators for AI workloads acceleration, where a good unified place for all the customer operator-level requests, which can match different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.

Some summary of the features:
* C++ level API
* Python level API
* The underneath kernel could come from triton/ck/asm
* Not just inference kernels, but also training kernels and GEMM+communication kernels—allowing for workarounds in any kernel-framework combination for any architecture limitation.

## Installation

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
```

If you forgot `--recursive` during clone:
```bash
git submodule sync && git submodule update --init --recursive
```

### Development Mode (JIT)

Kernels are compiled on first use — fastest to get started:
```bash
python3 setup.py develop
```

### Install with Precompiled Kernels

Precompile kernels at install time so there is no JIT overhead at runtime:
```bash
PREBUILD_KERNELS=2 GPU_ARCHS="gfx942" python3 setup.py install
```

| Variable | Description |
|---|---|
| `GPU_ARCHS` | Target GPU architecture(s), semicolon-separated. Use `"native"` to auto-detect. Common values: `gfx942` (MI300X), `gfx950` (MI350X), `gfx90a` (MI250X). Multi-target example: `"gfx942;gfx950"` |
| `PREBUILD_KERNELS` | `0` — no precompilation (JIT only, default). `1` — precompile core kernels (excludes tuning and most MHA variants). `2` — precompile inference kernels (excludes backward and tuning). `3` — precompile MHA kernels only (minimal build). |
| `MAX_JOBS` | Max parallel compilation threads (auto-calculated from CPU cores and memory if not set) |

### Triton Communication Support (Optional)

For multi-GPU communication primitives (reduce-scatter, all-gather) using the [Iris](https://github.com/ROCm/iris) library:
```bash
pip install -e .
pip install -r requirements-triton-comms.txt
```
See the [Triton Comms Guide](docs/triton_comms.md) for usage details.

## Supported Operators

| **Operator** | **Description** | **Guide** |
|---|---|---|
| Attention (MHA, PA) | Multi-Head Attention, Paged Attention (decode & prefill), Unified Attention, chunked prefill, GQA/MQA | [Attention Guide](docs/attention_variants_guide.md) |
| MLA | Multi-head Latent Attention — standard decode, persistent decode, prefill, sparse MLA, fused ops | [MLA Guide](docs/mla_kernel_support_report.md) |
| Fused MOE | Mixture of Experts — A8W8, A16W8, FP8 block-scale, MXFP4, 2-stage MOE, topK routing | [MOE Guide](docs/moe_variants_guide.md) |
| GEMM | Matrix multiply (A8W8, A16W16, A4W4, batched), DeepGEMM, Triton FFN fusions, CSV-based tuning | [GEMM Guide](docs/gemm_variants_guide.md) |
| Quantization | BF16/FP16 to FP8/MXFP4/INT4, per-tensor/token/block strategies, fused quant ops, SmoothQuant | [Quantization Guide](docs/quantization_guide.md) |
| Normalization (RMSNorm, LayerNorm) | RMSNorm, LayerNorm, GroupNorm — fused add/quant variants, SmoothQuant, distributed fusion | [Normalization Guide](docs/normalization_guide.md) |
| RoPE | Rotary Position Embedding — SBHD/THD/2D/3D formats, NeoX & GPT-J styles, scaling methods | [RoPE Guide](docs/rope_guide.md) |
| KV-Cache | Paged/flash/MLA cache layouts, quantized cache (FP8/INT8), fused RoPE + cache write | [KV-Cache Guide](docs/kv_cache_guide.md) |
| Elementwise & Activations | SiLU/GELU/sigmoid/tanh, SwiGLU gates, fused activation + quantize, binary arithmetic (+−×÷) | [Elementwise Guide](docs/elementwise_activation_guide.md) |
| Sampling | Greedy, random, mixed, top-k, top-p token sampling for LLM generation | [Sampling Guide](docs/sampling_guide.md) |
| Top-K | Top-k selection — MOE routing (grouped, biased), radix/bitonic sort, fused softmax+topk | [Top-K Guide](docs/topk_guide.md) |
| Communication (AllReduce) | Custom all-reduce, quick all-reduce, Iris reduce-scatter/all-gather | [Distributed Guide](docs/distributed_guide.md) |
| Causal Conv1D | Causal convolution for Mamba/SSM models — prefill, decode, fused QKV split, speculative decoding | [Causal Conv1D Guide](docs/causal_conv1d_guide.md) |
| Gated Delta Net | Gated delta rule recurrence — fused recurrent, chunk-based, sigmoid gating, GVA support | [GDN Guide](docs/gated_delta_net_guide.md) |
| Grouped GEMM | GMM (Triton) and DeepGEMM (CK) — MoE expert routing, variable-length grouped GEMM | [Grouped GEMM Guide](docs/grouped_gemm_guide.md) |

Each guide covers available variants, backend support (ASM / CK / Triton), Python API examples, and performance tuning advice.

Run operator tests with: `python3 op_tests/<test_file>.py` (e.g. `python3 op_tests/test_pa.py`)

## Infrastructure & Tuning

| Guide | What's Inside |
|-------|--------------|
| [JIT Compilation System](docs/jit_system_guide.md) | `@compile_ops` decorator, module config, build flow, cache, GPU detection |
| [GEMM Tuning & Gradlib](docs/gemm_tuning_guide.md) | CSV-based kernel dispatch, hipBLASLt/ASM tuning, gradlib framework |
| [Distributed Infrastructure](docs/distributed_guide.md) | Tensor parallelism, custom/quick all-reduce, Iris comms, shared memory broadcast |
| [Weight Shuffle & Preshuffle](docs/weight_shuffle_guide.md) | Weight layout transforms for CK/ASM/Triton GEMM, FP8/FP4 preshuffle |
| [BERT Padding & Variable-Length](docs/bert_padding_guide.md) | Pad/unpad utilities, variable-length attention, cumulative sequence lengths |

## Additional Resources
- [Triton-based Communication (Iris)](docs/triton_comms.md) — GPU-initiated reduce-scatter and all-gather
- [Autotuning Pipeline](docs/autotuning_pipeline.md) — CSV-based kernel selection and tuning workflow
- [Container Setup (Non-root)](docs/aiter_container_nonroot_setup.md) — Running AITER in Docker without root
