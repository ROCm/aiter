# aiter
![image](https://github.com/user-attachments/assets/9457804f-77cd-44b0-a088-992e4b9971c6)

AITER is AMD's centralized repository that support various of high performance AI operators for AI workloads acceleration, where a good unified place for all the customer operator-level requests, which can match different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.

Some summary of the features:
* C++ level API
* Python level API
* The underneath kernel could come from triton/ck/asm
* Not just inference kernels, but also training kernels and GEMM+communication kernels—allowing for workarounds in any kernel-framework combination for any architecture limitation.

## Documentation

Each guide covers available variants, backend support (ASM / CK / Triton), Python API examples, and performance tuning advice.

| Guide | What's Inside |
|-------|--------------|
| [Attention Variants](docs/attention_variants_guide.md) | MHA, Paged Attention (decode & prefill), Unified Attention, chunked prefill, GQA/MQA support |
| [MLA Variants](docs/mla_kernel_support_report.md) | Multi-head Latent Attention — standard decode, persistent decode, prefill, sparse MLA, fused operations |
| [Fused MOE Variants](docs/moe_variants_guide.md) | Mixture of Experts — A8W8, A16W8, FP8 block-scale, MXFP4, 2-stage MOE, topK routing |
| [GEMM Variants & Tuning](docs/gemm_variants_guide.md) | A8W8, A16W16, A4W4, batched GEMM, DeepGEMM, Triton FFN fusions, CSV-based tuning system |
| [Quantization & Precision](docs/quantization_guide.md) | QuantType strategies (per-tensor/token/block), fused quant ops, FP8/MXFP4/INT4, SmoothQuant |
| [Normalization](docs/normalization_guide.md) | RMSNorm, LayerNorm, GroupNorm — fused add/quant variants, SmoothQuant, distributed fusion |
| [RoPE (Rotary Embedding)](docs/rope_guide.md) | SBHD/THD/2D/3D formats, NeoX & GPT-J styles, scaling methods, fused QK norm + RoPE |
| [KV-Cache Management](docs/kv_cache_guide.md) | Paged/flash/MLA layouts, quantized cache (FP8/INT8), fused RoPE + cache write |
| [Elementwise & Activations](docs/elementwise_activation_guide.md) | SiLU/GELU/sigmoid/tanh, SwiGLU gates, fused activation + quantize, binary arithmetic |

Additional resources:
- [Triton-based Communication (Iris)](docs/triton_comms.md) — GPU-initiated reduce-scatter and all-gather via [Iris](https://github.com/ROCm/iris)
- [Autotuning Pipeline](docs/autotuning_pipeline.md) — CSV-based kernel selection and tuning workflow
- [Container Setup (Non-root)](docs/aiter_container_nonroot_setup.md) — Running AITER in Docker without root

## Installation
```
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

If you happen to forget the `--recursive` during `clone`, you can use the following command after `cd aiter`
```
git submodule sync && git submodule update --init --recursive
```

To install with Triton communication support:
```bash
pip install -e .
pip install -r requirements-triton-comms.txt
```

## Supported Operators

Run any operator test with: `python3 op_tests/test_layernorm2d.py`

| **Operator** | **Description** | **Guide** |
|---|---|---|
| MHA | Multi-Head Attention | [Attention Guide](docs/attention_variants_guide.md) |
| PA | Paged Attention (decode & prefill) | [Attention Guide](docs/attention_variants_guide.md) |
| MLA | Multi-head Latent Attention | [MLA Guide](docs/mla_kernel_support_report.md) |
| FusedMOE | Mixture of Experts | [MOE Guide](docs/moe_variants_guide.md) |
| GEMM | Matrix multiply (A8W8, A16W16, A4W4, batched) | [GEMM Guide](docs/gemm_variants_guide.md) |
| QUANT | BF16/FP16 to FP8/MXFP4/INT4 quantization | [Quantization Guide](docs/quantization_guide.md) |
| RMSNORM | Root Mean Square Normalization | [Normalization Guide](docs/normalization_guide.md) |
| LAYERNORM | Layer Normalization | [Normalization Guide](docs/normalization_guide.md) |
| ROPE | Rotary Position Embedding | [RoPE Guide](docs/rope_guide.md) |
| KVCACHE | KV-Cache management | [KV-Cache Guide](docs/kv_cache_guide.md) |
| AllREDUCE | Reduce + Broadcast | [Triton Comms](docs/triton_comms.md) |
| ELEMENT WISE | Element-wise ops: + - * / | [Elementwise Guide](docs/elementwise_activation_guide.md) |
| SIGMOID | Sigmoid activation | [Elementwise Guide](docs/elementwise_activation_guide.md) |
