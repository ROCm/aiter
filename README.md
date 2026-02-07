# aiter
![image](https://github.com/user-attachments/assets/9457804f-77cd-44b0-a088-992e4b9971c6)

AITER is AMD's centralized repository that support various of high performance AI operators for AI workloads acceleration, where a good unified place for all the customer operator-level requests, which can match different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.

Some summary of the features:
* C++ level API
* Python level API
* The underneath kernel could come from triton/ck/asm
* Not just inference kernels, but also training kernels and GEMM+communication kernels—allowing for workarounds in any kernel-framework combination for any architecture limitation.

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
| Communication (AllReduce) | GPU-initiated reduce-scatter and all-gather via [Iris](https://github.com/ROCm/iris) | [Triton Comms](docs/triton_comms.md) |

Each guide covers available variants, backend support (ASM / CK / Triton), Python API examples, and performance tuning advice.

Run operator tests with: `python3 op_tests/<test_file>.py` (e.g. `python3 op_tests/test_pa.py`)

## Additional Resources
- [Autotuning Pipeline](docs/autotuning_pipeline.md) — CSV-based kernel selection and tuning workflow
- [Container Setup (Non-root)](docs/aiter_container_nonroot_setup.md) — Running AITER in Docker without root
