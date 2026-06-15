# MoE kernel sources

Authoritative FlyDSL kernel implementation for the a4w4 two-stage MoE path.
Paths are repo-relative. These files are the source of truth; this example does
not duplicate them.

| Path | Role | Entry symbol |
| ---- | ---- | ------------ |
| `aiter/ops/flydsl/moe_kernels.py` | Public wrappers: build sorted-dispatch args, compile + launch stage1/stage2, glue the stages. | `flydsl_moe_stage1`, `flydsl_moe_stage2` |
| `aiter/ops/flydsl/moe_common.py` | Shared types for the stage1 gate/up computation strategy. | `GateMode` |
| `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` | Kernel bodies for the a4w4 (fp4 weight) path: stage1 gate/up GEMM with fused gated activation, and stage2 down GEMM. | `compile_mixed_moe_gemm1`, `compile_mixed_moe_gemm2` |

For the a4w4 path with bf16 output and no split-K (what this example runs), the
activation is fused inside `compile_mixed_moe_gemm1`. Two related files are NOT
on this exact path but belong to sibling MoE variants:

- `aiter/ops/flydsl/kernels/silu_and_mul_fq.py` (`build_silu_and_mul_fq_module`)
  — fused activation + quant + scale-sort, used only for fp4/fp8-quantized
  stage1 output or split-K (`k_batch > 1`).
- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
  (`compile_moe_gemm1` / `compile_moe_gemm2`) — the int4-weight + bf16-activation
  (a16wi4) path, a different quantization scheme.
