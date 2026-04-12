Supported Models
================

AITER provides optimized kernels used by inference engines (ATOM, vLLM, SGLang)
for a variety of model architectures. The table below lists validated model
families and the key AITER operations they use.

Model Matrix
------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Model Family
     - Architecture
     - Key AITER Ops
     - Tuned GEMM Configs
   * - DeepSeek-V3/R1
     - MoE (MLA + FusedMoE)
     - MLA decode/prefill, fused_moe, paged_attention
     - ``dsv3_bf16_tuned_gemm.csv``
   * - Kimi K2.5
     - MoE
     - fused_moe, paged_attention
     - ``kimik2_bf16_tuned_gemm.csv``
   * - GLM-5
     - Dense
     - GEMM (FP8 blockscale), paged_attention
     - ``glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv``
   * - GPT-OSS 120B
     - MoE
     - fused_moe, paged_attention
     - ``gptoss_bf16_tuned_gemm.csv``
   * - Qwen3/3.5
     - MoE
     - fused_moe, paged_attention
     - ``a8w8_blockscale_tuned_gemm_qwen3_5_397b_a13b.csv``
   * - MiniMax-M2.5
     - MoE
     - fused_moe, paged_attention
     - (uses default configs)
   * - Llama 3.x
     - Dense
     - GEMM, flash attention, paged_attention
     - (uses default configs)

.. note::

   Tuned GEMM configs are stored in ``aiter/configs/model_configs/``. To tune
   GEMM for a new model, see the :doc:`gemm_tuning` guide.
