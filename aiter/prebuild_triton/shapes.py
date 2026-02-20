# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Model shape configurations for Triton kernel precompilation.

Each model config specifies the dimensions that determine which kernel
variants need to be compiled. The precompilation registry uses these
to generate all (constexpr, signature) combinations.
"""

DEEPSEEK_V3 = {
    "num_heads": 128,
    "num_kv_heads": 1,
    "head_dim": 128,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_experts": 256,
    "top_k": 8,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 129280,
    "batch_sizes": [1, 4, 16, 64],
    "seq_lens": [1, 128, 512, 2048, 8192],
    "block_size": 16,
}

LLAMA3_70B = {
    "num_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 128,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "vocab_size": 128256,
    "batch_sizes": [1, 4, 16, 64],
    "seq_lens": [1, 128, 512, 2048, 8192],
    "block_size": 16,
}

LLAMA3_8B = {
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "batch_sizes": [1, 4, 16, 64],
    "seq_lens": [1, 128, 512, 2048, 8192],
    "block_size": 16,
}

QWEN2_72B = {
    "num_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 128,
    "hidden_size": 8192,
    "intermediate_size": 24576,
    "vocab_size": 152064,
    "batch_sizes": [1, 4, 16, 64],
    "seq_lens": [1, 128, 512, 2048, 8192],
    "block_size": 16,
}

ALL_MODELS = {
    "deepseek_v3": DEEPSEEK_V3,
    "llama3_70b": LLAMA3_70B,
    "llama3_8b": LLAMA3_8B,
    "qwen2_72b": QWEN2_72B,
}

# Common hidden sizes across models (for normalization/activation kernels)
COMMON_HIDDEN_SIZES = sorted(
    set(m["hidden_size"] for m in ALL_MODELS.values())
    | set(m["intermediate_size"] for m in ALL_MODELS.values())
)

# Common head dimensions
COMMON_HEAD_DIMS = sorted(set(m["head_dim"] for m in ALL_MODELS.values()))

# Common block sizes for paged attention
COMMON_BLOCK_SIZES = [16, 32]
