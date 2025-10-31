# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# ========================================================================
# How to use AOT gluon kernel for pa_mqa_logits on lower triton version (below 3.4.0):
#   1. Generate Gluon kernel based on rocm/triton/gluon_ext (3.5.0+gite392a058)
#      it requires zip installed.
#          $ cd ${AOT_DUMP_AITER_ROOT}
#          $ python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py --batch=1 -aot [-p]
#      "-p" means kernel could assume the stride of KVCache is aligned to 16B.
#      If enable it, the stride of KVCache in the AOT_load side must also be aligned to 16B.
#   2. Copy generated paged_mqa_logits_aot_kernel.zip to ${AOT_LOAD_AITER_ROOT}/aiter/ops/triton/configs
#      and unzip it.
#          $ cd ${AOT_LOAD_AITER_ROOT}
#          $ cd aiter/ops/triton/configs && unzip paged_mqa_logits_aot_kernel.zip && cd -
#   3. Set env variable to enable AOT gluon kernel loading
#          $ export AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1
#          $ python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py -kv_length=32768 --batch=2 -mtp=1 -p
#      Set AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=0 to disable AOT gluon kernel. It will backward
#      to triton JIT kernel
# ========================================================================

import os
import torch
import triton
from functools import lru_cache

from triton.backends.compiler import GPUTarget

enable_aot_gluon_pa_mqa_logits = os.environ.get(
    "AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS", "0"
)
enable_aot_gluon_pa_mqa_logits = enable_aot_gluon_pa_mqa_logits == "1"

if triton.__version__ >= "3.5.0":
    from triton.experimental.gluon._runtime import GluonASTSource as ASTSource
    from aiter.ops.triton._triton_kernels.pa_mqa_logits import (
        _deepgemm_fp8_paged_mqa_logits_stage1,
        _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
        _deepgemm_fp8_paged_mqa_logits,
    )
    from aiter.ops.triton.gluon.pa_mqa_logits import (
        _gluon_deepgemm_fp8_paged_mqa_logits,
    )

    enable_gluon_pa_mqa_logits = True
    enable_jit_gluon_pa_mqa_logits_kernel = True
else:
    from triton.compiler import ASTSource
    from aiter.ops.triton._triton_kernels.pa_mqa_logits import (
        _deepgemm_fp8_paged_mqa_logits_stage1,
        _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
        _deepgemm_fp8_paged_mqa_logits,
        _gluon_deepgemm_fp8_paged_mqa_logits,
    )

    assert triton.__version__ < "3.4.0"
    enable_gluon_pa_mqa_logits = enable_aot_gluon_pa_mqa_logits
    enable_jit_gluon_pa_mqa_logits_kernel = False


from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.utility.triton.triton_metadata_redirect import (
    AOTMetadataContext,
)


def deepgemm_fp8_paged_mqa_logits_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 64,
    SplitKV: int = 5,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fnuz)

    config = {
        "ChunkQ": heads,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }

    grid = (batch_size * next_n * config["SplitKV"],)
    _deepgemm_fp8_paged_mqa_logits_ragged_k[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fnuz)

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, 1, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkQ: int = 64,
    ChunkK: int = 256,
    TotalCuCount: int = 80,
    WavePerEU: int = 2,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_blk_len = kv_indices.size()

    TileQCount = batch_size * next_n * (heads // ChunkQ)
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fnuz)

    config = {
        "ChunkQ": ChunkQ,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * SplitKV),)
    _deepgemm_fp8_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_blk_len,
        waves_per_eu=WavePerEU,
        **config,
    )


@lru_cache(maxsize=None)
def _compile_deepgemm_fp8_paged_mqa_logits(
    ChunkQ,
    ChunkK,
    KVBlockSize,
    HiddenDim,
    is_padded_mode: bool,
    WavePerEU: int = 2,
):
    target = GPUTarget("hip", "gfx942", 64)

    fn_signature = {
        "batch_size": "i32",
        "next_n": "i32",
        "heads_num": "i32",
        "Q_buffer": "*fp8e4b8",
        "stride_q_batch": "i32",
        "stride_q_next_n": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": "*fp8e4b8",
        "stride_k_seq": "i32",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i32",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "SplitKV": "i32",
    }
    if not enable_jit_gluon_pa_mqa_logits_kernel:
        fn_signature["dummyPointerArg"] = "*i32"
    fn_signature["ChunkQ"] = "constexpr"
    fn_signature["ChunkK"] = "constexpr"
    fn_signature["KVBlockSize"] = "constexpr"
    fn_signature["HiddenDim"] = "constexpr"

    options = {
        "num_warps": 4,
        "waves_per_eu": WavePerEU,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": "gfx942",
        "backend_name": "hip",
        "warp_size": 64,
        "name": "_gluon_deepgemm_fp8_paged_mqa_logits",
    }

    kv_cache_attr = []
    if is_padded_mode:
        kv_cache_attr.append(["tt.divisibility", 16])

    src = ASTSource(
        fn=_gluon_deepgemm_fp8_paged_mqa_logits,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
        },
        attrs={
            (2,): [["tt.divisibility", 16]],  # heads_num
            (3,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q_buffer
            (4,): [["tt.divisibility", 16]],  # stride_q_batch
            (5,): [["tt.divisibility", 16]],  # stride_q_next_n
            (6,): [["tt.divisibility", 16]],  # stride_q_heads
            (7,): kv_cache_attr,  # KV_buffer
            (8,): kv_cache_attr,  # stride_k_seq
            (9,): kv_cache_attr,  # scale_buffer
            (10,): kv_cache_attr,  # stride_scale_seq
            (11,): [["tt.pointer_range", 32]],  # context_len_ptr
            (12,): [["tt.pointer_range", 32]],  # kv_indices
            (13,): [
                ["tt.divisibility", 16],
                ["tt.pointer_range", 32],
            ],  # weights
            (14,): [["tt.divisibility", 16]],  # stride_w_batch
            (15,): [["tt.pointer_range", 32]],  # OutLogits_buffer
        },
    )

    if enable_jit_gluon_pa_mqa_logits_kernel:
        kernel = triton.compile(
            src,
            target=target,
            options=options,
        )
    else:
        padded_str = "T" if is_padded_mode else "F"
        kernel_str = f"paged_mqa_logits_{ChunkQ}x{ChunkK}x{HiddenDim}_B{KVBlockSize}P{padded_str}W{WavePerEU}"
        metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/paged_mqa_logits/aot/{kernel_str}"
        with AOTMetadataContext(
            _gluon_deepgemm_fp8_paged_mqa_logits.fn.__name__,
            metadata_pth,
        ):
            kernel = triton.compile(
                src,
                target=target,
                options=options,
            )
    return kernel


def deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache,
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 1,
    ChunkK: int = 256,
    TotalCuCount: int = 80,
    WavePerEU: int = 2,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_block_len = kv_indices.size()

    TileQCount = batch_size * next_n
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    assert ChunkK % KVBlockSize == 0
    assert KVBlockSize == 1 and not Preshuffle

    kv_cache_fp8, kv_cache_scale = (
        kv_cache[..., :hidden_dim],
        kv_cache[..., hidden_dim:],
    )
    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fnuz)
    kv_cache_scale = kv_cache_scale.view(torch.float32)

    is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0
    kernel = _compile_deepgemm_fp8_paged_mqa_logits(
        ChunkQ=heads,
        ChunkK=ChunkK,
        KVBlockSize=KVBlockSize,
        HiddenDim=hidden_dim,
        is_padded_mode=is_padded_mode,
        WavePerEU=WavePerEU,
    )

    grid = (batch_size * next_n * SplitKV, 1, 1)
    if enable_gluon_pa_mqa_logits:
        if enable_jit_gluon_pa_mqa_logits_kernel:
            kernel[grid](
                batch_size,
                next_n,
                heads,
                q_fp8,
                q_fp8.stride(0),
                q_fp8.stride(1),
                q_fp8.stride(2),
                kv_cache_fp8,
                kv_cache_fp8.stride(0),
                kv_cache_scale,
                kv_cache_scale.stride(0),
                context_lens,
                kv_indices,
                weights,
                weights.stride(0),
                out_logits,
                out_logits.stride(0),
                max_model_len,
                max_block_len,
                SplitKV,
                # constexpr
                heads,
                ChunkK,
                KVBlockSize,
                hidden_dim,
            )
        else:  #  load AOT compiled gluon kernel
            kernel[grid](
                batch_size,
                next_n,
                heads,
                q_fp8,
                q_fp8.stride(0),
                q_fp8.stride(1),
                q_fp8.stride(2),
                kv_cache_fp8,
                kv_cache_fp8.stride(0),
                kv_cache_scale,
                kv_cache_scale.stride(0),
                context_lens,
                kv_indices,
                weights,
                weights.stride(0),
                out_logits,
                out_logits.stride(0),
                max_model_len,
                max_block_len,
                SplitKV,
                out_logits,  # dummyPointerArg for triton version < 3.4.0
                # constexpr
                heads,
                ChunkK,
                KVBlockSize,
                hidden_dim,
            )
    else:
        kernel = _deepgemm_fp8_paged_mqa_logits[grid](
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_fp8,
            kv_cache_fp8.stride(0),
            kv_cache_scale,
            kv_cache_scale.stride(0),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            waves_per_eu=WavePerEU,
            ChunkQ=heads,
            ChunkK=ChunkK,
            SplitKV=SplitKV,
            HiddenDim=hidden_dim,
        )
    return triton.runtime.cache.get_cache_manager(kernel.hash).key
