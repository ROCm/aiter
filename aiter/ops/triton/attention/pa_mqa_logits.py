# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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

import math
import os
from functools import cache

import torch
import triton
from packaging.version import Version
from triton.backends.compiler import GPUTarget

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.config_utils import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext

enable_aot_gluon_pa_mqa_logits = os.environ.get(
    "AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS", "0"
)
enable_aot_gluon_pa_mqa_logits = enable_aot_gluon_pa_mqa_logits == "1"
triton_version = Version(Version(triton.__version__).base_version)
_GLUON_PA_MQA_LOGITS_ARCHS = ("gfx942", "gfx950", "gfx1250")
if triton_version >= Version("3.5.0"):
    from triton.experimental.gluon._runtime import GluonASTSource as ASTSource

    from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits import (
        _deepgemm_fp8_paged_mqa_logits,
        _deepgemm_fp8_paged_mqa_logits_persistent_schedule,
        _deepgemm_fp8_paged_mqa_logits_ragged_k,
        _deepgemm_fp8_paged_mqa_logits_stage1,
        _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
        _deepgemm_fp8_paged_mqa_logits_varctx_schedule,
    )
    from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version
    from aiter.ops.triton.gluon.pa_mqa_logits import (
        _gluon_deepgemm_fp8_paged_mqa_logits,
        _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle,
        _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx,
    )

    enable_gluon_pa_mqa_logits = get_gfx() in _GLUON_PA_MQA_LOGITS_ARCHS
    enable_jit_gluon_pa_mqa_logits_kernel = not enable_aot_gluon_pa_mqa_logits
else:
    from triton.compiler import ASTSource

    from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits import (
        _deepgemm_fp8_paged_mqa_logits,
        _deepgemm_fp8_paged_mqa_logits_persistent_schedule,
        _deepgemm_fp8_paged_mqa_logits_ragged_k,
        _deepgemm_fp8_paged_mqa_logits_stage1,
        _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
        _deepgemm_fp8_paged_mqa_logits_varctx_schedule,
        _gluon_deepgemm_fp8_paged_mqa_logits,
        _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle,
        _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx,
    )

    enable_gluon_pa_mqa_logits = (
        enable_aot_gluon_pa_mqa_logits and get_gfx() in _GLUON_PA_MQA_LOGITS_ARCHS
    )
    enable_jit_gluon_pa_mqa_logits_kernel = False


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
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

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
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

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
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, block_size, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkQ: int = 64,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    num_blocks, block_size, num_kv_heads, packed_dim = kv_cache_fp8.size()
    _, max_blk_len = kv_indices.size()

    assert num_kv_heads == 1
    assert packed_dim == hidden_dim + 4, (
        "The stage1 kernel expects one fp32 scale after each packed FP8 token; "
        f"got q hidden_dim={hidden_dim} and packed KV dim={packed_dim}."
    )

    TileQCount = batch_size * next_n * (heads // ChunkQ)
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    packed_kv_cache = kv_cache_fp8.view(num_blocks, -1)
    value_elements = block_size * hidden_dim
    kv_cache_values = packed_kv_cache[:, :value_elements].view(
        num_blocks, block_size, hidden_dim
    )
    kv_cache_scale = packed_kv_cache[:, value_elements:].view(torch.float32)
    kv_cache_fp8 = kv_cache_values.view(dtypes.fp8)

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
        kv_cache_fp8.stride(1),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        kv_cache_scale.stride(1),
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
        KVBlockSize=block_size,
    )


@cache
def _compile_deepgemm_fp8_paged_mqa_logits(
    ChunkQ,
    ChunkK,
    Preshuffle,
    KVBlockSize,
    HiddenDim,
    is_padded_mode: bool,
    WavePerEU: int = 2,
    VarCtxOpt: bool = False,
    Persistent: bool = False,
    NextN: int = 1,
):
    gfx_version = get_gfx()
    assert gfx_version in _GLUON_PA_MQA_LOGITS_ARCHS
    is_gfx1250 = gfx_version == "gfx1250"
    if is_gfx1250:
        if Preshuffle:
            assert KVBlockSize > 1 and ChunkK % KVBlockSize == 0, (
                f"gfx1250 preshuffle (TDM block-load) requires KVBlockSize>1 "
                f"and ChunkK % KVBlockSize == 0 (ChunkK = N*KVBlockSize); got "
                f"KVBlockSize={KVBlockSize}, ChunkK={ChunkK}."
            )
        else:
            assert KVBlockSize == 1, (
                f"gfx1250 base kernel requires KVBlockSize==1; got "
                f"KVBlockSize={KVBlockSize}. Use Preshuffle=True for "
                f"KVBlockSize>1 (TDM block-load)."
            )
    cdna_version = get_cdna_version()
    warp_size = 32 if is_gfx1250 else 64
    target = GPUTarget("hip", gfx_version, warp_size)

    # gfx942 uses the AMD fnuz e4m3 variant (*fp8e4b8); gfx950 and gfx1250 use
    # the OCP/IEEE e4m3 variant (*fp8e4nv), matching utils.types.get_fp8_dtypes.
    gfx_fp8_pointer = "*fp8e4b8" if gfx_version == "gfx942" else "*fp8e4nv"

    fn_signature = {
        "batch_size": "i32",
        "next_n": "i32",
        "heads_num": "i32",
        "Q_buffer": gfx_fp8_pointer,
        "stride_q_batch": "i32",
        "stride_q_next_n": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": gfx_fp8_pointer,
        # The plain kernel forms per-token KV addresses from the page table, so
        # a cache past 2 GiB overflows a 32-bit stride product.
        "stride_k_seq": "i32" if Preshuffle else "i64",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32" if Preshuffle else "i64",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i64",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "num_block": "i32",
    }
    if VarCtxOpt:
        fn_signature["safe_chunks_per_cta_ptr"] = "*i32"
    else:
        fn_signature["SplitKV"] = "*i32" if Persistent else "i32"

    if triton_version < Version("3.4.0"):
        assert not enable_jit_gluon_pa_mqa_logits_kernel
        fn_signature["dummyPointerArg"] = "*i32"
    fn_signature["ChunkQ"] = "constexpr"
    fn_signature["ChunkK"] = "constexpr"
    fn_signature["KVBlockSize"] = "constexpr"
    fn_signature["HiddenDim"] = "constexpr"
    fn_signature["CDNA_VERSION"] = "constexpr"
    fn_signature["ARCH"] = "constexpr"
    persistent_constexpr = {}
    if triton_version >= Version("3.5.0") and not VarCtxOpt:
        fn_signature["PERSISTENT"] = "constexpr"
        fn_signature["PERSISTENT_NEXT_N"] = "constexpr"
        persistent_constexpr["PERSISTENT"] = Persistent
        persistent_constexpr["PERSISTENT_NEXT_N"] = NextN

    effective_wave_per_eu = 1 if is_gfx1250 and not Preshuffle else WavePerEU
    effective_num_warps = 1 if is_gfx1250 and Preshuffle else 4
    options = {
        "num_warps": effective_num_warps,
        "waves_per_eu": effective_wave_per_eu,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": gfx_version,
        "backend_name": "hip",
        "warp_size": warp_size,
        "name": (
            "_gluon_deepgemm_fp8_paged_mqa_logits"
            if not Preshuffle
            else (
                "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx"
                if VarCtxOpt
                else "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle"
            )
        ),
    }

    kv_cache_attr = []
    if is_padded_mode:
        kv_cache_attr.append(["tt.divisibility", 16])

    kernel_fn = (
        _gluon_deepgemm_fp8_paged_mqa_logits
        if not Preshuffle
        else (
            _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx
            if VarCtxOpt
            else _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle
        )
    )
    src = ASTSource(
        fn=kernel_fn,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
            "CDNA_VERSION": cdna_version,
            "ARCH": gfx_version,
            **persistent_constexpr,
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
            # OutLogits_buffer: NO tt.pointer_range 32 -- the output row base
            # offset (row * stride_out_batch) can exceed a 32-bit byte offset
            # for wide dense logits (e.g. max_model_len=1<<20). stride_out_batch
            # is i64 and the gluon kernel advances the base pointer in 64 bit
            # (buffer_store voffset stays int32) to avoid the 2**31 overflow.
        },
    )

    if enable_jit_gluon_pa_mqa_logits_kernel:
        kernel = triton.compile(
            src,
            target=target,
            options=options,
        )
    else:
        padded_str = "T" if is_padded_mode and not Preshuffle else "F"
        preshuffle_suffix = "_preshuffle" if Preshuffle else ""
        varctx_suffix = (
            f"_persistent_n{NextN}" if Persistent else "_varctx" if VarCtxOpt else ""
        )
        kernel_str = f"paged_mqa_logits{preshuffle_suffix}{varctx_suffix}_{ChunkQ}x{ChunkK}x{HiddenDim}_B{KVBlockSize}P{padded_str}W{WavePerEU}"
        metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/paged_mqa_logits/aot/{kernel_str}"
        with AOTMetadataContext(
            kernel_fn.fn.__name__,
            metadata_pth,
        ):
            kernel = triton.compile(
                src,
                target=target,
                options=options,
            )
    return kernel


def deepgemm_fp8_paged_mqa_logits_schedule(
    batch_size,
    next_n,
    context_lens: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    assert batch_size < TotalCuCount * WavePerEU // next_n

    max_chunks = math.ceil(max_model_len / ChunkK)
    schedule_waves_per_eu = 4
    grid = (TotalCuCount * schedule_waves_per_eu, 1, 1)
    TryCount = math.ceil(max_chunks / grid[0])
    align_power_of_2_batch = 1 << (batch_size - 1).bit_length()

    safe_chunks_per_cta = torch.empty(
        (1,),
        device="cuda",
        dtype=torch.int32,
    )
    _deepgemm_fp8_paged_mqa_logits_varctx_schedule[grid](
        batch_size,
        context_lens,
        safe_chunks_per_cta,
        TotalCuCount * WavePerEU // next_n,
        ChunkK,
        align_power_of_2_batch,
        TryCount,
        waves_per_eu=schedule_waves_per_eu,
    )
    return safe_chunks_per_cta


def deepgemm_fp8_paged_mqa_logits_persistent_schedule(
    batch_size: int,
    next_n: int,
    context_lens: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
    NumCTAs: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the FP8 decode persistent grid entirely on the GPU.

    The returned int32 ``[NumCTAs, 4]`` table contains packed query row, first
    chunk, chunk count and context length. Pass it as ``PersistentSchedule`` to
    :func:`deepgemm_fp8_paged_mqa_logits`. Rebuild it when context lengths,
    ``next_n``, ``ChunkK`` or ``max_model_len`` change. Both construction and
    execution can be captured in a CUDA graph; ``out`` allows storage reuse.

    The default starts at ``TotalCuCount * WavePerEU`` CTAs and adds grid waves
    when the upper bound exceeds 32 chunks per CTA, up to four times that
    budget. It is rounded to complete ``next_n`` groups and capped at one CTA
    per chunk. Explicit ``NumCTAs`` must accommodate one CTA per query.
    """
    if batch_size <= 0 or next_n <= 0 or ChunkK <= 0 or max_model_len < 0:
        raise ValueError(
            "batch_size, next_n and ChunkK must be positive; max_model_len >= 0"
        )
    if context_lens.ndim != 1 or context_lens.numel() != batch_size:
        raise ValueError("context_lens must have shape [batch_size]")
    if not context_lens.is_cuda or context_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("context_lens must be a CUDA int32 or int64 tensor")
    if NumCTAs is None:
        if TotalCuCount is None:
            TotalCuCount = get_num_sms()
        if TotalCuCount <= 0 or WavePerEU <= 0:
            raise ValueError("TotalCuCount and WavePerEU must be positive")
        base_slots = max(batch_size, triton.cdiv(TotalCuCount * WavePerEU, next_n))
        max_slots = batch_size * max(1, triton.cdiv(max_model_len, ChunkK))
        # Keep enough queued work for long contexts without letting a loose
        # output-width bound grow the persistent grid indefinitely.
        grid_waves = min(4, max(1, triton.cdiv(max_slots, 32 * base_slots)))
        slots = min(base_slots * grid_waves, max_slots)
        NumCTAs = slots * next_n
    if NumCTAs < batch_size * next_n or NumCTAs % next_n:
        raise ValueError(
            "NumCTAs must be a multiple of next_n and >= batch_size * next_n"
        )
    if out is None:
        out = torch.empty((NumCTAs, 4), dtype=torch.int32, device=context_lens.device)
    elif (
        out.shape != (NumCTAs, 4)
        or out.dtype != torch.int32
        or out.device != context_lens.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous int32 [NumCTAs, 4] on the context device"
        )
    slots = NumCTAs // next_n
    # Keep the prefix lookup small and within one wave for common decode
    # batches. Larger batches retain four waves to limit register pressure.
    block_s = 16
    schedule_warps = 1 if batch_size <= 128 else 4
    _deepgemm_fp8_paged_mqa_logits_persistent_schedule[(triton.cdiv(slots, block_s),)](
        context_lens.to(torch.int32).contiguous(),
        out,
        batch_size,
        slots,
        max_model_len,
        ChunkK=ChunkK,
        NEXT_N=next_n,
        BLOCK_B=triton.next_power_of_2(batch_size),
        BLOCK_S=block_s,
        num_warps=schedule_warps,
    )
    return out


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
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
    VarCtxSchedule: torch.Tensor = None,
    SplitKV: int | None = None,
    PersistentSchedule: torch.Tensor = None,
):
    """Compute paged FP8 decode logits, optionally using a persistent CTA table.

    Build ``PersistentSchedule`` with the same query geometry, context lengths,
    ``max_model_len`` and ``ChunkK`` using the persistent schedule helper above.
    The caller initializes logits outside the context to ``-inf`` as on the
    SplitKV path. Persistent AOT binaries use a separate ``_persistent`` name.
    """
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_Size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    Persistent = PersistentSchedule is not None
    if Persistent:
        if VarCtxSchedule is not None:
            raise ValueError(
                "PersistentSchedule and VarCtxSchedule are mutually exclusive"
            )
        if triton_version < Version("3.5.0") or get_gfx() not in ("gfx942", "gfx950"):
            raise ValueError(
                "PersistentSchedule requires Triton >= 3.5 and gfx942/gfx950"
            )
        if (
            PersistentSchedule.ndim != 2
            or PersistentSchedule.shape[1] != 4
            or PersistentSchedule.dtype != torch.int32
            or PersistentSchedule.device != q_fp8.device
            or not PersistentSchedule.is_contiguous()
            or PersistentSchedule.shape[0] < batch_size * next_n
            or PersistentSchedule.shape[0] % next_n
        ):
            raise ValueError(
                "PersistentSchedule must be contiguous int32 [NumCTAs, 4] on the query device"
            )

    if get_gfx() == "gfx1250":
        if Preshuffle and hidden_dim <= 128:
            WavePerEU = 4
        else:
            WavePerEU = 1

    TileQCount = batch_size * next_n
    if SplitKV is None:
        if get_gfx() == "gfx1250":
            SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU * 2
        else:
            # Two workgroups per CU, with the context split evenly between
            # them. Rounding SplitKV itself to a multiple of 5 leaves the grid
            # at a fraction of the CU count -- 2.5 workgroups per CU at batch
            # 16/32/64 -- so half the CUs run one more workgroup than the other
            # half and the kernel waits for them. Sizing the grid instead of
            # SplitKV keeps that from happening at any batch.
            SplitKV = max(1, -(-2 * TotalCuCount // TileQCount))
            chunks = max(1, -(-max_model_len // ChunkK))
            # Two per CU stops being enough once a workgroup's share of the
            # context passes ~32 chunks: at batch 128/256 it leaves each one
            # chewing 128-256 chunks with no queue behind it to rebalance
            # against. Split further there, by at most 2x -- `max_model_len` is
            # only an upper bound on the context, since the real lengths live in
            # a device tensor, and a loose bound must not over-split without
            # limit.
            SplitKV = min(max(SplitKV, -(-chunks // 32)), SplitKV * 2)
            # Splits past the last chunk return immediately; don't launch them.
            SplitKV = min(SplitKV, chunks)

    assert ChunkK % KVBlockSize == 0 or KVBlockSize % ChunkK == 0
    assert block_Size == KVBlockSize
    if Preshuffle:
        assert (
            KVBlockSize % 16 == 0
        ), f"Preshuffle mode only supports KVBlockSize aligned to 16. Got KVBlockSize={KVBlockSize}"

    kv_cache = kv_cache.view(-1, KVBlockSize * index_dim)
    num_block = kv_cache.shape[0]
    kv_cache_fp8, kv_cache_scale = (
        kv_cache[..., : KVBlockSize * hidden_dim],
        kv_cache[..., KVBlockSize * hidden_dim :],
    )
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)
    kv_cache_scale = kv_cache_scale.view(torch.float32)

    if VarCtxSchedule is not None and get_gfx() == "gfx1250":
        import warnings

        warnings.warn(
            "VarCtx schedule is not implemented on gfx1250 yet; ignoring it and "
            "falling back to the non-varctx preshuffle path."
        )
        VarCtxSchedule = None

    VarCtxOpt = VarCtxSchedule is not None
    if Persistent:
        grid = (PersistentSchedule.shape[0], 1, 1)
    elif VarCtxOpt:
        grid = (TotalCuCount * WavePerEU, 1, 1)
    else:
        grid = (batch_size * next_n * SplitKV, 1, 1)

    if enable_gluon_pa_mqa_logits:
        is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0
        kernel = _compile_deepgemm_fp8_paged_mqa_logits(
            ChunkQ=heads,
            ChunkK=ChunkK,
            Preshuffle=Preshuffle,
            KVBlockSize=KVBlockSize,
            HiddenDim=hidden_dim,
            is_padded_mode=is_padded_mode,
            WavePerEU=WavePerEU,
            VarCtxOpt=VarCtxOpt,
            Persistent=Persistent,
            NextN=next_n if Persistent else 1,
        )
        if triton_version >= Version("3.5.0"):
            cdna_version = get_cdna_version()
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
                num_block,
                (
                    PersistentSchedule
                    if Persistent
                    else VarCtxSchedule if VarCtxOpt else SplitKV
                ),
                # constexpr
                heads,
                ChunkK,
                KVBlockSize,
                hidden_dim,
                cdna_version,
                get_gfx(),
                *(() if VarCtxOpt else (Persistent, next_n if Persistent else 1)),
            )
        else:  #  load AOT compiled gluon kernel
            assert triton_version < Version(
                "3.4.0"
            ), "https://github.com/triton-lang/triton/pull/7258 involves a ABI-breaking change on triton3.4, "
            "which adding an extra pointer argument at the end of kernel arguments. To ensure compatibility"
            "with AOT compiled gluon kernel on triton3.5, a feasible solution is to add a pointer parameter "
            "at the end of the parameters and ensure that the Triton version used is before the ABI "
            "modification, i.e., verison<3.4.0"
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
                SplitKV if not VarCtxOpt else VarCtxSchedule,
                out_logits,  # dummyPointerArg for triton version < 3.4.0,
                # constexpr
                heads,
                ChunkK,
                KVBlockSize,
                hidden_dim,
            )
    else:
        assert not Preshuffle, "Preshuffle mode is only supported on gluon kernel."
        kv_cache_values = kv_cache_fp8.view(num_block, KVBlockSize, hidden_dim)
        kv_cache_scales = kv_cache_scale.view(num_block, KVBlockSize)
        kernel = _deepgemm_fp8_paged_mqa_logits[grid](
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_values,
            kv_cache_values.stride(0),
            kv_cache_values.stride(1),
            kv_cache_scales,
            kv_cache_scales.stride(0),
            kv_cache_scales.stride(1),
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
            KVBlockSize=KVBlockSize,
        )
    return triton.runtime.cache.get_cache_manager(kernel.hash).key
