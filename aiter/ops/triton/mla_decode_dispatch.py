# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# Copyright (C) 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

from typing import Optional
import functools
import json
import os
import triton
import triton.language as tl
import torch
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter import dtypes

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.backends.compiler import GPUTarget

enable_aot_gluon_mla = os.environ.get(
    "AITER_ENABLE_AOT_MLA", "0"
)
enable_aot_gluon_mla = enable_aot_gluon_mla == "1"

if triton.__version__ >= "3.5.0" and not enable_aot_gluon_mla:
    from triton.experimental.gluon._runtime import GluonASTSource as ASTSource
    enable_gluon_mla = True
    enable_jit_gluon_mla = True
    from aiter.ops.triton.gluon.mla_decode_mi355 import (
        _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64
    )
else:
    from triton.compiler import ASTSource
    enable_gluon_mla = enable_aot_gluon_mla
    from aiter.ops.triton._triton_kernels.mla_decode_mi355 import (
        _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64
    )
    enable_jit_gluon_mla = False

from aiter.utility.triton.triton_metadata_redirect import (
    AOTMetadataContext,
)


@functools.lru_cache(maxsize=None)
def _compile_mla(
    kv_lora_rank,
    qk_rope_head_dim,
    kv_group_num,
    head_num,
    batch,
    logit_cap,
    max_qo_len,
    PAGE_BLOCK_SIZE,
    BLOCK_C,
    BLOCK_R,
    BLOCK_N,
    BLOCK_H,
    NUM_KV_SPLITS,
    q_dtype,
    WavePerEU: int = 2,
):
    arch = arch_info.get_arch()
    dev = arch_info.get_device()
    target = GPUTarget("hip", arch, 64)
    if q_dtype == dtypes.fp8:
        dtype_str = "*fp8e4b8"
        dtype = "fp8"
    else:
        dtype_str = "*bf16"
        dtype = "bf16"

    fn_signature = {
        "Q": dtype_str,
        "K_Buffer": dtype_str,
        "V_buffer": dtype_str,
        "sm_scale": "fp32",
        "kv_indptr": "*i32",
        "kv_indices": "*i32",
        "Att_Out": "*fp32",
        "Att_Lse": "*fp32",
        "stride_qb": "i32",
        "stride_qh": "i32",
        "stride_buf_kbs": "i32",
        "stride_buf_kh":  "i32",
        "stride_mid_ob": "i32",
        "stride_mid_oh": "i32",
        "stride_mid_os": "i32",
        "stride_mid_lse_b": "i32",
        "stride_mid_lse_h": "i32",
        "stride_mid_lse_s": "i32",
        "stride_b_block_table": "i32",
    }
    if not enable_jit_gluon_mla:
        fn_signature["dummyPointerArg"] = "*i32"
    fn_signature["kv_lora_rank"] = "constexpr"
    fn_signature["qk_rope_head_dim"] = "constexpr"
    fn_signature["kv_group_num"] = "constexpr"
    fn_signature["q_head_num"] = "constexpr"
    fn_signature["batch"] = "constexpr"
    fn_signature["logit_cap"] = "constexpr"
    fn_signature["max_qo_len"] = "constexpr"
    fn_signature["BLOCK_C"] = "constexpr"
    fn_signature["BLOCK_R"] = "constexpr"
    fn_signature["BLOCK_N"] = "constexpr"
    fn_signature["BLOCK_H"] = "constexpr"
    fn_signature["NUM_KV_SPLITS"] = "constexpr"
    fn_signature["PAGE_BLOCK_SIZE"] = "constexpr"

    options = {
        "num_warps": 4,
        "waves_per_eu": WavePerEU,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": arch,
        "backend_name": "hip",
        "warp_size": 64,
        "name": "mla_n16x4_prefetch_k_paged_64",
    }
    kernel_fn = (
        _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64
    )
    # import pdb; pdb.set_trace()
    src = ASTSource(
        fn=kernel_fn,
        signature=fn_signature,
        constexprs={
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_group_num": kv_group_num,
            "q_head_num": head_num,
            "batch": batch,
            "logit_cap": logit_cap,
            "max_qo_len": max_qo_len,
            "BLOCK_C": BLOCK_C,
            "BLOCK_R": BLOCK_R,
            "BLOCK_N": BLOCK_N,
            "BLOCK_H": BLOCK_H,
            "NUM_KV_SPLITS": NUM_KV_SPLITS,
            "PAGE_BLOCK_SIZE": PAGE_BLOCK_SIZE,
        },
        attrs={
            (0,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q
            (1,): [["tt.divisibility", 16]],  # k_buffer 
            (2,): [["tt.divisibility", 16]],  # v_buffer
            (3,): [["tt.divisibility", 16]],  # sm_scale
            (1,): [["tt.divisibility", 16]],  # kv_indptr 
            (2,): [["tt.divisibility", 16]],  # kv_indices
            (1,): [["tt.divisibility", 16]],  # Att_Out 
            (2,): [["tt.divisibility", 16]],  # Att_Lse
            (4,): [["tt.divisibility", 16]],  # stride_qb
            (5,): [["tt.divisibility", 16]],  # stride_qh
            (6,): [["tt.divisibility", 16]],  # stride_buf_kbs
            (7,): [["tt.divisibility", 16]],  # stride_buf_kh
            (8,): [["tt.divisibility", 16]],  # stride_mid_ob
            (9,): [["tt.divisibility", 16]],  # stride_mid_oh
            (10,): [["tt.divisibility", 16]],  # stride_mid_os
            (11,): [["tt.divisibility", 16]],  # stride_mid_lse_b
            (12,): [["tt.divisibility", 16]],  # stride_mid_lse_h
            (13,): [["tt.divisibility", 16]],  # stride_mid_lse_s
            (14,): [["tt.divisibility", 16]],  # stride_b_block_table
        },
    )

    if enable_jit_gluon_mla:
        kernel = triton.compile(
            src,
            target=target,
            options=options,
        )
    else:
        kernel_str = f"mla_n16x4_prefetch_k_paged_64_{dtype}_{dev}"
        metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/mla/aot/{kernel_str}"
        # metadata_pth = f"./mla/aot/{kernel_str}"
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


def _decode_grouped_att_m_fwd(
    q,               # [b, sq, hq, 576]
    k_buffer,        # [pages, hk, 576]
    v_buffer,
    att_out,
    att_lse,
    kv_lora_rank,  # c
    kv_indptr,
    kv_indices,
    block_tables,
    num_kv_splits,
    sm_scale,
    logit_cap,
    mtp,
    config,
):
    qk_rope_head_dim = q.size(-1) - kv_lora_rank
    # batch, head_num = kv_indptr.size(0) - 1, q.size(1)
    batch, head_num = q.size(0), q.size(1)
    kv_group_num = q.size(1) // k_buffer.size(-2)
    page_block_size = k_buffer.size(1)

    # batch = batch*(mtp + 1)

    config["BLOCK_C"] = triton.next_power_of_2(kv_lora_rank)
    config["BLOCK_R"] = triton.next_power_of_2(qk_rope_head_dim)
    # print(batch, head_num, kv_group_num)

    config["BLOCK_H"] = ((kv_group_num + 15) // 16) * 16

    # print(config["BLOCK_H"])

    config["NUM_KV_SPLITS"] = num_kv_splits
    block_h = min(config["BLOCK_H"], kv_group_num)
    grid = (
        (head_num + block_h - 1) // block_h
        * batch
        * config["NUM_KV_SPLITS"],
        1,
        1,
    )
    # print(q.shape, grid)

    kernel = _compile_mla(
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num,
        head_num,
        batch,
        logit_cap,
        mtp + 1,
        PAGE_BLOCK_SIZE=64,
        BLOCK_C=config["BLOCK_C"],
        BLOCK_R=config["BLOCK_R"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_H=config["BLOCK_H"],
        NUM_KV_SPLITS=config["NUM_KV_SPLITS"],
        q_dtype=q.dtype,
        WavePerEU=config["waves_per_eu"],
    )
    if enable_gluon_mla:
        if enable_jit_gluon_mla:
            kernel[grid](
                q,
                k_buffer,
                v_buffer,
                sm_scale,
                kv_indptr,
                block_tables,
                att_out,
                att_lse,
                q.stride(0),
                q.stride(1),
                k_buffer.stride(-3),
                k_buffer.stride(-2),
                att_out.stride(0),
                att_out.stride(1),
                att_out.stride(2),
                att_lse.stride(0),
                att_lse.stride(1),
                att_lse.stride(2),
                block_tables.stride(0),
                kv_lora_rank,
                qk_rope_head_dim,
                kv_group_num,
                head_num,
                batch,
                logit_cap,
                mtp + 1,
                config["BLOCK_C"],
                config["BLOCK_R"],
                config["BLOCK_N"],
                config["BLOCK_H"],
                config["NUM_KV_SPLITS"],
                page_block_size,
            )
        else:
            kernel[grid](
                q,
                k_buffer,
                v_buffer,
                sm_scale,
                kv_indptr,
                block_tables,
                att_out,
                att_lse,
                q.stride(0),
                q.stride(1),
                k_buffer.stride(-3),
                k_buffer.stride(-2),
                att_out.stride(0),
                att_out.stride(1),
                att_out.stride(2),
                att_lse.stride(0),
                att_lse.stride(1),
                att_lse.stride(2),
                block_tables.stride(0),
                att_out,
                kv_lora_rank,
                qk_rope_head_dim,
                kv_group_num,
                head_num,
                batch,
                logit_cap,
                mtp + 1,
                config["BLOCK_C"],
                config["BLOCK_R"],
                config["BLOCK_N"],
                config["BLOCK_H"],
                config["NUM_KV_SPLITS"],
                page_block_size,
            )
            # pass

    # if page_block_size == 64:
    #     # import pdb; pdb.set_trace()
    #     config["PAGE_BLOCK_SIZE"] = page_block_size
    #     kernel = _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64[grid](
    #         q,
    #         k_buffer,
    #         v_buffer,
    #         sm_scale,
    #         kv_indptr,
    #         block_tables,
    #         att_out,
    #         att_lse,
    #         q.stride(0),
    #         q.stride(1),
    #         k_buffer.stride(-3),
    #         k_buffer.stride(-2),
    #         att_out.stride(0),
    #         att_out.stride(1),
    #         att_out.stride(2),
    #         att_lse.stride(0),
    #         att_lse.stride(1),
    #         att_lse.stride(2),
    #         block_tables.stride(0),
    #         kv_lora_rank=kv_lora_rank,
    #         qk_rope_head_dim=qk_rope_head_dim,
    #         kv_group_num=kv_group_num,
    #         q_head_num=head_num,
    #         batch=batch,
    #         logit_cap=logit_cap,
    #         max_qo_len=mtp + 1,
    #         **config,
    #     )
    return triton.runtime.cache.get_cache_manager(kernel.hash).key


@triton.jit
def _fwd_kernel_stage2(
    Att_Out,
    Att_Lse,
    O,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    batch: tl.constexpr,
    head_num: tl.constexpr,
    max_qo_len: tl.constexpr,
    PAGE_BLOCK_SIZE: gl.constexpr,
):
    pid = tl.program_id(0)

    pid = remap_xcd(pid, batch * head_num)
    cur_batch = pid % batch
    cur_head = (pid // batch) % head_num

    cur_batch_kv = cur_batch // max_qo_len
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch_kv + 1) - tl.load(
        kv_indptr + cur_batch_kv
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_lse_b + cur_head * stride_mid_lse_h
    # kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    cur_batch_block_nums = gl.cdiv(cur_batch_seq_len, PAGE_BLOCK_SIZE)
    blocks_per_split = gl.cdiv(cur_batch_block_nums, NUM_KV_SPLITS)

    for split_kv_id in range(0, NUM_KV_SPLITS):
        split_kv_start = blocks_per_split * split_kv_id * PAGE_BLOCK_SIZE
        split_kv_end = tl.minimum(split_kv_start + blocks_per_split * PAGE_BLOCK_SIZE, cur_batch_seq_len)

        if split_kv_end > split_kv_start:

            tv = tl.load(
                Att_Out + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Att_Lse + offs_logic + split_kv_id * stride_mid_lse_s)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    att_out,
    att_lse,
    q,
    o,
    k_buffer,
    kv_indptr,
    num_kv_splits,
    mtp,
    config,
):
    batch, head_num = q.size(0), q.size(1)
    page_block_size = k_buffer.size(1)
    Lv = o.size(-1)
    config["BLOCK_DV"] = triton.next_power_of_2(Lv)

    config["NUM_KV_SPLITS"] = num_kv_splits
    config["PAGE_BLOCK_SIZE"] = page_block_size

    grid = (batch * head_num,)
    _fwd_kernel_stage2[grid](
        att_out,
        att_lse,
        o,
        kv_indptr,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_lse.stride(0),
        att_lse.stride(1),
        att_lse.stride(2),
        o.stride(0) * (mtp + 1),
        o.stride(1),
        Lv=Lv,
        head_num=head_num,
        batch=batch,
        max_qo_len=mtp + 1,
        **config,
    )


@functools.lru_cache(maxsize=1024)
def _get_config():
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MLA_DECODE_ROPE-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict


def decode_attention_fwd_grouped(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lora_rank: int,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    logit_cap: Optional[float] = 0.0,
    mtp: Optional[int] = 1,
    config: Optional[dict[str, any]] = None,
):
    """
    Implements deepseek decode attention with grouped query attention and rotary positional encoding

    parameters:
    q: Query Tensor
    k_buffer: Key Cache Tensor
    v_buffer: Value Cache Tensor
    o: Output tensor containing the result of decode. Allocated by the caller
    kv_indptr:
    kv_indices:
    kv_lora_rank:
    attn_logits:
    num_kv_splits:
    sm_scale
    logit_cap:

    Returns:
    o: output Tensor

    """
    if config is None:
        config = _get_config()

    cache_key = _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_lora_rank,
        kv_indptr,
        kv_indices,
        block_tables,
        num_kv_splits,
        sm_scale,
        logit_cap,
        mtp,
        config["fwd_grouped_kernel_stage1_rope_fp8"] if q.dtype == dtypes.fp8 else config["fwd_grouped_kernel_stage1_rope"],
    )
    # import pdb ;pdb.set_trace()
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        k_buffer,
        kv_indptr,
        num_kv_splits,
        mtp,
        config["fwd_kernel_stage2"],
    )

    return cache_key
