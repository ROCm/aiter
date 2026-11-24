# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache

import torch
from jinja2 import Template

from csrc.cpp_itfs.torch_utils import torch_to_c_types, torch_to_hip_types
from csrc.cpp_itfs.utils import AITER_CORE_DIR, compile_template_op

MD_NAME = "pa_ps"
# The JIT build cache is keyed on MD_NAME and the template kwargs only, not on
# the source, so any change to the launcher's C signature must bump this.
_LAUNCHER_ABI_VERSION = 2

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_ps.cpp.jinja", "r") as f:
    src_template = Template(f.read())


@lru_cache(maxsize=128)
def compile(
    head_size: int,
    query_group_size: int,
    context_partition_num: int,
    out_dtype: str,
    logits_dtype: str,
    sink_dtype: str,
    use_sinks: bool,
    is_varlen: bool = False,
    folder: str | None = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_ps.cuh",
        ],
        head_size=head_size,
        query_group_size=query_group_size,
        context_partition_num=context_partition_num,
        out_dtype=out_dtype,
        logits_dtype=logits_dtype,
        sink_dtype=sink_dtype,
        use_sinks=use_sinks,
        is_varlen=is_varlen,
        abi_version=_LAUNCHER_ABI_VERSION,
        folder=folder,
    )


def launch_pa_decode_ps_reduce(
    output_ptr: torch.Tensor,
    exp_sums_ptr: torch.Tensor,
    max_logits_ptr: torch.Tensor,
    logits_ptr: torch.Tensor,
    sink_token_ptr: torch.Tensor,
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    stride_exp_sums_seq: int,
    stride_exp_sums_head: int,
    stride_exp_sums_part: int,
    stride_logits_seq: int,
    stride_logits_head: int,
    stride_logits_part: int,
    stride_logits_group: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    context_partition_num: int,
    query_start_loc: torch.Tensor | None = None,
):
    """Reduce the PS partitions into ``output``.

    Without ``query_start_loc`` the output is the padded 5D
    ``[batch, query_seq_len, num_kv_heads, group, head]`` view. With it, the
    output is packed per token as ``[1, num_tokens, num_kv_heads, group, head]``
    (``stride_output_bs`` is unused) and sequence ``b`` writes rows
    ``[query_start_loc[b], query_start_loc[b + 1])``.
    """
    if query_start_loc is None:
        batch_size = output_ptr.shape[0]
    else:
        assert query_start_loc.dtype == torch.int32
        batch_size = query_start_loc.shape[0] - 1
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if not (1 <= context_partition_num <= 64):
        raise ImportError(
            "C++ PS reduce: optimized kernel supports 1..64 partition slots"
        )
    if output_ptr.dtype not in supported_dtypes:
        raise ImportError(
            f"C++ PS reduce: unsupported output dtype {output_ptr.dtype!r}"
        )
    if logits_ptr.dtype not in supported_dtypes:
        raise ImportError(
            f"C++ PS reduce: unsupported logits dtype {logits_ptr.dtype!r}"
        )
    if sink_token_ptr is not None and sink_token_ptr.dtype not in supported_dtypes:
        raise ImportError(
            f"C++ PS reduce: unsupported sink dtype {sink_token_ptr.dtype!r}"
        )

    out_dtype = torch_to_hip_types(output_ptr.dtype)[0]
    logits_dtype = torch_to_hip_types(logits_ptr.dtype)[0]
    sink_dtype = torch_to_hip_types(
        (output_ptr if sink_token_ptr is None else sink_token_ptr).dtype
    )[0]

    func = compile(
        head_size=head_size,
        query_group_size=query_group_size,
        context_partition_num=context_partition_num,
        out_dtype=out_dtype,
        logits_dtype=logits_dtype,
        sink_dtype=sink_dtype,
        use_sinks=sink_token_ptr is not None,
        is_varlen=query_start_loc is not None,
    )

    func(
        *torch_to_c_types(
            output_ptr,
            exp_sums_ptr,
            max_logits_ptr,
            logits_ptr,
            sink_token_ptr,
            query_start_loc,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            batch_size,
            output_ptr.shape[2],
            query_seq_len,
            query_group_size,
            context_partition_num,
            torch.cuda.current_stream(output_ptr.device),
        )
    )

    return output_ptr
