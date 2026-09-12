import ctypes
import math
from functools import cache, lru_cache

from jinja2 import Template

from csrc.cpp_itfs.utils import AITER_CORE_DIR, compile_template_op, str_to_bool

MD_NAME = "pa_v1"

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_v1.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(
    gqa_ratio: int,
    head_size: int,
    npar_loops: int,
    dtype: str,
    kv_dtype: str,
    fp8_kv_dtype: str,
    out_dtype: str,
    block_size: int,
    alibi_enabled: bool,
    logits_soft_cap_enabled: bool,
    partition_size: int = 256,
    mtp: int = 1,
    sliding_window_enabled: bool = False,
    folder: str | None = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_kernels.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_v1.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",
            f"{AITER_CORE_DIR}/csrc/include",
            f"{AITER_CORE_DIR}/csrc/include/ck_tile/",
        ],
        gqa_ratio=gqa_ratio,
        head_size=head_size,
        npar_loops=npar_loops,
        dtype=dtype,
        kv_dtype=kv_dtype,
        fp8_kv_dtype=fp8_kv_dtype,
        out_dtype=out_dtype,
        block_size=block_size,
        alibi_enabled=alibi_enabled,
        logits_soft_cap_enabled=logits_soft_cap_enabled,
        partition_size=partition_size,
        mtp=mtp,
        sliding_window_enabled=sliding_window_enabled,
        folder=folder,
    )


def _validate_shapes(
    block_tables_shape,
    num_context_lens,
    num_heads,
    head_size,
    block_size,
    max_context_len,
    partition_size,
    mtp,
    q_elem_size,
):
    """Check the launch shapes and return the minimum workspace size in bytes.

    Takes plain shape scalars rather than tensors so that the result can be
    memoized per launch signature instead of recomputed on every call.
    """
    op_name = "paged_attention_v1"
    max_num_partitions = math.ceil(max_context_len / partition_size)
    # Use the launch-time upper bound rather than reading context_lens data.
    # context_lens may live on GPU, and scalar extraction would force a sync
    # and introduce tensor-data-dependent Python in torch.compile paths.
    min_blocks_per_seq = math.ceil(max_context_len / block_size)

    if len(block_tables_shape) != 2:
        raise ValueError(
            f"{op_name}: block_tables must be 2D "
            f"[num_seqs, max_num_blocks_per_seq], got {tuple(block_tables_shape)}"
        )
    num_seqs, max_num_blocks_per_seq = block_tables_shape
    if max_num_blocks_per_seq < min_blocks_per_seq:
        raise ValueError(
            f"{op_name}: block_tables.size(1)={max_num_blocks_per_seq} is too small "
            f"for max_context_len={max_context_len} and block_size={block_size}; "
            f"need at least {min_blocks_per_seq} block-table entries per sequence"
        )
    if num_context_lens != num_seqs:
        raise ValueError(
            f"{op_name}: context_lens.size(0)={num_context_lens} must match "
            f"block_tables.size(0)={num_seqs}"
        )

    return (
        num_seqs * mtp * num_heads * max_num_partitions * head_size * q_elem_size
        + 2 * num_seqs * mtp * num_heads * max_num_partitions * 4
    )


def _raise_workspace_too_small(
    workspace_bytes,
    required_bytes,
    max_context_len,
    partition_size,
    num_seqs,
    num_heads,
    head_size,
    mtp,
):
    raise ValueError(
        f"paged_attention_v1: workspace_buffer is too small ({workspace_bytes} bytes) "
        f"for max_context_len={max_context_len}, partition_size={partition_size}, "
        f"num_seqs={num_seqs}, num_heads={num_heads}, head_size={head_size}, mtp={mtp}; "
        f"need at least {required_bytes} bytes "
        f"({math.ceil(max_context_len / partition_size)} partition slots)"
    )


def validate_paged_attention_v1_workspace(
    workspace_buffer,
    query,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    partition_size,
    mtp=1,
):
    num_heads = query.size(1)
    head_size = query.size(2)
    required_bytes = _validate_shapes(
        tuple(block_tables.shape),
        context_lens.size(0),
        num_heads,
        head_size,
        block_size,
        max_context_len,
        partition_size,
        mtp,
        query.element_size(),
    )
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if workspace_bytes < required_bytes:
        _raise_workspace_too_small(
            workspace_bytes,
            required_bytes,
            max_context_len,
            partition_size,
            block_tables.size(0),
            num_heads,
            head_size,
            mtp,
        )


_NULL = ctypes.c_void_p(0)

# kv_cache_dtype -> query dtype -> (compute type, kv cache type) that the kernel
# template is instantiated with, plus out dtype -> output type. Built lazily so
# that importing this module (e.g. for ahead-of-time compilation) needs no torch.
_HIP_TYPES = None
_HIP_OUT_TYPES = None
_get_raw_stream = None


def _init_torch_tables():
    global _HIP_TYPES, _HIP_OUT_TYPES, _get_raw_stream
    import torch

    _HIP_TYPES = {
        "auto": {
            torch.bfloat16: ("__hip_bfloat16", "__hip_bfloat16"),
            torch.float16: ("_Float16", "_Float16"),
        },
        "fp8": {
            torch.bfloat16: ("__hip_bfloat16", "uint8_t"),
            torch.float16: ("_Float16", "uint8_t"),
        },
    }
    _HIP_TYPES["fp8_e4m3"] = _HIP_TYPES["fp8"]
    _HIP_OUT_TYPES = {
        torch.bfloat16: "__hip_bfloat16",
        torch.float16: "_Float16",
    }
    _get_raw_stream = torch._C._cuda_getCurrentRawStream


@cache
def _warp_size(device_index):
    import torch

    return torch.cuda.get_device_properties(device_index).warp_size


@lru_cache(maxsize=1024)
def _plan(
    q_dtype,
    out_dtype,
    kv_cache_dtype,
    gqa_ratio,
    head_size,
    block_size,
    alibi_enabled,
    logits_soft_cap_enabled,
    sliding_window_enabled,
    partition_size,
    mtp,
    scale,
    logits_soft_cap,
    sliding_window,
    block_tables_shape,
    num_context_lens,
    num_heads,
    num_kv_heads,
    q_stride,
    kv_block_stride,
    kv_head_stride,
    kv_seq_stride,
    max_context_len,
    q_elem_size,
    device_index,
):
    """Resolve the kernel and pre-build the by-value arguments for a call shape.

    Everything here is a function of shapes, strides and dtypes only, so it is
    done once per distinct launch signature. At batch=1 the host-side work used
    to cost more than the kernels themselves, which made single-request decode
    launch-bound; see ROCm/aiter#2495.
    """
    if _HIP_TYPES is None:
        _init_torch_tables()

    try:
        dtype, kv_dtype = _HIP_TYPES[kv_cache_dtype][q_dtype]
    except KeyError:
        raise ValueError(
            f"paged_attention_v1: unsupported kv_cache_dtype/query dtype "
            f"combination: {kv_cache_dtype}/{q_dtype}"
        ) from None
    try:
        out_hip_dtype = _HIP_OUT_TYPES[out_dtype]
    except KeyError:
        raise ValueError(f"Unsupported data type: {out_dtype}") from None

    required_bytes = _validate_shapes(
        block_tables_shape,
        num_context_lens,
        num_heads,
        head_size,
        block_size,
        max_context_len,
        partition_size,
        mtp,
        q_elem_size,
    )

    max_num_partitions = math.ceil(max_context_len / partition_size)
    func = compile(
        gqa_ratio,
        head_size,
        math.ceil(max_num_partitions / _warp_size(device_index)),
        dtype,
        kv_dtype,
        kv_cache_dtype,
        out_hip_dtype,
        block_size,
        alibi_enabled,
        logits_soft_cap_enabled,
        partition_size,
        mtp,
        sliding_window_enabled=sliding_window_enabled,
    )
    num_seqs, max_num_blocks_per_seq = block_tables_shape
    scalars = (
        ctypes.c_float(scale),
        ctypes.c_int(max_num_blocks_per_seq),
        ctypes.c_int(max_num_partitions),
        ctypes.c_float(logits_soft_cap),
        ctypes.c_int(num_seqs),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_heads),
        ctypes.c_int(q_stride),
        ctypes.c_int(kv_block_stride),
        ctypes.c_int(kv_head_stride),
        ctypes.c_int(kv_seq_stride),
        ctypes.c_int(sliding_window),
    )
    return func, scalars, required_bytes


def paged_attention_v1(
    out,
    workspace_buffer,
    query,
    key_cache,
    value_cache,
    scale: float,
    block_tables,
    cu_query_lens,
    context_lens,
    max_context_len: int,
    alibi_slopes,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    k_scale,
    v_scale,
    fp8_out_scale=None,
    partition_size: int = 256,
    mtp: int = 1,
    q_scale=None,
    sliding_window: int = 0,
):
    q_shape = query.shape
    kv_shape = key_cache.shape
    kv_stride = key_cache.stride()
    if kv_cache_layout == "HND":
        num_kv_heads, block_size = kv_shape[1], kv_shape[2]
        kv_head_stride, kv_seq_stride = kv_stride[1], kv_stride[2]
    else:
        block_size, num_kv_heads = kv_shape[1], kv_shape[2]
        kv_head_stride, kv_seq_stride = kv_stride[2], kv_stride[1]
    num_heads = q_shape[1]
    head_size = q_shape[2]
    device_index = query.get_device()

    func, scalars, required_bytes = _plan(
        query.dtype,
        out.dtype,
        kv_cache_dtype,
        num_heads // num_kv_heads,
        head_size,
        block_size,
        alibi_slopes is not None,
        logits_soft_cap > 0,
        sliding_window > 0,
        partition_size,
        mtp,
        scale,
        logits_soft_cap,
        sliding_window,
        block_tables.shape,
        context_lens.shape[0],
        num_heads,
        num_kv_heads,
        query.stride(0),
        kv_stride[0],
        kv_head_stride,
        kv_seq_stride,
        max_context_len,
        query.element_size(),
        device_index,
    )

    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if workspace_bytes < required_bytes:
        _raise_workspace_too_small(
            workspace_bytes,
            required_bytes,
            max_context_len,
            partition_size,
            block_tables.shape[0],
            num_heads,
            head_size,
            mtp,
        )

    c_void_p = ctypes.c_void_p
    func(
        c_void_p(out.data_ptr()),
        c_void_p(workspace_buffer.data_ptr()),
        c_void_p(query.data_ptr()),
        c_void_p(key_cache.data_ptr()),
        c_void_p(value_cache.data_ptr()),
        c_void_p(block_tables.data_ptr()),
        _NULL if cu_query_lens is None else c_void_p(cu_query_lens.data_ptr()),
        c_void_p(context_lens.data_ptr()),
        _NULL if alibi_slopes is None else c_void_p(alibi_slopes.data_ptr()),
        _NULL if q_scale is None else c_void_p(q_scale.data_ptr()),
        c_void_p(k_scale.data_ptr()),
        c_void_p(v_scale.data_ptr()),
        _NULL if fp8_out_scale is None else c_void_p(fp8_out_scale.data_ptr()),
        *scalars,
        c_void_p(_get_raw_stream(device_index)),
    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_ratio", type=int, required=True)
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--npar_loops", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--fp8_kv_dtype", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--alibi_enabled", type=str_to_bool, required=True)
    parser.add_argument("--logits_soft_cap_enabled", type=str_to_bool, required=True)
    parser.add_argument("--mtp", type=int, default=1)
    parser.add_argument("--folder", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))
