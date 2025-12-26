import random
from typing import List, Optional, Tuple, Union
import itertools
import torch
import aiter
import pytest
from aiter.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from aiter import pertoken_quant
from aiter import dtypes
from enum import Enum
from einops import rearrange
import argparse
import os
import numpy as np
from aiter import paged_attention_ragged

uniform_range = (-1, 1)


class PAVariant(Enum):
    Shomy = 1
    Asm = 2
    Naive = 3


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory_v2(
    num_blocks: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    key_cache_shape = (num_blocks, 1, num_heads, head_size)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, 1, num_heads, head_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


def kv_ptr_factory(
    num_seqs: int,
    ctx_lens: int,
    page_size: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # kv_indptr
    num_blocks_list = [ctx_lens] * num_seqs
    kv_indptr = torch.tensor([0] + num_blocks_list).cumsum(dim=0, dtype=torch.int)

    # kv_page_indices
    padded_ctx_lens = page_size * int(
        np.ceil(ctx_lens / page_size)
    )  # e.g., ctx_lens=10, page_size=3 --> padded_ctx_lens=12
    index_total = num_seqs * padded_ctx_lens
    head_per_row = int(np.ceil(ctx_lens / page_size))
    head_total = num_seqs * int(np.ceil(ctx_lens / page_size))

    # Generate heads (Start from 0, page_size, 2xpage_size, ...)
    all_heads = np.arange(0, index_total, page_size)
    np.random.shuffle(all_heads)
    row_chunks = all_heads.reshape(num_seqs, head_per_row)

    # Sort the chunks since the page indices are in ascending order.
    sorted_row_heads = np.sort(row_chunks, axis=1)
    extended_heads = np.repeat(sorted_row_heads, page_size, axis=1)

    # Create Offset matrix with shape (1, length)
    offset = np.tile(np.arange(page_size), head_per_row)
    print(sorted_row_heads.shape)

    # shape (bs, length)
    offset_tile = np.tile(offset, (num_seqs, 1))

    # Extend sorted_row_heads

    kv_page_indices = extended_heads + offset_tile
    kv_page_indices = kv_page_indices[:, :ctx_lens]
    kv_page_indices = torch.from_numpy(kv_page_indices).to(
        device="cuda:0", dtype=torch.int32
    )
    return kv_indptr, kv_page_indices.reshape(-1)


def run_aiter(
    output,
    workspace_buffer,
    query,
    key_cache,
    value_cache,
    scale,
    kv_indptr,
    kv_page_indices,
    kv_last_page_len,
    # page_size,      # New args
    block_size,
    max_num_partitions,
    alibi_slopes,
    kv_cache_dtype,
    kv_cache_layout,
    logits_soft_cap,
    k_scale,
    v_scale,
    fp8_out_scale,
    _PARTITION_SIZE_ROCM,
    version="GOLDEN",
):
    os.environ["QKV_VERSION"] = version
    torch.ops.aiter.paged_attention_ragged(
        output,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_len,
        # page_size,      # New args
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale,
        _PARTITION_SIZE_ROCM,
    )

    return workspace_buffer, output


def test_paged_attention(
    in_pt: str,
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    page_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    pa_variant: PAVariant,
    quant_cache_dtype: torch.dtype,
    seed: int,
    device: str,
    warmup_iter: int,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_default_device(device)
    block_size = 1

    if in_pt == None:
        # Using default kv_scale
        k_scale = v_scale = torch.tensor([1.0], dtype=dtypes.fp32)
        scale = float(1.0 / (head_size**0.5))
        num_query_heads, num_kv_heads = num_heads
        alibi_slopes = None
        if use_alibi:
            alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        max_seq_len = ctx_lens
        padded_ctx_lens = page_size * int(np.ceil(max_seq_len / page_size))  # e.g.,
        num_blocks = padded_ctx_lens * num_seqs

        # prepare inputs & golden output
        query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
        query.uniform_(*uniform_range)

        # Create the KV caches.
        key_caches, value_caches = kv_cache_factory_v2(
            num_blocks,
            page_size,
            1,
            num_kv_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            seed,
            device,
        )
        key_cache, value_cache = key_caches[0], value_caches[0]
        kv_indptr, kv_page_indices = kv_ptr_factory(num_seqs, ctx_lens, page_size)
        kv_last_page_len = torch.tensor(
            [block_size for i in range(num_seqs)], dtype=torch.int
        )
        block_size = key_cache.shape[2 if kv_cache_layout == "HND" else 1]
    else:  # Load from pt
        gpu_index = torch.cuda.current_device()
        TARGET_DEVICE = torch.device(f"cuda:{gpu_index}")

        data = torch.load(in_pt)
        query = data["q"].clone().detach().to(TARGET_DEVICE)
        workspace = torch.empty(*data["workspace_buffer_shape"]).to(TARGET_DEVICE)
        key_cache = torch.empty(*data["k_buffer_shape"]).to(TARGET_DEVICE)
        value_cache = torch.empty(*data["v_buffer_shape"]).to(TARGET_DEVICE)
        kv_indptr = data["kv_indptr"].clone().detach().to(TARGET_DEVICE)
        kv_page_indices = data["kv_indices"].clone().detach().to(TARGET_DEVICE)
        kv_last_page_len = data["kv_last_page_len"].clone().detach().to(TARGET_DEVICE)
        page_size = data["page_size"]
        block_size = data["block_size"]
        max_seq_len = kv_indptr[1] - kv_indptr[0]
        kv_cache_dtype = data["kv_cache_dtype"]
        kv_cache_layout = data["kv_cache_layout"]
        scale = data["scale"]
        alibi_slopes = (
            data["alibi_slopes"].to(TARGET_DEVICE)
            if isinstance(data["alibi_slopes"], torch.Tensor)
            else data["alibi_slopes"]
        )
        logits_soft_cap = data["logits_soft_cap"]
        k_scale = data["k_scale"]
        v_scale = data["v_scale"]

    _PARTITION_SIZE_ROCM = 256
    fp8_out_scale = None
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
        max_seq_len + _PARTITION_SIZE_ROCM - 1
    ) // _PARTITION_SIZE_ROCM
    assert _PARTITION_SIZE_ROCM % block_size == 0

    # will use single workspace buffer to accommodate following 3 intermediate tensors:
    #   1. tmp_output (shape=(num_seqs, num_heads, max_num_partitions, head_size), dtype=output.dtype)
    #   2. exp_sums (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    #   3. max_logits (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    output = torch.empty_like(query)
    nbyes_per_qo_elem = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * num_heads * max_num_partitions * head_size) * nbyes_per_qo_elem
        + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )

    cpa_fp8_out = False
    if fp8_out_scale is not None:
        output = torch.empty_like(output, dtype=dtypes.fp8)
        cpa_fp8_out = True
    torch.cuda.synchronize()

    # Debug
    print(
        f"[DEBUG pa_unit_test.py]  value_cache.is_contiguous()={value_cache.is_contiguous()}"
    )
    print(
        f"[DEBUG] kv_indptr.shape={kv_indptr.shape}, kv_page_indices.shape={kv_page_indices.shape}, kv_last_page_len.shape={kv_last_page_len.shape}"
    )
    print(
        f"[DEBUG] key_cache.shape={key_cache.shape}, value_cache.shape={value_cache.shape}"
    )
    print(f"[DEBUG] kv_page_indices={kv_page_indices}")
    print(f"[DEBUG] kv_indptr[-10:]={kv_indptr[-10:]}")
    print(
        f"[DEBUG] kv_page_indices.max()={kv_page_indices.max()}, num_seqs*ctx_lens={num_seqs*ctx_lens}"
    )
    # print(f"[DEBUG] kv_last_page_len={kv_last_page_len}")
    # print(f"kv_indptr={kv_indptr}")

    ARGS_TUPLE = (
        output,
        workspace_buffer,
        query,
        key_cache.contiguous(),
        value_cache.contiguous(),
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_len,
        # page_size,  # New args
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale if cpa_fp8_out else None,
        _PARTITION_SIZE_ROCM,
    )

    # Warmup
    for i in range(warmup_iter):
        _, _ = run_aiter(*ARGS_TUPLE, version="GOLDEN")
        _, _ = run_aiter(*ARGS_TUPLE, version="EXPERIMENTAL")
    workspace_golden, out_golden = run_aiter(*ARGS_TUPLE, version="GOLDEN")
    workspace_experi, out_experi = run_aiter(*ARGS_TUPLE, version="EXPERIMENTAL")

    # Grok1-bf16-TP8 + bs512-ilen2048:
    #    num_seqs=512, num_heads=6, max_num_partitions=8, head_size=128, nbyes_per_qo_elem=2

    # workspace_buffer size: from pa_ragged.cpp.jinja
    #     exp_sums_ptr:  = (num_seqs * num_heads * max_num_partitions) * 4 as type is float
    #                    = 512*6*8*4 bytes
    #     max_logits_ptr:= (num_seqs * num_heads * max_num_partitions) * 4 as type is float
    #                    = 512*6*8*4 bytes
    #     tmp_out_ptr:   = (num_seqs * num_heads * max_num_partitions * head_size) * nbyes_per_qo_elem
    #                    = 512*6*8*128*2 bytes
    # output size = torch.empty_like(query), dtype=dtype
    num_seqs, num_heads, head_size = query.shape
    block_size = key_cache.shape[2 if kv_cache_layout == "HND" else 1]
    _PARTITION_SIZE_ROCM = 256
    max_num_partitions = (
        max_seq_len + _PARTITION_SIZE_ROCM - 1
    ) // _PARTITION_SIZE_ROCM
    nbyes_per_qo_elem = torch.finfo(query.dtype).bits // 8
    bytes_sizes = [
        num_seqs * num_heads * max_num_partitions * 4,
        num_seqs * num_heads * max_num_partitions * 4,
        num_seqs * num_heads * max_num_partitions * head_size * nbyes_per_qo_elem,
    ]
    # print(f"[DEBUG] num_seqs={num_seqs}, num_heads={num_heads}, block_size={block_size}, "
    #             f"max_num_partitions={max_num_partitions}, head_size={head_size}, "
    #             f"nbyes_per_qo_elem={nbyes_per_qo_elem}, "
    #             f"_PARTITION_SIZE_ROCM={_PARTITION_SIZE_ROCM}")

    target_dtypes = [torch.float, torch.float, torch.bfloat16]
    import itertools

    accu_bytes = list(itertools.accumulate(bytes_sizes, initial=0))

    def split_workspace(workspace):
        blocks = []
        for i in range(len(bytes_sizes)):
            start_byte_idx = accu_bytes[i]
            end_byte_idx = accu_bytes[i + 1]
            byte_slice = workspace[start_byte_idx:end_byte_idx]
            block = byte_slice.view(target_dtypes[i])
            blocks.append(block)
        return blocks

    def NumericCheck(
        golden_tensor: torch.Tensor,
        experi_tensor: torch.Tensor,
        name: str = "Tensor",
        rtol: float = 1e-5,
        atol: float = 1e-8,
        max_display: int = 5,
    ):
        golden_tensor = golden_tensor.reshape(-1)
        experi_tensor = experi_tensor.reshape(-1)
        mismatch_mask = torch.abs(golden_tensor - experi_tensor) > (
            atol + rtol * torch.abs(experi_tensor)
        )
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
        mismatch_count = mismatch_mask.sum().item()
        if mismatch_count > 0:
            num_to_display = min(5, mismatch_count)
            print(
                f"Numeric Check [{name} Failed] Elem count: {exp_sums_golden.numel()}, mismatch_count = {mismatch_count}"
            )
            for i in range(num_to_display):
                idx = mismatch_indices[i].item()
                golden_val = exp_sums_golden[idx].item()
                experi_val = exp_sums_experi[idx].item()
                abs_diff = abs(golden_val - experi_val)
                print(
                    f"  Index [{idx}]: Golden={golden_val:.6e}, experi={experi_val:.6e}, Abs Diff={abs_diff:.2e}"
                )
        else:
            print(f"Numeric Check [{name} Success]")

    exp_sums_golden, max_logits_golden, tmp_out_golden = split_workspace(
        workspace_golden
    )
    exp_sums_experi, max_logits_experi, tmp_out_experi = split_workspace(
        workspace_experi
    )
    NumericCheck(exp_sums_golden, exp_sums_experi, "exp_sums")
    NumericCheck(max_logits_golden, max_logits_experi, "max_logits")
    NumericCheck(tmp_out_golden, tmp_out_experi, "tmp_out")
    NumericCheck(out_golden, out_experi, "out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test Paged Attention ragged.",
    )
    parser.add_argument(
        "-c",
        "--ctx_len",
        type=int,
        default=2048,
        help="""Context length.
    e.g. -c 128""",
    )
    parser.add_argument(
        "-p",
        "--pa_variant",
        type=str,
        choices=[member.name for member in PAVariant],
        default=[PAVariant.Shomy, PAVariant.Asm],
        nargs="*",
        help="It is not used. Just place an empty str",
    )
    parser.add_argument(
        "-q",
        "--quant_cache_dtype",
        type=str,
        choices=["none", "fp8", "i8"],
        default=["none", "fp8", "i8"],
        nargs="*",
        help="""Quantization cache dtype.
        e.g. -q fp8""",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=512,
        help="number of seqs",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="block size(page size)",
    )
    parser.add_argument(
        "--in-pt", type=str, default=None, help="Load data from pt file"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warmup iterations",
    )
    torch.set_printoptions(sci_mode=False)
    args = parser.parse_args()
    args.quant_cache_dtype = [
        None if i == "none" else dtypes.d_dtypes[i] for i in args.quant_cache_dtype
    ]

    ctx_len = args.ctx_len
    pa_variant = args.pa_variant
    quant_cache_dtype = args.quant_cache_dtype
    # print(f"[DEBUG pa_unit_test.py] ctx_len={ctx_len}, pa_variant={pa_variant}, quant_cache_dtype={quant_cache_dtype}")

    page_size = args.page_size  # Original block size is 1
    test_paged_attention(
        args.in_pt,
        ctx_len,
        args.n,
        (6, 1),  # num_heads: query and KV
        128,  # head_size
        False,  # use_alibi
        page_size,
        dtypes.bf16,  # dtype
        "auto",  # kv_cache_dtype
        "NHD",  # kv_cache_layout
        30.0,  # logits_soft_cap
        pa_variant,
        quant_cache_dtype,
        0,  # seed
        "cuda:0",  # device
        args.warmup,
    )


"""
# Even if the input length is 256, I use "context length = 2048"
# since I would like to know the performance of the kernel when
# the KV cache is longer than prefill 256 tokens.
python ~/Grok_SGLang0.4.9/pa_unit_test_v2.py -n 512 -c 2048 --page-size 1 --warmup 0

# E2E
RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1  python -m sglang.bench_one_batch \
	--batch-size 512 --input 256 --output 2048 --tp 8 --quantization fp8 --trust-remote-code \
    --model /data/huggingface/hub/amd/grok-1-W4A8KV8  \
	--tokenizer-path /data/huggingface/hub/Xenova/grok-1-tokenizer  \
    --attention-backend aiter


"""
