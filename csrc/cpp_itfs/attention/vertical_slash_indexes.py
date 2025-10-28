from jinja2 import Template
from csrc.cpp_itfs.utils import compile_template_op, AITER_CORE_DIR


MD_NAME = "vertical_slash_indexes"

with open(
    f"{AITER_CORE_DIR}/csrc/cpp_itfs/attention/vertical_slash_indexes.cpp.jinja", "r"
) as f:
    src_template = Template(f.read())


def compile(
    n_rows,
    block_size_m,
    block_size_n,
    nnz_v,
    nnz_s,
    folder: str = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/attention/vertical_slash_indexes.cuh",
            f"{AITER_CORE_DIR}/csrc/include",
        ],
        n_rows=n_rows,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        nnz_v=nnz_v,
        nnz_s=nnz_s,
        folder=folder,
    )


def convert_vertical_slash_index(
    q_seqlens,  # [BATCH, ]
    kv_seqlens,  # [BATCH, ]
    vertical_indexes,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes,  # [BATCH, N_HEADS, NNZ_S]
    context_size,
    block_size_M,
    block_size_N,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    # block_offset.fill_(torch.iinfo(block_offset.dtype).max)
    column_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    func = compile(n_rows=num_rows, block_size_m=block_size_M, block_size_n=block_size_N, nnz_v=nnz_vertical, nnz_s=nnz_slash)

    (
        q_seqlens_ptr,
        kv_seqlens_ptr,
        vertical_indexes_ptr,
        slash_indexes_ptr,
        block_count_ptr,
        block_offset_ptr,
        column_count_ptr,
        column_index_ptr,
        batch_size,
        num_heads
    ) = torch_to_c_types(
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        block_count,
        block_offset,
        column_count,
        column_index,
        batch_size,
        num_heads
    )

    func(
        q_seqlens_ptr,
        kv_seqlens_ptr,
        vertical_indexes_ptr,
        slash_indexes_ptr,
        block_count_ptr,
        block_offset_ptr,
        column_count_ptr,
        column_index_ptr,
        batch_size,
        num_heads,
    )
    return block_count, block_offset, column_count, column_index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--n_rows", type=int, required=True)
    parser.add_argument("--block_size_m", type=int, required=True)
    parser.add_argument("--block_size_n", type=int, required=True)
    parser.add_argument("--nnz_v", type=int, required=True)
    parser.add_argument("--nnz_s", type=int, required=True)
    args = parser.parse_args()
    compile(**vars(args))
