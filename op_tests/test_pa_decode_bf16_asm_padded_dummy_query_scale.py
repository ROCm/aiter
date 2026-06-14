"""Repro for PA BF16 ASM padded decode output poisoning.

Run inside the yhl_pa container, for example:

    cd /app/aiter
    HIP_VISIBLE_DEVICES=0 \
    PYTHONPATH=/app/aiter:/app/aiter/aiter/jit/utils \
    ENABLE_CK=0 \
    python /host/path/pa_asm_padded_dummy_query_scale_repro.py

The important condition is scheduled_bs < graph_bs. PA ASM has no work for the
padded row, so an uninitialized output row survives. ATOM then computes the
next PA ASM query scale with q.abs().max() over graph_bs rows, so the padded
row can poison real requests.
"""

import math

import torch

import aiter


def fill_padded_metadata(
    graph_bs: int,
    scheduled_bs: int,
    ctx_len: int,
    page_size: int,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    pages = math.ceil(ctx_len / page_size)
    context_lens[:scheduled_bs].fill_(ctx_len)
    context_lens[scheduled_bs:graph_bs].fill_(0)

    indptr = [0]
    for _ in range(scheduled_bs):
        indptr.append(indptr[-1] + pages)
    for _ in range(scheduled_bs, graph_bs):
        indptr.append(indptr[-1])

    kv_indptr[: graph_bs + 1].copy_(
        torch.tensor(indptr, dtype=torch.int32, device="cuda")
    )
    kv_indices[: scheduled_bs * pages].copy_(
        torch.arange(scheduled_bs * pages, dtype=torch.int32, device="cuda")
    )


def alloc_ps_metadata(graph_bs: int, max_q: int, kv_heads: int):
    keys = (
        "work_meta_data",
        "work_indptr",
        "work_info",
        "reduce_indptr",
        "reduce_final",
        "reduce_partial",
    )
    specs = aiter.get_ps_metadata_info_v1(
        graph_bs,
        kv_heads,
        max_q,
        qlen_granularity=max_q,
    )
    return {
        key: torch.empty(shape, dtype=dtype, device="cuda")
        for key, (shape, dtype) in zip(keys, specs)
    }


def refresh_ps_metadata(
    graph_bs: int,
    max_q: int,
    kv_heads: int,
    gqa: int,
    page_size: int,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    ps,
) -> None:
    aiter.get_ps_metadata_v1(
        cu_seqlens_q,
        kv_indptr[: graph_bs + 1],
        context_lens[:graph_bs],
        gqa,
        kv_heads,
        ps["work_meta_data"],
        ps["work_indptr"],
        ps["work_info"],
        ps["reduce_indptr"],
        ps["reduce_final"],
        ps["reduce_partial"],
        qhead_granularity=gqa,
        qlen_granularity=max_q,
        kvlen_granularity=page_size,
        block_size=page_size,
        is_causal=False,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")

    torch.cuda.set_device(0)
    torch.manual_seed(123)

    fp8 = aiter.dtypes.fp8
    graph_bs = 8
    scheduled_bs = 7
    ctx_len = 768
    page_size = 256
    max_q = 1
    kv_heads = 8
    gqa = 8
    heads = kv_heads * gqa
    head_dim = 64
    x = 16
    pages = math.ceil(ctx_len / page_size)
    num_pages = scheduled_bs * pages

    cu_seqlens_q = torch.arange(
        0, graph_bs + 1, dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.empty((graph_bs + 1,), dtype=torch.int32, device="cuda")
    kv_indices = torch.empty((num_pages,), dtype=torch.int32, device="cuda")
    context_lens = torch.empty((graph_bs,), dtype=torch.int32, device="cuda")

    q = (
        torch.randn(
            (graph_bs, max_q, kv_heads, gqa, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 8
    ).to(fp8)
    k = (
        torch.randn(
            (num_pages, kv_heads, head_dim // x, page_size, x),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 8
    ).to(fp8)
    v = (
        torch.randn(
            (num_pages, kv_heads, page_size // x, head_dim, x),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 8
    ).to(fp8)

    fill_padded_metadata(
        graph_bs,
        scheduled_bs,
        ctx_len,
        page_size,
        kv_indptr,
        kv_indices,
        context_lens,
    )
    ps = alloc_ps_metadata(graph_bs, max_q, kv_heads)
    refresh_ps_metadata(
        graph_bs,
        max_q,
        kv_heads,
        gqa,
        page_size,
        cu_seqlens_q,
        kv_indptr,
        context_lens,
        ps,
    )

    split_rows = int(ps["reduce_partial"].numel()) * max_q
    split_o = torch.empty(
        (split_rows, 1, heads, head_dim), dtype=torch.float32, device="cuda"
    )
    split_lse = torch.empty(
        (split_rows, 1, heads, 1), dtype=torch.float32, device="cuda"
    )
    final_lse = torch.empty((graph_bs * max_q, heads), dtype=torch.float32, device="cuda")
    sink = torch.zeros((heads,), dtype=torch.float32, device="cuda")
    q_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    sentinel = 1024.0
    out = torch.empty_like(q, dtype=torch.bfloat16)
    out.fill_(sentinel)
    split_o.zero_()
    split_lse.fill_(float("-inf"))

    aiter.pa_decode_bf16_asm(
        q,
        k,
        v,
        kv_indices,
        context_lens,
        1.0,
        kv_indptr,
        gqa=gqa,
        mtp=max_q - 1,
        query_scale=q_scale,
        key_scale=k_scale,
        value_scale=v_scale,
        qo_indptr=cu_seqlens_q,
        work_indptr=ps["work_indptr"],
        work_info=ps["work_info"],
        split_o=split_o,
        split_lse=split_lse,
        sink=sink,
        out=out,
    )
    aiter.pa_reduce_v1(
        split_o,
        split_lse,
        ps["reduce_indptr"],
        ps["reduce_final"],
        ps["reduce_partial"],
        max_q,
        out.view(graph_bs * max_q, heads, head_dim),
        final_lse,
    )
    torch.cuda.synchronize()

    real = out[:scheduled_bs]
    padded = out[scheduled_bs:]
    real_amax = real.abs().max()
    all_amax = out.abs().max()

    print(f"real_amax={real_amax.item():.6f}")
    print(f"all_amax_with_padded_row={all_amax.item():.6f}")
    print(f"padded_unique={padded.unique().detach().cpu().tolist()[:4]}")

    assert not torch.all(real == sentinel), "real rows were not written"
    assert torch.all(padded == sentinel), "padded row was unexpectedly written"
    assert all_amax > real_amax * 100, "padded row did not poison global q_amax"

    fixed = out.clone()
    fixed[scheduled_bs:].zero_()
    fixed_amax = fixed.abs().max()
    print(f"all_amax_after_zeroing_padded_row={fixed_amax.item():.6f}")
    assert torch.allclose(fixed_amax, real_amax)


if __name__ == "__main__":
    main()
