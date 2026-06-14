"""Minimal PA BF16 ASM CUDAGraph replay metadata repro.

This is intentionally a failing regression test for the current bug:
CUDAGraph capture records PA ASM with PS metadata generated for the capture
batch. Replay mutates context_lens / kv_indptr / kv_indices, but the captured
work_info / reduce metadata does not get regenerated. The replay graph then
does not match an eager launch with fresh metadata for the replay context.

Run on gfx1250:

    HIP_VISIBLE_DEVICES=3 PYTHONPATH=/app/aiter:/app/aiter/aiter/jit/utils \
      python op_tests/test_pa_decode_bf16_asm_cudagraph_metadata.py
"""

import math
import os

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx


def _fill_metadata_inputs(bs, ctx_len, page, kv_indptr, kv_indices, context_lens):
    pages = math.ceil(ctx_len / page)
    context_lens.fill_(ctx_len)
    kv_indptr[: bs + 1].copy_(
        torch.arange(0, (bs + 1) * pages, pages, dtype=torch.int32, device="cuda")
    )
    kv_indices[: bs * pages].copy_(torch.arange(bs * pages, dtype=torch.int32, device="cuda"))
    return pages


def _alloc_ps_metadata(bs, max_q, kv_heads):
    def alloc(spec):
        shape, dtype = spec
        return torch.empty(shape, dtype=dtype, device="cuda")

    specs = aiter.get_ps_metadata_info_v1(
        bs,
        kv_heads,
        max_q,
        qlen_granularity=max_q,
    )
    work_meta, work_indptr, work_info, reduce_indptr, reduce_final, reduce_partial = [
        alloc(s) for s in specs
    ]
    return {
        "work_meta_data": work_meta,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final": reduce_final,
        "reduce_partial": reduce_partial,
    }


def _refresh_ps_metadata(bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens, ps):
    aiter.get_ps_metadata_v1(
        cu,
        kv_indptr[: bs + 1],
        context_lens[:bs],
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
        kvlen_granularity=page,
        block_size=page,
        is_causal=False,
    )
    return ps


def _build_ps_metadata(bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens):
    ps = _alloc_ps_metadata(bs, max_q, kv_heads)
    return _refresh_ps_metadata(
        bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens, ps
    )


def _run_pa_decode(
    q,
    k,
    v,
    kv_indices,
    context_lens,
    kv_indptr,
    cu,
    ps,
    sink,
    out,
    split_o,
    split_lse,
    q_scale,
    k_scale,
    v_scale,
    reduce_output,
    final_lse=None,
):
    bs, max_q, kv_heads, gqa, head_dim = q.shape
    heads = kv_heads * gqa

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
        qo_indptr=cu,
        work_indptr=ps["work_indptr"],
        work_info=ps["work_info"],
        split_o=split_o,
        split_lse=split_lse,
        sink=sink,
        out=out,
    )
    if reduce_output:
        assert final_lse is not None
        aiter.pa_reduce_v1(
            split_o,
            split_lse,
            ps["reduce_indptr"],
            ps["reduce_final"],
            ps["reduce_partial"],
            max_q,
            out.view(bs * max_q, heads, head_dim),
            final_lse,
        )
    return out


def _diff_stats(actual, expected, label):
    diff = actual.float() - expected.float()
    finite = torch.isfinite(actual)
    expected_finite = torch.isfinite(expected)
    finite_diff = torch.where(torch.isfinite(diff), diff.abs(), torch.zeros_like(diff))
    max_abs = finite_diff.max().item()
    mean_abs = finite_diff.mean().item()
    nan_count = torch.isnan(actual).sum().item()
    inf_count = torch.isinf(actual).sum().item()
    expected_nan_count = torch.isnan(expected).sum().item()
    expected_inf_count = torch.isinf(expected).sum().item()
    print(
        f"{label}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"actual_finite={finite.sum().item()}/{actual.numel()} "
        f"expected_finite={expected_finite.sum().item()}/{expected.numel()} "
        f"actual_nan={nan_count} actual_inf={inf_count} "
        f"expected_nan={expected_nan_count} expected_inf={expected_inf_count}"
    )


def main():
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        print("SKIP: pa_decode_bf16_asm repro requires gfx1250")
        return

    torch.manual_seed(123)
    torch.cuda.set_device(0)
    fp8 = aiter.dtypes.fp8

    bs = int(os.getenv("PA_REPRO_BS", "1"))
    capture_ctx = int(os.getenv("PA_REPRO_CAPTURE_CTX", "1"))
    replay_ctx = int(os.getenv("PA_REPRO_REPLAY_CTX", "512"))
    page = 256
    max_q = 1
    kv_heads = 8
    gqa = 8
    heads = kv_heads * gqa
    head_dim = 64
    x = 16
    replay_pages = math.ceil(replay_ctx / page)
    total_pages = bs * replay_pages

    cu = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.empty((bs + 1,), dtype=torch.int32, device="cuda")
    kv_indices = torch.empty((total_pages,), dtype=torch.int32, device="cuda")
    context_lens = torch.empty((bs,), dtype=torch.int32, device="cuda")

    q = (
        torch.randn(
            (bs, max_q, kv_heads, gqa, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 8
    ).to(fp8).contiguous()
    k = (
        torch.randn(
            (total_pages, kv_heads, head_dim // x, page, x),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 8
    ).to(fp8).contiguous()
    v = (
        torch.randn(
            (total_pages, kv_heads, page // x, head_dim, x),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 8
    ).to(fp8).contiguous()
    sink = torch.randn((heads,), dtype=torch.float32, device="cuda")
    graph_out = torch.empty_like(q, dtype=torch.bfloat16)
    capture_kernel_out = torch.empty_like(q, dtype=torch.bfloat16)
    eager_out = torch.empty_like(q, dtype=torch.bfloat16)
    split_rows = bs * replay_pages * max_q
    graph_split_o = torch.empty(
        (split_rows, 1, heads, head_dim), dtype=torch.float32, device="cuda"
    )
    graph_split_lse = torch.empty(
        (split_rows, 1, heads, 1), dtype=torch.float32, device="cuda"
    )
    eager_split_o = torch.empty_like(graph_split_o)
    eager_split_lse = torch.empty_like(graph_split_lse)
    final_lse = torch.empty((bs * max_q, heads), dtype=torch.float32, device="cuda")
    q_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    _fill_metadata_inputs(bs, replay_ctx, page, kv_indptr, kv_indices, context_lens)
    replay_ps = _build_ps_metadata(
        bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens
    )
    control_graph_out = torch.empty_like(q, dtype=torch.bfloat16)
    control_kernel_out = torch.empty_like(q, dtype=torch.bfloat16)
    control_eager_out = torch.empty_like(q, dtype=torch.bfloat16)
    control_split_o = torch.empty_like(graph_split_o)
    control_split_lse = torch.empty_like(graph_split_lse)
    control_eager_split_o = torch.empty_like(graph_split_o)
    control_eager_split_lse = torch.empty_like(graph_split_lse)
    control_final_lse = torch.empty_like(final_lse)
    control_eager_final_lse = torch.empty_like(final_lse)

    _run_pa_decode(
        q,
        k,
        v,
        kv_indices,
        context_lens,
        kv_indptr,
        cu,
        replay_ps,
        sink,
        control_eager_out,
        control_eager_split_o,
        control_eager_split_lse,
        q_scale,
        k_scale,
        v_scale,
        reduce_output=True,
        final_lse=control_eager_final_lse,
    )

    def control_graph_launch():
        control_graph_out.copy_(
            _run_pa_decode(
                q,
                k,
                v,
                kv_indices,
                context_lens,
                kv_indptr,
                cu,
                replay_ps,
                sink,
                control_kernel_out,
                control_split_o,
                control_split_lse,
                q_scale,
                k_scale,
                v_scale,
                reduce_output=True,
                final_lse=control_final_lse,
            )
        )

    for _ in range(3):
        control_graph_launch()
    torch.cuda.synchronize()

    control_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(control_graph):
        control_graph_launch()
    torch.cuda.synchronize()
    control_graph.replay()
    torch.cuda.synchronize()
    _diff_stats(control_graph_out, control_eager_out, "fresh-metadata graph vs eager")
    assert torch.allclose(control_graph_out, control_eager_out, atol=1e-2, rtol=1e-2), (
        "Control failed: PA ASM CUDAGraph with fresh static metadata should match eager."
    )

    _fill_metadata_inputs(bs, capture_ctx, page, kv_indptr, kv_indices, context_lens)
    external_refresh_ps = _build_ps_metadata(
        bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens
    )
    external_refresh_graph_out = torch.empty_like(q, dtype=torch.bfloat16)
    external_refresh_kernel_out = torch.empty_like(q, dtype=torch.bfloat16)
    external_refresh_split_o = torch.empty_like(graph_split_o)
    external_refresh_split_lse = torch.empty_like(graph_split_lse)
    external_refresh_final_lse = torch.empty_like(final_lse)

    def external_refresh_graph_launch():
        external_refresh_graph_out.copy_(
            _run_pa_decode(
                q,
                k,
                v,
                kv_indices,
                context_lens,
                kv_indptr,
                cu,
                external_refresh_ps,
                sink,
                external_refresh_kernel_out,
                external_refresh_split_o,
                external_refresh_split_lse,
                q_scale,
                k_scale,
                v_scale,
                reduce_output=True,
                final_lse=external_refresh_final_lse,
            )
        )

    for _ in range(3):
        external_refresh_graph_launch()
    torch.cuda.synchronize()

    external_refresh_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(external_refresh_graph):
        external_refresh_graph_launch()
    torch.cuda.synchronize()

    _fill_metadata_inputs(bs, replay_ctx, page, kv_indptr, kv_indices, context_lens)
    _refresh_ps_metadata(
        bs,
        max_q,
        kv_heads,
        gqa,
        page,
        cu,
        kv_indptr,
        context_lens,
        external_refresh_ps,
    )
    torch.cuda.synchronize()
    external_refresh_graph.replay()
    torch.cuda.synchronize()
    _diff_stats(
        external_refresh_graph_out,
        control_eager_out,
        "external-refresh graph replay vs eager",
    )
    assert torch.allclose(
        external_refresh_graph_out, control_eager_out, atol=1e-2, rtol=1e-2
    ), (
        "External-refresh graph failed: replay should match eager when the "
        "captured metadata buffers are refreshed before graph replay and reduce "
        "is present in the captured graph."
    )

    if os.getenv("PA_REPRO_TRY_DYNAMIC_METADATA", "0") == "1":
        _fill_metadata_inputs(bs, capture_ctx, page, kv_indptr, kv_indices, context_lens)
        dynamic_ps = _build_ps_metadata(
            bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens
        )
        dynamic_graph_out = torch.empty_like(q, dtype=torch.bfloat16)
        dynamic_kernel_out = torch.empty_like(q, dtype=torch.bfloat16)
        dynamic_split_o = torch.empty_like(graph_split_o)
        dynamic_split_lse = torch.empty_like(graph_split_lse)
        dynamic_final_lse = torch.empty_like(final_lse)

        def dynamic_metadata_graph_launch():
            _refresh_ps_metadata(
                bs,
                max_q,
                kv_heads,
                gqa,
                page,
                cu,
                kv_indptr,
                context_lens,
                dynamic_ps,
            )
            dynamic_graph_out.copy_(
                _run_pa_decode(
                    q,
                    k,
                    v,
                    kv_indices,
                    context_lens,
                    kv_indptr,
                    cu,
                    dynamic_ps,
                    sink,
                    dynamic_kernel_out,
                    dynamic_split_o,
                    dynamic_split_lse,
                    q_scale,
                    k_scale,
                    v_scale,
                    reduce_output=True,
                    final_lse=dynamic_final_lse,
                )
            )

        for _ in range(3):
            dynamic_metadata_graph_launch()
        torch.cuda.synchronize()

        dynamic_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(dynamic_graph):
            dynamic_metadata_graph_launch()
        torch.cuda.synchronize()

        _fill_metadata_inputs(bs, replay_ctx, page, kv_indptr, kv_indices, context_lens)
        dynamic_graph.replay()
        torch.cuda.synchronize()
        _diff_stats(
            dynamic_graph_out,
            control_eager_out,
            "dynamic-metadata graph replay vs eager",
        )
        assert torch.allclose(
            dynamic_graph_out, control_eager_out, atol=1e-2, rtol=1e-2
        ), (
            "Dynamic-metadata graph failed: replay should match eager when metadata "
            "generation and reduce are recorded in the graph."
        )

    _fill_metadata_inputs(bs, capture_ctx, page, kv_indptr, kv_indices, context_lens)
    capture_ps = _build_ps_metadata(
        bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens
    )

    def graph_launch():
        # Mimics ATOM capture: uses capture_ps and no reduce, because capture_ctx
        # is one page. Replay mutates only context_lens / kv_indptr / kv_indices.
        graph_out.copy_(
            _run_pa_decode(
                q,
                k,
                v,
                kv_indices,
                context_lens,
                kv_indptr,
                cu,
                capture_ps,
                sink,
                capture_kernel_out,
                graph_split_o,
                graph_split_lse,
                q_scale,
                k_scale,
                v_scale,
                reduce_output=False,
            )
        )

    for _ in range(3):
        graph_launch()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_launch()
    torch.cuda.synchronize()

    _fill_metadata_inputs(bs, replay_ctx, page, kv_indptr, kv_indices, context_lens)
    graph.replay()
    torch.cuda.synchronize()
    stale_graph_out = graph_out.clone()

    fresh_ps = _build_ps_metadata(
        bs, max_q, kv_heads, gqa, page, cu, kv_indptr, context_lens
    )
    eager_out = _run_pa_decode(
        q,
        k,
        v,
        kv_indices,
        context_lens,
        kv_indptr,
        cu,
        fresh_ps,
        sink,
        eager_out,
        eager_split_o,
        eager_split_lse,
        q_scale,
        k_scale,
        v_scale,
        reduce_output=True,
        final_lse=final_lse,
    )
    torch.cuda.synchronize()

    max_abs = (stale_graph_out.float() - eager_out.float()).abs().max().item()
    mean_abs = (stale_graph_out.float() - eager_out.float()).abs().mean().item()
    print(
        f"capture_ctx={capture_ctx} replay_ctx={replay_ctx} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
    )
    _diff_stats(stale_graph_out, eager_out, "stale-capture graph vs fresh eager")

    assert torch.allclose(stale_graph_out, eager_out, atol=1e-2, rtol=1e-2), (
        "PA ASM CUDAGraph replay used stale capture PS metadata. "
        "Replay output differs from eager PA ASM with fresh replay metadata."
    )


if __name__ == "__main__":
    main()
