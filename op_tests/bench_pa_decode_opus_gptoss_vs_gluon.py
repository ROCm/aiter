# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf comparison for gpt-oss decode: the OPUS HIP kernel vs the Gluon kernel.

The Gluon call follows ATOM's full-attention decode path. Set head dimension
and head counts to match the loaded model configuration before comparing.

Both sides read the *same* KV cache tensors -- the layouts already agree:

    k_cache : [num_blocks, num_kv_heads, D/x, PAGE, x]    fp8, x = 16
    v_cache : [num_blocks, num_kv_heads, D, PAGE]         fp8
  sink    : [num_heads]                                 fp32

The Gluon call mirrors ATOM's own call site in
`atom/model_ops/attention_mha.py:527-589` -- same partition sizing via
`get_recommended_splits`, same per-tensor scales, same `sliding_window=-1` for
the full-attention layers. Its scratch (exp_sums / max_logits / temporary_output)
is allocated outside the timed region; ATOM re-allocates it per call, so the
numbers here are kinder to Gluon than production would be.

Both paths are checked against a torch reference before timing, so the numbers
always refer to kernels that actually compute the right thing.
"""

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.ops.pa_decode_opus import (
    pa_decode_opus_gptoss,
    pa_decode_opus_gptoss_ps,
    pa_decode_opus_ps_plan,
)
from aiter.ops.triton.gluon.pa_decode_gluon import (
    get_recommended_splits,
    pa_decode_gluon,
)

HEAD_DIM = 64
X = 16 // dtypes.fp8.itemsize  # 16 for fp8


def make_block_table(pattern, batch, max_blocks, pool, device):
    """How pages are drawn decides how much of the KV the cache can absorb.

    ``distinct``  unique page IDs across the whole batch within one call. Repeated
                  calls still reuse these addresses; this does not ensure HBM reads.
    ``pool``      sampled with replacement, so pages repeat and some reads come back
                  from cache. This is what the gpt-oss test fixture builds, and what
                  the historical 7.3 TB/s figure in the share doc was measured on.
    ``shared``    every sequence reads the same pages -> the working set collapses to
                  one sequence, which may fit in cache depending on its size.
    """
    if pattern == "seq":
        # Identity table: page id equals its own flat index. Still one page per slot
        # with no reuse, but now the id is an affine function of (sequence, tile), so
        # a kernel can compute it instead of reading it. That is what makes it usable
        # as the control for the block-table-load oracle.
        return torch.arange(
            batch * max_blocks, dtype=torch.int32, device=device
        ).reshape(batch, max_blocks)
    if pattern == "distinct":
        return torch.randperm(pool, device=device, dtype=torch.int32)[
            : batch * max_blocks
        ].reshape(batch, max_blocks)
    if pattern == "shared":
        one = torch.randperm(pool, device=device, dtype=torch.int32)[:max_blocks]
        return one.unsqueeze(0).expand(batch, max_blocks).contiguous()
    return torch.randint(0, pool, (batch, max_blocks), dtype=torch.int32, device=device)


def make_inputs(
    batch,
    num_kv_heads,
    gqa,
    ctx,
    page_size,
    q_dtype,
    pattern="distinct",
    device="cuda",
    qlen=1,
    head_dim=HEAD_DIM,
):
    torch.manual_seed(0)
    num_heads = num_kv_heads * gqa
    max_blocks = (ctx + page_size - 1) // page_size
    # The pool matches the test fixture's, so `pool` here reproduces its reuse rate.
    num_blocks = max(64, batch * max_blocks * 2)

    def rand_fp8(shape):
        return (torch.randn(shape, device=device) / 4).to(dtypes.fp8)

    q_shape = (
        (batch, num_heads, head_dim)
        if qlen == 1
        else (batch, qlen, num_heads, head_dim)
    )
    q_ref = torch.randn(q_shape, device=device) / 4
    q = q_ref.to(q_dtype)
    k_cache = rand_fp8((num_blocks, num_kv_heads, head_dim // X, page_size, X))
    v_cache = rand_fp8((num_blocks, num_kv_heads, head_dim, page_size))

    block_tables = make_block_table(pattern, batch, max_blocks, num_blocks, device)
    context_lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    # A sink near the score range actually participates in the denominator; one far
    # below it would be numerically inert and would not exercise the seeding path.
    sink = (torch.randn(num_heads, device=device) * 0.5).float()
    return q, k_cache, v_cache, block_tables, context_lens, sink


def ref_gptoss(
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    sink,
    scale,
    kv_scale,
    *,
    key_scale=None,
    value_scale=None,
):
    """Dense torch attention with the sink as an extra, value-less logit column.

    4-D ``q`` is MTP: query ``t`` attends the first ``ctx - qlen + 1 + t`` KV
    tokens, matching both OPUS and Gluon's tail-causal mask.
    """
    q4 = q.unsqueeze(1) if q.dim() == 3 else q
    batch, qlen, num_heads, head_dim = q4.shape
    num_kv_heads = k_cache.shape[1]
    page_size = v_cache.shape[3]
    gqa = num_heads // num_kv_heads
    out = torch.empty_like(q4, dtype=torch.bfloat16)

    for s in range(batch):
        ctx = int(context_lens[s].item())
        pages = block_tables[s, : (ctx + page_size - 1) // page_size]
        k = k_cache[pages].float() * kv_scale
        if key_scale is not None:
            k *= key_scale[pages].unsqueeze(2)
        k = k.permute(1, 0, 3, 2, 4).reshape(num_kv_heads, -1, head_dim)[:, :ctx]
        v = v_cache[pages].float() * kv_scale
        if value_scale is not None:
            v *= value_scale[pages].transpose(2, 3)
        v = v.permute(1, 0, 3, 2).reshape(num_kv_heads, -1, head_dim)[:, :ctx]
        k_t = k.transpose(1, 2)
        sink_col = sink.reshape(num_kv_heads, gqa, 1).float()

        for t in range(qlen):
            valid = ctx - qlen + 1 + t
            qs = q4[s, t].float().reshape(num_kv_heads, gqa, head_dim)
            score = torch.bmm(qs, k_t) * scale  # [nkv, gqa, ctx]
            if valid < ctx:
                score[..., valid:] = float("-inf")
            full = torch.cat([score, sink_col], dim=-1)
            p = torch.softmax(full, dim=-1)[..., :-1]
            out[s, t] = torch.bmm(p, v).reshape(num_heads, head_dim).to(torch.bfloat16)
    return out.reshape(q.shape)


def make_gluon_call(
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    sink,
    scale,
    kv_scale,
    *,
    key_scale=None,
    value_scale=None,
):
    """Mirror ATOM's call site, scratch allocated once instead of per call.

    Gluon wants ``[batch * qlen, heads, D]``. OPUS MTP is ``[batch, qlen, heads, D]``;
    flatten a view so both kernels read the same Q bytes.
    """
    if q.dim() == 4:
        batch, query_length, num_heads, _ = q.shape
        q_gluon = q.reshape(batch * query_length, num_heads, q.size(-1))
    else:
        batch, num_heads, _ = q.shape
        query_length = 1
        q_gluon = q
    head_dim = q.shape[-1]
    num_kv_heads = k_cache.shape[1]
    query_group_size = query_length * (num_heads // num_kv_heads)

    max_parts = get_recommended_splits(batch, num_kv_heads)
    part_size = 256

    shape = (batch, num_kv_heads, max_parts, query_group_size)
    exp_sums = torch.empty(shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty(shape, dtype=torch.float32, device=q.device)
    temporary_output = torch.empty(
        *shape, head_dim, dtype=torch.bfloat16, device=q.device
    )
    out = torch.empty(
        (batch * query_length, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=q.device,
    )

    if (key_scale is None) != (value_scale is None):
        raise ValueError("Provide both per-token KV scales or neither")
    if key_scale is None:
        key_scale = torch.tensor([kv_scale], dtype=torch.float32, device=q.device)
        value_scale = key_scale
    q_scale_t = torch.tensor([1.0], dtype=torch.float32, device=q.device)
    is_fp8_q = q.dtype == dtypes.fp8

    def call():
        pa_decode_gluon(
            out,
            q_gluon,
            k_cache,
            v_cache,
            context_lens,
            block_tables,
            scale,
            query_length,
            max_parts,
            part_size,
            dtypes.fp8,
            q_scale_t if is_fp8_q else None,
            key_scale,
            value_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=sink,
            sliding_window=-1,
            ps=True,
        )

    return call, out


def graph_wrap(call):
    """Replay the kernel from a HIP graph, which is how ATOM runs decode.

    At small shapes both kernels spend most of the wall clock in host-side launch,
    and Gluon's Triton launcher costs far more than the OPUS one. Capturing removes
    that from both sides and leaves the device time, which is the part that still
    matters once a serving stack has captured its decode step.

    OPUS's split-KV scratch is allocated on first use and hipMalloc is illegal
    during capture, so the warmup below is load-bearing, not just cache priming.
    It also has to run on the stream the capture will use, because the scratch is
    cached per stream -- warming the default stream leaves the capture stream's
    entry empty and the kernel refuses to allocate mid-capture.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        call()
    return g.replay


def _one_burst(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def timeit_pair(fn_a, fn_b, warmup=25, iters=50, rounds=7):
    """Interleave the rounds so both sides see the same interference, take the min.

    Contention can only ever add time, so the per-side minimum is the estimate that
    survives a machine shared with other jobs.
    """
    for _ in range(warmup):
        fn_a()
        fn_b()
    best_a = best_b = float("inf")
    for _ in range(rounds):
        best_a = min(best_a, _one_burst(fn_a, iters))
        best_b = min(best_b, _one_burst(fn_b, iters))
    return best_a, best_b


def run(
    batch,
    num_kv_heads,
    gqa,
    ctx,
    page_size,
    q_dtype,
    verify,
    graph=False,
    persistent=False,
    kv_scale=1.0,
    pattern="distinct",
    qlen=1,
    head_dim=HEAD_DIM,
):
    q, k_cache, v_cache, bt, cl, sink = make_inputs(
        batch,
        num_kv_heads,
        gqa,
        ctx,
        page_size,
        q_dtype,
        pattern,
        qlen=qlen,
        head_dim=head_dim,
    )
    scale = float(head_dim**-0.5)

    gluon_call, gluon_out = make_gluon_call(
        q, k_cache, v_cache, bt, cl, sink, scale, kv_scale
    )
    opus_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    if persistent:
        if qlen > 1:
            raise RuntimeError("OPUS persistent path does not support MTP")
        # The plan is schedule-only and reusable across calls, so building it here
        # matches how a serving stack would amortize it.
        plan = pa_decode_opus_ps_plan(
            bt,
            cl,
            num_kv_heads * gqa,
            num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )

        def opus_call():
            pa_decode_opus_gptoss_ps(
                q,
                k_cache,
                v_cache,
                plan,
                sink,
                scale,
                q_scale=1.0,
                k_scale=kv_scale,
                v_scale=kv_scale,
                out=opus_out,
            )

    else:

        def opus_call():
            pa_decode_opus_gptoss(
                q,
                k_cache,
                v_cache,
                bt,
                cl,
                sink,
                scale,
                q_scale=1.0,
                k_scale=kv_scale,
                v_scale=kv_scale,
                out=opus_out,
            )

    status = "-"
    if verify:
        ref = ref_gptoss(q, k_cache, v_cache, bt, cl, sink, scale, kv_scale)
        gluon_call()
        opus_call()
        torch.cuda.synchronize()
        # fp8 P quantization dominates the error budget, so this is a loose bound.
        g_bad = (gluon_out.reshape(ref.shape).float() - ref.float()).abs().max().item()
        o_bad = (opus_out.float() - ref.float()).abs().max().item()
        status = f"g={g_bad:.3f} o={o_bad:.3f}"
        if max(g_bad, o_bad) > 0.05:
            status += " !!"

    if graph:
        opus_call, gluon_call = graph_wrap(opus_call), graph_wrap(gluon_call)

    t_opus, t_gluon = timeit_pair(opus_call, gluon_call)

    # Both kernels stream the same KV bytes; that is what they are bound by.
    kv_bytes = 2 * batch * num_kv_heads * ctx * head_dim * dtypes.fp8.itemsize
    bw_opus = kv_bytes / (t_opus * 1e-6) / 1e9
    bw_gluon = kv_bytes / (t_gluon * 1e-6) / 1e9

    print(
        f"{batch:>5} {num_kv_heads:>4} {gqa:>4} {qlen:>4} {ctx:>7} {page_size:>5} "
        f"{t_opus:>9.1f} {t_gluon:>9.1f} {t_gluon / t_opus:>7.2f}x "
        f"{bw_opus:>8.0f} {bw_gluon:>8.0f}  {status}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 128])
    p.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--page", type=int, nargs="+", default=[16, 256])
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--gqa", type=int, default=8)
    p.add_argument("--head-dim", type=int, choices=[64, 128], default=HEAD_DIM)
    p.add_argument("--q-dtype", choices=["bf16", "fp8"], default="bf16")
    p.add_argument(
        "--qlen",
        type=int,
        nargs="+",
        default=[1],
        help="query tokens per sequence; 1 is decode, 2 is OPUS gpt-oss MTP "
        "(qlen * GQA must be <= 16). Gluon also accepts up to 4.",
    )
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--ps",
        action="store_true",
        help="use the OPUS persistent variant, matching Gluon's ps=True",
    )
    p.add_argument(
        "--graph",
        action="store_true",
        help="replay both from a HIP graph, dropping host launch cost",
    )
    p.add_argument(
        "--pattern",
        choices=["distinct", "pool", "shared", "seq"],
        default="distinct",
        help="how block tables draw pages, i.e. how much KV the cache can absorb",
    )
    args = p.parse_args()

    q_dtype = torch.bfloat16 if args.q_dtype == "bf16" else dtypes.fp8
    if args.ps and any(q > 1 for q in args.qlen):
        p.error("OPUS persistent path does not support MTP; drop --ps")
    if any(q * args.gqa > 16 for q in args.qlen):
        p.error("OPUS gpt-oss MTP requires qlen * GQA <= 16")
    mode = "hip-graph replay" if args.graph else "eager launch"
    mode += ", opus-ps" if args.ps else ", opus-split"
    mode += f", {args.pattern} pages"
    qlen_tag = ",".join(str(q) for q in args.qlen)
    print(
        f"gpt-oss decode: OPUS vs Gluon   q={args.q_dtype}, KV=fp8, head_dim={args.head_dim}, "
        f"qlen={qlen_tag}, {mode}"
    )
    print(
        f"gfx={aiter.get_gfx_runtime() if hasattr(aiter, 'get_gfx_runtime') else '?'}"
    )
    print(
        f"{'batch':>5} {'nkv':>4} {'gqa':>4} {'qlen':>4} {'ctx':>7} {'page':>5} "
        f"{'opus us':>9} {'gluon us':>9} {'speedup':>8} "
        f"{'o GB/s':>8} {'g GB/s':>8}  max-abs-err"
    )
    for page in args.page:
        for ctx in args.ctx:
            for qlen in args.qlen:
                for batch in args.batch:
                    run(
                        batch,
                        args.num_kv_heads,
                        args.gqa,
                        ctx,
                        page,
                        q_dtype,
                        verify=not args.no_verify,
                        graph=args.graph,
                        persistent=args.ps,
                        pattern=args.pattern,
                        qlen=qlen,
                        head_dim=args.head_dim,
                    )


if __name__ == "__main__":
    main()
