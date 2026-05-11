# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Side-by-side example: feed the SAME logical (Q, K, V) batch -- in a true
*varlen + paged* layout -- to

    1. CK Unified Attention -> aiter.ops.unified_attention.unified_attention_fwd
    2. CK FMHA varlen       -> aiter.ops.mha.mha_varlen_fwd
    3. (when supported)     CK FMHA non-split paged
                            -> aiter.ops.mha.mha_batch_prefill_func

Unlike compare_unified_vs_fmha_signatures.py (which used uniform seqlens and
called the simpler 4D-batched mha_fwd path), this script uses non-uniform
per-sequence Q/KV lengths and a paged KV cache, so both kernels are invoked
on their proper "sibling" entry point. That makes the remaining differences
purely about API surface (mask_type vs. is_causal/window/sink, FP8
scale_s/scale_k/... vs. q_descale/k_descale/..., return convention, etc.)
rather than "varlen vs. batched" or "paged vs. contiguous".

We run TWO problem configurations:

    A) (GQA-8, hdim=64, page_blk=64)   -- UA's d64 GQA-8 sweet spot
       UA works; mha_varlen_fwd (split-KV path) works; mha_batch_prefill_func
       has no compiled instance and fails at runtime.

    B) (MHA,   hdim=128, page_blk=32)  -- UA's d128 MHA sweet spot
       UA works; mha_batch_prefill_func works; mha_varlen_fwd may also work.
       This is the only configuration in this build where we can compare
       UA's single-pass paged kernel against an FMHA single-pass paged
       kernel (mha_batch_prefill_func) on the same problem.

Run with:

    python op_tests/compare_unified_vs_mha_varlen_signatures.py
"""

import torch

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.mha import mha_varlen_fwd, mha_batch_prefill_func


def run_config(label, query_lens, kv_lens, num_kv_heads, num_q_per_kv,
               head_size, page_blk_size):
    print()
    print("#" * 72)
    print(f"# CONFIG: {label}")
    print(f"#   head_size={head_size}  num_kv_heads={num_kv_heads}  "
          f"num_q_per_kv={num_q_per_kv}  page_blk_size={page_blk_size}")
    print("#" * 72)

    device = "cuda"
    dtype = torch.bfloat16

    batch          = len(query_lens)
    num_q_heads    = num_kv_heads * num_q_per_kv
    scale_s        = head_size ** -0.5

    total_q_tokens = sum(query_lens)
    max_seqlen_q   = max(query_lens)
    max_seqlen_k   = max(kv_lens)

    # ---------------------------------------------------------------
    # 2. Build shared varlen tensors:
    #      - q_packed   : [total_q_tokens, num_q_heads, head_size]
    #      - key_cache  : [num_pages,      page_blk_size, num_kv_heads, head_size]
    #      - value_cache: same shape as key_cache
    #      - block_tables : [batch, max_pages_per_seq]
    #      - cu_seqlens_q : [batch+1]
    #      - cu_seqlens_k : [batch+1]
    #      - seq_lens     : [batch]   (UA only; FMHA derives it from cu_seqlens_k)
    # ---------------------------------------------------------------
    torch.manual_seed(0)

    q_packed = torch.randn(total_q_tokens, num_q_heads, head_size,
                           dtype=dtype, device=device)
    out_ua   = torch.empty_like(q_packed)

    pages_per_seq    = [(kl + page_blk_size - 1) // page_blk_size for kl in kv_lens]
    max_pages_per_seq = max(pages_per_seq)
    total_pages      = sum(pages_per_seq)

    key_cache   = torch.zeros(total_pages, page_blk_size, num_kv_heads,
                              head_size, dtype=dtype, device=device)
    value_cache = torch.zeros_like(key_cache)

    # Random (non-identity) physical-page permutation, to exercise the
    # block_tables indirection on both kernels.
    perm = torch.randperm(total_pages, device=device, dtype=torch.int64).to(torch.int32)
    block_tables = torch.zeros(batch, max_pages_per_seq,
                               dtype=torch.int32, device=device)
    cursor = 0
    for b, n_pages in enumerate(pages_per_seq):
        block_tables[b, :n_pages] = perm[cursor:cursor + n_pages]
        cursor += n_pages

    # Fill paged buffers with random K/V, indexed via block_tables, so both
    # kernels read the identical KV bytes.
    for b, kl in enumerate(kv_lens):
        kb = torch.randn(kl, num_kv_heads, head_size, dtype=dtype, device=device)
        vb = torch.randn(kl, num_kv_heads, head_size, dtype=dtype, device=device)
        for p in range(pages_per_seq[b]):
            phys = int(block_tables[b, p].item())
            tok_start = p * page_blk_size
            tok_end   = min(tok_start + page_blk_size, kl)
            n = tok_end - tok_start
            key_cache[phys, :n]   = kb[tok_start:tok_end]
            value_cache[phys, :n] = vb[tok_start:tok_end]

    cu_seqlens_q = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0).tolist()),
                                dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0).tolist()),
                                dtype=torch.int32, device=device)
    seq_lens_t   = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # ---------------------------------------------------------------
    # 3. CK Unified Attention call
    #    output is written in-place; nothing returned.
    # ---------------------------------------------------------------
    unified_attention_fwd(
        out_ua,                  # [total_q_tokens, num_q_heads, head_size]
        q_packed,                # same shape as output
        key_cache,               # [num_pages, page_blk_size, num_kv_heads, head_size]
        value_cache,
        block_tables,            # [batch, max_pages_per_seq]
        seq_lens_t,              # [batch]    full context len per seq
        cu_seqlens_q,            # [batch+1]  cumulative query tokens
        mask_type=2,             # 0 = no mask, 2 = causal  (only options)
        scale_s=scale_s,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,  # FP8 quant scales
    )

    # ---------------------------------------------------------------
    # 4. CK FMHA mha_varlen_fwd call (varlen + paged)
    #    Returns (out, lse, P, rng_state).
    #
    #    NOTE: we set return_softmax_lse=True purely for kernel-dispatch
    #    reasons. mha_varlen_fwd splits its JIT-compiled kernels across
    #    different .so files per flag combo, and in this build the
    #    `_nlse_pagedkv_` variant was generated with no splitkv main
    #    kernels (only the combine half), so paged dispatch with
    #    return_softmax_lse=False fails with "invalid argument for
    #    fmha_fwd_splitkv". The `_lse_pagedkv_` build has the kernels.
    #    UA has no LSE output of its own, so we just discard `lse` here.
    # ---------------------------------------------------------------
    varlen_ok = True
    out_fmha = None
    try:
        out_fmha, lse, _, _ = mha_varlen_fwd(
            q_packed,                # [total_q_tokens, num_q_heads, head_size]
            key_cache,               # paged KV: [num_pages, page_blk_size, num_kv_heads, head_size]
            value_cache,
            cu_seqlens_q,            # [batch+1]
            cu_seqlens_k,            # [batch+1]  same KV-length info as UA's
                                     #            seq_lens, just cumulative:
                                     #            seq_lens[b] == cu_seqlens_k[b+1]
                                     #                          - cu_seqlens_k[b]
                                     #            UA doesn't take this because UA's
                                     #            KV is always paged, so a cumulative
                                     #            offset into a flat KV buffer would
                                     #            be meaningless -- block_tables[b]
                                     #            + seq_lens[b] is the natural form.
            max_seqlen_q,            # int        (UA infers from query_start_len)
            max_seqlen_k,            # int
            0,                       # min_seqlen_q     (FMHA-only knob)
            0.0,                     # dropout_p        (not in unified attention)
            scale_s,                 # softmax_scale
            0.0,                     # logits_soft_cap  (not in unified attention)
            False,                   # zero_tensors
            True,                    # is_causal        (vs. unified's mask_type=2)
            -1,                      # window_size_left (sliding window, not in UA)
            -1,                      # window_size_right
            0,                       # sink_size        (not in unified attention)
            True,                    # return_softmax_lse  -- see note below; UA never returns LSE
            False,                   # return_dropout_randval
            out=None,
            block_table=block_tables,    # SAME paged block-table as UA
            bias=None,                   # custom bias       (not in UA)
            alibi_slopes=None,           # ALiBi             (not in UA)
            q_descale=None,              # FP8 descale       (UA: scale_s/scale_k/scale_v/scale_out)
            k_descale=None,
            v_descale=None,
            gen=None,
            cu_seqlens_q_padded=None,
            cu_seqlens_k_padded=None,
            sink_ptr=None,
        )
        torch.cuda.synchronize()
    except Exception as e:                                              # noqa: BLE001
        varlen_ok = False
        print(f"  [info] mha_varlen_fwd unavailable at this shape: "
              f"{type(e).__name__}: {str(e).splitlines()[0]}")
        print(f"         skipping the UA-vs-mha_varlen_fwd comparison/bench;")
        print(f"         will still run UA + mha_batch_prefill_func below.")

    # ---------------------------------------------------------------
    # 5. Report
    # ---------------------------------------------------------------
    print("=" * 72)
    print("INPUTS (varlen + paged, shared by both kernels)")
    print("=" * 72)
    print(f"  batch={batch}  query_lens={query_lens}  kv_lens={kv_lens}")
    print(f"  total_q_tokens={total_q_tokens}  max_seqlen_q={max_seqlen_q}  "
          f"max_seqlen_k={max_seqlen_k}")
    print(f"  num_q_heads={num_q_heads}  num_kv_heads={num_kv_heads}  "
          f"num_q_per_kv={num_q_per_kv}  head_size={head_size}")
    print(f"  page_blk_size={page_blk_size}  total_pages={total_pages}  "
          f"max_pages_per_seq={max_pages_per_seq}")
    print()
    print(f"  shared tensors:")
    print(f"    q_packed     {tuple(q_packed.shape)}      {q_packed.dtype}")
    print(f"    key_cache    {tuple(key_cache.shape)}  paged")
    print(f"    block_tables {tuple(block_tables.shape)}  {block_tables.dtype}")
    print(f"    cu_seqlens_q {cu_seqlens_q.tolist()}")
    print(f"    cu_seqlens_k {cu_seqlens_k.tolist()}")
    print(f"    seq_lens     {seq_lens_t.tolist()}     (UA-only; FMHA derives from cu_seqlens_k)")
    print()

    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)
    print(f"  unified out  {tuple(out_ua.shape)}  {out_ua.dtype}  "
          f"abs_mean={out_ua.abs().mean().item():.6f}")
    if varlen_ok:
        print(f"  fmha    out  {tuple(out_fmha.shape)}  {out_fmha.dtype}  "
              f"abs_mean={out_fmha.abs().mean().item():.6f}")
        diff = (out_ua.float() - out_fmha.float()).abs()
        print(f"  max abs diff  : {diff.max().item():.4e}")
        print(f"  mean abs diff : {diff.mean().item():.4e}")

        # ---------------------------------------------------------------
        # 6. Correctness check (UA vs mha_varlen_fwd)
        #
        # bf16 has a 7-bit mantissa, so the per-element ULP at magnitude m
        # is ~m * 2^-7. Our outputs sit at O(0.1), so a single-ULP
        # disagreement is ~1e-3 and worst-case (one tile boundary) on
        # values near 1.0 is ~7.8e-3. atol=2e-2 leaves comfortable headroom
        # for tile/split-K reduction-order jitter while still catching
        # real divergence.
        # ---------------------------------------------------------------
        atol, rtol = 2e-2, 1e-2
        ok = torch.allclose(out_ua.float(), out_fmha.float(), atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print()
        print(f"  UA vs mha_varlen_fwd  agreement (atol={atol}, rtol={rtol}): {status}")
        if not ok:
            idx = int(diff.flatten().argmax().item())
            coord = torch.unravel_index(torch.tensor(idx), out_ua.shape)
            coord = tuple(int(c.item()) for c in coord)
            print(f"    worst element @ {coord}: "
                  f"ua={out_ua[coord].item():.6f}  fmha={out_fmha[coord].item():.6f}")
    else:
        print(f"  (mha_varlen_fwd output comparison skipped, kernel unavailable)")

    # ---------------------------------------------------------------
    # 7. Benchmarking
    #
    # We time the kernels in isolation using CUDA events. Warmup first so
    # JIT autotune / kernel caches are settled; then take the median over
    # `iters` runs to reduce jitter from other GPU activity.
    # ---------------------------------------------------------------
    warmup, iters = 10, 100

    def bench_ua():
        unified_attention_fwd(
            out_ua, q_packed, key_cache, value_cache,
            block_tables, seq_lens_t, cu_seqlens_q,
            mask_type=2, scale_s=scale_s,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )

    # Reuse one preallocated output for FMHA so we time the kernel only,
    # not the per-iter empty() allocation. mha_varlen_fwd writes into `out`
    # when given.
    out_fmha_buf = torch.empty_like(q_packed)

    def bench_fmha():
        mha_varlen_fwd(
            q_packed, key_cache, value_cache,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, 0,
            0.0, scale_s, 0.0, False,
            True, -1, -1, 0,
            True, False,                # return_softmax_lse=True (dispatch reasons; see above)
            out=out_fmha_buf,
            block_table=block_tables,
        )

    def time_fn(fn, warmup, iters):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return {
            "median": ms[iters // 2],
            "p10":    ms[max(0, iters // 10)],
            "mean":   sum(ms) / iters,
        }

    def time_graph(fn, warmup, iters):
        """CUDA-graph-mode timing.

        Captures `fn` once into a CUDA graph, then times pure replay. This
        eliminates Python/torch dispatch overhead entirely, so we measure
        only on-GPU kernel + DMA cost. The relative ratio between kernels
        should stay similar to time_fn, but absolute numbers shrink.

        We capture once outside the timing loop and time `g.replay()`,
        which submits the prerecorded launch list to the GPU.
        """
        # Warmup eager (also lets caching allocator settle for any tensors
        # the wrapper allocates per-call, so they get reused at capture).
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        try:
            g = torch.cuda.CUDAGraph()
            # Capture into a fresh stream as required by torch.cuda.graph.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # extra warmups on capture stream so allocator is warm there
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                fn()
        except Exception as e:                                  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}

        # Time pure replay
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            g.replay()
            ends[i].record()
        torch.cuda.synchronize()
        ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return {
            "median": ms[iters // 2],
            "p10":    ms[max(0, iters // 10)],
            "mean":   sum(ms) / iters,
        }

    ua_t = time_fn(bench_ua, warmup, iters)
    ua_g = time_graph(bench_ua, warmup, iters)

    fmha_t = time_fn(bench_fmha, warmup, iters) if varlen_ok else {"error": "kernel unavailable"}
    fmha_g = time_graph(bench_fmha, warmup, iters) if varlen_ok else {"error": "kernel unavailable"}

    def _row(name, t):
        if "error" in t:
            return f"  {name:<22} {'(skipped: ' + t['error'] + ')'}"
        return (f"  {name:<22} {t['median']:>10.4f} "
                f"{t['p10']:>10.4f} {t['mean']:>10.4f}")

    print()
    print("=" * 72)
    print(f"BENCHMARK -- eager  (warmup={warmup}, iters={iters}; times in ms)")
    print("=" * 72)
    print(f"  {'kernel':<22} {'median':>10} {'p10':>10} {'mean':>10}")
    print(_row('unified_attention_fwd', ua_t))
    print(_row('mha_varlen_fwd',        fmha_t))
    if "error" not in ua_t and "error" not in fmha_t:
        speedup = fmha_t["median"] / ua_t["median"]
        faster  = "UA" if speedup > 1 else "FMHA"
        gap_pct = (speedup - 1) * 100 if speedup > 1 else (1 / speedup - 1) * 100
        print(f"  ratio (median): UA / FMHA = {ua_t['median'] / fmha_t['median']:.3f}  "
              f"=> {faster} faster by {gap_pct:.1f}%")

    print()
    print("=" * 72)
    print(f"BENCHMARK -- CUDA graph replay  (iters={iters})")
    print("=" * 72)
    print(f"  {'kernel':<22} {'median':>10} {'p10':>10} {'mean':>10}")
    print(_row('unified_attention_fwd', ua_g))
    print(_row('mha_varlen_fwd',        fmha_g))
    if "error" not in ua_g and "error" not in fmha_g:
        speedup = fmha_g["median"] / ua_g["median"]
        faster  = "UA" if speedup > 1 else "FMHA"
        gap_pct = (speedup - 1) * 100 if speedup > 1 else (1 / speedup - 1) * 100
        print(f"  ratio (median): UA / FMHA = {ua_g['median'] / fmha_g['median']:.3f}  "
              f"=> {faster} faster by {gap_pct:.1f}%")
    if "error" not in ua_t and "error" not in ua_g:
        line = f"  graph-mode savings: UA   {(1 - ua_g['median']/ua_t['median'])*100:5.1f}%"
        if "error" not in fmha_t and "error" not in fmha_g:
            line += f"  FMHA {(1 - fmha_g['median']/fmha_t['median'])*100:5.1f}%"
        print(line)

    # ---------------------------------------------------------------
    # 8. Try the non-split paged FMHA: mha_batch_prefill_func
    #
    # mha_varlen_fwd(paged) is hardcoded to route through mha_fwd_splitkv,
    # so there is no way to "turn off" splitkv on that entry point. The
    # symmetric non-split paged FMHA is `mha_batch_prefill_func`, which
    # takes CSR-style paging (kv_indptr + kv_page_indices) instead of a
    # rectangular block_table. Per docs/unified_attention_vs_fmha.md, in
    # the current build its compiled instances exist for hdim=128 but not
    # hdim=64 -- so this attempt is *expected* to fail at our shape and
    # confirms that, on (GQA-8, hdim=64), UA is the only single-pass
    # paged kernel available (no splitkv path on the FMHA side).
    # ---------------------------------------------------------------
    print()
    print("=" * 72)
    print("EXTRA: mha_batch_prefill_func (non-split paged FMHA, CSR layout)")
    print("=" * 72)

    # Convert rectangular block_tables -> CSR (kv_indptr, kv_page_indices)
    pages_per_seq_t = torch.tensor(pages_per_seq, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = pages_per_seq_t.cumsum(0)
    kv_page_indices = torch.cat(
        [block_tables[b, :pages_per_seq[b]] for b in range(batch)]
    ).to(torch.int32)
    kv_last_page_lens = torch.tensor(
        [((kl - 1) % page_blk_size) + 1 for kl in kv_lens],
        dtype=torch.int32, device=device,
    )

    out_bp_buf = torch.empty_like(q_packed)

    def bench_bp():
        mha_batch_prefill_func(
            q_packed, key_cache, value_cache,
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            kv_page_indices=kv_page_indices,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale_s,
            causal=True,
            kv_last_page_lens=kv_last_page_lens,
            out=out_bp_buf,
        )

    try:
        bench_bp()  # one cold call to provoke any "no kernel" error
        torch.cuda.synchronize()
        bp_t = time_fn(bench_bp, warmup, iters)
        bp_g = time_graph(bench_bp, warmup, iters)
        diff_bp = (out_bp_buf.float() - out_ua.float()).abs()
        print(f"  ran successfully  abs_mean={out_bp_buf.abs().mean().item():.6f}  "
              f"max diff vs UA = {diff_bp.max().item():.4e}")
        print(f"  {'kernel':<24} {'median':>10} {'p10':>10} {'mean':>10}")
        print(f"  {'mha_batch_prefill (eager)':<24} {bp_t['median']:>10.4f} "
              f"{bp_t['p10']:>10.4f} {bp_t['mean']:>10.4f}")
        if "error" not in bp_g:
            print(f"  {'mha_batch_prefill (graph)':<24} {bp_g['median']:>10.4f} "
                  f"{bp_g['p10']:>10.4f} {bp_g['mean']:>10.4f}")
    except Exception as e:                                          # noqa: BLE001
        msg = str(e).splitlines()[0]
        print(f"  failed at this shape:")
        print(f"    {type(e).__name__}: {msg}")
        print()
        print("  Per docs/unified_attention_vs_fmha.md, mha_batch_prefill_func has")
        print("  compiled instances at (MHA, hdim=128) but not at (GQA-8, hdim=64),")
        print("  and UA is reciprocally narrow (d=128 MHA + d=64 GQA-8 only). So:")
        print("    - (GQA-8, d=64)  : only UA is single-pass paged here.")
        print("    - (MHA,   d=128) : UA and mha_batch_prefill_func both work,")
        print("                       which is the truly apples-to-apples shape.")


def main():
    # Config A -- UA's d64 GQA-8 sweet spot (mixed prefill + decode, paged KV).
    # mha_batch_prefill_func is expected to fail here.
    run_config(
        label="(A) GQA-8, hdim=64, page_blk=64  (UA's d64 sweet spot)",
        query_lens=[256, 1, 512, 1],
        kv_lens=[256, 2048, 512, 4096],
        num_kv_heads=8,
        num_q_per_kv=8,                    # GQA-8
        head_size=64,
        page_blk_size=64,
    )

    # Config B -- UA's d128 MHA sweet spot. UA dispatches to its hdim=128 MHA
    # instance which is compiled with kPageBlockSize=32 (BlockSize=32 because
    # HeadSize > 64 in unified_attention_impl.hpp), so we use page_blk_size=32
    # to match. This is the configuration where mha_batch_prefill_func also
    # has a compiled kernel, giving us a fair single-pass-paged-vs-single-
    # pass-paged comparison.
    run_config(
        label="(B) MHA, hdim=128, page_blk=32  (UA's d128 sweet spot, "
              "mha_batch_prefill_func also works)",
        query_lens=[256, 1, 512, 1],
        kv_lens=[256, 2048, 512, 4096],
        num_kv_heads=8,
        num_q_per_kv=1,                    # MHA
        head_size=128,
        page_blk_size=32,
    )


if __name__ == "__main__":
    main()
