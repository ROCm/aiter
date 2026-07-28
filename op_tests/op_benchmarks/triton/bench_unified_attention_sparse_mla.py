import argparse

import torch
import triton

import aiter.mla
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)
from op_tests.triton_tests.attention.test_unified_attention_sparse_mla import (
    Param,
    chunk_input,
    generate_test_data,
)


def reorder_for_sparse_pattern(indices_in_kvcache, pattern):
    # NSA-style top-k selection (what the test harness emits) clusters
    # selected indices by physical block — consecutive entries in a row tend
    # to share a block, so per-tile KV loads stay within few physical blocks.
    # DSA-style token-level selection picks indices scattered across the
    # sequence, so per-tile loads scatter across many blocks. Shuffling each
    # row independently reproduces that scatter without changing validity or
    # row length (and the -1 padding sentinels stay -1 after shuffle).
    if pattern == "nsa":
        return indices_in_kvcache
    out = indices_in_kvcache.clone()
    n_rows, n_cols = out.shape
    perm = torch.argsort(
        torch.rand(n_rows, n_cols, device=out.device), dim=1
    )
    out = torch.gather(out, 1, perm)
    return out


def dense_physical_indices_to_csr(indices_in_kvcache):
    num_rows = indices_in_kvcache.shape[0]
    chunks = []
    indptr = [0]
    max_sparse_len = 0
    for row in range(num_rows):
        valid = indices_in_kvcache[row] != -1
        row_indices = indices_in_kvcache[row, valid]
        chunks.append(row_indices)
        row_len = int(row_indices.numel())
        indptr.append(indptr[-1] + row_len)
        max_sparse_len = max(max_sparse_len, row_len)

    if chunks:
        kv_indices = torch.cat(chunks).to(torch.int32)
    else:
        kv_indices = torch.empty(
            (0,), dtype=torch.int32, device=indices_in_kvcache.device
        )
    kv_indptr = torch.tensor(
        indptr, dtype=torch.int32, device=indices_in_kvcache.device
    )
    return kv_indptr, kv_indices, max_sparse_len


def make_inputs(args):
    total_dim = args.lora_dim + args.rope_dim
    test_p = Param(
        args.batch,
        args.sq,
        args.sk,
        d=total_dim,
        dv=args.lora_dim,
        h_q=args.heads,
        block_size=args.block_size,
        is_varlen=args.varlen,
        is_causal=False,
        is_fp8=False,
        topk=args.top_k,
        test_performance=True,
    )
    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_test_data(test_p)
    )
    (
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        q,
        block_table,
        blocked_k,
        _,
        indices_in_kvcache,
    ) = chunk_input(
        cache_seqlens,
        q,
        block_table,
        blocked_k,
        abs_indices,
        indices_in_kvcache,
    )
    indices_in_kvcache = reorder_for_sparse_pattern(
        indices_in_kvcache, args.sparse_pattern
    )
    kv_indptr, kv_indices, max_sparse_len = dense_physical_indices_to_csr(
        indices_in_kvcache
    )
    return (
        q,
        blocked_k,
        block_table,
        indices_in_kvcache,
        kv_indptr,
        kv_indices,
        max_sparse_len,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
    )


def run(args):
    (
        q,
        blocked_k,
        block_table,
        topk_indices,
        kv_indptr,
        kv_indices,
        max_sparse_len,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
    ) = make_inputs(args)

    softmax_scale = args.lora_dim**-0.5
    out_topk = torch.empty((*q.shape[:-1], args.lora_dim), device=q.device, dtype=q.dtype)
    out_csr = torch.empty_like(out_topk)
    out_mla = torch.empty_like(out_topk)
    total_q = q.shape[0]
    qk_head_dim = args.lora_dim + args.rope_dim
    kv_buffer_mla = blocked_k.reshape(-1, 1, 1, qk_head_dim)
    qo_indptr_mla = torch.arange(
        total_q + 1, dtype=torch.int32, device=q.device
    )
    kv_last_page_lens = torch.ones(total_q, dtype=torch.int32, device=q.device)

    # FP8 path: quantize the KV cache (excluding NaN sentinels) to fp8_e4m3
    # with a single per-tensor scale, then route through unified_attention_sparse_mla
    # with k_scale=v_scale (single buffer is both K and V).
    fp8_enabled = args.dtype == "fp8"
    if fp8_enabled:
        blocked_k_clean = torch.where(
            torch.isnan(blocked_k),
            torch.zeros_like(blocked_k),
            blocked_k,
        )
        amax = blocked_k_clean.abs().float().max().clamp_min(1e-8)
        fp8_max = 448.0
        scale_val = (amax / fp8_max).item()
        blocked_kv_fp8 = (
            (blocked_k_clean.float() / scale_val).clamp(-fp8_max, fp8_max)
        ).to(torch.float8_e4m3fn)
        scale_t = torch.tensor(scale_val, dtype=torch.float32, device=q.device)
        kv_for_kernel = blocked_kv_fp8
    else:
        kv_for_kernel = blocked_k
        scale_t = None

    def run_topk():
        unified_attention_sparse_mla(
            q,
            kv_for_kernel,
            out_topk,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            topk_indices,
            block_table,
            args.lora_dim,
            k_scale=scale_t,
            v_scale=scale_t,
        )

    def run_csr():
        unified_attention_sparse_mla(
            q,
            kv_for_kernel,
            out_csr,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            None,
            block_table,
            args.lora_dim,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            max_sparse_len=max_sparse_len,
            k_scale=scale_t,
            v_scale=scale_t,
        )

    def run_mla_decode():
        aiter.mla.mla_decode_fwd(
            q,
            kv_buffer_mla,
            out_mla,
            qo_indptr_mla,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            1,
            1,
            1,
            softmax_scale,
        )

    if args.validate:
        run_topk()
        run_csr()
        torch.testing.assert_close(out_csr, out_topk, atol=2e-2, rtol=2e-2)
        if not fp8_enabled:
            run_mla_decode()
            torch.testing.assert_close(out_mla, out_csr, atol=1.5e-2, rtol=1e-2)

    topk_ms = triton.testing.do_bench(run_topk, warmup=args.warmup, rep=args.rep)
    csr_ms = triton.testing.do_bench(run_csr, warmup=args.warmup, rep=args.rep)
    if fp8_enabled or args.skip_mla_decode:
        mla_ms = float("nan")
    else:
        try:
            mla_ms = triton.testing.do_bench(
                run_mla_decode, warmup=args.warmup, rep=args.rep
            )
        except (AttributeError, RuntimeError) as e:
            print(f"# skipping mla_decode_fwd reference: {e}")
            mla_ms = float("nan")
    nnz = int(kv_indices.numel())
    print(
        "unified_attention_sparse_mla "
        f"batch={args.batch} sq={args.sq} sk={args.sk} heads={args.heads} "
        f"lora={args.lora_dim} rope={args.rope_dim} block={args.block_size} "
        f"top_k={args.top_k} dtype={args.dtype} pattern={args.sparse_pattern} "
        f"nnz={nnz} max_sparse_len={max_sparse_len}"
    )
    import math as _math
    print(f"topk_matrix_ms={topk_ms:.4f}")
    print(f"csr_ms={csr_ms:.4f}")
    has_mla = not _math.isnan(mla_ms)
    if has_mla:
        print(f"mla_decode_fwd_ms={mla_ms:.4f}")
    print(f"csr_speedup_vs_topk={topk_ms / csr_ms:.3f}x")
    if has_mla:
        print(f"mla_decode_fwd_speedup_vs_csr={csr_ms / mla_ms:.3f}x")
        print(f"mla_decode_fwd_speedup_vs_topk={topk_ms / mla_ms:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse MLA top-k vs CSR")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sq", type=int, default=1)
    parser.add_argument("--sk", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--lora-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--skip-mla-decode",
        action="store_true",
        help="Skip the ASM mla_decode_fwd reference (used when aiter HIP is unavailable).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp8"],
        default="bf16",
        help="KV cache dtype: bf16 (default) or fp8 (e4m3, per-tensor scale)",
    )
    parser.add_argument(
        "--sparse-pattern",
        choices=["nsa", "dsa"],
        default="nsa",
        help=(
            "Sparse-attention pattern. 'nsa' (default): block-quantized "
            "selection from the test harness; consecutive indices share a "
            "physical block. 'dsa': per-row shuffle so consecutive indices "
            "scatter across blocks, simulating DSA's lightning-indexer "
            "token-level selection."
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
