"""Stage1 probe for mla_decode_fwd_v4_nm.

Pre-fills the stage1 partial buffers (logits / attn_lse) with a sentinel,
runs the v4 nm kernel at the reproducing config, then classifies every cell of
the STAGE1 output as: sane / NaN / +-Inf / illegal-huge (|x|>1e30 finite) /
unwritten (still == sentinel). Reported per split index so we can tell whether
the garbage lives in valid split slots (=> stage1 asm bug) or only in
skipped/unwritten slots (=> stage2 merge reading uninitialized memory).
"""

import sys

import torch

sys.path.insert(0, "op_tests")
import test_mla_v4_kargpreld as T  # noqa: E402

import aiter  # noqa: E402
import aiter.mla  # noqa: E402
from aiter import dtypes  # noqa: E402

SENT = 123456.0


def classify(x, name):
    x = x.float()
    tot = x.numel()
    nan = torch.isnan(x)
    inf = torch.isinf(x)
    finite = torch.isfinite(x)
    huge = (x.abs() > 1e30) & finite
    unwritten = x == SENT
    print(
        f"  [{name}] total={tot}  nan={nan.sum().item()}  inf={inf.sum().item()}  "
        f"illegal_huge_finite={huge.sum().item()}  unwritten(sentinel)={unwritten.sum().item()}"
    )
    if finite.any():
        f = x[finite & ~unwritten]
        if f.numel():
            print(
                f"       finite(written) min={f.min().item():.4g} max={f.max().item():.4g} "
                f"absmax={f.abs().max().item():.4g}"
            )
    return nan, inf, huge, unwritten


def probe(batch, kv, q, gqa, splits, sink, seed):
    inp = T._build_bf16_inputs(
        batch=batch,
        kv_seq_lens=kv,
        q_seq_logical=q,
        seed=seed,
        gqa_ratio=gqa,
        attn_sink=sink,
    )
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])

    total_q = inp["q_bf16"].size(0)
    num_heads = T.NUM_KV_HEADS * gqa
    dev = "cuda"
    num_seqs = inp["qo_indptr"].numel() - 1

    logits = torch.full(
        (total_q, splits, num_heads, T.V_HEAD_DIM), SENT, dtype=dtypes.fp32, device=dev
    )
    lse = torch.full(
        (total_q, splits, num_heads, 1), SENT, dtype=dtypes.fp32, device=dev
    )
    out = torch.empty((total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    split_indptr = torch.tensor(
        [i * splits for i in range(num_seqs + 1)], dtype=torch.int32, device=dev
    )

    ret_logits, ret_lse = aiter.mla.mla_decode_fwd_v4_nm(
        q=qp,
        qrope=qr.contiguous(),
        kv_buffer=kvp,
        kvrope=kvr.contiguous(),
        output=out,
        qo_indptr=inp["qo_indptr"],
        kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"],
        kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=split_indptr,
        max_seqlen_q=inp["max_seqlen_q"],
        sink=inp["sink"],
        sm_scale=1.0 / (T._QUANT_D**0.5),
        out_16_nosplit=0,
        num_kv_splits=splits,
        logits=logits,
        attn_lse=lse,
    )
    torch.cuda.synchronize()

    print(
        f"=== config batch={batch} kv={kv} q={q} gqa={gqa} splits={splits} "
        f"sink={sink} seed={seed} ==="
    )
    print(" STAGE1 logits (partials, kernel-native [total_q, splits, H, dv]):")
    classify(ret_logits, "logits ALL")
    print(" STAGE1 attn_lse:")
    classify(ret_lse, "lse ALL")

    # Per-split breakdown of the logits partials.
    L = ret_logits.float()
    for s in range(splits):
        Ls = L[:, s]
        nan = torch.isnan(Ls).sum().item()
        inf = torch.isinf(Ls).sum().item()
        fin = torch.isfinite(Ls)
        huge = ((Ls.abs() > 1e30) & fin).sum().item()
        unw = (Ls == SENT).sum().item()
        print(
            f"   split {s}: nan={nan} inf={inf} illegal_huge={huge} unwritten={unw}"
            f"  (of {Ls.numel()})"
        )

    # Merged output classification.
    print(" MERGED output (stage2 -> BF16):")
    classify(out, "output")
    print()


def leak_test(batch=64, kv=271, q=1, gqa=64, splits=8, sink=True, iters=20):
    """Reproduce the test's buffer-reuse: allocate logits/lse ONCE, poison every
    cell with an extreme value (mimicking uninitialized torch.empty garbage that
    can be ~1e35/NaN), then call the kernel `iters` times reusing the SAME
    buffers WITHOUT re-poisoning. If the merged output ever shows huge/NaN/Inf,
    the stage2 merge is leaking the unwritten (invalid) split slots."""
    inp = T._build_bf16_inputs(
        batch=batch, kv_seq_lens=kv, q_seq_logical=q, seed=0, gqa_ratio=gqa,
        attn_sink=sink,
    )
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = inp["q_bf16"].size(0)
    num_heads = T.NUM_KV_HEADS * gqa
    dev = "cuda"
    num_seqs = inp["qo_indptr"].numel() - 1
    split_indptr = torch.tensor(
        [i * splits for i in range(num_seqs + 1)], dtype=torch.int32, device=dev
    )
    logits = torch.empty(
        (total_q, splits, num_heads, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev
    )
    lse = torch.empty((total_q, splits, num_heads, 1), dtype=dtypes.fp32, device=dev)
    out = torch.empty((total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    # Poison: half +1e35, and inject NaN/Inf too.
    logits.fill_(1e35)
    lse.fill_(1e35)
    logits.view(-1)[::2] = float("nan")
    lse.view(-1)[::3] = float("inf")

    print(f"=== LEAK TEST (poison=1e35/NaN/Inf, reuse buffers) x{iters} ===")
    bad = 0
    for it in range(iters):
        aiter.mla.mla_decode_fwd_v4_nm(
            q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(),
            output=out, qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
            kv_page_indices=inp["kv_page_indices"],
            kv_last_page_lens=inp["kv_last_page_lens"], split_indptr=split_indptr,
            max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
            sm_scale=1.0 / (T._QUANT_D**0.5), out_16_nosplit=0, num_kv_splits=splits,
            logits=logits, attn_lse=lse,
        )
        torch.cuda.synchronize()
        of = out.float()
        nan = torch.isnan(of).sum().item()
        inf = torch.isinf(of).sum().item()
        huge = ((of.abs() > 1e30) & torch.isfinite(of)).sum().item()
        absmax = of[torch.isfinite(of)].abs().max().item() if torch.isfinite(of).any() else float("nan")
        flag = "  <== LEAK" if (nan or inf or huge) else ""
        if nan or inf or huge:
            bad += 1
        print(
            f"  iter {it:2d}: out nan={nan} inf={inf} huge={huge} absmax={absmax:.4g}{flag}"
        )
    print(f" => {bad}/{iters} iterations leaked garbage into merged output\n")


def sync_test(batch=64, kv=271, q=1, gqa=64, splits=8, sink=True, iters=10):
    """Isolate the cross-iteration race: call the wrapper `iters` times reusing
    the SAME buffers, once WITH torch.cuda.synchronize() between calls and once
    WITHOUT. If only the no-sync run corrupts stage1, the bug is a missing
    barrier between iteration N's stage2 read and iteration N+1's stage1 write
    on the shared logits buffer."""
    inp = T._build_bf16_inputs(
        batch=batch, kv_seq_lens=kv, q_seq_logical=q, seed=0, gqa_ratio=gqa,
        attn_sink=sink,
    )
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = inp["q_bf16"].size(0)
    num_heads = T.NUM_KV_HEADS * gqa
    dev = "cuda"
    num_seqs = inp["qo_indptr"].numel() - 1
    split_indptr = torch.tensor(
        [i * splits for i in range(num_seqs + 1)], dtype=torch.int32, device=dev
    )
    out = torch.empty((total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    logits = torch.empty(
        (total_q, splits, num_heads, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev
    )
    lse = torch.empty((total_q, splits, num_heads, 1), dtype=dtypes.fp32, device=dev)
    common = dict(
        q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(),
        output=out, qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"],
        kv_last_page_lens=inp["kv_last_page_lens"], split_indptr=split_indptr,
        max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
        sm_scale=1.0 / (T._QUANT_D**0.5), out_16_nosplit=0, num_kv_splits=splits,
        logits=logits, attn_lse=lse,
    )

    for label, per_iter_sync in [
        ("per-iter sync", True),
        ("NO per-iter sync (back-to-back)", False),
    ]:
        # Fresh buffers for each phase so state can't leak between phases.
        logits.fill_(0)
        lse.fill_(0)
        out.fill_(0)
        torch.cuda.synchronize()
        for _ in range(iters):
            aiter.mla.mla_decode_fwd_v4_nm(**common)
            if per_iter_sync:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        Lf = logits.float()
        per = []
        for s in range(splits):
            xs = Lf[:, s]
            fin = torch.isfinite(xs)
            per.append(round(xs[fin].abs().max().item(), 1) if fin.any() else float("nan"))
        of = out.float()
        oh = ((of.abs() > 1e30) & torch.isfinite(of)).sum().item()
        oam = of[torch.isfinite(of)].abs().max().item()
        print(f"[{label}] stage1 per-split absmax={per}")
        print(f"    merged out huge={oh} absmax={oam:.4g}"
              f"{'  <== GARBAGE' if oh else '  (clean)'}")
    print()


def perf_repro(batch=64, kv=271, q=1, gqa=64, splits=8, sink=True, reps=6):
    """Reproduce the EXACT test path (run_perftest under a profiler, buffer
    reuse) and classify BOTH the returned stage1 partials and the merged output
    after the perf loop, to localize where the garbage enters."""
    from aiter.test_common import run_perftest

    for r in range(reps):
        inp = T._build_bf16_inputs(
            batch=batch, kv_seq_lens=kv, q_seq_logical=q, seed=0, gqa_ratio=gqa,
            attn_sink=sink,
        )
        qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
        kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
        total_q = inp["q_bf16"].size(0)
        num_heads = T.NUM_KV_HEADS * gqa
        dev = "cuda"
        num_seqs = inp["qo_indptr"].numel() - 1
        split_indptr = torch.tensor(
            [i * splits for i in range(num_seqs + 1)], dtype=torch.int32, device=dev
        )
        output_buf = torch.empty(
            (total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev
        )
        logits_buf = torch.empty(
            (total_q, splits, num_heads, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev
        )
        lse_buf = torch.empty(
            (total_q, splits, num_heads, 1), dtype=dtypes.fp32, device=dev
        )
        common = dict(
            q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(),
            output=output_buf, qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
            kv_page_indices=inp["kv_page_indices"],
            kv_last_page_lens=inp["kv_last_page_lens"], split_indptr=split_indptr,
            max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
            sm_scale=1.0 / (T._QUANT_D**0.5), num_kv_splits=splits,
            logits=logits_buf, attn_lse=lse_buf,
        )
        # fp8-dequant reference (same as the real test), for the exact compare.
        out_fp8_ref, _ = T._torch_attn_decode_fp8_dequant_ref(
            qp, qr, kvp, kvr, inp["qo_indptr"], inp["kv_indptr"],
            inp["kv_page_indices"], inp["kv_last_page_lens"],
            1.0 / (T._QUANT_D**0.5), attn_sink=inp["sink"],
        )

        (logits, _lse), us = run_perftest(
            aiter.mla.mla_decode_fwd_v4_nm, out_16_nosplit=0, **common,
            num_iters=10, num_warmup=2, num_rotate_args=1,
        )
        torch.cuda.synchronize()

        Lf = logits.float()
        of = output_buf.float()
        # per-split absmax of stage1 partials
        per_am = []
        for s in range(splits):
            xs = Lf[:, s]
            fin = torch.isfinite(xs)
            per_am.append(round(xs[fin].abs().max().item(), 1) if fin.any() else float("nan"))
        # merged output garbage
        on = torch.isnan(of).sum().item()
        oi = torch.isinf(of).sum().item()
        ofin = torch.isfinite(of)
        oh = ((of.abs() > 1e30) & ofin).sum().item()
        oam = of[ofin].abs().max().item() if ofin.any() else float("nan")
        # exact test-style delta
        delta = (of - out_fp8_ref.float()).abs()
        dmax = delta.max().item()
        nbad = (delta > 3e-2).sum().item()
        tot = delta.numel()
        print(
            f"rep {r}: stage1 per-split absmax={per_am}"
        )
        print(
            f"        MERGED out: nan={on} inf={oi} huge={oh} absmax={oam:.4g} "
            f"| vs ref: max_delta={dmax:.4g} bad={100*nbad/tot:.2f}% "
            f"{'<== GARBAGE' if (on or oi or oh or dmax > 1e3) else ''}"
        )
    print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "leak":
        leak_test(iters=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif mode == "perf":
        perf_repro(reps=int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif mode == "sync":
        for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
            sync_test()
    else:
        reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        for r in range(reps):
            probe(batch=64, kv=271, q=1, gqa=64, splits=8, sink=True, seed=r)
