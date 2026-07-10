"""Standalone perf comparison: asm sparse decode kernel vs ATOM's gfx1250 gluon
`pa_decode_sparse`, with pa invoked EXACTLY the way ATOM invokes it.

- asm       = the `..._sparse` .co kernel (fp8, native), num_kv_splits=1 so it
              produces its final output directly; we report its stage1 device time
              (there is no real stage2 at split=1).
- pa (full) = `pa_decode_sparse(...)` called the ATOM way (see ATOM
              model_ops/v4_kernels/paged_decode.py gfx1250 path): NO kv_splits (so
              it is auto-inferred) and NO skip_reduce (so stage1 split-K + stage2
              reduce BOTH run when the inferred split > 1). This is what ATOM
              actually pays. The inferred split is printed as `pa_split`.
- pa (s1)   = the same call forced with skip_reduce=True at the same inferred
              split -> ONLY the stage1 split kernel (kept for context, so the
              stage2 reduce cost = pa_full_us - pa_s1_us is visible).

NOT bit-exact / not memory-fair: pa is bf16 while asm is fp8, and D=512 here
(MLA QK is really 576-wide, so pa QK is ~11% under-counted). Perf only.
asm stays at split=1 (its natural fast path; the poc's asm stage2 merge is
pure-torch and unrepresentative), so the pa_full/asm ratio is "ATOM's real
full-pipeline cost vs asm's single-pass cost".

Usage:
  ENABLE_CK=0 python op_tests/_stage1_bench_pa.py              # default sweep
  ENABLE_CK=0 python op_tests/_stage1_bench_pa.py 64 512       # one combo: batch ctx
  ENABLE_CK=0 python op_tests/_stage1_bench_pa.py 64 512 128   # batch ctx gqa
  ENABLE_CK=0 python op_tests/_stage1_bench_pa.py 64 512 128 4 # batch ctx gqa asm_split

The asm_split only sets the asm pass's num_kv_splits (sweep the default list via
_ASM_SPLITS). pa's split is always auto-inferred the ATOM way and is unaffected.
"""
import sys
sys.path.insert(0, "op_tests")
sys.path.insert(0, "/home/amd/feifei/ATOM")

import torch
import test_mla_v4_kargpreld as T
import aiter, aiter.mla
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

Q = 1
SINK = True

# gqa + ctx mirror the kargpreld sweep; batch spans small->large so the
# auto-inferred pa split covers ATOM's large-split (stage2) and split=1 ends.
_GQA_LIST = [64, 128]
#_CTX_LENS = [256, 512, 1024]
#_BATCH_SIZES = [64]
#_ASM_SPLITS = [1]

_CTX_LENS = [1024]
_BATCH_SIZES = [64]
_ASM_SPLITS = [1, 2, 4]

# gfx1250/gluon constants used by pa_decode_sparse's kv_splits auto-infer.
_PA_BLOCK_K = 16
_PA_MAX_NUM_WG = 1024

# Perf iteration counts for run_perftest (mirrors test_mla_v4_kargpreld.py's
# _PERF usage). Kept as module constants so both bench fns share them.
_PERF = {"num_iters": 50, "num_warmup": 2}


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1


def _infer_kv_splits(n_tokens, n_heads, n_indices):
    """Replicate pa_decode_sparse's gfx1250 kv_splits auto-inference (so we can
    force the same split for the stage1-only timing and print it)."""
    block_h = max(_next_pow2(min(n_heads, 16)), 16)
    n_head_blocks = -(-n_heads // block_h)  # ceil
    max_kv_splits = max(1, -(-n_indices // _PA_BLOCK_K))
    ks = max(1, _PA_MAX_NUM_WG // max(1, n_tokens * n_head_blocks))
    ks = min(max_kv_splits, ks)
    return _next_pow2(ks)


def bench_asm(batch, ctx, gqa, split=1):
    # split=1 is asm's natural fast path (single pass produces the final output);
    # split>1 exercises the split-K stage1 + torch stage2 merge.
    inp = T._build_bf16_inputs(batch=batch, kv_seq_lens=ctx, q_seq_logical=Q,
                               seed=0, gqa_ratio=gqa, attn_sink=SINK)
    sm = 1.0 / (T._QUANT_D ** 0.5)
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = inp["q_bf16"].size(0)
    ns = inp["qo_indptr"].size(0) - 1
    nh = T.NUM_KV_HEADS * gqa
    dev = "cuda"
    ob = torch.empty((total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    sidx = torch.tensor([i * split for i in range(ns + 1)], dtype=torch.int32, device=dev)
    lb = torch.empty((total_q, split, nh, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev)
    eb = torch.empty((total_q, split, nh, 1), dtype=dtypes.fp32, device=dev)
    kw = dict(
        q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(), output=ob,
        qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"], kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=sidx, max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
        sm_scale=sm, out_16_nosplit=0, num_kv_splits=split, logits=lb, attn_lse=eb,
    )
    # asm mla_decode_fwd_v4_nm launches stage1 (sparse) + stage2 (merge). Profile
    # both device times separately so we can report stage1 / stage2 / total and
    # compare each against pa. (stage1 still lines up with the kargpreld table.)
    s1_us, s2_us = T._profile_stage_times(
        lambda: aiter.mla.mla_decode_fwd_v4_nm(**kw),
        iters=_PERF["num_iters"],
        warmup=_PERF["num_warmup"],
    )
    return s1_us, s2_us


def bench_pa(batch, ctx, gqa):
    """Time pa the ATOM way (auto split, full stage1+stage2) and, for context,
    the same call forced to stage1-only. Returns (pa_s1_us, pa_full_us, split)."""
    dev = "cuda"
    D = T.V_HEAD_DIM  # 512
    Tn = batch
    H = gqa
    q = torch.randn((Tn, H, D), dtype=torch.bfloat16, device=dev)
    total_pages = batch * ctx
    unified_kv = torch.randn((total_pages, D), dtype=torch.bfloat16, device=dev) * 0.1
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (Tn + 1) * ctx, ctx, dtype=torch.int32, device=dev)
    attn_sink = torch.randn((H,), dtype=torch.float32, device=dev)
    sm = 1.0 / (D ** 0.5)

    split = _infer_kv_splits(Tn, H, kv_indices.numel())

    # pa full: EXACTLY ATOM's gfx1250 call — no kv_splits (auto-inferred to
    # `split`), no skip_reduce => stage1 split + stage2 reduce both run.
    _, full_us = run_perftest(
        pa_decode_sparse,
        q, unified_kv, kv_indices, kv_indptr, attn_sink, sm,
        has_invalid=False,
        num_iters=_PERF["num_iters"],
        num_warmup=_PERF["num_warmup"],
    )
    # pa stage1-only: same split but skip_reduce=True => only the split kernel
    # (skip_reduce is a no-op when the inferred split==1, so s1==full there).
    _, s1_us = run_perftest(
        pa_decode_sparse,
        q, unified_kv, kv_indices, kv_indptr, attn_sink, sm,
        kv_splits=split, has_invalid=False, skip_reduce=True,
        num_iters=_PERF["num_iters"],
        num_warmup=_PERF["num_warmup"],
    )
    return s1_us, full_us, split


import itertools

def _ratio(pa, asm):
    return pa / asm if asm > 0 else float("nan")


# Three side-by-side comparisons: stage1, stage2, and total (asm s1+s2 vs pa full).
print(f"{'gqa':>4} {'batch':>6} {'ctx':>6} {'asm_split':>9} {'pa_split':>8} | "
      f"{'asm_s1':>8} {'pa_s1':>8} {'s1 pa/asm':>10} | "
      f"{'asm_s2':>8} {'pa_s2':>8} {'s2 pa/asm':>10} | "
      f"{'asm_tot':>8} {'pa_tot':>8} {'tot pa/asm':>11}")
print("-" * 128)

if len(sys.argv) > 1:
    batch = int(sys.argv[1])
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    gqa = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    asm_split = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    combos = [(gqa, batch, ctx, asm_split)]
else:
    combos = list(itertools.product(_GQA_LIST, _BATCH_SIZES, _CTX_LENS, _ASM_SPLITS))

# gqa outermost so rows mirror the kargpreld summary-table grouping.
for gqa, batch, ctx, asm_split in combos:
    asm_s1, asm_s2 = bench_asm(batch, ctx, gqa, asm_split)
    pa_s1_us, pa_full_us, split = bench_pa(batch, ctx, gqa)
    asm_tot = asm_s1 + asm_s2
    pa_s2_us = pa_full_us - pa_s1_us  # pa stage2 reduce = full - stage1
    print(f"{gqa:>4} {batch:>6} {ctx:>6} {asm_split:>9} {split:>8} | "
          f"{asm_s1:>8.2f} {pa_s1_us:>8.2f} {_ratio(pa_s1_us, asm_s1):>9.2f}x | "
          f"{asm_s2:>8.2f} {pa_s2_us:>8.2f} {_ratio(pa_s2_us, asm_s2):>9.2f}x | "
          f"{asm_tot:>8.2f} {pa_full_us:>8.2f} {_ratio(pa_full_us, asm_tot):>10.2f}x")
