"""Standalone *correctness* comparison: asm sparse decode kernel vs ATOM's
gfx1250 gluon `pa_decode_sparse`, invoked EXACTLY the way ATOM invokes it.

Companion to `_stage1_bench_pa.py` (which only compares stage1 *timing*). Here we
feed BOTH kernels the SAME bf16 q/kv and check their final outputs against a
shared pure-torch bf16 golden.

`pa_decode_sparse` is called the ATOM way (see ATOM
model_ops/v4_kernels/paged_decode.py::sparse_attn_v4_paged_decode gfx1250 path):
NO `kv_splits` (so it is auto-inferred from T/H/len(kv_indices)) and NO
`skip_reduce` (so BOTH stage1 split-K AND stage2 reduce run whenever the inferred
split > 1). The auto-inferred split is printed as the `pa_split` column. The asm
side stays at num_kv_splits=1 (its logits[:,0] is already the final result);
split count is just a parallelization strategy, so the merged pa output is still
mathematically the same attention as asm and the golden.

Why this is a fair *numeric* comparison (unlike the perf bench):
  * In this poc MLA both QK score and V output run over the full 512-wide latent
    (`_torch_attn_decode_bf16_golden` dots q.kv over all 512 dims for scores AND
    for the value). `pa_decode_sparse` with D=512 does exactly the same math, so
    D=512 is not an under-count here.
  * The only intrinsic difference is dtype: asm quantizes q/kv to fp8+e8m0
    (native), while pa runs in bf16. So:
        golden(bf16) vs pa(bf16)  -> should be TIGHT (both bf16, same math)
        golden(bf16) vs asm(fp8)  -> fp8 quant noise floor
        pa(bf16)     vs asm(fp8)  -> the two kernels' mutual difference
    The last row is the "are the results consistent?" number the user asked for;
    it is expected to sit near the fp8 quant floor, NOT at bf16 machine eps.

The batch grid includes small batches (1, 16) so the auto-inferred split lands
well above 1 there, exercising ATOM's real small-batch / large-split + stage2
path (batch=64 falls back toward split=1).

Usage:
  ENABLE_CK=0 python op_tests/_stage1_verify_pa.py             # default sweep
  ENABLE_CK=0 python op_tests/_stage1_verify_pa.py 64 512      # one combo: batch ctx
  ENABLE_CK=0 python op_tests/_stage1_verify_pa.py 64 512 128  # batch ctx gqa
"""
import sys
sys.path.insert(0, "op_tests")
sys.path.insert(0, "/home/amd/feifei/ATOM")

import itertools

import torch
import test_mla_v4_kargpreld as T
import aiter, aiter.mla
from aiter import dtypes
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

Q = 1          # q_seq_logical (decode)
SINK = True    # exercise the per-head softmax-denom sink on both paths

# gqa + ctx mirror _stage1_bench_pa.py; batch spans small->large so the
# auto-inferred pa split covers ATOM's large-split (stage2) and split=1 ends.
_GQA_LIST = [64, 128]
_CTX_LENS = [256, 512, 1024]
_BATCH_SIZES = [1, 16, 64]

# gfx1250/gluon constants used by pa_decode_sparse's kv_splits auto-infer.
_PA_BLOCK_K = 16
_PA_MAX_NUM_WG = 1024


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1


def _infer_kv_splits(n_tokens, n_heads, n_indices):
    """Replicate pa_decode_sparse's gfx1250 kv_splits auto-inference (for display
    only; the real value is chosen inside the kernel launcher identically)."""
    block_h = max(_next_pow2(min(n_heads, 16)), 16)
    n_head_blocks = -(-n_heads // block_h)  # ceil
    max_kv_splits = max(1, -(-n_indices // _PA_BLOCK_K))
    ks = max(1, _PA_MAX_NUM_WG // max(1, n_tokens * n_head_blocks))
    ks = min(max_kv_splits, ks)
    return _next_pow2(ks)


def _delta_stats(a, b):
    """max |a-b| and mismatch ratio (rtol=atol=3e-2) as a compact (float,float)."""
    a = a.float()
    b = b.float()
    maxd = float((a - b).abs().max().item())
    close = torch.isclose(a, b, rtol=3e-2, atol=3e-2)
    ratio = float((~close).sum().item()) / max(a.numel(), 1)
    return maxd, ratio


def run_golden(inp, sm):
    out, _ = T._torch_attn_decode_bf16_golden(
        inp["q_bf16"], inp["kv_bf16"], inp["qo_indptr"], inp["kv_indptr"],
        inp["kv_page_indices"], inp["kv_last_page_lens"], sm, attn_sink=inp["sink"],
    )
    return out  # [total_q, gqa, 512] bf16


def run_asm(inp, gqa, sm):
    """asm stage1 with num_kv_splits=1 -> logits[:,0] is the final fp32 partial."""
    dev = "cuda"
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = inp["q_bf16"].size(0)
    ns = inp["qo_indptr"].size(0) - 1
    nh = T.NUM_KV_HEADS * gqa
    ob = torch.empty((total_q, gqa, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    sidx = torch.arange(0, ns + 1, dtype=torch.int32, device=dev)  # split=1
    lb = torch.empty((total_q, 1, nh, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev)
    eb = torch.empty((total_q, 1, nh, 1), dtype=dtypes.fp32, device=dev)
    logits, _lse = aiter.mla.mla_decode_fwd_v4_nm(
        q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(),
        output=ob, qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"], kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=sidx, max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
        sm_scale=sm, out_16_nosplit=0, num_kv_splits=1, logits=lb, attn_lse=eb,
    )
    return logits[:, 0].to(dtypes.bf16)  # [total_q, gqa, 512]


def run_pa(inp, gqa, sm):
    """pa called EXACTLY as ATOM does on gfx1250: no kv_splits (auto-inferred)
    and no skip_reduce, so stage1 split-K + stage2 reduce both run when the
    inferred split > 1. Returns (final bf16 output, inferred_kv_splits).

    Same bf16 q/kv as the golden/asm: kv is the 512-latent pool, kv_indices/indptr
    are the asm page tables (page_size=1, last_page_len=1 => pages == kv tokens).
    """
    q = inp["q_bf16"].contiguous()                         # [total_q, gqa, 512]
    unified_kv = inp["kv_bf16"].reshape(-1, T.V_HEAD_DIM).contiguous()  # [pages, 512]
    kv_indices = inp["kv_page_indices"].to(torch.int32).contiguous()
    kv_indptr = inp["kv_indptr"].to(torch.int32).contiguous()
    out = pa_decode_sparse(
        q, unified_kv, kv_indices, kv_indptr, inp["sink"].contiguous(), sm,
        has_invalid=False,
    )
    ks = _infer_kv_splits(q.size(0), gqa, kv_indices.numel())
    return out, ks  # [total_q, gqa, 512] bf16, int


def verify(batch, ctx, gqa):
    inp = T._build_bf16_inputs(batch=batch, kv_seq_lens=ctx, q_seq_logical=Q,
                               seed=0, gqa_ratio=gqa, attn_sink=SINK)
    sm = 1.0 / (T._QUANT_D ** 0.5)
    golden = run_golden(inp, sm)
    asm = run_asm(inp, gqa, sm)
    pa, pa_split = run_pa(inp, gqa, sm)
    g_pa = _delta_stats(golden, pa)
    g_asm = _delta_stats(golden, asm)
    pa_asm = _delta_stats(pa, asm)
    return pa_split, g_pa, g_asm, pa_asm


hdr = (f"{'gqa':>4} {'batch':>6} {'ctx':>6} {'pa_split':>8} | "
       f"{'golden~pa (max|Δ|/mism%)':>26} | "
       f"{'golden~asm (max|Δ|/mism%)':>27} | "
       f"{'pa~asm (max|Δ|/mism%)':>24}")
print(hdr)
print("-" * len(hdr))

if len(sys.argv) > 1:
    batch = int(sys.argv[1])
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    gqa = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    combos = [(gqa, batch, ctx)]
else:
    combos = list(itertools.product(_GQA_LIST, _BATCH_SIZES, _CTX_LENS))

for gqa, batch, ctx in combos:
    pa_split, g_pa, g_asm, pa_asm = verify(batch, ctx, gqa)
    print(f"{gqa:>4} {batch:>6} {ctx:>6} {pa_split:>8} | "
          f"{g_pa[0]:>13.4g}/{g_pa[1]*100:>10.3f}% | "
          f"{g_asm[0]:>13.4g}/{g_asm[1]*100:>11.3f}% | "
          f"{pa_asm[0]:>10.4g}/{pa_asm[1]*100:>10.3f}%")
