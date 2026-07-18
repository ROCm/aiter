import sys, torch
sys.path.insert(0, "op_tests")
import test_mla_v4_kargpreld as T
import aiter, aiter.mla
from aiter import dtypes

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 64
KV = int(sys.argv[2]) if len(sys.argv) > 2 else 271
SPLITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GQA, Q, SINK = 64, 1, True
ATOL = RTOL = 3e-2
T._SEED = 0


def build_and_run():
    inp = T._build_bf16_inputs(batch=BATCH, kv_seq_lens=KV, q_seq_logical=Q, seed=0,
                               gqa_ratio=GQA, attn_sink=SINK)
    sm = 1.0 / (T._QUANT_D ** 0.5)
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    ref, _ = T._torch_attn_decode_fp8_dequant_ref(
        qp, qr, kvp, kvr, inp["qo_indptr"], inp["kv_indptr"],
        inp["kv_page_indices"], inp["kv_last_page_lens"], sm, attn_sink=inp["sink"])
    total_q = inp["q_bf16"].size(0); ns = inp["qo_indptr"].size(0) - 1
    nh = T.NUM_KV_HEADS * GQA; dev = "cuda"
    ob = torch.empty((total_q, GQA, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    sidx = torch.tensor([i * SPLITS for i in range(ns + 1)], dtype=torch.int32, device=dev)
    lb = torch.empty((total_q, SPLITS, nh, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev)
    eb = torch.empty((total_q, SPLITS, nh, 1), dtype=dtypes.fp32, device=dev)
    aiter.mla.mla_decode_fwd_v4_nm(
        q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(), output=ob,
        qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"], kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=sidx, max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
        sm_scale=sm, out_16_nosplit=0, num_kv_splits=SPLITS, logits=lb, attn_lse=eb)
    torch.cuda.synchronize()
    return ref.float(), ob.float()


tries = int(sys.argv[4]) if len(sys.argv) > 4 else 20
for t in range(tries):
    ref, asm = build_and_run()
    close = torch.isclose(ref, asm, rtol=RTOL, atol=ATOL)
    pct = (~close).float().mean().item()
    if pct > 0.02:
        print(f"[got FAIL on try {t}] mismatch={pct:.2%}")
        break
else:
    print(f"no >2% fail in {tries} tries; using last run (mismatch={pct:.2%})")

m = ~close
rv = ref[m]; av = asm[m]
delta = (av - rv).abs()
print(f"\n===== ctx={KV} batch={BATCH} split={SPLITS} : {m.sum().item()} mismatched elems =====")
print(f"ref (golden) at mismatches:  min={rv.min():.3f} max={rv.max():.3f} mean|.|={rv.abs().mean():.3f}")
print(f"asm (kernel) at mismatches:  min={av.min():.3f} max={av.max():.3f} mean|.|={av.abs().mean():.3f}")
print(f"|delta| at mismatches     :  min={delta.min():.4f} max={delta.max():.4f} mean={delta.mean():.4f}")
print(f"whole-tensor magnitude    :  ref |.|max={ref.abs().max():.3f}  asm |.|max={asm.abs().max():.3f}")
print(f"NaN/Inf in asm? {torch.isnan(asm).any().item()} / {torch.isinf(asm).any().item()}")
# neighbor vs wild: how big is delta relative to typical value scale?
scale = ref.abs().mean().item()
print(f"\ntypical |ref| scale (all)= {scale:.3f}")
print(f"delta/scale ratio: median={ (delta/ (rv.abs()+1e-6)).median():.2f}  max={(delta/(rv.abs()+1e-6)).max():.2f}")
# histogram of delta
import numpy as np
d = delta.cpu().numpy()
for thr in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e3, 1e6]:
    print(f"  |delta| <= {thr:>8}: {100.0*(d<=thr).mean():5.1f}%")
# sample pairs
idx = torch.argsort(delta, descending=True)[:12]
print("\ntop-12 worst (ref -> asm, delta):")
for i in idx.tolist():
    print(f"   {rv[i].item():+.4f} -> {av[i].item():+.4f}   d={delta[i].item():.4f}")
