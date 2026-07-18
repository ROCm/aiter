import sys, torch
sys.path.insert(0, "op_tests")
import test_mla_v4_kargpreld as T
import aiter, aiter.mla
from aiter import dtypes

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 60
KV = int(sys.argv[2]) if len(sys.argv) > 2 else 271
SPLITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
Q, GQA, SINK = 1, 64, True


def run_once():
    inp = T._build_bf16_inputs(batch=BATCH, kv_seq_lens=KV, q_seq_logical=Q, seed=0,
                               gqa_ratio=GQA, attn_sink=SINK)
    qp, qr = T._native_to_2buff_for_asm(inp["q_bf16"])
    kvp, kvr = T._native_to_2buff_for_asm(inp["kv_bf16"])
    total_q = inp["q_bf16"].size(0); nh = T.NUM_KV_HEADS * GQA; dev = "cuda"
    ns = inp["qo_indptr"].numel() - 1
    sidx = torch.tensor([i * SPLITS for i in range(ns + 1)], dtype=torch.int32, device=dev)
    ob = torch.empty((total_q, GQA, T.V_HEAD_DIM), dtype=dtypes.bf16, device=dev)
    lb = torch.zeros((total_q, SPLITS, nh, T.V_HEAD_DIM), dtype=dtypes.fp32, device=dev)
    eb = torch.full((total_q, SPLITS, nh, 1), float("-inf"), dtype=dtypes.fp32, device=dev)
    aiter.mla.mla_decode_fwd_v4_nm(
        q=qp, qrope=qr.contiguous(), kv_buffer=kvp, kvrope=kvr.contiguous(), output=ob,
        qo_indptr=inp["qo_indptr"], kv_indptr=inp["kv_indptr"],
        kv_page_indices=inp["kv_page_indices"], kv_last_page_lens=inp["kv_last_page_lens"],
        split_indptr=sidx, max_seqlen_q=inp["max_seqlen_q"], sink=inp["sink"],
        sm_scale=1.0 / (T._QUANT_D ** 0.5), out_16_nosplit=0, num_kv_splits=SPLITS,
        logits=lb, attn_lse=eb)
    torch.cuda.synchronize()
    return lb.float().clone(), eb.float().clone()


print(f"===== batch={BATCH} : per-split stage1 nondeterminism (3 runs, same input) =====")
L1, E1 = run_once()
L2, E2 = run_once()
L3, E3 = run_once()
print("split :  logits max|r2-r1|   logits max|r3-r1|    lse max|r2-r1|   status")
for s in range(SPLITS):
    d2 = (L2[:, s] - L1[:, s]).abs().max().item()
    d3 = (L3[:, s] - L1[:, s]).abs().max().item()
    fe = torch.isfinite(E2[:, s]) & torch.isfinite(E1[:, s])
    de = (E2[:, s] - E1[:, s])[fe].abs().max().item() if fe.any() else 0.0
    status = "RACE" if max(d2, d3) > 1e-6 else "stable"
    print(f"  {s}   :   {d2:>13.6g}   {d3:>13.6g}   {de:>13.6g}    {status}")
