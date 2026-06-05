"""Isolate gemm1 correctness at BM16 (matched tile_m=sort_block_m=16).

ref   : BM32 sort + moe_mxfp4_sort a1_scale + std gemm1 @ tile_m=32
test  : BM16 sort + torch a1_scale       + inline gemm1 @ tile_m=16 (sbm=16)
Both produce inter [token, topk, inter//2]; compare per-token (dequant cosine).
"""
import sys
import torch
import aiter
from aiter import dtypes
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter.utility.fp4_utils import moe_mxfp4_sort, mxfp4_to_f32

from bench_up_moe_v1 import KIMI, build_weights, build_inputs
from _test_iq_quantcheck import quant_like_kernel
from _test_iq_scalelayout import build_a1_scale_preshuffle


def _e(d): return torch.empty((0,), dtype=dtypes.bf16, device=d)


def sort_bufs(M, NE, topk, BM, device):
    active = min(NE, M * topk)
    cumsum_max = M * topk + active * (BM - 1)
    max_sorted = ((cumsum_max + BM - 1) // BM) * BM
    return dict(
        sorted_token_ids=torch.empty((max_sorted,), device=device, dtype=dtypes.i32),
        sorted_expert_ids=torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32),
        cumsum_tensor=torch.empty((1,), device=device, dtype=dtypes.i32),
        reverse_sorted=torch.empty((M * topk,), device=device, dtype=dtypes.i32),
        sorted_weights=torch.empty((max_sorted,), device=device, dtype=dtypes.fp32),
        masked_m=torch.empty((NE,), device=device, dtype=dtypes.i32),
        m_indices=torch.empty((max_sorted,), device=device, dtype=dtypes.i32),
    ), max_sorted


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    device = torch.device("cuda")
    shape = KIMI
    fly_w, _mx_w = build_weights(shape, device)
    hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
    w1u = fly_w["w1"].view(torch.uint8); w1sv = fly_w["w1_scale"].view(dtypes.fp8_e8m0)
    w1v = w1u.view(dtypes.fp4x2)
    NE = w1u.shape[0]; D_HIDDEN = hidden.shape[1]; D_INTER = w1u.shape[1] // 2
    topk = topk_ids.shape[1]; K32 = D_HIDDEN // 32

    # ---- reference: BM32 sort + moe_mxfp4_sort + std gemm1 @ tile_m=32 ----
    b32, ms32 = sort_bufs(M, NE, topk, 32, device)
    out32 = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)
    aiter.mxfp4_moe_sort(topk_ids=topk_ids, topk_weight=topk_weight, **b32,
        bf16_zero_out=_e(device), bf16_zero_workspace=_e(device),
        M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=32, prologue=1)
    a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((M, K32), device=device, dtype=torch.uint8)
    aiter.mxfp4_moe_quant(a_input=hidden, a_quant=a_quant, a_scale=a_scale,
        bf16_zero_out=out32, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=32)
    nv32 = b32["cumsum_tensor"].repeat(2)
    a1s32 = moe_mxfp4_sort(a_scale.view(dtypes.fp8_e8m0).view(M, 1, -1),
        sorted_ids=b32["sorted_token_ids"], num_valid_ids=nv32, token_num=M, block_size=32)
    ref = flydsl_moe_stage1(a=a_quant.view(dtypes.fp4x2), w1=w1v,
        sorted_token_ids=b32["sorted_token_ids"], sorted_expert_ids=b32["sorted_expert_ids"],
        num_valid_ids=nv32, topk=topk, tile_m=32, tile_n=128, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="fp4", w1_scale=w1sv, a1_scale=a1s32,
        sorted_weights=None, use_async_copy=False, waves_per_eu=2, b_nt=2, gate_mode="separated")
    ref_q = (ref[0] if isinstance(ref, tuple) else ref)

    # ---- test: BM16 sort + torch a1_scale + inline gemm1 @ tile_m=16 ----
    b16, ms16 = sort_bufs(M, NE, topk, 16, device)
    aiter.mxfp4_moe_sort(topk_ids=topk_ids, topk_weight=topk_weight, **b16,
        bf16_zero_out=_e(device), bf16_zero_workspace=_e(device),
        M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=16, prologue=0)
    nv16 = b16["cumsum_tensor"].repeat(2)
    _, e8 = quant_like_kernel(hidden)
    a1s16 = build_a1_scale_preshuffle(e8, b16["sorted_token_ids"], ms16, K32, 1)
    a1s16 = a1s16.view(torch.uint8).view(dtypes.fp8_e8m0)
    tst = flydsl_moe_stage1(a=hidden, w1=w1v,
        sorted_token_ids=b16["sorted_token_ids"], sorted_expert_ids=b16["sorted_expert_ids"],
        num_valid_ids=nv16, topk=topk, tile_m=16, tile_n=128, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="fp4", w1_scale=w1sv, a1_scale=a1s16,
        sorted_weights=None, use_async_copy=False, waves_per_eu=2, b_nt=2,
        gate_mode="separated", inline_quant=True)
    tst_q = (tst[0] if isinstance(tst, tuple) else tst)
    ref_s = (ref[1] if isinstance(ref, tuple) else None)
    tst_s = (tst[1] if isinstance(tst, tuple) else None)
    torch.cuda.synchronize()

    c = torch.nn.functional.cosine_similarity(
        mxfp4_to_f32(ref_q.view(dtypes.fp4x2)).reshape(-1),
        mxfp4_to_f32(tst_q.view(dtypes.fp4x2)).reshape(-1), dim=0).item()
    print(f"M={M}  gemm1 inter dequant cosine (BM16-inline vs BM32-std): {c:.5f}")
    print(f"  cumsum32={int(b32['cumsum_tensor'].item())} cumsum16={int(b16['cumsum_tensor'].item())}")

    # ---- validate gemm1 inter SCALE: read e8m0[sorted_row, kg] from preshuffle ----
    K32i = D_INTER // 32  # inter scale kgroups

    def read_scale_preshuffle(buf, nv):
        b = buf.reshape(-1).view(torch.uint8)
        NB = (K32i + 7) // 8
        rows = torch.arange(nv, device=device)
        kg = torch.arange(K32i, device=device)
        R = rows[:, None]; KG = kg[None, :]
        mb = R // 32; mh = (R % 32) // 16; rr = R % 16
        nb = KG // 8; nh = (KG % 8) // 4; cc = (KG % 8) % 4
        byte = mh + nh * 2
        dword = mb * (NB * 64) + nb * 64 + cc * 16 + rr
        boff = (dword * 4 + byte).reshape(-1).to(torch.int64)
        return b[boff].reshape(nv, K32i)

    def scale_by_pair(buf, sti, nv):
        sc = read_scale_preshuffle(buf, nv).to(torch.int64)  # [nv, K32i]
        r = sti[:nv].to(torch.int64); t = r & 0xFFFFFF; s = r >> 24
        v = t < M
        out = torch.zeros((M * topk, K32i), dtype=torch.int64, device=device)
        key = torch.where(v, t * topk + s, torch.zeros_like(t))
        out[key] = torch.where(v[:, None], sc, out[key])
        return out

    sc16 = scale_by_pair(tst_s, b16["sorted_token_ids"], int(b16["cumsum_tensor"].item()))
    sc32 = scale_by_pair(ref_s, b32["sorted_token_ids"], int(b32["cumsum_tensor"].item()))
    smatch = (sc16 == sc32).float().mean().item()
    print(f"  inter SCALE match (BM16 vs BM32, per pair): {smatch*100:.3f}%")
    dif = (sc16 != sc32)
    if dif.any():
        ij = dif.nonzero()[:4]
        for p in ij:
            i, j = p.tolist()
            print(f"    pair {i} kg {j}: bm16={int(sc16[i,j])} bm32={int(sc32[i,j])}")

    # ---- gemm2 both paths ----
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    w2u = fly_w["w2"].view(torch.uint8); w2sv = fly_w["w2_scale"].view(dtypes.fp8_e8m0)
    w2v = w2u.view(dtypes.fp4x2)

    out32 = torch.zeros((M, D_HIDDEN), dtype=dtypes.bf16, device=device)
    flydsl_moe_stage2(inter_states=ref_q, w2=w2v,
        sorted_token_ids=b32["sorted_token_ids"], sorted_expert_ids=b32["sorted_expert_ids"],
        num_valid_ids=nv32, topk=topk, out=out32, tile_m=32, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2sv, a2_scale=ref_s, sorted_weights=b32["sorted_weights"], b_nt=2)

    g2tn = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    g2tk = 256
    # torch-recomputed sorted_weights from topk_weight + sorted_token_ids decode
    sti16 = b16["sorted_token_ids"]
    raw = sti16.to(torch.int64)
    tid = raw & 0xFFFFFF; slot = raw >> 24
    valid = tid < M
    tw_flat = topk_weight.reshape(-1).float()
    idx = torch.where(valid, tid * topk + slot, torch.zeros_like(tid))
    sw_torch = torch.where(valid, tw_flat[idx], torch.zeros_like(tw_flat[idx])).to(torch.float32)
    # diff vs cheap-sort weights
    swc = b16["sorted_weights"]
    nvv = int(b16["cumsum_tensor"].item())
    wdiff = (sw_torch[:nvv] - swc[:nvv]).abs().max().item()
    print(f"  sorted_weights max|torch-cheap| (valid rows)={wdiff:.4f}")

    out16 = torch.zeros((M, D_HIDDEN), dtype=dtypes.bf16, device=device)
    flydsl_moe_stage2(inter_states=tst_q, w2=w2v,
        sorted_token_ids=b16["sorted_token_ids"], sorted_expert_ids=b16["sorted_expert_ids"],
        num_valid_ids=nv16, topk=topk, out=out16, tile_m=16, tile_n=g2tn, tile_k=g2tk,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2sv, a2_scale=tst_s, sorted_weights=b16["sorted_weights"],
        b_nt=2, sort_block_m=16)
    out16t = torch.zeros((M, D_HIDDEN), dtype=dtypes.bf16, device=device)
    flydsl_moe_stage2(inter_states=tst_q, w2=w2v,
        sorted_token_ids=b16["sorted_token_ids"], sorted_expert_ids=b16["sorted_expert_ids"],
        num_valid_ids=nv16, topk=topk, out=out16t, tile_m=16, tile_n=g2tn, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2sv, a2_scale=tst_s, sorted_weights=sw_torch,
        b_nt=2, sort_block_m=16)
    torch.cuda.synchronize()
    print(f"  [gemm2 tile_n={g2tn}]")

    mxref = fused_moe(hidden, _mx_w["w1"], _mx_w["w2"], topk_weight, topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        w1_scale=_mx_w["w1_scale"], w2_scale=_mx_w["w2_scale"]).clone()
    torch.cuda.synchronize()

    def cos(a, b): return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0).item()
    print(f"  out32(BM32 flydsl)      vs mxref: {cos(out32, mxref):.5f}  norm={out32.float().norm():.2f}")
    print(f"  out16(cheap-sort sw)    vs mxref: {cos(out16, mxref):.5f}  norm={out16.float().norm():.2f}")
    print(f"  out16t(torch sw)        vs mxref: {cos(out16t, mxref):.5f}  norm={out16t.float().norm():.2f}")
    print(f"  out16 vs out32: {cos(out16, out32):.5f}   mxref norm={mxref.float().norm():.2f}")

    # ---- gemm2@16 FLAT (per-sorted-row, no atomic reduce) + torch reduce ----
    flat = torch.zeros((ms16, D_HIDDEN), dtype=dtypes.bf16, device=device)
    flydsl_moe_stage2(inter_states=tst_q, w2=w2v,
        sorted_token_ids=b16["sorted_token_ids"], sorted_expert_ids=b16["sorted_expert_ids"],
        num_valid_ids=nv16, topk=topk, out=flat, tile_m=16, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w2_scale=w2sv, a2_scale=tst_s, sorted_weights=None, b_nt=2,
        sort_block_m=16, flat_output=True)
    torch.cuda.synchronize()
    # torch reduce: out[token] += weight[row] * flat[row]
    nvv = int(b16["cumsum_tensor"].item())
    outf = torch.zeros((M, D_HIDDEN), dtype=torch.float32, device=device)
    rr = sti16[:nvv].to(torch.int64); rtid = rr & 0xFFFFFF; rsl = rr >> 24
    rvalid = rtid < M
    contrib = flat[:nvv].float() * sw_torch[:nvv, None]
    outf.index_add_(0, torch.where(rvalid, rtid, torch.zeros_like(rtid)),
                    torch.where(rvalid[:, None], contrib, torch.zeros_like(contrib)))
    print(f"  out16-FLAT+torchreduce vs mxref: {cos(outf, mxref):.5f}  norm={outf.float().norm():.2f}")

    # ---- characterize: gemm2@32 FLAT (correct acc) vs gemm2@16 FLAT per (token,slot) ----
    flat32b = torch.zeros((ms32, D_HIDDEN), dtype=dtypes.bf16, device=device)
    flydsl_moe_stage2(inter_states=ref_q, w2=w2v,
        sorted_token_ids=b32["sorted_token_ids"], sorted_expert_ids=b32["sorted_expert_ids"],
        num_valid_ids=nv32, topk=topk, out=flat32b, tile_m=32, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w2_scale=w2sv, a2_scale=ref_s, sorted_weights=None, b_nt=2, flat_output=True)
    torch.cuda.synchronize()

    def flat_by_pair(flatbuf, sti, nv):
        r = sti[:nv].to(torch.int64); t = r & 0xFFFFFF; s = r >> 24
        v = t < M
        out = torch.zeros((M * topk, D_HIDDEN), dtype=torch.float32, device=device)
        key = torch.where(v, t * topk + s, torch.zeros_like(t))
        out[key] = torch.where(v[:, None], flatbuf[:nv].float(), out[key])
        return out
    f16p = flat_by_pair(flat, b16["sorted_token_ids"], int(b16["cumsum_tensor"].item()))
    f32p = flat_by_pair(flat32b, b32["sorted_token_ids"], int(b32["cumsum_tensor"].item()))
    valid_pair = (f32p.abs().sum(1) > 0)
    fr = (f16p[valid_pair] / (f32p[valid_pair] + 1e-6))
    print(f"  FLAT per-pair cos: {cos(f16p[valid_pair], f32p[valid_pair]):.5f}")
    print(f"  FLAT elem ratio f16/f32: mean={fr.mean():.3f} std={fr.std():.3f} "
          f"med={fr.median():.3f}")
    # per-column mean ratio (first 8 cols) to see if N-dependent
    colr = (f16p[valid_pair].mean(0) / (f32p[valid_pair].mean(0).abs() + 1e-6))
    # per-row(pair) mean ratio variance
    rowr = f16p[valid_pair].norm(dim=1) / (f32p[valid_pair].norm(dim=1) + 1e-6)
    print(f"  per-pair norm ratio: mean={rowr.mean():.3f} std={rowr.std():.4f} "
          f"min={rowr.min():.3f} max={rowr.max():.3f}")
    # which sorted rows (BM16) are wrong? correlate ratio with row position-in-block
    sti16b = b16["sorted_token_ids"]; nvv16 = int(b16["cumsum_tensor"].item())
    rr = sti16b[:nvv16].to(torch.int64); tt = rr & 0xFFFFFF; ss = rr >> 24
    vv = tt < M
    rowr16 = flat[:nvv16].float().norm(dim=1)
    f32_by_pair_norm = f32p.norm(dim=1)
    pair_key = tt * topk + ss
    ref_norm = torch.where(vv, f32_by_pair_norm[torch.where(vv, pair_key, torch.zeros_like(pair_key))], torch.ones_like(rowr16))
    ratio_row = rowr16 / (ref_norm + 1e-6)
    pos_in_blk = torch.arange(nvv16, device=device) % 16
    blk_idx = torch.arange(nvv16, device=device) // 16
    bad = vv & (ratio_row < 0.9)
    good = vv & (ratio_row >= 0.9)
    print(f"  valid rows={int(vv.sum())} bad(<0.9)={int(bad.sum())} good={int(good.sum())}")
    if bad.any():
        bp = pos_in_blk[bad]
        print(f"  bad rows pos-in-block: {sorted(set(bp.tolist()))[:20]}")
        print(f"  good rows pos-in-block: {sorted(set(pos_in_blk[good].tolist()))[:20]}")
        # block parity (even/odd 16-block within 32-group)
        print(f"  bad block-idx parity (blk%2): 0->{int(((blk_idx[bad]%2)==0).sum())} 1->{int(((blk_idx[bad]%2)==1).sum())}")
    # per-token norms
    pn32 = out32.float().norm(dim=1)
    pn16 = out16.float().norm(dim=1)
    ratio = (pn16 / (pn32 + 1e-6))
    print(f"  per-token norm ratio out16/out32: min={ratio.min():.3f} max={ratio.max():.3f} mean={ratio.mean():.3f}")
    print(f"  ratio per token: {[round(x,2) for x in ratio.tolist()]}")
    sw16 = b16["sorted_weights"]; sw32 = b32["sorted_weights"]
    print(f"  sw16: sum={sw16.sum():.2f} nonzero={int((sw16!=0).sum())}/{sw16.numel()}  "
          f"sw32: sum={sw32.sum():.2f} nonzero={int((sw32!=0).sum())}/{sw32.numel()}")
    # count valid (token,slot) pairs in sorted ids
    def npairs(sti, nv):
        r = sti[:nv].to(torch.int64); t = r & 0xFFFFFF; s = r >> 24
        v = t < M
        key = (t * topk + s)[v]
        return int(v.sum()), int(torch.unique(key).numel())
    n16, u16 = npairs(b16["sorted_token_ids"], int(b16["cumsum_tensor"].item()))
    n32, u32 = npairs(b32["sorted_token_ids"], int(b32["cumsum_tensor"].item()))
    print(f"  valid rows/unique-pairs: BM16={n16}/{u16}  BM32={n32}/{u32}  (expect pairs={M*topk})")


if __name__ == "__main__":
    main()
