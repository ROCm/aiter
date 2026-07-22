# ql=3 (mg path) bitwise/cosine check: SAME per-token fp8 data packed into
# block_size=16 vs 64 -> compare kernel outputs. Prints input tensor shapes.
import sys, math, argparse
import torch
from aiter import dtypes
from aiter.paged_attn import PagedAttention

FP8 = dtypes.fp8
NQH, NKV, HD, PART = 8, 1, 128, 256


def build_tokens(bs, ctx, mtp, seed=0, qdt="fp8"):
    """Per-token logical data, quantized ONCE (block-size independent).

    qdt='fp8'  -> query pre-quantized to fp8 + external q_scale.
    qdt='bf16' -> query stays bf16, q_scale=None (kernel quantizes in-kernel).
    """
    torch.manual_seed(seed)
    fmax = torch.finfo(FP8).max
    q_bf = torch.randn(bs * mtp, NQH, HD, dtype=torch.bfloat16, device="cuda")
    k_bf = torch.randn(bs, ctx, HD, dtype=torch.bfloat16, device="cuda")
    v_bf = torch.randn(bs, ctx, HD, dtype=torch.bfloat16, device="cuda")

    if qdt == "fp8":
        q_scale = (q_bf.float().abs().amax(-1) / fmax).clamp_min(1e-6)          # [bs*mtp, 8]
        query = (q_bf.float() / q_scale[..., None]).clamp(-fmax, fmax).to(FP8)  # [bs*mtp, 8, 128]
    else:  # bf16 query: kernel does in-kernel max|Q| quant, external q_scale ignored
        query = q_bf                                                            # [bs*mtp, 8, 128] bf16
        q_scale = None

    k_scale = (k_bf.float().abs().amax(-1) / fmax).clamp_min(1e-6)          # [bs, ctx]  per-token
    k_fp8 = (k_bf.float() / k_scale[..., None]).clamp(-fmax, fmax).to(FP8)  # [bs, ctx, 128]

    v_scale = (v_bf.float().abs().amax() / fmax).clamp_min(1e-6).reshape(1)  # [1]  per-tensor
    v_fp8 = (v_bf.float() / v_scale).clamp(-fmax, fmax).to(FP8)             # [bs, ctx, 128]
    return query, q_scale, k_fp8, k_scale, v_fp8, v_scale


def pack(bs, ctx, B, k_fp8, k_scale, v_fp8):
    """Reshuffle identical per-token bytes into a block_size=B paged layout."""
    assert ctx % B == 0
    bps = ctx // B
    nb = bs * bps
    # K cache [nb, 1, HD//16, B, 16]
    k = (k_fp8.view(bs, bps, B, HD // 16, 16)
              .permute(0, 1, 3, 2, 4).reshape(nb, NKV, HD // 16, B, 16).contiguous())
    # k_scale [nb, 1, B]  (per token = per (block, slot))
    ks = k_scale.view(bs, bps, B).reshape(nb, NKV, B).contiguous()
    # V cache 4D [nb, 1, HD, B]
    v = (v_fp8.view(bs, bps, B, HD)
             .permute(0, 1, 3, 2).reshape(nb, NKV, HD, B).contiguous())
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(bs, bps)
    return k, ks, v, bt


def run(q_fp8, q_scale, k, ks, v, v_scale, bt, seq_lens, ctx, mtp):
    scale = 1.0 / math.sqrt(HD)
    return PagedAttention.forward_decode(
        q_fp8, k, v, bt, seq_lens, ctx, kv_cache_dtype="auto",
        num_kv_heads=NKV, scale=scale, alibi_slopes=None,
        k_scale=ks, v_scale=v_scale, q_scale=q_scale, mtp=mtp,
        p_scale=None, p_scale_inv=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--mtp", type=int, default=3)
    ap.add_argument("--qdt", type=str, default="fp8", choices=["fp8", "bf16"])
    ap.add_argument("--show_shapes", action="store_true")
    args = ap.parse_args()
    mtp = args.mtp
    qdt = args.qdt

    # print shapes for one representative config
    bs0, ctx0 = args.bs[0], args.ctx[0]
    q, qs, kf, ks, vf, vs = build_tokens(bs0, ctx0, mtp, qdt=qdt)
    k16, ks16, v16, bt16 = pack(bs0, ctx0, 16, kf, ks, vf)
    k64, ks64, v64, bt64 = pack(bs0, ctx0, 64, kf, ks, vf)
    print(f"===== INPUT SHAPES (bs={bs0} ctx={ctx0} mtp={mtp} q={qdt}) =====")
    print(f"query      : {tuple(q.shape)}      dtype={q.dtype}   # [bs*mtp, num_q_heads, head_size]")
    print(f"q_scale    : {('None (in-kernel quant)' if qs is None else str(tuple(qs.shape))+'  dtype='+str(qs.dtype))}  # per-(token,head)")
    print(f"k_cache b16: {tuple(k16.shape)}  dtype={k16.dtype}  # [nb, kv_heads, hd//16, block, 16]")
    print(f"k_cache b64: {tuple(k64.shape)}   dtype={k64.dtype}")
    print(f"k_scale b16: {tuple(ks16.shape)}       dtype={ks16.dtype}  # [nb, kv_heads, block] per-token")
    print(f"k_scale b64: {tuple(ks64.shape)}        dtype={ks64.dtype}")
    print(f"v_cache b16: {tuple(v16.shape)}    dtype={v16.dtype}  # [nb, kv_heads, head_size, block]")
    print(f"v_cache b64: {tuple(v64.shape)}     dtype={v64.dtype}")
    print(f"v_scale    : {tuple(vs.shape)}             dtype={vs.dtype}  # per-tensor (per kv-head)")
    print(f"block_tables b16: {tuple(bt16.shape)}   b64: {tuple(bt64.shape)}   dtype={bt16.dtype}")
    print(f"context_lens: ({bs0},)  dtype=int32   value={ctx0}")
    del k16, ks16, v16, k64, ks64, v64, q, qs, kf, vf
    torch.cuda.empty_cache()

    print(f"\n===== ql={mtp} q={qdt}  block16 vs block64 (SAME per-token input) =====")
    print(f"{'bs':>4} {'ctx':>7} | {'bitwise':>7} {'cosine':>14} {'max_abs':>11} {'ndiff':>10}")
    all_eq = True
    for bs in args.bs:
        for ctx in args.ctx:
            try:
                q, qs, kf, ks, vf, vs = build_tokens(bs, ctx, mtp, qdt=qdt)
                seq = torch.full((bs,), ctx, dtype=torch.int32, device="cuda")
                k16, ks16, v16, bt16 = pack(bs, ctx, 16, kf, ks, vf)
                k64, ks64, v64, bt64 = pack(bs, ctx, 64, kf, ks, vf)
                o16 = run(q, qs, k16, ks16, v16, vs, bt16, seq, ctx, mtp)
                o64 = run(q, qs, k64, ks64, v64, vs, bt64, seq, ctx, mtp)
                eq = torch.equal(o16, o64)
                a = o16.flatten().double(); b = o64.flatten().double()
                cos = (a @ b / (a.norm() * b.norm())).item()
                mx = (o16.float() - o64.float()).abs().max().item()
                nd = int((o16 != o64).sum().item())
                all_eq = all_eq and eq
                print(f"{bs:>4} {ctx:>7} | {str(eq):>7} {cos:>14.10f} "
                      f"{mx:>11.3e} {nd:>4}/{o16.numel():<5}", flush=True)
                del k16, ks16, v16, k64, ks64, v64, o16, o64, q, qs, kf, vf
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"{bs:>4} {ctx:>7} | OOM", flush=True)
                torch.cuda.empty_cache()
    print("\nRESULT:", "ALL BITWISE IDENTICAL" if all_eq else "MISMATCH FOUND")


if __name__ == "__main__":
    main()
