"""D_INTER=192 chunk-invariance test for mxfp4_moe_run.

Exercises the non-256-multiple inter dim path:
  * gemm1 N_OUT = 2*192 = 384  -> BN=128 (exact, no waste)
  * gemm2 K = 192 -> padded to 256 (MFMA-K=128 / BK / scale layout need %256);
    K_TILES_TOTAL==1 single-tile kernel path; the [192,256) tail is physically
    zero in w2 / inter (host K-pad in mxfp4_moe_run) and contributes nothing.

MoE is per-token independent, so running N tokens at once must equal running
them in chunks. Weights are kept small-magnitude so the atomic gemm2 bf16
reduction's FP-reordering noise floor stays well under tolerance.

    python op_tests/test_mxfp4_moe_e192_chunk.py --tokens 512 --chunk 256
"""
import argparse
import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import mxfp4_moe_run

torch.set_default_device("cuda")

NE, D_HIDDEN, D_INTER, TOPK = 385, 7168, 192, 9
KN1 = f"mxfp4_moe_g1_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_BM32_CACHED"
KN2 = f"mxfp4_moe_g2_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_TOPK{TOPK}_BM32_ATOMIC"


def make_weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    # small e8m0 exponents -> small magnitudes -> tiny atomic noise floor
    e8 = lambda *s: torch.randint(118, 120, s, dtype=torch.uint8, generator=g)
    w1 = u8(NE, 2 * D_INTER, D_HIDDEN // 2)
    w1s = e8(NE, 2 * D_INTER, D_HIDDEN // 32)
    w2 = u8(NE, D_HIDDEN, D_INTER // 2)      # K=192 (96 bytes); host pads to 256
    w2s = e8(NE, D_HIDDEN, D_INTER // 32)
    return w1, w2, w1s, w2s


def make_routing(num_tokens, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    topk_ids = torch.randint(0, NE - 1, (num_tokens, TOPK), dtype=torch.int32, generator=g)
    topk_ids[:, -1] = NE - 1
    topk_weight = torch.rand((num_tokens, TOPK), dtype=torch.float32, generator=g)
    return topk_ids, topk_weight


def run(hidden, topk_ids, topk_weight, w):
    w1, w2, w1s, w2s = w
    return mxfp4_moe_run(
        hidden, w1, w2, TOPK, topk_ids, topk_weight,
        kernelName1=KN1, kernelName2=KN2,
        w1_scale=w1s, w2_scale=w2s,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args()
    NT, CH = args.tokens, args.chunk
    assert NT % CH == 0
    print(f"E192: tokens={NT} chunk={CH} nchunks={NT // CH}")

    g = torch.Generator(device="cuda").manual_seed(2)
    hidden = (torch.randn(NT, D_HIDDEN, dtype=torch.float32, generator=g) * 0.1).to(dtypes.bf16)
    topk_ids, topk_weight = make_routing(NT)
    w = make_weights()

    full = run(hidden, topk_ids, topk_weight, w)
    torch.cuda.synchronize()
    chunks = [run(hidden[c * CH:(c + 1) * CH].contiguous(),
                  topk_ids[c * CH:(c + 1) * CH].contiguous(),
                  topk_weight[c * CH:(c + 1) * CH].contiguous(), w)
              for c in range(NT // CH)]
    chunked = torch.cat(chunks, dim=0)
    torch.cuda.synchronize()

    a, b = full.float(), chunked.float()
    diff = (a - b).abs()
    bad = (diff > args.atol + args.rtol * b.abs()).any(dim=1)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    print(f"[E192 full vs chunk] mismatch rows={int(bad.sum())}/{NT} "
          f"max|Δ|={diff.max().item():.4e} finite={finite} "
          f"out_abs_mean={a.abs().mean().item():.4e}")
    ok = int(bad.sum()) == 0 and finite and a.abs().mean().item() > 0
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    assert ok


if __name__ == "__main__":
    main()
