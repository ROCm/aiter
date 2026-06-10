# Direct gen-kernel (NOT dispatched/persistent) vs Triton, uniform + skew,
# do_bench. Answers: is jagged_dense_bmm_gen itself on par with Triton on skew?
import sys
import torch
import triton
import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import jagged_dense_bmm as gen_bmm
from generative_recommenders.ops.triton.triton_jagged import triton_jagged_dense_bmm_add_fwd

SHAPES = [(120, 256, 256), (120, 512, 512), (1024, 256, 256), (1024, 512, 512)]


def seqlens(B, msl, regime, seed):
    if regime == "uniform":
        mi = [msl] * B
    elif regime == "u4":  # heavy skew: msl*U^4, ~20% empty (bench_jdbba_vs_triton dist)
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(B, generator=g)
        t = (msl * (u ** 4)).floor().to(torch.int64)
        t[: max(1, B // 5)] = 0
        t[-1] = msl
        if B > 1:
            t[-2] = int(0.9 * msl)
        mi = [int(x) for x in t.tolist()]
    else:  # "lin": 0.95*U(1,msl), lighter skew
        g = torch.Generator().manual_seed(seed)
        mi = [max(1, int(0.95 * torch.rand(1, generator=g).item() * msl)) for _ in range(B)]
        mi[0] = 0
        mi[1] = 1
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + mi[i]
    return so, mi


def run(regime, seed=1234):
    print(f"\n=== regime={regime} (gen-direct vs triton, do_bench ms) ===")
    print(f"{'shape':22}{'meanMi':>8}{'empty%':>8}{'flydsl':>10}{'triton':>10}{'t/f':>8}")
    for (B, D, Kout) in SHAPES:
        N, K, msl = Kout, D, 7680
        so, mi = seqlens(B, msl, regime, seed)
        amsl = max(mi) if regime != "uniform" else msl
        L = int(so[-1])
        torch.manual_seed(0)
        jag = torch.randn(max(L, 1), K, dtype=torch.bfloat16).cuda()
        dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
        bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
        sod = so.cuda()
        dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
        bf = bias.reshape(B * N).contiguous()
        out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
        tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
        tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
        st = torch.cuda.current_stream()
        uni = regime == "uniform"

        def ffn():
            gen_bmm(tC, tA, dt, bf, sod, B, amsl, stream=st, uniform_seqlen=uni)

        so64 = sod.to(torch.int64)

        def tfn():
            return triton_jagged_dense_bmm_add_fwd(amsl, so64, jag, dense, bias)

        # correctness
        ref = torch.zeros(L, N, dtype=torch.bfloat16).cuda()
        for b in range(B):
            s, e = int(so[b]), int(so[b + 1])
            if e > s:
                ref[s:e] = (jag[s:e].float() @ dense[b].float() + bias[b].float()).to(torch.bfloat16)
        ffn(); torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(out[:L].float().flatten(), ref.float().flatten(), dim=0).item()
        assert cos > 0.999, f"FAIL cos={cos} {(B,D,Kout)} {regime}"

        fms = triton.testing.do_bench(ffn, warmup=25, rep=100)
        tms = triton.testing.do_bench(tfn, warmup=25, rep=100)
        mean_mi = L / B
        empty = 100.0 * sum(1 for m in mi if m == 0) / B
        print(f"B{B}_D{D}_K{Kout:<10}{mean_mi:>8.0f}{empty:>8.0f}{fms:>10.4f}{tms:>10.4f}{tms/fms:>8.2f}x")


if __name__ == "__main__":
    regimes = sys.argv[1:] or ["uniform", "u4", "lin"]
    for r in regimes:
        run(r)
