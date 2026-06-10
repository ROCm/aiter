# Skew experiment: XCD-aware persistent clone vs baseline persist_dev vs Triton.
# Sweeps xcd_c on the clone. do_bench, kernel-only (pre-built CUM passed in so the
# timing isolates the visiting-order change, not the CUM build).
import sys
import torch
import triton
import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels import jagged_dense_bmm_persist_dev as base
from aiter.ops.flydsl.kernels import jagged_dense_bmm_persist_xcd as xcd
from generative_recommenders.ops.triton.triton_jagged import triton_jagged_dense_bmm_add_fwd

SHAPES = [(120, 256, 256), (120, 512, 512), (1024, 256, 256), (1024, 512, 512)]
CSWEEP = [0, 8, 16, 32, 60, 120]  # 0 == baseline persist_dev (no remap)


def skew_seqlens(B, msl, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(B, generator=g)
    t = (msl * (u ** 4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0
    t[-1] = msl
    if B > 1:
        t[-2] = int(0.9 * msl)
    mi = [int(x) for x in t.tolist()]
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + mi[i]
    return so, mi


def run(seed=1234):
    print(f"\n=== skew seed={seed} (do_bench ms, kernel-only; lower=better) ===")
    hdr = f"{'shape':20}{'empty%':>7}{'triton':>9}{'base':>9}"
    for c in CSWEEP[1:]:
        hdr += f"{'c='+str(c):>9}"
    print(hdr)
    for (B, D, Kout) in SHAPES:
        N, K, msl = Kout, D, 7680
        so, mi = skew_seqlens(B, msl, seed)
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
        cum = base.build_cum(sod, B, N)

        # reference
        ref = torch.zeros(L, N, dtype=torch.bfloat16).cuda()
        for b in range(B):
            s, e = int(so[b]), int(so[b + 1])
            if e > s:
                ref[s:e] = (jag[s:e].float() @ dense[b].float() + bias[b].float()).to(torch.bfloat16)

        def check(fn):
            out.zero_(); fn(); torch.cuda.synchronize()
            return torch.nn.functional.cosine_similarity(
                out[:L].float().flatten(), ref.float().flatten(), dim=0).item()

        so64 = sod.to(torch.int64)
        tfn = lambda: triton_jagged_dense_bmm_add_fwd(msl, so64, jag, dense, bias)
        tms = triton.testing.do_bench(tfn, warmup=25, rep=100)

        bfn = lambda: base.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, cum=cum)
        assert check(bfn) > 0.999, f"base FAIL {(B,D,Kout)}"
        bms = triton.testing.do_bench(bfn, warmup=25, rep=100)

        empty = 100.0 * sum(1 for m in mi if m == 0) / B
        line = f"B{B}_D{D}_K{Kout:<7}{empty:>7.0f}{tms:>9.4f}{bms:>9.4f}"
        for c in CSWEEP[1:]:
            xfn = lambda c=c: xcd.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, cum=cum, xcd_c=c)
            cos = check(xfn)
            if cos < 0.999:
                line += f"{'FAIL':>9}"
                continue
            xms = triton.testing.do_bench(xfn, warmup=25, rep=100)
            line += f"{xms:>9.4f}"
        print(line)


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [1234]
    for s in seeds:
        run(s)
