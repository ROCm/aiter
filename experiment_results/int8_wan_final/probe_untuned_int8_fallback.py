import torch
import aiter
from aiter.ops.shuffle import shuffle_weight

for N, K in ((5120, 5120), (13824, 5120), (5120, 13824)):
    for M in (512, 9450, 9472, 18900, 37800, 75600):
        x = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda")
        ws = shuffle_weight(w, layout=(16, 16))
        xs = torch.rand(M, 1, device="cuda", dtype=torch.float32) * 0.01
        wsc = torch.rand(1, N, device="cuda", dtype=torch.float32) * 0.01
        try:
            y = aiter.gemm_a8w8_bpreshuffle(x, ws, xs, wsc, None, torch.bfloat16)
            torch.cuda.synchronize()
            ref = (x.float() @ w.float().T) * xs * wsc
            cos = torch.nn.functional.cosine_similarity(
                y.float().flatten(), ref.flatten(), dim=0
            ).item()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            for _ in range(5):
                aiter.gemm_a8w8_bpreshuffle(x, ws, xs, wsc, None, torch.bfloat16)
            s.record()
            for _ in range(20):
                aiter.gemm_a8w8_bpreshuffle(x, ws, xs, wsc, None, torch.bfloat16)
            e.record()
            torch.cuda.synchronize()
            us = s.elapsed_time(e) / 20 * 1000
            tf = 2 * M * N * K / us / 1e6
            print(f"RESULT M={M} N={N} K={K} cos={cos:.6f} us={us:.1f} TOPS={tf:.0f}", flush=True)
        except Exception as e:
            print(f"RESULT M={M} N={N} K={K} FAIL {type(e).__name__} {str(e)[:300]}", flush=True)
