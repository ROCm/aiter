

import torch
import argparse
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.triton_tests.test_fp8_mqa_logits import per_custom_dims_cast_to_fp8, generate_cp_test_data

def calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms):
    time_s = time_ms * 1e-3
    causal = True
    start_inds = start_inds.to("cpu")
    end_inds = end_inds.to("cpu")
    total_flops = 0.0
    for i in range(len(start_inds)):
        start = start_inds[i]
        end = end_inds[i]

        total_flops += (
            2.0 * num_heads_q * head_dim * (end - start)
        )

    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)

    return tflops
def main():
    parser = argparse.ArgumentParser(description="FP8 MQA Logits Benchmark")
    parser.add_argument('--num_heads_q', type=int, default=64, help='num. q heads')
    parser.add_argument('--head_dim', type=int, default=128, help='head dim size')
    parser.add_argument('--seq_q_l', type=int, default=1024, help='Input sequence length')
    parser.add_argument('--seq_kv_l', type=int, default=1024, help='Output sequence length')
    args = parser.parse_args()

    cache_size = eval(args.cache_size)
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
    clear_cache = lambda: cache.zero_()

    seq_len = args.seq_q_l
    seq_len_kv = args.seq_kv_l
    num_heads, head_dim = args.num_heads_q, args.head_dim
    repeat = args.repeat

    disable_cp = True
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)


    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)


    q_fp8 = q.to(e4m3_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0, ), False)

    func = lambda: fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)


    time_ms = triton.testing.do_bench(func, warmup=25, rep=100)
    tflops = calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms)


if __name__ == "__main__":
    main()

