from functools import partial

import torch
import triton

from aiter.ops.triton.routing import (
    _routing_sigmoid_top1_kernel,
    routing_sigmoid_top1,
    torch_routing_sigmoid_top1,
)


def _get_compiled(fn):
    compiled_fn = torch.compile(
        fn, backend="inductor", fullgraph=True, options={"max_autotune": True}
    )
    return compiled_fn


def benchmark(M, N, K):
    TOPK = 1

    torch.manual_seed(7)

    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn((M, K), dtype=dtype, device=device)
    w = torch.randn((K, N), dtype=dtype, device=device) * 0.1

    dummy_ids = torch.ones((M, 1), dtype=torch.int32, device="cuda") * N
    dummy_weights = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    _eager = partial(
        torch_routing_sigmoid_top1, dummy_ids=dummy_ids, dummy_weights=dummy_weights
    )
    _compiled = _get_compiled(_eager)

    def eager_fn():
        return _eager(x, w, TOPK)

    def triton_fn():
        return routing_sigmoid_top1(x, w, TOPK)

    def compile_fn():
        return _compiled(x, w, TOPK)

    # warmup
    for _ in range(5):
        eager_fn()
        triton_fn()
        compile_fn()

    with torch.cuda.stream(torch.cuda.Stream()):
        ms_eager_time = triton.testing.do_bench_cudagraph(eager_fn)

    with torch.cuda.stream(torch.cuda.Stream()):
        ms_triton_time = triton.testing.do_bench_cudagraph(triton_fn)

    with torch.cuda.stream(torch.cuda.Stream()):
        ms_compile_time = triton.testing.do_bench_cudagraph(compile_fn)

    print(
        f"{M=} {K=} {N=} {TOPK=}, "
        f"{ms_eager_time=:.3f}, {ms_triton_time=:.3f}, {ms_compile_time=:.3f}, "
        f"speedup_vs_eager: {ms_eager_time / ms_triton_time:.3f}, "
        f"speedup_vs_compile: {ms_compile_time / ms_triton_time:.3f}\n"
        f"best triton config: {getattr(_routing_sigmoid_top1_kernel, 'best_config', "")}"
    )


def benchmkar_prefill():
    print("=== PREFILL SHAPEs ===")
    benchmark(M=1024, K=5120, N=128)
    benchmark(M=1024, K=5120, N=16)
    benchmark(M=2048, K=5120, N=128)
    benchmark(M=2048, K=5120, N=16)
    benchmark(M=4096, K=5120, N=128)
    benchmark(M=4096, K=5120, N=16)
    benchmark(M=8192, K=5120, N=128)
    benchmark(M=8192, K=5120, N=16)


def benchmark_decode():
    print("=== DECODE SHAPEs ===")
    benchmark(M=64, K=5120, N=128)
    benchmark(M=64, K=5120, N=16)
    benchmark(M=128, K=5120, N=128)
    benchmark(M=128, K=5120, N=16)
    benchmark(M=256, K=5120, N=128)
    benchmark(M=256, K=5120, N=16)


def main():
    benchmkar_prefill()
    # no gain for decode shape
    benchmark_decode()


if __name__ == "__main__":
    main()
