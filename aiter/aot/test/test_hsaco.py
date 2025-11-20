from csrc.cpp_itfs.utils import compile_hsaco_from_triton, run_hsaco
from aiter.aot.test.matmul_fp16 import matmul_fp16
import torch

if __name__ == "__main__":
    func_name = compile_hsaco_from_triton(
        matmul_fp16,
        torch.float16,
        torch.float16,
        torch.float16,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        16,
        16,
        16,
        grid=(1, 1, 1),
        num_warps=4,
        num_stages=2,
    )
    A = torch.ones(1024, 1024, dtype=torch.float16, device="cuda")
    B = torch.ones(1024, 1024, dtype=torch.float16, device="cuda")
    C = torch.zeros(1024, 1024, dtype=torch.float16, device="cuda")
    run_hsaco(
        matmul_fp16.fn.__name__,
        C,
        A,
        B,
        1024,
        1024,
        1024,
        1024,
        1,
        1024,
        1,
        1024,
        1,
        grid=((1024 + 16 - 1) / 16, (1024 + 16 - 1) / 16, 1),
        constants={"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16},
    )
    print(C)
