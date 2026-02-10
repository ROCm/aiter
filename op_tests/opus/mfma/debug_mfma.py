# SPDX-License-Identifier: MIT
# Debug MFMA 32x32x8 (block_v2, swap_ab): kernel C = B @ A^T. Compare with torch.
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def main():
    import torch

    try:
        import opus_mfma
    except ModuleNotFoundError:
        import subprocess

        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=_THIS_DIR,
            check=True,
        )
        import opus_mfma

    M, N, K = 32, 32, 8
    device = torch.device("cuda")
    dtype = torch.float16

    # Test 1: A=1, B=1 -> C_ref[i,j] = 8 for all i,j
    A1 = torch.ones(M, K, device=device, dtype=dtype)
    B1 = torch.ones(N, K, device=device, dtype=dtype)
    C1 = torch.empty(M, N, device=device, dtype=dtype)
    opus_mfma.run_mfma_32x32x8_f16(A1, B1, C1)
    C1_ref = torch.mm(B1.float(), A1.float().t()).to(dtype)  # block_v2: C = B @ A^T

    print("=== A=1, B=1 (expect C = 8 everywhere) ===")
    print("Kernel C first 8x8:")
    print(C1.float()[:8, :8].cpu().numpy())
    print("Reference C_ref first 8x8:")
    print(C1_ref.float()[:8, :8].cpu().numpy())
    diff1 = (C1.float() - C1_ref.float()).abs()
    print("Max diff:", diff1.max().item())
    print(
        "Min kernel C:",
        C1.float().min().item(),
        "Max kernel C:",
        C1.float().max().item(),
    )
    print("Positions where kernel != 8 (first 20):")
    wrong = (C1.float().abs() - 8.0).abs().gt(0.01)
    idx = wrong.nonzero(as_tuple=False)[:20]
    for i in range(idx.size(0)):
        r, c = idx[i, 0].item(), idx[i, 1].item()
        print(f"  ({r},{c}) kernel={C1[r,c].float().item():.4f} ref=8.0")

    # Test 2: Random (same as main test) - print first row/col
    torch.manual_seed(12345)
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(N, K, device=device, dtype=dtype)
    C = torch.empty(M, N, device=device, dtype=dtype)
    opus_mfma.run_mfma_32x32x8_f16(A, B, C)
    C_ref = torch.mm(B.float(), A.float().t()).to(dtype)  # block_v2: C = B @ A^T
    print("\n=== Random A,B: first row and first column ===")
    print("Kernel C[0,:]:", C[0, :].float().cpu().numpy())
    print("Ref   C[0,:]:", C_ref[0, :].float().cpu().numpy())
    print("Kernel C[:,0]:", C[:, 0].float().cpu().numpy())
    print("Ref   C[:,0]:", C_ref[:, 0].float().cpu().numpy())
    print("Max diff:", (C.float() - C_ref.float()).abs().max().item())


if __name__ == "__main__":
    main()
