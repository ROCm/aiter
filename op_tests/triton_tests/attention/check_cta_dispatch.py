"""
Verify whether Triton CTA dispatch on MI300X follows the assumed pattern:
  CTA program_id i -> CU (i % 304) -> XCD (i % 304) // 38

Reads hardware CU ID from AMDGCN HW_ID register for each CTA and records it.
Then checks: does actual XCD match (program_id % 304) // 38 ?

Usage:
  python op_tests/triton_tests/attention/check_cta_dispatch.py

Expected output if assumption holds:
  XCD mismatch rate: 0.00%  (routing formula is correct)

Expected output if assumption fails:
  XCD mismatch rate: XX.X%  (routing formula is wrong -> xcd_privatized is broken)
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import triton
import triton.language as tl


@triton.jit
def _read_cu_id(
    CuId_ptr,    # [N] int32 output: actual CU ID for each CTA
    XcdId_ptr,   # [N] int32 output: actual XCD ID for each CTA
    N: tl.constexpr,
    CUS_PER_XCD: tl.constexpr,  # 38 for MI300X
):
    """
    One CTA per program_id. Reads the hardware CU ID from the AMDGCN HW_ID
    SGPR register (bits [13:10] = CU_ID within SE, bits [17:14] = SE_ID on
    some generations; on gfx942 the combined field gives the physical CU).

    We use s_getreg_b32 with HW_REG_HW_ID (reg=4).
    Bit layout for gfx942 HW_ID:
      [3:0]   WAVE_ID
      [5:4]   SIMD_ID
      [9:6]   CU_ID    (within the shader array)
      [13:10] SA_ID    (shader array within XCD)
      [17:14] SE_ID    (shader engine = XCD index on MI300X)
    """
    pid = tl.program_id(0)

    # Read HW_ID register: hwreg(HW_REG_HW_ID=4, offset=0, size=32)
    hw_id = tl.inline_asm_elementwise(
        "s_getreg_b32 $0, hwreg(4, 0, 32)",
        "=s",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )

    # Extract SE_ID = XCD index on MI300X (bits [17:14])
    xcd_id = (hw_id >> 14) & 0xF

    # Extract combined CU index within XCD (bits [9:6] = CU_ID, bits [13:10] = SA_ID)
    cu_within_sa = (hw_id >> 6) & 0xF   # CU within shader array (0-9 on MI300X)
    sa_id        = (hw_id >> 10) & 0xF  # shader array within XCD (0-3 on MI300X)
    cu_id_local  = sa_id * 10 + cu_within_sa  # approximate physical CU within XCD

    tl.store(CuId_ptr + pid, cu_id_local)
    tl.store(XcdId_ptr + pid, xcd_id)


def main():
    device = "cuda"
    # Use same grid as the dKV kernel
    T = 4096
    num_cus = 304
    cus_per_xcd = 38
    num_xcd = 8

    cu_ids  = torch.zeros(T, dtype=torch.int32, device=device)
    xcd_ids = torch.zeros(T, dtype=torch.int32, device=device)

    # Warm up to ensure all CTAs are dispatched in a single wave pattern
    for _ in range(3):
        _read_cu_id[(T,)](cu_ids, xcd_ids, N=T, CUS_PER_XCD=cus_per_xcd,
                          num_warps=1)
    torch.cuda.synchronize()

    _read_cu_id[(T,)](cu_ids, xcd_ids, N=T, CUS_PER_XCD=cus_per_xcd,
                      num_warps=1)
    torch.cuda.synchronize()

    cu_ids  = cu_ids.cpu()
    xcd_ids = xcd_ids.cpu()

    # What xcd_privatized assumes:
    predicted_xcd = (torch.arange(T) % num_cus) // cus_per_xcd

    mismatches = (xcd_ids != predicted_xcd).sum().item()
    print(f"\nMI300X CTA dispatch analysis (grid={T}, 304 CUs, 38 CUs/XCD)")
    print(f"{'pid':>6s}  {'actual_xcd':>10s}  {'predicted_xcd':>13s}  {'match':>5s}")
    print("-" * 42)
    # Print first 32 CTAs as sample
    for i in range(min(32, T)):
        match = "OK" if xcd_ids[i] == predicted_xcd[i] else "MISMATCH"
        print(f"{i:6d}  {xcd_ids[i].item():10d}  {predicted_xcd[i].item():13d}  {match:>8s}")

    print(f"\nTotal CTAs:      {T}")
    print(f"XCD mismatches:  {mismatches}")
    print(f"Mismatch rate:   {100.0 * mismatches / T:.2f}%")

    if mismatches == 0:
        print("\nConclusion: dispatch assumption HOLDS -> H1 ruled out.")
        print("  xcd_privatized routing is correct; bottleneck is intra-XCD L2 atomic serialization (H2).")
    else:
        print(f"\nConclusion: dispatch assumption is WRONG ({mismatches}/{T} CTAs on wrong XCD) -> H1 confirmed.")
        print("  xcd_privatized writes to wrong copies -> cross-XCD contention remains.")

    # Also show actual XCD distribution
    print(f"\nActual XCD distribution:")
    for x in range(num_xcd):
        count = (xcd_ids == x).sum().item()
        print(f"  XCD {x}: {count} CTAs")


if __name__ == "__main__":
    main()
