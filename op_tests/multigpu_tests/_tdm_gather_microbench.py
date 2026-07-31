"""Validate TDM gather-store from padded LDS rows."""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, tdm_ops
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    lds_store_b32_raw,
    workgroup_barrier,
)
from aiter.ops.flydsl.kernels.tdm_gather_shim import (
    make_tensor_gather_descriptor,
    tensor_store_gather,
)

R = 8      # rows to scatter (<=8 for 32-bit index)
N = 32     # cols per row
ELEM_BYTES = 4
ROW_BYTES = N * ELEM_BYTES
LDS_ROW_BYTES = ((ROW_BYTES + 15) // 16) * 16
if LDS_ROW_BYTES % 32 == 0:
    LDS_ROW_BYTES += 16
P = LDS_ROW_BYTES // ELEM_BYTES
M = 16     # dst rows
PER = (R * P) // 32  # elements each of the 32 lanes writes


@flyc.jit
def run(dst: fx.Tensor, idx: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
    @flyc.kernel(name="tdm_gather_mb", known_block_size=[32, 1, 1])
    def kernel(dst: fx.Tensor, idx: fx.Pointer):
        tid = fx.thread_idx.x
        base_ptr = fx.SharedAllocator(static=False).allocate(R * P * ELEM_BYTES)._ptr
        lds_base = fx.index_cast(T.index, fx.ptrtoint(base_ptr))

        for j in range_constexpr(PER):
            e = tid * PER + j
            lds_store_b32_raw(lds_base, e * ELEM_BYTES, e)
        workgroup_barrier()

        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=4,
        )
        idx_iter = fx.recast_iter(i32_ptr, idx)
        row_indices = [
            fx.ptr_load(idx_iter + r, result_type=T.i32) for r in range_constexpr(R)
        ]

        desc = make_tensor_gather_descriptor(
            global_ptr=dst,
            lds_base_idx=lds_base,
            row_indices=row_indices,
            row_width=P,
            tensor_dim0=N,
            tensor_dim1=M,
            stride=N,
            elem_bytes=ELEM_BYTES,
            index_size=32,
        )
        tensor_store_gather(desc)
        tdm_ops.tensor_wait(0)

    kernel(dst, idx).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


if __name__ == "__main__":
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    dev = torch.device("cuda")
    dst = torch.full((M, N), -1, dtype=torch.int32, device=dev)
    # unique dst rows: a permutation prefix of [0, M)
    idx = torch.randperm(M, device=dev)[:R].to(torch.int32)

    run(dst, ptr_arg(idx))
    torch.cuda.synchronize()

    dst_c = dst.cpu()
    idx_c = idx.cpu()
    ref = torch.full((M, N), -1, dtype=torch.int32)
    for r in range(R):
        ref[idx_c[r]] = torch.arange(r * P, r * P + N, dtype=torch.int32)

    if torch.equal(dst_c, ref):
        print("PASS: TDM gather-store scattered LDS rows to Dst[Idx[r]] correctly")
    else:
        diff = (dst_c != ref)
        print(f"FAIL: {diff.sum().item()}/{dst_c.numel()} elems differ")
        # show first few mismatching rows
        for r in range(M):
            if diff[r].any():
                print(f"  row{r}: got={dst_c[r][:8].tolist()} exp={ref[r][:8].tolist()}")
