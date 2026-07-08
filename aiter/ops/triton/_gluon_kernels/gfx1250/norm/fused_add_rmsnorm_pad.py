import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor * weight
    return rms_norm


@gluon.jit
def _gluon_fused_add_rmsnorm_pad_kernel(
    x_ptr,
    res_ptr,
    out_ptr,
    res_out_ptr,
    weight_ptr,
    eps,
    M,
    N,
    N_out,
    x_stride_m,
    x_stride_n,
    res_stride_m,
    res_stride_n,
    out_stride_m,
    out_stride_n,
    res_out_stride_m,
    res_out_stride_n,
    HAS_RES: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
):
    start_pid = gl.program_id(0)

    # create 1d layout for n_offs
    n_offs_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_SIZE_N + 127) // 128],  # size per thread
        [32],  # threads per warp
        [4],  # warps per cta
        [0],  # order
    )

    # create shared layouts for x, res, weights, out
    sharedLayout1D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # create tensor descriptors for x, res, weights, out
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x_ptr,
        [M, N],
        [x_stride_m, x_stride_n],
        [1, BLOCK_SIZE_N],
        sharedLayout1D,
    )
    weights_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        weight_ptr,
        [1, N],
        [0, 1],
        [1, BLOCK_SIZE_N],
        sharedLayout1D,
    )
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        out_ptr,
        [M, N_out],
        [out_stride_m, out_stride_n],
        [1, BLOCK_SIZE_N],
        sharedLayout1D,
    )

    # create tensor descriptors for res, res_out if applicable
    if HAS_RES:
        res_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            res_ptr,
            [M, N],
            [res_stride_m, res_stride_n],
            [1, BLOCK_SIZE_N],
            sharedLayout1D,
        )
        res_out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            res_out_ptr,
            [M, N],
            [res_out_stride_m, res_out_stride_n],
            [1, BLOCK_SIZE_N],
            sharedLayout1D,
        )

    # allocate shared memory for x, weights, out
    smemX = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [1, BLOCK_SIZE_N], sharedLayout1D
    )
    smemWeights = gl.allocate_shared_memory(
        weight_ptr.dtype.element_ty, [1, BLOCK_SIZE_N], sharedLayout1D
    )
    smemOut = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [1, BLOCK_SIZE_N], sharedLayout1D
    )

    # allocate shared memory for res, res_out if applicable
    if HAS_RES:
        smemRes = gl.allocate_shared_memory(
            res_ptr.dtype.element_ty, [1, BLOCK_SIZE_N], sharedLayout1D
        )
        smemResOut = gl.allocate_shared_memory(
            res_out_ptr.dtype.element_ty, [1, BLOCK_SIZE_N], sharedLayout1D
        )

    # load x, res (if applicable), and weights
    gl.amd.gfx1250.tdm.async_load(x_desc, [start_pid, 0], smemX)
    gl.amd.gfx1250.tdm.async_load(weights_desc, [0, 0], smemWeights)
    if HAS_RES:
        gl.amd.gfx1250.tdm.async_load(res_desc, [start_pid, 0], smemRes)
    gl.amd.gfx1250.tdm.async_wait(0)
    smemX_1d = smemX.reshape([BLOCK_SIZE_N])
    x = smemX_1d.load(n_offs_layout).to(gl.float32)

    # reshape res and add to x if applicable
    if HAS_RES:
        smemRes_1d = smemRes.reshape([BLOCK_SIZE_N])
        res = smemRes_1d.load(n_offs_layout).to(gl.float32)
        x = x + res

    smemWeights_1d = smemWeights.reshape([BLOCK_SIZE_N])
    w = smemWeights_1d.load(n_offs_layout).to(gl.float32)
    # compute rmsnorm
    out = _rmsmorm_op(x, w, N, eps).to(out_ptr.dtype.element_ty)
    out = out.reshape([1, BLOCK_SIZE_N])

    # write out to LDS
    smemOut.store(out.to(out_ptr.dtype.element_ty))
    if HAS_RES:
        x = x.reshape([1, BLOCK_SIZE_N])
        smemResOut.store(x.to(res_out_ptr.dtype.element_ty))

    gl.amd.gfx1250.tdm.async_store(out_desc, [start_pid, 0], smemOut)
    if HAS_RES:
        gl.amd.gfx1250.tdm.async_store(res_out_desc, [start_pid, 0], smemResOut)

    gl.amd.gfx1250.tdm.async_wait(0)


_KERNEL_MAP = {
    "tdm": _gluon_fused_add_rmsnorm_pad_kernel,
}
