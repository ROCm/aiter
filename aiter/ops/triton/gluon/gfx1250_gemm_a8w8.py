from typing import Optional
import functools
import json
import triton
import torch
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

USE_TDM = True
TRANSPOSE_B = False

@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: gl.constexpr,
                NUM_BUFFERS: gl.constexpr, TRANSPOSE_B: gl.constexpr, pred=1):
    # pred is a hardware predicate passed to async_load for conditional execution without branch divergence
    # Convert boolean pred to i32 for hardware predicate (i1 -> i32)
    pred_i32 = pred.to(gl.int32) if hasattr(pred, 'to') else pred
    gl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    pred=pred_i32)
    if not TRANSPOSE_B:
        gl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred_i32)
    else:
        gl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred_i32)
    producer += 1
    return producer


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    K_DIM: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 function
    below

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 4],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2], [1, 0]],  # [2, 4] layout
        reg_bases=[],
        instr_shape=[16, 16, K_DIM],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    ) # CHECK VEC=16, rn doing ds_load_2addr_stride64_b64. Will ds_load_b128 be better?
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=16
    )

    # Load first blocks of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk))
    ) % M
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    if EVEN_K:
        a = gl.amd.gfx1250.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
        )
    else:
        a = gl.amd.gfx1250.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
            mask=(offs_ak[None, :] < K),
        )

    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn))
    ) % N
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if EVEN_K:
        b = gl.amd.gfx1250.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
        )
    else:
        b = gl.amd.gfx1250.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            mask=(offs_bk[:, None] < K),
        )

    # Load scales
    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
    )
    a_scale = gl.amd.gfx1250.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        mask=offs_a_scale < M,
    )
    b_scale = gl.amd.gfx1250.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        mask=offs_b_scale < N,
    )

    # Create shared memories
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # LDS write first block of A
    smem_a.store(a)

    acc_dtype = gl.float32 if a_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)

    # num_stages:2
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K) - 1):

        # advance pointers for block A and B
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

        # load next block of A
        if EVEN_K:
            a = gl.amd.gfx1250.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
            )
        else:
            a = gl.amd.gfx1250.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_SIZE_K),
            )

        # LDS write current block of B
        smem_b.store(b)

        # read current block of A from LDS
        cur_a = smem_a.load(layout=dot_a_layout)

        # load next block of B
        if EVEN_K:
            b = gl.amd.gfx1250.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
            )
        else:
            b = gl.amd.gfx1250.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_SIZE_K),
            )

        # read current block of B from LDS
        cur_b = smem_b.load(layout=dot_b_layout)

        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        
        # write next block of A to LDS
        smem_a.store(a)

    # ======= Epilogue ========

    # write last block of B to LDS
    smem_b.store(b)

    # read last blocks of A and B from LDS
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = smem_b.load(layout=dot_b_layout)

    acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)

    # apply scales to accumulator
    acc *= a_scale[:, None] * b_scale[None, :]

    # add bias
    if HAS_BIAS:
        bias = gl.amd.gfx1250.buffer_load(
            ptr=bias_ptr,
            offsets=offs_b_scale,
            mask=offs_b_scale < N,
        )
        acc = acc.to(bias_ptr.type.element_ty) + bias[None, :]

    c = acc.to(c_ptr.type.element_ty)

    # store block C back to global memory with masks
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
    )
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.gfx1250.buffer_store(stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask)

@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):

    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        if dev != "gfx1250":
            raise ValueError(
                "This kernel is not supported on this device (requires gfx1250)."
            )
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict["any"]

@triton.heuristics(
    {
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)

@gluon.jit
def _gemm_a8w8_kernel_async(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    K_DIM: gl.constexpr,
    TRANSPOSE_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 function
    below

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_K, 16]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0])
    if not TRANSPOSE_B:
        shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_N, 32]], [BLOCK_SIZE_K, BLOCK_SIZE_N], [1, 0])
        # this dont make sense need to look more into this
    else:
        shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_K, 16]], [BLOCK_SIZE_N, BLOCK_SIZE_K], [1, 0])

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, K),
                                                         strides=(stride_am, stride_ak), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), layout=shared_a)
    if not TRANSPOSE_B:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(K, N),
                                                         strides=(stride_bk, stride_bn), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), layout=shared_b)
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(N, K),
                                                         strides=(stride_bn, stride_bk), block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K), layout=shared_b)

    a_buffer = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2], [1, 0]],  # [2, 4] layout
        reg_bases=[],
        instr_shape=[16, 16, K_DIM],
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=16
    )

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    # Load scales
    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
    )
    a_scale = gl.amd.gfx1250.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        mask=offs_a_scale < M,
    )
    b_scale = gl.amd.gfx1250.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        mask=offs_b_scale < N,
    )

    producer=0
    consumer=0

    # prologue
    producer = issue_loads(producer, a_desc, b_desc, offs_am, offs_bn, a_buffer, b_buffer, BLOCK_SIZE_K, NUM_BUFFERS, TRANSPOSE_B)
    #change this to k_iter

    acc_dtype = gl.float32 if a_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)
    for _ in range(0, gl.cdiv(K, BLOCK_SIZE_K) - 1):
        producer = issue_loads(producer, a_desc, b_desc, offs_am, offs_bn, a_buffer, b_buffer, BLOCK_SIZE_K, NUM_BUFFERS, TRANSPOSE_B)
        gl.amd.gfx1250.tdm.async_wait(2) # wait for a and b async load
        a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=dot_a_layout)
        if not TRANSPOSE_B:
            b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=dot_b_layout)
        else:
            b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=dot_b_layout) #wmma layout expects (K, N)
        acc = gl.amd.gfx1250.wmma(a, b, acc)
        consumer += 1

    #epilogue
    gl.amd.gfx1250.tdm.async_wait(0)
    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=dot_a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=dot_b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=dot_b_layout)
    acc = gl.amd.gfx1250.wmma(a, b, acc)

    # apply scales to accumulator
    acc *= a_scale[:, None] * b_scale[None, :]

    # add bias
    if HAS_BIAS:
        bias = gl.amd.gfx1250.buffer_load(
            ptr=bias_ptr,
            offsets=offs_b_scale,
            mask=offs_b_scale < N,
        )
        acc = acc.to(bias_ptr.type.element_ty) + bias[None, :]

    c = acc.to(c_ptr.type.element_ty)

    # store block C back to global memory with masks
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
    )
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.gfx1250.buffer_store(stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask)
    
def gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output with shape (M, N) in higher precision format.
    """


    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"
    assert x.dtype == w.dtype, "Input types must be the same"
    M, K = x.shape
    N, K = w.shape
    
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)
    
    if USE_TDM:
        if not TRANSPOSE_B: # (K , N)
            w = w.T.contiguous()
        else: # (N, K)
            w = w.contiguous() # TDM needs the inner dimension to be contiguous

        x = x.contiguous()
        grid = (
            triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        )
        _gemm_a8w8_kernel_async[grid](
            x,
            w,
            x_scale,
            w_scale,
            bias,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0) if TRANSPOSE_B else w.stride(1),
            w.stride(1) if TRANSPOSE_B else w.stride(0),
            y.stride(0),
            y.stride(1),
            bias is not None,
            NUM_WARPS=config["num_warps"],
            **config,
            K_DIM=64 if x.dtype == torch.int8 else 128,
            TRANSPOSE_B=TRANSPOSE_B,
            NUM_BUFFERS=2,
        )

    else:

        # Transpose w (kernel expects (K, N))
        w = w.T

        grid = (
            triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        )
        _gemm_a8w8_kernel[grid](
            x,
            w,
            x_scale,
            w_scale,
            bias,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            bias is not None,
            NUM_XCDS=0,
            NUM_WARPS=config["num_warps"],
            K_DIM=64 if x.dtype == torch.int8 else 128,
            **config
        )

    return y
