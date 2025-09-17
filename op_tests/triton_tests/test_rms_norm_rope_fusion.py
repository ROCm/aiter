import triton
import triton.language as tl
import torch
from aiter.ops.triton.rmsnorm import rmsnorm3d_fwd_with_rope, _rmsnorm_forward_with_add
# from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms

def num_programs(x):
    return min(x.shape[0], get_num_sms())


def block_size(x):
    return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))


def use_blocked(x):
    return x.shape[1] > block_size(x)


from itertools import product
def get_hip_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [2, 4, 8, 16])]

def get_autotune_config():
    get_hip_autotune_config()

@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def rms_kernel(output_ptr, input_ptr, g_ptr, rsigma_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
               ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr,
               NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # as older version Triton doesn't support tl.assume and BUFF OPS, comment out for now
    # tl.assume(input_row_stride >= 0)
    # tl.assume(output_row_stride >= 0)
    # tl.assume(row_start >= 0)

    if USE_BLOCKED:

        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # sum_squares: tl.float32 = 0.
            # however, with type promoting rule in triton, sum_squares should be always fp32 with below init
            sum_squares = 0.
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            rsigma_output_ptr = rsigma_ptr + row_idx
            tl.store(rsigma_output_ptr, norm_factor)

            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)




@triton.autotune(configs=get_autotune_config(), key=['n_rows1', 'n_cols'], use_cuda_graph=False)
@triton.jit
def fused_rmsnorm_rope_kernel(
    # Pointers to matrices
    input_ptr1,
    input_ptr2,
    out_ptr1,
    out_ptr2,
    g_ptr1,
    g_ptr2,
    pos_id_ptr,
    cos_sin_cache_ptr,
    input_stride_1,
    input_stride_2,
    nhead_1,
    nhead_2,
    n_rows1,
    n_rows2,
    n_cols,
    epsilon,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKING: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
    IS_NEOX_STYPE: tl.constexpr,
):

    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    total_rows = n_rows1
    # tl.device_print("pid is: ", row_start)

    if not BLOCKING:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, total_rows, NUM_PRGMS, num_stages=2):
            if row_idx < n_rows1:
                pos_idx = row_idx // nhead_1
                # tl.device_print("row_idx: ", row_idx)
                input_ptrs = input_ptr1 + row_idx * input_stride_1 + col_offsets
                output_ptrs = out_ptr1 + row_idx * input_stride_1
                g_ptr = g_ptr1 + col_offsets
            else:
                new_row_idx = row_idx - n_rows1
                # tl.device_print("row_idx: ", new_row_idx)
                pos_idx = new_row_idx // nhead_2
                input_ptrs = input_ptr2 + new_row_idx * input_stride_2 + col_offsets
                output_ptrs = out_ptr2 + new_row_idx * input_stride_2
                g_ptr = g_ptr2 + col_offsets

            tl.multiple_of(input_ptrs, 16)
            tl.multiple_of(output_ptrs, 16)
            tl.multiple_of(g_ptr, 16)

            # rms_norm
            row = tl.load(input_ptrs, mask=mask)
            dtype = row.dtype
            row = row.to(tl.float32)
            g = tl.load(g_ptr, mask=mask)
            row_norm = row * row
            # row_norm = tl.where(mask, row_norm, 0.0)
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
            rms_norm = (row * norm_factor).to(dtype) * g


            # rope
            rope_mask = (col_offsets < 2 * ROPE_DIM) & mask
            pos_ids = tl.load(pos_id_ptr + pos_idx)
            if BLOCK_SIZE == ROPE_DIM * 2:
                cos_sin = tl.load(cos_sin_cache_ptr + pos_ids * 2 * ROPE_DIM + col_offsets, mask=rope_mask)
                if IS_NEOX_STYPE:
                    rms_norm_x1, rms_norm_x2 = rms_norm.reshape(2, BLOCK_SIZE // 2).permute(1, 0).split()
                else:
                    rms_norm_x1, rms_norm_x2 = rms_norm.reshape(BLOCK_SIZE // 2, 2).split()
                cos, sin = cos_sin.reshape(2, BLOCK_SIZE // 2).permute(1, 0).split()
                o1 = rms_norm_x1 * cos - rms_norm_x2 * sin
                o2 = rms_norm_x2 * cos + rms_norm_x1 * sin
                rms_norm_out = tl.join(o1, o2).permute(1, 0).reshape(BLOCK_SIZE)
                output_ptrs = output_ptrs + col_offsets
                tl.multiple_of(output_ptrs, 16)
                tl.store(output_ptrs, rms_norm_out, mask=mask)
            else:
                # tl.device_print("in other path")
                rope_idx = tl.arange(0, ROPE_DIM)
                rope_load_idx = tl.arange(0, ROPE_DIM * 2)
                cos_sin = tl.load(cos_sin_cache_ptr + pos_ids * ROPE_DIM * 2 + rope_load_idx)
                if IS_NEOX_STYPE:
                    rms_norm_x1 = tl.gather(rms_norm, rope_idx, axis=0)
                    rms_norm_x2 = tl.gather(rms_norm, rope_idx + ROPE_DIM, axis=0)
                else:
                    rms_norm_x1 = tl.gather(rms_norm, rope_idx * 2, axis=0)
                    rms_norm_x2 = tl.gather(rms_norm, rope_idx * 2 + 1, axis=0)
                cos, sin = cos_sin.reshape(2, ROPE_DIM).permute(1, 0).split()
                o1 = rms_norm_x1 * cos - rms_norm_x2 * sin
                o2 = rms_norm_x2 * cos + rms_norm_x1 * sin
                rms_norm_out = tl.join(o1, o2).permute(1, 0).reshape(ROPE_DIM * 2)
                output_rope_ptrs = output_ptrs + rope_load_idx
                output_non_rope_ptrs = output_ptrs + col_offsets
                rope_store_mask = col_offsets >= 2 * ROPE_DIM
                tl.store(output_rope_ptrs, rms_norm_out)
                tl.store(output_non_rope_ptrs, rms_norm, mask=rope_store_mask)
    else:
        # tl.device_print("in other path")
        for row_idx in tl.range(row_start, total_rows, NUM_PRGMS, num_stages=1):
            if row_idx < n_rows1:
                row_base_input_ptr = input_ptr1 + row_idx * input_stride_1
                row_base_output_ptr = out_ptr1 + row_idx * input_stride_1
                g_base_ptr = g_ptr1
                pos_id_ptr = pos_id_ptr + row_idx
            else:
                new_row_idx = row_idx - n_rows1
                row_base_input_ptr = input_ptr2 + new_row_idx * input_stride_2
                row_base_output_ptr = out_ptr2 + new_row_idx * input_stride_2
                g_base_ptr = g_ptr2
                pos_id_ptr = pos_id_ptr + new_row_idx
            n_cols_blks = n_cols // BLOCK_SIZE
            sum_sq = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                mask = col_offsets < (n_cols - blk_idx * BLOCK_SIZE)
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_base_input_ptr + cols
                tl.multiple_of(input_ptrs, 16)
                x = tl.load(input_ptrs, mask=mask, cache_modifier=".cg").to(tl.float32)
                sum_sq += tl.sum(x * x, axis=0)
            norm_factor = tl.rsqrt(sum_sq / cols + epsilon)
            pos_idx = tl.load(pos_id_ptr)
            cos_ptr = cos_sin_cache_ptr + pos_idx * 2 * ROPE_DIM
            sin_ptr = cos_sin_cache_ptr + pos_idx * 2 * ROPE_DIM + ROPE_DIM

            # process the rope part
            for blk_idx in tl.range(0, ROPE_DIM, BLOCK_SIZE, num_stages=2):
                # We assume the rope mask is valid on both cos and sin
                rope_mask = col_offsets + blk_idx < ROPE_DIM
                input_x1_ptrs = row_base_input_ptr + blk_idx + col_offsets
                input_x2_ptrs = row_base_input_ptr + blk_idx + ROPE_DIM + col_offsets
                cos_ptrs = cos_ptr + blk_idx + col_offsets
                sin_ptrs = sin_ptr + blk_idx + col_offsets
                g1_ptrs = g_base_ptr + col_offsets
                g2_ptrs = g_base_ptr + col_offsets + ROPE_DIM
                output_o1_ptrs = row_base_output_ptr + blk_idx + col_offsets
                output_o2_ptrs = row_base_output_ptr + blk_idx + ROPE_DIM + col_offsets
                tl.multiple_of(input_x1_ptrs, 16)
                tl.multiple_of(g1_ptrs, 16)
                tl.multiple_of(input_x2_ptrs, 16)
                tl.multiple_of(g2_ptrs, 16)
                tl.multiple_of(cos_ptr, 16)
                tl.multiple_of(sin_ptr, 16)
                x1 = tl.load(input_x1_ptrs, mask=rope_mask).to(tl.float32)
                g1 = tl.load(g1_ptrs, mask=rope_mask)
                x2 = tl.load(input_x2_ptrs, mask=rope_mask).to(tl.float32)
                g2 = tl.load(g2_ptrs, mask=rope_mask)
                cos = tl.load(cos_ptrs, mask=rope_mask)
                sin = tl.load(sin_ptrs, mask=rope_mask)

                rms_norm_x1 = x1 * norm_factor * g1
                rms_norm_x2 = x2 * norm_factor * g2
                o1 = rms_norm_x1 * cos - rms_norm_x2 * sin
                o2 = rms_norm_x2 * cos + rms_norm_x1 * sin

                tl.store(output_o1_ptrs, o1, mask=rope_mask)
                tl.store(output_o2_ptrs, o2, mask=rope_mask)

            # process the remaining part of rmsnorm
            for blk_idx in tl.range(ROPE_DIM * 2, n_cols, BLOCK_SIZE, num_stages=2):
                mask = col_offsets + blk_idx < n_cols
                cols = blk_idx + col_offsets
                input_ptrs = row_base_input_ptr + cols
                g_ptrs = g_base_ptr + cols
                output_ptrs = row_base_output_ptr + cols
                tl.multiple_of(input_ptrs, 16)
                tl.multiple_of(g_ptrs, 16)
                x = tl.load(input_ptrs, mask=mask).to(tl.float32)
                g = tl.load(g_ptrs, mask=mask).to(tl.float32)
                rms_norm = x * norm_factor * g

                tl.store(output_ptrs, rms_norm, mask=mask)


def rmsnorm3d_fwd_with_rope(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    epsilon: float,
    pos_embedding: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool
):
    """
    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result followed by a quantization.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N, K).
    - Input: The input tensor to be normalized with shape (M, N).
    - Residual_in: The tensor to be added to the Input tensor with shape (M, N).
    - Residual_out: The tensor in which the addition result will be stored with shape (M, N).
    - Yscale: The tensor where the scale for each row will be stored with shape (M, ).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.
    """
    import numpy as np
    # input1 = input1.view(-1, input1.shape[-1])
    # 
    # # add = torch.empty_like(input1)
    # rsigma = torch.empty([input1.shape[-1]], dtype=input1.dtype, device="cuda")
    # _rmsnorm_forward_with_add(input1, input1, input1, input1, weight1, rsigma, epsilon)
    # return input1, None
    n_row1 = int(np.prod(input1.shape[:-1]))
    n_row2 = int(np.prod(input2.shape[:-1]))
    head_dim = input1.shape[-1]
    n_head = input1.shape[-2]
    n_kv_head = input2.shape[-2]
    query_stride = input1.stride(-2)
    key_stride = input2.stride(-2)
    ROPE_DIM = cos_sin_cache.size(-1) // 2
    # def get_blk_size(head_dim):
    #     return min(1024, triton.next_power_of_2(head_dim))
    # WARP_SIZE = 64
    # NUM_WARPS = triton.next_power_of_2(head_dim // WARP_SIZE)
    input1 = input1.view(-1, input1.shape[-1])
    input2 = input2.view(-1, input2.shape[-1])
    out1 = torch.empty_like(input1)
    out2 = torch.empty_like(input2)
    BLOCK_SIZE = block_size(input1)
    USE_BLOCKED = use_blocked(input1)
    NUM_PRGMS = num_programs(input1)
    # BLOCKING = False
    # if BLOCK_SIZE < head_dim:
    #     BLOCKING = False
    # NUM_PRGMS = min(n_row1, get_num_sms())
    # print("num prog: ", NUM_PRGMS, n_row1 + n_row2)
    # print("num programs: ", NUM_PRGMS)
    grid = lambda meta: (NUM_PRGMS, 1, 1)  # noqa: E731
    assert input1.shape[-1] == head_dim
    assert input2.shape[-1] == head_dim
    assert weight1.shape[-1] == input1.shape[-1]
    assert weight2.shape[-1] == input1.shape[-1]
    rsigma = torch.empty((n_row1, ), device=input1.device, dtype=torch.float32)

    rms_kernel[grid](out1, input1, weight1, rsigma, input1.stride(0), out1.stride(0), n_row1, head_dim, epsilon, False,
                        BLOCK_SIZE, USE_BLOCKED, NUM_PRGMS)
    # print("blocking: ", BLOCKING)
    # fused_rmsnorm_rope_kernel[grid](
    #     input1,
    #     input2,
    #     out1,
    #     out2,
    #     weight1,
    #     weight2,
    #     pos_embedding,
    #     cos_sin_cache,
    #     query_stride,
    #     key_stride,
    #     n_head,
    #     n_kv_head,
    #     n_row1,
    #     0,
    #     head_dim,
    #     epsilon,
    #     ROPE_DIM=ROPE_DIM,
    #     BLOCK_SIZE=BLOCK_SIZE,
    #     BLOCKING=USE_BLOCKED,
    #     NUM_PRGMS=NUM_PRGMS,
    #     IS_NEOX_STYPE=is_neox_style
    # )
    return out1, out2

@triton.jit
def test_slice_tr(
  a_ptr,
  end,
  o_ptr,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(0)
  tl.device_print("pid: ", pid)
  # offset = tl.arange(0, BLOCK_SIZE)
  # mask = offset < 100
  # register = tl.load(a_ptr + offset, mask=mask)
  # tl.device_print("init data is: ", register)
  # for i in tl.range(pid, 35, 35):
  #   tl.device_print("idx is: ", i)
  # register = register.reshape(2, BLOCK_SIZE//2).permute(1, 0)
  # register_sum = tl.sum(register, axis=-1)
  # # register_view1, register_view2 = register.split()
  # # register_sum = register_view1 + register_view2
  # tl.device_print("registre is: ", register_sum)
  # register = tl.interleave(register_view1, register_view2)
  # register_view = register.view(32)
  # tl.device_print("register view:", register_view)
  # register = tl.flip(register, dim=1)
  # register = register.reshape(BLOCK_SIZE)
  # tl.store(o_ptr + offset, register, mask=mask)


def test_slice():
  a = torch.randn(128).cuda()
  b = torch.randn(128).cuda()
  BLOCK_SIZE=128
  grid = (1, 1, 1)
  test_slice_tr[grid](
    a,
    128,
    b,
    BLOCK_SIZE=128,
    num_warps=2
  )
  print(a)
  print(b)
  return b


def rocm_aiter_rms_norm_impl(x: torch.Tensor, weight: torch.Tensor,
                             variance_epsilon: float) -> torch.Tensor:

    import aiter as rocm_aiter
    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rocm_aiter.rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rocm_aiter.rms_norm(x, weight, variance_epsilon)

def ref_func(
    query: torch.Tensor,
    key: torch.Tensor,
    weight_query: torch.Tensor,
    weight_key: torch.Tensor, 
    epsilon: float,
    pos_embed: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool
):

  def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float):
    ret_type = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return (x).to(ret_type) * weight
  
  def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor, is_neox: bool):
    rope_dim = cos_sin.size(-1) // 2
    cos = cos_sin[..., :rope_dim].unsqueeze(-2)
    sin = cos_sin[..., rope_dim:].unsqueeze(-2)
    x_rotate = x[...,:rope_dim * 2]

    if is_neox:
      x1, x2 = torch.chunk(x_rotate, 2, dim=-1)
    else:
      x1, x2 = x_rotate[..., ::2], x_rotate[..., 1::2]
    ret = torch.empty_like(x_rotate)
    ret[..., :rope_dim] = x1 * cos - x2 * sin
    ret[..., rope_dim:] = x2 * cos + x1 * sin
    x[..., :rope_dim * 2] = ret
    return x

  # query = rms_norm(query, weight_query, epsilon)
  # key = rms_norm(key, weight_key, epsilon)
  query = rocm_aiter_rms_norm_impl(query, weight_query, epsilon)
  key = rocm_aiter_rms_norm_impl(key, weight_key, epsilon)
  # cos_sin = torch.index_select(cos_sin_cache, 0, pos_embed)
  # print("size of cos sin: ", cos_sin.size())
  # query = apply_rope(query, cos_sin, is_neox_style)
  # key = apply_rope(key, cos_sin, is_neox_style)
  return query, key




def test_rope(
    num_tokens,
    num_heads,
    num_kv_heads,
    head_dim,
    rope_dim,
    is_neox_style,
    dtype,
    max_positions = 10000,
    epsilon=1e-6
):
  import time
  query = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype).cuda()
  key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype).cuda()
  cos_sin_cache = torch.randn(max_positions, 2 * rope_dim, dtype=dtype).cuda()
  pos_embed = torch.randint(0, max_positions, (num_tokens, )).cuda()
  weight_query = torch.ones(head_dim, dtype=dtype).cuda()
  weight_key = torch.ones(head_dim, dtype=dtype).cuda()
  # opt_ref_func = torch.compile(ref_func)
  opt_ref_func = ref_func
  # warm up
  for i in range(3):
    q_ref, k_ref = opt_ref_func(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  
  for i in range(3):
    q_out, k_out = rmsnorm3d_fwd_with_rope(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  
  torch.cuda.synchronize()
  time_start = time.time()
  for i in range(10):
    q_ref, k_ref = opt_ref_func(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  torch.cuda.synchronize()
  elaps_ref = time.time() - time_start
  time_start = time.time()
  for i in range(10):
    q_out, k_out = rmsnorm3d_fwd_with_rope(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  torch.cuda.synchronize()
  elaps_fusion = time.time() - time_start
  def normal_kernel():
    return opt_ref_func(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  def fused_kenrel():
    return rmsnorm3d_fwd_with_rope(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  q_out, k_out = rmsnorm3d_fwd_with_rope(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  # q_ref, k_ref = ref_func(query, key, weight_query, weight_key, epsilon, pos_embed, cos_sin_cache, is_neox_style)
  # assert torch.allclose(q_out, q_ref, rtol=1e-3, atol=1e-3)
  # assert torch.allclose(k_out, k_ref, rtol=1e-3, atol=1e-3)
  print("q out: ", q_out.size())
  print("q ref: ", q_ref.size())
  # abs_diff = torch.abs(q_out - q_ref)
  # diff_mask = abs_diff > 1e-3
  # print(abs_diff[diff_mask])
  # print("idx: ", torch.nonzero(diff_mask))
  # print("diff ratio: ", abs_diff[diff_mask].numel() / abs_diff.numel())
  print("ref time: ", elaps_ref)
  print("fuse time: ", elaps_fusion)
  normal_ms = triton.testing.do_bench(normal_kernel)
  fused_ms = triton.testing.do_bench(fused_kenrel)
  print("normal ms: ", normal_ms)
  print("fused ms: ", fused_ms)
  

test_rope(
  1024,
  32,
  4,
  128,
  64,
  False,
  torch.bfloat16
)
# test_slice()
