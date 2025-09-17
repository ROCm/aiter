import triton
import triton.language as tl
import torch
from aiter.ops.triton.rmsnorm import rmsnorm3d_fwd_with_rope, _rmsnorm_forward_with_add
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

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
  cos_sin = torch.index_select(cos_sin_cache, 0, pos_embed)
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
