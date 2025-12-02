import torch
import aiter
from aiter.test_common import checkAllclose, perftest, run_perftest
from aiter import dtypes
from aiter import per_tensor_quant

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
import math
# current supported case in ps decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16
# qdtype fp8, kdtype fp8: nhead16, nhead128
# qdtype fp8, kdtype bf16: nhead16


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8_q=False,
    is_fp8_kvc=False,
    q_scale=None,
    kv_scale=None,
):

    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))

    l = attn_weights_exp.sum(-1)

    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtype)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0, 1).unsqueeze(-1)

    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse

@perftest()
def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        # print("torch q", q.shape)
        # print("torch k", k.shape)
        # print("torch v", v.shape)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


batch_size = 1
seq_len = 522
nhead = 1
nhead_kv = 1
qk_head_dim = 192  # 128 (kv_lora_rank) + 64 (qk_rope_head_dim)
v_head_dim = 128
page_size = 1
is_causal = True


total_q = batch_size * seq_len
num_page = math.ceil(total_q / page_size)

# num_page = 256
total_kv = total_q


# seq_len_qo = [seq_len] * batch_size
# seq_len_kv = [seq_len] * batch_size

# qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
# kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)

# for i in range(batch_size):
#     qo_indptr[i + 1] = qo_indptr[i] + seq_len_qo[i]
#     kv_indptr[i + 1] = kv_indptr[i] + seq_len_kv[i]


qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)

# ctx_lens = 256
for i in range(batch_size):
    # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
    # seq_lens_qo[i] = max(
    #     min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
    # )
    seq_lens_kv.fill_(seq_len)
    seq_lens_qo.fill_(seq_len)


kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)

# kv_indices = torch.arange(0, kv_indptr[-1].item(), dtype=torch.int)
kv_indices = torch.randint(0, num_page, (num_page,), dtype=torch.int32)
# Or use random: torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)

qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
max_seqlen_qo = seq_lens_qo.max().item()
max_seqlen_kv = seq_lens_kv.max().item()
total_qo = qo_indptr[-1].item()
total_kv = kv_indptr[-1].item()

# Q_bf16 = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)
# K_bf16 = torch.randn((total_kv, nhead_kv, qk_head_dim), dtype=torch.bfloat16)
# V_bf16 = torch.randn((total_kv, nhead_kv, v_head_dim), dtype=torch.bfloat16)
Q_bf16 = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)
K_bf16 = torch.randn((total_kv, nhead_kv, qk_head_dim), dtype=torch.bfloat16)
V_bf16 = K_bf16[:, :, :v_head_dim].contiguous()

quant_dtype = dtypes.fp8
q_quant, q_scale = per_tensor_quant(Q_bf16, scale=torch.tensor(1), quant_dtype=quant_dtype)
k_quant, k_scale = per_tensor_quant(K_bf16, scale=torch.tensor(1), quant_dtype=quant_dtype)
v_quant, v_scale = per_tensor_quant(V_bf16, scale=torch.tensor(1), quant_dtype=quant_dtype)

output = torch.empty((total_qo, nhead, v_head_dim), dtype=torch.bfloat16)

print("\nTensor shapes:")
print(f"  Q_bf16: {Q_bf16.shape}, q_quant: {q_quant.shape}")
print(f"  K_bf16: {K_bf16.shape}, k_quant: {k_quant.shape}")
print(f"  V_bf16: {V_bf16.shape}, v_quant: {v_quant.shape}")
print(f"  num_page: {num_page}, total_kv: {total_kv}")
# num_cus = torch.cuda.get_device_properties(0).multi_processor_count

# work_indptr = torch.tensor([0, 1] + [1] * (num_cus - 1), dtype=torch.int32)

available_tgs = 2
work_indptr_ = [0]
current_work_id = 0
for tg_idx in range(available_tgs):
    current_work_id += 1
    work_indptr_.append(current_work_id)
work_indptr = torch.tensor(work_indptr_, dtype=torch.int32, device=qo_indptr.device)

# work_indptr = torch.tensor([0, 1], dtype=torch.int32)
# work_info_set = torch.tensor(
#     [0, -1, 0, 256, 0, 256, 0, 655360], dtype=torch.int32)

# work_indptr = torch.tensor([0, 1], dtype=torch.int32)
# work_info_set = torch.tensor(
#     [0, -1, 0, 256, 0, 256, 0, 655360], dtype=torch.int32)


# work_info_set = torch.tensor(
#     [0, -1, 0, 256, 0, 256, 0, 655360], dtype=torch.int32)

work_info_set = torch.tensor([[0, -1, 0, 256, 0, 522, 0, 655360],
                              [0, 0, 256, 512, 0, 384, 0, 655360],
                              [0, 256, 256, 512, 384, 522, 0, 655360],
                              [0, -1, 512, 522, 0, 522, 0, 655360]], dtype=torch.int32)


# work_info_set = torch.zeros((1, 8), dtype=torch.int32, device="cuda")
# for i in range(batch_size):
#     work_info_set[i, 0] = i                        # bs_index
#     work_info_set[i, 1] = -1                       # partial_index: -1 means no split
#     work_info_set[i, 2] = qo_indptr[i].item()      # q_start
#     work_info_set[i, 3] = qo_indptr[i+1].item()    # q_end
#     work_info_set[i, 4] = kv_indptr[i].item()      # kv_start
#     work_info_set[i, 5] = kv_indptr[i+1].item()    # kv_end


# reduce_indptr, reduce_final_map, reduce_partial_map
tile_q = 256
def ceil_div(a, b):
    return (a + b - 1) // b

num_q_tile = ceil_div(seq_len, tile_q)  # 522 / 256 = 3 tiles
padded_num_tokens = num_q_tile * tile_q  # 3 * 256 = 768

reduce_indptr = torch.tensor([0, 2], dtype=torch.int32)
# [<qo_start, qo_end, q_head_start, q_head_end>]
# reduce_final_map = torch.tensor([[0, 256, 0, 1], [256, 512, 0, 1], [256, 512, 0, 1], [512, 522, 0, 1]], dtype=torch.int32)
reduce_final_map = torch.tensor([[256, 512, 0, 1]], dtype=torch.int32)


# [<partial_qo_loc, q_head_start, q_head_end>]
reduce_partial_map = torch.tensor(
    [
        # [-1, 0, 1],   # partial_qo_loc=-1
        [0, 0, 1],   # partial_qo_loc=0
        [256, 0, 1], # partial_qo_loc=256
        # [-1, 0, 1],   # partial_qo_loc=-1
    ],
    dtype=torch.int32
)

max_seqlen_q = seq_len
softmax_scale = 1.0 / (qk_head_dim ** 0.5)
# q_scale = torch.ones([1], dtype=torch.float32)
# k_scale = torch.ones([1], dtype=torch.float32)
# v_scale = torch.ones([1], dtype=torch.float32)

print("Calling mla_ps_prefill_fwd...")
print(f"  Q shape: {q_quant.shape}")
print(f"  K shape: {k_quant.shape}")
print(f"  V shape: {v_quant.shape}")
print(f"  work_indptr shape: {work_indptr}")
print(f"  work_info_set shape: {work_info_set.shape}")
print(f"  work_indptr: {work_indptr}")
print(f"  work_info_set: {work_info_set}")
print(f"  max_seqlen_q: {max_seqlen_q}")
print(f"  is_causal: {is_causal}")
print(f"  qo_indptr: {qo_indptr}")
print(f"  kv_indptr: {kv_indptr}")
print(f"  kv_indices: {kv_indices}")
print(f"  kv_indices shape: {kv_indices.shape}")

# print(f"\nFirst work item:")
# print(f"  bs_idx: {work_info_set[0, 0]}")
# print(f"  partial_idx: {work_info_set[0, 1]}")
# print(f"  q_start: {work_info_set[0, 2]}")
# print(f"  q_end: {work_info_set[0, 3]}")
# print(f"  kv_start: {work_info_set[0, 4]}")
# print(f"  kv_end: {work_info_set[0, 5]}")
# print(f"  q_head_range: {work_info_set[0, 7]} (head {work_info_set[0, 7]} to {work_info_set[0, 7] >> 16})")

_, us_asm_prefill = run_perftest(
    aiter.mla.mla_ps_prefill_fwd,
    q_quant,
    k_quant,
    v_quant,
    output,
    qo_indptr,
    kv_indptr,
    kv_indices,
    work_indptr,
    work_info_set,
    max_seqlen_q,
    is_causal,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    softmax_scale,
    q_scale,
    k_scale,
    v_scale,
)

# print('This is output', output)

out_dtype = torch.bfloat16
kv_lora_rank = 128
# kv_lora_rank = 512
qk_rope_head_dim = 64

# kv_buffer must use the SAME data as K_bf16 (not different random data!)
# kv_buffer shape for reference: [num_page * page_size, nhead_kv, qk_head_dim]
kv_buffer = K_bf16.view(num_page * page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim)


# output2 = torch.empty((total_q, nhead, v_head_dim), dtype=torch.bfloat16)

# kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
# _ = run_perftest(
#     aiter.mla.mla_prefill_fwd,
#     Q,
#     kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
#     output2,
#     qo_indptr,
#     kv_indptr,
#     kv_indices,
#     kv_last_page_lens,
#     max_seqlen_q,
#     softmax_scale,
# )

# err = checkAllclose(
#     output2,
#     output,
#     msg=f"mla_prefill_fwd-mla_ps_prefill_fwd    [mla_ps_prefill_fwd vs mla_ps_prefill_fwd]: us......",
# )

# CRITICAL: Use bf16 tensors for reference (not fp8), and same data as kernel
(out_ref, lse_ref), us_torch_prefill = torch_mla_extend(
    Q_bf16,  # Use bf16 version, not Q which doesn't exist
    kv_buffer,  # Now uses same data as K_bf16
    qo_indptr,
    kv_indptr,
    kv_indices,
    softmax_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype=out_dtype,
    is_causal=True,  # Match kernel's is_causal setting
)
print("\nKernel executed successfully!")
# print(output)
err = checkAllclose(
    out_ref,
    output,
    rtol=5e-2, atol=5e-2,
    msg="mla_ps_prefill-absorb    [torch vs aiter_asm]: us......",
)
print(f"us_torch_prefill: {us_torch_prefill}")
print(f"us_asm_prefill: {us_asm_prefill}")
# print(f"Output shape: {result.shape}")
# print(f"Output sample: {result.view(-1)[:5]}")
# print(f"LSE shape: {attn_lse.shape}")

