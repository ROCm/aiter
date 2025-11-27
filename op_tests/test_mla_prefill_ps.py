import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random


torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

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
nhead = 16
nhead_kv = 16
qk_head_dim = 192  # 128 (kv_lora_rank) + 64 (qk_rope_head_dim)
v_head_dim = 128
page_size = 1
is_causal = True


total_q = batch_size * seq_len
num_page = 1044
total_kv = total_q

# Q, K, V
# Q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)
# K = torch.randn((num_page * page_size, nhead_kv, qk_head_dim), dtype=torch.bfloat16)
# V, _ = torch.split(K, [v_head_dim, qk_head_dim - v_head_dim], dim=-1)

# # Output
# output = torch.empty((total_q, nhead, v_head_dim), dtype=torch.bfloat16)

# qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32)
# kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32)
# kv_indices = torch.randint(0, num_page, (total_kv,), dtype=torch.int32)

qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)

ctx_lens = 256
for i in range(batch_size):
    seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
    seq_lens_qo[i] = max(
        min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
    )


kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
kv_indices = torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)
qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
max_seqlen_qo = seq_lens_qo.max().item()
max_seqlen_kv = seq_lens_kv.max().item()
total_qo = qo_indptr[-1].item()
total_kv = kv_indptr[-1].item()


Q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)
K = torch.randn((total_kv, nhead_kv, qk_head_dim), dtype=torch.bfloat16)
V = torch.randn((total_kv, nhead_kv, v_head_dim), dtype=torch.bfloat16)
output = torch.empty((total_qo, nhead, v_head_dim), dtype=torch.bfloat16)

num_cus = torch.cuda.get_device_properties(0).multi_processor_count

work_indptr = torch.tensor([0, 1] + [1] * (num_cus - 1), dtype=torch.int32)


work_info_set = torch.zeros((1, 8), dtype=torch.int32, device="cuda")
for i in range(batch_size):
    work_info_set[i, 0] = i                        # bs_index
    work_info_set[i, 1] = -1                       # partial_index: -1 means no split
    work_info_set[i, 2] = qo_indptr[i].item()      # q_start
    work_info_set[i, 3] = qo_indptr[i+1].item()    # q_end
    work_info_set[i, 4] = kv_indptr[i].item()      # kv_start
    work_info_set[i, 5] = kv_indptr[i+1].item()    # kv_end
    work_info_set[i, 6] = 0                        # kv_offset
    work_info_set[i, 7] = 0                        # pad


# reduce_indptr, reduce_final_map, reduce_partial_map
reduce_indptr = torch.tensor([0, 1], dtype=torch.int32)
reduce_final_map = torch.tensor([[0, 0]], dtype=torch.int32)  # [batch_idx, tile_idx]
reduce_partial_map = torch.tensor([0], dtype=torch.int32)

max_seqlen_q = seq_len
softmax_scale = 1.0 / (qk_head_dim ** 0.5)
q_scale = torch.ones([1], dtype=torch.float32)
k_scale = torch.ones([1], dtype=torch.float32)
v_scale = torch.ones([1], dtype=torch.float32)

print("Calling mla_ps_prefill_fwd...")
print(f"  Q shape: {Q.shape}")
print(f"  K shape: {K.shape}")
print(f"  V shape: {V.shape}")
print(f"  work_indptr shape: {work_indptr.shape}")
print(f"  work_info_set shape: {work_info_set.shape}")
print(f"  work_indptr: {work_indptr}")
print(f"  work_info_set: {work_info_set}")
print(f"  max_seqlen_q: {max_seqlen_q}")
print(f"  is_causal: {is_causal}")
print(f"  qo_indptr: {qo_indptr}")
print(f"  kv_indptr: {kv_indptr}")
print(f"  kv_indices: {kv_indices}")

(attn_logits, attn_lse), us_asm_decode = run_perftest(
    aiter.mla.mla_ps_prefill_fwd,
    Q,
    K,
    V,
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
qk_rope_head_dim = 64
kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )


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

out_ref, lse_ref = torch_mla_extend(
    Q,
    kv_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    softmax_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype=out_dtype,
    is_causal=True,
)

print("\nKernel executed successfully!")
err = checkAllclose(
    out_ref,
    output,
    msg=f"mla_ps_prefill-absorb    [torch vs aiter_asm]: us......",
)
# print(f"Output shape: {result.shape}")
# print(f"Output sample: {result.view(-1)[:5]}")
# print(f"LSE shape: {attn_lse.shape}")

