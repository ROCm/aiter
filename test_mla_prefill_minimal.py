import torch
import aiter

torch.set_default_device("cuda")

# ????
batch_size = 1
seq_len = 256
nhead = 16
nhead_kv = 1
qk_head_dim = 192  # 128 (kv_lora_rank) + 64 (qk_rope_head_dim)
v_head_dim = 128
page_size = 1
is_causal = True

# 1. ???? tensors
total_q = batch_size * seq_len
num_page = 2048
total_kv = total_q

# Q, K, V
Q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)
K = torch.randn((num_page * page_size, nhead_kv, qk_head_dim), dtype=torch.bfloat16)
V, _ = torch.split(K, [v_head_dim, qk_head_dim - v_head_dim], dim=-1)

# Output
output = torch.empty((total_q, nhead, v_head_dim), dtype=torch.bfloat16)

# 2. ?? indptr ? indices
qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32)
kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32)
kv_indices = torch.randint(0, num_page, (total_kv,), dtype=torch.int32)

# 3. ???? metadata
# ?? CU ??
num_cus = torch.cuda.get_device_properties(0).multi_processor_count

# work_indptr: ????? CU
work_indptr = torch.tensor([0, 1] + [1] * (num_cus - 1), dtype=torch.int32)

# work_info_set: [num_work_items, 8]
# ??: [bs_index, partial_index, q_start, q_end, kv_start, kv_end, kv_offset, pad]
work_info_set = torch.tensor([
    [0,    # bs_index: batch 0
     -1,   # partial_index: -1 ?????
     0,    # q_start: Q ????
     seq_len,  # q_end: Q ????
     0,    # kv_start: KV ????
     seq_len,  # kv_end: KV ????
     0,    # kv_offset: ?? KV ??
     0]    # pad
], dtype=torch.int32)

# reduce_indptr, reduce_final_map, reduce_partial_map
reduce_indptr = torch.tensor([0, 1], dtype=torch.int32)
reduce_final_map = torch.tensor([[0, 0]], dtype=torch.int32)  # [batch_idx, tile_idx]
reduce_partial_map = torch.tensor([0], dtype=torch.int32)

# 4. ????
max_seqlen_q = seq_len
softmax_scale = 1.0 / (qk_head_dim ** 0.5)
q_scale = torch.ones([1], dtype=torch.float32)
k_scale = torch.ones([1], dtype=torch.float32)
v_scale = torch.ones([1], dtype=torch.float32)

# 5. ?? kernel
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

result, attn_lse = aiter.mla.mla_ps_prefill_fwd(
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

print("\nKernel executed successfully!")
print(f"Output shape: {result.shape}")
print(f"Output sample: {result.view(-1)[:5]}")
print(f"LSE shape: {attn_lse.shape}")

