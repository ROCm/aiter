import triton
import triton.language as tl


@triton.jit
def _split_qkv_kernel(
    qkv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    qkv_stride_0,
    q_stride_0,
    k_stride_0,
    v_stride_0,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
):

    wid = tl.program_id(0)

    q_offs = tl.arange(0, Q_SIZE)
    kv_offs = tl.arange(0, KV_SIZE)

    q_load_ptrs = qkv_ptr + (wid * qkv_stride_0) + q_offs
    k_load_ptrs = qkv_ptr + (wid * qkv_stride_0) + Q_SIZE + kv_offs
    v_load_ptrs = qkv_ptr + (wid * qkv_stride_0) + Q_SIZE + KV_SIZE + kv_offs

    q = tl.load(q_load_ptrs)
    k = tl.load(k_load_ptrs)
    v = tl.load(v_load_ptrs)

    q_store_ptrs = q_ptr + (wid * q_stride_0) + q_offs
    k_store_ptrs = k_ptr + (wid * k_stride_0) + kv_offs
    v_store_ptrs = v_ptr + (wid * v_stride_0) + kv_offs

    tl.store(q_store_ptrs, q)
    tl.store(k_store_ptrs, k)
    tl.store(v_store_ptrs, v)
