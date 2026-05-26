import triton
import triton.language as tl


@triton.jit
def _cat_and_cache_mla_kernel(
    k_nope_ptr,
    k_pe_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_blk,
    kv_cache_stride_d,
    k_scale_ptr,
    KH: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1,
    SHUFFLED_KV_CACHE: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope).to(tl.int64)
    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    pid_b = pid // KH
    pid_hk = pid % KH
    pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
    if pid_slot >= 0:
        if BLOCK_SIZE > 1:
            pid_t_slot = pid_slot // BLOCK_SIZE
            pid_blk = pid_slot % BLOCK_SIZE
        else:
            pid_t_slot = pid_slot
            pid_blk = 0
        if HAVE_K_SCALE:
            k_scale = tl.load(k_scale_ptr)
        else:
            k_scale = 1

        k_nope_ptrs = (
            k_nope_ptr
            + pid_b * k_nope_stride_b
            + pid_hk * k_nope_stride_h
            + d_nope_offs * k_nope_stride_d
        )
        k_pe_ptrs = (
            k_pe_ptr
            + pid_b * k_pe_stride_b
            + pid_hk * k_pe_stride_h
            + d_pe_offs * k_pe_stride_d
        )
        k_nope = tl.load(k_nope_ptrs)
        k_pe = tl.load(k_pe_ptrs)
        k_scale_rcprl = (1 / k_scale).to(tl.float32)
        k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(
            kv_cache_ptr.dtype.element_ty
        )
        k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)

        if SHUFFLED_KV_CACHE:
            if kv_cache_ptr.dtype.element_ty == tl.bfloat16:
                K_WIDTH: tl.constexpr = 8
            else:
                K_WIDTH: tl.constexpr = 16
            dk_nope_offs_shfl = tl.arange(0, BLOCK_D_nope // K_WIDTH).to(tl.int64)
            d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe // K_WIDTH).to(tl.int64)
            k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
            k_nope = k_nope.reshape((BLOCK_D_nope // K_WIDTH, K_WIDTH))
            k_pe = k_pe.reshape((BLOCK_D_pe // K_WIDTH, K_WIDTH))

            kv_cache_ptrs = (
                kv_cache_ptr
                + pid_t_slot * kv_cache_stride_b
                + pid_hk * kv_cache_stride_h
            )
            kv_cache_nope_offs = (
                (pid_blk // 16) * BLOCK_D_nope * 16
                + (pid_blk % 16) * K_WIDTH
                + dk_nope_offs_shfl[:, None] * K_WIDTH * 16
                + k_width_shfl[None, :]
            ) * kv_cache_stride_d
            kv_cache_pe_offs = (
                (pid_blk // 16) * BLOCK_D_pe * 16
                + (pid_blk % 16) * K_WIDTH
                + d_pe_offs_shfl[:, None] * K_WIDTH * 16
                + k_width_shfl[None, :]
                + BLOCK_SIZE * BLOCK_D_nope
            ) * kv_cache_stride_d

            tl.store(kv_cache_ptrs + kv_cache_nope_offs, k_nope)
            tl.store(kv_cache_ptrs + kv_cache_pe_offs, k_pe)
        else:
            kv_cache_ptrs = (
                kv_cache_ptr
                + pid_t_slot * kv_cache_stride_b
                + pid_hk * kv_cache_stride_h
            )
            tl.store(kv_cache_ptrs + d_nope_offs * kv_cache_stride_d, k_nope)
            tl.store(
                kv_cache_ptrs + (d_pe_offs + BLOCK_D_nope) * kv_cache_stride_d,
                k_pe,
            )
