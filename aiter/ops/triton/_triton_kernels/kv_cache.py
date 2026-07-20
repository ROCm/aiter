import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _nvfp4_quant_op


@triton.jit
def _store_mla_kv_cache(
    kv_cache_ptr,
    pid_t_slot,
    pid_hk,
    pid_blk,
    d_nope_offs,
    d_pe_offs,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_nope,
    k_pe,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SHUFFLED_KV_CACHE: tl.constexpr,
    SCALE_K_WIDTH_NOPE: tl.constexpr,
    SCALE_K_WIDTH_ROPE: tl.constexpr,
):
    if SHUFFLED_KV_CACHE:
        if kv_cache_ptr.dtype.element_ty == tl.bfloat16:
            # BF16
            K_WIDTH: tl.constexpr = 8
        else:
            # FP8 E4M3 or packed FP4 E2M1
            K_WIDTH: tl.constexpr = 16

        if kv_cache_ptr.dtype.element_ty == tl.uint8:
            NVFP4_QUANT_BLOCK_SIZE: tl.constexpr = 16
            k_nope, k_nope_scales = _nvfp4_quant_op(
                k_nope, BLOCK_D_nope, 1, NVFP4_QUANT_BLOCK_SIZE
            )
            k_pe, k_pe_scales = _nvfp4_quant_op(
                k_pe, BLOCK_D_pe, 1, NVFP4_QUANT_BLOCK_SIZE
            )
            BLOCK_D_nope_STORE: tl.constexpr = BLOCK_D_nope // 2
            BLOCK_D_pe_STORE: tl.constexpr = BLOCK_D_pe // 2
        else:
            BLOCK_D_nope_STORE: tl.constexpr = BLOCK_D_nope
            BLOCK_D_pe_STORE: tl.constexpr = BLOCK_D_pe

        d_nope_offs_shfl = tl.arange(0, BLOCK_D_nope_STORE // K_WIDTH).to(tl.int64)
        d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_STORE // K_WIDTH).to(tl.int64)
        k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
        k_nope = k_nope.reshape((BLOCK_D_nope_STORE // K_WIDTH, K_WIDTH))
        k_pe = k_pe.reshape((BLOCK_D_pe_STORE // K_WIDTH, K_WIDTH))

        kv_cache_ptrs = (
            kv_cache_ptr + pid_t_slot * kv_cache_stride_b + pid_hk * kv_cache_stride_h
        )
        kv_cache_nope_offs = (
            (pid_blk // 16) * BLOCK_D_nope_STORE * 16
            + (pid_blk % 16) * K_WIDTH
            + d_nope_offs_shfl[:, None] * K_WIDTH * 16
            + k_width_shfl[None, :]
        ) * kv_cache_stride_d

        if kv_cache_ptr.dtype.element_ty == tl.uint8:
            nope_scale_offset: tl.constexpr = BLOCK_D_nope // NVFP4_QUANT_BLOCK_SIZE
        else:
            nope_scale_offset: tl.constexpr = 0
        kv_cache_pe_offs = (
            BLOCK_SIZE * (BLOCK_D_nope_STORE + nope_scale_offset)
            + (pid_blk // 16) * BLOCK_D_pe_STORE * 16
            + (pid_blk % 16) * K_WIDTH
            + d_pe_offs_shfl[:, None] * K_WIDTH * 16
            + k_width_shfl[None, :]
        ) * kv_cache_stride_d

        tl.store(
            kv_cache_ptrs + kv_cache_nope_offs, k_nope.to(kv_cache_ptr.dtype.element_ty)
        )
        tl.store(
            kv_cache_ptrs + kv_cache_pe_offs, k_pe.to(kv_cache_ptr.dtype.element_ty)
        )

        if kv_cache_ptr.dtype.element_ty == tl.uint8:
            BLOCK_D_nope_scales: tl.constexpr = BLOCK_D_nope // NVFP4_QUANT_BLOCK_SIZE
            BLOCK_D_pe_scales: tl.constexpr = BLOCK_D_pe // NVFP4_QUANT_BLOCK_SIZE
            d_nope_offs_shfl = tl.arange(
                0, BLOCK_D_nope_scales // SCALE_K_WIDTH_NOPE
            ).to(tl.int64)
            d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_scales // SCALE_K_WIDTH_ROPE).to(
                tl.int64
            )
            k_nope_width_shfl = tl.arange(0, SCALE_K_WIDTH_NOPE).to(tl.int64)
            k_pe_width_shfl = tl.arange(0, SCALE_K_WIDTH_ROPE).to(tl.int64)
            k_nope_scales = k_nope_scales.reshape(
                (BLOCK_D_nope_scales // SCALE_K_WIDTH_NOPE, SCALE_K_WIDTH_NOPE)
            )
            k_pe_scales = k_pe_scales.reshape(
                (BLOCK_D_pe_scales // SCALE_K_WIDTH_ROPE, SCALE_K_WIDTH_ROPE)
            )
            pid_sub_blk = pid_blk % 128
            kv_cache_nope_scales_offs = (
                BLOCK_SIZE * BLOCK_D_nope_STORE
                + (pid_blk // 128) * BLOCK_D_nope_scales * 128
                + d_nope_offs_shfl[:, None] * SCALE_K_WIDTH_NOPE * 128
                + (pid_sub_blk % 32) * 4 * SCALE_K_WIDTH_NOPE
                + (pid_sub_blk // 32) * SCALE_K_WIDTH_NOPE
                + k_nope_width_shfl[None, :]
            ) * kv_cache_stride_d
            kv_cache_pe_scales_offs = (
                BLOCK_SIZE
                * (BLOCK_D_nope_STORE + BLOCK_D_nope_scales + BLOCK_D_pe_STORE)
                + (pid_blk // 128) * BLOCK_D_pe_scales * 128
                + d_pe_offs_shfl[:, None] * SCALE_K_WIDTH_ROPE * 128
                + (pid_sub_blk % 32) * 4 * SCALE_K_WIDTH_ROPE
                + (pid_sub_blk // 32) * SCALE_K_WIDTH_ROPE
                + k_pe_width_shfl[None, :]
            ) * kv_cache_stride_d
            e4m3_dtype = tl.float8e4nv
            tl.store(
                kv_cache_ptrs + kv_cache_nope_scales_offs,
                k_nope_scales.to(e4m3_dtype).to(
                    kv_cache_ptr.dtype.element_ty, bitcast=True
                ),
            )
            tl.store(
                kv_cache_ptrs + kv_cache_pe_scales_offs,
                k_pe_scales.to(e4m3_dtype).to(
                    kv_cache_ptr.dtype.element_ty, bitcast=True
                ),
            )
    else:
        # non-shuffled KV cache
        kv_cache_ptrs = (
            kv_cache_ptr + pid_t_slot * kv_cache_stride_b + pid_hk * kv_cache_stride_h
        )
        kv_cache_nope_offs = d_nope_offs * kv_cache_stride_d
        kv_cache_pe_offs = (d_pe_offs + BLOCK_D_nope) * kv_cache_stride_d
        tl.store(
            kv_cache_ptrs + kv_cache_nope_offs, k_nope.to(kv_cache_ptr.dtype.element_ty)
        )
        tl.store(
            kv_cache_ptrs + kv_cache_pe_offs, k_pe.to(kv_cache_ptr.dtype.element_ty)
        )


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
    kv_cache_stride_d,
    k_scale_ptr,
    KH: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1,
    SHUFFLED_KV_CACHE: tl.constexpr = False,
    SCALE_K_WIDTH_NOPE: tl.constexpr = 4,
    SCALE_K_WIDTH_ROPE: tl.constexpr = 4,
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
        k_nope = k_nope.to(tl.float32) * k_scale_rcprl
        k_pe = k_pe.to(tl.float32) * k_scale_rcprl

        _store_mla_kv_cache(
            kv_cache_ptr,
            pid_t_slot,
            pid_hk,
            pid_blk,
            d_nope_offs,
            d_pe_offs,
            kv_cache_stride_b,
            kv_cache_stride_h,
            kv_cache_stride_d,
            k_nope,
            k_pe,
            BLOCK_D_nope,
            BLOCK_D_pe,
            BLOCK_SIZE,
            SHUFFLED_KV_CACHE,
            SCALE_K_WIDTH_NOPE,
            SCALE_K_WIDTH_ROPE,
        )

        # if SHUFFLED_KV_CACHE:
        #     if kv_cache_ptr.dtype.element_ty == tl.bfloat16:
        #         # BF16
        #         K_WIDTH: tl.constexpr = 8
        #     else:
        #         # FP8 E4M3 or packed FP4 E2M1
        #         K_WIDTH: tl.constexpr = 16

        #     if kv_cache_ptr.dtype.element_ty == tl.uint8:
        #         NVFP4_QUANT_BLOCK_SIZE: tl.constexpr = 16
        #         k_nope, k_nope_scales = _nvfp4_quant_op(
        #             k_nope, BLOCK_D_nope, 1, NVFP4_QUANT_BLOCK_SIZE
        #         )
        #         k_pe, k_pe_scales = _nvfp4_quant_op(
        #             k_pe, BLOCK_D_pe, 1, NVFP4_QUANT_BLOCK_SIZE
        #         )
        #         BLOCK_D_nope_STORE: tl.constexpr = BLOCK_D_nope // 2
        #         BLOCK_D_pe_STORE: tl.constexpr = BLOCK_D_pe // 2
        #     else:
        #         BLOCK_D_nope_STORE: tl.constexpr = BLOCK_D_nope
        #         BLOCK_D_pe_STORE: tl.constexpr = BLOCK_D_pe

        #     d_nope_offs_shfl = tl.arange(0, BLOCK_D_nope_STORE // K_WIDTH).to(tl.int64)
        #     d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_STORE // K_WIDTH).to(tl.int64)
        #     k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
        #     k_nope = k_nope.reshape((BLOCK_D_nope_STORE // K_WIDTH, K_WIDTH))
        #     k_pe = k_pe.reshape((BLOCK_D_pe_STORE // K_WIDTH, K_WIDTH))

        #     kv_cache_ptrs = (
        #         kv_cache_ptr
        #         + pid_t_slot * kv_cache_stride_b
        #         + pid_hk * kv_cache_stride_h
        #     )
        #     kv_cache_nope_offs = (
        #         (pid_blk // 16) * BLOCK_D_nope_STORE * 16
        #         + (pid_blk % 16) * K_WIDTH
        #         + d_nope_offs_shfl[:, None] * K_WIDTH * 16
        #         + k_width_shfl[None, :]
        #     ) * kv_cache_stride_d

        #     if kv_cache_ptr.dtype.element_ty == tl.uint8:
        #         nope_scale_offset: tl.constexpr = BLOCK_D_nope // NVFP4_QUANT_BLOCK_SIZE
        #     else:
        #         nope_scale_offset: tl.constexpr = 0
        #     kv_cache_pe_offs = (
        #         BLOCK_SIZE * (BLOCK_D_nope_STORE + nope_scale_offset)
        #         + (pid_blk // 16) * BLOCK_D_pe_STORE * 16
        #         + (pid_blk % 16) * K_WIDTH
        #         + d_pe_offs_shfl[:, None] * K_WIDTH * 16
        #         + k_width_shfl[None, :]
        #     ) * kv_cache_stride_d

        #     tl.store(kv_cache_ptrs + kv_cache_nope_offs, k_nope.to(kv_cache_ptr.dtype.element_ty))
        #     tl.store(kv_cache_ptrs + kv_cache_pe_offs, k_pe.to(kv_cache_ptr.dtype.element_ty))

        #     if kv_cache_ptr.dtype.element_ty == tl.uint8:
        #         BLOCK_D_nope_scales: tl.constexpr = BLOCK_D_nope // NVFP4_QUANT_BLOCK_SIZE
        #         BLOCK_D_pe_scales: tl.constexpr = BLOCK_D_pe // NVFP4_QUANT_BLOCK_SIZE
        #         d_nope_offs_shfl = tl.arange(0, BLOCK_D_nope_scales // SCALE_K_WIDTH_NOPE).to(tl.int64)
        #         d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_scales // SCALE_K_WIDTH_ROPE).to(tl.int64)
        #         k_nope_width_shfl = tl.arange(0, SCALE_K_WIDTH_NOPE).to(tl.int64)
        #         k_pe_width_shfl = tl.arange(0, SCALE_K_WIDTH_ROPE).to(tl.int64)
        #         k_nope_scales = k_nope_scales.reshape((BLOCK_D_nope_scales // SCALE_K_WIDTH_NOPE, SCALE_K_WIDTH_NOPE))
        #         k_pe_scales = k_pe_scales.reshape((BLOCK_D_pe_scales // SCALE_K_WIDTH_ROPE, SCALE_K_WIDTH_ROPE))
        #         pid_sub_blk = pid_blk % 128
        #         kv_cache_nope_scales_offs = (
        #             BLOCK_SIZE * BLOCK_D_nope_STORE
        #             + (pid_blk // 128) * BLOCK_D_nope_scales * 128
        #             + d_nope_offs_shfl[:, None] * SCALE_K_WIDTH_NOPE * 128
        #             + (pid_sub_blk % 32) * 4 * SCALE_K_WIDTH_NOPE
        #             + (pid_sub_blk // 32) * SCALE_K_WIDTH_NOPE + k_nope_width_shfl[None, :]
        #         ) * kv_cache_stride_d
        #         kv_cache_pe_scales_offs = (
        #             BLOCK_SIZE * (BLOCK_D_nope_STORE + BLOCK_D_nope_scales + BLOCK_D_pe_STORE)
        #             + (pid_blk // 128) * BLOCK_D_pe_scales * 128
        #             + d_pe_offs_shfl[:, None] * SCALE_K_WIDTH_ROPE * 128
        #             + (pid_sub_blk % 32) * 4 * SCALE_K_WIDTH_ROPE
        #             + (pid_sub_blk // 32) * SCALE_K_WIDTH_ROPE + k_pe_width_shfl[None, :]
        #         ) * kv_cache_stride_d
        #         e4m3_dtype = tl.float8e4nv
        #         tl.store(kv_cache_ptrs + kv_cache_nope_scales_offs, k_nope_scales.to(e4m3_dtype).to(kv_cache_ptr.dtype.element_ty, bitcast = True))
        #         tl.store(kv_cache_ptrs + kv_cache_pe_scales_offs, k_pe_scales.to(e4m3_dtype).to(kv_cache_ptr.dtype.element_ty, bitcast = True))
        # else:
        #     kv_cache_ptrs = (
        #         kv_cache_ptr
        #         + pid_t_slot * kv_cache_stride_b
        #         + pid_hk * kv_cache_stride_h
        #     )
        #     kv_cache_nope_offs = d_nope_offs * kv_cache_stride_d
        #     kv_cache_pe_offs = (d_pe_offs + BLOCK_D_nope) * kv_cache_stride_d
        #     tl.store(kv_cache_ptrs + kv_cache_nope_offs, k_nope.to(kv_cache_ptr.dtype.element_ty))
        #     tl.store(kv_cache_ptrs + kv_cache_pe_offs, k_pe.to(kv_cache_ptr.dtype.element_ty))
