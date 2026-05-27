import triton

try:
    _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
except RuntimeError:
    from jax._src.lib import gpu_triton as triton_kernel_call_lib

    _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]


def get_arch():
    return _CACHED_ARCH


def is_gluon_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp4_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950", "gfx1250", "gfx1200", "gfx1201")


def is_mx_scale_preshuffling_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_tdm_avail():
    return get_arch() in ("gfx1250",)


_LDS_CAP_BYTES = {"gfx950": 163840, "gfx942": 65536}


def _padded_size_32_4(n):
    pad = (n >> 5) << 2
    if (n & 31) == 0 and pad >= 4:
        pad -= 4
    return n + pad


def _padded_size_pow2(n, interval, padding):
    log2_i = (interval - 1).bit_length()
    log2_p = (padding - 1).bit_length() if padding else 0
    pad = (n >> log2_i) << log2_p
    if n % interval == 0 and pad >= padding:
        pad -= padding
    return n + pad


def _gemm_lds_bytes(
    block_m, block_n, block_k, bits_a, bits_b, num_stages, use_async_padding
):
    elem_a = block_m * block_k
    elem_b = block_k * block_n
    if use_async_padding:
        # Padded shared encoding + N buffers (matches TensorAtlas
        # _estimate_triton_lds_async_copy / tritonBLAS origami).
        pa = _padded_size_32_4(elem_a)
        pb = _padded_size_32_4(elem_b)
        if block_k & (block_k - 1) == 0:
            pa = max(pa, _padded_size_pow2(elem_a, block_k, 8))
        if block_n & (block_n - 1) == 0:
            pb = max(pb, _padded_size_pow2(elem_b, block_n, 8))
        return num_stages * (pa * bits_a + pb * bits_b) // 8
    # Non-async: (N-1) extra buffer pairs beyond the active stage.
    LDSA = elem_a * bits_a
    LDSB = elem_b * bits_b
    if num_stages <= 1:
        return max(LDSA, LDSB) // 8
    return (LDSA + LDSB) * (num_stages - 1) // 8


def pick_gemm_num_stages(
    arch, block_m, block_n, block_k, bits_a, bits_b, use_async_padding=False
):
    # bits_a / bits_b: element bit-widths (8 for fp8, 4 for mxfp4).
    # use_async_padding: True when the kernel lowers to async direct-to-LDS
    # with padded shared encoding (e.g. a4w4 on gfx950).
    cap = _LDS_CAP_BYTES.get(arch)
    if cap is None:
        return 2
    lds = _gemm_lds_bytes(
        block_m, block_n, block_k, bits_a, bits_b, 2, use_async_padding
    )
    return 2 if lds <= cap else 1
