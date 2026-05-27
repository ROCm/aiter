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


def _gemm_lds_bytes(block_m, block_n, block_k, bits_a, bits_b, num_stages):
    # Non-async LDS usage (matches TensorAtlas calculate_lds_usage).
    LDSA_bits = block_m * block_k * bits_a
    LDSB_bits = block_n * block_k * bits_b
    if num_stages <= 1:
        return max(LDSA_bits, LDSB_bits) // 8
    return (LDSA_bits + LDSB_bits) * (num_stages - 1) // 8


def pick_gemm_num_stages(arch, block_m, block_n, block_k, bits_a, bits_b):
    # bits_a / bits_b: element bit-widths (8 for fp8, 4 for mxfp4).
    cap = _LDS_CAP_BYTES.get(arch)
    if cap is None:
        return 2
    return 2 if _gemm_lds_bytes(block_m, block_n, block_k, bits_a, bits_b, 2) <= cap else 1
