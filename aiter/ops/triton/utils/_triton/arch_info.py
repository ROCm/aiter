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


def pick_gemm_num_stages(arch, block_m, block_n, block_k, elem_a, elem_b):
    # gfx942 has no async_copy and num_stages=2 always fits.
    # gfx950 enables async_copy; pick ns=1 only when ns=2 overflows the 160 KB cap.
    if arch != "gfx950":
        return 2
    lds_bytes = int(block_m * block_k * elem_a + block_n * block_k * elem_b)
    return 2 if lds_bytes <= 163840 else 1
