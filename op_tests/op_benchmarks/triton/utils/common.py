import triton
import torch
import triton.language as tl


def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        return -1
    if target.arch == "gfx942":
        return 3
    if target.arch == "gfx950":
        return 4
    return -1


# OCP mixed-format fp4 (mxfp4) has two elements packed in one uint8
str_to_torch_dtype = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": (
        torch.float8_e4m3fn if get_cdna_version() == 4 else torch.float8_e4m3fnuz
    ),
    "fp8_e5m2": torch.float8_e5m2 if get_cdna_version() == 4 else torch.float8_e5m2fnuz,
    "mxfp4": torch.uint8,
}


torch_to_tl_dtype = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e5m2: tl.float8e5,
    torch.float8_e5m2fnuz: tl.float8e5b16,
    torch.uint8: tl.uint8,
}
