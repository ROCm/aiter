import torch
import triton
import triton.language as tl
from . import arch_info 

e5m2_dtype, e4m3_dtype = arch_info.get_fp8_dtypes()
str_to_torch_dtype = {
    "float64"           : torch.float64,
    "float32"           : torch.float32,
    "fp32"              : torch.float32,
    "bfloat16"          : torch.bfloat16,
    "bf16"              : torch.bfloat16,
    "float16"           : torch.float16,
    "fp16"              : torch.float16,
    "float8_e5m2"       : e5m2_dtype,
    "float8_e4m3fn"     : e4m3_dtype,
    "e5m2fnuz"          : e5m2_dtype,
    "e4m3fnuz"          : e4m3_dtype,
    "int64"             : torch.int64,
    "int32"             : torch.int32,
    "int16"             : torch.int16,
    "uint8"             : torch.uint8,
    "int8"              : torch.int8,
    "mxfp4_e2m1"        : torch.uint8  #OCP MXFP4 packs two 4-bits into 8-bit
}

torch_to_triton_dtype = {
    torch.float64       : tl.float64,
    torch.float32       : tl.float32,
    torch.float16       : tl.float16,
    torch.bfloat16      : tl.bfloat16,
    torch.float8_e4m3fn : tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e5m2   : tl.float8e5,
    torch.float8_e5m2fnuz: tl.float8e5b16,
    torch.int64         : tl.int64,
    torch.int32         : tl.int32,
    torch.int16         : tl.int16,
    torch.int8          : tl.int8,
    torch.uint8         : tl.uint8,
}
