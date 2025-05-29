import torch
import triton
import pytest
import numpy as np
from aiter.ops.triton.gemm_a8wfp4 import gemm_a8wfp4

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32

# Debug flags
DEBUG = True
DEBUG_INPUT = True
ZERO_OUTPUT = True

MXFP4_TABLE = [
        0.0,   # 0000
        0.5,   # 0001
        1.0,   # 0010
        1.5,   # 0011
        2.0,   # 0100
        3.0,   # 0101
        4.0,   # 0110
        6.0,   # 0111
        -0.0,  # 1000
        -0.5,  # 1001
        -1.0,  # 1010
        -1.5,  # 1011
        -2.0,  # 1100
        -3.0,  # 1101
        -4.0,  # 1110
        -6.0,  # 1111
    ]

def generate_a_8bit_inputs(M, K, dtype):
    if DEBUG_INPUT:
        # x = torch.arange(1, M * K + 1, dtype=torch.float32, device="cuda").reshape(M, K)
        x = torch.ones((M, K), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    max_x = x.abs().float().amax(dim=1, keepdim=True)
    dtype_max = torch.iinfo(dtype).max if dtype == torch.int8 else torch.finfo(dtype).max
    x_scale = max_x / dtype_max
    x = x / x_scale
    x = x.to(dtype)
    return x, x_scale


def generate_b_fp4_inputs(N, K):
    torch.manual_seed(5)
    if DEBUG_INPUT:
        # scale up
        w_fp32 = torch.ones((N, K), dtype=torch.float32, device="cuda")
        if False:
            max_w = w_fp32.abs().float().amax(dim=1, keepdim=True)
            mxfp4_max = 6.0
            w_scale = max_w / mxfp4_max  # 1.0 / 6.0 ≈ 0.1667
        else:
            w_scale = torch.ones((N, 1), dtype=torch.float32, device="cuda")
        w_scaled = w_fp32 / w_scale
        
        # find nearest MXFP4 value for each element
        w_fp4_indices = torch.zeros((N, K), dtype=torch.uint8, device="cuda")
        mxfp4_values = torch.tensor(MXFP4_TABLE, device="cuda", dtype=torch.float32)
        for i in range(N):
            for j in range(K):
                # Find closest value in MXFP4 lookup table
                diffs = (mxfp4_values - w_scaled[i, j]).abs()
                w_fp4_indices[i, j] = diffs.argmin()

        # pack two FP4 values into one uint8. 1.0 is 0010 in fp4. See the MXFP4 table. Two packed 1.0 values (0010 0010). Which is 0x22 in hex or 34 in uint8.
        w_packed = torch.zeros((N, K // 2), dtype=torch.uint8, device="cuda")
        w_packed = (w_fp4_indices[:, 1::2] << 4) | w_fp4_indices[:, ::2]
        
        # convert scale factor to e8m0 format
        w_scales_e8m0 = (torch.log2(w_scale) + 127).round().clamp(0, 255).to(torch.uint8)
        w_scales_e8m0 = w_scales_e8m0.T.repeat(K // SCALE_GROUP_SIZE, 1)
        
        w = w_packed
        w_scales = w_scales_e8m0
    else:
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w = w_low | w_high << 4
        # Scale of 1.0 in e8m0, bias 127.
        w_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda")
    return w.T, w_scales.T

def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(2 ** (v - 1), 4096 * v, 4096 * v) for v in range(1, 6)]
    # x_vals = [(128, 1024, 4096)]
    x_vals += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
    return x_vals


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_in_f32 = torch.tensor(MXFP4_TABLE, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales, dtype):
    # convert int8/fp8 A to f32. a scales are in fp32
    x_f32 = x.to(torch.float32)
    x_f32 = x_f32 * x_scales
    
    # convert fp4 B to fp32. b scales are in e8m0 so converted to fp32
    w_f32 = mxfp4_to_f32(w.T)
    w_f32 = w_f32.T
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32).to(dtype)

def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == "gfx950"

e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

# @pytest.mark.parametrize("M, N, K", get_x_vals())
# @pytest.mark.parametrize("M, N, K", [(512, 512, 512)])
# @pytest.mark.parametrize("M, N, K", [(64, 64, 32)])
@pytest.mark.parametrize("M, N, K", [(2, 2, 32)])
@pytest.mark.parametrize("a_dtype", [e4m3_type]) # [e4m3_type, e5m2_type, torch.int8]
@pytest.mark.parametrize("out_dtype", [torch.float16])
def test_gemm_a8wfp4(M: int, N: int, K: int, a_dtype, out_dtype):
    if not is_cdna4():
        pytest.skip("MXFP4 not supported on this architecture")

    x, x_scales = generate_a_8bit_inputs(M, K, a_dtype)
    w, w_scales = generate_b_fp4_inputs(N, K)
    if DEBUG:
        print()
        print("x", x, x.shape)
        print("x_scales", x_scales, x_scales.shape)
        print("w", w , w.shape)
        print("w_scales", w_scales, w_scales.shape)
        print(f"NOTE: we have shape {M}x{K} for A (fp8) and {N}x{K//2} for B (fp4). 2 fp4 values are packed into each uint8 value in the B tensor.")
        print("=== Debug: Matrix Values  ===")
        x_f32 = x.to(torch.float32) * x_scales
        print(x_f32, x_f32.shape)
        w_f32 = mxfp4_to_f32(w.T).T
        w_scales_f32 = e8m0_to_f32(w_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
        w_f32 = w_f32 * w_scales_f32.T
        print(w_f32, w_f32.shape)
        print(f"Expected result: each element should be {K} (sum of {K} ones)")

        print("=== What Triton Kernel Will See ===")
        print("A matrix raw bytes (what tl.load will return):")
        x_uint8 = x.view(torch.uint8)
        print(f"x as uint8: {x_uint8}")
        print(f"These are the raw byte values - 448 in fp8_e4m3fn is encoded as byte value {x_uint8[0, 0]}")
        
        print("B matrix raw bytes:")
        print(f"w as uint8: {w}")
        print(f"0x22 = {0x22} = two packed fp4 values: lower nibble = 2 (1.0), upper nibble = 2 (1.0)")
        
        print("Scale values:")
        print(f"a_scales (fp32): {x_scales.flatten()}")
        print(f"b_scales (e8m0 as uint8): {w_scales.flatten()}")
        print(f"b_scales decoded to fp32: {e8m0_to_f32(w_scales).flatten()}")
    torch_out = run_torch(x, w, x_scales, w_scales, out_dtype).to(out_dtype)
    if DEBUG:
        print("torch_out", torch_out, torch_out.shape)

    if ZERO_OUTPUT:
        triton_out = torch.zeros(x.shape[0], w.shape[1], device=x.device, dtype=out_dtype)
    else:
        triton_out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=out_dtype)
    gemm_a8wfp4(x, w, triton_out, x_scales, w_scales, out_dtype)
    if DEBUG:
        print("out:", triton_out, triton_out.shape)

    # np.savetxt(f"gemm_torch.csv", torch_out.cpu().numpy(), delimiter=",", fmt="%.6f")
    # np.savetxt(f"gemm_triton.csv", triton_out.cpu().numpy(), delimiter=",", fmt="%.6f")


    # torch.testing.assert_close(torch_out, out, atol=0.01, rtol=1e-2)
    torch.testing.assert_close(torch_out, triton_out)