from aiter.ops.flydsl.moe_kernels import (
    build_flydslv2_gemm2_name,
    get_flydsl_stage2_v2_kernels,
)
from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel


def test_name_roundtrip():
    name = build_flydslv2_gemm2_name(
        "fp4",
        "fp4",
        "bf16",
        tm=32,
        epilog="atomic",
        persist=False,
        use_nt=True,
        sbm=32,
    )
    assert name == "flydslv2_moe2_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32"
    cfg = parse_flydsl_v2_gemm2_kernel(name)
    assert cfg["tile_m"] == 32 and cfg["tile_n"] == 256 and cfg["tile_k"] == 256
    assert cfg["epilog"] == "atomic" and cfg["use_nt"] and not cfg["persist"]
    assert cfg["sort_block_m"] == 32


def test_stage2_v2_kernels_fp4_persist_only():
    # fp4: persist in {False, True}; block_m=64 -> BM in {16,32,64}
    ks = get_flydsl_stage2_v2_kernels("fp4", "fp4", "bf16", block_m=64)
    assert all(parse_flydsl_v2_gemm2_kernel(k) is not None for k in ks)
    bms = {parse_flydsl_v2_gemm2_kernel(k)["tile_m"] for k in ks}
    assert bms <= {16, 32, 64} and 64 in bms
    persists = {parse_flydsl_v2_gemm2_kernel(k)["persist"] for k in ks}
    assert persists == {False, True}


def test_stage2_v2_kernels_fp8_no_persist():
    ks = get_flydsl_stage2_v2_kernels("fp8", "fp4", "bf16", block_m=32)
    persists = {parse_flydsl_v2_gemm2_kernel(k)["persist"] for k in ks}
    assert persists == {False}  # fp8-A persist is fail-fast
