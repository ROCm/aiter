# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused gather + kv_b_proj (DeepSeek MLA prefix expansion) tests.

Covers the one configuration the FlyDSL backend implements: page_size 1, fp8 KV
cache, fp8 ``shuffle_weight((16,16))`` weight, per-output-row weight scale,
per-tensor activation scale, bf16 or scaled fp8 outputs, gfx950.

Checked against two independent references:
  * a float32 torch reference (ground truth), and
  * the Triton op on the same preshuffled weight -- the two backends consume the
    same preshuffled tensor, which is itself the thing being asserted.
Usage:
    pytest op_tests/test_flydsl_gather_kv_b_proj.py -q
    python op_tests/test_flydsl_gather_kv_b_proj.py      # + perf
    python op_tests/test_flydsl_gather_kv_b_proj.py --fp8-output
"""

import argparse

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gather_kv_b_proj import gather_kv_b_proj_flydsl
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gather_kv_b_proj import (
    gather_kv_b_proj as triton_gather_kv_b_proj,
)
from aiter.test_common import checkAllclose, run_perftest

KV_C_DIM = 512
KV_PE_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128

SUPPORTED_GFX = ("gfx950",)

_SKIP = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="gfx950 FlyDSL required",
)


def _make_case(
    num_tokens,
    n_heads,
    alloc=None,
    duplicate_indices=False,
    k_scale_value=1.0,
    scale_mode="row",
    num_blocks=None,
    seed=0,
    device="cuda",
):
    """Build one page_size-1 gather case.

    ``alloc`` models the real caller: the chunk workspace is preallocated at its
    maximum and only ``num_tokens`` rows are live.
    """
    torch.manual_seed(seed)
    alloc = alloc or num_tokens
    num_blocks = num_blocks or max(alloc, 64)
    weight_n = n_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)

    if num_blocks > 1 << 17:
        tile = (
            torch.randn(4096, KV_C_DIM + KV_PE_DIM, device=device)
            .to(dtypes.fp8)
            .view(torch.uint8)
        )
        k_buffer = (
            tile.repeat(-(-num_blocks // 4096), 1)[:num_blocks]
            .view(dtypes.fp8)
            .view(num_blocks, 1, KV_C_DIM + KV_PE_DIM)
        )
    else:
        k_buffer = torch.randn(
            (num_blocks, 1, KV_C_DIM + KV_PE_DIM), device=device, dtype=torch.float32
        ).to(dtypes.fp8)
    k_scale = torch.full((1,), k_scale_value, device=device, dtype=torch.float32)

    if duplicate_indices:
        # The prefix cache legitimately repeats slot ids across tokens.
        kv_indices = torch.randint(0, num_blocks, (alloc,), device=device)
    else:
        kv_indices = torch.randperm(num_blocks, device=device)[:alloc]
    kv_indices = kv_indices.to(torch.int32)

    kv_indptr = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    cu_seqlens_k = kv_indptr

    weight = torch.randn((weight_n, KV_C_DIM), device=device).to(dtypes.fp8)
    if scale_mode == "row":
        weight_scale = (
            torch.rand((weight_n, 1), device=device, dtype=torch.float32) + 0.5
        )
    else:  # 128x128 block scale -- the DeepSeek default quantization
        weight_scale = (
            torch.rand(
                (weight_n // 128, KV_C_DIM // 128), device=device, dtype=torch.float32
            )
            + 0.5
        )

    k_prefix = torch.zeros(
        (alloc, n_heads, QK_NOPE_HEAD_DIM + KV_PE_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (alloc, n_heads, V_HEAD_DIM), device=device, dtype=torch.bfloat16
    )
    return {
        "k_buffer": k_buffer,
        "k_scale": k_scale,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "cu_seqlens_k": cu_seqlens_k,
        "weight": weight,
        "weight_scale": weight_scale,
        "k_prefix": k_prefix,
        "v_prefix": v_prefix,
        "num_tokens": num_tokens,
        "n_heads": n_heads,
        "scale_mode": scale_mode,
    }


def _torch_ref(case):
    """float32 ground truth for page_size 1."""
    m = case["num_tokens"]
    n_heads = case["n_heads"]
    idx = case["kv_indices"][:m].long()
    latent = case["k_buffer"][idx].reshape(m, KV_C_DIM + KV_PE_DIM).float()
    kv_c, k_pe = latent.split([KV_C_DIM, KV_PE_DIM], dim=-1)

    if case["scale_mode"] == "row":
        w = case["weight"].float() * case["weight_scale"].float()
    else:
        ws = case["weight_scale"]
        w = (
            case["weight"].float().view(ws.shape[0], 128, ws.shape[1], 128)
            * ws[:, None, :, None]
        ).reshape(case["weight"].shape)
    scale = case["k_scale"].float()
    proj = ((kv_c @ w.T) * scale).view(m, n_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    k_nope, v = proj.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    rope = (k_pe * scale).unsqueeze(1).expand(-1, n_heads, -1)
    return torch.cat([k_nope, rope], dim=-1), v


def _run_flydsl(case, weight_preshuffle=True, **kw):
    w = (
        shuffle_weight(case["weight"], layout=(16, 16))
        if weight_preshuffle
        else case["weight"]
    )
    gather_kv_b_proj_flydsl(
        case["k_buffer"],
        case["k_scale"],
        case["kv_indptr"],
        case["kv_indices"],
        case["cu_seqlens_k"],
        w,
        case["weight_scale"],
        case["k_prefix"],
        case["v_prefix"],
        num_tokens=case["num_tokens"],
        weight_preshuffle=weight_preshuffle,
        **kw,
    )


@_SKIP
@pytest.mark.parametrize(
    "num_tokens, n_heads, alloc, duplicate_indices, k_scale_value",
    [
        (512, 12, None, False, 1.0),
        (2048, 12, None, False, 1.0),
        (8192, 12, None, False, 1.0),
        # M not a multiple of BLOCK_M=256: exercises the row tail.
        (1000, 12, 1024, False, 1.0),
        (1, 12, 256, False, 1.0),
        # The prefix cache repeats slot ids.
        (777, 12, 1024, True, 1.0),
        # k_scale is 1.0 in the current deployment but must not be assumed.
        (512, 12, None, False, 0.37),
        (512, 16, None, False, 1.0),
    ],
)
def test_gather_kv_b_proj_flydsl(
    num_tokens, n_heads, alloc, duplicate_indices, k_scale_value
):
    case = _make_case(num_tokens, n_heads, alloc, duplicate_indices, k_scale_value)
    _run_flydsl(case)
    m = num_tokens
    k_ref, v_ref = _torch_ref(case)
    checkAllclose(
        k_ref,
        case["k_prefix"][:m].float(),
        atol=1e-2,
        rtol=1e-2,
        msg="k_prefix vs torch f32",
    )
    checkAllclose(
        v_ref,
        case["v_prefix"][:m].float(),
        atol=1e-2,
        rtol=1e-2,
        msg="v_prefix vs torch f32",
    )

    cos = torch.nn.functional.cosine_similarity(
        case["v_prefix"][:m].float().flatten(), v_ref.flatten(), dim=0
    )
    assert cos > 0.999, f"cosine similarity {cos:.6f} too low"


@_SKIP
@pytest.mark.parametrize(
    "num_tokens, n_heads, alloc, k_scale_value",
    [
        (512, 12, None, 1.0),
        (2048, 12, None, 1.0),
        (1000, 12, 1024, 1.0),  # row tail
        (777, 12, 1024, 0.37),  # non-unit activation scale
        (512, 16, None, 1.0),
    ],
)
def test_gather_kv_b_proj_flydsl_block_scale(num_tokens, n_heads, alloc, k_scale_value):
    """128x128 block scale -- the DeepSeek default quantization.

    The scale varies along K, so it cannot be applied once at the end; the kernel
    renormalises the accumulator between K tiles instead of keeping a second
    accumulator. This is the test that the renormalisation is exact.
    """
    case = _make_case(
        num_tokens, n_heads, alloc, k_scale_value=k_scale_value, scale_mode="block"
    )
    _run_flydsl(case)
    m = num_tokens
    k_ref, v_ref = _torch_ref(case)
    checkAllclose(
        k_ref,
        case["k_prefix"][:m].float(),
        atol=1e-2,
        rtol=1e-2,
        msg="k_prefix, block scale",
    )
    checkAllclose(
        v_ref,
        case["v_prefix"][:m].float(),
        atol=1e-2,
        rtol=1e-2,
        msg="v_prefix, block scale",
    )


@_SKIP
@pytest.mark.parametrize("num_tokens", [512, 1000])
def test_gather_kv_b_proj_flydsl_row_major_weight(num_tokens):
    """Row-major (un-preshuffled) weight must match the preshuffled path exactly.

    Same GEMM, only the B-side global->LDS address map and the K-tile stride
    differ, so any difference here is an addressing bug, not arithmetic.
    """
    case = _make_case(num_tokens, 12)
    _run_flydsl(case, weight_preshuffle=False)
    k_ref, v_ref = _torch_ref(case)
    m = num_tokens
    checkAllclose(
        k_ref,
        case["k_prefix"][:m].float(),
        atol=1e-2,
        rtol=1e-2,
        msg="k_prefix, row-major weight",
    )
    checkAllclose(
        v_ref,
        case["v_prefix"][:m].float(),
        atol=1e-2,
        rtol=1e-2,
        msg="v_prefix, row-major weight",
    )

    shuffled = _make_case(num_tokens, 12)
    _run_flydsl(shuffled, weight_preshuffle=True)
    assert torch.equal(
        case["k_prefix"], shuffled["k_prefix"]
    ), "row-major != preshuffled"
    assert torch.equal(
        case["v_prefix"], shuffled["v_prefix"]
    ), "row-major != preshuffled"


@_SKIP
@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"shuffled_kv_cache": True}, "shuffled_kv_cache"),
        ({"block_m": 192}, "BLOCK_M"),
    ],
)
def test_gather_kv_b_proj_flydsl_rejects_unsupported(kwargs, needle):
    """Unsupported configurations must raise, never silently miscompute."""
    case = _make_case(256, 12)
    with pytest.raises(ValueError, match=needle):
        _run_flydsl(case, **kwargs)


@_SKIP
@pytest.mark.parametrize("num_tokens, block_m", [(16384, 256), (8192, 128)])
def test_gather_kv_b_proj_flydsl_determinism_large_m(num_tokens, block_m):
    """Repeated identical launches must agree bitwise."""
    case = _make_case(num_tokens, 12, num_blocks=3_000_000)
    _run_flydsl(case, block_m=block_m)
    first_k = case["k_prefix"].clone()
    first_v = case["v_prefix"].clone()
    for run in range(2, 17):
        case["k_prefix"].zero_()
        case["v_prefix"].zero_()
        _run_flydsl(case, block_m=block_m)
        assert torch.equal(case["k_prefix"], first_k), f"k_prefix differs on run {run}"
        assert torch.equal(case["v_prefix"], first_v), f"v_prefix differs on run {run}"


@_SKIP
@pytest.mark.parametrize("block_m", [128, 256, 384])
def test_gather_kv_b_proj_flydsl_rope_is_complete(block_m):
    """Every rope row must be written, and be a bitwise copy, for any BLOCK_M.

    The fused rope copy maps 512 threads onto 256 rows per pass, so BLOCK_M > 256
    needs more than one pass. A single pass leaves rows 256.. of every tile
    unwritten while the GEMM half stays perfectly correct -- silent missing data
    that an accuracy check on k_nope / v cannot see.
    """
    # 1536 is divisible by all three block_m under test, so no tail masking
    # confounds the completeness check.
    case = _make_case(1536, 12)
    case["k_prefix"].fill_(float("nan"))
    _run_flydsl(case, block_m=block_m)

    rope = case["k_prefix"][:, :, QK_NOPE_HEAD_DIM:]
    assert not torch.isnan(rope).any(), f"block_m={block_m}: rope rows left unwritten"

    idx = case["kv_indices"][: case["num_tokens"]].long()
    want = case["k_buffer"][idx].reshape(-1, KV_C_DIM + KV_PE_DIM)[:, KV_C_DIM:]
    want = want.to(torch.bfloat16).unsqueeze(1).expand(-1, case["n_heads"], -1)
    # k_scale is 1.0 here, so the rope is a pure copy and must match bitwise.
    assert torch.equal(rope, want), f"block_m={block_m}: rope is not a bitwise copy"


@_SKIP
def test_gather_kv_b_proj_flydsl_tail_is_untouched():
    """Rows past ``num_tokens`` must not be written.

    This is the check that catches a wrong ``num_records_bytes``: the caller
    preallocates the workspace at its maximum, so a bound derived from the
    tensor extent instead of the live row count would let tail workgroups
    scribble on data the caller still owns.
    """
    m, alloc = 1000, 1024
    case = _make_case(m, 12, alloc)
    case["k_prefix"].fill_(float("nan"))
    case["v_prefix"].fill_(float("nan"))
    _run_flydsl(case)
    assert torch.isnan(case["k_prefix"][m:]).all(), "k_prefix tail was clobbered"
    assert torch.isnan(case["v_prefix"][m:]).all(), "v_prefix tail was clobbered"
    assert not torch.isnan(case["k_prefix"][:m]).any(), "k_prefix live rows unwritten"
    assert not torch.isnan(case["v_prefix"][:m]).any(), "v_prefix live rows unwritten"


@_SKIP
@pytest.mark.parametrize("num_tokens", [512, 2048])
def test_gather_kv_b_proj_flydsl_matches_triton(num_tokens):
    """Both backends consume the same preshuffled weight tensor."""
    n_heads = 12
    case = _make_case(num_tokens, n_heads)
    w_shuffled = shuffle_weight(case["weight"], layout=(16, 16))
    _run_flydsl(case)

    k_tri = torch.zeros_like(case["k_prefix"])
    v_tri = torch.zeros_like(case["v_prefix"])
    triton_gather_kv_b_proj(
        case["k_buffer"],
        case["k_scale"],
        case["kv_indptr"],
        case["kv_indices"],
        case["cu_seqlens_k"],
        w_shuffled,
        case["weight_scale"],
        k_tri,
        v_tri,
        weight_preshuffle=True,
    )
    checkAllclose(
        k_tri.float(),
        case["k_prefix"].float(),
        atol=2e-2,
        rtol=2e-2,
        msg="k_prefix: flydsl vs triton",
    )
    checkAllclose(
        v_tri.float(),
        case["v_prefix"].float(),
        atol=2e-2,
        rtol=2e-2,
        msg="v_prefix: flydsl vs triton",
    )


def _bench(num_tokens, n_heads):
    case = _make_case(num_tokens, n_heads)
    w_shuffled = shuffle_weight(case["weight"], layout=(16, 16))
    args = (
        case["k_buffer"],
        case["k_scale"],
        case["kv_indptr"],
        case["kv_indices"],
        case["cu_seqlens_k"],
        w_shuffled,
        case["weight_scale"],
        case["k_prefix"],
        case["v_prefix"],
    )
    _, us_tri = run_perftest(triton_gather_kv_b_proj, *args, weight_preshuffle=True)
    _, us_fly = run_perftest(gather_kv_b_proj_flydsl, *args, num_tokens=num_tokens)
    weight_n = n_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    tflops = 2 * num_tokens * weight_n * KV_C_DIM / us_fly * 1e-6
    out_gb = (
        num_tokens
        * n_heads
        * (QK_NOPE_HEAD_DIM + KV_PE_DIM + V_HEAD_DIM)
        * 2
        / us_fly
        * 1e-3
    )
    return us_tri, us_fly, tflops, out_gb


@_SKIP
@pytest.mark.parametrize("scale_mode", ["row", "block"])
@pytest.mark.parametrize("weight_preshuffle", [False, True])
@pytest.mark.parametrize("num_tokens,block_m", [(1, 128), (257, 256), (8192, 256)])
def test_fp8_output(scale_mode, weight_preshuffle, num_tokens, block_m):
    case = _make_case(
        num_tokens,
        16,
        alloc=num_tokens + 31,
        duplicate_indices=True,
        k_scale_value=0.37,
        scale_mode=scale_mode,
    )
    for key in ("k_prefix", "v_prefix"):
        case[key] = torch.full_like(case[key], 7).to(torch.float8_e4m3fn)
    ks = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    vs = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    _run_flydsl(
        case,
        weight_preshuffle=weight_preshuffle,
        block_m=block_m,
        k_out_scale=ks,
        v_out_scale=vs,
    )
    refs = _torch_ref(case)
    for key, ref, scale in zip(("k_prefix", "v_prefix"), refs, (ks, vs)):
        actual = case[key][:num_tokens].float() * scale
        # E4M3 has at most 1/16 relative rounding error; the absolute term
        # covers subnormals and fp32 GEMM cancellation near zero.
        torch.testing.assert_close(actual, ref, rtol=0.065, atol=0.001)
        assert (case[key][num_tokens:].float() == 7).all()
    # RoPE must use K's descale too, despite bypassing the GEMM epilogue.
    expected_rope = (refs[0][..., 128:] / ks).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(
        case["k_prefix"][:num_tokens, :, 128:].view(torch.int8),
        expected_rope.view(torch.int8),
    )


@_SKIP
@pytest.mark.parametrize("zero", [False, True])
def test_fp8_saturation_and_zero(zero):
    case = _make_case(259, 2, k_scale_value=2.0)
    case["k_buffer"].fill_(0 if zero else 1)
    case["weight"].fill_(1)
    case["weight"][128:256].fill_(-1)
    case["weight_scale"].fill_(1)
    for key in ("k_prefix", "v_prefix"):
        case[key] = case[key].to(torch.float8_e4m3fn)
    scale = torch.tensor([0.001], device="cuda")
    _run_flydsl(case, k_out_scale=scale, v_out_scale=scale)
    for key, ref in zip(("k_prefix", "v_prefix"), _torch_ref(case)):
        actual = case[key].float()
        expected = (ref / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.isfinite().all()
        if not zero:
            assert (actual.abs() == 448).any()


@_SKIP
@pytest.mark.parametrize(
    "invalid",
    ["missing", "shape", "dtype", "device", "mixed_outputs", "bf16_scales", "strided"],
)
def test_fp8_output_validation(invalid):
    case = _make_case(256, 2)
    if invalid != "bf16_scales":
        for key in ("k_prefix", "v_prefix"):
            case[key] = case[key].to(torch.float8_e4m3fn)
    ks = vs = torch.ones(1, device="cuda")
    if invalid == "missing":
        ks = None
    elif invalid == "shape":
        ks = torch.ones(2, device="cuda")
    elif invalid == "dtype":
        ks = ks.to(torch.bfloat16)
    elif invalid == "device":
        ks = ks.cpu()
    elif invalid == "mixed_outputs":
        case["v_prefix"] = case["v_prefix"].to(torch.bfloat16)
    elif invalid == "strided":
        case["k_prefix"] = case["k_prefix"].transpose(0, 1).contiguous().transpose(0, 1)
    with pytest.raises(ValueError):
        _run_flydsl(case, k_out_scale=ks, v_out_scale=vs)


@_SKIP
def test_fp8_graph_replay_uses_current_scales():
    case = _make_case(257, 2)
    for key in ("k_prefix", "v_prefix"):
        case[key] = case[key].to(torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda")
    weight = shuffle_weight(case["weight"], layout=(16, 16))

    def run():
        gather_kv_b_proj_flydsl(
            case["k_buffer"],
            case["k_scale"],
            case["kv_indptr"],
            case["kv_indices"],
            case["cu_seqlens_k"],
            weight,
            case["weight_scale"],
            case["k_prefix"],
            case["v_prefix"],
            k_out_scale=scale,
            v_out_scale=scale,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    scale.fill_(0.5)
    graph.replay()
    for key, ref in zip(("k_prefix", "v_prefix"), _torch_ref(case)):
        torch.testing.assert_close(
            case[key].float() * scale, ref, rtol=0.065, atol=0.001
        )


def _bench_fp8(num_tokens, n_heads):
    from aiter.ops.quant import per_tensor_quant_hip

    case = _make_case(num_tokens, n_heads)
    args = (
        case["k_buffer"],
        case["k_scale"],
        case["kv_indptr"],
        case["kv_indices"],
        case["cu_seqlens_k"],
        shuffle_weight(case["weight"], layout=(16, 16)),
        case["weight_scale"],
    )
    k, v = case["k_prefix"], case["v_prefix"]

    def separate():
        gather_kv_b_proj_flydsl(*args, k, v)
        # These benchmark shapes admit 256 rows of complete 16-element vectors.
        # Per-tensor quantization is shape-independent; a compact row count
        # avoids inflating the HIP amax kernel's atomic-reduction overhead.
        return tuple(
            per_tensor_quant_hip(x.view(256, -1), quant_dtype=torch.float8_e4m3fn)
            for x in (k, v)
        )

    (_, ks), (_, vs) = separate()
    k8, v8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in (k, v))

    def fused():
        gather_kv_b_proj_flydsl(*args, k8, v8, k_out_scale=ks, v_out_scale=vs)

    fused()
    for actual, ref, scale in zip((k8, v8), _torch_ref(case), (ks, vs)):
        torch.testing.assert_close(actual.float() * scale, ref, rtol=0.065, atol=0.01)
    _, separate_us = run_perftest(separate)
    _, fused_us = run_perftest(fused)
    return separate_us, fused_us


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL gather_kv_b_proj unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("-heads", type=int, default=12, help="tp_k_head_num")
    parser.add_argument(
        "--fp8-output",
        action="store_true",
        help="compare BF16 gather + HIP K/V quantization against direct FP8 output",
    )
    args = parser.parse_args()

    if args.fp8_output:
        print(f"\n## FP8 gather output, {args.heads} heads, K=512\n")
        print("Caller-provided scales are prepared before timing the fused path.")
        print("The HIP per-tensor baseline uses a 256-row view of each output.")
        print("| M | BF16 gather + quant us | FP8 gather us | speedup |")
        print("|---|---|---|---|")
        for m in (2048, 8192, 16384):
            separate_us, fused_us = _bench_fp8(m, args.heads)
            print(
                f"| {m} | {separate_us:.2f} | {fused_us:.2f} | "
                f"{separate_us / fused_us:.2f}x |"
            )
        return

    rows = []
    for m in (2048, 8192, 16384):
        rows.append((m, *_bench(m, args.heads)))

    print(f"\n## gather_kv_b_proj, {args.heads} heads, K=512\n")
    print("| M | triton us | flydsl us | speedup | flydsl TFLOPS | out GB/s |")
    print("|---|---|---|---|---|---|")
    for m, us_t, us_f, tf, gb in rows:
        print(
            f"| {m} | {us_t:.2f} | {us_f:.2f} | {us_t / us_f:.2f}x | {tf:.1f} | {gb:.1f} |"
        )


if __name__ == "__main__":
    main()
