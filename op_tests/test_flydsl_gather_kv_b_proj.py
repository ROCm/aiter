# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused gather + kv_b_proj (MLA prefix expansion) tests.

Covers the one configuration the FlyDSL backend implements: page_size 1, fp8 KV
cache, fp8 ``shuffle_weight((16,16))`` weight, per-output-row weight scale,
per-tensor activation scale, bf16 or scaled fp8 outputs, gfx950 -- at both the DeepSeek
128+128 head and the GLM-5.2 192+256 one.

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
from aiter.ops.flydsl.gather_kv_b_proj import (
    gather_kv_b_proj_flydsl,
    gather_kv_b_proj_flydsl_supported,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gather_kv_b_proj import (
    gather_kv_b_proj as triton_gather_kv_b_proj,
)
from aiter.test_common import checkAllclose, run_perftest

KV_C_DIM = 512
KV_PE_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128

# (qk_nope_head_dim, v_head_dim). DeepSeek fills both B LDS halves exactly; GLM-5.2
# is unequal, so the k half pads to 256 and its last 64 columns must be dropped.
DIMS_DEEPSEEK = (QK_NOPE_HEAD_DIM, V_HEAD_DIM)
DIMS_GLM = (192, 256)

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
    dims=DIMS_DEEPSEEK,
    output_dtype=torch.bfloat16,
):
    """Build one page_size-1 gather case.

    ``alloc`` models the real caller: the chunk workspace is preallocated at its
    maximum and only ``num_tokens`` rows are live.
    """
    torch.manual_seed(seed)
    nope, v_dim = dims
    alloc = alloc or num_tokens
    num_blocks = num_blocks or max(alloc, 64)
    weight_n = n_heads * (nope + v_dim)

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
        (alloc, n_heads, nope + KV_PE_DIM),
        device=device,
        dtype=output_dtype,
    )
    v_prefix = torch.zeros((alloc, n_heads, v_dim), device=device, dtype=output_dtype)
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
        "nope": nope,
        "v_dim": v_dim,
        "out_scales": (
            {
                "k_out_scale": torch.tensor([0.73], device=device),
                "v_out_scale": torch.tensor([0.53], device=device),
            }
            if output_dtype == torch.float8_e4m3fn
            else {}
        ),
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
    nope, v_dim = case["nope"], case["v_dim"]
    proj = ((kv_c @ w.T) * scale).view(m, n_heads, nope + v_dim)
    k_nope, v = proj.split([nope, v_dim], dim=-1)
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
        **{**case["out_scales"], **kw},
    )


_OUTPUT_DTYPES = pytest.mark.parametrize(
    "output_dtype", [torch.bfloat16, torch.float8_e4m3fn]
)


def _check_output(case):
    for key, ref, scale_name in zip(
        ("k_prefix", "v_prefix"), _torch_ref(case), ("k_out_scale", "v_out_scale")
    ):
        actual = case[key][: case["num_tokens"]].float()
        if case["out_scales"]:
            # BF16 absolute tolerance plus E4M3's 1/16 normal-value rounding error.
            actual *= case["out_scales"][scale_name]
            assert (
                checkAllclose(
                    actual, ref, rtol=0.065, atol=1e-2, tol_err_ratio=0, msg=key
                )
                == 0
            )
        else:
            checkAllclose(ref, actual, atol=1e-2, rtol=1e-2, msg=key)


@_SKIP
@pytest.mark.parametrize("dims", [DIMS_DEEPSEEK, DIMS_GLM])
@pytest.mark.parametrize(
    "num_tokens, n_heads, alloc, duplicate_indices, k_scale_value",
    [
        (512, 12, None, False, 1.0),
        # Enough tiles that the default BLOCK_M steps up to 256 for either head.
        (4096, 12, None, False, 1.0),
        # M not a multiple of BLOCK_M, and k_scale is 1.0 in the current
        # deployment but must not be assumed.
        (1000, 12, 1024, False, 0.37),
        # The prefix cache repeats slot ids.
        (777, 16, 1024, True, 1.0),
        (1, 12, 256, False, 1.0),
    ],
)
@_OUTPUT_DTYPES
def test_gather_kv_b_proj_flydsl(
    num_tokens, n_heads, alloc, duplicate_indices, k_scale_value, dims, output_dtype
):
    case = _make_case(
        num_tokens,
        n_heads,
        alloc,
        duplicate_indices,
        k_scale_value,
        output_dtype=output_dtype,
        dims=dims,
    )
    _run_flydsl(case)
    m = num_tokens
    _check_output(case)
    _, v_ref = _torch_ref(case)

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
@_OUTPUT_DTYPES
def test_gather_kv_b_proj_flydsl_block_scale(
    num_tokens, n_heads, alloc, k_scale_value, output_dtype
):
    """Check accumulator rescaling between K tiles with 128x128 weight scales."""
    case = _make_case(
        num_tokens,
        n_heads,
        alloc,
        k_scale_value=k_scale_value,
        scale_mode="block",
        output_dtype=output_dtype,
    )
    _run_flydsl(case)
    _check_output(case)


@_SKIP
@pytest.mark.parametrize("num_tokens", [512, 1000])
@_OUTPUT_DTYPES
@pytest.mark.parametrize(
    "dims,scale_mode",
    [(DIMS_DEEPSEEK, "row"), (DIMS_DEEPSEEK, "block"), (DIMS_GLM, "row")],
)
def test_gather_kv_b_proj_flydsl_row_major_weight(
    num_tokens, dims, output_dtype, scale_mode
):
    """Row-major and preshuffled weight layouts must produce identical outputs."""
    case = _make_case(
        num_tokens, 12, output_dtype=output_dtype, scale_mode=scale_mode, dims=dims
    )
    _run_flydsl(case, weight_preshuffle=False)
    _check_output(case)
    first = [case[key].clone() for key in ("k_prefix", "v_prefix")]
    _run_flydsl(case, weight_preshuffle=True)
    for key, expected in zip(("k_prefix", "v_prefix"), first):
        assert torch.equal(
            case[key].view(torch.uint8), expected.view(torch.uint8)
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


def _supported(case, **kw):
    return gather_kv_b_proj_flydsl_supported(
        case["k_buffer"],
        shuffle_weight(case["weight"], layout=(16, 16)),
        case["weight_scale"],
        case["k_prefix"],
        case["v_prefix"],
        **kw,
    )


@_SKIP
@pytest.mark.parametrize("dims", [DIMS_DEEPSEEK, DIMS_GLM])
@_OUTPUT_DTYPES
def test_gather_kv_b_proj_flydsl_supported_agrees_with_the_op(dims, output_dtype):
    """The support predicate must agree with launch validation."""
    ok = _make_case(256, 12, output_dtype=output_dtype, dims=dims)
    assert _supported(ok)
    _run_flydsl(ok)  # and it really runs

    for kw, needle in (
        ({"shuffled_kv_cache": True}, "shuffled_kv_cache"),
        ({"block_m": 192}, "BLOCK_M"),
    ):
        assert not _supported(ok, **kw)
        with pytest.raises(ValueError, match=needle):
            _run_flydsl(ok, **kw)


@_SKIP
@pytest.mark.parametrize("break_it", ["bf16_cache", "no_scale", "mxfp4_weight"])
def test_gather_kv_b_proj_flydsl_declines_what_triton_covers(break_it):
    """The three shapes ATOM hands this op that only the Triton one can serve.

    Two of them took down a CI accuracy job apiece: the FlyDSL op raised, and
    ATOM had gated on "did the import succeed" rather than on this predicate,
    so the engine died instead of using the fallback sitting next to it.
    """
    case = _make_case(256, 12)
    if break_it == "bf16_cache":
        case["k_buffer"] = case["k_buffer"].to(torch.bfloat16)  # GLM-5.2
    elif break_it == "no_scale":
        case["weight_scale"] = None  # Kimi-K3 DSpark
    else:
        case["weight"] = case["weight"].view(torch.uint8)  # MXFP4 kv_b_proj

    assert not _supported(case)
    with pytest.raises(ValueError):
        _run_flydsl(case)


def _sparse_case(num_blocks, m, n_heads, lo, dims=DIMS_DEEPSEEK):
    """A case whose cache is zero except the ``m`` rows it gathers, all at or
    above row ``lo``.

    ``_make_case`` tiles its content every 4096 rows, so a misaddressed load
    there can return data that still looks plausible. Zero everywhere else means
    a wrong address can only come back as zeros, which the reference never is.
    """
    torch.manual_seed(0)
    row = KV_C_DIM + KV_PE_DIM
    k_buffer = torch.zeros((num_blocks, 1, row), device="cuda", dtype=torch.uint8)
    kv_indices = (torch.randperm(num_blocks - lo, device="cuda")[:m] + lo).to(
        torch.int32
    )
    # 0x01..0x76 is positive-finite e4m3 -- no NaN to poison the reference.
    k_buffer[kv_indices.long()] = torch.randint(
        1, 0x77, (m, 1, row), device="cuda", dtype=torch.uint8
    )
    case = _make_case(m, n_heads, num_blocks=4096, dims=dims)
    case["k_buffer"] = k_buffer.view(dtypes.fp8)
    case["kv_indices"] = kv_indices
    return case


def _check_sparse_case(case):
    """Assert on cosine, not on checkAllclose, which only warns.

    These cases feed the full positive-finite e4m3 range through a 512-deep dot
    product, so the bf16 result carries a real absolute error at the top of the
    range; cosine is the scale-free statement, and a gather that reads the wrong
    row lands nowhere near 1 because everything off the gathered rows is zero.
    """
    k_ref, v_ref = _torch_ref(case)
    _run_flydsl(case)
    checkAllclose(k_ref, case["k_prefix"].float(), rtol=2e-2, atol=2e-2, msg="k_prefix")
    checkAllclose(v_ref, case["v_prefix"].float(), rtol=2e-2, atol=2e-2, msg="v_prefix")
    for name, got, ref in (
        ("k_prefix", case["k_prefix"], k_ref),
        ("v_prefix", case["v_prefix"], v_ref),
    ):
        cos = torch.nn.functional.cosine_similarity(
            got.float().flatten(), ref.flatten(), dim=0
        )
        assert cos > 0.999, f"{name} cosine {cos:.6f} -- gathered the wrong rows"
    assert (case["k_prefix"] != 0).any(), "every gathered row read back as zero"
    torch.cuda.empty_cache()


@_SKIP
def test_gather_kv_b_proj_flydsl_spans_past_2gib():
    """A cache past 2**31 bytes, gathered only above that boundary.

    The DeepSeek-R1-0528 tp4 shape, and two distinct 32-bit edges: FlyDSL packs
    memref shapes as i32, so the cache goes in as a pointer plus ``num_blocks``
    rather than a flattened memref; and the descriptor form's ``idx * 576`` is
    an i32 whose wrap is harmless only because voffset is read unsigned.
    """
    row = KV_C_DIM + KV_PE_DIM
    num_blocks = 4_458_592
    assert 2**31 < num_blocks * row < 2**32, "must land in the descriptor's range"
    _check_sparse_case(_sparse_case(num_blocks, 256, 12, 2**31 // row + 1))


@_SKIP
@pytest.mark.parametrize(
    "num_blocks, wide, dims",
    [
        # Brackets the switch: the widest cache one descriptor still spans, then
        # the narrowest it does not. Both gather only from the top of the cache.
        ((2**32 - 1) // (KV_C_DIM + KV_PE_DIM), False, DIMS_DEEPSEEK),
        (2**32 // (KV_C_DIM + KV_PE_DIM) + 1, True, DIMS_DEEPSEEK),
        (9_000_000, True, DIMS_DEEPSEEK),  # 4.83 GiB -- well past, not just over
        # The wide A path addresses the cache; a split head moves B and the
        # epilogue's column origin. Nothing is shared but the rope copy, whose
        # source one changes and destination the other, so pin them composing.
        (9_000_000, True, DIMS_GLM),
    ],
)
def test_gather_kv_b_proj_flydsl_brackets_the_descriptor_span(num_blocks, wide, dims):
    """Either side of 4 GiB must agree with the reference.

    Past it the kernel drops the buffer descriptor for ``global_load_lds`` over
    64-bit per-lane addresses: a gather's row is per-lane, so it can never be
    folded into an SRsrc base the way a tile scan's can.
    """
    row = KV_C_DIM + KV_PE_DIM
    assert (num_blocks * row >= 2**32) == wide
    lo = 2**31 // row + 1
    _check_sparse_case(_sparse_case(num_blocks, 256, 12, lo, dims=dims))


@_SKIP
def test_gather_kv_b_proj_flydsl_rejects_block_scale_unequal_dims():
    """A 128x128 block scale cuts across the k/v split unless both halves are 128,
    which the one-scalar-per-half epilogue cannot represent."""
    case = _make_case(256, 8, scale_mode="block", dims=DIMS_GLM)
    assert not _supported(case)
    with pytest.raises(ValueError, match="block scale"):
        _run_flydsl(case)


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
@pytest.mark.parametrize("dims", [DIMS_DEEPSEEK, DIMS_GLM])
@_OUTPUT_DTYPES
def test_gather_kv_b_proj_flydsl_rope_is_complete(dims, block_m, output_dtype):
    """Check every RoPE row, including the extra copy pass for BLOCK_M > 256."""
    # Divisible by all tested tile heights to exclude tail masking.
    case = _make_case(
        1536,
        12,
        output_dtype=output_dtype,
        dims=dims,
        k_scale_value=0.43 if output_dtype == torch.float8_e4m3fn else 1.0,
    )
    case["k_prefix"].fill_(float("nan"))
    _run_flydsl(case, block_m=block_m)

    rope = case["k_prefix"][:, :, case["nope"] :]
    assert not torch.isnan(
        rope.float()
    ).any(), f"block_m={block_m}: rope rows left unwritten"

    idx = case["kv_indices"][: case["num_tokens"]].long()
    want = case["k_buffer"][idx].reshape(-1, KV_C_DIM + KV_PE_DIM)[:, KV_C_DIM:]
    if case["out_scales"]:
        scale = case["k_scale"] * case["out_scales"]["k_out_scale"].reciprocal()
        want = (want.float() * scale).clamp(-448, 448)
    want = want.to(output_dtype).unsqueeze(1).expand(-1, case["n_heads"], -1)
    assert torch.equal(
        rope.view(torch.uint8), want.view(torch.uint8)
    ), f"block_m={block_m}: incorrect rope"


@_SKIP
@pytest.mark.parametrize("dims", [DIMS_DEEPSEEK, DIMS_GLM])
@_OUTPUT_DTYPES
def test_gather_kv_b_proj_flydsl_tail_is_untouched(dims, output_dtype):
    """Store bounds must use the live row count, preserving the allocated tail."""
    m, alloc = 1000, 1024
    case = _make_case(m, 12, alloc, output_dtype=output_dtype, dims=dims)
    case["k_prefix"].fill_(float("nan"))
    case["v_prefix"].fill_(float("nan"))
    _run_flydsl(case)
    assert torch.isnan(
        case["k_prefix"][m:].float()
    ).all(), "k_prefix tail was clobbered"
    assert torch.isnan(
        case["v_prefix"][m:].float()
    ).all(), "v_prefix tail was clobbered"
    assert not torch.isnan(
        case["k_prefix"][:m].float()
    ).any(), "k_prefix live rows unwritten"
    assert not torch.isnan(
        case["v_prefix"][:m].float()
    ).any(), "v_prefix live rows unwritten"


@_SKIP
@pytest.mark.parametrize("num_tokens", [512, 2048])
@pytest.mark.parametrize("dims", [DIMS_DEEPSEEK, DIMS_GLM])
def test_gather_kv_b_proj_flydsl_matches_triton(num_tokens, dims):
    """Both backends consume the same preshuffled weight tensor."""
    n_heads = 12
    case = _make_case(num_tokens, n_heads, dims=dims)
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


def _bench(num_tokens, n_heads, dims):
    case = _make_case(num_tokens, n_heads, dims=dims)
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
    nope, v_dim = dims
    weight_n = n_heads * (nope + v_dim)
    tflops = 2 * num_tokens * weight_n * KV_C_DIM / us_fly * 1e-6
    out_gb = num_tokens * n_heads * (nope + KV_PE_DIM + v_dim) * 2 / us_fly * 1e-3
    return us_tri, us_fly, tflops, out_gb


@_SKIP
@pytest.mark.parametrize(
    "descale,projection",
    [(0.37, 7.029999732971191), (0.73, 78.83999633789062)],
)
def test_fp8_reciprocal_rounding(descale, projection):
    # Exact dot products that hit E4M3 ties with reciprocal multiplication,
    # but fall below the ties with division.
    case = _make_case(
        257, 12, k_scale_value=projection, output_dtype=torch.float8_e4m3fn
    )
    case["k_buffer"].zero_()
    case["k_buffer"][:, 0, 0] = 1
    case["k_buffer"][:, 0, 512:] = 1
    case["weight"].zero_()
    case["weight"][:, 0] = 1
    case["weight_scale"].fill_(1)
    scale = torch.tensor([descale], device="cuda", dtype=torch.float32)
    # CPU reference prevents device compiler rewrites of division.
    value = torch.tensor([projection], dtype=torch.float32)
    scale_cpu = torch.tensor([descale], dtype=torch.float32)
    expected = (value * scale_cpu.reciprocal()).to(torch.float8_e4m3fn)
    divided = (value / scale_cpu).to(torch.float8_e4m3fn)
    assert not torch.equal(expected.view(torch.int8), divided.view(torch.int8))
    expected = expected.to(device="cuda")
    _run_flydsl(case, k_out_scale=scale, v_out_scale=scale)
    for key in ("k_prefix", "v_prefix"):
        assert torch.equal(
            case[key].view(torch.int8),
            expected.view(torch.int8).expand_as(case[key]),
        )


@_SKIP
@pytest.mark.parametrize("zero", [False, True])
def test_fp8_saturation_and_zero(zero):
    case = _make_case(259, 12, k_scale_value=2.0, output_dtype=torch.float8_e4m3fn)
    case["k_buffer"].fill_(0 if zero else 1)
    case["weight"].fill_(1)
    case["weight"][128:256].fill_(-1)
    case["weight_scale"].fill_(1)
    scale = torch.tensor([0.001], device="cuda")
    _run_flydsl(case, k_out_scale=scale, v_out_scale=scale)
    for key, ref in zip(("k_prefix", "v_prefix"), _torch_ref(case)):
        actual = case[key].float()
        expected = (
            (ref * scale.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        )
        assert (
            checkAllclose(actual, expected, rtol=0, atol=0, tol_err_ratio=0, msg=key)
            == 0
        )
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
    case = _make_case(257, 12, output_dtype=torch.float8_e4m3fn)
    _run_flydsl(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_flydsl(case)
    case["out_scales"]["k_out_scale"].mul_(1.5)
    case["out_scales"]["v_out_scale"].mul_(2.0)
    graph.replay()
    _check_output(case)


def _bench_fp8(num_tokens, n_heads, dims):
    from aiter.ops.quant import per_tensor_quant_hip

    case = _make_case(num_tokens, n_heads, dims=dims)
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

    def dynamic_quant():
        gather_kv_b_proj_flydsl(*args, k, v)
        # A 256-row view reduces HIP amax overhead while preserving vector alignment.
        return tuple(
            per_tensor_quant_hip(x.view(256, -1), quant_dtype=torch.float8_e4m3fn)
            for x in (k, v)
        )

    (_, ks), (_, vs) = dynamic_quant()
    k8, v8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in (k, v))

    def supplied_scale_quant():
        gather_kv_b_proj_flydsl(*args, k, v)
        return tuple(
            per_tensor_quant_hip(
                x.view(256, -1), scale=scale, quant_dtype=torch.float8_e4m3fn
            )
            for x, scale in ((k, ks), (v, vs))
        )

    def fused():
        gather_kv_b_proj_flydsl(*args, k8, v8, k_out_scale=ks, v_out_scale=vs)

    fused()
    for actual, ref, scale in zip((k8, v8), _torch_ref(case), (ks, vs)):
        assert (
            checkAllclose(
                actual.float() * scale,
                ref,
                rtol=0.065,
                atol=0.01,
                tol_err_ratio=0,
                msg="fp8 gather benchmark",
            )
            == 0
        )
    _, dynamic_us = run_perftest(dynamic_quant)
    _, supplied_scale_us = run_perftest(supplied_scale_quant)
    _, fused_us = run_perftest(fused)
    return dynamic_us, supplied_scale_us, fused_us


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL gather_kv_b_proj unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("-heads", type=int, default=12, help="tp_k_head_num")
    parser.add_argument(
        "-dims", type=int, nargs=2, default=DIMS_GLM, help="qk_nope_head_dim v_head_dim"
    )
    parser.add_argument(
        "--fp8-output",
        action="store_true",
        help="compare BF16 gather + HIP K/V quantization against direct FP8 output",
    )
    args = parser.parse_args()
    dims = tuple(args.dims)

    if args.fp8_output:
        print(
            f"\n## FP8 gather output {dims[0]}+{dims[1]}, {args.heads} heads, K=512\n"
        )
        print("The supplied-scale and fused paths reuse scales prepared before timing.")
        print("The HIP per-tensor baseline uses a 256-row view of each output.")
        print(
            "| M | BF16 gather + dynamic quant us | BF16 gather + supplied-scale quant us "
            "| FP8 gather us | speedup vs dynamic | speedup vs supplied-scale |"
        )
        print("|---|---|---|---|---|---|")
        for m in (2048, 8192, 16384):
            dynamic_us, supplied_scale_us, fused_us = _bench_fp8(m, args.heads, dims)
            print(
                f"| {m} | {dynamic_us:.2f} | {supplied_scale_us:.2f} | {fused_us:.2f} | "
                f"{dynamic_us / fused_us:.2f}x | {supplied_scale_us / fused_us:.2f}x |"
            )
        return

    rows = []
    for m in (2048, 8192, 16384):
        rows.append((m, *_bench(m, args.heads, dims)))

    print(f"\n## gather_kv_b_proj {dims[0]}+{dims[1]}, {args.heads} heads, K=512\n")
    print("| M | triton us | flydsl us | speedup | flydsl TFLOPS | out GB/s |")
    print("|---|---|---|---|---|---|")
    for m, us_t, us_f, tf, gb in rows:
        print(
            f"| {m} | {us_t:.2f} | {us_f:.2f} | {us_t / us_f:.2f}x | {tf:.1f} | {gb:.1f} |"
        )


if __name__ == "__main__":
    main()
