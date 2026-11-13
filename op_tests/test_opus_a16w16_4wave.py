# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 a16w16 4-wave kids (pre-built .co device images, gen_co/gfx950).

Kids 1000-1007 (``a16w16_4wave``) launch directly (bias fused in the kernel);
kids 1008-1015 (``a16w16_4wave_ws``) always write fp32 split-K partials into a
workspace and reduce them (bias folded by the reduce). Every kid detects the
operand layout (NT / NN / TT / TN) from the tensor strides at runtime.

Run: ``pytest op_tests/test_opus_a16w16_4wave.py``
"""

import os

import pytest
import torch

from aiter.ops.opus import gemm_a16w16_opus, opus_bmm, opus_gemm
from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
from csrc.opus_gemm.opus_gemm_common import (
    A16W16_4WAVE_LAYOUTS,
    CO_KERNELS_GFX950_JSON,
    DEFAULT_COMPILED_KIDS_BY_NAMESPACE,
    GFX950_A16W16_4WAVE_KIDS,
    KERNEL_REGISTRY,
    OPUS_KERNEL_TAGS_BY_ARCH_FAMILY,
    a16w16_4wave_co_image_paths,
    a16w16_4wave_co_images,
    get_kernel_instance,
    get_registry,
    gfx950_a16w16_4wave_kernels_declared,
    group_kids,
    kernel_needs_external_workspace,
)

DIRECT_KIDS = tuple(range(1000, 1008))
WS_KIDS = tuple(range(1008, 1016))
TILES = ((256, 256), (256, 128), (128, 256), (128, 128), (256, 64), (64, 256), (128, 64), (64, 128))
# layout -> (A stored [K, M], B stored [K, N])
LAYOUTS = {"nt": (False, False), "nn": (False, True), "tt": (True, False), "tn": (True, True)}


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(props.gcnArchName).split(":", 1)[0].lower()


requires_gfx950 = pytest.mark.skipif(
    _runtime_arch() != "gfx950", reason="requires gfx950 hardware"
)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _operand(batch, rows, K, transposed, pad, gen):
    """Logical [batch, rows, K] bf16 view over padded storage.

    transposed=False: stored [rows, K] with row pitch K + pad (K-contiguous).
    transposed=True:  stored [K, rows] with row pitch round_up(rows, 8) + pad.
    """
    if not transposed:
        ld = K + pad
        base = torch.randn((batch, rows, ld), generator=gen, device="cuda")
        return base.to(torch.bfloat16)[:, :, :K]
    ld = _round_up(rows, 8) + pad
    base = torch.randn((batch, K, ld), generator=gen, device="cuda")
    return base.to(torch.bfloat16)[:, :, :rows].transpose(1, 2)


def _golden(A, B, bias=None):
    out = A.float() @ B.float().transpose(-1, -2)
    if bias is not None:
        out = out + bias.float()
    return out


def _assert_close(actual, golden, msg):
    # One final bf16 rounding of an fp32 accumulation (same bound as the other
    # A16W16 bf16-output tests).
    torch.testing.assert_close(actual.float(), golden, rtol=0.03, atol=0.5, msg=msg)


# ---------------------------------------------------------------------------
# Registry / plan (no GPU needed)
# ---------------------------------------------------------------------------


def test_registry_declares_16_kids_with_images():
    assert set(gfx950_a16w16_4wave_kernels_declared) == set(DIRECT_KIDS + WS_KIDS)
    assert GFX950_A16W16_4WAVE_KIDS == frozenset(DIRECT_KIDS + WS_KIDS)
    for kid, (bm, bn) in zip(DIRECT_KIDS + WS_KIDS, TILES + TILES):
        inst = gfx950_a16w16_4wave_kernels_declared[kid]
        assert (inst.B_M, inst.B_N, inst.B_K, inst.BLOCK_SIZE) == (bm, bn, 64, 256)
        assert inst.WG_PER_CU == (1 if bm * bn > 128 * 128 else 2)
        workspace = kid in WS_KIDS
        assert inst.kernel_tag == ("a16w16_4wave_ws" if workspace else "a16w16_4wave")
        assert inst.splitk_workspace_dtype == ("fp32_t" if workspace else None)
        images = a16w16_4wave_co_images(inst)
        assert len(images) == (4 if workspace else 8)
        assert [layout for layout, _, _ in images[:: 1 if workspace else 2]] == list(
            A16W16_4WAVE_LAYOUTS
        )
        for path in a16w16_4wave_co_image_paths(inst):
            assert os.path.isfile(path), path
            assert os.path.dirname(path) == os.path.join(
                os.path.dirname(CO_KERNELS_GFX950_JSON), "gfx950"
            )
    # 8 tiles x (8 + 4) images
    assert sum(len(a16w16_4wave_co_images(i)) for i in gfx950_a16w16_4wave_kernels_declared.values()) == 96


@pytest.mark.parametrize("cu_num", (128, 256))
def test_registry_namespaces_and_groups(cu_num):
    registry = get_registry("a16w16", "gfx950", cu_num)
    assert set(DIRECT_KIDS + WS_KIDS) <= set(registry)
    assert {"a16w16_4wave", "a16w16_4wave_ws"} <= OPUS_KERNEL_TAGS_BY_ARCH_FAMILY["gfx950"]["a16w16"]
    assert set(DIRECT_KIDS + WS_KIDS) <= DEFAULT_COMPILED_KIDS_BY_NAMESPACE[("gfx950", cu_num)]
    assert set(DIRECT_KIDS + WS_KIDS) <= group_kids("BIAS_AWARE_KIDS", "gfx950", cu_num=cu_num)
    assert set(WS_KIDS) <= group_kids("SPLITK_KIDS", "gfx950", cu_num=cu_num)
    assert set(DIRECT_KIDS) <= group_kids("NON_SPLITK_KIDS", "gfx950", cu_num=cu_num)
    for kid in DIRECT_KIDS + WS_KIDS:
        assert kernel_needs_external_workspace("gfx950", "a16w16", kid, cu_num=cu_num) == (
            kid in WS_KIDS
        )
        # bf16 Y only (also for the workspace kids, whose reducer writes bf16).
        assert get_kernel_instance("gfx950", "a16w16", kid, "bf16", cu_num=cu_num) is not None
        assert get_kernel_instance("gfx950", "a16w16", kid, "fp32", cu_num=cu_num) is None
    # gfx950 records never leak into the other arches.
    for gfx, tables in KERNEL_REGISTRY["a16w16"].items():
        if gfx == "gfx950":
            continue
        for table in tables.values():
            assert not any(
                inst.kernel_tag in ("a16w16_4wave", "a16w16_4wave_ws") for inst in table.values()
            )


def _plan(kid, *, K=512, split_k=0, M=200, N=136, batch=1, has_bias=False):
    return _get_cached_a16w16_launch_plan(
        "gfx950", M, N, K, batch, 256, has_bias, torch.bfloat16, torch.bfloat16, kid, split_k
    )


@pytest.mark.parametrize(
    ("K", "requested", "expected"),
    ((128, 0, 1), (128, 4, 1), (256, 4, 2), (512, 3, 3), (512, 9, 4), (4096, 16, 16)),
)
def test_ws_plan_clamps_split_k_and_sizes_workspace(K, requested, expected):
    plan = _plan(1011, K=K, split_k=requested, M=131, N=197, batch=3)  # 128x128
    assert plan.abi_split_k == expected
    assert plan.workspace_spec.shape == (expected, 3, 256, 256)
    assert plan.workspace_spec.dtype == torch.float32


def test_plan_rejections():
    assert _plan(1000).workspace_spec is None
    with pytest.raises(ValueError, match="does not support split-K"):
        _plan(1000, split_k=2)
    for kid in (1000, 1008):
        for K in (64, 160, 1000):
            with pytest.raises(ValueError, match="incompatible with shape"):
                _plan(kid, K=K)
    # Any N (the direct kernel stores the column-N tail per element).
    assert _plan(1003, N=131).workspace_spec is None   # accepted (no split-K plan)
    assert _plan(1011, N=131, split_k=1).abi_split_k == 1
    with pytest.raises(ValueError, match="does not support output dtype"):
        _get_cached_a16w16_launch_plan(
            "gfx950", 64, 64, 256, 1, 256, False, torch.bfloat16, torch.float32, 1003, 0
        )


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


@requires_gfx950
@pytest.mark.parametrize("kid", DIRECT_KIDS)
def test_direct_kid_all_layouts_bias(kid):
    gen = torch.Generator(device="cuda").manual_seed(kid)
    cases = (
        # batch, M, N, K, pad (odd M/N, padded leading dims, batch > 1)
        (1, 193, 136, 256, 8),
        (2, 64, 264, 512, 0),
        (3, 301, 72, 384, 24),
        (1, 77, 131, 256, 8),   # odd N: column-N tail stored per element, odd-length bias
        (2, 129, 7, 384, 0),
    )
    for layout, (a_t, b_t) in LAYOUTS.items():
        for batch, M, N, K, pad in cases:
            A = _operand(batch, M, K, a_t, pad, gen)
            B = _operand(batch, N, K, b_t, pad, gen)
            for bias_kind in (None, "1d", "2d"):
                bias = None
                if bias_kind == "1d":
                    bias = torch.randn((N,), generator=gen, device="cuda").to(torch.bfloat16)
                elif bias_kind == "2d":
                    bias = torch.randn((batch, N), generator=gen, device="cuda").to(torch.bfloat16)
                Y = torch.full((batch, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
                out = opus_bmm(A, B, Y, family="a16w16", kid=kid, bias=bias)
                torch.cuda.synchronize()
                assert out is Y
                golden = _golden(A, B, None if bias is None else bias.view(-1, 1, N) if bias.dim() == 2 else bias)
                _assert_close(Y, golden, f"kid {kid} {layout} b{batch} {M}x{N}x{K} pad{pad} bias={bias_kind}")


@requires_gfx950
@pytest.mark.parametrize("kid", WS_KIDS)
def test_ws_kid_all_layouts_split_k(kid):
    gen = torch.Generator(device="cuda").manual_seed(kid)
    batch, M, N, K = 2, 131, 197, 512  # 8 K tiles -> split_k up to 4
    for layout, (a_t, b_t) in LAYOUTS.items():
        pad = 16 if layout in ("nn", "tt") else 0
        A = _operand(batch, M, K, a_t, pad, gen)
        B = _operand(batch, N, K, b_t, pad, gen)
        for split_k in (1, 2, 3, 4):
            # odd N: the reducer's bias read must keep the last (odd) column
            bias = golden_bias = None
            if split_k == 2:
                bias = golden_bias = torch.randn((N,), generator=gen, device="cuda").to(torch.bfloat16)
            elif split_k == 4:
                bias = torch.randn((batch, N), generator=gen, device="cuda").to(torch.bfloat16)
                golden_bias = bias.view(batch, 1, N)
            Y = torch.full((batch, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
            opus_bmm(A, B, Y, family="a16w16", kid=kid, bias=bias, split_k=split_k)
            torch.cuda.synchronize()
            _assert_close(Y, _golden(A, B, golden_bias), f"kid {kid} {layout} split_k={split_k} bias={bias is not None}")


@requires_gfx950
@pytest.mark.parametrize(("kid", "split_k"), ((1003, 0), (1001, 1), (1011, 2), (1013, 3)))
@pytest.mark.parametrize("layout", tuple(LAYOUTS))
def test_logical_2d_gemm_views(kid, split_k, layout):
    """opus_gemm on 2D views: x_t = randn(K, M).t() etc. (batch-one launch)."""
    a_t, b_t = LAYOUTS[layout]
    gen = torch.Generator(device="cuda").manual_seed(7 + kid)
    M, N, K = 77, (520 if kid < 1008 else 515), 768
    A = _operand(1, M, K, a_t, 0, gen)[0]
    B = _operand(1, N, K, b_t, 8, gen)[0]
    bias = torch.randn((N,), generator=gen, device="cuda").to(torch.bfloat16)
    Y = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    opus_gemm(A, B, Y, family="a16w16", kid=kid, split_k=split_k, bias=bias)
    torch.cuda.synchronize()
    _assert_close(Y, _golden(A, B, bias), f"kid {kid} {layout}")


@requires_gfx950
def test_m_or_n_equal_one_and_shared_weight_batch_stride():
    gen = torch.Generator(device="cuda").manual_seed(11)
    K = 256
    # M == 1 with a transposed view: strides (.., 1, 1) are read as K-contiguous.
    A = torch.randn((2, K, 1), generator=gen, device="cuda").to(torch.bfloat16).transpose(1, 2)
    B = _operand(2, 40, K, True, 0, gen)
    Y = torch.empty((2, 1, 40), device="cuda", dtype=torch.bfloat16)
    opus_bmm(A, B, Y, family="a16w16", kid=1007)
    torch.cuda.synchronize()
    _assert_close(Y, _golden(A, B), "M=1")
    # One weight shared by every batch (batch stride 0), N-contiguous.
    W = _operand(1, 96, K, True, 0, gen).expand(3, -1, -1)
    X = _operand(3, 50, K, False, 0, gen)
    Y = torch.empty((3, 50, 96), device="cuda", dtype=torch.bfloat16)
    opus_bmm(X, W, Y, family="a16w16", kid=1014, split_k=2)
    torch.cuda.synchronize()
    _assert_close(Y, _golden(X, W), "batch stride 0")


@requires_gfx950
def test_shape_driven_api_explicit_and_tuned_kid(monkeypatch):
    """gemm_a16w16_opus: explicit kernelId, and a tuned CSV row (solidx/splitK)."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    M, N, K = 96, 200, 1024
    A = _operand(1, M, K, True, 0, gen)[0]  # M-contiguous [M, K] view
    B = _operand(1, N, K, True, 0, gen)[0]  # N-contiguous [N, K] view
    out = gemm_a16w16_opus(A, B, kernelId=1006)
    torch.cuda.synchronize()
    _assert_close(out, _golden(A, B), "explicit kernelId")

    import aiter.ops.opus.policy as policy

    monkeypatch.setattr(
        policy, "lookup_a16w16_opus_config", lambda **_: {"solidx": 1009, "splitK": 3}
    )
    bias = torch.randn((N,), generator=gen, device="cuda").to(torch.bfloat16)
    out = gemm_a16w16_opus(A, B, bias)
    torch.cuda.synchronize()
    _assert_close(out, _golden(A, B, bias), "tuned row kid 1009 splitK 3")


@requires_gfx950
def test_layout_contract_errors():
    gen = torch.Generator(device="cuda").manual_seed(5)
    M, N, K = 128, 128, 256
    A_t = _operand(1, M, K, True, 0, gen)
    B = _operand(1, N, K, False, 0, gen)
    Y = torch.empty((1, M, N), device="cuda", dtype=torch.bfloat16)
    # Other gfx950 kids keep the K-contiguous contract.
    with pytest.raises(NotImplementedError, match="K-contiguous"):
        opus_bmm(A_t, B, Y, family="a16w16", kid=92)
    # Leading dimension not a multiple of 8 elements.
    bad = torch.randn((1, K, M + 4), device="cuda").to(torch.bfloat16)[:, :, :M].transpose(1, 2)
    with pytest.raises(NotImplementedError, match="multiples of 8"):
        opus_bmm(bad, B, Y, family="a16w16", kid=1003)
    # Neither K- nor M-contiguous.
    strided = torch.randn((1, M, 2 * K), device="cuda").to(torch.bfloat16)[:, :, ::2]
    with pytest.raises(NotImplementedError, match="M-contiguous"):
        opus_bmm(strided, B, Y, family="a16w16", kid=1003)
    # Direct kids reject split-K; fp32 Y is not supported.
    with pytest.raises(ValueError, match="does not support split-K"):
        opus_bmm(A_t, B, Y, family="a16w16", kid=1003, split_k=2)
    with pytest.raises(ValueError):
        opus_bmm(A_t, B, Y.float(), family="a16w16", kid=1011, split_k=2)
    # The C++ launcher re-validates layouts on its own (bypassing Python checks).
    from aiter.ops.opus.gemm_op_a16w16 import _launch_a16w16_backend

    with pytest.raises(RuntimeError, match="multiples of 8"):
        _launch_a16w16_backend(bad, B, Y, None, None, 1003, 0)
