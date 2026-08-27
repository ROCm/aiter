# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests and perf sweep for the FlyDSL fused GDN K5+K6 forward.

Verifies ``chunk_gated_delta_rule_fwd_h_o_flydsl`` (inter-chunk state scan fused
with the inter/intra-chunk output) against a known-correct reference:

    reference o  =  Triton K6( q, k, v_new_ref, h_ref, g )

where ``h_ref`` / ``v_new_ref`` come from the pure-PyTorch K5 reference
(``ref_chunk_gated_delta_rule_fwd_h``). This decouples the fused output check
from the FlyDSL K5 kernel itself (whose correctness is covered by
``test_flydsl_linear_attention_prefill.py``).

Correctness levels:
  * ``test_fused_unit``     — call the fused wrapper directly.
  * ``test_fused_pipeline`` — call the end-to-end pipeline with
    ``fusion=K5K6Fusion.ALWAYS`` and compare to the NEVER/Triton baseline.

Invocation (mirrors ``test_flydsl_linear_attention_prefill.py``): running the
file as a script drives pytest, so single cases select the usual way::

    python op_tests/test_flydsl_gdn_fused_k5k6.py                 # all correctness tests
    python op_tests/test_flydsl_gdn_fused_k5k6.py -k test_fused_unit
    pytest op_tests/test_flydsl_gdn_fused_k5k6.py::test_fused_final_state

The perf sweep is opt-in behind ``--bench`` and never runs under pytest::

    python op_tests/test_flydsl_gdn_fused_k5k6.py --bench               # fused sweep
    python op_tests/test_flydsl_gdn_fused_k5k6.py --bench --kernel k5   # state scan
    python op_tests/test_flydsl_gdn_fused_k5k6.py --bench \
        --kernel k5 --k5-variants all                          # per-variant K5
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is importable so ``from op_tests....`` resolves under a
# bare ``python op_tests/test_flydsl_gdn_fused_k5k6.py`` invocation (CI runs the
# file this way). Under pytest the rootdir is already on sys.path, so this is a
# no-op there. This file imports a sibling test module (unlike the other
# op_tests, which are self-contained), hence the bootstrap.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL fused K5+K6 tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.gdn_fused_gfx942_kernels import (
        chunk_gated_delta_rule_fwd_h_o_flydsl,
    )
    from op_tests.gdn_common import (
        _RMSE_TOL,
        _make_inputs,
        _pipeline_inputs,
        _reference_o,
        _rmse_ratio,
        ref_chunk_gated_delta_rule_fwd_h,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Unable to import FlyDSL fused K5+K6 dependencies: {exc}",
        allow_module_level=True,
    )

# NOTE: this is process-wide and leaks into any test module that runs after this
# one in the same session -- tensors created without an explicit device land on
# the GPU. Keep new tests device-explicit rather than relying on it.
torch.set_default_device("cuda")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unit test: fused wrapper vs reference
# --------------------------------------------------------------------------- #
# BV in {16, 32, 64} plus the wave-widened bv64w8 (NR_SPLIT=2, splits b_A across
# the V-split waves) are all supported. ``None`` exercises the auto heuristic.
@pytest.mark.parametrize("gate", ["g", "gk"])
@pytest.mark.parametrize(
    "H,Hg",
    [(12, 12), (24, 24), (4, 2)],  # MHA (KDA), MHA (KDA TP4), GQA (GDN)
)
@pytest.mark.parametrize("seq_lens", [[512], [512, 512], [640, 384, 512]])
@pytest.mark.parametrize("variant", ["bv16", "bv32", "bv64", "bv64w8", None])
def test_fused_unit(gate, H, Hg, seq_lens, variant):
    """Fused wrapper output matches pure-PyTorch K5 ref + Triton K6."""
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    K = V = 128
    T_flat = sum(seq_lens)
    scale = K**-0.5
    use_exp2 = False  # inputs are natural-log gates (see K5 bench rationale)

    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)

    o_ref = _reference_o(inp, scale=scale, use_exp2=use_exp2)

    o_fused, _ = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=inp["u_hm"],
        g=inp["g"],
        gk=inp["gk"],
        scale=scale,
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
        variant=variant,
    )

    ratio = _rmse_ratio(o_fused, o_ref)
    assert ratio < _RMSE_TOL, (
        f"fused o mismatch: rmse_ratio={ratio:.3e} "
        f"(gate={gate} H={H} Hg={Hg} seqs={seq_lens} variant={variant})"
    )


@pytest.mark.parametrize("gate", ["gk", "g"])
@pytest.mark.parametrize("B", [2, 3])
def test_fused_dense_multibatch(gate, B):
    """Dense B>1 matches the reference for both gate kinds.

    Gate addressing: the kernel folds the batch into
    the token index (``bos = i_n * T``), so any bound or offset that forgets
    the batch term shows up here and nowhere else.

    ``gk_decay`` is deliberately gentle: the default saturates exp(gk_last) to
    zero, which would make this insensitive to a gate that is merely wrong
    rather than absent.
    """
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    H = Hg = 8
    K = V = 128
    seq_lens = [512]
    T_flat = sum(seq_lens)
    scale = K**-0.5
    use_exp2 = False

    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate, B=B, gk_decay=-0.002)
    assert inp["cu"] is None, "dense multi-batch test must not be varlen"

    o_ref = _reference_o(inp, scale=scale, use_exp2=use_exp2)
    o_fused, _ = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=inp["u_hm"],
        g=inp["g"],
        gk=inp["gk"],
        scale=scale,
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
    )

    # Assert per batch: a whole-tensor RMSE is dominated by batch 0, which is
    # correct even when the batch term is broken, and would mask the bug.
    for b in range(B):
        ratio = _rmse_ratio(o_fused[b], o_ref[b])
        assert ratio < _RMSE_TOL, (
            f"fused dense multibatch mismatch in batch {b}/{B}: "
            f"rmse_ratio={ratio:.3e} (gate={gate})"
        )


@pytest.mark.parametrize("poison", [float("inf"), float("-inf"), float("nan")])
def test_fused_tail_mask_suppresses_nonfinite(poison):
    """A non-finite value in the tail-mask shadow does not leak into a neighbour.

    ``seq_lens=[100, 128]``: sequence 0's last chunk covers tokens 64..127, of
    which only 64..99 are its own -- tokens 100..127 are sequence 1's live
    data, read and then masked away. Planting Inf/NaN there is the case a
    multiplicative mask cannot handle, since ``Inf * 0`` and ``NaN * 0`` are
    both NaN. Sequence 0's output and final state must stay finite.

    Sequence 1 legitimately consumes the poison, so only sequence 0 is checked.
    """
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    H = Hg = 8
    K = V = 128
    seq_lens = [100, 128]
    T_flat = sum(seq_lens)
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, "g")

    # u is head-major [B, H, T_flat, V]; poison the tokens that fall inside
    # sequence 0's tail-chunk shadow but belong to sequence 1.
    u_hm = inp["u_hm"].clone()
    u_hm[:, :, seq_lens[0] : 128, :] = poison

    o, fs = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=u_hm,
        g=inp["g"],
        scale=K**-0.5,
        initial_state=inp["h0"],
        output_final_state=True,
        cu_seqlens=inp["cu"],
        use_exp2=False,
    )

    o_seq0 = o[0, : seq_lens[0]]
    assert torch.isfinite(o_seq0).all(), (
        f"non-finite leaked into sequence 0's output via the tail mask "
        f"(poison={poison}): "
        f"{(~torch.isfinite(o_seq0)).sum().item()} / {o_seq0.numel()} elements"
    )
    assert torch.isfinite(fs[0]).all(), (
        f"non-finite leaked into sequence 0's final state via the tail mask "
        f"(poison={poison})"
    )


def test_fused_gate_rank_spellings_interop():
    """Both advertised ``g`` spellings work back-to-back in one process.

    The compile caches key on shape parameters only, but the compiled kernel's
    C ABI is packed per tensor rank. Calling with ``[B, H, T]`` and then
    ``[H, T]`` for the same shape used to hit the same cache entry with a
    mismatched ABI and fail in argument packing.
    """
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    H = Hg = 8
    K = V = 128
    T_flat = 256
    inp = _make_inputs(H, Hg, K, V, T_flat, [T_flat], "g", B=1)
    g2d = inp["g"].reshape(H, T_flat).contiguous()
    g3d = g2d.reshape(1, H, T_flat)

    def run(g):
        o, _ = chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=inp["q"],
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=g,
            scale=K**-0.5,
            initial_state=inp["h0"],
            output_final_state=False,
            cu_seqlens=None,
            use_exp2=False,
        )
        return o

    # Order matters: 3-D first populates the cache, 2-D second is the call that
    # used to fail.
    o3 = run(g3d)
    o2 = run(g2d)
    torch.testing.assert_close(o2, o3, rtol=0, atol=0)


def test_fused_rejects_undersized_gk():
    """A ``gk`` sized for one batch is rejected, not silently mis-read.

    Without the host-side check this shape reaches the kernel, which addresses
    ``gk`` as a flat token-major buffer and would read past the tensor for
    every batch after the first.
    """
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    H = Hg = 8
    K = V = 128
    T_flat = 512
    B = 2
    inp = _make_inputs(H, Hg, K, V, T_flat, [T_flat], "gk", B=B)

    with pytest.raises(ValueError, match="gk must be"):
        chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=inp["q"],
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=None,
            gk=inp["gk"][0].contiguous(),  # only batch 0's worth
            scale=K**-0.5,
            initial_state=inp["h0"],
            output_final_state=False,
            cu_seqlens=None,
            use_exp2=False,
        )


def test_fused_final_state():
    """output_final_state=True returns a final state matching the K5 ref."""
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    H = Hg = 12
    K = V = 128
    seq_lens = [512, 512]
    T_flat = sum(seq_lens)
    scale = K**-0.5
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, "g")

    (
        _,
        fs_ref,
    ) = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"],
        w=inp["w_tm"],
        u=inp["u_tm"],
        g=inp["g_ref"],
        gk=inp["gk"],
        initial_state=inp["h0"],
        output_final_state=True,
        cu_seqlens=inp["cu"],
        g_head_major=True,
    )[0::2]

    _, fs_fused = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=inp["u_hm"],
        g=inp["g"],
        scale=scale,
        initial_state=inp["h0"],
        output_final_state=True,
        cu_seqlens=inp["cu"],
        use_exp2=False,
    )
    assert fs_fused is not None
    ratio = _rmse_ratio(fs_fused, fs_ref)
    assert ratio < _RMSE_TOL, f"final_state mismatch: rmse_ratio={ratio:.3e}"


@pytest.mark.parametrize(
    "seq_lens",
    [
        [613],  # dense, not a multiple of BT=64 → exercises tail masking in K6
        [512, 613],  # varlen with a ragged tail sequence → exercises causal-column mask
    ],
)
def test_fused_nonaligned(seq_lens):
    """Fused output matches reference when sequence length is not a multiple of BT=64."""
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    H = Hg = 8
    K = V = 128
    T_flat = sum(seq_lens)
    scale = K**-0.5
    use_exp2 = False

    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, "g")
    o_ref = _reference_o(inp, scale=scale, use_exp2=use_exp2)
    o_fused, _ = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=inp["u_hm"],
        g=inp["g"],
        scale=scale,
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
    )
    ratio = _rmse_ratio(o_fused, o_ref)
    assert (
        ratio < _RMSE_TOL
    ), f"fused nonaligned mismatch: rmse_ratio={ratio:.3e} (seq_lens={seq_lens})"


# --------------------------------------------------------------------------- #
# Pipeline tests: the K5K6Fusion API on chunk_gated_delta_rule_fwd_opt_vk.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,Hg", [(4, 2), (8, 4)])
@pytest.mark.parametrize("seq_lens", [[512], [512, 512]])
def test_fused_pipeline(H, Hg, seq_lens):
    """fusion=ALWAYS (FlyDSL backend) matches the pure-Triton baseline."""
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    from aiter.ops.gated_delta_rule_fusion import K5K6Fusion
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )

    common = _pipeline_inputs(H, Hg, seq_lens)
    # Baseline: pure Triton (no FlyDSL backend -> fusion ignored regardless).
    _, o_base, fs_base = chunk_gated_delta_rule_fwd_opt_vk(**common)
    # Force the fused FlyDSL kernel. NOTE: fusion is gated behind
    # use_chunk_flydsl, so both flags are required.
    _, o_fused, fs_fused = chunk_gated_delta_rule_fwd_opt_vk(
        use_chunk_flydsl=True, fusion=K5K6Fusion.ALWAYS, **common
    )

    ratio_o = _rmse_ratio(o_fused, o_base)
    assert ratio_o < _RMSE_TOL, f"pipeline o mismatch: rmse_ratio={ratio_o:.3e}"
    ratio_fs = _rmse_ratio(fs_fused, fs_base)
    assert ratio_fs < _RMSE_TOL, f"pipeline final_state mismatch: {ratio_fs:.3e}"


@pytest.mark.parametrize(
    "kwargs,reason_fragment",
    [
        ({"use_chunk_flydsl": False}, "FlyDSL path is disabled"),
        (
            {"use_chunk_flydsl": True, "inplace_final_state": True},
            "in-place write-back",
        ),
        (
            {"use_chunk_flydsl": True, "snapshot_dtype": torch.float32},
            "snapshot_dtype",
        ),
    ],
)
def test_fused_always_raises_when_unsupported(kwargs, reason_fragment):
    """``ALWAYS`` refuses loudly instead of silently running the separate path."""
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    from aiter.ops.gated_delta_rule_fusion import K5K6Fusion
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )

    common = _pipeline_inputs(8, 4, [512])
    with pytest.raises(ValueError, match="fusion=ALWAYS") as exc:
        chunk_gated_delta_rule_fwd_opt_vk(fusion=K5K6Fusion.ALWAYS, **kwargs, **common)
    assert reason_fragment in str(
        exc.value
    ), f"the error should name the actual reason; got: {exc.value}"


def test_fused_always_overrides_the_perf_heuristic():
    """``ALWAYS`` still fuses a shape AUTO would route to the separate path."""
    if get_gfx() != "gfx942":
        pytest.skip(f"fused K5+K6 kernel is gfx942-only; arch={get_gfx()}")
    from aiter.ops.flydsl.gdn_fused_gfx942_kernels import (
        fused_k5k6_gfx942_supported,
        should_use_fused_k5k6_gfx942,
    )
    from aiter.ops.gated_delta_rule_fusion import K5K6Fusion
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )

    H, Hg = 4, 2
    assert fused_k5k6_gfx942_supported()
    # A tiny grid: one sequence, few heads -> fill well under _FUSED_MIN_FILL.
    assert not should_use_fused_k5k6_gfx942(
        H=H, N=1, V=128
    ), "shape chosen for this test is no longer AUTO-rejected; pick a smaller one"

    common = _pipeline_inputs(H, Hg, [512])
    _, o_base, fs_base = chunk_gated_delta_rule_fwd_opt_vk(**common)
    # Must not raise, and must agree with the baseline.
    _, o_fused, fs_fused = chunk_gated_delta_rule_fwd_opt_vk(
        use_chunk_flydsl=True, fusion=K5K6Fusion.ALWAYS, **common
    )
    assert _rmse_ratio(o_fused, o_base) < _RMSE_TOL
    assert _rmse_ratio(fs_fused, fs_base) < _RMSE_TOL


# --------------------------------------------------------------------------- #
# Fused selection rules (both closed-form): H*N variant rule + fill>=0.45 routing
# --------------------------------------------------------------------------- #
def test_fused_variant_hn_rule():
    """``select_fused_variant`` is the H*N tile rule: <=32 bv16, <=80 bv32, else
    bv64w8; None only if BV illegal for V."""
    from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import (
        select_fused_variant,
    )

    V = 128  # all BVs legal
    # (H, N, expected tag) spanning both sides of the 32 and 80 boundaries.
    cases = [
        (4, 1, "bv16"),
        (32, 1, "bv16"),
        (8, 4, "bv16"),  # H*N <= 32
        (8, 8, "bv32"),
        (16, 4, "bv32"),
        (12, 4, "bv32"),  # 32 < H*N <= 80
        (12, 6, "bv32"),
        (16, 5, "bv32"),  # H*N = 72 / 80: bv32, not bv64w8
        (16, 8, "bv64w8"),
        (96, 1, "bv64w8"),
        (24, 4, "bv64w8"),  # H*N > 80
    ]
    for H, N, exp in cases:
        got = select_fused_variant(H=H, N=N, V=V)
        assert got == exp, f"H={H} N={N} H*N={H * N}: got {got}, want {exp}"
    # Legality fallback: bv64w8 needs a legal BV=64; V=16 makes it illegal.
    assert select_fused_variant(H=96, N=8, V=16) is None


def test_fused_selection_heuristic():
    """``should_use_fused_k5k6_gfx942`` fuses iff ceil(V/BV_run)*N*H/CU >= 0.45, with
    BV_run from the same H*N variant rule the launcher uses."""
    import math

    from aiter.ops.flydsl.gdn_fused_gfx942_kernels import (
        _FUSED_MIN_FILL,
        should_use_fused_k5k6_gfx942,
    )
    from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import (
        select_fused_variant,
    )
    from aiter.ops.flydsl.kernels.k5_variants import _bv_of_variant
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        _ARCH,
        _device_cu_count,
    )

    if _ARCH != "gfx942":
        pytest.skip(f"fusion heuristic is gfx942-only; arch={_ARCH}")

    # Pin the calibrated threshold: a silent change here would reroute shapes.
    assert _FUSED_MIN_FILL == 0.45

    cu = _device_cu_count()
    V = 128
    for H, N in [
        (24, 8),
        (12, 8),
        (24, 4),
        (12, 4),
        (4, 8),
        (96, 1),
        (12, 1),
        (8, 1),
    ]:
        tag = select_fused_variant(H=H, N=N, V=V)
        bv = _bv_of_variant(tag) if tag else 64
        fill = math.ceil(V / bv) * N * H / cu
        expect = fill >= _FUSED_MIN_FILL
        got = should_use_fused_k5k6_gfx942(H=H, N=N, V=V)
        assert (
            got == expect
        ), f"H={H} N={N} bv={bv} fill={fill:.2f}: got {got}, want {expect}"


@pytest.mark.parametrize(
    "H,Hg,seq_lens,expect_fused",
    [
        (24, 24, [512] * 8, True),  # bv64w8 fill=2.53 -> fused
        (8, 4, [512] * 1, False),  # bv16 N=1 fill=0.21 -> separate
    ],
)
def test_fused_auto_routing(H, Hg, seq_lens, expect_fused):
    """fusion=AUTO (with use_chunk_flydsl) routes by the heuristic, stays correct.

    High-fill routes to the fused kernel (matches the pure-Triton baseline within
    tolerance). Low-fill routes to the separate FlyDSL-K5 + Triton-K6 path, which
    differs from pure Triton only in the K5 backend -- still within tolerance.
    Both must match; the routing itself is asserted via should_use_fused_k5k6_gfx942.
    """
    from aiter.ops.flydsl.gdn_fused_gfx942_kernels import should_use_fused_k5k6_gfx942
    from aiter.ops.flydsl.linear_attention_prefill_kernels import _ARCH
    from aiter.ops.gated_delta_rule_fusion import K5K6Fusion
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )

    N = len(seq_lens)
    if _ARCH == "gfx942":
        assert (
            should_use_fused_k5k6_gfx942(H=H, N=N, V=128) is expect_fused
        ), "heuristic prediction changed; update the test expectation"

    common = _pipeline_inputs(H, Hg, seq_lens)
    _, o_base, _ = chunk_gated_delta_rule_fwd_opt_vk(**common)  # pure Triton
    _, o_auto, _ = chunk_gated_delta_rule_fwd_opt_vk(
        use_chunk_flydsl=True, fusion=K5K6Fusion.AUTO, **common
    )
    ratio = _rmse_ratio(o_auto, o_base)
    assert ratio < _RMSE_TOL, f"auto-routed o mismatch: rmse={ratio:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
