# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GQA coverage for the Opus GDN prefill families.

GQA here means grouped value heads: q/k carry ``Hg`` key heads while
v/g/beta/o/state carry ``H`` value heads, and ``H / Hg`` value heads share one
key head. Every case is validated against the same problem expanded back to MHA
with ``repeat_interleave``, which the Triton reference already supports.

The W/U families (WS, WF) cover gfx942/gfx950; the C families (CF, CS) are
gfx942-only and dense-only, matching their launcher.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import aiter.ops.gdn_prefill as adapter
from aiter.ops.opus_gdn_c_prefill import opus_gdn_c_prefill_fwd
from aiter.ops.opus_gdn_wu_prefill import opus_gdn_wu_prefill_fwd
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

_D = 128


def _require_opus_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Opus GDN GQA requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gfx = properties.gcnArchName.split(":", 1)[0]
    if gfx not in ("gfx942", "gfx950"):
        pytest.skip(f"Opus W/U kernels require gfx942/gfx950, got {gfx}")


def _require_c_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Opus GDN GQA requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gfx = properties.gcnArchName.split(":", 1)[0]
    if gfx != "gfx942":
        pytest.skip(f"Opus C-input kernels require gfx942, got {gfx}")


def _cu_from_lens(lens: list[int]) -> torch.Tensor:
    values = [0]
    for length in lens:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _make_inputs(
    B: int,
    T: int,
    key_heads: int,
    value_heads: int,
    sequences: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = F.normalize(
        torch.randn(B, T, key_heads, _D, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(B, T, key_heads, _D, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    v = (
        torch.randn(B, T, value_heads, _D, device="cuda", dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, value_heads, device="cuda", dtype=torch.float32))
    beta = torch.sigmoid(torch.randn_like(g)).to(torch.bfloat16)
    initial_state = (
        torch.randn(sequences, value_heads, _D, _D, device="cuda", dtype=torch.float32)
        * 0.01
    )
    return q, k, v, g, beta, initial_state


def _expanded_mha_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ratio = v.shape[2] // q.shape[2]
    return chunk_gated_delta_rule_opt_vk(
        q=q.repeat_interleave(ratio, dim=2).contiguous(),
        k=k.repeat_interleave(ratio, dim=2).contiguous(),
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize(
    ("B", "T", "key_heads", "value_heads"),
    (
        pytest.param(2, 256, 2, 4, id="ratio2"),
        pytest.param(1, 512, 2, 8, id="ratio4"),
        pytest.param(1, 128, 1, 8, id="single-key-head"),
        pytest.param(1, 100, 2, 4, id="padded-T"),
        pytest.param(2, 128, 4, 4, id="mha-baseline"),
    ),
)
@pytest.mark.parametrize(
    ("with_initial_state", "output_final_state"),
    (
        pytest.param(False, False, id="stateless"),
        pytest.param(True, True, id="state-io"),
    ),
)
def test_dense_gqa_matches_expanded_mha(
    B: int,
    T: int,
    key_heads: int,
    value_heads: int,
    with_initial_state: bool,
    output_final_state: bool,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, state = _make_inputs(
        B, T, key_heads, value_heads, B, seed=20260811 + T + value_heads
    )
    initial_state = state if with_initial_state else None

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        k2_mode=2,
        use_env_overrides=False,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=None,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    if output_final_state:
        assert actual_final is not None and expected_final is not None
        assert tuple(actual_final.shape) == (B, value_heads, _D, _D)
        torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)
    else:
        assert actual_final is None


@pytest.mark.parametrize(
    ("lens", "key_heads", "value_heads"),
    (
        pytest.param([15], 1, 4, id="single-tail"),
        pytest.param([1, 63, 64, 65, 129], 2, 4, id="boundary-mix"),
        pytest.param([64, 128, 256], 2, 8, id="aligned-packed"),
        pytest.param([15, 85, 200, 900], 4, 4, id="mha-baseline"),
    ),
)
@pytest.mark.parametrize(
    ("with_initial_state", "output_final_state"),
    (
        pytest.param(False, False, id="stateless"),
        pytest.param(True, True, id="state-io"),
    ),
)
def test_packed_gqa_matches_expanded_mha(
    lens: list[int],
    key_heads: int,
    value_heads: int,
    with_initial_state: bool,
    output_final_state: bool,
) -> None:
    _require_opus_device()
    cu_seqlens = _cu_from_lens(lens)
    q, k, v, g, beta, state = _make_inputs(
        1,
        sum(lens),
        key_heads,
        value_heads,
        len(lens),
        seed=20260811 + sum(lens) + value_heads,
    )
    initial_state = state if with_initial_state else None

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_env_overrides=False,
        cu_seqlens=cu_seqlens,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    if output_final_state:
        assert actual_final is not None and expected_final is not None
        assert tuple(actual_final.shape) == (len(lens), value_heads, _D, _D)
        torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize(
    ("lens", "key_heads", "value_heads"),
    (
        pytest.param([1, 63, 64, 65, 129], 2, 4, id="boundary-mix"),
        pytest.param([64, 128, 256], 2, 8, id="aligned-packed"),
        pytest.param([15, 85, 200], 4, 4, id="mha-baseline"),
    ),
)
def test_packed_gqa_fused_wu_matches_expanded_mha(
    lens: list[int],
    key_heads: int,
    value_heads: int,
) -> None:
    _require_opus_device()
    cu_seqlens = _cu_from_lens(lens)
    q, k, v, g, beta, state = _make_inputs(
        1,
        sum(lens),
        key_heads,
        value_heads,
        len(lens),
        seed=20260812 + sum(lens) + value_heads,
    )

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        k2_mode=1,
        use_env_overrides=False,
        cu_seqlens=cu_seqlens,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert tuple(actual_final.shape) == (len(lens), value_heads, _D, _D)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("k2_mode", (0, 1, 2), ids=("auto", "wf", "ws"))
@pytest.mark.parametrize(
    ("B", "T", "key_heads", "value_heads"),
    (
        pytest.param(2, 256, 2, 4, id="ratio2"),
        pytest.param(1, 512, 2, 8, id="ratio4"),
        pytest.param(1, 128, 1, 8, id="single-key-head"),
        pytest.param(2, 128, 4, 4, id="mha-baseline"),
    ),
)
def test_dense_gqa_every_k2_mode(
    k2_mode: int, B: int, T: int, key_heads: int, value_heads: int
) -> None:
    """WS, WF and the auto envelope must all address q/k per key head."""
    _require_opus_device()
    q, k, v, g, beta, state = _make_inputs(
        B, T, key_heads, value_heads, B, seed=101 + T + value_heads
    )

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        k2_mode=k2_mode,
        use_env_overrides=False,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=None,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("wf_variant", (0, 1, 2, 3, 4, 5, 6), ids=lambda v: f"wf{v}")
def test_gqa_fused_wu_variants(
    wf_variant: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fused W/U specialization re-reads q/k, so all of them need Hg.

    Production picks the variant from the serial-work envelope; the benchmark
    override is the only way to reach the others.
    """
    _require_opus_device()
    monkeypatch.setenv("OPUS_GDN_WF_VARIANT", str(wf_variant))
    q, k, v, g, beta, state = _make_inputs(1, 256, 2, 8, 1, seed=131 + wf_variant)

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        num_warps=8,
        k2_mode=1,
        use_env_overrides=True,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=None,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("c_mode", (0, 1, 2), ids=("auto", "cf", "cs"))
@pytest.mark.parametrize(
    ("B", "T", "key_heads", "value_heads"),
    (
        pytest.param(2, 256, 2, 4, id="ratio2"),
        pytest.param(1, 512, 2, 8, id="ratio4"),
        pytest.param(1, 128, 1, 8, id="single-key-head"),
        pytest.param(1, 100, 2, 4, id="padded-T"),
        pytest.param(2, 128, 4, 4, id="mha-baseline"),
    ),
)
def test_dense_gqa_c_input_paths(
    c_mode: int, B: int, T: int, key_heads: int, value_heads: int
) -> None:
    """CF and CS reach q/k through K1-C, the C scan and the shared K6."""
    _require_c_device()
    q, k, v, g, beta, state = _make_inputs(
        B, T, key_heads, value_heads, B, seed=149 + T + value_heads
    )

    actual, actual_final = opus_gdn_c_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        c_mode=c_mode,
        use_env_overrides=False,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=None,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("scan_bv", (16, 32, 64), ids=lambda v: f"scan{v}")
@pytest.mark.parametrize("out_bv", (64, 128), ids=lambda v: f"outbv{v}")
def test_gqa_c_split_scan_and_output_variants(
    scan_bv: int, out_bv: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every CS scan V tile and its shared K6 V tile must carry the key-head stride."""
    _require_c_device()
    monkeypatch.setenv("OPUS_GDN_K2C_SCAN_BV", str(scan_bv))
    monkeypatch.setenv("OPUS_GDN_K2C_OUT_BV", str(out_bv))
    q, k, v, g, beta, state = _make_inputs(1, 256, 2, 8, 1, seed=151 + scan_bv + out_bv)

    actual, actual_final = opus_gdn_c_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        c_mode=2,
        use_env_overrides=True,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=None,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("ref_bv", (16, 32, 64), ids=lambda v: f"ref{v}")
@pytest.mark.parametrize("out_bv", (32, 64, 128), ids=lambda v: f"outbv{v}")
@pytest.mark.parametrize("packed", (False, True), ids=("dense", "packed"))
def test_gqa_scan_and_output_variants(
    ref_bv: int,
    out_bv: int,
    packed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every state-scan V tile and K6 V tile must carry the key-head stride.

    Production picks these from the shape envelope, so the benchmark overrides
    are the only way to reach the tiles a given test shape would skip.
    """
    _require_opus_device()
    monkeypatch.setenv("OPUS_GDN_REF", str(ref_bv))
    monkeypatch.setenv("OPUS_GDN_OUT_BV", str(out_bv))
    lens = [64, 128, 65]
    cu_seqlens = _cu_from_lens(lens) if packed else None
    total = sum(lens) if packed else 256
    q, k, v, g, beta, state = _make_inputs(
        1, total, 2, 8, len(lens) if packed else 1, seed=20260811 + out_bv + ref_bv
    )

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        k2_mode=2,
        use_env_overrides=True,
        cu_seqlens=cu_seqlens,
    )
    expected, expected_final = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize(
    ("BT", "k1_algo"),
    (
        pytest.param(16, 1, id="BT16"),
        pytest.param(64, 0, id="k1-basic"),
    ),
)
def test_gqa_rejects_non_neumann_k1(BT: int, k1_algo: int) -> None:
    _require_opus_device()
    q, k, v, g, beta, _ = _make_inputs(1, 128, 2, 4, 1, seed=107)

    with pytest.raises(ValueError, match="GQA requires BT=64 and k1_algo=1"):
        opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            BT=BT,
            k1_algo=k1_algo,
            k2_mode=2,
            use_env_overrides=False,
        )


def test_rejects_indivisible_head_counts() -> None:
    _require_opus_device()
    q, k, v, g, beta, _ = _make_inputs(1, 128, 4, 4, 1, seed=109)
    v = v[:, :, :3].contiguous()
    g = g[:, :, :3].contiguous()
    beta = beta[:, :, :3].contiguous()

    with pytest.raises(ValueError, match="must be a positive multiple"):
        opus_gdn_wu_prefill_fwd(q, k, v, g, beta, k2_mode=2, use_env_overrides=False)


@pytest.mark.parametrize("path", ("auto", "ws", "wu"))
def test_adapter_routes_gqa_to_ws(path: str) -> None:
    _require_opus_device()
    q, k, v, g, beta, _ = _make_inputs(1, 256, 2, 8, 1, seed=113)

    assert adapter.select_gdn_prefill_path(q, k, v, g=g, beta=beta, path=path) == "ws"

    actual, _ = adapter.gdn_prefill(q, k, v, g=g, beta=beta, path=path)
    expected, _ = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("path", ("wf", "cf", "cs", "c"))
def test_adapter_serves_gqa_on_explicit_paths(path: str) -> None:
    """Auto keeps GQA on the measured WS family; an explicit family is honored."""
    if path in ("c", "cf", "cs"):
        _require_c_device()
    else:
        _require_opus_device()
    q, k, v, g, beta, _ = _make_inputs(1, 256, 2, 8, 1, seed=127)

    selected = adapter.select_gdn_prefill_path(q, k, v, g=g, beta=beta, path=path)
    assert selected in ("wf", "cf", "cs")

    actual, _ = adapter.gdn_prefill(q, k, v, g=g, beta=beta, path=path)
    expected, _ = _expanded_mha_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_adapter_rejects_gqa_on_triton() -> None:
    _require_opus_device()
    q, k, v, g, beta, _ = _make_inputs(1, 128, 2, 4, 1, seed=139)

    with pytest.raises(ValueError, match="does not support"):
        adapter.select_gdn_prefill_path(q, k, v, g=g, beta=beta, path="triton")
