# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the routed ``a2a_gemm_fused`` v1 contract and native reference.

The provider interface is exercised without a built AITER (``aiter.plugin`` imports
only the standard library; the op module itself needs torch). The native reference runs
on ordinary torch CPU tensors -- fp8 is decoded before the matmul -- and simulates the
SP4 exchange in-process, because it is a non-exchanging single-rank stand-in. The
independent oracle below decodes the logical weight from the bytes the fixture built,
so it does not reuse the op's own ``view/permute`` preshuffle math.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

# the public package must expose the lazy op
import aiter
import aiter.ops.gemm_fusion.a2a_gemm.op as op_module
from aiter import plugin
from aiter.ops.gemm_fusion.a2a_gemm.op import (
    FINGERPRINT,
    SIGNATURE,
    a2a_gemm_fused,
)

_WORLD = 4
_K = 8192
_N = 6144
_SHARD = 2048
_M_MAX = 16384
_FROZEN_FINGERPRINT = "388d9816496c260e"
_FROZEN_SIGNATURE = (
    "x:positional, x_scale:positional, w:positional, w_scale:positional, "
    "out:positional, total_tokens:keyword-only, bucket_m:keyword-only=None, "
    "registered:keyword-only=False, group:keyword-only=None, "
    "stream:keyword-only=None, -> None"
)
_BASELINE = Path(__file__).resolve().parents[1] / "tests" / "plugin_contracts.json"


# ----------------------------------------------------------------- helpers


def _preshuffle(logical):
    """AITER ``shuffle_weight(layout=(16,16))`` forward layout, as tested upstream."""

    rows, k = logical.shape
    return (
        logical.view(rows // 16, 16, k // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(rows, k)
    )


def _render_signature(function):
    """Render an inspect.Signature in the baseline's exact spelling."""

    kinds = {
        inspect.Parameter.POSITIONAL_ONLY: "positional-only",
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional",
        inspect.Parameter.KEYWORD_ONLY: "keyword-only",
    }
    parts = []
    for parameter in inspect.signature(function).parameters.values():
        part = f"{parameter.name}:{kinds[parameter.kind]}"
        if parameter.default is not inspect.Parameter.empty:
            part += f"={parameter.default!r}"
        parts.append(part)
    annotation = inspect.signature(function).return_annotation
    if annotation is not inspect.Signature.empty:
        parts.append(f"-> {annotation}")
    return ", ".join(parts)


def _seeded_fp8(shape, seed, *, gain=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * gain).to(torch.float8_e4m3fn)


@pytest.fixture(scope="module")
def weight():
    """A full [6144, 8192] preshuffled fp8 weight + the logical weight it encodes."""

    generator = torch.Generator().manual_seed(20261113)
    logical = (torch.randn((_N, _K), generator=generator) * 0.1).to(torch.float8_e4m3fn)
    w_scale = (torch.rand((_N,), generator=generator) * 0.5 + 0.25).to(torch.float32)
    return _preshuffle(logical), logical, w_scale


def _valid_case(weight, m_real=2, bucket_m=None, seed=11):
    """A well-formed call: (args list, kwargs dict, out buffer)."""

    w_phys, _logical, w_scale = weight
    generator = torch.Generator().manual_seed(seed)
    x = _seeded_fp8((_WORLD * m_real, _SHARD), seed, gain=1.0)
    x_scale = (torch.rand((m_real,), generator=generator) * 0.5 + 0.75).to(
        torch.float32
    )
    rows = m_real if bucket_m is None else bucket_m
    out = torch.zeros((rows, _N), dtype=torch.bfloat16)
    args = [x, x_scale, w_phys, w_scale, out]
    kwargs = {"total_tokens": _WORLD * m_real, "bucket_m": bucket_m}
    return args, kwargs, out


def _assemble_reference(peers, destination, m_real):
    """Independent gather: source s block d, concatenated along K in source order."""

    lo = destination * m_real
    blocks = [peers[s][lo : lo + m_real].float().double() for s in range(_WORLD)]
    return torch.cat(blocks, dim=1)


def _math_reference(activation, logical_w, x_scale, w_scale):
    """Independent FP64 decoded-FP8 GEMM with row/channel scales, one BF16 rounding."""

    acc = activation @ logical_w.float().double().t()
    scaled = acc * x_scale.float().double().reshape(-1, 1)
    scaled = scaled * w_scale.float().double().reshape(1, -1)
    return scaled.to(torch.bfloat16)


class _FakeGroup:
    def __init__(self, rank, size=_WORLD):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture(autouse=True)
def _clean_plugin(monkeypatch):
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    plugin._reset_for_testing()
    yield
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    plugin._reset_for_testing()


# ------------------------------------------------------------- contract id


def test_contract_fingerprint_is_the_frozen_value():
    assert SIGNATURE == _FROZEN_SIGNATURE
    assert FINGERPRINT == _FROZEN_FINGERPRINT
    assert (
        hashlib.sha256(SIGNATURE.encode("utf-8")).hexdigest()[:16]
        == _FROZEN_FINGERPRINT
    )
    assert _render_signature(a2a_gemm_fused) == _FROZEN_SIGNATURE, (
        "the routed signature drifted from the frozen contract; a provider "
        "implements against exactly this string"
    )


def test_committed_baseline_records_the_same_contract():
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert "a2a_gemm_fused" in baseline
    assert baseline["a2a_gemm_fused"] == {
        "fingerprint": _FROZEN_FINGERPRINT,
        "signature": _FROZEN_SIGNATURE,
        "source": "aiter/ops/gemm_fusion/a2a_gemm/op.py",
        "supersedes": [],
        "version": 1,
    }


def test_lazy_export_resolves_to_the_routed_op():
    from aiter.ops.gemm_fusion import a2a_gemm_fused as exported

    assert aiter.a2a_gemm_fused is a2a_gemm_fused
    assert exported is a2a_gemm_fused
    assert a2a_gemm_fused.__aiter_plugin_op__ == plugin.Op("a2a_gemm_fused", 1)
    routed = plugin.status()["routed_ops"]
    assert routed["a2a_gemm_fused"]["version"] == 1
    assert "sp_qkv_fused" in routed, "the existing export must stay routed"


# --------------------------------------------------------- native semantics


@pytest.mark.parametrize("m_real", [1, 3])
def test_native_matches_independent_fp64_reference(weight, m_real):
    args, kwargs, out = _valid_case(weight, m_real=m_real)
    assert a2a_gemm_fused(*args, **kwargs) is None

    _, logical, w_scale = weight
    x, x_scale = args[0], args[1]
    activation = _assemble_reference((x, x, x, x), 0, m_real)
    expected = _math_reference(activation, logical, x_scale, w_scale)

    assert out.dtype == torch.bfloat16 and out.shape == (m_real, _N)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_native_models_the_group_rank_destination(weight):
    m_real = 2
    w_phys, logical, w_scale = weight
    x = torch.zeros((_WORLD * m_real, _SHARD), dtype=torch.float8_e4m3fn)
    for destination in range(_WORLD):
        x[destination * m_real : (destination + 1) * m_real].fill_(
            (destination + 1) / 4.0
        )
    x_scale = torch.ones(m_real, dtype=torch.float32)

    out = torch.zeros((m_real, _N), dtype=torch.bfloat16)
    a2a_gemm_fused(
        x,
        x_scale,
        w_phys,
        w_scale,
        out,
        total_tokens=_WORLD * m_real,
        group=_FakeGroup(rank=2),
    )

    activation = _assemble_reference((x, x, x, x), 2, m_real)
    expected = _math_reference(activation, logical, x_scale, w_scale)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    other = _math_reference(
        _assemble_reference((x, x, x, x), 0, m_real), logical, x_scale, w_scale
    )
    assert not torch.equal(out, other), "rank 2 must not compute destination 0"


def test_assemble_from_peers_gathers_each_sources_destination_block():
    m_real = 2
    # Values stay in fp8 e4m3fn's exact range for their magnitude (step 0.5..1).
    peers = []
    for source in range(_WORLD):
        peer = torch.zeros((_WORLD * m_real, _SHARD), dtype=torch.float8_e4m3fn)
        for destination in range(_WORLD):
            peer[destination * m_real : (destination + 1) * m_real].fill_(
                4.0 * source + destination + 1
            )
        peers.append(peer)

    assembled = op_module._assemble_from_peers(tuple(peers), 2, m_real, torch)

    assert assembled.shape == (m_real, _K)
    for source in range(_WORLD):
        quarter = assembled[:, source * _SHARD : (source + 1) * _SHARD]
        assert torch.equal(
            quarter.float(), torch.full((m_real, _SHARD), 4.0 * source + 3)
        ), f"source {source}'s destination-2 block is in the wrong K quarter"


def test_preshuffle_round_trip_is_exact(weight):
    w_phys, logical, _w_scale = weight
    recovered = op_module._preshuffle_to_linear(w_phys, torch)
    assert torch.equal(recovered, logical.float())

    generator = torch.Generator().manual_seed(7)
    small = (torch.randn((32, 64), generator=generator)).to(torch.float8_e4m3fn)
    assert torch.equal(
        op_module._preshuffle_to_linear(_preshuffle(small), torch), small.float()
    )

    with pytest.raises(ValueError, match="16x32-block aligned"):
        op_module._preshuffle_to_linear(
            torch.zeros((32, 48), dtype=torch.float8_e4m3fn), torch
        )


# ------------------------------------------------------------------- Pad M


def test_pad_m_writes_only_the_real_rows(weight):
    m_real, bucket_m = 2, 5
    args, kwargs, out = _valid_case(weight, m_real=m_real, bucket_m=bucket_m)
    sentinel = 1.5
    out.fill_(sentinel)
    tail = out[m_real:].clone()

    assert a2a_gemm_fused(*args, **kwargs) is None

    _, logical, w_scale = weight
    x, x_scale = args[0], args[1]
    expected = _math_reference(
        _assemble_reference((x, x, x, x), 0, m_real), logical, x_scale, w_scale
    )
    torch.testing.assert_close(out[:m_real], expected, rtol=2e-2, atol=2e-2)
    assert torch.equal(out[m_real:], tail), "rows [M_real:bucket_m] must be untouched"


def test_bucket_m_defaults_to_m_real(weight):
    m_real = 2
    args, kwargs, out = _valid_case(weight, m_real=m_real)
    assert kwargs["bucket_m"] is None and out.shape == (m_real, _N)
    a2a_gemm_fused(*args, **kwargs)

    _, logical, w_scale = weight
    x, x_scale = args[0], args[1]
    expected = _math_reference(
        _assemble_reference((x, x, x, x), 0, m_real), logical, x_scale, w_scale
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


# -------------------------------------------------------------- registered


def test_registered_is_a_no_op_in_the_native_reference(weight):
    args, kwargs, out = _valid_case(weight)
    a2a_gemm_fused(*args, **dict(kwargs, registered=False))
    first = out.clone()

    out.zero_()
    a2a_gemm_fused(*args, **dict(kwargs, registered=True))
    assert torch.equal(out, first), "registered is a provider hint, not semantics"


# --------------------------------------------------------- invalid payloads


def _mutate_x_dtype(case, weight):
    case[0][0] = case[0][0].to(torch.bfloat16)


def _mutate_x_k(case, weight):
    case[0][0] = torch.zeros((8, 1024), dtype=torch.float8_e4m3fn)


def _mutate_x_rows(case, weight):
    case[0][0] = torch.zeros((6, _SHARD), dtype=torch.float8_e4m3fn)


def _mutate_x_scale_dtype(case, weight):
    case[0][1] = case[0][1].double()


def _mutate_x_scale_shape(case, weight):
    case[0][1] = torch.ones(case[1]["total_tokens"] // _WORLD + 1)


def _mutate_w_dtype(case, weight):
    case[0][2] = case[0][2].float()


def _mutate_w_shape(case, weight):
    case[0][2] = torch.zeros((_N, _K // 2), dtype=torch.float8_e4m3fn)


def _mutate_w_scale_shape(case, weight):
    case[0][3] = torch.ones(5, dtype=torch.float32)


def _mutate_out_dtype(case, weight):
    case[0][4] = torch.zeros_like(case[0][4], dtype=torch.float32)


def _mutate_out_shape(case, weight):
    case[0][4] = torch.zeros((case[0][4].shape[0] + 1, _N), dtype=torch.bfloat16)


def _mutate_total_tokens(case, weight):
    case[1]["total_tokens"] = case[1]["total_tokens"] - _WORLD


def _mutate_total_tokens_bool(case, weight):
    case[1]["total_tokens"] = True


def _mutate_bucket_below_m(case, weight):
    case[1]["bucket_m"] = case[1]["total_tokens"] // _WORLD - 1


def _mutate_bucket_above_max(case, weight):
    case[1]["bucket_m"] = _M_MAX + 1


def _mutate_bucket_bool(case, weight):
    case[1]["bucket_m"] = True


def _mutate_registered(case, weight):
    case[1]["registered"] = 1


def _mutate_group_world(case, weight):
    case[1]["group"] = _FakeGroup(rank=0, size=8)


def _mutate_group_rank(case, weight):
    case[1]["group"] = _FakeGroup(rank=4)


_INVALID = [
    pytest.param(_mutate_x_dtype, ValueError, "x must be", id="x-dtype"),
    pytest.param(_mutate_x_k, ValueError, "x K must be", id="x-k"),
    pytest.param(_mutate_x_rows, ValueError, "x rows must be", id="x-rows"),
    pytest.param(_mutate_x_scale_dtype, ValueError, "x_scale must be", id="xs-dtype"),
    pytest.param(
        _mutate_x_scale_shape, ValueError, "x_scale must have shape", id="xs-shape"
    ),
    pytest.param(_mutate_w_dtype, ValueError, "w must be", id="w-dtype"),
    pytest.param(_mutate_w_shape, ValueError, "w must have shape", id="w-shape"),
    pytest.param(_mutate_w_scale_shape, ValueError, "w_scale must have", id="ws-shape"),
    pytest.param(_mutate_out_dtype, ValueError, "out must be", id="out-dtype"),
    pytest.param(_mutate_out_shape, ValueError, "out must have shape", id="out-shape"),
    pytest.param(
        _mutate_total_tokens, ValueError, "total_tokens must equal", id="total-mismatch"
    ),
    pytest.param(
        _mutate_total_tokens_bool,
        ValueError,
        "total_tokens must be an int",
        id="total-bool",
    ),
    pytest.param(
        _mutate_bucket_below_m, ValueError, "bucket_m must be in", id="bucket-small"
    ),
    pytest.param(
        _mutate_bucket_above_max, ValueError, "bucket_m must be in", id="bucket-large"
    ),
    pytest.param(
        _mutate_bucket_bool, ValueError, "bucket_m must be an int", id="bucket-bool"
    ),
    pytest.param(
        _mutate_registered, ValueError, "registered must be a bool", id="registered"
    ),
    pytest.param(
        _mutate_group_world,
        NotImplementedError,
        "SP world 4",
        id="group-world",
    ),
    pytest.param(_mutate_group_rank, ValueError, "group rank must be", id="group-rank"),
]


@pytest.mark.parametrize(("mutate", "error", "match"), _INVALID)
def test_invalid_inputs_are_rejected_before_any_write(weight, mutate, error, match):
    case = list(_valid_case(weight))
    case[0] = list(case[0])
    before = case[2].clone()
    mutate(case, weight)

    with pytest.raises(error, match=match):
        a2a_gemm_fused(*case[0], **case[1])

    assert torch.equal(case[2], before), "a rejected call must not write out"


# ----------------------------------------------------- provider-facing shape


class _EntryPoint:
    def __init__(self, name, provider):
        self.name = name
        self._provider = provider

    def load(self):
        return self._provider


class _RecordingProvider:
    name = "fake-a2a-provider"
    api_versions: ClassVar[set] = {plugin.API_VERSION}
    op_versions: ClassVar[dict] = {"a2a_gemm_fused": {1}}

    def __init__(self):
        self.seen = []
        self.ran = []

    def select(self, op, args, kwargs):
        if op.name != "a2a_gemm_fused":
            return None
        self.seen.append((tuple(args), dict(kwargs)))
        return plugin.KernelCandidate(
            provider=self.name,
            kernel_id="fake.kernel",
            artifact_digest="sha256:fake",
            run=self.run,
        )

    def run(self, *args, **kwargs):
        self.ran.append((tuple(args), dict(kwargs)))


def test_provider_receives_the_canonical_contract(weight, monkeypatch):
    provider = _RecordingProvider()
    m_real = 2
    args, kwargs, out = _valid_case(weight, m_real=m_real)
    before = out.clone()
    monkeypatch.setattr(
        plugin, "_entry_points", lambda: (_EntryPoint(provider.name, provider),)
    )
    plugin.configure(source="cpu-test", enable=(provider.name,))

    a2a_gemm_fused(*args, **kwargs)

    assert len(provider.seen) == 1
    positionals, keywords = provider.seen[0]
    # The canonicalized contract: five positional values (defaults applied to
    # the keyword-only tail), in contract order.
    assert len(positionals) == 5
    for got, expected in zip(positionals, args):
        assert got is expected
    assert list(keywords) == [
        "total_tokens",
        "bucket_m",
        "registered",
        "group",
        "stream",
    ]
    assert keywords["total_tokens"] == _WORLD * m_real
    # The canonicalizer applies the *signature* default; the semantic default
    # (bucket_m := M_real) is the provider's to resolve, as contract.py does.
    assert keywords["bucket_m"] is None
    assert keywords["registered"] is False
    assert keywords["group"] is None
    assert keywords["stream"] is None
    assert torch.equal(out, before), "the provider ran, not the native"

    decision = plugin.last_decision()
    assert decision is not None
    assert (decision.route, decision.provider) == ("provider", provider.name)
    assert decision.op == plugin.Op("a2a_gemm_fused", 1)
