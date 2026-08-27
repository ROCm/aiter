# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the gfx942 f32 -> bf16 converter (``_to_bf16_fast``).

The converter is tested directly through a one-line probe kernel rather than
end-to-end, because the interesting inputs cannot be reached through the GDN
kernels' public surface: every bf16 input widens to f32 with its NaN payload in
the high bits, and every f32 input passes through arithmetic that quietens NaN
(setting bit 22) before the conversion. A low-payload NaN therefore only exists
at the converter's own boundary -- which is exactly where the rounding bias used
to turn it into infinity, so that is where it has to be tested.
"""

import struct

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx942", reason="_to_bf16_fast is the gfx942 converter"
)


def _f32_from_bits(bits: int) -> torch.Tensor:
    return torch.tensor([bits], dtype=torch.int32).view(torch.float32)


def _bits_of_bf16(t: torch.Tensor) -> int:
    return int(t.view(torch.int16).to(torch.int32).item() & 0xFFFF)


# (name, f32 bit pattern, what the bf16 result must be)
#
# "nan" means any encoding with exponent 0xFF and a non-zero mantissa; the exact
# payload is not pinned because bf16 cannot represent most of them. "+inf" /
# "-inf" pin the encoding exactly.
_CASES = [
    ("low-payload NaN (smallest)", 0x7F800001, "nan"),
    ("low-payload NaN (bit 15)", 0x7F808000, "nan"),
    ("low-payload NaN, negative", 0xFF800001, "nan"),
    ("signaling NaN", 0x7FA00000, "nan"),
    ("quiet NaN", 0x7FC00000, "nan"),
    ("quiet NaN, negative", 0xFFC00000, "nan"),
    ("+Inf", 0x7F800000, 0x7F80),
    ("-Inf", 0xFF800000, 0xFF80),
    ("+0.0", 0x00000000, 0x0000),
    ("-0.0", 0x80000000, 0x8000),
    ("1.0", 0x3F800000, 0x3F80),
    ("-1.0", 0xBF800000, 0xBF80),
    ("FLT_MAX", 0x7F7FFFFF, None),  # must stay finite-or-inf, never NaN
]


@pytest.fixture(scope="module")
def convert():
    """Compile a probe kernel that applies ``_to_bf16_fast`` elementwise."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import _to_bf16_fast

    THREADS = 64

    def _flat(tensor):
        buf = fx.rocdl.make_buffer_tensor(tensor, max_size=False)
        n = fx.get_scalar(fx.cosize(fx.get_layout(buf)))
        return fx.Tensor(fx.make_view(fx.get_iter(buf), fx.make_layout((n,), (1,))))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def probe(src_tensor: fx.Tensor, dst_tensor: fx.Tensor, n_val: fx.Int32):
        src = _flat(src_tensor)
        dst = _flat(dst_tensor)
        idx = fx.block_idx.x * fx.Int32(THREADS) + fx.thread_idx.x
        if (idx < n_val).ir_value():

            def _store(off, value):
                dst[(off,)] = value

            _store(idx, _to_bf16_fast(fx.Float32(src[(idx,)])))

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, n_val: fx.Int32, stream: fx.Stream):
        probe(src, dst, n_val).launch(
            grid=((n_val + fx.Int32(THREADS - 1)) // fx.Int32(THREADS), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    def run(bit_patterns):
        src = (
            torch.tensor(
                [struct.unpack("<i", struct.pack("<I", b))[0] for b in bit_patterns],
                dtype=torch.int32,
                device="cuda",
            )
            .view(torch.float32)
            .contiguous()
        )
        dst = torch.empty(len(bit_patterns), dtype=torch.bfloat16, device="cuda")
        launch(src, dst, len(bit_patterns), torch.cuda.current_stream())
        torch.cuda.synchronize()
        return dst

    return run


@pytest.mark.parametrize("name,bits,expected", _CASES, ids=[c[0] for c in _CASES])
def test_to_bf16_fast_classification(convert, name, bits, expected):
    """NaN stays NaN, Inf stays Inf with its sign, finite values round correctly."""
    out = convert([bits])[0]
    got = _bits_of_bf16(out)

    if expected == "nan":
        assert torch.isnan(out.float()), (
            f"{name} (0x{bits:08X}) converted to 0x{got:04X} "
            f"(={out.float().item()}), which is not NaN. A NaN that rounds to "
            f"infinity is the low-payload carry bug."
        )
    elif expected is None:
        assert not torch.isnan(
            out.float()
        ), f"{name} (0x{bits:08X}) converted to NaN (0x{got:04X})"
    else:
        assert (
            got == expected
        ), f"{name} (0x{bits:08X}) -> 0x{got:04X}, expected 0x{expected:04X}"


def test_to_bf16_fast_low_payload_nan_is_not_inf(convert):
    """The specific regression: low-payload NaNs must not become infinity.

    ``(bits + 0x8000) >> 16`` carries into the exponent for any NaN whose
    payload sits below bit 16, producing 0x7F80 -- positive infinity.
    """
    patterns = [0x7F800001, 0x7F800002, 0x7F804000, 0x7F808000, 0xFF800001]
    out = convert(patterns).float()
    became_inf = torch.isinf(out)
    assert not became_inf.any(), (
        "low-payload NaNs converted to infinity at positions "
        f"{became_inf.nonzero().flatten().tolist()} of {patterns}"
    )
    assert torch.isnan(out).all(), f"expected all NaN, got {out.tolist()}"


def test_to_bf16_fast_rounding_still_round_half_away(convert):
    """The NaN guard must not disturb rounding of ordinary finite values."""
    # 1.0 + 2^-9 sits exactly halfway between two bf16 values and must round
    # away from zero (up), which is this converter's documented contract.
    halfway = 0x3F808000  # 1.00390625
    assert _bits_of_bf16(convert([halfway])[0]) == 0x3F81

    # A spread of ordinary values must land within half a bf16 ulp.
    # ``device="cpu"`` is explicit: another test module in the same session may
    # have installed a default CUDA device, which would silently put ``vals`` on
    # the GPU and make this comparison a cross-device error.
    torch.manual_seed(0)
    vals = torch.randn(1024, dtype=torch.float32, device="cpu") * 100.0
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in vals.tolist()]
    got = convert(bits).float().cpu()
    ulp = vals.abs() * (2.0**-8)
    assert torch.all((got - vals).abs() <= ulp + 1e-30), "rounding error exceeds 1 ulp"
