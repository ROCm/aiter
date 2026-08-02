from __future__ import annotations

import pytest

from aiter.ops.flydsl.kernels import mxmoe_dispatcher as dispatcher
from aiter.ops.flydsl.kernels import mxmoe_gemm_v2

G2_COMMON = dict(
    BM=32,
    BN=128,
    BK=256,
    use_nt=True,
    HIDDEN_MAX=8192,
    epilog="atomic",
    INTER_MAX=8192,
    a_dtype="fp4",
)


def _install_g2_compile_collector(monkeypatch, spart):
    compile_calls = []

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        return object()

    monkeypatch.setattr(dispatcher, "G2_CACHE", {})
    monkeypatch.setattr(dispatcher, "compile_gemm2_a4w4_port", fake_compile)
    monkeypatch.setenv("MXFP4_G2_KSTAGES", "2")
    monkeypatch.setenv("MXFP4_G2_BHOIST", "1")
    monkeypatch.setenv("MXFP4_G2_ASCALE_PF", "1")
    monkeypatch.setenv("MXFP4_G2_SPART", str(spart))
    monkeypatch.setenv("MXFP4_G2_BF16_LDS", "0")
    return compile_calls


def test_get_g2_keys_and_forwards_spart_opt(monkeypatch):
    compile_calls = _install_g2_compile_collector(monkeypatch, spart=402)

    monkeypatch.setenv("MXFP4_G2_SPART_OPT", "0")
    dispatcher.get_g2(**G2_COMMON)
    monkeypatch.setenv("MXFP4_G2_SPART_OPT", "1")
    dispatcher.get_g2(**G2_COMMON)

    assert [call["g2_spart_opt"] for call in compile_calls] == [False, True]
    assert [call["g2_kstages"] for call in compile_calls] == [2, 2]


def test_get_g2_normalizes_spart_opt_off_when_spart_is_disabled(monkeypatch):
    compile_calls = _install_g2_compile_collector(monkeypatch, spart=0)

    monkeypatch.setenv("MXFP4_G2_SPART_OPT", "0")
    dispatcher.get_g2(**G2_COMMON)
    monkeypatch.setenv("MXFP4_G2_SPART_OPT", "1")
    dispatcher.get_g2(**G2_COMMON)

    assert len(compile_calls) == 1
    assert compile_calls[0]["g2_spart_opt"] is False
    assert compile_calls[0]["g2_kstages"] == 2


def test_get_g2_normalizes_spart_opt_off_when_persistent(monkeypatch):
    compile_calls = _install_g2_compile_collector(monkeypatch, spart=402)

    monkeypatch.setenv("MXFP4_G2_SPART_OPT", "0")
    dispatcher.get_g2(**G2_COMMON, persist=True, cu_num=8)
    monkeypatch.setenv("MXFP4_G2_SPART_OPT", "1")
    dispatcher.get_g2(**G2_COMMON, persist=True, cu_num=8)

    assert len(compile_calls) == 1
    assert compile_calls[0]["g2_spart_opt"] is False
    assert compile_calls[0]["g2_kstages"] == 2


def test_unsigned_spart_helpers_wrap_operands(monkeypatch):
    calls = []

    def uint32(value):
        calls.append(("u32", value))
        return value

    def int32(value):
        calls.append(("i32", value))
        return value

    monkeypatch.setattr(dispatcher.fx, "Uint32", uint32)
    monkeypatch.setattr(dispatcher.fx, "Int32", int32)

    assert dispatcher._udiv_i32(9, 4) == 2
    assert calls == [("u32", 9), ("u32", 4), ("i32", 2)]

    calls.clear()
    assert dispatcher._umod_i32(9, 4) == 1
    assert calls == [("u32", 9), ("u32", 4), ("i32", 1)]


def test_resolve_tile_coords_precomputed_and_fallback_paths():
    class NoDivide:
        def __floordiv__(self, _other):
            raise AssertionError("precomputed coordinates must bypass division")

    assert mxmoe_gemm_v2._resolve_tile_coords(NoDivide(), 48, 3, 7) == (3, 7)
    assert mxmoe_gemm_v2._resolve_tile_coords(99, 48) == (2, 3)

    with pytest.raises(AssertionError):
        mxmoe_gemm_v2._resolve_tile_coords(99, 48, 3, None)
    with pytest.raises(AssertionError):
        mxmoe_gemm_v2._resolve_tile_coords(99, 48, None, 7)


class HostPredicate:
    def __init__(self, value):
        self.value = bool(value)

    def select(self, true_value, false_value):
        return true_value if self.value else false_value


class HostInt:
    def __init__(self, value):
        self.value = int(value)

    @staticmethod
    def _value(other):
        return other.value if isinstance(other, HostInt) else int(other)

    def __add__(self, other):
        return HostInt(self.value + self._value(other))

    __radd__ = __add__

    def __sub__(self, other):
        return HostInt(self.value - self._value(other))

    def __rsub__(self, other):
        return HostInt(self._value(other) - self.value)

    def __mul__(self, other):
        return HostInt(self.value * self._value(other))

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return HostInt(self.value // self._value(other))

    def __mod__(self, other):
        return HostInt(self.value % self._value(other))

    def __lt__(self, other):
        return HostPredicate(self.value < self._value(other))

    def __le__(self, other):
        return HostPredicate(self.value <= self._value(other))

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"HostInt({self.value})"


def _ck_spart_output_tiles(m0, n0, group_num, m01):
    tile_count = m0 * n0
    group_size = (tile_count + group_num - 1) // group_num
    big_group_num = group_num - (group_size * group_num - tile_count)
    m0_mod = m0 % m01
    tiles = []

    for block_id in range(tile_count):
        group_id_y, group_id_x = divmod(block_id, group_num)
        if group_id_x <= big_group_num:
            remap = group_id_x * group_size + group_id_y
        else:
            remap = group_id_x * group_size + big_group_num - group_id_x + group_id_y

        idx_m0, idx_n0 = divmod(remap, n0)
        m01_adapt = m01 if idx_m0 < m0 - m0_mod else m0_mod
        idx_m00, idx_m01 = divmod(idx_m0, m01)
        n_out, loc_mod = divmod(idx_n0 + idx_m01 * n0, m01_adapt)
        tiles.append((loc_mod + idx_m00 * m01, n_out))

    return tiles


@pytest.mark.parametrize(
    ("m0", "n0", "group_num", "m01"),
    [
        pytest.param(257, 48, 4, 2, id="production"),
        pytest.param(1, 1, 4, 2, id="singleton"),
        pytest.param(3, 5, 4, 7, id="m0-lt-m01"),
        pytest.param(3, 5, 4, 2, id="uneven-small"),
        pytest.param(4, 7, 4, 2, id="m0-divisible-by-m01"),
        pytest.param(7, 6, 4, 3, id="partial-final-m01"),
        pytest.param(7, 6, 5, 3, id="uneven-non-power-of-two"),
        pytest.param(8, 11, 5, 3, id="non-power-of-two-group-m01"),
    ],
)
def test_spart_output_tile_index_matches_ck_reference(
    monkeypatch, m0, n0, group_num, m01
):
    uint32_calls = []

    def int32(value):
        return value if isinstance(value, HostInt) else HostInt(value)

    def uint32(value):
        uint32_calls.append(int(value))
        return value if isinstance(value, HostInt) else HostInt(value)

    monkeypatch.setattr(dispatcher.fx, "Int32", int32)
    monkeypatch.setattr(dispatcher.fx, "Uint32", uint32)

    def map_all(use_unsigned):
        return [
            tuple(
                int(coord)
                for coord in dispatcher._spart_output_tile_index(
                    HostInt(block),
                    HostInt(m0),
                    n0,
                    group_num,
                    m01,
                    use_unsigned=use_unsigned,
                )
            )
            for block in range(m0 * n0)
        ]

    signed = map_all(False)
    assert uint32_calls == []
    unsigned = map_all(True)
    assert uint32_calls
    reference = _ck_spart_output_tiles(m0, n0, group_num, m01)

    assert signed == reference
    assert unsigned == reference
    assert all(0 <= m < m0 and 0 <= n < n0 for m, n in signed)
    assert len(signed) == m0 * n0
    assert set(signed) == {(m, n) for m in range(m0) for n in range(n0)}
