# SPDX-License-Identifier: MIT
"""CPU-only exact E4M3 conversion models; no torch/FlyDSL/GPU imports.

The legacy scalar semantics deliberately decode 0x7f/0xff as +/-480.
The native model uses IEEE E4M3FN (endpoints NaN), then the proposed
positive-magnitude pk8 BF16 conversion + packed min + original-sign fix.

FTZ domains are separate: the legacy half-slot multiply has FP16 subnormal
inputs; direct native BF16 conversion has no such intermediate. Its smallest
finite output is BF16-normal. This CPU proof does NOT establish whether
hardware flushes *FP8 source* subnormals; that is an independent ISA/GPU gate.
Run: python aiter/ops/flydsl/kernels/mqa_logits/_g2/test_fp8_cvt_cpu.py
"""
from __future__ import annotations

import sys as _n64_sys, pathlib as _n64_path  # ROUND-NATIVE64 seam support
_n64_sys.path.insert(0, str(_n64_path.Path(__file__).resolve().parent))
from native64_seam import ensure_i32  # noqa: E402

import ast
from contextlib import contextmanager
from pathlib import Path
import random
import struct
import sys
import types
import unittest

U32 = 0xffffffff
KERNEL_PATH = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def half_from_bits(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xffff))[0]


def half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def legacy_scalar_f32(b):
    """Literal numerical mirror of existing scalar integer formula."""
    sign = (b & 0x80) << 24
    normal = struct.unpack("<f", struct.pack("<I", ((b & 0x7f) << 20) + 0x3c000000 | sign))[0]
    sub = struct.unpack("<f", struct.pack("<I", 0x46800000 | (b & 7)))[0] - 16384.0
    return (-sub if b & 0x80 else sub) if b & 0x78 == 0 else normal


def e4m3_byte_to_bf16_bits(b):
    """Independent integer decoder: exact for all 256 legacy codes."""
    magnitude = b & 0x7f
    sign = (b & 0x80) << 8
    if magnitude >= 8:
        return sign | (0x3c00 + (magnitude << 4))
    if magnitude == 0:
        return sign
    leading = magnitude.bit_length() - 1
    # magnitude * 2**-9; exponent = leading - 9, fraction normalized to 7 bits.
    return sign | ((leading + 118) << 7) | ((magnitude - (1 << leading)) << (7 - leading))


def legacy_half_slot_bits(b):
    return ((b & 0x80) << 8) | ((b & 0x7f) << 7)


def flush_half_input(bits, ftz):
    if ftz and 0 < (bits & 0x7fff) < 0x0400:
        return bits & 0x8000
    return bits


def existing_half_slot_model(b, slot, ftz_input=False, ftz_output=False):
    """One code in low/high half; multiplication separately models both FTZ domains."""
    if slot not in (0, 1):
        raise ValueError("half slot must be 0 or 1")
    packed = legacy_half_slot_bits(b) << (16 * slot)
    h = (packed >> (16 * slot)) & 0xffff
    product = half_bits(half_from_bits(flush_half_input(h, ftz_input)) * 256.0)
    product = flush_half_input(product, ftz_output)
    return f32_bits(half_from_bits(product)) >> 16


def perm_bytes(first, second, selector):
    """v_perm_b32 model: selectors 0..3 name second, 4..7 name first."""
    data = second.to_bytes(4, "little") + first.to_bytes(4, "little")
    return sum(data[(selector >> (8 * i)) & 7] << (8 * i) for i in range(4))


def pk_min_u16(a, b):
    return min(a & 0xffff, b & 0xffff) | (min(a >> 16 & 0xffff, b >> 16 & 0xffff) << 16)


def native_positive_bf16(b, nan_payload=0x7fc0, ftz_output=False, flush_fp8_source=False):
    """Ideal positive native decode; output FTZ has no effect at E8M0 scale127.

    nan_payload may be any BF16 NaN encoding with either sign, not a fixed
    canonical encoding: a positive source need not produce a positive NaN.
    flush_fp8_source intentionally models a *different*, hypothetical policy.
    """
    b &= 0x7f
    if b == 0x7f:
        if not (0 <= nan_payload <= 0xffff and 0x7f81 <= (nan_payload & 0x7fff) <= 0x7fff):
            raise ValueError("BF16 NaN payload with either sign required")
        return nan_payload
    if flush_fp8_source and 0 < b < 8:
        return 0
    result = e4m3_byte_to_bf16_bits(b)
    if ftz_output and 0 < result < 0x0080:
        return 0
    return result


def native_pk8_min_model(x0, x1, *, nan_payload=0x7fc0, ftz_output=False, flush_fp8_source=False):
    """Exact parent candidate: 2 source masks + native pk8 + 4*(min,perm,andor).

    Returns four dwords, sequential pairs of little-endian source bytes.
    Finite outputs are positive because source signs are cleared. Either sign
    of native NaN is >0x43f0 as unsigned16 and is clamped before sign restoration.
    """
    original = (x0 & U32, x1 & U32)
    positive = tuple(x & 0x7f7f7f7f for x in original)
    results = []
    for word, source in zip(positive, original):
        converted = [native_positive_bf16(word >> (8 * i) & 0xff, nan_payload,
                     ftz_output, flush_fp8_source) for i in range(4)]
        for pair, selector in ((0, 0x01010000), (1, 0x03030202)):
            native = converted[2 * pair] | (converted[2 * pair + 1] << 16)
            clamped = pk_min_u16(native, 0x43f043f0)
            signs = perm_bytes(source, source, selector) & 0x80008000
            results.append(clamped | signs)
    return tuple(results)


def reference_pk8(x0, x1):
    results = []
    for word in (x0, x1):
        codes = [e4m3_byte_to_bf16_bits(word >> (8 * i) & 0xff) for i in range(4)]
        results.extend((codes[0] | codes[1] << 16, codes[2] | codes[3] << 16))
    return tuple(results)


class _Vector:
    """Small stdlib mock sufficient to execute the ACTUAL legacy source helper."""
    ftz_input = False
    ftz_output = False

    def __init__(self, data, dtype=None):
        if isinstance(data, _Vector):
            self.data, self.dtype = data.data[:], data.dtype if dtype is None else dtype
        else:
            self.data, self.dtype = list(data), dtype

    def ir_value(self):
        return self

    @classmethod
    def make_type(cls, count, dtype):
        return count, dtype

    @classmethod
    def from_elements(cls, data, dtype):
        return cls(data, dtype)

    @classmethod
    def filled(cls, count, value, dtype):
        return cls([value] * count, dtype)

    def bitcast(self, dtype):
        if self.dtype is int and dtype == "f16":
            halves = [half_from_bits(x >> (16 * i) & 0xffff) for x in self.data for i in range(2)]
            return _Vector(halves, dtype)
        if self.dtype == "f32" and dtype is int:
            return _Vector([f32_bits(x) for x in self.data], dtype)
        raise TypeError((self.dtype, dtype))

    def __mul__(self, other):
        result = []
        for a, b in zip(self.data, other.data):
            a = half_from_bits(flush_half_input(half_bits(a), self.ftz_input))
            bits = half_bits(a * b)
            result.append(half_from_bits(flush_half_input(bits, self.ftz_output)))
        return _Vector(result, self.dtype)

    def to(self, dtype):
        return _Vector(self.data, dtype)

    def __getitem__(self, index):
        return self.data[index]


def load_actual_legacy_helper():
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_e4m3x4_to_bf16x2")
    module = ast.Module(body=[selected], type_ignores=[])
    namespace = {"fx": types.SimpleNamespace(Int32=int, Float16="f16", Float32="f32"),
                 "Vec": _Vector, "_E4M3_F16_MAG": 0x3f803f80,
                 "_E4M3_F16_SIGN": 0x80008000, "_BF16_LO_MASK": 0xffff,
                 "_BF16_HI_MASK": 0xffff0000}
    exec(compile(module, str(KERNEL_PATH), "exec"), namespace)
    return namespace["_e4m3x4_to_bf16x2"]


class _Bool:
    def __init__(self, value):
        self.value = bool(value)

    def __bool__(self):
        return self.value

    def select(self, yes, no):
        return yes if self.value else no


class _I32(int):
    ir_type = "i32"

    def __new__(cls, value):
        return int.__new__(cls, int(value) & U32)

    def ir_value(self):
        return self

    def __eq__(self, other):
        return _Bool(int(self) == int(other))


class _MemoryPointer:
    def __init__(self, words, index=0, loads=None):
        self.words, self.index = words, index
        self.loads = [] if loads is None else loads

    def __add__(self, offset):
        return _MemoryPointer(self.words, self.index + int(offset), self.loads)

    def load(self, result_type):
        count, dtype = result_type
        assert 0 <= self.index <= len(self.words) - count
        self.loads.append((self.index * 4, count * 4))
        return _Vector(self.words[self.index:self.index + count], dtype)


class _Asm:
    nan_payload = 0x7fc0
    flush_fp8_source = False
    operations = []

    @classmethod
    def inline_asm(cls, result_type, operands, text, constraints, has_side_effects):
        assert has_side_effects is False
        cls.operations.append(text)
        if text == "v_cvt_scale_pk8_bf16_fp8 $0, $1, 0x7f":
            # Native pk8 outputs must not overlap its two source registers.
            assert constraints == "=&v,v" and result_type == (4, _I32)
            src = operands[0].data
            assert len(src) == 2
            assert all(int(word) & 0x80808080 == 0 for word in src)
            values = [native_positive_bf16(int(word) >> (8 * i) & 255,
                      cls.nan_payload, flush_fp8_source=cls.flush_fp8_source)
                      for word in src for i in range(4)]
            return _Vector([values[i] | values[i + 1] << 16 for i in range(0, 8, 2)], _I32)
        if text == "v_pk_min_u16 $0, $1, $2":
            assert constraints == "=v,v,s"
            return _I32(pk_min_u16(int(operands[0]), int(operands[1])))
        if text.startswith("v_perm_b32 $0, $1, $1, "):
            assert constraints == "=v,v"
            word = int(operands[0])
            return _I32(perm_bytes(word, word, int(text.rsplit(" ", 1)[1], 16)))
        if text == "v_and_or_b32 $0, $1, $2, $3":
            assert constraints == "=v,v,s,v"
            return _I32((int(operands[0]) & int(operands[1])) | int(operands[2]))
        raise AssertionError("unmodeled inline assembly: " + text)


@contextmanager
def mocked_llvm_import():
    names = ("flydsl", "flydsl._mlir", "flydsl._mlir.dialects")
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules[names[-1]].llvm = types.SimpleNamespace(inline_asm=_Asm.inline_asm)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_actual_fast_helpers():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    wanted = {"_e4m3x8_to_bf16x4_fast", "_pair_frag_tdm_fast", "_pair_frag_tdm"}
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise AssertionError("actual fast helpers are missing")
    fx = types.SimpleNamespace(Int32=_I32, ptr_load=lambda ptr, result_type: ptr.load(result_type))
    namespace = {"fx": fx, "Vec": _Vector, "range_constexpr": range,
                 "_s32": lambda value: value - 2**32 if value >= 2**31 else value,
                 "_e4m3x4_to_bf16x2": load_actual_legacy_helper()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(KERNEL_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in ("_e4m3x8_to_bf16x4_fast", "_pair_frag_tdm_fast", "_pair_frag_tdm"))


def actual_cvt_environment(value):
    """Execute only the production environment assignment and validation AST."""
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_FP8_CVT" for t in node.targets))
    environment = {} if value is None else {"QSA_FP8_CVT": value}
    namespace = {"os": types.SimpleNamespace(environ=environment)}
    exec(compile(ast.Module(body=tree.body[start:start + 2], type_ignores=[]),
                 str(KERNEL_PATH), "exec"), namespace)
    return namespace["_FP8_CVT"]


def actual_cvt_route(value, rows, k_load, access="raw"):
    """Execute the real resolver/selectors/name, without compiling a kernel."""
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    build = top["_build_wmma_fp8_kernel"]
    helpers = {n.name: n.name for n in build.body if isinstance(n, ast.FunctionDef)}
    names = {"k_load", "use_tdm", "fast_cvt", "pair_storage", "pair_stage",
             "pair_base", "pair_frag", "kernel_name"}
    selected = [top["_resolve_fp8_k_load"]]
    for node in build.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                               for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "fast_cvt":
            selected.append(node)
    namespace = dict(helpers, _FP8_CVT=actual_cvt_environment(value), _FP8_K_LOAD="auto",
                     _ACCESS=access, rows_per_block=rows, k_load=k_load,
                     HEAD_DIM=128, NUM_HEADS=4, PAGE_SIZE=4, WMMA_COLS=64)
    ensure_i32(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(KERNEL_PATH), "exec"), namespace)
    return namespace


class ConversionTests(unittest.TestCase):
    def test_actual_environment_default_current_fast_and_invalid(self):
        self.assertEqual(actual_cvt_environment(None), "fast")
        for value, expected in (("current", "current"), ("fast", "fast"),
                                (" CURRENT ", "current"), (" FAST ", "fast")):
            self.assertEqual(actual_cvt_environment(value), expected)
        for value in ("", " ", "auto", "0", "1", "faster", "current,fast"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "QSA_FP8_CVT"):
                actual_cvt_environment(value)

    def test_actual_default_current_fast_routing_and_unchanged_paths(self):
        for rows in (16, 32):
            for k_load in (None, "auto", "raw", "tdm"):
                for access in ("raw", "buffer"):
                    routes = {mode: actual_cvt_route(mode, rows, k_load, access)
                              for mode in (None, "current", "fast")}
                    eligible = rows == 32 and k_load != "raw" and access == "raw"
                    for mode, route in routes.items():
                        fast = eligible and mode != "current"
                        self.assertEqual(route["fast_cvt"], fast)
                        self.assertEqual(route["pair_frag"], "_pair_frag_tdm_fast" if fast else
                                         "_pair_frag_tdm" if eligible else "_pair_frag_raw")
                        self.assertEqual(route["kernel_name"].endswith("_cvtfast"), fast)
                    keys = ("pair_storage", "pair_stage", "pair_base")
                    for key in keys:
                        self.assertEqual(routes[None][key], routes["current"][key])
                    self.assertEqual(routes[None]["kernel_name"], routes["fast"]["kernel_name"])
                    if not eligible:
                        self.assertEqual(routes[None]["kernel_name"], routes["current"]["kernel_name"])

    def test_independent_integer_vs_scalar_all_codes(self):
        for b in range(256):
            self.assertEqual(e4m3_byte_to_bf16_bits(b), f32_bits(legacy_scalar_f32(b)) >> 16)
        self.assertEqual(e4m3_byte_to_bf16_bits(0x7f), 0x43f0)
        self.assertEqual(e4m3_byte_to_bf16_bits(0xff), 0xc3f0)

    def test_half_slots_all_codes_and_four_denormal_modes(self):
        bad_codes = {1, 2, 3, 4, 5, 6, 7, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87}
        for slot in (0, 1):
            for ftz_input in (False, True):
                for ftz_output in (False, True):
                    mismatches = set()
                    for b in range(256):
                        got = existing_half_slot_model(b, slot, ftz_input, ftz_output)
                        if got != e4m3_byte_to_bf16_bits(b):
                            mismatches.add(b)
                            self.assertEqual(got, (b & 0x80) << 8)
                    self.assertEqual(mismatches, bad_codes if ftz_input else set())

    def test_actual_source_helper_all_codes_all_byte_positions(self):
        helper = load_actual_legacy_helper()
        for ftz in (False, True):
            _Vector.ftz_input = ftz
            for position in range(4):
                for b in range(256):
                    # Signed zero / endpoint / subnormal sentinels expose ordering errors.
                    codes = [0x80, 0x7f, 0xff, 0x03]
                    codes[position] = b
                    word = sum(code << (8 * i) for i, code in enumerate(codes))
                    expected = []
                    for pair in (0, 1):
                        values = [existing_half_slot_model(codes[2 * pair + i], i, ftz) for i in (0, 1)]
                        expected.append(values[0] | values[1] << 16)
                    self.assertEqual(tuple(helper(word)), tuple(expected))
        _Vector.ftz_input = False

    def test_native_all_codes_all_eight_byte_positions_and_bf16_output_ftz(self):
        for ftz in (False, True):
            for position in range(8):
                for b in range(256):
                    codes = [0, 0x80, 0x7f, 0xff, 1, 0x87, 0x7e, 0xfe]
                    codes[position] = b
                    words = [sum(codes[j + i] << (8 * i) for i in range(4)) for j in (0, 4)]
                    self.assertEqual(native_pk8_min_model(*words, ftz_output=ftz), reference_pk8(*words))

    def test_native_all_nan_payloads_both_signs_and_half_positions(self):
        for payload in tuple(range(0x7f81, 0x8000)) + tuple(range(0xff81, 0x10000)):
            for position in range(8):
                for endpoint in (0x7f, 0xff):
                    codes = [0x80, 0, 0x87, 1, 0x7e, 0xfe, 3, 0x83]
                    codes[position] = endpoint
                    words = [sum(codes[j + i] << (8 * i) for i in range(4)) for j in (0, 4)]
                    self.assertEqual(native_pk8_min_model(*words, nan_payload=payload), reference_pk8(*words))

    def test_native_all_adjacent_code_pairs(self):
        for a in range(256):
            for b in range(256):
                word = a | b << 8 | (a ^ 0x80) << 16 | (b ^ 0x80) << 24
                self.assertEqual(native_pk8_min_model(word, word ^ 0x80808080), reference_pk8(word, word ^ 0x80808080))

    def test_native_random_full_words(self):
        rng = random.Random(455)
        for _ in range(10000):
            words = rng.getrandbits(32), rng.getrandbits(32)
            self.assertEqual(native_pk8_min_model(*words), reference_pk8(*words))

    def test_actual_fast_source_all_codes_all_positions_both_ambient_modes(self):
        helper, _, _ = load_actual_fast_helpers()
        with mocked_llvm_import():
            for ambient_half_ftz in (False, True):
                _Vector.ftz_input = ambient_half_ftz
                for position in range(8):
                    legacy_mismatches = set()
                    for b in range(256):
                        codes = [0] * 8
                        codes[position] = b
                        words = [sum(codes[j + i] << (8 * i) for i in range(4)) for j in (0, 4)]
                        _Asm.operations = []
                        actual = helper(*words)
                        self.assertEqual(tuple(map(int, actual.data)), reference_pk8(*words))
                        self.assertEqual(len(_Asm.operations), 13)  # plus two DSL source masks
                        self.assertEqual(sum(op.startswith("v_cvt_scale_pk8") for op in _Asm.operations), 1)
                        reference_old = existing_half_slot_model(b, position % 2, ambient_half_ftz)
                        if reference_old != e4m3_byte_to_bf16_bits(b):
                            legacy_mismatches.add(b)
                    self.assertEqual(legacy_mismatches,
                        set(range(1, 8)) | set(range(0x81, 0x88)) if ambient_half_ftz else set())
        _Vector.ftz_input = False

    def test_actual_fast_source_all_nan_payloads(self):
        helper, _, _ = load_actual_fast_helpers()
        with mocked_llvm_import():
            for payload in tuple(range(0x7f81, 0x8000)) + tuple(range(0xff81, 0x10000)):
                _Asm.nan_payload = payload
                # Every pair combines both endpoints, exercising both half slots/signs.
                words = (0xff7fff7f, 0x7fff7fff)
                self.assertEqual(tuple(map(int, helper(*words).data)), reference_pk8(*words))
        _Asm.nan_payload = 0x7fc0

    def test_actual_fast_tdm_consumer_all_rows_chunks_lanes(self):
        _, fast_consumer, old_consumer = load_actual_fast_helpers()
        # Load the existing independent FP8 TDM CPU model without aiter/FlyDSL/GPU.
        model_path = Path(__file__).with_name("test_fp8_k_tdm_cpu.py")
        module_name = "_qsa_fp8_cvt_tdm_reference"
        model = types.ModuleType(module_name)
        model.__file__ = str(model_path)
        sys.modules[module_name] = model
        exec(compile(model_path.read_text(encoding="utf-8"), str(model_path), "exec"), model.__dict__)
        source = bytes((i * 37 + (i // 128) * 11) & 255 for i in range(8 * 512))
        table = [0, 1, -1, 99, 4, 5, 6, 7, 2, 0, 7, 1, 3, 4, 5, 6]
        tile, _, _, _ = model.stage_tile(source, table)
        tile_words = list(struct.unpack("<2304I", tile))
        _Vector.ftz_input = False
        with mocked_llvm_import():
            for nt in range(4):
                for chunk in range(4):
                    for lane in range(32):
                        base = (nt * 16 + lane % 16) * 144 + chunk * 32
                        fast_ptr, old_ptr = _MemoryPointer(tile_words), _MemoryPointer(tile_words)
                        actual = fast_consumer(fast_ptr, base, _I32(lane // 16))
                        original = old_consumer(old_ptr, base, _I32(lane // 16))
                        _, expected_bytes, _ = model.raw_fragment(source, table, 0, nt, chunk, lane, 64, 64)
                        expected = struct.unpack("<8I", expected_bytes)
                        self.assertEqual(tuple(map(int, actual.data)), expected)
                        self.assertEqual(tuple(map(int, original.data)), expected)
                        self.assertEqual(fast_ptr.loads, [(base, 16), (base + 16, 16)])
                        self.assertEqual(fast_ptr.loads, old_ptr.loads)
        sys.modules.pop(module_name, None)

    def test_hypothetical_fp8_source_ftz_distinction(self):
        mismatches = []
        for b in range(256):
            got = native_pk8_min_model(b, 0, flush_fp8_source=True)[0] & 0xffff
            if got != e4m3_byte_to_bf16_bits(b):
                mismatches.append(b)
        self.assertEqual(mismatches, list(range(1, 8)) + list(range(0x81, 0x88)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
    # unittest.main exits, so all statements above are the executable interface.
