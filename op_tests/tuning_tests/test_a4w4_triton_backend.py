# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the triton backend of gemm_a4w4.

The a4w4_blockscale table can name a backend in its ``libtype`` column: a
``triton`` row routes :func:`aiter.gemm_a4w4` to triton's preshuffled MXFP4
kernel, and the a4w4 tuner times that kernel as one more candidate for every
supported shape.  These tests pin down both halves:

* ``test_*`` in :class:`TestA4W4TritonDispatch` exercise the CSV-driven
  selection / fallback rules and the operand-layout conversion.
* ``test_*`` in :class:`TestA4W4TunerAddsTriton` exercise the tuner's candidate
  list and the ``libtype`` recording.

No GPU, torch build or pandas is required: when a dependency cannot be
imported it is replaced by a small stand-in (same approach as
``test_asm_splitk_guard.py``).  Run:

    python3 -m unittest op_tests.tuning_tests.test_a4w4_triton_backend -v
"""

import importlib
import importlib.util
import math
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DISPATCH_PATH = _REPO_ROOT / "aiter" / "ops" / "gemm_op_a4w4.py"
_TUNER_PATH = (
    _REPO_ROOT / "csrc" / "ck_gemm_a4w4_blockscale" / "gemm_a4w4_blockscale_tune.py"
)

# ---------------------------------------------------------------------------
# Minimal stand-ins for torch / pandas / the aiter extension modules.
# ---------------------------------------------------------------------------


class _DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, _DType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class _Tensor:
    """Only used as the ``torch.Tensor`` annotation on the decorated ops."""


class FakeTensor:
    """Just enough of a tensor for the dispatch/validation code paths."""

    def __init__(self, shape, dtype=None, device="cpu", contiguous=True):
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self.data = object()

    def __repr__(self):
        return f"FakeTensor({self.shape}, {self.dtype})"

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return 1

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return FakeTensor(self.shape, self.dtype, self.device, True)

    def _resolve(self, args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = tuple(args[0])
        if args.count(-1) > 1:
            raise ValueError(f"only one -1 allowed, got {args}")
        known = 1
        for dim in args:
            if dim != -1:
                known *= dim
        if -1 in args:
            if known == 0 or self.numel() % known:
                raise ValueError(f"cannot reshape {self.shape} to {args}")
            return tuple(self.numel() // known if d == -1 else d for d in args)
        total = 1
        for dim in args:
            total *= dim
        if total != self.numel():
            raise ValueError(f"cannot reshape {self.shape} to {args}")
        return tuple(args)

    def view(self, *args):
        # tensor.view(torch.uint8) is the dtype-reinterpret form.
        if len(args) == 1 and not isinstance(args[0], (int, tuple, list)):
            if not self._contiguous:
                raise RuntimeError("view(dtype) requires a contiguous tensor")
            return FakeTensor(self.shape, args[0], self.device, True)
        return FakeTensor(self._resolve(args), self.dtype, self.device, True)

    def reshape(self, *args):
        return FakeTensor(self._resolve(args), self.dtype, self.device, True)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            shape = list(self.shape)
            for axis, selection in enumerate(index):
                if isinstance(selection, slice):
                    shape[axis] = len(range(*selection.indices(shape[axis])))
            return FakeTensor(shape, self.dtype, self.device, True)
        if isinstance(index, slice):
            start, stop, _step = index.indices(self.shape[0])
            return FakeTensor(
                (stop - start,) + self.shape[1:], self.dtype, self.device, True
            )
        raise TypeError(f"FakeTensor only supports slicing, got {index!r}")

    def fill_(self, _value):
        return self

    def to(self, *_args, **_kwargs):
        return self


def _isna(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _make_stub(name, **attrs):
    mod = types.ModuleType(name)
    # Package-looking so `from .sub import x` works for later stubs.
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _empty(shape, dtype=None, device=None, **_kwargs):
    return FakeTensor(shape, dtype, device or "cpu")


def _build_stubs():
    dtypes_mod = _make_stub(
        "aiter.dtypes",
        bf16=_DType("bf16"),
        fp16=_DType("fp16"),
        fp32=_DType("fp32"),
        fp8=_DType("fp8"),
        fp8_e8m0=_DType("e8m0"),
        fp4x2=_DType("fp4x2"),
    )
    torch_mod = _make_stub(
        "torch",
        Tensor=_Tensor,
        dtype=_DType("torch.dtype"),
        uint8=_DType("uint8"),
        float32=_DType("float32"),
        empty=_empty,
        set_printoptions=lambda *a, **k: None,
        random=SimpleNamespace(manual_seed=lambda *a, **k: None),
    )
    pandas_mod = _make_stub("pandas", isna=_isna)
    aiter_mod = _make_stub("aiter", dtypes=dtypes_mod)
    import logging

    aiter_mod.logger = logging.getLogger("aiter")

    def _perftest(*_args, **_kwargs):
        return lambda func: func

    def _compile_ops(*_args, **_kwargs):
        return lambda func: func

    def _torch_compile_guard(*_args, **_kwargs):
        return lambda func: func

    class _StubGemmCommonTuner:
        ARG_DEFAULTS: ClassVar[dict] = {
            "verbose": False,
            "tune_file": "",
            "untune_file": "",
            "errRatio": 0.05,
            "batch": 100,
            "profile_file": "",
            "timeout": None,
            "warmup": 5,
            "iters": 101,
            "min_improvement_pct": 3.0,
            "sort": True,
        }

        def __init__(self, name, key=None, resultList=None, description=None):
            self.name = name
            self.keys = list(key) if key else ["gfx", "cu_num", "M", "N", "K"]
            self.columns = self.keys + list(resultList or [])

        def get_gfx(self):
            return "gfx950"

        def get_cu_num(self):
            return 256

        def _get_run_config_err_ratio_limit(self, _row, _args):
            return 0.05, "0.05"

    stubs = {
        "aiter": aiter_mod,
        "aiter.dtypes": dtypes_mod,
        "torch": torch_mod,
        "pandas": pandas_mod,
        "aiter.jit": _make_stub("aiter.jit"),
        "aiter.jit.core": _make_stub(
            "aiter.jit.core",
            AITER_CONFIG_GEMM_A4W4="",
            AITER_LOG_TUNED_CONFIG=False,
            AITER_CONFIGS=SimpleNamespace(
                AITER_CONFIG_GEMM_A4W4_FILE="",
                AITER_CONFIG_GEMM_A4W4_MXSCALE_BPRESHUFFLE_FILE="",
            ),
            compile_ops=_compile_ops,
            get_asm_dir=lambda: "/nonexistent",
        ),
        "aiter.jit.utils": _make_stub("aiter.jit.utils"),
        "aiter.jit.utils.torch_guard": _make_stub(
            "aiter.jit.utils.torch_guard", torch_compile_guard=_torch_compile_guard
        ),
        "aiter.jit.utils.asm_guard": _make_stub(
            "aiter.jit.utils.asm_guard", require_gfx1250_asm=lambda *a, **k: None
        ),
        "aiter.jit.utils.chip_info": _make_stub(
            "aiter.jit.utils.chip_info",
            get_cu_num=lambda: 256,
            get_gfx=lambda: "gfx950",
            get_gfx_runtime=lambda: "gfx950",
        ),
        "aiter.ops": _make_stub("aiter.ops"),
        "aiter.ops.gemm_op_common": _make_stub(
            "aiter.ops.gemm_op_common", get_padded_m=lambda m, *_args: m
        ),
        "aiter.ops.shuffle": _make_stub(
            "aiter.ops.shuffle", shuffle_weight=lambda *a, **k: None
        ),
        "aiter.test_common": _make_stub(
            "aiter.test_common",
            perftest=_perftest,
            checkAllclose=lambda *a, **k: 0.0,
        ),
        "aiter.utility": _make_stub("aiter.utility", dtypes=dtypes_mod),
        "aiter.utility.fp4_utils": _make_stub("aiter.utility.fp4_utils"),
        "aiter.utility.base_tuner": _make_stub(
            "aiter.utility.base_tuner", GemmCommonTuner=_StubGemmCommonTuner
        ),
        "aiter.utility.mp_tuner": _make_stub(
            "aiter.utility.mp_tuner", mp_tuner=lambda *a, **k: []
        ),
        "gemm_a4w4_blockscale_common": _make_stub(
            "gemm_a4w4_blockscale_common", kernels_list={}
        ),
    }
    stubs["aiter"].compute_gemm_SplitK = lambda *a, **k: 0
    # The stubs are only installed while the two files under test are loaded
    # (and while the tuner runs); leaving them in sys.modules would shadow the
    # real packages for every other test in the session.
    return stubs


_STUB_MODULES = _build_stubs()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return mod


sys.path.insert(0, str(_TUNER_PATH.parent))
with mock.patch.dict(sys.modules, _STUB_MODULES):
    DISPATCH = _load_module("aiter.ops._a4w4_triton_backend_under_test", _DISPATCH_PATH)
    TUNER_MOD = _load_module("a4w4_blockscale_tune_under_test", _TUNER_PATH)

_KERNELS = {
    0: SimpleNamespace(
        name="a4w4_blockscale_fake_kernel",
        MPerBLOCK=128,
        NPerBLOCK=128,
        KPerBLOCK=128,
    )
}


@contextmanager
def _fake_triton_preshuffle(func):
    """Stub the lazy import without initializing a real GPU/extension runtime."""
    names = [
        "aiter.ops.triton",
        "aiter.ops.triton.gemm",
        "aiter.ops.triton.gemm.basic",
        "aiter.ops.triton.gemm.basic.gemm_afp4wfp4",
    ]
    mods = {name: _make_stub(name) for name in names}
    mods[names[-1]].gemm_afp4wfp4_preshuffle = func
    mods[names[-1]].gemm_afp4wfp4 = func
    with mock.patch.dict(sys.modules, mods):
        yield


class _MiniFrame:
    """The small slice of pandas.DataFrame that get_GEMM_config uses."""

    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.columns = list(rows[0]) if rows else []

    def drop_duplicates(self):
        unique = []
        for row in self._rows:
            if row not in unique:
                unique.append(row)
        self._rows = unique
        return self

    def set_index(self, keys):
        self._keys = list(keys)
        return self

    def to_dict(self, _orient):
        if _orient == "records":
            return [dict(row) for row in self._rows]
        return {tuple(row[key] for key in self._keys): dict(row) for row in self._rows}


class TestA4W4TritonDispatch(unittest.TestCase):
    def setUp(self):
        self.mod = DISPATCH
        self.dtypes = self.mod.dtypes
        self.u8 = self.mod.torch.uint8
        self.bf16 = self.dtypes.bf16
        self.calls = []

    # -- helpers ---------------------------------------------------------
    def _operands(self, m=128, n=512, k=7168, scale_rows=None, dtype=None):
        A = FakeTensor((m, k // 2), dtype=self.dtypes.fp4x2)
        B = FakeTensor((n, k // 2), dtype=self.dtypes.fp4x2)
        xs = FakeTensor((scale_rows or (m + 255) // 256 * 256, k // 32), dtype=self.u8)
        ws = FakeTensor(((n + 255) // 256 * 256, k // 32), dtype=self.u8)
        return A, B, xs, ws

    @staticmethod
    def _config(**overrides):
        config = {
            "kernelId": 0,
            "splitK": 0,
            "us": 10.0,
            "kernelName": "a4w4_blockscale_fake_kernel",
            "tflops": 1.0,
            "bw": 1.0,
            "errRatio": 0.0,
        }
        config.update(overrides)
        return config

    def _accessible(self, A, B, xs, ws, **overrides):
        kwargs = {
            "dtype": self.bf16,
            "bpreshuffle": True,
            "bias": None,
            "alpha": 1.0,
            "beta": 0.0,
            "global_A_scale": None,
            "global_B_scale": None,
        }
        kwargs.update(overrides)
        return self.mod._a4w4_triton_accessible(A, B, xs, ws, **kwargs)

    def _dispatch(self, config, **overrides):
        operands = overrides.pop("operands", None) or self._operands()
        with mock.patch.object(
            self.mod, "get_GEMM_config", return_value=config
        ), mock.patch.object(
            self.mod, "get_gfx", return_value="gfx950"
        ), mock.patch.object(
            self.mod, "gemm_a4w4_triton", side_effect=self._on_triton
        ), mock.patch.object(
            self.mod, "gemm_a4w4_blockscale", side_effect=self._on_ck
        ), mock.patch.object(
            self.mod, "gemm_a4w4_asm", side_effect=self._on_asm
        ):
            return self.mod.gemm_a4w4(*operands, **overrides)

    def _on_triton(self, *args, **kwargs):
        self.calls.append(("triton", args, kwargs))
        return "TRITON"

    def _on_ck(self, *args, **kwargs):
        self.calls.append(("ck", args, kwargs))
        return args[4]  # Out

    def _on_asm(self, *args, **kwargs):
        self.calls.append(("asm", args, kwargs))
        return args[4]

    def _backends(self):
        return [call[0] for call in self.calls]

    # -- CSV lookup ------------------------------------------------------
    def _lookup_config(
        self, rows, m=128, n=512, k=7168, bpreshuffle=True, scale_preshuffle=True
    ):
        mod = self.mod
        mod.get_GEMM_config.cache_clear()
        if hasattr(mod.get_GEMM_config, "gemm_dict"):
            del mod.get_GEMM_config.gemm_dict
        try:
            with mock.patch.object(
                mod.pd, "read_csv", return_value=_MiniFrame(rows), create=True
            ), mock.patch.object(
                mod, "get_gfx", return_value="gfx950"
            ), mock.patch.object(
                mod, "get_cu_num", return_value=256
            ):
                return mod.get_GEMM_config(m, n, k, bpreshuffle, scale_preshuffle)
        finally:
            mod.get_GEMM_config.cache_clear()
            if hasattr(mod.get_GEMM_config, "gemm_dict"):
                del mod.get_GEMM_config.gemm_dict

    def test_csv_triton_row_survives_lookup(self):
        config = self._lookup_config(
            [
                {
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "M": 128,
                    "N": 512,
                    "K": 7168,
                    "libtype": "triton",
                    "kernelId": 0,
                    "splitK": 0,
                    "us": 9.0,
                    "kernelName": "triton",
                }
            ]
        )
        self.assertEqual(config["libtype"], "triton")
        self.assertEqual(config["kernelName"], "triton")
        self.assertEqual(self._dispatch(config), "TRITON")

    def test_csv_without_libtype_column_keeps_legacy_route(self):
        # The shipped table has no libtype column yet: nothing may change for it.
        config = self._lookup_config(
            [
                {
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "M": 128,
                    "N": 512,
                    "K": 7168,
                    "kernelId": 0,
                    "splitK": 0,
                    "us": 9.0,
                    "kernelName": "a4w4_blockscale_fake_kernel",
                }
            ]
        )
        self.assertNotIn("libtype", config)
        self._dispatch(config)
        self.assertEqual(self._backends(), ["ck"])

    # -- libtype decoding ------------------------------------------------
    def test_config_libtype_decoding(self):
        decode = self.mod._a4w4_config_libtype
        self.assertEqual(decode(self._config(libtype="triton")), "triton")
        self.assertEqual(decode(self._config(libtype=" ck ")), "ck")
        self.assertEqual(decode(self._config(libtype=float("nan"))), "")
        self.assertEqual(decode(self._config()), "")
        self.assertEqual(decode(None, ""), "")
        # A table whose libtype column was stripped keeps working when the
        # kernelName itself says triton. The same holds after a merged
        # multi-file config backfills the missing column with 0.
        self.assertEqual(decode({"kernelName": "triton"}, "triton"), "triton")
        self.assertEqual(decode({"libtype": 0}, "triton"), "triton")
        self.assertEqual(decode({"libtype": 0.0}, "a4w4_ck_name"), "")

    def test_csv_layouts_never_share_a_winner(self):
        legacy = {
            "gfx": "gfx950",
            "cu_num": 256,
            "M": 128,
            "N": 512,
            "K": 7168,
            "kernelName": "a4w4_ck",
            "splitK": 0,
        }
        raw = dict(
            legacy,
            bpreshuffle=0,
            scale_preshuffle=0,
            kernelName="triton",
            libtype="triton",
        )
        mixed = dict(
            raw, scale_preshuffle=1, kernelName="_ZN_noBpreShuffle_asm", libtype="asm"
        )
        rows = [legacy, raw, mixed]
        self.assertEqual(self._lookup_config(rows)["kernelName"], "a4w4_ck")
        self.assertEqual(
            self._lookup_config(rows, bpreshuffle=False, scale_preshuffle=False)[
                "kernelName"
            ],
            "triton",
        )
        self.assertEqual(self._lookup_config(rows, bpreshuffle=False)["libtype"], "asm")
        self.assertIsNone(self._lookup_config([legacy], bpreshuffle=False))
        self.assertIsNone(self._lookup_config([legacy], scale_preshuffle=False))

    def test_layout_flags_reject_invalid_values(self):
        for value in (None, float("nan"), "", "1", 1, True):
            self.assertEqual(self.mod._a4w4_layout_flag(value), 1)
        for value in (0, False, "0", "0.0"):
            self.assertEqual(self.mod._a4w4_layout_flag(value), 0)
        for value in (2, -1, "raw", 0.5):
            with self.assertRaises(ValueError):
                self.mod._a4w4_layout_flag(value)

    # -- routing ---------------------------------------------------------
    def test_legacy_ck_row_keeps_ck_route(self):
        result = self._dispatch(self._config())
        self.assertEqual(self._backends(), ["ck"])
        kwargs = self.calls[0][2]
        self.assertEqual(kwargs["kernelName"], "a4w4_blockscale_fake_kernel")
        self.assertEqual(kwargs["splitK"], 0)
        self.assertEqual(result.shape, (128, 512))

    def test_triton_row_routes_to_triton(self):
        result = self._dispatch(self._config(libtype="triton", kernelName="triton"))
        self.assertEqual(result, "TRITON")
        self.assertEqual(self._backends(), ["triton"])
        args, kwargs = self.calls[0][1], self.calls[0][2]
        self.assertEqual(args[0].shape, (128, 7168 // 2))
        self.assertEqual(kwargs["dtype"], self.bf16)

    def test_kernelname_triton_without_libtype_routes_to_triton(self):
        result = self._dispatch({"kernelName": "triton"})
        self.assertEqual(result, "TRITON")
        self.assertEqual(self._backends(), ["triton"])

    def test_triton_row_with_non_default_operands_falls_back_to_asm(self):
        A, B, xs, ws = self._operands()
        bias = FakeTensor((1, 512), dtype=self.dtypes.fp32)
        result = self._dispatch(
            self._config(libtype="triton"), operands=(A, B, xs, ws), bias=bias
        )
        self.assertEqual(self._backends(), ["asm"])
        # No untuned CK/asm name is invented either: it runs the default asm.
        self.assertEqual(self.calls[0][1][5], "")
        self.assertEqual(result.shape, (128, 512))

    def test_triton_row_with_fp16_output_falls_back_to_asm(self):
        self._dispatch(self._config(libtype="triton"), dtype=self.dtypes.fp16)
        self.assertEqual(self._backends(), ["asm"])

    def test_triton_row_does_not_accept_preshuffled_activations(self):
        self._dispatch(self._config(libtype="triton"), apreshuffle=True)
        self.assertEqual(self._backends(), ["asm"])

    def test_triton_row_with_unsupported_n_falls_back_to_asm(self):
        A, B, xs, ws = self._operands(n=520)
        self._dispatch(self._config(libtype="triton"), operands=(A, B, xs, ws))
        self.assertEqual(self._backends(), ["asm"])

    # -- shape gate ------------------------------------------------------
    def test_shape_gate(self):
        supported = self.mod.a4w4_triton_shape_supported
        self.assertTrue(supported(32, 512, 7168))
        self.assertTrue(supported(128, 512, 7168))
        self.assertFalse(supported(31, 512, 7168))
        self.assertFalse(supported(128, 520, 7168))
        self.assertFalse(supported(128, 512, 7040))  # K % 256 != 0
        self.assertTrue(supported(128, 512, 6912))
        self.assertTrue(supported(1, 17, 32, bpreshuffle=False))
        self.assertTrue(supported(31, 512, 7040, bpreshuffle=False))
        self.assertFalse(supported(32, 512, 33, bpreshuffle=False))

    # -- operand gate ----------------------------------------------------
    def test_accessible_accepts_default_preshuffled_operands(self):
        self.assertTrue(self._accessible(*self._operands()))

    def test_accessible_rejects_other_contracts(self):
        A, B, xs, ws = self._operands()
        self.assertFalse(self._accessible(A, B, xs, ws, dtype=self.dtypes.fp16))
        self.assertTrue(self._accessible(A, B, xs, ws, bpreshuffle=False))
        self.assertFalse(self._accessible(A, B, xs, ws, bias=FakeTensor((1, 512))))
        self.assertFalse(self._accessible(A, B, xs, ws, alpha=2.0))
        self.assertFalse(self._accessible(A, B, xs, ws, beta=1.0))
        self.assertFalse(
            self._accessible(A, B, xs, ws, global_A_scale=FakeTensor((1,)))
        )
        self.assertFalse(
            self._accessible(A, B, xs, ws, global_B_scale=FakeTensor((1,)))
        )
        # A_scale columns must describe the whole K.
        self.assertFalse(self._accessible(A, B, FakeTensor((256, 200), self.u8), ws))
        # A_scale rows must be whole 32-row stripes.
        self.assertFalse(self._accessible(A, B, FakeTensor((40, 224), self.u8), ws))
        self.assertFalse(self._accessible(A, FakeTensor((512, 1024), self.u8), xs, ws))
        self.assertFalse(
            self._accessible(A, FakeTensor(B.shape, self.u8, contiguous=False), xs, ws)
        )
        self.assertFalse(self._accessible(A, B, FakeTensor((32, 224), self.u8), ws))
        self.assertFalse(self._accessible(A, B, xs, FakeTensor((32, 224), self.u8)))

    # -- scale layout ----------------------------------------------------
    def test_scale_becomes_one_row_of_k_per_32_row_stripe(self):
        scale = FakeTensor((256, 224), dtype=self.u8)
        out = self.mod._a4w4_triton_scale(scale, 7168)
        self.assertEqual(out.shape, (8, 7168))
        self.assertEqual(out.dtype, self.u8)

    def test_scale_accepts_e8m0_and_strided_input(self):
        e8m0 = FakeTensor((512, 224), dtype=self.dtypes.fp8_e8m0)
        out = self.mod._a4w4_triton_scale(e8m0, 7168)
        self.assertEqual(out.dtype, self.u8)
        self.assertEqual(out.shape, (16, 7168))

        strided = FakeTensor((512, 224), dtype=self.dtypes.fp8_e8m0, contiguous=False)
        out = self.mod._a4w4_triton_scale(strided, 7168)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.shape, (16, 7168))

    def test_scale_rejects_mismatched_k_and_rows(self):
        with self.assertRaises(ValueError):
            self.mod._a4w4_triton_scale(FakeTensor((256, 200), self.u8), 7168)
        with self.assertRaises(ValueError):
            self.mod._a4w4_triton_scale(FakeTensor((40, 224), self.u8), 7168)

    # -- kernel wrapper --------------------------------------------------
    def test_gemm_a4w4_triton_reshapes_operands_and_forwards_config(self):
        captured = {}

        def fake_preshuffle(x, w, xs, ws, dtype=None, y=None, config=None, **_kw):
            captured.update(x=x, w=w, xs=xs, ws=ws, dtype=dtype, y=y, config=config)
            if y is not None:
                return y
            return FakeTensor((x.shape[0], w.shape[0] * 16), dtype)

        A, B, xs, ws = self._operands()
        config = {"BLOCK_SIZE_M": 32}
        with _fake_triton_preshuffle(fake_preshuffle):
            y = self.mod.gemm_a4w4_triton(A, B, xs, ws, config=config)

        self.assertEqual(y.shape, (128, 512))
        self.assertEqual(captured["x"].shape, (128, 3584))
        self.assertEqual(captured["w"].shape, (32, 57344))
        self.assertEqual(captured["x"].dtype, self.u8)
        self.assertEqual(captured["w"].dtype, self.u8)
        self.assertEqual(captured["xs"].shape, (8, 7168))
        self.assertEqual(captured["ws"].shape, (16, 7168))
        self.assertIs(captured["config"], config)
        self.assertEqual(captured["y"].shape, (128, 512))
        self.assertEqual(captured["y"].dtype, self.bf16)

    def test_gemm_a4w4_triton_honors_out(self):
        captured = {}

        def fake_preshuffle(_x, _w, _xs, _ws, dtype=None, y=None, **_kw):
            captured["y"] = y
            return y

        A, B, xs, ws = self._operands()
        out = FakeTensor((128, 512), dtype=self.bf16)
        with _fake_triton_preshuffle(fake_preshuffle):
            y = self.mod.gemm_a4w4_triton(A, B, xs, ws, out=out)
        self.assertIs(captured["y"], out)
        self.assertEqual(y.shape, (128, 512))

        bad = FakeTensor((64, 512), dtype=self.bf16)
        with _fake_triton_preshuffle(fake_preshuffle), self.assertRaises(ValueError):
            self.mod.gemm_a4w4_triton(A, B, xs, ws, out=bad)

    def test_gemm_a4w4_triton_rejects_unsupported_m(self):
        A, B, xs, ws = self._operands(m=16)
        with self.assertRaises(ValueError):
            self.mod.gemm_a4w4_triton(A, B, xs, ws)

    def test_raw_triton_uses_raw_bytes_without_scale_conversion(self):
        A, B, _, _ = self._operands(m=1, n=17, k=256)
        xs, ws = FakeTensor((1, 8), self.u8), FakeTensor((17, 8), self.u8)
        captured = {}

        def plain(x, w, sx, sw, dtype=None, y=None, config=None):
            captured.update(x=x, w=w, sx=sx, sw=sw)
            return y

        with _fake_triton_preshuffle(plain):
            result = self.mod.gemm_a4w4_triton(
                A, B, xs, ws, bpreshuffle=False, scale_preshuffle=False
            )
        self.assertEqual(result.shape, (1, 17))
        self.assertEqual(captured["w"].shape, (17, 128))
        self.assertEqual(captured["sx"].shape, (1, 8))
        self.assertEqual(captured["sw"].shape, (17, 8))
        self.assertEqual(captured["w"].dtype, self.u8)

    def test_wrong_ck_layout_is_rejected_before_kernel_launch(self):
        with self.assertRaises(ValueError):
            self._dispatch(self._config(), bpreshuffle=False)
        self.assertEqual(self.calls, [])

    def test_raw_weight_asm_output_has_full_tile_storage(self):
        for m in (32, 128, 257):
            self._dispatch(None, operands=self._operands(m=m), bpreshuffle=False)
            self.assertEqual(self.calls[-1][0], "asm")
            self.assertEqual(self.calls[-1][1][4].shape, ((m + 255) // 256 * 256, 512))

    def test_nonpreshuffled_triton_route_keeps_layout_arguments(self):
        self._dispatch(
            self._config(libtype="triton", kernelName="triton"), bpreshuffle=False
        )
        self.assertEqual(self._backends(), ["triton"])
        self.assertFalse(self.calls[0][2]["bpreshuffle"])
        self.assertTrue(self.calls[0][2]["scale_preshuffle"])


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        return self._rows[index]


class _Frame:
    def __init__(self, rows):
        self._rows = rows
        self.iloc = _Rows(rows)

    def __len__(self):
        return len(self._rows)


class TestA4W4TunerAddsTriton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner_cls = TUNER_MOD.GemmA4W4BlockScaleTuner

    def _make_tuner(self):
        return self.tuner_cls("a4w4-triton-test")

    def _run_tune(self, rows, libtype):
        tuner = self._make_tuner()
        args = SimpleNamespace(
            splitK=False,
            mp=1,
            shape_grouped=False,
            errRatio=0.05,
            timeout=None,
            verbose=False,
            libtype=libtype,
        )
        captured = {"tasks": [], "in_datas": []}

        def fake_mp(tasks, in_datas, *_args, **_kwargs):
            captured["tasks"] = list(tasks)
            captured["in_datas"] = list(in_datas)
            return []

        lazy_env = {**_STUB_MODULES, "aiter.ops.gemm_op_a4w4": DISPATCH}
        with mock.patch.dict(sys.modules, lazy_env), mock.patch.object(
            _STUB_MODULES["aiter.jit.utils.chip_info"],
            "get_gfx_runtime",
            return_value="gfx950",
        ), mock.patch.object(TUNER_MOD, "kernels_list", _KERNELS), mock.patch.object(
            TUNER_MOD, "mp_tuner", fake_mp
        ), mock.patch.object(
            tuner, "get_asm_kernels", return_value={}
        ), mock.patch.object(
            tuner, "get_gfx", return_value="gfx950"
        ), mock.patch.object(
            tuner, "get_cu_num", return_value=256
        ):
            tuner.tune(_Frame(rows), _Frame([]), args)
        return captured

    def test_result_list_records_libtype(self):
        self.assertEqual(self.tuner_cls.RESULT_LIST[0], "libtype")
        tuner = self._make_tuner()
        self.assertIn("libtype", tuner.columns)

    def test_libtype_of_kernel_name(self):
        classify = self.tuner_cls._libtype_of
        self.assertEqual(classify("triton"), "triton")
        self.assertEqual(classify("_ZN5aiter41some_asm_kernel"), "asm")
        self.assertEqual(classify("a4w4_blockscale_256x64x128"), "ck")

    def test_get_kernel_name(self):
        with mock.patch.object(TUNER_MOD, "kernels_list", _KERNELS):
            tuner = self._make_tuner()
            self.assertEqual(tuner.getKernelName(0), "a4w4_blockscale_fake_kernel")
            self.assertEqual(tuner.getKernelName(0, "triton"), "triton")
            self.assertIsNone(tuner.getKernelName(999))

    def test_triton_is_a_candidate_for_supported_shapes(self):
        captured = self._run_tune([{"M": 128, "N": 512, "K": 7168}], "all")
        names = [task[0][3] for task in captured["tasks"]]
        self.assertIn("triton", names)
        self.assertEqual(len(names), 2)  # one CK kernel + triton
        triton_task = captured["tasks"][names.index("triton")]
        info = triton_task[0]
        self.assertEqual(info[0], ("gfx950", 256, 128, 512, 7168, 1, 1))
        self.assertEqual(info[3], "triton")
        self.assertEqual(triton_task[5], {"num_warmup": 10, "num_iters": 101})
        # The triton candidate writes the exact-M output buffer.
        self.assertIn("out_triton", triton_task[4][0])
        self.assertEqual(captured["in_datas"], [(2, ())])

    def test_libtype_triton_alone_tunes_only_triton(self):
        captured = self._run_tune([{"M": 128, "N": 512, "K": 7168}], "triton")
        self.assertEqual([task[0][3] for task in captured["tasks"]], ["triton"])

    def test_raw_layout_candidate_uses_raw_operands_and_excludes_ck(self):
        captured = self._run_tune(
            [{"M": 31, "N": 512, "K": 7040, "bpreshuffle": 0, "scale_preshuffle": 0}],
            "all",
        )
        self.assertEqual([t[0][3] for t in captured["tasks"]], ["triton"])
        task = captured["tasks"][0]
        self.assertEqual(task[0][0][-2:], (0, 0))
        self.assertEqual(
            task[4], (["x", "w", "x_scales", "w_scales", "out_triton"], False, False)
        )

    def test_triton_skipped_for_narrow_m(self):
        captured = self._run_tune([{"M": 16, "N": 512, "K": 7168}], "triton")
        # No candidate at all => mp_tuner is never invoked.
        self.assertEqual(captured["tasks"], [])

    def test_triton_skipped_for_k_not_multiple_of_256(self):
        captured = self._run_tune([{"M": 128, "N": 512, "K": 7040}], "triton")
        self.assertEqual(captured["tasks"], [])


if __name__ == "__main__":
    unittest.main()
