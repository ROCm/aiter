# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-name contract for the FlyDSL a8w8 bpreshuffle dispatch.

No GPU: these exercise the name parser and the tuned CSVs as data, which is
why they live apart from the device-gated tests in
``test_gemm_a8w8_bpreshuffle_pad_k.py``.
"""

import csv
import glob
import os
import re
from unittest import mock

from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
    kernelInstance,
)

# Rows whose kernelName the current catalog cannot build. These predate the
# knob-field schema that `kernelInstance.name` emits and carry a fifth numeric
# token, so both the runtime parser here and the AOT parser in
# `aiter/aot/flydsl/gemm.py` reject them: the shapes run on the CK default
# heuristic and are never AOT-compiled. Regenerating the file with the current
# tuner is what removes this entry -- it is a ratchet, so a *new* dead row in
# any other file fails the test.
KNOWN_UNBUILDABLE_ROWS = {
    "model_configs/"
    "a8w8_bpreshuffle_tuned_gemm_qwen3_5_397b_a17b_mxfp4_attnfp8.csv": 136,
}


def _config_root() -> str:
    import aiter

    return os.path.join(os.path.dirname(os.path.abspath(aiter.__file__)), "configs")


def test_generated_name_round_trips():
    """Every name the tuner emits must parse back to the knobs it encoded."""
    inst = kernelInstance(
        tile_m=32,
        tile_n=64,
        tile_k=512,
        q_dtype_a="fp8",
        q_dtype_w="fp8",
        dtype="bf16",
        use_async_copy=1,
        waves_per_eu=2,
        xcd_swizzle=8,
        lds_stage=2,
    )

    parsed = gemm_mod._parse_flydsl_kernel_name(inst.name)

    assert parsed is not None, f"tuner emitted an unparseable name: {inst.name}"
    tm, tn, tk, acp, wpe, xcd_swizzle, lds_stage, scheduler, k_split = parsed
    assert (tm, tn, tk) == (32, 64, 512)
    assert (acp, wpe, xcd_swizzle, lds_stage) == (1, 2, 8, 2)
    assert scheduler.lower() == "default"
    assert k_split == 1


def test_five_token_knob_field_is_rejected():
    """The knob field carries exactly four tokens.

    `kernelInstance.name` joins use_async_copy, waves_per_eu, xcd_swizzle and
    lds_stage. A fifth token comes from a schema this tree no longer has, so
    the name cannot be mapped onto `flydsl_preshuffle_gemm_a8` arguments and
    must not be silently accepted as something else.
    """
    name = "flydsl_bpreshuflle_32x64x512_F8_F8_B16_2x0x1x3x0_default"

    assert gemm_mod._parse_flydsl_kernel_name(name) is None


def test_unknown_kernel_name_warns_once():
    """An unbuildable name must say so instead of degrading silently.

    Falling back to the CK default heuristic without a message makes a dead
    tuned row look like a bad tuning result.
    """
    name = "flydsl_bpreshuflle_32x64x512_F8_F8_B16_9x9x9x9x9_default"
    gemm_mod._UNKNOWN_FLYDSL_BPRESHUFFLE_WARNED.discard(name)

    # The "aiter" logger does not propagate, so pytest's caplog never sees it.
    with mock.patch.object(gemm_mod.logger, "warning") as warning:
        gemm_mod._warn_unknown_flydsl_bpreshuffle_kernel(name)
        gemm_mod._warn_unknown_flydsl_bpreshuffle_kernel(name)

    assert warning.call_count == 1, "expected exactly one warning per distinct name"
    message = warning.call_args.args[0]
    assert name in message and "not recognized" in message


def test_shipped_tuned_rows_name_buildable_kernels():
    """Ratchet: no tuned CSV may gain a kernelName the catalog cannot build."""
    root = _config_root()
    unbuildable: dict[str, int] = {}
    total = 0

    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        with open(path, newline="", errors="replace") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            name = (row.get("kernelName") or "").strip()
            if not name.startswith("flydsl_bpreshuflle_"):
                continue
            total += 1
            if gemm_mod._parse_flydsl_kernel_name(name) is None:
                rel = os.path.relpath(path, root)
                unbuildable[rel] = unbuildable.get(rel, 0) + 1

    assert total > 0, "no flydsl_bpreshuflle rows found -- is the config root right?"
    assert unbuildable == KNOWN_UNBUILDABLE_ROWS, (
        "set of tuned rows naming unbuildable kernels changed.\n"
        f"  found:    {unbuildable}\n"
        f"  expected: {KNOWN_UNBUILDABLE_ROWS}\n"
        "A new entry means a tuner wrote names this catalog cannot build; a "
        "smaller count means a file was regenerated -- update the constant."
    )


_AOT_KNOB_RE = re.compile(
    r"^flydsl_bpreshuflle_\d+x\d+x\d+_[A-Z0-9]+_[A-Z0-9]+_[A-Z0-9]+_"
    r"(?P<knobs>\d+(?:x\d+)*)_"
)


def test_runtime_and_aot_parsers_agree_on_the_knob_count():
    """The two parsers must accept the same knob-token counts.

    `aiter/aot/flydsl/gemm.py` decides what gets pre-compiled and
    `gemm_op_a8w8.py` decides what runs. A name accepted by one and rejected
    by the other is either a missing AOT kernel or a silent runtime fallback.

    Scoped to the knob field. The two regexes also differ on the scheduler
    token -- optional at runtime, mandatory for AOT -- but no shipped tuned row
    omits it, so that divergence is latent and left alone here.
    """
    from aiter.aot.flydsl.gemm import _parse_preshuffle_kernel_name

    names = [
        "flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x1_default",
        "flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x1x4_default",
        "flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x1x4x2_default",
        "flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x1x4x2_default_ks4",
        "flydsl_bpreshuflle_32x64x512_F8_F8_B16_2x0x1x3x0_default",
    ]

    for name in names:
        runtime_ok = gemm_mod._parse_flydsl_kernel_name(name) is not None
        aot_ok = _parse_preshuffle_kernel_name(name) is not None
        knobs = _AOT_KNOB_RE.match(name).group("knobs")
        assert runtime_ok == aot_ok, (
            f"parsers disagree on {name!r} (knob tokens: {knobs.count('x') + 1}): "
            f"runtime={runtime_ok}, aot={aot_ok}"
        )
