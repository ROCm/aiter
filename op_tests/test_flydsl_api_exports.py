# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest


def test_fp4_mqa_lazy_exports():
    pytest.importorskip("flydsl")
    from aiter.ops import flydsl

    names = (
        "flydsl_pa_mqa_logits_fp4",
        "flydsl_pa_mqa_logits_fp4_gfx942",
        "flydsl_pa_mqa_logits_fp4_gfx942_longctx",
        "flydsl_pa_mqa_logits_fp4_prefill",
        "flydsl_pa_mqa_logits_fp4_prefill_gfx942",
        "flydsl_pa_mqa_logits_fp4_varqlen",
        "flydsl_pa_mqa_logits_fp4_varqlen_gfx942",
        "flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942",
        "rowpack_prefill_inputs_gfx942",
        "recommended_gfx942_fp4_longctx_config",
    )
    for name in names:
        assert name in flydsl.__all__
        assert callable(getattr(flydsl, name))
