# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aiter.fused_moe import (
    AUX_SORT_OPUS,
    AUX_SORT_THREESTAGE,
    MOEMetadata,
    _override_output_aux,
)


def _metadata(output_aux):
    return MOEMetadata(
        stage1=None,
        stage2=None,
        block_m=32,
        ksplit=0,
        output_aux=output_aux,
    )


@pytest.mark.parametrize("backend", ["", AUX_SORT_OPUS, AUX_SORT_THREESTAGE])
def test_output_aux_override_keeps_non_aux_config(backend):
    metadata = _metadata(False)

    assert _override_output_aux(metadata, backend) is metadata


def test_empty_output_aux_keeps_configured_backend():
    metadata = _metadata(AUX_SORT_OPUS)

    assert _override_output_aux(metadata, "") is metadata


@pytest.mark.parametrize("backend", [AUX_SORT_OPUS, AUX_SORT_THREESTAGE])
def test_output_aux_override_reselects_supported_backend(backend):
    metadata = _metadata(AUX_SORT_OPUS)

    updated = _override_output_aux(metadata, backend)

    assert updated.output_aux == backend
    assert metadata.output_aux == AUX_SORT_OPUS


def test_output_aux_override_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown output_aux"):
        _override_output_aux(_metadata(False), "unknown")
