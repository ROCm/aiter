# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decision-level test for the MLA-decode persistent-vs-non-persistent gate.

``mla_decode_fwd`` chooses the kernel purely via
``persistent_mode = _use_persistent_mla_decode(bs, nhead, max_seqlen_q,
q_dtype, kv_dtype)`` (in ``aiter/mla.py``), so testing that predicate directly
deterministically covers which kernel is selected -- no GPU metadata, no
dispatch spies. The gate only differentiates on the characterized gfx950
bf16/bf16 nhead=16 qseqlen=1 profile; anything out of scope returns True.

CI runs this via ``python3 op_tests/test_mla_decode_gate.py`` (also
pytest-collectable).
"""

import os

import pytest

from aiter import dtypes
from aiter.mla import _persistent_mla_decode_max_batch, _use_persistent_mla_decode
from aiter.jit.utils.chip_info import get_gfx

try:
    from unittest.mock import patch
except ImportError:  # pragma: no cover
    from mock import patch

bf16 = dtypes.bf16
fp8 = dtypes.fp8


def _is_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="gate only differentiates on gfx950"
)


@pytest.fixture(autouse=True)
def _reset_max_batch_cache():
    # The env read is memoized (lru_cache) so it costs nothing on the hot path;
    # clear it around each test so per-test AITER_MLA_DECODE_PERSISTENT_MAX_BATCH
    # overrides are actually observed instead of a stale first read.
    _persistent_mla_decode_max_batch.cache_clear()
    yield
    _persistent_mla_decode_max_batch.cache_clear()


def test_defaults():
    assert _use_persistent_mla_decode(8, 16, 1, bf16, bf16) is True
    assert _use_persistent_mla_decode(64, 16, 1, bf16, bf16) is False


def test_env_lowers_threshold():
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "4"}):
        assert _use_persistent_mla_decode(8, 16, 1, bf16, bf16) is False
        assert _use_persistent_mla_decode(2, 16, 1, bf16, bf16) is True


def test_env_disabled():
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "0"}):
        assert _use_persistent_mla_decode(64, 16, 1, bf16, bf16) is True


def test_env_raises_threshold():
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "128"}):
        assert _use_persistent_mla_decode(64, 16, 1, bf16, bf16) is True
        assert _use_persistent_mla_decode(200, 16, 1, bf16, bf16) is False


def test_out_of_scope():
    # A tight threshold that WOULD flip an in-scope big batch to non-persistent.
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "4"}):
        assert _use_persistent_mla_decode(64, 16, 1, fp8, bf16) is True
        assert _use_persistent_mla_decode(64, 16, 1, bf16, fp8) is True
        assert _use_persistent_mla_decode(64, 128, 1, bf16, bf16) is True
        assert _use_persistent_mla_decode(64, 16, 2, bf16, bf16) is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
