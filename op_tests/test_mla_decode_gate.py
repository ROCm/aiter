# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decision-level tests for MLA-decode kernel selection and split-count metadata.

Two independent decisions, both cheap to test directly:

1. ``mla_decode_fwd`` chooses the kernel purely via
   ``persistent_mode = _use_persistent_mla_decode(bs, nhead, max_seqlen_q,
   q_dtype, kv_dtype)`` (in ``aiter/mla.py``), so testing that predicate
   directly deterministically covers which kernel is selected -- no GPU
   metadata, no dispatch spies. The gate only differentiates on the
   characterized gfx950 bf16/bf16 nhead=16 qseqlen=1 profile; anything out of
   scope returns True.
2. On the non-persistent side ``get_meta_param`` resolves the split count, and
   its fp8 branch keys ``get_block_n_fp8`` by the FOLDED query width
   ``nhead * max_seqlen_q``. See the second test group.

CI runs this via ``python3 op_tests/test_mla_decode_gate.py`` (also
pytest-collectable).
"""

import os

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import (
    _persistent_mla_decode_max_batch,
    _use_persistent_mla_decode,
    get_meta_param,
)

try:
    from unittest.mock import patch
except ImportError:  # pragma: no cover
    from unittest.mock import patch

bf16 = dtypes.bf16
fp8 = dtypes.fp8


def _is_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:  # noqa: BLE001
        return False


# Scoped to the gate tests rather than module-level: the get_meta_param group
# below is arch-independent pure-Python table logic and must also run on gfx942.
gate_only = pytest.mark.skipif(
    not _is_gfx950(), reason="gate only differentiates on gfx950"
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="get_meta_param allocates the split indptr on cuda",
)


@pytest.fixture(autouse=True)
def _reset_max_batch_cache():
    # The env read is memoized (lru_cache) so it costs nothing on the hot path;
    # clear it around each test so per-test AITER_MLA_DECODE_PERSISTENT_MAX_BATCH
    # overrides are actually observed instead of a stale first read.
    _persistent_mla_decode_max_batch.cache_clear()
    yield
    _persistent_mla_decode_max_batch.cache_clear()


@gate_only
def test_defaults():
    assert _use_persistent_mla_decode(8, 16, 1, bf16, bf16) is True
    assert _use_persistent_mla_decode(64, 16, 1, bf16, bf16) is False


@gate_only
def test_env_lowers_threshold():
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "4"}):
        assert _use_persistent_mla_decode(8, 16, 1, bf16, bf16) is False
        assert _use_persistent_mla_decode(2, 16, 1, bf16, bf16) is True


@gate_only
def test_env_disabled():
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "0"}):
        assert _use_persistent_mla_decode(64, 16, 1, bf16, bf16) is True


@gate_only
def test_env_raises_threshold():
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "128"}):
        assert _use_persistent_mla_decode(64, 16, 1, bf16, bf16) is True
        assert _use_persistent_mla_decode(200, 16, 1, bf16, bf16) is False


@gate_only
def test_out_of_scope():
    # A tight threshold that WOULD flip an in-scope big batch to non-persistent.
    with patch.dict(os.environ, {"AITER_MLA_DECODE_PERSISTENT_MAX_BATCH": "4"}):
        assert _use_persistent_mla_decode(64, 16, 1, fp8, bf16) is True
        assert _use_persistent_mla_decode(64, 16, 1, bf16, fp8) is True
        assert _use_persistent_mla_decode(64, 128, 1, bf16, bf16) is True
        assert _use_persistent_mla_decode(64, 16, 2, bf16, bf16) is True


# ---------------------------------------------------------------------------
# get_meta_param: the fp8 folded-query-width lookup into get_block_n_fp8.
#
# The fp8 branch keys the table by `nhead * max_seqlen_q` -- the FOLDED query
# width, not nhead -- and used to index it directly. Plain decode has
# max_seqlen_q == 1, so the key is just nhead and every listed entry is hit;
# max_seqlen_q > 1 multiplies the key and can leave the table entirely.
#
# That is reachable on a supported path. get_meta_param is only called from the
# NON-persistent branch of mla_decode_fwd, and for gqa_ratio=128 asm_mla.cu sets
# config_max_seqlen_q = 0 and asserts nothing -- the shipped
# `fp8,fp8,Gqa=128,ps=0,qSeqLen=0` kernel is qlen-agnostic. Widths 128/256/384/512
# (qlen 1..4) are listed; qlen >= 5 is not, and raised KeyError in Python before
# reaching a kernel that would have run.
#
# Scope note: these cover the split-count metadata only. Whether the asm kernel
# *supports* a given qlen is a separate constraint in asm_mla.cu -- fp8 gqa16
# needs persistent mode on gfx950 for qlen > 4 -- so a passing test here is not
# a claim that any qlen runs end-to-end on the non-persistent path.
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("max_seqlen_q", [1, 2, 3, 4, 5, 6, 7, 8])
def test_get_meta_param_fp8_unlisted_width_on_qlen_agnostic_kernel(max_seqlen_q):
    """nhead=128: qlen 1..4 are listed widths, qlen >= 5 (640+) are not.

    The gqa=128 ps=0 kernel takes any max_seqlen_q, so every one of these must
    resolve a block size rather than raise.
    """
    num_kv_splits, split_indptr = get_meta_param(None, 1, 4096, 128, max_seqlen_q, fp8)
    assert num_kv_splits >= 1
    assert split_indptr.numel() == 2  # bs + 1


@requires_cuda
def test_get_meta_param_fp8_arbitrary_unlisted_width_falls_back():
    """Any width with no table entry (nhead=16 x qlen=9 -> 144) must not raise."""
    num_kv_splits, _ = get_meta_param(None, 1, 1024, 16, 9, fp8)
    assert num_kv_splits >= 1


@requires_cuda
def test_get_meta_param_fp8_block_cap_still_bites():
    """The width -> block_n cap must still clamp, not just avoid the KeyError.

    bs=1, total_kv=128, unlisted width 640 -> fallback is the widest listed
    width's block (512 -> 32) -> ceil(128/32) = 4 splits. A fallback that
    silently disabled the cap would let this exceed 4.
    """
    num_kv_splits, _ = get_meta_param(None, 1, 128, 128, 5, fp8)
    assert 1 <= num_kv_splits <= 4


@requires_cuda
def test_get_meta_param_fp8_listed_width_unchanged():
    """The fallback must not perturb a width the table does list.

    bs=1, total_kv=1024, width 512 -> min_block_n 32 -> ceil(1024/32) = 32,
    which is above the search's 16-split ceiling, so the cap does not bind.
    """
    num_kv_splits, _ = get_meta_param(None, 1, 1024, 128, 4, fp8)
    assert num_kv_splits >= 1


@requires_cuda
def test_get_meta_param_bf16_never_consults_the_fp8_table():
    """Non-fp8 skips the table entirely, so unlisted widths were always fine."""
    num_kv_splits, _ = get_meta_param(None, 1, 1024, 128, 5, bf16)
    assert num_kv_splits >= 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
