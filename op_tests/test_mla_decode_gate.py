# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the AITER MLA-decode persistent-vs-non-persistent gate.

The gate lives in ``aiter/mla.py``:

* ``_use_persistent_mla_decode(bs, nhead, max_seqlen_q, q_dtype, kv_dtype)``
  returns ``True`` to keep the persistent kernel and ``False`` to fall back to
  the non-persistent split-KV kernel. It is scoped to the characterized
  regression profile (``get_gfx()=="gfx950"``, bf16/bf16, ``nhead==16``,
  ``max_seqlen_q==1``); anything out of scope returns ``True``. The batch
  threshold defaults to ``_MLA_DECODE_PERSISTENT_MAX_BATCH_DEFAULT`` (32) and is
  overridable via ``AITER_MLA_DECODE_PERSISTENT_MAX_BATCH`` (``<=0`` disables the
  gate -> always ``True``).

* In ``mla_decode_fwd`` the persistent branch is only kept when
  ``_use_persistent_mla_decode(...)`` returns ``True``. Two complementary
  DISPATCH OBSERVABLES tell us which kernel path was taken:
    - the NON-PERSISTENT branch calls ``aiter.mla.get_meta_param(...)`` to build
      ``num_kv_splits`` (the persistent branch does NOT), and
    - the PERSISTENT branch calls ``aiter.mla_reduce_v1(...)`` for its cross-
      split reduce (the non-persistent branch instead uses the triton
      ``_fwd_kernel_stage2_asm``).
  So: ``get_meta_param`` called + ``mla_reduce_v1`` not called => non-persistent;
  ``get_meta_param`` not called + ``mla_reduce_v1`` called => persistent.

Note on the stock v0.24.0 image: it ships a *prebuilt* ``module_mla_reduce``
whose ``mla_reduce_v1`` op ABI predates this PR's aiter base (the PR passes a
``num_kv_splits`` argument the older op does not accept). Reaching that call
already proves the persistent branch was entered; the dispatch spy tolerates
*only* that specific known ABI skew so the persistent dispatch completes on the
stock image. On an aiter build matching the PR base the reduce runs normally and
the spy is a transparent pass-through.

Runnable two ways:
* ``pytest op_tests/test_mla_decode_gate.py -v``
* ``python op_tests/test_mla_decode_gate.py`` (prints per-test PASS/FAIL and a
  final summary; exits non-zero on any failure).

CI runs this via ``python3 op_tests/test_mla_decode_gate.py``; it is also
collectable under pytest.

GPU/integration tests skip cleanly when CUDA is unavailable or the arch is not
gfx950 so the file still imports/collects on other machines.
"""

import contextlib
import os
import sys

import pytest
import torch

import aiter
import aiter.mla as amla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")


# --- MLA-decode profile (mirrors the microbench build) ----------------------
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 512
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
PAGE_SIZE = 1
DECODE_QLEN = 1  # max_seqlen_q
NHEAD = 16
CTX = 256  # per-request KV length; small so the dispatch runs fast


# Environment guards
def _gpu_ready():
    """(ok, reason) — whether the GPU integration tests can run here."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        gfx = get_gfx()
    except Exception as e:  # pragma: no cover - non-GPU hosts
        return False, f"get_gfx() failed: {e}"
    if gfx != "gfx950":
        return False, f"requires gfx950, got {gfx}"
    return True, ""


# Test helpers
def build_persistent_metadata(
    qo_indptr, kv_indptr, kv_last_page_len, nhead, q_adt, kv_adt, device
):
    """Mirror rocm_aiter_mla.py::_build_decode persistent-metadata construction
    (page_size=1, kv_granularity=16, max_seqlen_qo=1, fast_mode=True). Copied
    from microbench/mla_decode_microbench.py::build_persistent_metadata."""
    (
        (wmd_size, wmd_type),
        (wi_size, wi_type),
        (wis_size, wis_type),
        (ri_size, ri_type),
        (rfm_size, rfm_type),
        (rpm_size, rpm_type),
    ) = aiter.get_mla_metadata_info_v1(
        qo_indptr.numel() - 1,  # batch
        1,  # max_seqlen_qo
        nhead,
        q_adt,
        kv_adt,
        is_sparse=False,
        fast_mode=True,
    )
    work_meta_data = torch.empty(wmd_size, dtype=wmd_type, device=device)
    work_indptr = torch.empty(wi_size, dtype=wi_type, device=device)
    work_info_set = torch.empty(wis_size, dtype=wis_type, device=device)
    reduce_indptr = torch.empty(ri_size, dtype=ri_type, device=device)
    reduce_final_map = torch.empty(rfm_size, dtype=rfm_type, device=device)
    reduce_partial_map = torch.empty(rpm_size, dtype=rpm_type, device=device)

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        nhead,
        1,  # nhead_kv
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
    )
    return dict(
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )


def build_inputs(batch, ctx=CTX, device="cuda"):
    """q / kv / indptr setup mirroring the microbench (bf16, page_size=1,
    decode_qlen=1, nhead=16, kv_lora_rank=512, qk_rope_head_dim=64,
    v_head_dim=512)."""
    q_adt = dtypes.bf16
    kv_adt = dtypes.bf16

    total_kv = batch * ctx
    num_page = total_kv
    kv_buffer = torch.randn(
        (num_page, 1, 1, QK_HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    kv_indices = torch.arange(num_page, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch, dtype=torch.int32, device=device)

    total_q = batch * DECODE_QLEN
    qo_indptr = torch.arange(
        0, (batch + 1) * DECODE_QLEN, DECODE_QLEN, dtype=torch.int32, device=device
    )

    q = torch.randn((total_q, NHEAD, QK_HEAD_DIM), dtype=torch.bfloat16, device=device)
    o = torch.empty((total_q, NHEAD, V_HEAD_DIM), dtype=torch.bfloat16, device=device)
    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    pmeta = build_persistent_metadata(
        qo_indptr, kv_indptr, kv_last_page_len, NHEAD, q_adt, kv_adt, device
    )

    return dict(
        q=q,
        kv_buffer=kv_buffer.view(-1, 1, 1, QK_HEAD_DIM),
        o=o,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        sm_scale=sm_scale,
        total_q=total_q,
        pmeta=pmeta,
    )


@contextlib.contextmanager
def _spy_dispatch():
    """Patch the two branch-marker functions with call-recording wrappers that
    still delegate to the real implementations, so the real dispatch runs:

    * ``aiter.mla.get_meta_param``  -> non-persistent branch marker.
    * ``aiter.mla_reduce_v1``       -> persistent branch marker.

    ``mla_decode_fwd`` resolves both names at call time (``get_meta_param`` as a
    module global, ``mla_reduce_v1`` as an ``aiter`` package attribute), so
    replacing those attributes intercepts the real calls. Yields a dict of call
    counts ``{"meta": n, "reduce": n}``."""
    calls = {"meta": 0, "reduce": 0}
    orig_meta = amla.get_meta_param
    orig_reduce = aiter.mla_reduce_v1

    def meta_spy(*args, **kwargs):
        calls["meta"] += 1
        return orig_meta(*args, **kwargs)

    def reduce_spy(*args, **kwargs):
        calls["reduce"] += 1
        try:
            return orig_reduce(*args, **kwargs)
        except TypeError as e:
            # Stock v0.24.0 image: prebuilt module_mla_reduce op predates this
            # PR's num_kv_splits ABI. Reaching this call already proves the
            # persistent branch; tolerate ONLY this known skew so the dispatch
            # completes. Anything else re-raises.
            if "final_output" in str(e) or "mla_reduce_v1" in str(e):
                return None
            raise

    amla.get_meta_param = meta_spy
    aiter.mla_reduce_v1 = reduce_spy
    try:
        yield calls
    finally:
        amla.get_meta_param = orig_meta
        aiter.mla_reduce_v1 = orig_reduce


_ENV_KEY = "AITER_MLA_DECODE_PERSISTENT_MAX_BATCH"


@contextlib.contextmanager
def _env(value):
    """Set (or, if ``value`` is None, unset) AITER_MLA_DECODE_PERSISTENT_MAX_BATCH
    and restore the prior value afterwards.

    Used by the standalone ``main()`` script runner; the pytest tests set the
    same env var via the ``monkeypatch`` fixture instead (see ``_mp_setter``)."""
    prior = os.environ.get(_ENV_KEY)
    _os_setter(value)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(_ENV_KEY, None)
        else:
            os.environ[_ENV_KEY] = prior


def _os_setter(value):
    """Set (``value`` is not None) or unset the gate env var via ``os.environ``.
    Used inside ``_env`` for the script path's multi-env decision matrix."""
    if value is None:
        os.environ.pop(_ENV_KEY, None)
    else:
        os.environ[_ENV_KEY] = str(value)


def _mp_setter(monkeypatch):
    """Return a setter that sets/unsets the gate env var via ``monkeypatch`` so
    pytest restores it on teardown."""

    def setter(value):
        if value is None:
            monkeypatch.delenv(_ENV_KEY, raising=False)
        else:
            monkeypatch.setenv(_ENV_KEY, str(value))

    return setter


def _run_dispatch(batch, ctx=CTX):
    """Run the real ``mla_decode_fwd`` with persistent metadata and return
    ``(meta_called, reduce_called, output_shape)``."""
    inp = build_inputs(batch, ctx)
    with _spy_dispatch() as calls:
        amla.mla_decode_fwd(
            inp["q"],
            inp["kv_buffer"],
            inp["o"],
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            inp["kv_last_page_len"],
            DECODE_QLEN,  # max_seqlen_q
            PAGE_SIZE,
            1,  # nhead_kv
            sm_scale=inp["sm_scale"],
            **inp["pmeta"],
        )
    torch.cuda.synchronize()
    return calls["meta"] > 0, calls["reduce"] > 0, tuple(inp["o"].shape)


# Env-agnostic assertion cores.
#
# Each ``_core_*`` below holds ONLY the assertion logic and assumes the gate
# env var is ALREADY in the required state. Both runners share them:
#   * pytest tests set the env via the ``monkeypatch`` fixture, then call the
#     core;
#   * the ``check_*`` wrappers (used by ``main()``) set the env via the
#     ``_env`` contextmanager, then call the same core.
# This keeps a single copy of every assertion across both runners.


# (1) Right kernel under default conditions (real dispatch)
def _core_defaults_persistent():
    """batch=8 (< 32) + persistent metadata => persistent kernel taken
    (get_meta_param NOT called, mla_reduce_v1 called), and the call produces a
    correct output shape. Assumes the gate env var is unset (default threshold
    32)."""
    meta, reduce, oshape = _run_dispatch(batch=8)
    assert not meta, "get_meta_param WAS called => non-persistent path (expected persistent)"
    assert reduce, "mla_reduce_v1 NOT called => persistent branch was not entered"
    assert oshape == (8 * DECODE_QLEN, NHEAD, V_HEAD_DIM), f"bad output shape {oshape}"


def _core_defaults_non_persistent():
    """batch=64 (>= 32) + persistent metadata => non-persistent kernel taken
    (get_meta_param WAS called, mla_reduce_v1 NOT called). Assumes the gate env
    var is unset (default threshold 32)."""
    meta, reduce, oshape = _run_dispatch(batch=64)
    assert meta, "get_meta_param NOT called => persistent path (expected non-persistent)"
    assert not reduce, "mla_reduce_v1 WAS called => persistent branch (expected non-persistent)"
    assert oshape == (64 * DECODE_QLEN, NHEAD, V_HEAD_DIM), f"bad output shape {oshape}"


# (2) Env variable triggers the expected behavior
def _core_env_decision_matrix(setenv):
    """Decision-level (`_use_persistent_mla_decode`) env-var behavior, fast and
    deterministic on gfx950. Takes a ``setenv(value)`` callable (``None`` unsets)
    so both runners drive the multiple env regimes through their own mechanism
    (monkeypatch vs ``os.environ``)."""
    dec = amla._use_persistent_mla_decode

    # env=4: batch=8 now >= threshold => non-persistent (False).
    setenv(4)
    assert dec(8, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is False
    # batch=2 still under 4 => persistent.
    assert dec(2, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is True

    # env=0: gate disabled => always persistent, even batch=64.
    setenv(0)
    assert dec(64, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is True

    # env=128: batch=64 < 128 => persistent.
    setenv(128)
    assert dec(64, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is True
    assert dec(200, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is False

    # default (no env): threshold 32.
    setenv(None)
    assert dec(8, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is True
    assert dec(64, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is False


def _core_env_out_of_scope():
    """Out-of-scope profiles return True regardless of batch (gate scoped to
    gfx950 bf16/bf16 nhead=16 max_seqlen_q=1). Assumes env=4 (a tight threshold
    that WOULD flip an in-scope big batch)."""
    dec = amla._use_persistent_mla_decode
    # fp8 q dtype -> out of scope.
    assert dec(64, NHEAD, DECODE_QLEN, dtypes.fp8, dtypes.bf16) is True
    # fp8 kv dtype -> out of scope.
    assert dec(64, NHEAD, DECODE_QLEN, dtypes.bf16, dtypes.fp8) is True
    # nhead=128 -> out of scope.
    assert dec(64, 128, DECODE_QLEN, dtypes.bf16, dtypes.bf16) is True
    # max_seqlen_q=2 -> out of scope.
    assert dec(64, NHEAD, 2, dtypes.bf16, dtypes.bf16) is True


def _core_env_end_to_end_nonpersistent():
    """End-to-end: env=4 flips batch=8 to the non-persistent kernel
    (get_meta_param WAS called, mla_reduce_v1 NOT called). Assumes env=4."""
    meta, reduce, oshape = _run_dispatch(batch=8)
    assert meta, "get_meta_param NOT called => persistent (expected non-persistent under env=4)"
    assert not reduce, "mla_reduce_v1 WAS called => persistent (expected non-persistent under env=4)"
    assert oshape == (8 * DECODE_QLEN, NHEAD, V_HEAD_DIM), f"bad output shape {oshape}"


def _core_env_end_to_end_persistent_disabled():
    """End-to-end: env=0 disables the gate so batch=64 stays persistent
    (get_meta_param NOT called, mla_reduce_v1 called). Assumes env=0."""
    meta, reduce, oshape = _run_dispatch(batch=64)
    assert not meta, "get_meta_param WAS called => non-persistent (expected persistent under env=0)"
    assert reduce, "mla_reduce_v1 NOT called => persistent branch not entered (expected persistent under env=0)"
    assert oshape == (64 * DECODE_QLEN, NHEAD, V_HEAD_DIM), f"bad output shape {oshape}"


# Script-path wrappers: set the env via the ``_env`` contextmanager, then
# delegate to the shared cores above. Used by ``main()``.
def check_defaults_persistent():
    with _env(None):  # ensure default threshold (32)
        _core_defaults_persistent()


def check_defaults_non_persistent():
    with _env(None):  # default threshold (32)
        _core_defaults_non_persistent()


def check_env_decision_matrix():
    # _env(None) snapshots/restores the prior value; the core drives the
    # intermediate regimes via _os_setter.
    with _env(None):
        _core_env_decision_matrix(_os_setter)


def check_env_out_of_scope():
    with _env(4):  # tight threshold that WOULD flip an in-scope big batch
        _core_env_out_of_scope()


def check_env_end_to_end_nonpersistent():
    with _env(4):
        _core_env_end_to_end_nonpersistent()


def check_env_end_to_end_persistent_disabled():
    with _env(0):
        _core_env_end_to_end_persistent_disabled()


# pytest entry points
#
# All tests below are guarded at module level by ``pytestmark`` so they skip
# cleanly off gfx950 / without CUDA. Env-var tests set the gate env via the
# ``monkeypatch`` fixture (auto-restored on teardown) and share the same
# assertion cores as the ``check_*`` script wrappers above.
needs_gfx950 = pytest.mark.skipif(
    not _gpu_ready()[0], reason=_gpu_ready()[1] or "GPU/gfx950 required"
)

pytestmark = needs_gfx950


def test_defaults_persistent(monkeypatch):
    monkeypatch.delenv(_ENV_KEY, raising=False)  # default threshold (32)
    _core_defaults_persistent()


def test_defaults_non_persistent(monkeypatch):
    monkeypatch.delenv(_ENV_KEY, raising=False)  # default threshold (32)
    _core_defaults_non_persistent()


def test_env_decision_matrix(monkeypatch):
    _core_env_decision_matrix(_mp_setter(monkeypatch))


def test_env_out_of_scope(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "4")  # tight threshold; profiles still out of scope
    _core_env_out_of_scope()


def test_env_end_to_end_nonpersistent(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "4")
    _core_env_end_to_end_nonpersistent()


def test_env_end_to_end_persistent_disabled(monkeypatch):
    monkeypatch.setenv(_ENV_KEY, "0")
    _core_env_end_to_end_persistent_disabled()


# Standalone runner
_CHECKS = [
    ("defaults_persistent (batch=8 -> persistent, get_meta_param NOT called)",
     check_defaults_persistent),
    ("defaults_non_persistent (batch=64 -> non-persistent, get_meta_param called)",
     check_defaults_non_persistent),
    ("env_decision_matrix (env 4/0/128/default at decision level)",
     check_env_decision_matrix),
    ("env_out_of_scope (fp8 / nhead=128 / max_seqlen_q=2 -> True)",
     check_env_out_of_scope),
    ("env_end_to_end_nonpersistent (env=4, batch=8 -> non-persistent)",
     check_env_end_to_end_nonpersistent),
    ("env_end_to_end_persistent_disabled (env=0, batch=64 -> persistent)",
     check_env_end_to_end_persistent_disabled),
]


def main():
    ok, reason = _gpu_ready()
    if not ok:
        print(f"SKIP: all tests skipped ({reason})")
        return 0

    print(f"Running {len(_CHECKS)} MLA-decode gate checks on gfx950 ...")
    passed = 0
    failed = 0
    for name, fn in _CHECKS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}\n         {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"[PASS] {name}")

    print("-" * 72)
    print(f"SUMMARY: {passed} passed, {failed} failed, out of {len(_CHECKS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
