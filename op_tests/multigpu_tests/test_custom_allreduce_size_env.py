# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check how AITER_CUSTOM_{AR,AG,RS}_MAX_SIZE size the custom allreduce input
pool and route each collective between the custom kernels and RCCL.

Single process, no process group: the resolver is a pure function and the size
gates only read attributes the constructor sets.

Run with: python3 op_tests/multigpu_tests/test_custom_allreduce_size_env.py
      or: pytest op_tests/multigpu_tests/test_custom_allreduce_size_env.py
"""

import os
from contextlib import contextmanager

import torch

from aiter.dist.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    _resolve_custom_sizes,
)

MiB = 1 << 20
GiB = 1 << 30
ENV_VARS = (
    "AITER_CUSTOM_AR_MAX_SIZE",
    "AITER_CUSTOM_AG_MAX_SIZE",
    "AITER_CUSTOM_RS_MAX_SIZE",
    "AITER_CUSTOM_AR_MIN_SIZE",
)


@contextmanager
def size_env(**values):
    """Set the given AITER_CUSTOM_* vars (AR_MAX="..." etc.) and clear the rest."""
    saved = {k: os.environ.get(k) for k in ENV_VARS}
    try:
        for k in ENV_VARS:
            os.environ.pop(k, None)
        for short, v in values.items():
            os.environ[f"AITER_CUSTOM_{short}_SIZE"] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def resolve(gfx1250, ws=4, max_size=GiB, **env):
    with size_env(**env):
        return _resolve_custom_sizes(gfx1250, ws, max_size)


def test_defaults_unchanged():
    for gfx1250 in (True, False):
        s = resolve(gfx1250)
        assert s.pool == GiB, s
        assert s.ar == 64 * MiB and s.ar_prefill == 64 * MiB, s
        assert s.rs == 64 * MiB, s
        # max_size / (2 * world_size), the cap should_custom_ag always used.
        assert s.ag == GiB // 8, s
    assert resolve(True, ws=8).ag == 64 * MiB
    assert resolve(True, ws=2).ag == 256 * MiB
    # A small constructor pool (test_sp_head_exchange passes 64 MiB) is kept,
    # and the defaults are capped to it.
    s = resolve(True, max_size=64 * MiB)
    assert s.pool == 64 * MiB and s.ag == 8 * MiB, s
    s = resolve(False, max_size=16 * MiB)
    assert s.pool == 16 * MiB and s.ar == 16 * MiB and s.ar_prefill == 8 * MiB, s


def test_gfx1250_pool_grows_to_largest_set_cutoff():
    s = resolve(True, AG_MAX=2 * GiB)
    assert s.pool == 2 * GiB and s.ag == 2 * GiB, s
    # The other ops keep their own defaults.
    assert s.ar == 64 * MiB and s.rs == 64 * MiB, s

    rs_max = 3 * GiB + 512 * MiB
    s = resolve(True, AR_MAX=3 * GiB, AG_MAX=2 * GiB, RS_MAX=rs_max)
    assert (s.pool, s.ar, s.ag, s.rs) == (rs_max, 3 * GiB, 2 * GiB, rs_max), s

    # Rounded up to 2 MiB.
    s = resolve(True, RS_MAX=2 * GiB + 1)
    assert s.pool == 2 * GiB + 2 * MiB and s.rs == 2 * GiB + 1, s


def test_gfx1250_pool_never_shrinks():
    s = resolve(True, AR_MAX=16 * MiB, AG_MAX=16 * MiB, RS_MAX=16 * MiB)
    assert s.pool == GiB, s
    # 0 disables an op and does not size the pool.
    s = resolve(True, AR_MAX=0)
    assert s.pool == GiB and s.ar == 0 and s.ar_prefill == 0, s


def test_gfx1250_prefill_not_halved():
    s = resolve(True, AR_MAX=2 * GiB)
    assert s.pool == 2 * GiB and s.ar == 2 * GiB and s.ar_prefill == 2 * GiB, s


INT32_MAX = (1 << 31) - 1


def test_gfx1250_caps_at_kernel_index_range():
    # AG: the output in 16-byte packs must fit int32, per rank
    # 16 * (INT32_MAX // world_size) bytes, whatever the dtype.
    for ws in (2, 4, 8):
        s = resolve(True, ws=ws, AG_MAX=64 * GiB)
        cap = 16 * (INT32_MAX // ws)
        assert s.ag == cap, (ws, s)
        assert (cap // 16) * ws <= INT32_MAX < (cap // 16 + 1) * ws, ws
        assert s.pool == (cap + 2 * MiB - 1) // (2 * MiB) * (2 * MiB), (ws, s)
    # AR / RS: the element count must fit int32 for 2-byte dtypes.
    s = resolve(True, AR_MAX=8 * GiB, RS_MAX=8 * GiB)
    assert s.ar == s.ar_prefill == s.rs == 2 * INT32_MAX, s
    assert s.pool == 4 * GiB, s
    # Below the caps nothing changes.
    s = resolve(True, ws=8, AG_MAX=2 * GiB, AR_MAX=3 * GiB)
    assert s.ag == 2 * GiB and s.ar == 3 * GiB, s
    # gfx9 keeps its pool cap and is not affected.
    s = resolve(False, ws=8, AG_MAX=64 * GiB)
    assert s.pool == GiB and s.ag == GiB, s


def test_gfx1250_ag_gate_takes_large_bf16():
    # 1.25 GiB of bf16 a rank at world_size 4 is 2.68e9 output elements, past
    # int32, but only 3.4e8 16-byte packs, which the kernels index. The gate is
    # in bytes and admits it.
    ca = _stub(AG_MAX=2 * GiB)
    inp = torch.empty(5 * GiB // 8, dtype=torch.bfloat16)
    assert inp.numel() * 4 > INT32_MAX
    assert ca.should_custom_ag(inp)


def test_gfx9_pool_fixed_and_cutoffs_capped():
    s = resolve(False, AR_MAX=4 * GiB, AG_MAX=4 * GiB, RS_MAX=4 * GiB)
    assert s.pool == GiB, s
    assert s.ar == GiB and s.ag == GiB and s.rs == GiB, s
    # 2-stage write mode needs the 2x tmp region: prefill stays at half.
    assert s.ar_prefill == GiB // 2, s

    s = resolve(False, AR_MAX=256 * MiB)
    assert s.ar == 256 * MiB and s.ar_prefill == 256 * MiB, s


def test_ops_are_independent():
    # RS does not follow AR, in either direction.
    s = resolve(True, AR_MAX=0)
    assert s.rs == 64 * MiB, s
    s = resolve(True, AR_MAX=512 * MiB)
    assert s.rs == 64 * MiB and s.ag == GiB // 8, s
    s = resolve(True, RS_MAX=0)
    assert s.ar == 64 * MiB and s.rs == 0, s


def test_bad_values_fall_back_to_defaults():
    s = resolve(True, AR_MAX="abc", AG_MAX="-5", RS_MAX="1.5e9")
    assert s.ar == 64 * MiB and s.ag == GiB // 8 and s.rs == 64 * MiB, s
    assert s.pool == GiB, s


def _stub(gfx1250=True, ws=4, min_size=0, **env):
    """A CustomAllreduce with only what the size gates read."""
    ca = CustomAllreduce.__new__(CustomAllreduce)
    s = resolve(gfx1250, ws=ws, **env)
    ca.disabled = False
    ca._is_gfx1250 = gfx1250
    ca.world_size = ws
    ca.fully_connected = True
    ca.max_size = s.pool
    ca._car_max_size = s.ar
    ca._car_prefill_max_size = s.ar_prefill
    ca._cag_max_size = s.ag
    ca._crs_max_size = s.rs
    ca._car_min_size = min_size
    return ca


def _t(nbytes):
    # CPU tensors: the gates only look at size and layout. A stride-0 view
    # stands in for a big input without allocating it.
    return torch.empty(1, dtype=torch.uint8).expand(nbytes)


def test_gates_use_their_own_cutoff():
    ca = _stub(AR_MAX=32 * MiB, AG_MAX=16 * MiB, RS_MAX=128 * MiB)

    def u8(n):
        return torch.empty(n, dtype=torch.uint8)

    assert ca.should_custom_ar(u8(32 * MiB))
    assert not ca.should_custom_ar(u8(32 * MiB + 16))
    assert ca.should_custom_rs(u8(128 * MiB), 0)
    assert not ca.should_custom_rs(u8(128 * MiB + 64), 0)
    assert ca.should_custom_ag(u8(16 * MiB))
    assert not ca.should_custom_ag(u8(16 * MiB + 16))


def test_zero_disables_only_that_op():
    small = torch.empty(4 * MiB, dtype=torch.uint8)
    ca = _stub(AR_MAX=0)
    assert not ca.should_custom_ar(small)
    assert ca.should_custom_rs(small, 0) and ca.should_custom_ag(small)
    ca = _stub(RS_MAX=0)
    assert not ca.should_custom_rs(small, 0)
    assert ca.should_custom_ar(small) and ca.should_custom_ag(small)
    ca = _stub(AG_MAX=0)
    assert not ca.should_custom_ag(small)
    assert ca.should_custom_ar(small) and ca.should_custom_rs(small, 0)


def test_ar_min_applies_to_ar_only():
    small = torch.empty(1 * MiB, dtype=torch.uint8)
    ca = _stub(min_size=2 * MiB)
    assert not ca.should_custom_ar(small)
    assert ca.should_custom_rs(small, 0) and ca.should_custom_ag(small)


def test_gfx9_prefill_gate():
    ca = _stub(gfx1250=False, AR_MAX=4 * GiB)
    big = _t(768 * MiB)
    assert ca._fits_custom_ar_size(big)
    assert not ca._fits_custom_ar_size(big, prefill_support=True)
    assert not ca._fits_custom_ar_size(_t(GiB + 16))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
