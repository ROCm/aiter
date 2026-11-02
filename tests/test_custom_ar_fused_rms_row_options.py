# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_ar_rms row options reach only the kernels that implement them.

The gfx1250 op takes 11 arguments and has no ``num_norm_rows`` / ``skip_residual``;
the other archs take 13. No GPU or process group is needed: the op is a fake.
"""

import pytest
import torch

from aiter.dist.device_communicators import custom_all_reduce as car
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

ROWS, HIDDEN = 4, 512


class _Pool:
    data_ptr = 0
    max_size = 0


def _make(is_gfx1250):
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca.disabled = True  # the destructor reads it
    ca._is_gfx1250 = is_gfx1250
    ca._ptr = 0
    ca._pool = {"input": _Pool}
    ca.calls = []
    ca._ops_fused_ar_rms = lambda *args: ca.calls.append(args)
    ca.can_fuse_ar_rms = lambda *args: True
    return ca


def _call(ca, **kw):
    inp = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    return ca.fused_ar_rms(
        inp,
        torch.zeros_like(inp),
        w=torch.ones(HIDDEN, dtype=torch.bfloat16),
        eps=1e-6,
        registered=True,
        **kw,
    )


def test_gfx1250_op_gets_its_own_eleven_arguments():
    ca = _make(True)
    _call(ca)
    assert [len(a) for a in ca.calls] == [11]


def test_other_archs_get_the_row_options():
    ca = _make(False)
    _call(ca, num_norm_rows=2, skip_residual=True)
    (args,) = ca.calls
    assert len(args) == 13 and args[-2:] == (2, True)


@pytest.mark.parametrize("opts", [{"skip_residual": True}, {"num_norm_rows": 2}])
def test_gfx1250_rejects_row_options(opts):
    ca = _make(True)
    with pytest.raises(ValueError, match="gfx1250"):
        _call(ca, **opts)
    assert not ca.calls


def test_gfx1250_rejects_a_padded_out(monkeypatch):
    """A caller-provided out wider than the row needs the pad kernel, which gfx1250
    does not have. The capability check must use out's width, not the input's."""
    ca = _make(True)
    del ca.can_fuse_ar_rms  # the real check
    pad_calls = []
    monkeypatch.setattr(
        car.ops, "fused_allreduce_rmsnorm_pad", lambda *a: pad_calls.append(a)
    )
    with pytest.raises(RuntimeError, match="gfx1250"):
        _call(ca, out=torch.empty(ROWS, HIDDEN + 64, dtype=torch.bfloat16))
    assert not ca.calls and not pad_calls
