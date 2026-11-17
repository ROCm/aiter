# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which mqa_logits_mxfp4 launches write lane maxima, asked of a device before any config exists
and of a config -- no GPU.

Only the schedule-free gfx1250 H = 32 kernels write them (gfx1250-strict runs the gfx1250
kernels). The device-level query and MqaLogitsConfig.writes_lane_max must give the same answer
from the same rule, so a caller that sizes the buffer from the query never allocates one no
launch writes, nor skips one a launch would write.
"""

import itertools
from types import SimpleNamespace

import pytest

M = pytest.importorskip("aiter.ops.opus.pa_mqa_logits_mxfp4")

_ARCHES = {"gfx1250": True, "gfx1250-strict": True, "gfx950": False}
_HEADS = {32: True, 64: False}
_CASES = list(itertools.product(_ARCHES, _HEADS))


@pytest.mark.parametrize("arch,heads", _CASES)
def test_device_query(monkeypatch, arch, heads):
    seen = []

    def fake_arch(device):
        seen.append(device)
        return arch

    monkeypatch.setattr(M, "_device_arch", fake_arch)
    expected = _ARCHES[arch] and _HEADS[heads]
    assert M.mqa_logits_mxfp4_writes_lane_max("cuda:3", heads) is expected
    assert seen == ["cuda:3"]


@pytest.mark.parametrize("arch,heads", _CASES)
@pytest.mark.parametrize("mode", [None, M.TABLE, M.FREE, M.PREFILL])
def test_config_agrees_with_device_query(monkeypatch, arch, heads, mode):
    # A config holds the base arch (mqa_logits_mxfp4_config maps gfx1250-strict to gfx1250);
    # in a schedule-free mode it writes lane maxima exactly where the device query says so.
    monkeypatch.setattr(M, "_device_arch", lambda device: arch)
    cfg = SimpleNamespace(_mode=mode, arch=M.base_gfx_arch(arch), heads=heads)
    writes = M.MqaLogitsConfig.writes_lane_max.fget(cfg)
    if mode in (None, M.TABLE):
        assert writes is False
    else:
        assert writes is M.mqa_logits_mxfp4_writes_lane_max(0, heads)
