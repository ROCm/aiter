"""The host side of the GEMM epilogue's Q/K prep (``qk_prep``): the ops a spec's names
select, the specs it rejects, the gains it folds into the column vectors, its operand
layouts and its kernel-name tags. The epilogue itself is checked on gfx950 by
``op_tests/test_gemm_a8w8.py`` and ``op_tests/test_gemm_a6w6_host.py``.

These run on CPU.
"""

import pytest
import torch

from aiter.ops.flydsl.kernels import qk_prep
from aiter.ops.flydsl.kernels.qk_prep import (
    HEAD_DIM,
    Hadamard,
    PrepOp,
    PrepSpec,
    RMSNorm,
    RoPE,
)

WAN = PrepSpec(norm="rms_across_heads", rope="interleaved", bias=True, hadamard=True)


@pytest.mark.parametrize(
    "spec, ops, suffix",
    [
        (
            WAN,
            (RMSNorm(True, 1e-6), RoPE("interleaved"), Hadamard()),
            "_nacross_rinter_h",
        ),
        (
            PrepSpec("rms_per_head", "rotate_half", False, False, eps=1e-5),
            (RMSNorm(False, 1e-5), RoPE("rotate_half")),
            "_nhead_eps1e-05_rhalf",
        ),
        (PrepSpec("none", "none", True, False), (), ""),
    ],
)
def test_names_select_ops_and_kernel_name_tags(spec, ops, suffix):
    assert spec.ops == ops
    assert spec.suffix == suffix


@pytest.mark.parametrize("norm, rope", [("layer_norm", "none"), ("none", "sideways")])
def test_rejects_unknown_names(norm, rope):
    with pytest.raises(ValueError):
        PrepSpec(norm, rope, False, False)


class _Scale(PrepOp):
    """Not linear, so it cannot follow a row-factor op."""

    @property
    def tag(self):
        return "scale"


def test_rejects_an_op_that_cannot_follow_a_row_factor(monkeypatch):
    monkeypatch.setitem(qk_prep.ROPES, "scale", lambda spec: _Scale())
    PrepSpec("rms_per_head", "scale", False, False)
    with pytest.raises(ValueError):
        PrepSpec("rms_across_heads", "scale", False, False)


def test_a_column_takes_the_gains_of_the_linear_ops_after_it():
    assert WAN.gains() == {"weight": HEAD_DIM**-0.5}
    assert PrepSpec("rms_per_head", "none", False, False).gains() == {"weight": 1.0}


def test_operand_layouts():
    heads, role = 2, 2 * HEAD_DIM
    scales, bias = torch.rand(3 * role), torch.rand(3 * role)
    q, k = torch.rand(HEAD_DIM), torch.rand(role)  # Q's one head's weight, repeated
    gain = HEAD_DIM**-0.5
    columns = WAN.pack_columns(scales, bias, [{"weight": q}, {"weight": k}], heads)
    expected = [scales, bias, q.repeat(heads) * gain, k * gain, torch.ones(role)]
    assert torch.equal(columns, torch.cat(expected))
    cos, sin = torch.rand(3, 64), torch.rand(3, 64)
    assert torch.equal(
        WAN.pack_tables({"cos": cos, "sin": sin}), torch.cat((cos, sin), dim=1)
    )
