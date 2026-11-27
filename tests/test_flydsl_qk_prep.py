"""The host side of the GEMM epilogue's Q/K prep (``qk_prep``): the ops a spec's names
select, the specs it rejects, the gains it folds into the column vectors, its operand
layouts, its Hadamard and its kernel-name tags. The epilogue itself is checked on gfx950
by ``op_tests/test_gemm_a8w8_int8_4wave_epilogue.py`` and
``op_tests/test_gemm_a6w6_host.py``.

These run on CPU.
"""

import re

import pytest
import torch

from aiter.ops.flydsl.kernels import qk_prep
from aiter.ops.flydsl.kernels.gemm_hooks import GemmHooks, name_suffix
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
            "_nhead_eps1em05_rhalf",
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

    def apply(self, x, ctx, state, data):
        return x

    def reference(self, y, params, tables):
        return y, None

    @property
    def tag(self):
        return "scale"


def test_an_op_implements_the_interface():
    with pytest.raises(TypeError):
        PrepOp()


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
    roles = [{"weight": q}, {"weight": k}]
    columns = WAN.pack_columns(scales, bias, roles, heads)
    expected = [scales, bias, q.repeat(heads) * gain, k * gain, torch.ones(role)]
    assert torch.equal(columns, torch.cat(expected))
    # A host that applies no column scales.
    columns = WAN.pack_columns(None, bias, roles, heads, n=3 * role)
    assert torch.equal(columns, torch.cat(expected[1:]))
    cos, sin = torch.rand(3, 64), torch.rand(3, 64)
    assert torch.equal(
        WAN.pack_tables({"cos": cos, "sin": sin}), torch.cat((cos, sin), dim=1)
    )


def test_hadamard_is_orthogonal_in_its_own_row_order():
    # Row x of H is the transform of the basis vector e_x.
    basis = torch.eye(HEAD_DIM).reshape(HEAD_DIM, 1, HEAD_DIM)
    h = Hadamard().reference(basis, {}, {})[0].reshape(HEAD_DIM, HEAD_DIM)
    assert torch.equal(h @ h.T, HEAD_DIM * torch.eye(HEAD_DIM))
    bits = torch.arange(HEAD_DIM)
    parity = torch.tensor([v.bit_count() % 2 for v in range(HEAD_DIM)])
    sylvester = 1.0 - 2.0 * parity[bits[:, None] & bits[None, :]].float()
    assert not torch.equal(h, sylvester)
    # Each row is one of Sylvester's, permuted and with 32 of them negated.
    match = h @ sylvester.T / HEAD_DIM
    assert torch.equal(match.abs().sum(1), torch.ones(HEAD_DIM))
    assert torch.equal(match.abs().sum(0), torch.ones(HEAD_DIM))
    assert int((match.sum(1) < 0).sum()) == 32


@pytest.mark.parametrize("eps", [1e-5, 1e-7, 1.5e-5, 1e-4, 0.25])
def test_kernel_name_parts_are_symbols(eps):
    tag = RMSNorm(True, eps).tag
    assert re.fullmatch(r"\w+", tag), tag
    assert tag != RMSNorm(True, 1e-6).tag
    hooks = GemmHooks(role_end=lambda *args: None, key="Q/K-v.2")
    suffix = name_suffix(roles=3, role_cols=5120, hooks=hooks, col_amax_offset=None)
    assert suffix == "_roles3x5120_hookQ_K_v_2"
