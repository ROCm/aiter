# SPDX-License-Identifier: MIT
from types import SimpleNamespace as NS

import pytest
import torch

from atom.models.deepseek_v4_mono import config, dispatch
from atom.mono.runtime import deployment


def cfg(profile=config.FLASH):
    hf = NS(
        hidden_size=profile.hidden,
        num_attention_heads=profile.heads,
        q_lora_rank=profile.q_rank,
        n_routed_experts=profile.experts,
        hc_mult=4,
        head_dim=512,
        index_n_heads=64,
        index_head_dim=128,
        num_experts_per_tok=6,
        n_shared_experts=1,
        hc_sinkhorn_iters=20,
    )
    return NS(
        hf_config=hf,
        tensor_parallel_size=profile.tp,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        parallel_config=NS(data_parallel_size=1),
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
        compilation_config=NS(level=0, cudagraph_mode=None),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        speculative_config=None,
    )


@pytest.fixture(autouse=True)
def native(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_DSV4_MONO", "1")
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)


@pytest.mark.parametrize("profile", [config.FLASH, config.PRO])
def test_measured_deployments(profile):
    assert config.config_refusal(cfg(profile), arch="gfx1250:sramecc+") is None


@pytest.mark.parametrize("profile", [config.FLASH, config.PRO])
def test_rows_and_prefill_contract(profile):
    assert config.step_supported(profile.max_rows, profile, prefill=False)
    assert not config.step_supported(profile.max_rows + 1, profile, prefill=False)
    assert not config.step_supported(0, profile, prefill=False)
    assert not config.step_supported(1, profile, prefill=True)
    assert not config.step_supported(1, profile, prefill=False, ubatches=True)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("tensor_parallel_size", 2, "geometries"),
        ("enable_expert_parallel", True, "expert parallel"),
        ("enable_dp_attention", True, "DP attention"),
        ("speculative_config", NS(method="eagle"), "speculative"),
        ("kv_cache_dtype", "fp8", "KV cache"),
        ("enable_tbo", True, "TBO"),
    ],
)
def test_deployment_refusals(field, value, reason):
    c = cfg()
    setattr(c, field, value)
    assert reason in config.config_refusal(c, arch="gfx1250")


def test_wrong_arch_and_geometry():
    assert config.config_refusal(cfg(), arch="gfx950") == "not gfx1250"
    c = cfg()
    c.hf_config.hc_mult = 8
    assert "hc_mult" in config.config_refusal(c, arch="gfx1250")


def test_piecewise_refused():
    c = cfg()
    c.compilation_config.cudagraph_mode = NS(
        requires_piecewise_compilation=lambda: True
    )
    assert "piecewise" in config.config_refusal(c, arch="gfx1250")


def test_disabled_installer_does_not_query_device(monkeypatch):
    monkeypatch.setenv("ATOM_DSV4_MONO", "0")
    model = torch.nn.Identity()
    assert dispatch.install_mono_decode(model, None, None, None) is model


def test_native_forward_and_logits_delegation(monkeypatch):
    class Native(torch.nn.Module):
        marker = "native"
        model = NS(layers=[NS(attn=NS(unified_kv=None))])

        def forward(self, ids, positions):
            return ids + positions

        def compute_logits(self, x):
            return x * 2

    wrapper = dispatch.MonoDecodeModel(Native(), config.FLASH)
    monkeypatch.setattr(
        dispatch, "get_forward_context", lambda: NS(context=NS(is_prefill=True))
    )
    # A prefill must not bind a runner at all.
    wrapper._mono._create = lambda: pytest.fail("prefill created a runner")
    ids = torch.tensor([1, 2])
    positions = torch.tensor([3, 4])
    assert torch.equal(wrapper(ids, positions), ids + positions)
    assert torch.equal(wrapper.compute_logits(ids), ids * 2)
    assert wrapper.marker == "native"


def test_decode_routes_and_publishes_hash_ids(monkeypatch):
    native = torch.nn.Identity()
    native.model = NS(layers=[NS(attn=NS(unified_kv=torch.empty(0)))])
    wrapper = dispatch.MonoDecodeModel(native, config.FLASH)
    state = NS(is_prefill=False, input_ids=None)
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: NS(context=state))
    ids = torch.tensor([4, 7])
    expected = torch.zeros(2, 4, 4096)
    wrapper._mono = NS(
        ready=lambda rows: True, runner=NS(forward=lambda i, p: expected)
    )
    assert wrapper(ids, ids) is expected
    assert torch.equal(state.input_ids, ids)


def test_shipped_gemms_keep_unsplit_accumulation():
    # A real C8 Pro input differed by one BF16 element under split-K.
    # Keep the shipped profiles on the validated single-accumulator order.
    import json
    from pathlib import Path

    root = Path(config.__file__).parent / "configs"
    for name in ("flash_tp1", "pro_tp4"):
        table = json.loads((root / f"{name}.json").read_text())
        assert table
        assert all(value["NUM_KSPLIT"] == 1 for value in table.values())
