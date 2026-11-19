import importlib

import torch


def test_add_dynamicquant_forwards_gemma_norm_to_hip(monkeypatch):
    rmsnorm = importlib.import_module("aiter.ops.rmsnorm")
    called = {}

    def fake_add_rmsnorm_quant(*args, **kwargs):
        called["args"] = args
        called["gemma_norm"] = kwargs.get("gemma_norm")

    monkeypatch.setattr(rmsnorm, "add_rmsnorm_quant", fake_add_rmsnorm_quant)
    shape = (2, 8)
    x = torch.empty(shape, dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    out = torch.empty(shape, dtype=torch.float8_e4m3fnuz)
    scale = torch.empty((shape[0], 1), dtype=torch.float32)
    weight = torch.empty(shape[-1], dtype=torch.bfloat16)

    rmsnorm.rmsnorm2d_fwd_with_add_dynamicquant(
        out,
        x,
        residual,
        residual_out,
        scale,
        weight,
        1e-6,
        gemma_norm=True,
    )

    assert called["gemma_norm"] is True
    assert called["args"][:7] == (
        out,
        x,
        residual,
        residual_out,
        scale,
        weight,
        1e-6,
    )


def test_gfx1250_gemma_quant_fallback_uses_fused_norm_quant(monkeypatch):
    import aiter

    communicator = importlib.import_module(
        "aiter.dist.device_communicators.communicator_cuda"
    )
    called = {}

    def fake_fused_norm_quant(*args, **kwargs):
        called["args"] = args
        called["gemma_norm"] = kwargs.get("gemma_norm")

    monkeypatch.setattr(communicator, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(
        aiter, "rmsnorm2d_fwd_with_add_dynamicquant", fake_fused_norm_quant
    )

    class FakeCommunicator:
        ca_comm = None
        _ar_quant_max_bytes = -1
        _ar_quant_no_prefill_max_bytes = -1

        def all_reduce(self, input_, **kwargs):
            called["all_reduce_kwargs"] = kwargs
            return input_.clone()

    shape = (2, 6144)
    x = torch.empty(shape, dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    weight = torch.empty(shape[-1], dtype=torch.bfloat16)
    out, residual_out, scale = (
        communicator.CudaCommunicator.fused_allreduce_rmsnorm_quant(
            FakeCommunicator(), x, residual, weight, 1e-6, gemma_norm=True
        )
    )

    assert called["gemma_norm"] is True
    assert called["all_reduce_kwargs"] == {"prefill_support": False}
    assert called["args"][0] is out
    assert called["args"][3] is residual_out
    assert called["args"][4] is scale
    assert out.shape == shape
    assert residual_out.shape == shape
    assert scale.shape == (shape[0], 1)


def test_gfx1250_gemma_quant_fallback_respects_payload_limit(monkeypatch):
    import aiter

    communicator = importlib.import_module(
        "aiter.dist.device_communicators.communicator_cuda"
    )
    called = {}

    def unexpected_fused_norm_quant(*args, **kwargs):
        raise AssertionError("oversized payload used the gfx1250 fallback fusion")

    def fake_quant(out, *, quant_dtype):
        called["quant_dtype"] = quant_dtype
        return (
            torch.empty(out.shape, dtype=quant_dtype),
            torch.empty(out.shape[:-1] + (1,), dtype=torch.float32),
        )

    monkeypatch.setattr(communicator, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(communicator, "get_hip_quant", lambda quant_type: fake_quant)
    monkeypatch.setattr(
        aiter,
        "rmsnorm2d_fwd_with_add_dynamicquant",
        unexpected_fused_norm_quant,
    )

    class FakeCommunicator:
        ca_comm = None
        _ar_quant_max_bytes = -1
        _ar_quant_no_prefill_max_bytes = 1

        def fused_allreduce_rmsnorm(
            self,
            input_,
            res_inp_,
            weight_,
            eps,
            prefill_support,
            *,
            gemma_norm,
        ):
            called["legacy"] = (prefill_support, gemma_norm)
            return input_.clone(), res_inp_.clone()

    shape = (2, 6144)
    x = torch.empty(shape, dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    weight = torch.empty(shape[-1], dtype=torch.bfloat16)
    out, residual_out, scale = (
        communicator.CudaCommunicator.fused_allreduce_rmsnorm_quant(
            FakeCommunicator(), x, residual, weight, 1e-6, gemma_norm=True
        )
    )

    assert called["legacy"] == (False, True)
    assert called["quant_dtype"] == out.dtype
    assert residual_out.shape == shape
    assert scale.shape == (shape[0], 1)
