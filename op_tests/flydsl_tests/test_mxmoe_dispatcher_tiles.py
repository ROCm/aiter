import pytest

import aiter.ops.flydsl.kernels.mxmoe_dispatcher as dispatcher


def test_get_g2_forwards_bk_and_keys_it(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatcher, "G2_CACHE", {})
    monkeypatch.setattr(
        dispatcher,
        "compile_gemm2_a4w4_port",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setenv("MXFP4_G2_BHOIST", "1")
    monkeypatch.setenv("MXFP4_G2_ASCALE_PF", "1")
    monkeypatch.setenv("MXFP4_G2_SPART", "0")
    monkeypatch.setenv("MXFP4_G2_BF16_LDS", "0")

    common = {
        "BM": 32,
        "BN": 128,
        "use_nt": False,
        "HIDDEN_MAX": 8192,
        "epilog": "atomic",
        "INTER_MAX": 8192,
        "a_dtype": "fp4",
    }
    dispatcher.get_g2(**common, BK=128)
    dispatcher.get_g2(**common, BK=256)

    assert [call["BK"] for call in calls] == [128, 256]


@pytest.mark.parametrize(("bn", "bk"), [(96, 256), (128, 64)])
def test_compile_rejects_unsupported_tiles(bn, bk):
    with pytest.raises(AssertionError):
        dispatcher.compile_gemm2_a4w4_port(
            BM=32,
            BN=bn,
            BK=bk,
            HIDDEN_MAX=8192,
            INTER_MAX=8192,
            g2_spart=0,
        )


def test_runtime_rejects_inter_dim_not_divisible_by_bk(monkeypatch):
    class DummyTensor:
        @staticmethod
        def data_ptr():
            return 0

    dummy = DummyTensor()
    monkeypatch.setattr(dispatcher, "get_g2", lambda *args, **kwargs: object())
    monkeypatch.setattr(dispatcher, "run_compiled", lambda *args: None)

    with pytest.raises(AssertionError, match="must be a multiple of BK"):
        dispatcher.mxfp4_moe_gemm2(
            inter_sorted_quant=dummy,
            inter_sorted_shuffled_scale=dummy,
            w2_u8=dummy,
            w2_scale_u8=dummy,
            sorted_expert_ids=dummy,
            cumsum_tensor=dummy,
            sorted_token_ids=dummy,
            sorted_weights=dummy,
            out=dummy,
            M_logical=1,
            max_sorted=32,
            NE=1,
            D_HIDDEN=256,
            D_INTER=384,
            topk=1,
            BM=32,
            BN=256,
            BK=256,
            stream=0,
        )
