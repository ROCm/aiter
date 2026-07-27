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


def _capture_runtime_grid(monkeypatch, **overrides):
    class DummyTensor:
        @staticmethod
        def data_ptr():
            return 0

    dummy = DummyTensor()
    captured = {}

    def capture_run_compiled(
        _launch,
        _inter_sorted_quant,
        _inter_sorted_shuffled_scale,
        _w2_u8,
        _w2_scale_u8,
        _sorted_expert_ids,
        _cumsum_tensor,
        _sorted_token_ids,
        _sorted_weights,
        _m_logical,
        max_m_blocks,
        grid_blocks,
        *_rest,
    ):
        captured.update(max_m_blocks=max_m_blocks, grid_blocks=grid_blocks)

    monkeypatch.setattr(dispatcher, "get_g2", lambda *args, **kwargs: object())
    monkeypatch.setattr(dispatcher, "run_compiled", capture_run_compiled)

    kwargs = dict(
        inter_sorted_quant=dummy,
        inter_sorted_shuffled_scale=dummy,
        w2_u8=dummy,
        w2_scale_u8=dummy,
        sorted_expert_ids=dummy,
        cumsum_tensor=dummy,
        sorted_token_ids=dummy,
        sorted_weights=dummy,
        out=dummy,
        M_logical=16,
        max_sorted=8384,
        NE=257,
        D_HIDDEN=6144,
        D_INTER=512,
        topk=9,
        BM=32,
        BN=128,
        BK=256,
        stream=0,
    )
    kwargs.update(overrides)
    dispatcher.mxfp4_moe_gemm2(**kwargs)
    return captured


@pytest.mark.parametrize(
    ("m_logical", "max_sorted", "bm", "sbm", "expected_grid_blocks"),
    [
        (16, 8384, 32, None, 144),
        (16, 8384, 16, 32, 288),
        (32, 8384, 32, None, 258),
        (16, 64, 32, None, 2),
    ],
)
def test_runtime_tightens_nonpersistent_grid_to_active_expert_bound(
    monkeypatch, m_logical, max_sorted, bm, sbm, expected_grid_blocks
):
    captured = _capture_runtime_grid(
        monkeypatch,
        M_logical=m_logical,
        max_sorted=max_sorted,
        BM=bm,
        SBM=sbm,
    )

    assert captured["max_m_blocks"] == (max_sorted + bm - 1) // bm
    assert captured["grid_blocks"] == expected_grid_blocks


@pytest.mark.parametrize(
    ("overrides", "expected_grid_blocks"),
    [
        ({"n_sorted_padded": 64}, 2),
        ({"persist": True, "cu_num": 8}, 8),
    ],
)
def test_runtime_preserves_explicit_grid_selection(
    monkeypatch, overrides, expected_grid_blocks
):
    captured = _capture_runtime_grid(monkeypatch, **overrides)

    assert captured["grid_blocks"] == expected_grid_blocks
