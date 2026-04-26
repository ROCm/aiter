import pytest
import torch

from aiter.dsv4_validate import dsv4_validate_sparse_attn_metadata


def _ok_meta(B=1, M=1, K=4, N=12, D=64, num_tokens=1, pool=4, head=2):
    """Construct a known-valid metadata tuple for happy-path tests."""
    return dict(
        q=torch.zeros(B, M, head, D),
        kv=torch.zeros(B, N, D),
        topk_idxs=torch.zeros(B, M, K, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.long),
        positions=torch.zeros(num_tokens, dtype=torch.long),
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32),
        pool_capacity=pool,
    )


class TestShapeRank:
    def test_q_must_be_4d(self):
        m = _ok_meta()
        m["q"] = torch.zeros(2, 3)  # 2D, wrong
        with pytest.raises(ValueError, match="q must be 4-D"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_kv_must_be_3d(self):
        m = _ok_meta()
        m["kv"] = torch.zeros(2, 3)  # 2D, wrong
        with pytest.raises(ValueError, match="kv must be 3-D"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_qkv_batch_must_match(self):
        m = _ok_meta(B=1)
        m["kv"] = torch.zeros(2, 12, 64)
        with pytest.raises(ValueError, match=r"q\.B=1 != kv\.B=2"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_qkv_head_dim_must_match(self):
        m = _ok_meta(D=64)
        m["kv"] = torch.zeros(1, 12, 32)  # head_dim mismatch
        with pytest.raises(ValueError, match="head_dim=64.*kv.*head_dim=32"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_must_be_3d(self):
        m = _ok_meta()
        m["topk_idxs"] = torch.zeros(4, dtype=torch.int32)
        with pytest.raises(ValueError, match="topk_idxs must be 3-D"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_first_two_dims_must_match_q(self):
        m = _ok_meta(B=1, M=1)
        m["topk_idxs"] = torch.zeros(2, 1, 4, dtype=torch.int32)  # B mismatch
        with pytest.raises(ValueError, match=r"topk_idxs\.shape\[:2\]"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_happy_path_passes(self):
        # Should not raise on valid input
        dsv4_validate_sparse_attn_metadata(**_ok_meta())


class TestDtype:
    def test_topk_must_be_int32(self):
        m = _ok_meta()
        m["topk_idxs"] = torch.zeros(1, 1, 4, dtype=torch.int64)
        with pytest.raises(ValueError, match="topk_idxs.dtype must be int32"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_slot_mapping_must_be_int(self):
        m = _ok_meta()
        m["slot_mapping"] = torch.zeros(1, dtype=torch.float32)
        with pytest.raises(ValueError, match="slot_mapping.dtype must be int"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_positions_must_be_int(self):
        m = _ok_meta()
        m["positions"] = torch.zeros(1, dtype=torch.float32)
        with pytest.raises(ValueError, match="positions.dtype must be int"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_cu_must_be_int(self):
        m = _ok_meta()
        m["cu_seqlens_q"] = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match="cu_seqlens_q.dtype must be int"):
            dsv4_validate_sparse_attn_metadata(**m)


class TestHappyPathVariousShapes:
    def test_passes_for_diverse_shapes(self):
        for kw in [
            dict(B=2, M=8, K=16, N=24, D=64, num_tokens=16, head=4),
            dict(B=4, M=1, K=4, N=8, D=128, num_tokens=4, head=1),
            dict(B=1, M=64, K=8, N=64, D=64, num_tokens=64, head=8),
        ]:
            dsv4_validate_sparse_attn_metadata(**_ok_meta(**kw))


class TestDeviceContiguity:
    def test_kv_device_must_match_q(self):
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for cross-device test")
        m = _ok_meta()
        m["q"] = m["q"].to("cuda")
        # kv stays on CPU
        with pytest.raises(ValueError, match=r"kv\.device=.*q\.device="):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_must_be_contiguous(self):
        # Use transpose to construct non-contiguous topk_idxs
        # Shape [2,4,4] transposed to [4,4,2] is non-contiguous
        non_contig = torch.zeros(2, 4, 4, dtype=torch.int32).transpose(0, 1)
        assert not non_contig.is_contiguous(), "test setup must produce non-contig"
        # non_contig.shape == [4, 2, 4]; matching q [4, 2, 2, 64], kv [4, 12, 64]
        m = _ok_meta(B=4, M=2, K=4, N=12, num_tokens=8)
        m["q"] = torch.zeros(4, 2, 2, 64)
        m["kv"] = torch.zeros(4, 12, 64)
        m["topk_idxs"] = non_contig
        m["positions"] = torch.zeros(8, dtype=torch.long)
        m["slot_mapping"] = torch.zeros(8, dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0, 8], dtype=torch.int32)
        with pytest.raises(ValueError, match="topk_idxs must be contiguous"):
            dsv4_validate_sparse_attn_metadata(**m)


class TestTopkDomain:
    def test_topk_below_sentinel_rejected(self):
        m = _ok_meta(B=1, M=1, K=4)
        m["topk_idxs"] = torch.tensor([[[-2, 0, 1, 2]]], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"topk_idxs contains values < -1"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_max_must_be_lt_kv_n(self):
        m = _ok_meta(B=1, M=1, K=4, N=12)
        m["topk_idxs"] = torch.tensor([[[0, 1, 2, 128]]], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"topk_idxs max=128 >= kv\.size\(N\)=12"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_all_negative_one_sentinels_pass(self):
        m = _ok_meta(B=1, M=1, K=4)
        m["topk_idxs"] = torch.full((1, 1, 4), -1, dtype=torch.int32)
        # Should not raise — -1 is the skip sentinel
        dsv4_validate_sparse_attn_metadata(**m)

    def test_empty_topk_passes(self):
        m = _ok_meta(B=1, M=0, K=4)
        m["q"] = torch.zeros(1, 0, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 0, 4, dtype=torch.int32)
        m["positions"] = torch.zeros(0, dtype=torch.long)
        m["slot_mapping"] = torch.zeros(0, dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0], dtype=torch.int32)
        dsv4_validate_sparse_attn_metadata(**m)
