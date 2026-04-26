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
