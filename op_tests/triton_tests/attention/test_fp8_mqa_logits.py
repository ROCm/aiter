# tests are adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py
import pytest
import torch

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()
fp8_info = torch.finfo(e4m3_type)
fp8_max = fp8_info.max


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / fp8_max
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(e4m3_type)
    return x_scaled, sf.squeeze()


def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    cost_only: bool = False,
):
    seq_len_kv = kv.shape[0]

    if cost_only:
        start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
        end = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
        count_ones_per_row = (end - start).clamp(min=0)
        return count_ones_per_row.sum()

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    return logits, cost


def generate_cp_test_data(seq_len, seq_len_kv):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    # Select an arbitrary CP rank
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


@pytest.mark.parametrize(
    "s_q, s_k",
    [
        (1, 1),
        (1, 16),
        (1, 113),
        (17, 76),
        (61, 113),
        (61, 1024),
        (128, 1024),
        (1024, 1024),
        (1024, 1560),
    ],
)
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("disable_cp", [True, False])
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_fp8_mqa_logits(
    s_q: int,
    s_k: int,
    num_heads: int,
    head_dim: int,
    disable_cp: bool,
    clean_logits: bool,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    # to respect the aseert in generate_cp_test_data
    if disable_cp or s_k % s_q != 0 or s_q % 2 != 0:
        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)
    else:
        ks, ke = generate_cp_test_data(s_q, s_k)

    q_fp8 = q.to(e4m3_type)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref_logits, _ref_cost = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )

    logits = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

    # If clean_logits is not set, clean the rest for testing
    if not clean_logits:
        assert logits.size() == (s_q, s_k)
        tmp = torch.full((s_q, s_k), float("-inf"), device="cuda")
        for i in range(s_q):
            tmp[i, ks[i] : ke[i]] = logits[i, : ke[i] - ks[i]]
        logits = tmp

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)
    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)
    diff = calc_diff(logits, ref_logits)
    if ref_neginf_mask.all():
        return  # nothing left to compare
    assert diff < 1e-3, f"{diff=}"


@pytest.mark.parametrize("num_heads", [32, 64])
@torch.inference_mode()
def test_fp8_mqa_logits_large_kv(num_heads: int) -> None:
    """A logits buffer past 2**31 elements.

    The kernel bakes the row and the tile into the store base pointer, so that
    base has to be formed in int64 -- in int32 elements it wraps here. The dense
    reference is far too large at this shape, so a sample of rows is recomputed
    instead, biased towards the high row ids where the base is largest.
    """
    s_q, s_k = 8192, 270336  # 2.21e9 logits elements, 8.85 GiB of fp32
    head_dim = 128
    need = s_q * s_k * 4 + 4 * 1024**3
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 1024**3:.0f} GiB of free device memory")

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q) + 1

    logits = fp8_mqa_logits(q.to(e4m3_type), kv_fp8, scales, weights, ks, ke)

    # Every masked position, and only those, must have been written as -inf.
    assert int(torch.isfinite(logits).sum()) == int((ke - ks).clamp(min=0).sum())

    kv_f32 = kv_fp8.to(torch.float32) * scales.reshape(-1, 1)
    rows = [0, 1, s_q // 2, s_q - 2, s_q - 1] + torch.randint(0, s_q, (5,)).tolist()
    for r in sorted(set(rows)):
        lo, hi = int(ks[r]), min(int(ke[r]), s_k)
        score = (q[r].to(torch.float32) @ kv_f32[lo:hi].T).relu()
        ref = (score * weights[r][:, None]).sum(0)
        diff = calc_diff(logits[r, lo:hi], ref)
        assert diff < 1e-3, f"row {r}: {diff=}"


def _check_rows(logits, q, kv_fp8, scales, weights, lo, hi, rows):
    """Recompute `rows` against a reference; the dense one is far too large."""
    kv_f32 = kv_fp8[lo:hi].to(torch.float32) * scales[lo:hi].reshape(-1, 1)
    for r in rows:
        score = (q[r].to(torch.float32) @ kv_f32.T).relu()
        ref = (score * weights[r][:, None]).sum(0)
        diff = calc_diff(logits[r, lo:hi], ref)
        assert diff < 1e-3, f"row {r}: {diff=}"


@pytest.mark.parametrize("num_heads", [32, 64])
@torch.inference_mode()
def test_fp8_mqa_logits_long_query(num_heads: int) -> None:
    """More query rows than a 2 GiB Q tensor holds.

    The buffer descriptor for Q covers one row, so the row has to go into its
    base pointer in int64. Folded into the 32-bit offset instead, rows past
    2**31 / (num_heads * head_dim) read out of range and silently come back as
    zeros rather than faulting.
    """
    head_dim, s_k = 128, 512
    s_q = 2**31 // (num_heads * head_dim) + 8192  # just over 2 GiB of Q
    need = s_q * (num_heads * head_dim + s_k * 4) + 2 * 1024**3
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 1024**3:.0f} GiB of free device memory")

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.full((s_q,), s_k, dtype=torch.int, device="cuda")

    logits = fp8_mqa_logits(q.to(e4m3_type), kv_fp8, scales, weights, ks, ke, False)
    _check_rows(logits, q, kv_fp8, scales, weights, 0, s_k, [0, s_q // 2, s_q - 1])


@torch.inference_mode()
def test_fp8_mqa_logits_kv_over_2gib() -> None:
    """A KV tensor larger than 2 GiB, with the window past that point.

    Same story on the load side: the KV descriptor is rebased once per tile, so
    `row_offset * stride_kv_s` must be int64. Keeping it exact is what lets the
    launcher stay on buffer loads here instead of the slower global path.
    """
    num_heads, head_dim = 64, 128
    s_q, s_k = 64, 17_039_360  # 2.03 GiB of KV
    lo = 16_700_000  # window starts past the 2 GiB mark
    need = s_k * head_dim + s_q * s_k * 4 + 4 * 1024**3
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 1024**3:.0f} GiB of free device memory")

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    del kv
    torch.cuda.empty_cache()
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.full((s_q,), lo, dtype=torch.int, device="cuda")
    ke = torch.full((s_q,), s_k, dtype=torch.int, device="cuda")

    logits = fp8_mqa_logits(q.to(e4m3_type), kv_fp8, scales, weights, ks, ke, False)
    _check_rows(logits, q, kv_fp8, scales, weights, lo, s_k, [0, s_q // 2, s_q - 1])
