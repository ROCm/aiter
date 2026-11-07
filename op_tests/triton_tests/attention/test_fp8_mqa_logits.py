# tests are adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py
import pytest
import torch

from aiter.ops.triton.attention.fp8_mqa_logits import (
    ASYNC_COPY_SUPPORTS_DISTRIBUTED,
    FOLDED_REDUCTED_SUPPORT,
    arch as mqa_arch,
    fp8_mqa_logits,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()
fp8_info = torch.finfo(e4m3_type)
fp8_max = fp8_info.max


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def check_logits(logits, ref_logits, where="", check_mask=True, tol=1e-3):
    ref_neginf_mask = ref_logits == float("-inf")
    if check_mask:
        assert torch.equal(logits == float("-inf"), ref_neginf_mask), f"mask {where}"
    if ref_neginf_mask.all():
        return  # nothing left to compare
    diff = calc_diff(
        logits.masked_fill(ref_neginf_mask, 0),
        ref_logits.masked_fill(ref_neginf_mask, 0),
    )
    assert diff < tol, f"{where}{diff=}"


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
    kv = kv_fp8.to(torch.float32) * scales.reshape(-1, 1)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    # to respect the assert in generate_cp_test_data
    if disable_cp or s_k % s_q != 0 or s_q % 2 != 0:
        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)
    else:
        ks, ke = generate_cp_test_data(s_q, s_k)

    q_fp8 = q.to(e4m3_type)
    q = q_fp8.to(torch.float32)

    ref_logits, _ref_cost = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )

    logits = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)
    assert logits.size() == (s_q, s_k)
    check_logits(logits, ref_logits, check_mask=clean_logits, tol=1e-4)


@pytest.mark.parametrize("s_q, s_k", [(8192, 65664), (8192, 98304)])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_fp8_mqa_logits_logits_past_2gib(
    s_q: int, s_k: int, num_heads: int, head_dim: int, clean_logits: bool
) -> None:
    """Prefill shapes whose fp32 logits tensor exceeds 2 GiB.

    Tests for potential buffer store/load issues and split-k.
    """
    # logit bytes with alignment
    logits_bytes = s_q * ((s_k + 255) // 256 * 256) * 4
    # the previous parametrization's logits are still held by the caching
    # allocator, and they are the same size as this one's
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < logits_bytes * 2:
        pytest.skip(f"needs {logits_bytes * 2 / 2**30:.1f} GiB free")

    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)

    q_fp8 = q.to(e4m3_type)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    logits = fp8_mqa_logits(
        q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=clean_logits
    )
    assert logits.shape == (s_q, s_k)

    # Slicing to reduce memory usage when testing
    ROWS_PER_CHECK = 32
    for a in range(0, s_q, ROWS_PER_CHECK):
        b = min(a + ROWS_PER_CHECK, s_q)
        ref_logits, _ = ref_fp8_mqa_logits(
            q=q[a:b],
            kv=kv,
            weights=weights[a:b],
            cu_seqlen_ks=ks[a:b],
            cu_seqlen_ke=ke[a:b],
        )
        check_logits(
            logits[a:b], ref_logits, where=f"rows {a}:{b}: ", check_mask=clean_logits
        )


@pytest.mark.parametrize("cu_start", [1, 128, 2529])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_fp8_mqa_logits_nonzero_cu_start(
    cu_start: int, num_heads: int, head_dim: int
) -> None:
    """Regression: rows whose kv window does not start at column 0 -- under
    vLLM, every row of the 2nd+ prefill request packed into one indexer
    chunk. The epilogue store masks scaled start_ind by BLOCK_KV."""
    torch.manual_seed(0)
    block_kv = 128  # BLOCK_KV on gfx1250
    seg_lens = [
        n * block_kv + tail
        for n in range(6)
        for tail in (0, 37)
        if n * block_kv + tail > 0
    ]
    s_q = len(seg_lens)
    s_k = cu_start + max(seg_lens)

    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = kv_fp8.to(torch.float32) * scales.reshape(-1, 1)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)

    ks = torch.full((s_q,), cu_start, dtype=torch.int, device="cuda")
    ke = ks + torch.tensor(seg_lens, dtype=torch.int, device="cuda")

    q_fp8 = q.to(e4m3_type)
    q = q_fp8.to(torch.float32)

    # kv_fp8/scales are the kernel's KV; kv is what they dequantize to, so the
    # reference works from the same values. Re-quantizing kv here would rebind
    # kv_fp8/scales and hand the kernel a slightly different KV.
    ref_logits, _ref_cost = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    logits = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, True)

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)
    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)
    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-4, f"{diff=}"


def _strided_test_tensor(tensor):
    storage = torch.empty(
        tuple(2 * size + 1 for size in tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = storage[tuple(slice(1, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


def _long_kv_test_inputs(s_q, s_k, num_heads, head_dim, strided=False):
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    q_fp8 = q.to(e4m3_type)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    if strided:
        q_fp8, kv_fp8, weights = (
            _strided_test_tensor(tensor) for tensor in (q_fp8, kv_fp8, weights)
        )

    row = torch.arange(s_q, dtype=torch.int32, device="cuda")
    phase = (row + row // 8) % 8
    starts = (row * 131) % (s_k - 1)
    starts = torch.where(phase == 5, s_k - 1, starts)
    starts = torch.where(phase == 7, 0, starts)
    lengths = torch.tensor(
        [0, 1, 63, 64, 65, 1, 127, s_k], dtype=torch.int32, device="cuda"
    )[phase]
    lengths = torch.where(phase == 6, 127 + (row // 8) % 3, lengths)
    ends = (starts + lengths).clamp_max(s_k)
    return q_fp8, kv_fp8, scales, weights, starts, ends


def _check_long_kv_logits(logits, inputs, clean_logits=False):
    q, kv, scales, weights, starts, ends = inputs
    assert logits.shape == (q.shape[0], kv.shape[0])
    assert logits.dtype == torch.float32
    kv_ref = kv.float() * scales[:, None]
    for first in range(0, q.shape[0], 32):
        last = min(first + 32, q.shape[0])
        ref, _ = ref_fp8_mqa_logits(
            q[first:last].float(),
            kv_ref,
            weights[first:last],
            starts[first:last],
            ends[first:last],
        )
        check_logits(logits[first:last], ref, check_mask=clean_logits, tol=1e-4)
        if q.shape[0] <= 24:
            for local_row in range(last - first):
                actual = logits[first + local_row]
                expected = ref[local_row]
                check_logits(actual, expected, check_mask=clean_logits, tol=1e-4)
                check_logits(
                    actual[-2:], expected[-2:], check_mask=clean_logits, tol=1e-4
                )


@pytest.mark.parametrize(
    "s_q,s_k,num_heads,head_dim,clean_logits,strided",
    [
        pytest.param(7, 4096, 32, 128, False, False, id="row-below-eight"),
        pytest.param(8, 4095, 32, 128, False, False, id="short-kv"),
        pytest.param(8, 4096, 32, 128, False, False, id="eight-rows"),
        pytest.param(8, 4097, 32, 128, False, False, id="key-tail"),
        pytest.param(9, 4097, 32, 128, False, False, id="row-tail"),
        pytest.param(24, 4223, 32, 128, False, False, id="three-groups"),
        pytest.param(4096, 4097, 32, 128, False, False, id="many-groups"),
        pytest.param(4098, 4097, 32, 128, False, False, id="two-row-fallback"),
        pytest.param(4104, 4097, 32, 128, False, False, id="many-groups-tail"),
        pytest.param(8, 4097, 32, 128, True, False, id="clean-output"),
        pytest.param(8, 4097, 64, 128, False, False, id="other-head-count"),
        pytest.param(8, 4097, 32, 64, False, False, id="other-head-size"),
        pytest.param(16, 4097, 32, 128, False, True, id="strided-inputs"),
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_windows(
    s_q, s_k, num_heads, head_dim, clean_logits, strided
):
    """Check long-KV row groups and their public-wrapper fallback cases."""
    torch.manual_seed(0)
    inputs = _long_kv_test_inputs(s_q, s_k, num_heads, head_dim, strided)
    logits = fp8_mqa_logits(*inputs, clean_logits=clean_logits)
    _check_long_kv_logits(logits, inputs, clean_logits)


@pytest.mark.parametrize(
    "s_k,window_length",
    [(4097, length) for length in (1, 63, 64, 65, 127, 128, 129)]
    + [(65537, 129), (65537, 65501)],
)
@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_nonzero_union(s_k, window_length):
    """Check narrow and split KV ranges without a full-range query row."""
    torch.manual_seed(0)
    inputs = _long_kv_test_inputs(8, s_k, 32, 128)
    inputs[-2].fill_(17)
    inputs[-1].fill_(17 + window_length)
    logits = fp8_mqa_logits(*inputs, clean_logits=False)
    _check_long_kv_logits(logits, inputs)


@pytest.mark.parametrize("s_q,s_k", [(0, 4097), (8, 0), (0, 0)])
@pytest.mark.parametrize("clean_logits", [False, True])
@torch.inference_mode()
def test_fp8_mqa_logits_empty_output(s_q, s_k, clean_logits):
    q = torch.empty(s_q, 32, 128, device="cuda", dtype=e4m3_type)
    kv = torch.empty(s_k, 128, device="cuda", dtype=e4m3_type)
    scales = torch.empty(s_k, device="cuda", dtype=torch.float32)
    weights = torch.empty(s_q, 32, device="cuda", dtype=torch.float32)
    starts = torch.zeros(s_q, device="cuda", dtype=torch.int32)
    ends = torch.zeros_like(starts)
    logits = fp8_mqa_logits(q, kv, scales, weights, starts, ends, clean_logits)
    assert logits.shape == (s_q, s_k)
    assert logits.dtype == torch.float32
    assert logits.device == q.device


@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_graph_replay():
    """Alternate two graphs with fresh inputs and changing valid windows."""
    torch.manual_seed(0)
    states = []
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        inputs = _long_kv_test_inputs(16, 4223, 32, 128)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                fp8_mqa_logits(*inputs, clean_logits=False)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = fp8_mqa_logits(*inputs, clean_logits=False)
        states.append((graph, inputs, logits))

    windows = ("full", "ragged", "empty", "full", "ragged", "full")
    for step, window in enumerate(windows):
        graph, inputs, logits = states[step % 2]
        fresh = _long_kv_test_inputs(16, 4223, 32, 128)
        for target, source in zip(inputs, fresh):
            target.copy_(source)
        if window == "full":
            inputs[-2].zero_()
            inputs[-1].fill_(4223)
        elif window == "empty":
            inputs[-2].fill_(17)
            inputs[-1].fill_(17)

        other_logits = states[1 - step % 2][2]
        other_before = other_logits.clone()
        logits.fill_(float("nan"))
        graph.replay()
        _check_long_kv_logits(logits, inputs)
        torch.testing.assert_close(
            other_logits, other_before, rtol=0, atol=0, equal_nan=True
        )


@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_auxiliary_strides():
    if not (
        mqa_arch == "gfx950"
        and e4m3_type == torch.float8_e4m3fn
        and ASYNC_COPY_SUPPORTS_DISTRIBUTED
        and FOLDED_REDUCTED_SUPPORT
    ):
        pytest.skip("Requires the gfx950 eight-row path")
    torch.manual_seed(0)
    inputs = list(_long_kv_test_inputs(8, 4097, 32, 128))
    inputs[-2].fill_(17)
    inputs[-1].fill_(146)
    for i in (2, 4, 5):
        inputs[i] = _strided_test_tensor(inputs[i])
    logits = fp8_mqa_logits(*inputs, clean_logits=False)
    _check_long_kv_logits(logits, inputs)
