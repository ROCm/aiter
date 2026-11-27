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


@pytest.mark.parametrize(
    "s_q, s_k", [(1, 113), (61, 1024), (1024, 1560), (1019, 40000)]
)
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize("layout", ["aligned", "tight", "offset"])
@torch.inference_mode()
def test_fp8_mqa_logits_into_caller_buffer(
    s_q: int, s_k: int, num_heads: int, clean_logits: bool, layout: str
) -> None:
    """`out=` gets the allocating path's scores in each row's range and writes
    nothing past seq_len_kv columns or outside the view."""
    torch.manual_seed(0)
    head_dim = 128
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.randint(0, s_k // 2 + 1, (s_q,), dtype=torch.int, device="cuda")
    span = torch.randint(1, s_k + 1, (s_q,), dtype=torch.int, device="cuda")
    ke = torch.minimum(ks + span, torch.full_like(ks, s_k))
    q_fp8 = q.to(e4m3_type)
    expected = fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, True)

    aligned = (s_k + 255) // 256 * 256
    pitch, offset = {
        "aligned": (aligned, 0),
        "tight": (s_k, 0),
        "offset": (aligned + 17, 37),
    }[layout]
    guard = 4096
    plane = torch.full(
        (offset + s_q * pitch + guard,),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )
    rows = plane[offset : offset + s_q * pitch].view(s_q, pitch)
    view = rows[:, :s_k]
    logits = fp8_mqa_logits(
        q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits, out=view
    )
    assert logits.data_ptr() == view.data_ptr()
    cols = torch.arange(s_k, device="cuda")
    inside = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
    assert torch.equal(logits[inside], expected[inside])
    if clean_logits:
        assert torch.isneginf(logits[~inside]).all()
    assert torch.isnan(plane[:offset]).all()
    assert torch.isnan(plane[offset + s_q * pitch :]).all()
    assert torch.isnan(rows[:, s_k:]).all()


@pytest.mark.parametrize(
    "bad, message",
    [
        ("shape", r"out must be \["),
        ("pitch", "unit column stride and a row stride"),
        ("column_stride", "unit column stride and a row stride"),
        ("dtype", "2-D float32"),
        ("device", "out is on cpu"),
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_refuses_a_malformed_buffer(bad: str, message: str) -> None:
    s_q, s_k, num_heads, head_dim = 4, 300, 32, 128
    q_fp8 = torch.randn(
        s_q, num_heads, head_dim, device="cuda", dtype=torch.float32
    ).to(e4m3_type)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(
        torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16), (0,), False
    )
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.full((s_q,), s_k, dtype=torch.int, device="cuda")
    f32 = {"device": "cuda", "dtype": torch.float32}
    buffers = {
        "shape": torch.empty(s_q, s_k + 256, **f32)[:, : s_k - 1],
        "pitch": torch.empty(s_q * s_k, **f32).as_strided((s_q, s_k), (s_k - 1, 1)),
        "column_stride": torch.empty(s_k, s_q, **f32).t(),
        "dtype": torch.empty(s_q, s_k, device="cuda", dtype=torch.float16),
        "device": torch.empty(s_q, s_k, device="cpu", dtype=torch.float32),
    }
    with pytest.raises(ValueError, match=message):
        fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, False, out=buffers[bad])


@pytest.mark.parametrize(
    "rows,keys", [(9, 1025), (4097, 1025), (9, 65539), (1019, 40000)]
)
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("clean_logits", [False, True])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_fp8_mqa_logits_packed(rows, keys, heads, clean_logits, offset_dtype):
    """Packed rows match the plane and leave gaps and guards untouched."""
    torch.manual_seed(7517)
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.float32).to(e4m3_type)
    kv = torch.randn(keys, 128, device="cuda", dtype=torch.float32).to(e4m3_type)
    scales = torch.rand(keys, device="cuda", dtype=torch.float32) + 0.5
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.float32)
    pattern = [0, 1, 63, 64, 65, 127, 128, 129, keys - 17]
    lengths = [pattern[i % len(pattern)] for i in range(rows)]
    visible = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    starts = (
        torch.arange(rows, device="cuda", dtype=torch.int32)
        * 193
        % (keys - visible + 1)
    )
    ends = starts + visible
    # Reverse the storage order to cover negative distances between row bases.
    locations, capacity = [0] * rows, 0
    for row in reversed(range(rows)):
        locations[row] = capacity
        capacity += lengths[row] + 7
    offsets = torch.tensor(locations, device="cuda", dtype=offset_dtype)
    storage = torch.full(
        (capacity + 34,), float("nan"), device="cuda", dtype=torch.float32
    )
    packed = storage[17:-17]
    dense = fp8_mqa_logits(q, kv, scales, weights, starts, ends, clean_logits=False)
    actual = fp8_mqa_logits(
        q,
        kv,
        scales,
        weights,
        starts,
        ends,
        clean_logits=clean_logits,
        out=packed,
        out_offsets=offsets,
    )
    assert actual is packed
    row_ids = torch.repeat_interleave(torch.arange(rows, device="cuda"), visible)
    first = visible.to(torch.int64).cumsum(0) - visible
    columns = torch.arange(sum(lengths), device="cuda") - first[row_ids]
    destinations = offsets[row_ids] + columns
    torch.testing.assert_close(
        packed[destinations], dense[row_ids, starts[row_ids] + columns], rtol=0, atol=0
    )
    untouched = torch.ones(capacity, device="cuda", dtype=torch.bool)
    untouched[destinations] = False
    assert torch.isnan(packed[untouched]).all()
    assert torch.isnan(storage[:17]).all() and torch.isnan(storage[-17:]).all()


@pytest.mark.parametrize("rows", [0, 5])
@torch.inference_mode()
def test_fp8_mqa_logits_packed_empty_rows(rows):
    q = torch.zeros(rows, 32, 128, device="cuda", dtype=e4m3_type)
    kv = torch.zeros(73, 128, device="cuda", dtype=e4m3_type)
    scales = torch.ones(73, device="cuda", dtype=torch.float32)
    weights = torch.ones(rows, 32, device="cuda", dtype=torch.float32)
    bounds = torch.zeros(rows, device="cuda", dtype=torch.int32)
    packed = torch.full((19,), float("nan"), device="cuda", dtype=torch.float32)
    assert (
        fp8_mqa_logits(
            q, kv, scales, weights, bounds, bounds, out=packed, out_offsets=bounds
        )
        is packed
    )
    assert torch.isnan(packed).all()


@pytest.mark.parametrize(
    "bad, message",
    [
        ("missing", "requires a 1-D float32 out"),
        ("rank", "requires a 1-D float32 out"),
        ("dtype", "requires a 1-D float32 out"),
        ("stride", "packed out must be contiguous"),
        ("offset_dtype", "out_offsets must be contiguous"),
        ("offset_shape", "out_offsets must be contiguous"),
        ("offset_stride", "out_offsets must be contiguous"),
        ("offset_device", "out_offsets must be contiguous"),
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_rejects_malformed_packed_output(bad, message):
    q = torch.empty(4, 32, 128, device="cuda", dtype=e4m3_type)
    kv = torch.empty(73, 128, device="cuda", dtype=e4m3_type)
    scales = torch.ones(73, device="cuda", dtype=torch.float32)
    weights = torch.ones(4, 32, device="cuda", dtype=torch.float32)
    bounds = torch.zeros(4, device="cuda", dtype=torch.int32)
    packed = torch.empty(512, device="cuda", dtype=torch.float32)
    offsets = torch.zeros(4, device="cuda", dtype=torch.int32)
    if bad == "missing":
        packed = None
    elif bad == "rank":
        packed = packed.view(4, 128)
    elif bad == "dtype":
        packed = packed.to(torch.float16)
    elif bad == "stride":
        packed = packed[::2]
    elif bad == "offset_dtype":
        offsets = offsets.to(torch.float32)
    elif bad == "offset_shape":
        offsets = offsets[:3]
    elif bad == "offset_stride":
        offsets = torch.zeros(8, device="cuda", dtype=torch.int32)[::2]
    else:
        offsets = torch.zeros(4, device="cpu", dtype=torch.int32)
    with pytest.raises(ValueError, match=message):
        fp8_mqa_logits(
            q, kv, scales, weights, bounds, bounds, out=packed, out_offsets=offsets
        )


def _modes_case(rows, keys, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, 32, 128, device="cuda", generator=g).to(e4m3_type)
    kv = torch.randn(keys, 128, device="cuda", generator=g).to(e4m3_type)
    scales = torch.rand(keys, device="cuda", generator=g) + 0.5
    weights = torch.randn(rows, 32, device="cuda", generator=g)
    starts = torch.randint(0, keys // 2 + 1, (rows,), device="cuda", generator=g)
    span = torch.randint(0, keys + 1, (rows,), device="cuda", generator=g)
    ends = torch.minimum(starts + span, torch.full_like(starts, keys))
    return q, kv, scales, weights, starts.int(), ends.int()


def _modes_reference(q, kv, scales, weights, starts, ends):
    s = torch.einsum("rhd,kd->rhk", q.float(), kv.float()).relu()
    out = (s * weights[:, :, None]).sum(1) * scales[None, :]
    cols = torch.arange(kv.shape[0], device="cuda")
    inside = (cols[None, :] >= starts[:, None]) & (cols[None, :] < ends[:, None])
    return out, inside


# fp32 head sum (2048 to 4097 rows), the few-row config, the matrix-core head sum
@pytest.mark.parametrize(
    "rows,keys", [(2048, 1025), (2051, 3001), (4097, 777), (5, 3001), (300, 9000)]
)
@torch.inference_mode()
def test_fp8_mqa_logits_modes_bit_identical(rows, keys):
    _check_modes(*_modes_case(rows, keys, rows + keys))


# the per-row kernel for short windows, and KV splits sized for the window
@pytest.mark.parametrize("rows,keys,window", [(5, 5000, 300), (64, 9000, 2000)])
@torch.inference_mode()
def test_fp8_mqa_logits_max_window_modes_bit_identical(rows, keys, window):
    q, kv, scales, weights, starts, ends = _modes_case(rows, keys, rows * keys)
    starts = torch.clamp(ends - (ends - starts) % (window + 1), min=0).int()
    hint = int((ends - starts).clamp(min=0).max())
    _check_modes(q, kv, scales, weights, starts, ends, max_window=hint)


@torch.inference_mode()
def test_fp8_mqa_logits_tuning_kwargs_bit_identical():
    case = _modes_case(4099, 24577, 7)
    ref = fp8_mqa_logits(*case).view(torch.int32)
    for kw in ({"big_tiles": False}, {"key_vec": 1}, {"key_vec": 2, "k_early": False}):
        assert torch.equal(fp8_mqa_logits(*case, **kw).view(torch.int32), ref), kw


def _check_modes(q, kv, scales, weights, starts, ends, **hint):
    rows, keys = q.shape[0], kv.shape[0]
    ref, inside = _modes_reference(q, kv, scales, weights, starts, ends)

    dense = fp8_mqa_logits(
        q, kv, scales, weights, starts, ends, clean_logits=False, **hint
    )
    err = (dense[inside] - ref[inside]).abs().max() / ref[inside].abs().max()
    assert err < 1e-3, f"max-abs relative error {err:.2e}"

    clean = fp8_mqa_logits(
        q, kv, scales, weights, starts, ends, clean_logits=True, **hint
    )
    assert torch.equal(clean[inside], dense[inside])
    assert torch.isneginf(clean[~inside]).all()

    out = torch.full((rows, keys), float("nan"), device="cuda")
    fp8_mqa_logits(
        q, kv, scales, weights, starts, ends, clean_logits=False, out=out, **hint
    )
    assert torch.equal(out[inside], dense[inside])

    lens = (ends - starts).clamp(min=0).long()
    offsets = torch.zeros_like(lens)
    offsets[1:] = lens.cumsum(0)[:-1]
    packed = torch.full((int(lens.sum()) + 8,), float("nan"), device="cuda")
    fp8_mqa_logits(
        q,
        kv,
        scales,
        weights,
        starts,
        ends,
        out=packed[4:-4],
        out_offsets=offsets,
        **hint,
    )
    row_ids = torch.repeat_interleave(torch.arange(rows, device="cuda"), lens)
    first = lens.cumsum(0) - lens
    columns = torch.arange(int(lens.sum()), device="cuda") - first[row_ids]
    assert torch.equal(
        packed[4:-4][offsets[row_ids] + columns],
        dense[row_ids, starts[row_ids] + columns],
    )
    assert torch.isnan(packed[:4]).all() and torch.isnan(packed[-4:]).all()


_ROWSPLIT_KEYS = (
    "BM",
    "NUM_WARPS",
    "BLOCK_KV",
    "SUB_N",
    "NUM_BUFFERS",
    "MIN_TILES_PER_SPLIT",
)
_ROWSPLIT_LAUNCHES = [
    (1, 1, 128, 32, 3, 1),
    (1, 4, 128, 32, 3, 2),
    (2, 4, 128, 16, 2, 4),
    (2, 2, 64, 32, 4, 1),
]


@pytest.mark.parametrize("clean_logits", [False, True])
@pytest.mark.parametrize("headsum_mma", [0, 1])
@torch.inference_mode()
def test_fp8_mqa_logits_rowsplit_launch_independent(
    monkeypatch, clean_logits, headsum_mma
):
    """gfx1250 32 x 128: every row-split launch shape and key_vec / k_early agree."""
    import aiter.ops.triton.attention.fp8_mqa_logits as mod

    if mod.arch not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("gfx1250 row-split kernel")
    rows, keys = 67, 2049
    q, kv, scales, weights, starts, ends = _modes_case(rows, keys, 7 * rows + keys)
    base_cfg = mod._gfx1250_launch_config(keys, 32, 128)
    outs = []
    for li, launch in enumerate(_ROWSPLIT_LAUNCHES):
        if headsum_mma and launch[0] % 2:
            continue  # the matrix-core head sum needs an even BM
        for ti, target in enumerate((1, 4096)):
            cfg = dict(base_cfg)
            rs = dict(
                base_cfg["rowsplit"],
                TARGET_PROGRAMS=target,
                waves_per_eu=0,
                HEADSUM_MMA=headsum_mma,
                KEY_VEC=(1, 2, 4)[(li + ti) % 3],
                K_EARLY=(li + ti) % 2,
            )
            rs.update(zip(_ROWSPLIT_KEYS, launch))
            rs["SMALL_ROWS"] = 0
            rs["PR_MAX_ROWS"] = 0  # always the row-split kernel
            cfg["rowsplit"] = rs
            monkeypatch.setattr(mod, "_gfx1250_launch_config", lambda *a, _c=cfg: _c)
            outs.append(
                fp8_mqa_logits(
                    q, kv, scales, weights, starts, ends, clean_logits=clean_logits
                )
            )
    _, inside = _modes_reference(q, kv, scales, weights, starts, ends)
    for out in outs[1:]:
        assert torch.equal(out[inside], outs[0][inside])
        if clean_logits:
            assert torch.equal(out, outs[0])


# page-sharing kernel (next_n 1, one and two rows per warp), per-row kernel (hint)
@pytest.mark.parametrize(
    "batch,next_n,ctx,ctx_per_row,hint",
    [
        (1, 1, 100, False, False),
        (2, 5, 3000, False, False),
        (5, 8, 1500, True, False),
        (3, 4, 777, True, True),
        (5, 2, 64, False, True),
    ],
)
@torch.inference_mode()
def test_fp8_paged_mqa_logits(batch, next_n, ctx, ctx_per_row, hint):
    import aiter.ops.triton.attention.fp8_mqa_logits as mod

    if mod.arch not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("gfx1250 paged kernel")
    g = torch.Generator(device="cuda").manual_seed(batch * 1000 + ctx)
    bs, max_model_len = 64, 4096
    pages = (ctx + bs - 1) // bs
    num_blocks = batch * pages + 3
    kv = torch.randn(num_blocks, bs, 128, device="cuda", generator=g).to(e4m3_type)
    scales = torch.rand(num_blocks, bs, device="cuda", generator=g) + 0.5
    cache = torch.cat(
        [
            kv.view(torch.uint8).reshape(num_blocks, -1),
            scales.view(torch.uint8).reshape(num_blocks, -1),
        ],
        1,
    ).view(num_blocks, bs, 1, 132)
    perm = torch.randperm(num_blocks, device="cuda", generator=g)
    block_tables = perm[: batch * pages].reshape(batch, pages).int()
    q = torch.randn(batch, next_n, 32, 128, device="cuda", generator=g).to(e4m3_type)
    weights = torch.randn(batch * next_n, 32, device="cuda", generator=g)
    row_ctx = (ctx - next_n + 1 + torch.arange(next_n, device="cuda")).repeat(batch)
    if ctx_per_row:
        row_ctx = (
            row_ctx - torch.randint(0, 40, row_ctx.shape, device="cuda", generator=g)
        ).clamp(min=0)
        context_lens = row_ctx.int().reshape(batch, next_n)
    else:
        context_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    out = torch.full((batch * next_n, max_model_len), float("nan"), device="cuda")
    mod.fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        out=out,
        max_ctx=ctx if hint else None,
    )
    keys = kv[block_tables.long()].reshape(batch, -1, 128).float()
    sc = scales[block_tables.long()].reshape(batch, -1)
    s = torch.einsum("bnhd,bkd->bnhk", q.float(), keys).relu()
    ref = (s * weights.view(batch, next_n, 32, 1)).sum(2) * sc[:, None, :]
    ref = ref.reshape(batch * next_n, -1)
    inside = torch.arange(ref.shape[1], device="cuda")[None, :] < row_ctx[:, None]
    got = out[:, : ref.shape[1]]
    assert torch.isnan(got[~inside]).all() and torch.isnan(out[:, ref.shape[1] :]).all()
    err = (got[inside] - ref[inside]).abs().max() / ref[inside].abs().max()
    assert err < 1e-3, f"max-abs relative error {err:.2e}"
