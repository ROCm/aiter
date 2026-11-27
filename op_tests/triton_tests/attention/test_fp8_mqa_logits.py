# tests are adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py
import functools

import pytest
import torch

import aiter.ops.triton.attention.fp8_mqa_logits as mqa_module
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


# The tests from here to test_fp8_mqa_logits_rows8_empty_output cover the gfx950
# eight-row path. It needs a Triton build with the folded head reduction.
requires_rows8 = pytest.mark.skipif(
    not (
        mqa_module.arch == "gfx950"
        and e4m3_type == torch.float8_e4m3fn
        and mqa_module.ASYNC_COPY_SUPPORTS_DISTRIBUTED
        and mqa_module.FOLDED_REDUCTED_SUPPORT
    ),
    reason="Requires the gfx950 eight-row path",
)


class _RouteProbe(Exception):
    """Stops a call after the wrapper selects a kernel."""


@pytest.fixture
def mqa_launches(monkeypatch):
    """Record the Gluon kernel of each wrapper launch: "existing" or "rows8".

    With ``launches.probe = True`` the launch raises _RouteProbe and no kernel
    runs.
    """

    class _Launches(list):
        probe = False

    launches = _Launches()

    class _Recorder:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            launches.append(self.name)
            if launches.probe:
                raise _RouteProbe(self.name)
            return self.kernel[grid]

    for name, attr in (
        ("existing", "_gluon_fp8_mqa_logits_kernel"),
        ("rows8", "_gluon_fp8_mqa_logits_rows8_kernel"),
    ):
        kernel = getattr(mqa_module, attr)
        if kernel is not None:
            monkeypatch.setattr(mqa_module, attr, _Recorder(name, kernel))
    return launches


@pytest.fixture
def rows8_forced(monkeypatch):
    """Skip the minimum grid size, so that small shapes run the eight-row kernel."""
    monkeypatch.setattr(
        mqa_module,
        "_gfx950_rows8_launch",
        functools.partial(mqa_module._gfx950_rows8_launch, check_min_programs=False),
    )


@requires_rows8
def test_fp8_mqa_logits_rows8_kernel_is_available():
    """A failed import of the eight-row kernel must not pass as a fallback."""
    assert mqa_module._gluon_fp8_mqa_logits_rows8_kernel is not None


# fmt: off
_ROWS8_GATE_CASES = [
    (8, 4096, False), (8, 4097, False), (24, 4223, False), (4096, 4095, False),
    (2040, 4096, False), (1024, 9216, False), (256, 46080, False), (64, 193536, False),
    (2040, 6145, True), (2048, 4096, True), (4096, 4096, True), (4096, 8192, True),
    (4104, 8192, True), (2048, 32768, True), (1024, 9217, True), (1024, 65536, True),
    (1856, 116480, True), (256, 46081, True), (64, 193537, True), (64, 1048576, True),
]
# fmt: on


@requires_rows8
@pytest.mark.parametrize("s_q,s_k,expected", _ROWS8_GATE_CASES)
def test_fp8_mqa_logits_rows8_gate(s_q, s_k, expected):
    """The eight-row path needs min_programs programs and min_seq_len_kv keys."""
    assert (mqa_module._gfx950_rows8_launch(s_q, s_k) is not None) == expected


def _strided_test_tensor(tensor):
    storage = torch.empty(
        tuple(2 * size + 1 for size in tensor.shape),
        dtype=tensor.dtype,
        device="cuda",
    )
    view = storage[tuple(slice(1, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


def _long_kv_test_inputs(
    s_q, s_k, num_heads=32, head_dim=128, strided=False, requests=0, out_of_range=False
):
    """FP8 inputs with ragged windows. With ``requests``, each request has causal
    windows in its own keys, and its first row is inside an eight-row group.
    With ``out_of_range``, windows also start before key 0 and end after the
    last key."""
    q = torch.randn(s_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    q_fp8 = q.to(e4m3_type)
    weights = torch.randn(s_q, num_heads, device="cuda", dtype=torch.float32)
    if strided:
        q_fp8, weights = _strided_test_tensor(q_fp8), _strided_test_tensor(weights)

    row = torch.arange(s_q, dtype=torch.int32, device="cuda")
    if requests:
        # Request r starts at row r * (s_q // requests) + r.
        first_row = torch.arange(requests, dtype=torch.int32, device="cuda")
        first_row = (first_row * (s_q // requests) + first_row).clamp_max(s_q)
        owner = torch.searchsorted(first_row, row, right=True).to(torch.int32) - 1
        keys = s_k // requests
        starts = owner * keys
        ends = (starts + 1 + (row - first_row[owner]) * 3).clamp_max(starts + keys)
        return q_fp8, kv_fp8, scales, weights, starts, ends
    phase = (row + row // 8) % 8
    starts = (row * 131) % (s_k - 1)
    starts = torch.where(phase == 5, s_k - 1, starts)
    starts = torch.where(phase == 7, 0, starts)
    lengths = torch.tensor(
        [0, 1, 63, 64, 65, 1, 127, s_k], dtype=torch.int32, device="cuda"
    )[phase]
    lengths = torch.where(phase == 6, 127 + (row // 8) % 3, lengths)
    ends = (starts + lengths).clamp_max(s_k)
    if out_of_range:
        # Four phases start 200 keys early and four end 200 keys late. Two rows
        # in each 64 have a window that is fully outside the keys.
        starts = torch.where(phase < 4, starts - 200, starts)
        ends = torch.where(phase >= 4, ends + 200, ends)
        starts = torch.where(row % 64 == 3, s_k + 5, starts)
        ends = torch.where(row % 64 == 3, s_k + 50, ends)
        starts = torch.where(row % 64 == 11, -50, starts)
        ends = torch.where(row % 64 == 11, -5, ends)
    return q_fp8, kv_fp8, scales, weights, starts, ends


def _window_mask(inputs):
    starts, ends = inputs[4], inputs[5]
    cols = torch.arange(inputs[1].shape[0], device="cuda")
    return (cols[None, :] >= starts[:, None]) & (cols[None, :] < ends[:, None])


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


@requires_rows8
@pytest.mark.parametrize(
    "s_q,s_k,variant,forced,route",
    [
        pytest.param(8, 4096, "", False, "existing", id="few-rows"),
        pytest.param(4096, 4097, "", False, "rows8", id="many-groups"),
        pytest.param(4104, 4097, "", False, "rows8", id="key-tail"),
        pytest.param(4096, 8192, "requests", False, "rows8", id="requests"),
        pytest.param(2048, 32768, "", False, "rows8", id="two-splits"),
        pytest.param(4096, 4097, "out-of-range", False, "rows8", id="out-of-range"),
        pytest.param(4098, 4097, "", False, "existing", id="row-tail"),
        pytest.param(4096, 4095, "", False, "existing", id="short-kv"),
        pytest.param(4096, 4097, "clean", False, "existing", id="clean"),
        pytest.param(4096, 4097, "64-heads", False, "existing", id="64-heads"),
        pytest.param(4096, 4097, "64-wide", False, "existing", id="64-wide"),
        pytest.param(8, 4096, "", True, "rows8", id="forced-one-group"),
        pytest.param(8, 4097, "", True, "rows8", id="forced-key-tail"),
        pytest.param(24, 4223, "", True, "rows8", id="forced-groups"),
        pytest.param(24, 4223, "requests", True, "rows8", id="forced-requests"),
        pytest.param(16, 4097, "strided", True, "rows8", id="forced-strided"),
        pytest.param(9, 4097, "", True, "existing", id="forced-row-tail"),
        pytest.param(8, 4095, "", True, "existing", id="forced-short-kv"),
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_windows(
    s_q, s_k, variant, forced, route, request, mqa_launches
):
    """Each case runs the expected kernel and matches the FP32 reference."""
    if forced:
        request.getfixturevalue("rows8_forced")
    torch.manual_seed(0)
    inputs = _long_kv_test_inputs(
        s_q,
        s_k,
        num_heads=64 if variant == "64-heads" else 32,
        head_dim=64 if variant == "64-wide" else 128,
        strided=variant == "strided",
        requests=3 if variant == "requests" else 0,
        out_of_range=variant == "out-of-range",
    )
    logits = fp8_mqa_logits(*inputs, clean_logits=variant == "clean")
    assert mqa_launches == [route]
    _check_long_kv_logits(logits, inputs, variant == "clean")


@requires_rows8
@pytest.mark.parametrize(
    "s_q,s_k,requests,out_of_range",
    [
        (4096, 4097, 0, False),
        (4096, 4097, 0, True),
        (4104, 8192, 3, False),
        (2048, 32768, 0, False),
        (1024, 65536, 4, False),
        (256, 245761, 0, False),
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_rows8_matches_existing_route(
    s_q, s_k, requests, out_of_range, mqa_launches
):
    """The eight-row scores equal the clean and packed scores bit for bit.

    The dense output starts as NaN, so a key that the kernel does not write
    fails the comparison.
    """
    torch.manual_seed(0)
    inputs = _long_kv_test_inputs(
        s_q, s_k, requests=requests, out_of_range=out_of_range
    )
    inside = _window_mask(inputs)
    lengths = inside.sum(dim=1)
    offsets = lengths.cumsum(0) - lengths
    packed = torch.empty(int(lengths.sum()), device="cuda", dtype=torch.float32)
    dense = torch.full(inside.shape, float("nan"), device="cuda", dtype=torch.float32)
    fp8_mqa_logits(*inputs, clean_logits=False, out=dense)
    clean = fp8_mqa_logits(*inputs, clean_logits=True)
    fp8_mqa_logits(*inputs, out=packed, out_offsets=offsets)
    assert mqa_launches == ["rows8", "existing", "existing"]
    assert torch.equal(dense[inside], clean[inside])
    assert torch.equal(dense[inside], packed)


def _request_batch_inputs(layout):
    """FP8 inputs for a batch of requests, (rows, keys) each, one after the
    other. Row i of a request sees its first ``keys - rows + i + 1`` keys;
    ``keys > rows`` is a resumed request."""
    s_q, s_k = sum(rows for rows, _ in layout), sum(keys for _, keys in layout)
    q = torch.randn(s_q, 32, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, 128, device="cuda", dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    weights = torch.randn(s_q, 32, device="cuda", dtype=torch.float32)
    starts, ends, base = [], [], 0
    for rows, keys in layout:
        row = torch.arange(rows, dtype=torch.int32, device="cuda")
        starts.append(torch.full_like(row, base))
        ends.append(base + keys - rows + row + 1)
        base += keys
    return q.to(e4m3_type), kv_fp8, scales, weights, torch.cat(starts), torch.cat(ends)


@requires_rows8
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param([(1022, 8192)] * 4, id="resumed-requests"),
        pytest.param([(3000, 3000), (500, 9000), (596, 600)], id="resumed-among-fresh"),
        pytest.param(
            [(1138, 1138), (2315, 2315), (118, 118), (4417, 4417), (204, 9227)],
            id="short-resumed-last",
        ),
        pytest.param([(2429, 2429), (1667, 1667)], id="two-fresh"),
        pytest.param([(9, 64)] * 448, id="two-requests-in-each-block"),
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_rows8_request_batches(layout, mqa_launches):
    """Requests of arbitrary lengths in one call, resumed and fresh, with
    request boundaries inside eight-row blocks: the eight-row scores equal the
    clean scores bit for bit and every key of a window is written."""
    torch.manual_seed(0)
    inputs = _request_batch_inputs(layout)
    inside = _window_mask(inputs)
    dense = torch.full(inside.shape, float("nan"), device="cuda", dtype=torch.float32)
    fp8_mqa_logits(*inputs, clean_logits=False, out=dense)
    clean = fp8_mqa_logits(*inputs, clean_logits=True)
    assert mqa_launches == ["rows8", "existing"]
    assert torch.equal(dense[inside], clean[inside])


@requires_rows8
@pytest.mark.parametrize(
    "change",
    [
        "kv_stride",
        "kv_scales_stride",
        "cu_starts_stride",
        "cu_ends_stride",
        "kv_scales_dtype",
        "packed",
        "row_pitch",
    ],
)
@torch.inference_mode()
def test_fp8_mqa_logits_rows8_keeps_existing_route(change, rows8_forced, mqa_launches):
    """An input that the eight-row kernel cannot read selects the existing kernel."""
    torch.manual_seed(0)
    s_q, s_k = 8, 4097
    inputs = list(_long_kv_test_inputs(s_q, s_k))
    kwargs = {}
    if change == "kv_stride":
        inputs[1] = _strided_test_tensor(inputs[1])
    elif change == "kv_scales_stride":
        inputs[2] = _strided_test_tensor(inputs[2])
    elif change == "cu_starts_stride":
        inputs[4] = _strided_test_tensor(inputs[4])
    elif change == "cu_ends_stride":
        inputs[5] = _strided_test_tensor(inputs[5])
    elif change == "kv_scales_dtype":
        inputs[2] = inputs[2].double()
    elif change == "packed":
        kwargs["out"] = torch.empty(s_q * s_k, device="cuda", dtype=torch.float32)
        kwargs["out_offsets"] = torch.arange(s_q, device="cuda") * s_k
    else:
        # Seven rows of this pitch do not fit the 32-bit store offsets.
        pitch = 2**31 // 28 + 1
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * s_q * pitch * 4:
            pytest.skip("needs a 2.3 GiB caller buffer")
        storage = torch.empty(s_q * pitch, device="cuda", dtype=torch.float32)
        kwargs["out"] = storage.view(s_q, pitch)[:, :s_k]
    mqa_launches.probe = True
    with pytest.raises(_RouteProbe, match="existing"):
        fp8_mqa_logits(*inputs, clean_logits=False, **kwargs)


@requires_rows8
@pytest.mark.parametrize(
    "s_k,window_length",
    [(4097, length) for length in (1, 63, 64, 65, 127, 128, 129)]
    + [(65537, 129), (65537, 65501)],
)
@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_nonzero_union(
    s_k, window_length, rows8_forced, mqa_launches
):
    """Check narrow and split KV ranges without a full-range query row.

    The output starts as NaN, so a key that the kernel does not write fails the
    reference check.
    """
    torch.manual_seed(0)
    inputs = _long_kv_test_inputs(8, s_k)
    inputs[-2].fill_(17)
    inputs[-1].fill_(17 + window_length)
    out = torch.full((8, s_k), float("nan"), device="cuda", dtype=torch.float32)
    fp8_mqa_logits(*inputs, clean_logits=False, out=out)
    assert mqa_launches == ["rows8"]
    _check_long_kv_logits(out, inputs)


@requires_rows8
@torch.inference_mode()
def test_fp8_mqa_logits_rows8_skips_empty_windows(rows8_forced, mqa_launches):
    """An empty window adds no keys to the walk of its row group.

    The kernel then writes nothing in the columns that no row of the group needs.
    """
    torch.manual_seed(0)
    s_k = 65537
    inputs = _long_kv_test_inputs(8, s_k)
    inputs[-2].fill_(s_k - 1)
    inputs[-1].fill_(s_k)
    inputs[-2][3] = 0
    inputs[-1][3] = 0
    out = torch.full((8, s_k), float("nan"), device="cuda", dtype=torch.float32)
    fp8_mqa_logits(*inputs, clean_logits=False, out=out)
    assert mqa_launches == ["rows8"]
    _check_long_kv_logits(out, inputs)
    assert torch.isnan(out[:, : s_k - 1]).all()


@requires_rows8
@pytest.mark.parametrize("s_q,s_k,forced", [(16, 4223, True), (4096, 4097, False)])
@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_caller_buffer(s_q, s_k, forced, request, mqa_launches):
    """The eight-row kernel writes the same scores into a wider caller buffer."""
    if forced:
        request.getfixturevalue("rows8_forced")
    torch.manual_seed(0)
    inputs = _long_kv_test_inputs(s_q, s_k)
    storage = torch.full((s_q, s_k + 64), 7.0, dtype=torch.float32, device="cuda")
    out = storage[:, :s_k]
    logits = fp8_mqa_logits(*inputs, clean_logits=False, out=out)
    assert mqa_launches == ["rows8"]
    assert logits.data_ptr() == out.data_ptr()
    _check_long_kv_logits(out, inputs)
    assert torch.all(storage[:, s_k:] == 7.0)


@requires_rows8
@torch.inference_mode()
def test_fp8_mqa_logits_long_kv_graph_replay(rows8_forced, mqa_launches):
    """Alternate two graphs with fresh inputs and changing valid windows."""
    torch.manual_seed(0)
    states = []
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        inputs = _long_kv_test_inputs(16, 4223)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                fp8_mqa_logits(*inputs, clean_logits=False)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = fp8_mqa_logits(*inputs, clean_logits=False)
        states.append((graph, inputs, logits))
    assert set(mqa_launches) == {"rows8"}

    windows = ("full", "ragged", "empty", "full", "ragged", "full")
    for step, window in enumerate(windows):
        graph, inputs, logits = states[step % 2]
        fresh = _long_kv_test_inputs(16, 4223)
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


@requires_rows8
@pytest.mark.parametrize("s_q,s_k", [(0, 4097), (8, 0), (0, 0)])
@pytest.mark.parametrize("clean_logits", [False, True])
@torch.inference_mode()
def test_fp8_mqa_logits_rows8_empty_output(s_q, s_k, clean_logits, mqa_launches):
    """An empty output does not reach the eight-row kernel."""
    q = torch.empty(s_q, 32, 128, device="cuda", dtype=e4m3_type)
    kv = torch.empty(s_k, 128, device="cuda", dtype=e4m3_type)
    scales = torch.empty(s_k, device="cuda", dtype=torch.float32)
    weights = torch.empty(s_q, 32, device="cuda", dtype=torch.float32)
    starts = torch.zeros(s_q, device="cuda", dtype=torch.int32)
    ends = torch.zeros_like(starts)
    logits = fp8_mqa_logits(q, kv, scales, weights, starts, ends, clean_logits)
    assert "rows8" not in mqa_launches
    assert logits.shape == (s_q, s_k)
    assert logits.dtype == torch.float32
    assert logits.device == q.device


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
