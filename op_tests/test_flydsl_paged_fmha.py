# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Native paged FP8 correctness checks on gfx950."""

import math
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.flydsl import flydsl_flash_attn_paged_fp8_func
from aiter.ops.flydsl.kernels import flash_attn_paged_fp8_func_gfx950 as paged
from aiter.test_common import checkAllclose

# Match the paged-FMHA benchmark for active rows; masked rows stay exact.
FP8_RTOL = 0.02
FP8_ATOL = 0.02

LAYOUTS = [
    (1, "linear3d"),
    (1, "linear"),
    (16, "vectorized"),
    (64, "vectorized"),
    (1024, "vectorized"),
]
DIMS = [(128, 128), (192, 128), (192, 192)]
_arch = (
    torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if torch.cuda.is_available()
    else ""
)
gfx950 = pytest.mark.skipif(
    _arch != "gfx950", reason="native paged FP8 requires gfx950"
)


def _quantize(x):
    scale = (x.abs().max() / 448).clamp_min(1e-8).reshape(1)
    return (x / scale).to(torch.float8_e4m3fn), scale


def make_case(
    page,
    layout,
    d,
    dv,
    *,
    qlens=(300, 65, 17),
    klens=(513, 193, 81),
    heads=(6, 2),
    seed=17,
):
    """Build independent physical caches and a packed query batch."""
    torch.manual_seed(seed)
    hq, hkv = heads
    counts = [(n + page - 1) // page for n in klens]
    maxkv = ((max(klens, default=0) + 127) // 128) * 128
    capacity = (maxkv + page - 1) // page
    npages = max(1, sum(counts))
    ids = torch.randperm(npages, device="cuda", dtype=torch.int32)
    table = torch.full((len(qlens), capacity), -1, device="cuda", dtype=torch.int32)
    offset = 0
    for b, count in enumerate(counts):
        table[b, :count] = ids[offset : offset + count]
        offset += count
    q, qs = _quantize(torch.randn(max(sum(qlens), 1), hq, d, device="cuda") * 0.2)
    q = q[: sum(qlens)]
    klin, ks = _quantize(torch.randn(npages, page, hkv, d, device="cuda") * 0.2)
    vlin, vs = _quantize(torch.randn(npages, page, hkv, dv, device="cuda") * 0.2 + 0.25)
    if layout == "vectorized":
        k = (
            klin.view(npages, page, hkv, d // 16, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        v = (
            vlin.view(npages, page // 16, 16, hkv, dv)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
        )
    elif layout == "linear3d":
        k, v = klin[:, 0], vlin[:, 0]
    else:
        k, v = klin, vlin
    cuq = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor(klens, dtype=torch.int32, device="cuda")
    output = torch.full(
        (sum(qlens), hq, dv), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    return SimpleNamespace(
        page=page,
        layout=layout,
        d=d,
        dv=dv,
        hq=hq,
        hkv=hkv,
        q=q,
        k=k,
        v=v,
        qs=qs,
        ks=ks,
        vs=vs,
        cuq=cuq,
        lengths=lengths,
        table=table,
        qlens=list(qlens),
        klens=list(klens),
        maxq=max(qlens, default=0),
        maxkv=maxkv,
        out=output,
    )


def reference(case, *, scale=None, lengths=None):
    """Logical token reconstruction and FP32 attention, without kernel helpers."""
    scale = case.d**-0.5 if scale is None else scale
    lengths = case.klens if lengths is None else lengths
    output = []
    offset = 0
    for b, (qlen, klen) in enumerate(zip(case.qlens, lengths)):
        if klen == 0:
            output.append(torch.zeros(qlen, case.hq, case.dv, device="cuda"))
            offset += qlen
            continue
        physical = case.table[b, : (klen + case.page - 1) // case.page].long()
        k, v = case.k[physical], case.v[physical]
        if case.layout == "vectorized":
            k, v = k.permute(0, 3, 1, 2, 4), v.permute(0, 2, 4, 1, 3)
        k = k.reshape(-1, case.hkv, case.d)[:klen].float() * case.ks
        v = v.reshape(-1, case.hkv, case.dv)[:klen].float() * case.vs
        q = case.q[offset : offset + qlen].float() * case.qs
        offset += qlen
        k = k.repeat_interleave(case.hq // case.hkv, dim=1)
        v = v.repeat_interleave(case.hq // case.hkv, dim=1)
        scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2) * scale
        allowed = (
            torch.arange(klen, device="cuda")[None, :]
            <= torch.arange(qlen, device="cuda")[:, None] + klen - qlen
        )
        probability = torch.softmax(
            scores.masked_fill(~allowed, float("-inf")), dim=-1
        ).nan_to_num(0)
        output.append((probability @ v.transpose(0, 1)).transpose(0, 1))
    return (
        torch.cat(output) if output else torch.empty_like(case.out, dtype=torch.float32)
    )


def run_case(case, **kwargs):
    options = {
        "block_table": case.table,
        "seqlen_k": case.lengths,
        "q_descale": case.qs,
        "k_descale": case.ks,
        "v_descale": case.vs,
        "out": case.out,
    }
    options.update(kwargs)
    return flydsl_flash_attn_paged_fp8_func(
        case.q, case.k, case.v, case.cuq, case.maxq, case.maxkv, **options
    )


def csr_metadata(case, prefix=0):
    counts = [(length + case.page - 1) // case.page for length in case.klens]
    offsets = [prefix]
    chunks = [torch.full((prefix,), -1, dtype=torch.int32, device=case.q.device)]
    for batch, count in enumerate(counts):
        offsets.append(offsets[-1] + count)
        chunks.append(case.table[batch, :count])
    chunks.append(torch.full((3,), -1, dtype=torch.int32, device=case.q.device))
    last = [(length - 1) % case.page + 1 if length else 0 for length in case.klens]
    return {
        "block_table": None,
        "kv_indptr": torch.tensor(offsets, dtype=torch.int32, device=case.q.device),
        "kv_page_indices": torch.cat(chunks),
        "kv_last_page_lens": torch.tensor(
            last, dtype=torch.int32, device=case.q.device
        ),
    }


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_csr_ragged_and_odd_page_bases(page, layout, d, dv):
    case = make_case(
        page,
        layout,
        d,
        dv,
        qlens=(0, 65, 300, 17),
        klens=(33, 0, max(129, page + 1), 65),
    )
    options = csr_metadata(case, prefix=5)
    check_case(case, **options)


@gfx950
@pytest.mark.parametrize("layout", ["linear3d", "linear"])
@pytest.mark.parametrize("d,dv", DIMS)
def test_csr_page1_optional_last_lengths(layout, d, dv):
    case = make_case(1, layout, d, dv)
    options = csr_metadata(case)
    explicit = check_case(case, **options).clone()
    options["kv_last_page_lens"] = None
    torch.testing.assert_close(check_case(case, **options), explicit, rtol=0, atol=0)


def check_accuracy(case, actual, expected, *, msg="paged FP8"):
    """Check masked rows exactly and all active rows with the benchmark tolerance."""
    offset = 0
    actual, expected = actual.float(), expected.float()
    for batch, (qlen, klen) in enumerate(zip(case.qlens, case.klens)):
        masked = min(qlen, max(0, qlen - klen))
        for begin, end, rtol, atol in (
            (0, masked, 0, 0),
            (masked, qlen, FP8_RTOL, FP8_ATOL),
        ):
            if begin == end:
                continue
            rows = slice(offset + begin, offset + end)
            context = f"{msg}, batch={batch}, rows[{rows.start}:{rows.stop}]: "
            error = checkAllclose(
                actual[rows],
                expected[rows],
                rtol=rtol,
                atol=atol,
                tol_err_ratio=0,
                msg=context,
            )
            assert error == 0, (
                f"{context}{error:.2%} of elements outside tolerance "
                f"(rtol={rtol}, atol={atol})"
            )
        offset += qlen
    assert offset == actual.shape[0] == expected.shape[0]
    return 0.0


def check_case(case, **kwargs):
    actual = run_case(case, **kwargs)
    torch.cuda.synchronize()
    assert actual is kwargs.get("out", case.out)
    assert bool(actual.isfinite().all())
    check_accuracy(
        case,
        actual,
        reference(case, scale=kwargs.get("softmax_scale")),
    )
    return actual


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_native_layouts_ragged_multiblock(page, layout, d, dv):
    check_case(make_case(page, layout, d, dv))


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_empty_requests_and_fully_masked_rows(page, layout, d, dv):
    case = make_case(
        page, layout, d, dv, qlens=(0, 65, 300, 17), klens=(33, 0, 129, 65)
    )
    actual = check_case(case)
    assert torch.count_nonzero(actual[:65]) == 0
    assert torch.count_nonzero(actual[65 : 65 + 171]) == 0


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_runtime_scale_reuses_compiled_launcher(page, layout, d, dv):
    case = make_case(page, layout, d, dv)
    default = check_case(case).clone()
    misses = paged._build.cache_info().misses
    for scale in (case.d**-0.5, 0.037, 0.137):
        actual = check_case(case, softmax_scale=scale)
        if scale == case.d**-0.5:
            torch.testing.assert_close(actual, default, rtol=0, atol=0)
    assert paged._build.cache_info().misses == misses


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS[:3])
@pytest.mark.parametrize("d,dv", DIMS)
def test_bounded_matches_wide_pointer_cache(monkeypatch, page, layout, d, dv):
    case = make_case(page, layout, d, dv)
    bounded = check_case(case).clone()
    build = paged._build

    def wide(**kwargs):
        assert kwargs["buffered"]
        kwargs["buffered"] = False
        return build(**kwargs)

    monkeypatch.setattr(paged, "_build", wide)
    torch.testing.assert_close(check_case(case), bounded, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [(1, "linear3d", 128, 128), (1, "linear", 192, 128), (16, "vectorized", 192, 192)],
)
def test_wide_cache_with_byte_offset_base(monkeypatch, page, layout, d, dv):
    case = make_case(page, layout, d, dv)

    def offset_copy(tensor):
        storage = torch.empty(
            tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device
        )
        view = storage[1:].view(tensor.shape)
        view.copy_(tensor)
        assert view.data_ptr() % 4 != 0
        return view

    case.k, case.v = offset_copy(case.k), offset_copy(case.v)
    build = paged._build

    def wide(**kwargs):
        kwargs["buffered"] = False
        return build(**kwargs)

    monkeypatch.setattr(paged, "_build", wide)
    check_case(case)


@gfx950
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("heads", [(16, 1), (8, 4)])
def test_scalar_and_paired_page64_paths(d, dv, heads):
    case = make_case(64, "vectorized", d, dv, heads=heads)
    even = check_case(case).clone()
    case.maxkv -= 64
    torch.testing.assert_close(check_case(case), even, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [
        (1, "linear3d", 192, 192),
        (16, "vectorized", 192, 128),
        (64, "vectorized", 128, 128),
        (1024, "vectorized", 192, 192),
    ],
)
def test_graph_reads_updated_lengths(page, layout, d, dv):
    case = make_case(page, layout, d, dv, qlens=(17,), klens=(max(257, page + 1),))
    run_case(case, softmax_scale=0.137)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_case(case, softmax_scale=0.137)
    assert captured is case.out
    for length in (65, 129, case.klens[0], 0, 33):
        case.lengths[0] = length
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(case.out.isfinite().all())
        torch.testing.assert_close(
            case.out.float(),
            reference(case, scale=0.137, lengths=[length]),
            rtol=FP8_RTOL,
            atol=FP8_ATOL,
        )


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
def test_strided_metadata_and_scalar_descales(page, layout):
    case = make_case(page, layout, 192, 128)
    case.cuq = case.cuq.repeat_interleave(2)[::2]
    case.lengths = case.lengths.repeat_interleave(2)[::2]
    case.table = case.table.repeat_interleave(2, dim=1)[:, ::2]
    expected = reference(case)
    actual = run_case(
        case,
        q_descale=case.qs.reshape(()),
        k_descale=case.ks.as_strided((1,), (0,)),
        v_descale=case.vs.reshape(1, 1),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=FP8_RTOL, atol=FP8_ATOL)


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [
        (1, "linear", 128, 128),
        (1, "linear3d", 192, 128),
        (16, "vectorized", 192, 192),
        (64, "vectorized", 192, 128),
        (1024, "vectorized", 192, 192),
    ],
)
def test_copies_follow_nondefault_stream(page, layout, d, dv):
    case = make_case(page, layout, d, dv)
    expected = check_case(case).clone()

    def strided(tensor):
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2),
            dtype=tensor.dtype,
            device="cuda",
        )
        view = storage[..., ::2]
        view.copy_(tensor)
        return view

    case.q, case.k, case.v = map(strided, (case.q, case.k, case.v))
    stream = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    run_case(case, stream=stream)
    stream.synchronize()
    torch.cuda._sleep(3_000_000_000)
    blocked = torch.cuda.Event()
    blocked.record()
    run_case(case, stream=stream)
    stream.synchronize()
    assert not blocked.query(), "copies waited on the blocked default stream"
    torch.cuda.synchronize()
    torch.testing.assert_close(case.out, expected, rtol=0, atol=0)


@gfx950
def test_side_stream_keeps_copy_source_alive():
    case = make_case(64, "vectorized", 128, 128)
    stream = torch.cuda.Stream()
    expected = check_case(case, stream=stream).clone()
    storage_shape = (*case.v.shape[:-1], case.v.shape[-1] * 2)
    storage = torch.empty(storage_shape, dtype=case.v.dtype, device="cuda")
    strided = storage[..., ::2]
    strided.copy_(case.v)
    case.v = strided
    source_begin = storage.data_ptr()
    source_end = source_begin + storage.numel() * storage.element_size()
    del storage, strided
    torch.cuda.synchronize()

    # Delay the copy so the caller can release its source and pressure the
    # creation-stream allocator before the side stream consumes the bytes.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(3_000_000_000)
    run_case(case, stream=stream)
    del case.v
    replacements = []
    source_reused = False
    pending = False
    try:
        for _ in range(32):
            replacement = torch.empty(
                storage_shape, dtype=torch.float8_e4m3fn, device="cuda"
            )
            replacements.append(replacement)
            if not stream.query():
                pending = True
                begin = replacement.data_ptr()
                end = begin + replacement.numel() * replacement.element_size()
                source_reused |= begin < source_end and end > source_begin
            replacement.zero_()
    finally:
        torch.cuda.synchronize()
    assert pending, "side stream completed before allocator pressure"
    assert not source_reused, "copy source was recycled while side stream was pending"
    torch.testing.assert_close(case.out, expected, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("rectangular", [False, True])
def test_public_batch_prefill_routes_to_flydsl(
    monkeypatch, page, layout, d, dv, rectangular
):
    from aiter.ops import mha

    case = make_case(page, layout, d, dv)
    metadata = csr_metadata(case, prefix=5)
    indptr, indices = metadata.pop("kv_indptr"), metadata.pop("kv_page_indices")
    if rectangular:
        metadata["block_table"] = case.table
        indices.fill_(-1)  # The rectangular table must take precedence.
    seen = []
    direct = paged.flydsl_flash_attn_paged_fp8_func

    def observed(*args, **kwargs):
        seen.append(kwargs)
        return direct(*args, **kwargs)

    def unexpected_ck(*args, **kwargs):
        raise AssertionError("supported paged FP8 request reached CK")

    monkeypatch.setattr(paged, "flydsl_flash_attn_paged_fp8_func", observed)
    monkeypatch.setattr(mha, "_mha_batch_prefill", unexpected_ck)
    actual = mha.mha_batch_prefill_func(
        case.q,
        case.k,
        case.v,
        case.cuq,
        indptr,
        indices,
        case.maxq,
        case.maxkv,
        causal=True,
        softmax_scale=0.137,
        out=case.out,
        seqlen_k=case.lengths,
        q_descale=case.qs,
        k_descale=case.ks,
        v_descale=case.vs,
        **metadata,
    )
    torch.cuda.synchronize()
    assert actual is case.out and len(seen) == 1
    torch.testing.assert_close(
        actual.float(), reference(case, scale=0.137), rtol=FP8_RTOL, atol=FP8_ATOL
    )


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv",
    [
        (1, "linear3d", 192, 128),
        (16, "vectorized", 192, 192),
        (64, "vectorized", 128, 128),
        (1024, "vectorized", 192, 128),
    ],
)
def test_csr_graph_updates_ranges_and_lengths(page, layout, d, dv):
    case = make_case(
        page, layout, d, dv, qlens=(17, 65), klens=(max(513, page + 1), 81)
    )
    metadata = csr_metadata(case, prefix=5)
    run_case(case, **metadata, softmax_scale=0.137)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_case(case, **metadata, softmax_scale=0.137)
    for first_length in (65, 129, case.klens[0], 0, 33):
        lengths = [first_length, 81]
        counts = [(n + page - 1) // page for n in lengths]
        offsets = [5, 5 + counts[0], 5 + sum(counts)]
        metadata["kv_indptr"].copy_(
            torch.tensor(offsets, dtype=torch.int32, device="cuda")
        )
        metadata["kv_last_page_lens"].copy_(
            torch.tensor(
                [(n - 1) % page + 1 if n else 0 for n in lengths],
                dtype=torch.int32,
                device="cuda",
            )
        )
        metadata["kv_page_indices"].fill_(-1)
        for batch, count in enumerate(counts):
            metadata["kv_page_indices"][offsets[batch] : offsets[batch + 1]].copy_(
                case.table[batch, :count]
            )
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(case.out.isfinite().all())
        torch.testing.assert_close(
            case.out.float(),
            reference(case, scale=0.137, lengths=lengths),
            rtol=FP8_RTOL,
            atol=FP8_ATOL,
        )


@gfx950
@pytest.mark.parametrize("page,layout", [(1, "linear3d"), (64, "vectorized")])
def test_csr_strided_metadata_on_side_stream(page, layout):
    case = make_case(page, layout, 192, 128)
    metadata = csr_metadata(case, prefix=5)
    for name in ("kv_indptr", "kv_page_indices", "kv_last_page_lens"):
        metadata[name] = metadata[name].repeat_interleave(2)[::2]
    stream = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    run_case(case, **metadata, stream=stream)
    stream.synchronize()
    torch.cuda._sleep(3_000_000_000)
    blocked = torch.cuda.Event()
    blocked.record()
    run_case(case, **metadata, stream=stream)
    stream.synchronize()
    assert not blocked.query(), "CSR normalization waited on the default stream"
    torch.cuda.synchronize()
    torch.testing.assert_close(
        case.out.float(), reference(case), rtol=FP8_RTOL, atol=FP8_ATOL
    )


@gfx950
@pytest.mark.parametrize("page,layout", [LAYOUTS[i] for i in (0, 2, 3, 4)])
@pytest.mark.parametrize("d,dv", DIMS)
def test_lazy_and_eager_match_reference(page, layout, d, dv):
    # Layout/metadata coverage uses lazy mode above. Check both softmax modes
    # for each page size/width without duplicating the rank/metadata product.
    case = make_case(page, layout, d, dv)
    options = csr_metadata(case, prefix=5)
    lazy = check_case(case, **options).clone()
    eager = check_case(case, **options, dualwave_swp_lazy_rescale=False)
    torch.testing.assert_close(eager, lazy, rtol=FP8_RTOL, atol=FP8_ATOL)


def _large_case(page, layout, d, dv, high_page):
    hkv = 2
    npages = high_page + 1
    needed = npages * page * hkv * (d + dv)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < needed + 2 * 2**30:
        pytest.skip(f"requires {needed / 2**30:.1f} GiB cache plus 2 GiB headroom")
    case = make_case(page, layout, d, dv, qlens=(1,), klens=(1,))
    if layout == "vectorized":
        kshape = (npages, hkv, d // 16, page, 16)
        vshape = (npages, hkv, page // 16, dv, 16)
    elif layout == "linear3d":
        kshape, vshape = (npages, hkv, d), (npages, hkv, dv)
    else:
        kshape, vshape = (npages, page, hkv, d), (npages, page, hkv, dv)
    case.k = torch.empty(kshape, dtype=torch.float8_e4m3fn, device="cuda")
    case.v = torch.empty(vshape, dtype=torch.float8_e4m3fn, device="cuda")
    case.k[high_page].zero_()
    case.v[high_page].fill_(float("nan"))
    first = (1.0 + torch.arange(hkv * dv, device="cuda").reshape(hkv, dv) % 7 / 8).to(
        torch.float8_e4m3fn
    )
    if layout == "vectorized":
        case.v[high_page, :, 0, :, 0].copy_(first)
    else:
        case.v[high_page].reshape(page, hkv, dv)[0].copy_(first)
    case.vs.fill_(1.0)
    case.table.fill_(-1)
    case.table[0, 0] = high_page
    expected = first.float().repeat_interleave(case.hq // hkv, dim=0)[None]
    return case, expected


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv,csr",
    [
        (1, "linear3d", 128, 128, False),
        (1, "linear", 192, 128, True),
        (16, "vectorized", 192, 192, True),
        (64, "vectorized", 192, 128, False),
        (1024, "vectorized", 128, 128, True),
    ],
)
def test_cache_offsets_above_4gib(page, layout, d, dv, csr):
    high_page = math.ceil(2**32 / (page * 2 * min(d, dv)))
    case, expected = _large_case(page, layout, d, dv, high_page)
    options = {}
    if csr:
        options = {
            "block_table": None,
            "kv_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            "kv_page_indices": torch.tensor(
                [high_page], dtype=torch.int32, device="cuda"
            ),
            "kv_last_page_lens": torch.tensor([1], dtype=torch.int32, device="cuda"),
        }
    actual = run_case(case, **options)
    torch.cuda.synchronize()
    assert high_page * page * case.hkv * d >= 2**32
    assert high_page * page * case.hkv * dv >= 2**32
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(actual.float(), expected, rtol=FP8_RTOL, atol=FP8_ATOL)


@gfx950
@pytest.mark.parametrize("csr", [False, True])
def test_page16_scalar_offset_near_i32_limit(monkeypatch, csr):
    page, d, dv = 16, 192, 192
    high_page = paged.PAGED_FP8_BUFFER_LIMIT_BYTES // (page * 2 * d) - 1
    case, expected = _large_case(page, "vectorized", d, dv, high_page)
    assert case.k.numel() <= paged.PAGED_FP8_BUFFER_LIMIT_BYTES
    assert high_page * page * case.hkv * d > 2**31 - 16384
    original = paged._build
    selected = []

    def build(**kwargs):
        selected.append(kwargs["buffered"])
        return original(**kwargs)

    monkeypatch.setattr(paged, "_build", build)
    options = {}
    if csr:
        options = {
            "block_table": None,
            "kv_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            "kv_page_indices": torch.tensor(
                [high_page], dtype=torch.int32, device="cuda"
            ),
            "kv_last_page_lens": torch.tensor([1], dtype=torch.int32, device="cuda"),
        }
    actual = run_case(case, **options)
    torch.cuda.synchronize()
    assert selected == [True]
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(actual.float(), expected, rtol=FP8_RTOL, atol=FP8_ATOL)


def _native_cache(case, key, value):
    pages = key.shape[0]
    if case.layout == "vectorized":
        key = key.view(pages, case.page, case.hkv, case.d // 16, 16).permute(
            0, 2, 3, 1, 4
        )
        value = value.view(pages, case.page // 16, 16, case.hkv, case.dv).permute(
            0, 3, 1, 4, 2
        )
    elif case.layout == "linear3d":
        key, value = key[:, 0], value[:, 0]
    case.k, case.v = key.contiguous(), value.contiguous()


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
@pytest.mark.parametrize("csr", [False, True])
def test_inactive_nan_bytes_and_empty_request(page, layout, d, dv, csr):
    # Cross token groups, 64/128-token tiles and the 1024-token page boundary
    # through the real attention pipeline, with inactive K/V bytes poisoned.
    for length in (1, 16, 17, 63, 64, 65, 127, 128, 129, 513, 1025):
        case = make_case(page, layout, d, dv, qlens=(17, 17), klens=(length, 0))
        case.q.zero_()
        key = torch.full(
            (case.k.shape[0], page, case.hkv, d),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        value = torch.full(
            (case.k.shape[0], page, case.hkv, dv),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        count = (length + page - 1) // page
        logical_key, logical_value = torch.full_like(
            key, float("nan")
        ), torch.full_like(value, float("nan"))
        logical_key.view(-1, case.hkv, d)[:length].zero_()
        logical_value.view(-1, case.hkv, dv)[:length].fill_(1)
        physical = case.table[0, :count].long()
        key[physical], value[physical] = logical_key[:count], logical_value[:count]
        _native_cache(case, key.to(case.q.dtype), value.to(case.q.dtype))
        case.qs.fill_(1)
        case.ks.fill_(1)
        case.vs.fill_(1)
        actual = run_case(case, **(csr_metadata(case, prefix=3) if csr else {}))
        expected = torch.zeros_like(actual)
        expected[max(0, 17 - length) : 17].fill_(1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@gfx950
@pytest.mark.parametrize("page,layout", LAYOUTS)
@pytest.mark.parametrize("d,dv", DIMS)
def test_shared_pages_loose_maxima_and_output_canaries(page, layout, d, dv):
    case = make_case(page, layout, d, dv, qlens=(17, 65), klens=(129, 65))
    count = (65 + page - 1) // page
    case.table[1, :count] = case.table[0, :count]
    # Oversized backing buffers make a broken bound observable as a canary
    # failure instead of an out-of-allocation memory access.
    case.maxq = 512
    query = torch.zeros((1024, case.hq, d), dtype=case.q.dtype, device="cuda")
    query[:82].copy_(case.q)
    case.q = query[:82]
    storage = torch.full((1026, case.hq, dv), 123, dtype=torch.bfloat16, device="cuda")
    case.out = storage[1:83]
    for csr in (False, True):
        case.out.fill_(float("nan"))
        check_case(case, **(csr_metadata(case, prefix=1) if csr else {}))
        assert bool((storage[:1] == 123).all())
        assert bool((storage[83:] == 123).all())


@gfx950
@pytest.mark.parametrize("mode", ["bounded", "escape", "mixed-waves", "negative"])
@pytest.mark.parametrize("csr", [False, True])
def test_d128_query_bound_preserves_rescaling(mode, csr):
    batch, hkv = {
        "bounded": (2, 1),
        "escape": (3, 2),
        "mixed-waves": (3, 2),
        "negative": (5, 1),
    }[mode]
    case = make_case(
        64,
        "vectorized",
        128,
        128,
        qlens=(300, 65, 257, 33, 127)[:batch],
        klens=(1024, 512, 768, 256, 384)[:batch],
        heads=(16, hkv),
        seed=29,
    )
    query = torch.zeros_like(case.q, dtype=torch.float32)
    key = torch.zeros_like(case.k, dtype=torch.float32)
    signs = torch.where(torch.arange(hkv, device="cuda") % 2 == 0, 1.0, -1.0)
    offset = 0
    for b, (qlen, klen) in enumerate(zip(case.qlens, case.klens)):
        rows = torch.arange(qlen, device="cuda")
        coefficients = torch.ones(qlen, device="cuda")
        if mode == "mixed-waves":
            coefficients = torch.where((rows // 32) % 2 == 0, 1.0, 4.0)
        query[offset : offset + qlen, :, 0] = coefficients[:, None]
        offset += qlen
        count = klen // 64
        levels = torch.where(
            (torch.arange(count, device="cuda") // 2) % 2 == 1, 1.0, -1.0
        )
        levels[:2] = 0
        if mode == "negative":
            levels.fill_(-1)
        key[case.table[b, :count].long(), :, 0, :, 0] = (
            levels[:, None, None] * signs[None, :, None] * 448
        )
    case.q, case.k = query.to(case.q.dtype), key.to(case.k.dtype)
    case.qs.fill_(1)
    # Use a non-default runtime scale: the query-bound proof must use the
    # same logit scale as the softmax, not an implicit rsqrt(D).
    scale = 0.137
    peak = {"bounded": 3.0, "escape": 16.0, "mixed-waves": 3.0, "negative": 2.5}[mode]
    case.ks.fill_(peak / (448 * scale * math.log2(math.e)))
    expected = reference(case, scale=scale)
    options = csr_metadata(case, prefix=5) if csr else {}
    for lazy in (True, False):
        actual = run_case(
            case, **options, softmax_scale=scale, dualwave_swp_lazy_rescale=lazy
        )
        assert bool(actual.isfinite().all())
        torch.testing.assert_close(
            actual.float(), expected, rtol=FP8_RTOL, atol=FP8_ATOL
        )


@gfx950
@pytest.mark.parametrize(
    "page,layout,d,dv,csr",
    [
        (1, "linear3d", 128, 128, False),
        (1, "linear", 192, 128, True),
        (16, "vectorized", 192, 192, True),
        (64, "vectorized", 192, 128, False),
        (1024, "vectorized", 128, 128, True),
    ],
)
def test_explicit_compile_then_launch(monkeypatch, page, layout, d, dv, csr):
    case = make_case(page, layout, d, dv, qlens=(65, 17), klens=(128, 97))
    original = paged._build
    original.cache_clear()
    compile_results = []

    def build(**kwargs):
        launcher = original(**kwargs)

        def run(*args, **options):
            before = case.out.view(torch.uint8).clone()
            compile_results.append(launcher.compile(*args, **options))
            torch.cuda.synchronize()
            torch.testing.assert_close(
                case.out.view(torch.uint8), before, rtol=0, atol=0
            )
            return launcher(*args, **options)

        return run

    monkeypatch.setattr(paged, "_build", build)
    for _ in range(2):  # Cold preload, then preload after a real launch.
        case.out.fill_(123)
        check_case(case, **(csr_metadata(case, prefix=1) if csr else {}))
    assert len(compile_results) == 2  # Older no-dispatch runtimes return None.
