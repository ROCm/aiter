# A8K4V4PT fp4pt V fused into the gfx1250 fused Q/K norm + MRoPE + cache kernels (2026-11-16):
#  - V bytes and scales == fp4pt_quant_v_cache (bit-exact), slots never written stay untouched
#  - Q (fp8) / k_out / K cache == the A8K4V8 fused kernel's (bitwise)
#  over token counts on both sides of the row-kernel / K-Q-V-kernel crossover, with each path forced, TDM full-tile
#  stores on and off, packed sequences starting mid-tile / straddling pages (non-contiguous pages), pad slots, a
#  slot table longer than the batch, and V as a strided view into qkv or a separate contiguous tensor.
import random

import pytest
import torch

import aiter.ops.triton.fusions.fused_qk_norm_mrope as fm
from aiter.ops.triton.fusions.fused_qk_norm_mrope import (
    fused_qk_norm_mrope_cache,
    fused_qk_norm_mrope_cache_k4v4pt,
)
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.shuffle import shuffle_weight

# Reference fp4pt V writer, kept here so this test stands alone (the same as fp4pt_quant_v_cache in the A8K4V4PT
# unified-attention tests).
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _e2m1_rne_codes(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 codes (uint8, 0..15), round to nearest even, saturating at 6."""
    mag = x.abs().contiguous()
    bounds = torch.tensor(_E2M1_BOUNDS, dtype=x.dtype, device=x.device)
    # ties go to the lower code; a tie on an odd code's upper bound goes up, to the even code
    code = torch.bucketize(mag, bounds, right=False)
    tie = mag == bounds[code.clamp(max=len(_E2M1_BOUNDS) - 1)]
    code = code + (tie & (code % 2 == 1)).to(code.dtype)
    return (code | ((x < 0).to(code.dtype) << 3)).to(torch.uint8)


def fp4pt_quant_v_cache(value_cache: torch.Tensor, ref: bool = False):
    """A8K4V4PT V cache layout. value_cache [num_blocks, block_size, num_kv_heads, head_size] ->
    [num_blocks, num_kv_heads, block_size, head_size // 2 + 4] torch.float4_e2m1fn_x2; one (block, kv head) row =
    [block_size * head_size / 2 B e2m1, shuffled as the NVFP4 V data (shuffle_weight, 16 x 16 B tiles) |
    block_size fp32 scales, token order]. Per (token, kv head): s = amax(|v|) / 6, e2m1 = rne(v / s)."""
    assert not ref
    num_blocks, block_size, num_kv_heads, head_size = value_cache.shape
    x = value_cache.permute(0, 2, 1, 3).to(torch.float32)  # [nb, kh, bs, d]
    scale = x.abs().amax(dim=-1, keepdim=True) / 6.0
    codes = _e2m1_rne_codes(x / torch.where(scale > 0, scale, 1.0))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)  # low nibble = even element
    data = shuffle_weight(packed, arch="gfx950").reshape(num_blocks, num_kv_heads, block_size * head_size // 2)
    tail = scale.reshape(num_blocks, num_kv_heads, block_size).contiguous()
    v = (
        torch.cat([data.view(torch.uint8), tail.view(torch.uint8)], dim=-1)
        .contiguous()
        .view(num_blocks, num_kv_heads, block_size, head_size // 2 + 4)
        .view(torch.float4_e2m1fn_x2)
    )
    return v, None

D, BS = 256, 128


def make_slots(t, num_blocks, layout, gen):
    """Slots for t tokens. 'aligned': sequences start at page offset 0 and run over whole pages (full tiles);
    'packed': several sequences of random lengths, each starting at a random page offset (mid-tile), running over
    randomly chosen (non-contiguous) pages, with ~3% pad slots (-1)."""
    rnd = random.Random(t * 31 + len(layout))
    pages = list(range(num_blocks))
    rnd.shuffle(pages)
    slots = []
    remaining = t
    while remaining > 0:
        n = remaining if layout == "aligned" else min(remaining, rnd.randint(1, max(1, t // 3 + 1)))
        start = 0 if layout == "aligned" else rnd.randrange(BS)
        pos = start
        page = pages.pop()
        for _ in range(n):
            if pos == BS:
                page = pages.pop()
                pos = 0
            slots.append(page * BS + pos)
            pos += 1
        remaining -= n
    s = torch.tensor(slots, dtype=torch.int64)
    if layout == "packed":
        pad = torch.rand(t, generator=torch.Generator().manual_seed(t)) < 0.03
        s[pad] = -1
    return s


def run_case(t, q_heads, k_heads, layout, v_contig, path, tdm, extra_slots=0):
    gen = torch.Generator(device="cuda").manual_seed(t * 7 + q_heads + (layout == "packed"))
    num_blocks = t + 2 * ((t + BS - 1) // BS) + 8  # every packed sequence may take its own pages
    qkv = torch.randn(t, (q_heads + 2 * k_heads) * D, generator=gen, device="cuda").to(torch.bfloat16)
    q = qkv[:, : q_heads * D]
    k = qkv[:, q_heads * D : (q_heads + k_heads) * D]
    v = qkv[:, (q_heads + k_heads) * D :]
    # a few rows with exact e2m1 ties and an all-zero row
    if t > 2:
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda").repeat(D // 8)
        v[1, :D] = ties.to(torch.bfloat16)
        v[2, :] = 0
    if v_contig:
        v = v.contiguous()
    qw = (torch.randn(D, generator=gen, device="cuda") * 0.1).to(torch.bfloat16)
    kw = (torch.randn(D, generator=gen, device="cuda") * 0.1).to(torch.bfloat16)
    pos = torch.randint(0, 8192, (3, t), generator=gen, device="cuda")
    inv = 1.0 / (10000 ** (torch.arange(0, 32, device="cuda").float() / 32))
    ang = torch.arange(8192, device="cuda").float()[:, None] * inv[None, :]
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    sec = [11, 11, 10]
    slots = make_slots(t, num_blocks, layout, gen).cuda()
    if extra_slots:
        slots = torch.cat([slots, torch.full((extra_slots,), -1, dtype=torch.int64, device="cuda")])
    # reference: the A8K4V8 fused kernel (Q, K) and the reference fp4pt writer (V)
    kc_a = torch.zeros(num_blocks, k_heads, BS, D // 2 + D // 16, dtype=torch.uint8, device="cuda")
    vc_a = torch.zeros(num_blocks, k_heads, BS // 16, D, 16, dtype=e4m3_dtype, device="cuda")
    qa, ka = fused_qk_norm_mrope_cache(q, k, v, qw, kw, 1e-6, pos, cos, sin, sec, kc_a, vc_a, slots, kv_scale=1.0)
    # the fused k4v4pt call, with the path forced
    old = (fm._K4V4PT_PAGE_MIN_TOKENS, fm._K4V4PT_V_TDM, fm._K4V8_ROWS_MIN_Q_HEADS)
    fm._K4V4PT_PAGE_MIN_TOKENS = 1 if path == "kqv" else 10**9
    fm._K4V4PT_V_TDM = tdm
    # the K / Q / V kernel at any head count (by default the row kernel runs every call from 16 Q heads)
    if path == "kqv":
        fm._K4V8_ROWS_MIN_Q_HEADS = 0
    try:
        kc_b = torch.zeros_like(kc_a)
        vc_b = torch.full((num_blocks, k_heads, BS, D // 2 + 4), 0xCD, dtype=torch.uint8, device="cuda")
        qb, kb = fused_qk_norm_mrope_cache_k4v4pt(
            q, k, qw, kw, 1e-6, pos, cos, sin, sec, kc_b, slots, kv_scale=1.0, v=v,
            v_cache=vc_b.view(torch.float4_e2m1fn_x2),
        )
        torch.cuda.synchronize()
    finally:
        fm._K4V4PT_PAGE_MIN_TOKENS, fm._K4V4PT_V_TDM, fm._K4V8_ROWS_MIN_Q_HEADS = old
    assert torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)), "Q differs from A8K4V8"
    assert torch.equal(ka, kb), "k_out differs"
    assert torch.equal(kc_a, kc_b), "K cache differs"
    # V: reference over the written slots
    st = slots[:t]
    valid = st >= 0
    written = st[valid]
    vv = v.view(t, k_heads, D)[valid]
    dense = torch.zeros(num_blocks * BS, k_heads, D, dtype=torch.bfloat16, device="cuda")
    dense[written] = vv
    ref = fp4pt_quant_v_cache(dense.view(num_blocks, BS, k_heads, D), ref=False)[0].view(torch.uint8)
    ref = ref.reshape(num_blocks, k_heads, -1)
    got = vc_b.reshape(num_blocks, k_heads, -1)
    is_w = torch.zeros(num_blocks * BS, dtype=torch.bool, device="cuda")
    is_w[written] = True
    is_w = is_w.view(num_blocks, BS)
    tt = torch.arange(BS, device="cuda")
    c = torch.arange(D // 2, device="cuda")
    data_off = (tt[:, None] // 16) * (D // 2) * 16 + (tt[:, None] % 16) * 16 + (c[None, :] // 16) * 256 + c[None, :] % 16
    s_off = BS * D // 2 + 4 * tt[:, None] + torch.arange(4, device="cuda")[None, :]
    # explicit per-byte "written" mask
    wm = torch.zeros(num_blocks, BS * (D // 2 + 4), dtype=torch.bool, device="cuda")
    for off in (data_off, s_off):
        flat = off.reshape(1, -1).expand(num_blocks, -1)
        rowm = is_w[:, :, None].expand(num_blocks, BS, off.shape[1]).reshape(num_blocks, -1)
        wm.scatter_(1, flat, rowm)
    wm = wm[:, None, :].expand(-1, k_heads, -1)
    nbad = int(((got != ref) & wm).sum())
    assert nbad == 0, f"{nbad} written V bytes differ from fp4pt_quant_v_cache"
    assert bool((got[~wm] == 0xCD).all()), "a slot that was not written changed"
    return int(written.numel())


TOKENS = [1, 4, 15, 16, 17, 64, 256, 1000, 4096, 8192]


@pytest.mark.parametrize("t", TOKENS)
@pytest.mark.parametrize("path,tdm", [("row", True), ("kqv", True), ("kqv", False)])
@pytest.mark.parametrize("layout", ["aligned", "packed"])
@pytest.mark.parametrize("heads", [(16, 2), (4, 1)])
def test_fused_v_bitexact(t, path, tdm, layout, heads):
    run_case(t, heads[0], heads[1], layout, v_contig=False, path=path, tdm=tdm)


@pytest.mark.parametrize("t", [17, 1000, 5000])
@pytest.mark.parametrize("path", ["row", "kqv"])
def test_fused_v_contig_v_and_long_slot_table(t, path):
    run_case(t, 16, 2, "packed", v_contig=True, path=path, tdm=True, extra_slots=8)


@pytest.mark.parametrize("t", TOKENS)
def test_fused_v_default_selection(t):
    # the shipped selection (_k4v4pt_min_tokens: the row kernel at every size from 16 Q heads) as is
    min_tokens = fm._k4v4pt_min_tokens(16)
    path = "kqv" if 0 < min_tokens <= t else "row"
    run_case(t, 16, 2, "packed", v_contig=False, path=path, tdm=fm._K4V4PT_V_TDM)


@pytest.mark.parametrize("t", [64, 4096])
def test_fused_v_cuda_graph(t):
    """Captured in a CUDA graph (as ATOM's decode / prefill token graphs), replayed on new inputs: the same bytes as an
    eager call on those inputs."""
    qh, kh = 16, 2
    nb = t // BS + 16
    g = torch.Generator(device="cuda").manual_seed(t + 5)
    qkv = torch.randn(t, (qh + 2 * kh) * D, generator=g, device="cuda").to(torch.bfloat16)
    q, k, v = qkv[:, : qh * D], qkv[:, qh * D : (qh + kh) * D], qkv[:, (qh + kh) * D :]
    qw = (torch.randn(D, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    kw = (torch.randn(D, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    pos = torch.randint(0, 8192, (3, t), generator=g, device="cuda")
    inv = 1.0 / (10000 ** (torch.arange(0, 32, device="cuda").float() / 32))
    ang = torch.arange(8192, device="cuda").float()[:, None] * inv[None, :]
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    slots = make_slots(t, nb, "packed", g).cuda()
    kc = torch.zeros(nb, kh, BS, D // 2 + D // 16, dtype=torch.uint8, device="cuda")
    vc = torch.zeros(nb, kh, BS, D // 2 + 4, dtype=torch.uint8, device="cuda")
    call = lambda kc_, vc_: fused_qk_norm_mrope_cache_k4v4pt(
        q, k, qw, kw, 1e-6, pos, cos, sin, [11, 11, 10], kc_, slots, kv_scale=1.0, v=v,
        v_cache=vc_.view(torch.float4_e2m1fn_x2))
    call(kc, vc)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(kc, vc)
    qkv.copy_(torch.randn(qkv.shape, generator=g, device="cuda").to(torch.bfloat16))
    kc.zero_(); vc.zero_()
    graph.replay()
    torch.cuda.synchronize()
    kc2, vc2 = torch.zeros_like(kc), torch.zeros_like(vc)
    ref = call(kc2, vc2)
    torch.cuda.synchronize()
    assert torch.equal(out[0].view(torch.uint8), ref[0].view(torch.uint8)) and torch.equal(out[1], ref[1])
    assert torch.equal(kc, kc2) and torch.equal(vc, vc2)
