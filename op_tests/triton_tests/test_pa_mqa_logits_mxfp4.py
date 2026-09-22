# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (cache_format,
                                            paged_mxfp4_mqa_logits,
                                            preshuffle_cache,
                                            unshuffle_scales, unshuffle_values)

SCALE_GROUP = 32
SEED = 0
TOL = 1e-12
E8M0_BIAS = 127
FP4_MAX = 6.0
_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _grid(device):
    mag = torch.tensor(_MAG, dtype=torch.float32, device=device)
    return torch.cat([mag, -mag])


def quantize(x, block=SCALE_GROUP):
    *prefix, d = x.shape
    xb = x.float().reshape(*prefix, d // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    exp = torch.where(amax > 0,
                      torch.ceil(torch.log2(amax.clamp(min=1e-30) / FP4_MAX)), 0.0)
    byte = (exp + E8M0_BIAS).clamp(0.0, 254.0)
    scaled = xb / torch.pow(2.0, byte - E8M0_BIAS)
    nib = (scaled.unsqueeze(-1) - _grid(x.device)).abs().argmin(dim=-1).to(torch.uint8)
    nib = nib.reshape(*prefix, d)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), byte.squeeze(-1).to(torch.uint8).contiguous()


def dequantize(packed, e8m0, block=SCALE_GROUP):
    *prefix, dh = packed.shape
    d = dh * 2
    nib = torch.empty(*prefix, d, dtype=torch.uint8, device=packed.device)
    nib[..., 0::2] = packed & 0xF
    nib[..., 1::2] = (packed >> 4) & 0xF
    vals = _grid(packed.device)[nib.long()]
    scale = torch.pow(2.0, e8m0.float() - E8M0_BIAS)
    return (vals.reshape(*prefix, d // block, block) * scale.unsqueeze(-1)).reshape(
        *prefix, d)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


def reference(q_deq, kv_deq, weights, ctx_lens, max_model_len, cu_ends=None):
    batch, next_n = q_deq.shape[0], q_deq.shape[1]
    out = torch.full((batch * next_n, max_model_len), float("-inf"),
                     dtype=torch.float32, device=q_deq.device)
    for b in range(batch):
        ctx = int(ctx_lens[b])
        if ctx == 0:
            continue
        k = kv_deq[b, :ctx].float()
        for n in range(next_n):
            row = (torch.relu(q_deq[b, n].float() @ k.T)
                   * weights[b * next_n + n].float()[:, None]).sum(dim=0)
            # Exclusive, and clamped to the context the way the kernel's
            # store_hi clamps it. Without a tensor it is the kernel's own rule.
            end = (ctx - next_n + n + 1 if cu_ends is None
                   else min(int(cu_ends[b * next_n + n]), ctx))
            pos = torch.arange(ctx, device=q_deq.device)
            out[b * next_n + n, :ctx] = torch.where(
                pos < end, row, torch.full_like(row, float("-inf")))
    return out


def cu_ends_for(kind, ctx_lens, next_n, ratio=2, device="cuda"):
    """Per-row exclusive bounds, indexed like weights.

    compressed  one key per `ratio` tokens, so the bound steps every `ratio` rows
    padded      parked rows, including a block's last -- so its furthest row is
                not its last
    """
    ends = []
    for b, ctx in enumerate(ctx_lens):
        for n in range(next_n):
            if kind == "compressed":
                e = (ratio * ctx - next_n + n + 1) // ratio
            elif kind == "padded":
                tail = next_n > 1 and n == next_n - 1
                e = 0 if (b % 2 or tail) else ctx - next_n + n + 1
            else:
                raise ValueError(kind)
            ends.append(max(e, 0))
    return torch.tensor(ends, dtype=torch.int32, device=device)


def run_case(batch, next_n, num_heads, head_size, ctx_lens, page_size,
             page_offset=0, seed=SEED, check_inf=True, preshuffle=1,
             clean_logits=True, dynamic=0, cu_ends=None):
    dev = "cuda"
    torch.manual_seed(seed)
    ctx = [int(c) for c in ctx_lens]
    t_max = max(ctx)
    per_seq = (t_max + page_size - 1) // page_size
    used = batch * per_seq
    num_pages = page_offset + used

    q = torch.randn(batch, next_n, num_heads, head_size, device=dev,
                    dtype=torch.bfloat16)
    kv = torch.randn(batch, per_seq * page_size, head_size, device=dev,
                     dtype=torch.bfloat16)
    weights = torch.randn(batch * next_n, num_heads, device=dev,
                          dtype=torch.float32)
    q4, q4s = quantize(q.reshape(-1, head_size))
    q4 = q4.reshape(batch, next_n, num_heads, head_size // 2)
    q4s = q4s.reshape(batch, next_n, num_heads, head_size // SCALE_GROUP)
    kv4, kv4s = quantize(kv.reshape(-1, head_size))
    kv4 = kv4.reshape(batch, -1, head_size // 2)
    kv4s = kv4s.reshape(batch, -1, head_size // SCALE_GROUP)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(used, generator=g) + page_offset
    block_table = perm.reshape(batch, per_seq).to(dev).to(torch.int32)

    # The pool is allocated once in its packed layout and only the pages the
    # block table names are written, so this stays affordable when page_offset
    # pushes num_pages into the millions
    hb, ns = head_size // 2, head_size // SCALE_GROUP
    cache = torch.zeros(num_pages, page_size, 1, hb + ns, dtype=torch.uint8,
                        device=dev)
    v_used = kv4.reshape(used, page_size, hb)
    s_used = kv4s.reshape(used, page_size, ns)
    fmt = cache_format(num_heads, head_size, page_size)
    if preshuffle:
        sv, ss = preshuffle_cache(v_used, s_used, num_heads, head_size)
    else:
        sv, ss = v_used, s_used
    flat = cache.view(num_pages, -1)
    phys = block_table.reshape(-1).long()
    flat[phys, :page_size * hb] = sv.reshape(used, -1)
    flat[phys, page_size * hb:] = ss.reshape(used, -1)
    del v_used, s_used, sv, ss, flat
    qp, qsp = q4, q4s

    mml = per_seq * page_size
    cl_t = torch.tensor(ctx, dtype=torch.int32, device=dev)
    out = paged_mxfp4_mqa_logits(
        qp, qsp, cache, weights, cl_t, block_table, mml,
        preshuffle=preshuffle, clean_logits=clean_logits, dynamic=dynamic,
        cu_ends=cu_ends)
    torch.cuda.synchronize()

    ref = reference(dequantize(q4, q4s), dequantize(kv4, kv4s), weights, ctx, mml,
                    cu_ends)
    # Only the in-window positions are defined without clean_logits, and they
    # are the ones a top-k reads either way.
    fin = torch.isfinite(ref)
    # Every row parked: the -inf pattern is the whole check, and calc_diff over
    # nothing is a nan.
    diff = float(calc_diff(out[fin], ref[fin])) if bool(fin.any()) else 0.0
    inf_ok = (bool(torch.equal(torch.isinf(out), torch.isinf(ref)))
              if (check_inf and clean_logits) else True)
    gib = num_pages * page_size * (head_size // 2 + head_size // SCALE_GROUP) / 2**30

    del cache, out, ref, kv4, kv4s
    torch.cuda.empty_cache()
    return diff, inf_ok, gib



SHAPES = [
    ("decode b1", 1, 1, [1047]),
    ("decode b8", 8, 1, [2048, 1024, 4096, 512, 3000, 777, 64, 129]),
    ("spec n=2", 4, 2, [1024, 2048, 999, 100]),
    ("spec n=6", 4, 6, [2048, 1500, 601, 64]),
    ("spec n=7", 2, 7, [3000, 64]),
    ("spec n=8", 3, 8, [4096, 513, 64]),
    ("prefill 256", 1, 256, [4096]),
    ("prefill 512", 1, 512, [8192]),
    ("tiny ctx", 3, 1, [1, 33, 64]),
    ("ragged", 5, 3, [97, 4096, 1, 2049, 512]),
]

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("head_size", [128,])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("preshuffle", [1, 0])
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize("dynamic", [0, 1])
def test_shape(shape, num_heads, head_size, page_size, preshuffle,
               clean_logits, dynamic):
    _, batch, next_n, ctx_lens = shape
    diff, inf_ok, _ = run_case(batch, next_n, num_heads, head_size, ctx_lens,
                               page_size, seed=SEED, preshuffle=preshuffle,
                               clean_logits=clean_logits, dynamic=dynamic)
    assert diff <= TOL, f"residual {diff:.3e}"
    assert inf_ok, "the -inf pattern does not match the reference"


CU_ENDS_SHAPES = [
    ("decode b8", 8, 1, [2048, 1024, 4096, 512, 3000, 777, 64, 129]),
    ("spec n=6", 4, 6, [2048, 1500, 601, 64]),
    ("chunk n=16", 4, 16, [4096, 97, 1024, 33]),
    # Contexts shorter than the chunk
    ("short ctx n=16", 4, 16, [8, 12, 4, 16]),
]


@pytest.mark.parametrize("shape", CU_ENDS_SHAPES,
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("kind", ["compressed", "padded"])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("dynamic", [0, 1])
def test_cu_ends(shape, kind, num_heads, dynamic):
    """Bounds the kernel's own rule cannot express: a compressed cache, whose
    bound steps once per two rows, and parked rows."""
    _, batch, next_n, ctx_lens = shape
    ends = cu_ends_for(kind, ctx_lens, next_n)
    diff, inf_ok, _ = run_case(batch, next_n, num_heads, 128, ctx_lens, 64,
                               seed=SEED, dynamic=dynamic, cu_ends=ends)
    assert diff <= TOL, f"residual {diff:.3e}"
    assert inf_ok, "the -inf pattern does not match the reference"




@pytest.mark.parametrize("gib", [5.0, ])
@pytest.mark.parametrize("preshuffle", [1, 0])
def test_addressing(gib, preshuffle):
    """A buffer op addresses through a 32-bit offset, so put the sequence at the
    top of a multi-gigabyte pool."""
    head_size, page_size, ctx = 128, 64, 8192
    page_bytes = page_size * (head_size // 2 + head_size // SCALE_GROUP)
    num_pages = max(1, int(gib * 2**30) // page_bytes)
    per_seq = (ctx + page_size - 1) // page_size
    assert num_pages >= per_seq, (
        f"{gib} GiB is only {num_pages} pages, smaller than the sequence's "
        f"{per_seq}")
    free, _ = torch.cuda.mem_get_info()
    if free < (gib + 2) * 2**30:
        pytest.skip(f"needs about {gib + 2:.1f} GiB free")
    try:
        diff, inf_ok, real = run_case(
            1, 1, 32, head_size, [ctx], page_size, seed=SEED,
            preshuffle=preshuffle,
            # the rest of the pool goes below, so the sequence sits at the top
            # and a truncated offset wraps down into real, wrong pages
            page_offset=num_pages - per_seq)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("out of memory")
    assert diff <= TOL, f"residual {diff:.3e} at {real:.1f} GiB"
    assert inf_ok, "the -inf pattern does not match the reference"
