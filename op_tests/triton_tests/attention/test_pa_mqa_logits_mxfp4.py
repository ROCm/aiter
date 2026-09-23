# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (cache_format,
                                            paged_mxfp4_mqa_logits,
                                            preshuffle_cache,
                                            unshuffle_scales, unshuffle_values)
from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (
    build_candidate_gather)
from aiter.ops.triton.attention.pa_mqa_logits_mxfp4_gather import build_gather

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


def _make_case(batch, next_n, num_heads, head_size, ctx_lens, page_size,
               page_offset=0, seed=SEED, preshuffle=1):
    """The quantised inputs and the packed cache, seeded so two calls agree."""
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

    return dict(q4=q4, q4s=q4s, kv4=kv4, kv4s=kv4s, cache=cache,
                weights=weights, block_table=block_table, ctx=ctx,
                cl=torch.tensor(ctx, dtype=torch.int32, device=dev),
                mml=per_seq * page_size, num_pages=num_pages, dev=dev)


def run_case(batch, next_n, num_heads, head_size, ctx_lens, page_size,
             page_offset=0, seed=SEED, check_inf=True, preshuffle=1,
             clean_logits=True, dynamic=0, cu_ends=None):
    st = _make_case(batch, next_n, num_heads, head_size, ctx_lens, page_size,
                    page_offset, seed, preshuffle)
    q4, q4s, kv4, kv4s = st["q4"], st["q4s"], st["kv4"], st["kv4s"]
    cache, weights, ctx, mml = st["cache"], st["weights"], st["ctx"], st["mml"]
    num_pages = st["num_pages"]

    out = paged_mxfp4_mqa_logits(
        q4, q4s, cache, weights, st["cl"], st["block_table"], mml,
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


GATHER_SHAPES = [
    ("decode b4", 4, 1, [2048, 1024, 512, 129]),
    ("spec n=4", 2, 4, [2048, 601]),
]


def _gather_run(st, num_heads, head_size, next_n, block, preshuffle,
                positions, cu_ends, dynamic=0):
    meta = build_gather(positions,
                        st["block_table"].repeat_interleave(next_n, 0),
                        st["cache"], num_heads, head_size, block,
                        preshuffle=preshuffle)
    out = paged_mxfp4_mqa_logits(
        st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
        st["block_table"], st["mml"], preshuffle=preshuffle, use_gather=True, candidates=meta,
        cu_ends=cu_ends, dynamic=dynamic)
    torch.cuda.synchronize()
    return out


def _identity_list(st, next_n, block):
    """The candidate list [0, block, 2*block, ...] and each row's slot count.

    The count is the kernel's own causal bound, so compact column j is KV
    position j and the two arms must agree bit for bit.
    """
    dev, mml = st["dev"], st["mml"]
    rows = len(st["ctx"]) * next_n
    r = torch.arange(rows, device=dev) % next_n
    ctx_r = torch.tensor(st["ctx"], device=dev).repeat_interleave(next_n)
    ends = torch.clamp(ctx_r - next_n + r + 1, min=0).to(torch.int32)
    k = (mml + block - 1) // block
    base = (torch.arange(k, device=dev) * block)[None, :]
    last = (((ends.long() - 1) // block) * block).clamp(min=0)[:, None]
    return torch.minimum(base.expand(rows, k), last).contiguous(), ends


@pytest.mark.parametrize("shape", GATHER_SHAPES,
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("preshuffle", [1, 0])
@pytest.mark.parametrize("block", [8, 16, 32])
def test_gather_identity(shape, num_heads, preshuffle, block):
    """The primary gate on the gather.

    The block-to-byte map is not linear -- a page's 8-token blocks start at
    0, 128, 256, 384, 2048, ... -- so a stride multiply gives plausible garbage
    no tolerance catches. On the identity list the gather walks the same bytes
    in the same order as the dense path, so only bit-identity will do.
    """
    _, batch, next_n, ctx_lens = shape
    st = _make_case(batch, next_n, num_heads, 128, ctx_lens, 64,
                    preshuffle=preshuffle)
    ref = paged_mxfp4_mqa_logits(
        st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
        st["block_table"], st["mml"], preshuffle=preshuffle).clone()
    torch.cuda.synchronize()
    pos, ends = _identity_list(st, next_n, block)
    got = _gather_run(st, num_heads, 128, next_n, block, preshuffle, pos, ends)
    nd = int((ref.view(torch.int32) != got.view(torch.int32)).sum())
    assert nd == 0, f"{nd} differing words"


@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("preshuffle", [1, 0])
def test_gather_scattered(num_heads, preshuffle):
    """A genuinely scattered list, against the dequantised cache.

    Also checks cu_ends is read in slot space: the output is compact, so row r
    holds its candidates at [0, cu_ends[r]) and everything past stays -inf.
    """
    batch, next_n, block, nb = 2, 2, 8, 16
    st = _make_case(batch, next_n, num_heads, 128, [1024, 768], 64,
                    preshuffle=preshuffle)
    dev, rows = st["dev"], batch * next_n
    g = torch.Generator(device=dev).manual_seed(7)
    pos = torch.stack([
        torch.randperm(768 // block, device=dev, generator=g)[:nb].sort().values
        for _ in range(rows)]).long() * block
    ends = torch.full((rows,), nb * block, dtype=torch.int32, device=dev)
    got = _gather_run(st, num_heads, 128, next_n, block, preshuffle, pos, ends)

    kv_deq = dequantize(st["kv4"], st["kv4s"])
    q_deq = dequantize(st["q4"], st["q4s"])
    slots = (pos[:, :, None]
             + torch.arange(block, device=dev)[None, None, :]).reshape(rows, -1)
    ref = torch.full_like(got, float("-inf"))
    for r in range(rows):
        b = r // next_n
        k = kv_deq[b].index_select(0, slots[r]).float()
        qk = torch.relu(q_deq[b, r % next_n].float() @ k.T)
        ref[r, :nb * block] = (qk * st["weights"][r].float()[:, None]).sum(0)
    assert float(calc_diff(got[:, :nb * block], ref[:, :nb * block])) <= TOL
    assert bool(torch.isinf(got[:, nb * block:]).all()), "tail is not -inf"


def test_gather_needs_cu_ends():
    """cu_ends is the gather's walk length, so it is not optional there."""
    st = _make_case(1, 1, 32, 128, [512], 64)
    pos, _ = _identity_list(st, 1, 8)
    with pytest.raises(AssertionError, match="cu_ends"):
        _gather_run(st, 32, 128, 1, 8, 1, pos, None)


def test_gather_ignores_dynamic():
    """A gather launch does not build a device schedule: every row walks the
    same tile count, so there is no spread to even out. dynamic=1 set for the
    step must leave the result alone."""
    st = _make_case(4, 1, 32, 128, [2048, 1024, 512, 129], 64)
    pos, ends = _identity_list(st, 1, 8)
    a = _gather_run(st, 32, 128, 1, 8, 1, pos, ends).clone()
    b = _gather_run(st, 32, 128, 1, 8, 1, pos, ends, dynamic=1)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


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


# block maxima


def block_scores_reference(logits, ends, block):
    """Per-row max over each `block` columns, out-of-window read as -inf.

    Max rounds nothing and torch.amax propagates NaN the way the kernel's
    reduce does, so this is a bit-identity reference, not a tolerance one.
    Pinned like the kernel: the block holding each row's newest key is +inf.
    """
    rows, width = logits.shape
    nb = (width + block - 1) // block
    x = logits
    if nb * block > width:
        x = torch.nn.functional.pad(x, (0, nb * block - width),
                                    value=float("-inf"))
    col = torch.arange(nb * block, device=logits.device)
    x = torch.where(col[None, :] < ends[:, None], x,
                    torch.full_like(x, float("-inf")))
    out = x.reshape(rows, nb, block).amax(-1)
    live = ends > 0
    out[torch.arange(rows, device=out.device)[live],
        (ends[live].long() - 1) // block] = float("inf")
    return out


def row_ends(ctx_lens, next_n, cu_ends, device="cuda"):
    """The exclusive per-row column bound the kernel's stores answer to."""
    out = []
    for b, ctx in enumerate(ctx_lens):
        for n in range(next_n):
            e = (ctx - next_n + n + 1 if cu_ends is None
                 else int(cu_ends[b * next_n + n]))
            out.append(max(min(e, ctx), 0))
    return torch.tensor(out, dtype=torch.int32, device=device)


def _bscore_run(st, num_heads, next_n, block, preshuffle=1, clean_logits=True,
                dynamic=0, cu_ends=None, only=False):
    """One launch of the fused reduce. `only` drops the logits store.

    Returns what the launcher returned and the score tensor -- which are the
    same object under `only`, that being the mode's whole output.
    """
    rows, mml = len(st["ctx"]) * next_n, st["mml"]
    nb = (mml + block - 1) // block
    bs = torch.full((rows, nb), float("-inf"), dtype=torch.float32,
                    device=st["dev"])
    out = paged_mxfp4_mqa_logits(
        st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
        st["block_table"], mml, preshuffle=preshuffle,
        clean_logits=clean_logits, dynamic=dynamic, cu_ends=cu_ends,
        block_scores=bs, calc_logits=not only, calc_block_scores=True,
        candidate_block_size=block)
    torch.cuda.synchronize()
    # "both" hands back a pair; the callers below want the logits half.
    return (out if only else out[0]), bs


def _bscore_arms(st, num_heads, next_n, block, **kw):
    """The same launch twice: maxima beside the logits, then instead of them.

    The mode is allowed to change exactly one thing -- that no logits are
    stored. Every block maximum must therefore be the same word, which is what
    the callers below assert. Returns (logits, beside, alone).
    """
    logits, beside = _bscore_run(st, num_heads, next_n, block, **kw)
    ret, alone = _bscore_run(st, num_heads, next_n, block, only=True, **kw)
    assert ret is alone, "calc_logits off must hand back the score tensor"
    return logits, beside, alone


def _same_words(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


BSCORE_SHAPES = [
    ("decode b1", 1, 1, [1047]),
    ("decode b8", 8, 1, [2048, 1024, 4096, 512, 3000, 777, 64, 129]),
    ("spec n=6", 4, 6, [2048, 1500, 601, 64]),
    ("prefill 512", 1, 512, [8192]),
    ("tiny ctx", 3, 1, [1, 33, 64]),
    ("ragged", 5, 3, [97, 4096, 1, 2049, 512]),
]


@pytest.mark.parametrize("shape", BSCORE_SHAPES[:2],
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("dynamic", [0, 1])
@pytest.mark.parametrize("clean_logits", [True, False])
def test_block_scores_knobs(shape, num_heads, dynamic, clean_logits):
    """The split plan and the store relaxation must not move a maximum.

    A block divides BLOCK_KV so none straddles a split; clean_logits off drops
    the per-row select from the logits store, which the reduce still needs.
    """
    _, batch, next_n, ctx_lens = shape
    st = _make_case(batch, next_n, num_heads, 128, ctx_lens, 64)
    logits, bs = _bscore_run(st, num_heads, next_n, 8, clean_logits=clean_logits,
                             dynamic=dynamic)
    ends = row_ends(st["ctx"], next_n, None)
    ref = block_scores_reference(logits, ends, 8)
    nd = int((ref.view(torch.int32) != bs.view(torch.int32)).sum())
    assert nd == 0, f"{nd} differing words"


@pytest.mark.parametrize("shape", BSCORE_SHAPES[:3],
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("num_heads", [32, 64])
def test_block_scores_leaves_logits_alone(shape, num_heads):
    """The maxima come out beside the logits, not instead of them.

    The producer still needs its own logits, so they must be word-identical to
    a launch without the side output.
    """
    _, batch, next_n, ctx_lens = shape
    st = _make_case(batch, next_n, num_heads, 128, ctx_lens, 64)
    plain = paged_mxfp4_mqa_logits(
        st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
        st["block_table"], st["mml"]).clone()
    torch.cuda.synchronize()
    fused, _ = _bscore_run(st, num_heads, next_n, 8)
    nd = int((plain.view(torch.int32) != fused.view(torch.int32)).sum())
    assert nd == 0, f"{nd} differing words in the logits"


def test_block_scores_rejects_gather():
    """The producer is dense and the consumers gather; no launch is both."""
    st = _make_case(1, 1, 32, 128, [512], 64)
    pos, ends = _identity_list(st, 1, 8)
    meta = build_gather(pos, st["block_table"], st["cache"], 32, 128, 8)
    bs = torch.full((1, st["mml"] // 8), float("-inf"), dtype=torch.float32,
                    device=st["dev"])
    with pytest.raises(AssertionError, match="dense producer"):
        paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], st["mml"], use_gather=True, candidates=meta, cu_ends=ends,
            block_scores=bs, calc_block_scores=True)


def test_block_scores_needs_room():
    """A score tensor too narrow for max_model_len is a caller error."""
    st = _make_case(1, 1, 32, 128, [512], 64)
    bs = torch.full((1, st["mml"] // 8 - 1), float("-inf"),
                    dtype=torch.float32, device=st["dev"])
    with pytest.raises(AssertionError, match="blocks wide"):
        paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], st["mml"], block_scores=bs,
            calc_block_scores=True)


# block maxima with no logits at all


@pytest.mark.parametrize("shape", BSCORE_SHAPES[:3],
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("block", [8, 32])
@pytest.mark.parametrize("preshuffle", [1, 0])
def test_scores_only(shape, num_heads, block, preshuffle):
    """The store-free mode must not move a maximum.

    Dropping the store frees registers and the scheduler reallocates, so check
    both arms against each other and against the reference. Covers mode 1 too.
    """
    _, batch, next_n, ctx_lens = shape
    st = _make_case(batch, next_n, num_heads, 128, ctx_lens, 64,
                    preshuffle=preshuffle)
    logits, beside, alone = _bscore_arms(st, num_heads, next_n, block,
                                         preshuffle=preshuffle)
    nd = _same_words(beside, alone)
    assert nd == 0, f"{nd} differing words against the alongside arm"
    ref = block_scores_reference(logits, row_ends(st["ctx"], next_n, None), block)
    nd = _same_words(ref, alone)
    assert nd == 0, f"{nd} differing words against the reference"


@pytest.mark.parametrize("shape", BSCORE_SHAPES[:4],
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("dynamic", [0, 1])
def test_scores_only_knobs(shape, num_heads, dynamic):
    """The split plan must not move a maximum with the store gone either.

    No clean_logits axis: the mode rejects it rather than ignoring it.
    """
    _, batch, next_n, ctx_lens = shape
    st = _make_case(batch, next_n, num_heads, 128, ctx_lens, 64)
    logits, beside, alone = _bscore_arms(st, num_heads, next_n, 8,
                                         dynamic=dynamic)
    assert _same_words(beside, alone) == 0
    ref = block_scores_reference(logits, row_ends(st["ctx"], next_n, None), 8)
    nd = _same_words(ref, alone)
    assert nd == 0, f"{nd} differing words"


@pytest.mark.parametrize("shape", CU_ENDS_SHAPES,
                         ids=lambda s: s[0].replace(" ", "_"))
@pytest.mark.parametrize("kind", ["compressed", "padded"])
def test_scores_only_cu_ends(shape, kind):
    """The row bound still reaches the reduce with no store to share it with.

    The boundary's -inf select belongs to the block max, not to the store.
    """
    _, batch, next_n, ctx_lens = shape
    ends_t = cu_ends_for(kind, ctx_lens, next_n)
    st = _make_case(batch, next_n, 32, 128, ctx_lens, 64)
    logits, beside, alone = _bscore_arms(st, 32, next_n, 8, cu_ends=ends_t)
    assert _same_words(beside, alone) == 0
    ref = block_scores_reference(logits, row_ends(st["ctx"], next_n, ends_t), 8)
    nd = _same_words(ref, alone)
    assert nd == 0, f"{nd} differing words"


@pytest.mark.parametrize("block", [8, 16])
def test_scores_only_nan(block):
    """A poisoned e8m0 byte must still propagate with the store gone."""
    batch, next_n, page_size = 2, 1, 64
    st = _make_case(batch, next_n, 32, 128, [1024, 777], page_size)
    flat = st["cache"].view(st["cache"].shape[0], -1)
    scales = flat[:, page_size * 64:]
    for i in st["block_table"].reshape(-1).tolist()[:4]:
        scales[i, 5] = 255
    logits, beside, alone = _bscore_arms(st, 32, next_n, block)
    assert int(torch.isnan(alone).sum()) > 0, "the NaN did not reach a block"
    assert _same_words(beside, alone) == 0
    ref = block_scores_reference(logits, row_ends(st["ctx"], next_n, None), block)
    nd = _same_words(ref, alone)
    assert nd == 0, f"{nd} differing words"


def test_scores_only_allocates_no_logits():
    """The reason the mode exists: the [rows, ctx] tensor never happens.

    16 MB here, 748 MB at the producer's real shape -- absent, not small.
    """
    batch, next_n, num_heads = 1, 512, 32
    st = _make_case(batch, next_n, num_heads, 128, [8192], 64)
    rows, mml = next_n, st["mml"]
    logits_bytes = rows * mml * 4

    def extra(only):
        bs = torch.full((rows, mml // 8), float("-inf"), dtype=torch.float32,
                        device=st["dev"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], mml, block_scores=bs,
            calc_logits=not only, calc_block_scores=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    with_store = extra(False)
    without = extra(True)
    assert with_store >= logits_bytes, with_store
    assert without < logits_bytes // 2, without


@pytest.mark.parametrize("want_logits,want_scores",
                         [(True, False), (False, True), (True, True)])
def test_outputs_allocates(want_logits, want_scores):
    """Every requested output comes back whether or not the caller owns it."""
    st = _make_case(1, 4, 32, 128, [512], 64)
    rows, mml = 4, st["mml"]
    got = paged_mxfp4_mqa_logits(
        st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
        st["block_table"], mml, calc_logits=want_logits,
        calc_block_scores=want_scores, candidate_block_size=8)
    logits, scores = (got if want_logits and want_scores else
                      (got, None) if want_logits else (None, got))
    if logits is not None:
        assert logits.shape == (rows, mml) and logits.dtype == torch.float32
    if scores is not None:
        assert scores.shape == (rows, mml // 8)
        # every block inside the context was written, none is left unset
        assert not (scores == float("-inf")).all(1).any()


def test_block_scores_rejects_logits_only():
    """A score tensor with nothing asked to write it is a caller error."""
    st = _make_case(1, 1, 32, 128, [512], 64)
    bs = torch.full((1, st["mml"] // 8), float("-inf"), dtype=torch.float32,
                    device=st["dev"])
    with pytest.raises(AssertionError, match="calc_block_scores off"):
        paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], st["mml"], block_scores=bs)


def test_scores_only_rejects_gather():
    """Mutual exclusion with the gather survives the third BSCORE value.

    Pass 1 of a producer whose pass 2 is a gather: two launches, never one.
    """
    st = _make_case(1, 1, 32, 128, [512], 64)
    pos, ends = _identity_list(st, 1, 8)
    meta = build_gather(pos, st["block_table"], st["cache"], 32, 128, 8)
    bs = torch.full((1, st["mml"] // 8), float("-inf"), dtype=torch.float32,
                    device=st["dev"])
    with pytest.raises(AssertionError, match="dense producer"):
        paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], st["mml"], use_gather=True, candidates=meta, cu_ends=ends,
            block_scores=bs, calc_logits=False, calc_block_scores=True)


def _expand_ref(ids, ends, block):
    """The sort and slot count build_candidate_gather fuses, in torch."""
    nb = (ends + block - 1) // block
    ok = (ids >= 0) & (ids < nb[:, None])
    n_valid = ok.sum(1)
    max_id = torch.where(ok, ids, torch.full_like(ids, -1)).max(1).values
    key = torch.where(ok, ids, torch.full_like(ids, 0x7FFFFFFF)).sort(1).values
    last = ((nb - 1) * block).long()
    pos = torch.where(key == 0x7FFFFFFF, last[:, None], key.long() * block)
    tail = torch.minimum(torch.full_like(ends, block), ends - max_id.int() * block)
    cu = torch.where(n_valid > 0, (n_valid.int() - 1) * block + tail,
                     torch.zeros_like(ends))
    return pos, cu.to(torch.int32)


@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("block", [8, 32])
def test_candidate_gather(num_heads, block):
    """The fused builder is build_gather plus the sort, word for word."""
    rows = 4
    st = _make_case(1, rows, num_heads, 128, [4096], 64)
    ctx = st["ctx"][0]
    nb, K = (ctx + block - 1) // block, 128
    g = torch.Generator(device=st["dev"]).manual_seed(5)
    ids = torch.rand(rows, nb, generator=g, device=st["dev"]).argsort(1)
    ids = ids[:, :K].to(torch.int32)
    ids[torch.rand(rows, K, generator=g, device=st["dev"]) < 0.1] = -1
    ends = torch.randint(1, ctx + 1, (rows,), dtype=torch.int32, device=st["dev"])
    bt = st["block_table"].repeat_interleave(rows, 0).contiguous()

    pos_r, cu_r = _expand_ref(ids, ends, block)
    ref = build_gather(pos_r, bt, st["cache"], num_heads, 128, block)
    got, cu = build_candidate_gather(ids, ends, bt, st["cache"], num_heads,
                                     128, block)
    assert torch.equal(got["voff"], ref["voff"]), "value offsets differ"
    assert torch.equal(got["soff"], ref["soff"]), "scale offsets differ"
    assert torch.equal(got["positions"], pos_r) and torch.equal(cu, cu_r)


@pytest.mark.parametrize("num_heads", [32, 64])
def test_candidates_implicit(num_heads):
    """Handing the launch ids matches building the pool outside it."""
    rows, block, K = 4, 8, 128
    st = _make_case(1, rows, num_heads, 128, [4096], 64)
    ctx = st["ctx"][0]
    g = torch.Generator(device=st["dev"]).manual_seed(7)
    ids = torch.rand(rows, ctx // block, generator=g,
                     device=st["dev"]).argsort(1)[:, :K].to(torch.int32)
    ends = row_ends(st["ctx"], rows, None)
    bt = st["block_table"].repeat_interleave(rows, 0).contiguous()
    meta, cu = build_candidate_gather(ids, ends, bt, st["cache"], num_heads,
                                      128, block)
    a = paged_mxfp4_mqa_logits(st["q4"], st["q4s"], st["cache"], st["weights"],
                               st["cl"], st["block_table"], K * block,
                               use_gather=True, candidates=meta, cu_ends=cu)
    b = paged_mxfp4_mqa_logits(st["q4"], st["q4s"], st["cache"], st["weights"],
                               st["cl"], st["block_table"], K * block,
                               use_gather=True, candidates=ids, cu_ends=ends)
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("pad", [256, 9472])
def test_strided_pages(num_heads, page_size, pad):
    """Pages a whole block apart, as vLLM's block-major pool lays them out."""
    st = _make_case(2, 1, num_heads, 128, [4096, 2048], page_size)
    cache = st["cache"]
    pages, page_bytes = cache.shape[0], cache[0].numel()
    stride = page_bytes + pad
    pool = torch.zeros(pages * stride, dtype=torch.uint8, device=st["dev"])
    torch.as_strided(pool, (pages, page_bytes), (stride, 1), 0).copy_(
        cache.view(pages, -1))
    strided = torch.as_strided(pool, cache.shape,
                               (stride,) + cache.stride()[1:], 0)

    args = (st["q4"], st["q4s"])
    rest = (st["weights"], st["cl"], st["block_table"], st["mml"])
    ref = paged_mxfp4_mqa_logits(*args, cache, *rest)
    got = paged_mxfp4_mqa_logits(*args, strided, *rest)
    torch.cuda.synchronize()
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))


@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("block", [8, 32])
def test_gather_offsets_i64(num_heads, block):
    """A 64-bit candidate list, for a pool past the i32 reach, walks the same."""
    rows, K = 4, 128
    st = _make_case(1, rows, num_heads, 128, [4096], 64)
    g = torch.Generator(device=st["dev"]).manual_seed(11)
    ids = torch.rand(rows, st["ctx"][0] // block, generator=g,
                     device=st["dev"]).argsort(1)[:, :K].to(torch.int32)
    ends = row_ends(st["ctx"], rows, None)
    bt = st["block_table"].repeat_interleave(rows, 0).contiguous()

    out = []
    for width in (torch.int32, torch.int64):
        meta, cu = build_candidate_gather(ids, ends, bt, st["cache"], num_heads,
                                          128, block, offsets=width)
        assert meta["voff"].dtype == width and meta["soff"].dtype == width
        out.append(paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], K * block, use_gather=True, candidates=meta,
            cu_ends=cu, candidate_block_size=block))
    torch.cuda.synchronize()
    assert torch.equal(out[0].view(torch.int32), out[1].view(torch.int32))


def test_gather_span_window():
    """A pool between 2 and 4 GiB: the i32 list still reaches, but the buffer
    path would not -- it multiplies the list back to bytes in i32."""
    rows, block, K, page_size = 4, 8, 64, 64
    page_bytes = page_size * (64 + 4)
    st = _make_case(1, rows, 32, 128, [2048], page_size,
                    page_offset=2 ** 31 // page_bytes + 512)
    g = torch.Generator(device=st["dev"]).manual_seed(13)
    ids = torch.rand(rows, st["ctx"][0] // block, generator=g,
                     device=st["dev"]).argsort(1)[:, :K].to(torch.int32)
    ends = row_ends(st["ctx"], rows, None)
    bt = st["block_table"].repeat_interleave(rows, 0).contiguous()

    out = []
    for width in (None, torch.int64):
        meta, cu = build_candidate_gather(ids, ends, bt, st["cache"], 32, 128,
                                          block, offsets=width)
        out.append(paged_mxfp4_mqa_logits(
            st["q4"], st["q4s"], st["cache"], st["weights"], st["cl"],
            st["block_table"], K * block, use_gather=True, candidates=meta,
            cu_ends=cu, candidate_block_size=block))
    torch.cuda.synchronize()
    assert torch.equal(out[0].view(torch.int32), out[1].view(torch.int32))


def test_gather_rejects_short_offsets():
    """Pinning i32 on a cache it cannot reach is an assert, not a truncation."""
    rows, block, K = 2, 8, 32
    # meta: the guard reads the cache's shape and strides, never its bytes
    cache = torch.empty(2_000_000, 64, 1, 68, dtype=torch.uint8, device="meta")
    ids = torch.zeros(rows, K, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), 512, dtype=torch.int32, device="cuda")
    bt = torch.zeros(rows, 16, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="do not reach"):
        build_candidate_gather(ids, ends, bt, cache, 32, 128, block,
                               offsets=torch.int32)
