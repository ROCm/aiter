"""Contract tests for the sampled top-k path: argument layout, NaN order, and a
row whose candidates overflow. `sampled` only offers a faster algorithm, so each
test holds it to what the entry it serves already does."""

import os

import torch

import aiter
from aiter.ops import topk as T
from aiter.ops import topk_select as S

N = 131072
K = 2048


def _use_sampled(on):
    os.environ["AITER_DISABLE_TOPK_SAMPLED"] = "0" if on else "1"
    S._available.cache_clear()
    S._choose.cache_clear()


def _bounds(rows, width):
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    return starts, ends


def _sign_key(v):
    """fp32 bits made monotone: a negative NaN below -inf, a positive one above +inf."""
    b = v.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b | 0x80000000)


def _nan_high_key(v):
    """topk_select's order: every NaN above +inf, tied with the others."""
    return torch.where(
        torch.isnan(v), torch.full_like(_sign_key(v), 0xFFFFFFFF), _sign_key(v)
    )


def _selected_keys_match(x, idx, k, key):
    ref = torch.topk(key(x), k, dim=1).values.sort(dim=1).values
    got = key(x.gather(1, idx.long())).sort(dim=1).values
    return torch.equal(got, ref)


def test_sampled_layout_contract():
    """The sampled entry refuses tensors its kernels would address wrongly, and
    the router leaves such calls on the path that served them before."""
    rows = 2
    if not aiter.topk_sampled_supports(rows, N, K):
        print("[sampled_layout] SKIP: `sampled` does not serve this shape here")
        return
    torch.manual_seed(0)
    x = torch.randn(rows, N, device="cuda")
    rs, re = _bounds(rows, N)

    def refused(**over):
        a = {
            "logits": x,
            "rowStarts": rs,
            "rowEnds": re,
            "indices": torch.empty(rows, K, dtype=torch.int32, device="cuda"),
            "values": None,
        }
        a.update(over)
        try:
            aiter.top_k_per_row_prefill_sampled(
                a["logits"],
                a["rowStarts"],
                a["rowEnds"],
                a["indices"],
                a["values"],
                rows,
                N,
                1,
                K,
            )
        except ValueError:
            return True
        return False

    guard = torch.full((rows, K + 17), -7, dtype=torch.int32, device="cuda")
    assert refused(indices=guard[:, :K]), "a row-strided indices view must be refused"
    assert bool((guard == -7).all()), "a refused call must not write"
    vguard = torch.zeros(rows, K + 17, device="cuda")
    assert refused(values=vguard[:, :K]), "a row-strided values view must be refused"
    assert refused(rowStarts=rs.long()), "int64 rowStarts must be refused"
    assert refused(
        rowEnds=torch.full((rows, 2), N, dtype=torch.int32, device="cuda")[:, 0]
    )
    assert refused(indices=torch.empty(rows, K, dtype=torch.int32)), "a CPU output"

    # Through the router a strided output is what the pre-`sampled` path makes of
    # it: same writes, same untouched slots.
    out = {}
    for on in (True, False):
        _use_sampled(on)
        g = torch.full((rows, K + 17), -7, dtype=torch.int32, device="cuda")
        T.top_k_per_row_prefill(x, rs, re, g[:, :K], None, rows, N, 1, K)
        torch.cuda.synchronize()
        out[on] = g.view(-1).sort().values
    _use_sampled(True)
    assert torch.equal(out[True], out[False]), "the router changed a strided call"
    print("[sampled_layout] PASS")


def test_sampled_nan_order():
    """topk_select ranks a NaN of either sign above +inf and returns the input's
    own bits; top_k_per_row_prefill keeps the sign order its radix paths use."""
    rows, k = 256, 128
    if not aiter.topk_sampled_supports(rows, N, k):
        print("[sampled_nan] SKIP: `sampled` does not serve this shape here")
        return
    torch.manual_seed(1)
    x = torch.rand(rows, N, device="cuda") + 1.0
    bits = x.view(torch.int32)
    payloads = (-4194304, -4194303, 0x7FC00001, 0x7F800001, -8388607)
    for j, p in enumerate(payloads):
        bits[:, 1000 * j + 7] = p
    nan_cols = torch.tensor([1000 * j + 7 for j in range(len(payloads))], device="cuda")

    _use_sampled(True)
    assert S.topk_select_backend(rows, N, k, S._available(N, k, 64, False, True)) == (
        "sampled"
    ), "this shape must exercise `sampled`"
    vals, idx = aiter.topk_select(x, k, return_value=True)
    torch.cuda.synchronize()
    assert _selected_keys_match(x, idx, k, _nan_high_key), "topk_select NaN order"
    assert bool(torch.isin(nan_cols, idx).all()), "every NaN must be selected"
    assert torch.equal(
        vals.view(torch.int32), x.gather(1, idx.long()).view(torch.int32)
    ), "returned values must be the input's bits"

    rs, re = _bounds(rows, N)
    pidx = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    pval = torch.empty(rows, k, device="cuda")
    T.top_k_per_row_prefill(x, rs, re, pidx, pval, rows, N, 1, k)
    torch.cuda.synchronize()
    assert _selected_keys_match(x, pidx, k, _sign_key), "prefill sign order"
    assert torch.equal(
        pval.view(torch.int32), x.gather(1, pidx.long()).view(torch.int32)
    ), "prefill values must be the input's bits"
    print("[sampled_nan] PASS")


def test_sampled_overflow_all_equal():
    """Every score equal: the threshold admits the whole row, every wave
    overflows its reservation, and the row is answered by the exact fallback."""
    rows = 128
    if not aiter.topk_sampled_supports(rows, N, K):
        print("[sampled_overflow] SKIP: `sampled` does not serve this shape here")
        return
    x = torch.ones(rows, N, device="cuda")
    rs, re = _bounds(rows, N)
    idx = torch.empty(rows, K, dtype=torch.int32, device="cuda")
    val = torch.empty(rows, K, device="cuda")
    aiter.top_k_per_row_prefill_sampled(x, rs, re, idx, val, rows, N, 1, K)
    torch.cuda.synchronize()
    i = idx.long()
    assert bool(((i >= 0) & (i < N)).all()), "index out of range"
    s = i.sort(dim=1).values
    assert not bool((s[:, 1:] == s[:, :-1]).any()), "duplicate index"
    assert bool((val == 1.0).all()), "values"
    print("[sampled_overflow] PASS")


if __name__ == "__main__":
    test_sampled_layout_contract()
    test_sampled_nan_order()
    test_sampled_overflow_all_equal()
