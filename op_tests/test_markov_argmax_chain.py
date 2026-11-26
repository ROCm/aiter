# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""markov_argmax_chain against a float64 torch reference.

Every step is checked on the kernel's own previous id, so one near-tie cannot
cascade: the W1 row must be the gathered row bit for bit, and the chosen id
must reach the exact maximum of the biased logits within the fp32
accumulation bound, and be torch.argmax wherever the top two are further
apart than that. Also covered: K changing between launches on one workspace
(including a full wrap of the 14-bit launch tag), int32 and int64 ids, strided
ids / logits / output rows, a smaller grid, NaN and -inf steps, the
supported() boundaries, the launcher's argument checks, and a launch whose
workgroups cannot all be resident (a CU-masked stream) setting the error word.
"""

import argparse
import ctypes
import sys

import torch

from aiter.jit.utils.hip_runtime import load_hip_runtime
from aiter.ops.markov_argmax_chain import (
    MARKOV_ARGMAX_CHAIN_MAX_STEPS,
    markov_argmax_chain,
    markov_argmax_chain_supported,
    markov_argmax_chain_workspace,
)

DEV = "cuda"
TAG_PERIOD = 1 << 14  # launch tags the slots tell apart


def make_inputs(V, R, K, seed, id_dtype=torch.int32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scale = R**-0.25  # the bias about as wide as the base logits
    w1 = (torch.randn(V, R, generator=g, device=DEV) * scale).bfloat16()
    w2 = (torch.randn(V, R, generator=g, device=DEV) * scale).bfloat16()
    base = torch.randn(K, V, generator=g, device=DEV).bfloat16()
    ids = torch.zeros(K + 1, dtype=id_dtype, device=DEV)
    ids[0] = int(torch.randint(0, V, (1,), generator=g, device=DEV))
    return base, ids, w1, w2


def check_chain(base, ids, w1, w2, rows, what):
    """Return the number of near-tie steps; raise on a wrong one."""
    K, V = base.shape
    R = w1.shape[1]
    w2d = w2.double()
    ids_h = ids.long().tolist()
    near = 0
    for k in range(K):
        x, y = ids_h[k], ids_h[k + 1]
        assert 0 <= y < V, f"{what} step {k}: id {y} out of range"
        assert torch.equal(rows[k], w1[x]), f"{what} step {k}: rows_out != W1[x_k]"
        row = w1[x].double()
        b = base[k].double()
        exact = b + w2d @ row
        exact = torch.where(torch.isnan(exact), -torch.inf, exact)  # NaN never wins
        if bool(torch.isneginf(exact).all()) and bool(torch.isnan(b).all()):
            assert y == 0, f"{what} step {k}: an all-NaN step answered {y}, not 0"
            continue
        # fp32 sum of R exact bf16 products, then + base: |fp32 - exact| <= tol
        mag = (b.abs().nan_to_num(0, 0, 0) + w2d.abs() @ row.abs()).max().item()
        tol = (R + 2) * 2.0**-24 * mag
        top2 = torch.topk(exact, 2).values.tolist()
        best = top2[0]
        assert (
            exact[y].item() >= best - 2 * tol
        ), f"{what} step {k}: id {y} logit {exact[y].item()} vs max {best} (tol {tol})"
        if top2[0] - top2[1] > 2 * tol:
            want = int(torch.argmax(exact))
            assert y == want, f"{what} step {k}: id {y}, torch.argmax {want}"
        else:
            near += 1
    return near


def run(base, ids, w1, w2, ws, rows=None):
    rows = markov_argmax_chain(base, ids, w1, w2, ws, rows)
    torch.cuda.synchronize()
    return rows


def test_shapes(cus):
    # (V, R, K, id dtype): a 200K vocabulary at rank 256, 129280 / 152064 ids,
    # a 32000-id vocabulary at rank 128, one 16-id tile, and the most ids 256
    # workgroups hold at either rank.
    shapes = [
        (200064, 256, 7, torch.int32),
        (129280, 256, 4, torch.int64),
        (152064, 128, 3, torch.int32),
        (32000, 128, 8, torch.int64),
        (16, 256, 2, torch.int32),
        (16 * (54 * cus - 1), 256, 8, torch.int32),
        (min(16 * (64 * cus - 1), (1 << 18) - 16), 128, 5, torch.int64),
    ]
    near = 0
    for i, (V, R, K, id_dtype) in enumerate(shapes):
        assert markov_argmax_chain_supported(V, R, K, id_dtype=id_dtype), (V, R, K)
        base, ids, w1, w2 = make_inputs(V, R, K, 100 + i, id_dtype)
        ws = markov_argmax_chain_workspace(K)
        rows = run(base, ids, w1, w2, ws)
        near += check_chain(base, ids, w1, w2, rows, f"V={V} R={R} K={K}")
        assert ws.err.item() == 0
    print(f"shapes: {len(shapes)} passed ({near} near-tie steps)")


def test_strided_and_small_grid():
    V, R, K = 16 * 347, 256, 6  # 347 tiles: a ragged split over 7 and 64 workgroups
    base, ids, w1, w2 = make_inputs(V, R, K, 7, torch.int64)
    wide = torch.full((K, V + 48), float("nan"), dtype=torch.bfloat16, device=DEV)
    wide[:, :V] = base
    ids2 = torch.zeros(K + 1, 2, dtype=torch.int64, device=DEV)
    ids2[0, 0] = ids[0]
    rows_buf = torch.zeros(K, R + 64, dtype=torch.bfloat16, device=DEV)
    for ncta in (7, 64):
        assert markov_argmax_chain_supported(V, R, K, num_workgroups=ncta)
        ws = markov_argmax_chain_workspace(K, num_workgroups=ncta)
        run(wide[:, :V], ids2[:, 0], w1, w2, ws, rows_buf[:, :R])
        check_chain(base, ids2[:, 0], w1, w2, rows_buf[:, :R], f"grid {ncta}")
        assert torch.isnan(wide[:, V:]).all() and (rows_buf[:, R:] == 0).all()
        assert (ids2[:, 1] == 0).all()
    print("strided operands, grids of 7 and 64 workgroups: passed")


def test_nan_and_inf_steps():
    V, R, K = 129280, 256, 4
    base, ids, w1, w2 = make_inputs(V, R, K, 11)
    base[0, 1000:2000] = float("nan")  # some NaN: never chosen
    base[0, 5] = 1e4  # ... and an obvious winner beside them
    base[1] = float("nan")  # no non-NaN logit: id 0
    base[2] = float("-inf")
    base[2, V - 1] = 0  # the one finite logit, in the last tile
    ws = markov_argmax_chain_workspace(K)
    rows = run(base, ids, w1, w2, ws)
    check_chain(base, ids, w1, w2, rows, "NaN/-inf")
    assert ids[1:4].tolist() == [5, 0, V - 1], ids.tolist()
    print("NaN / -inf steps: passed")


def test_steps_change_between_launches():
    """One workspace, K changing launch to launch, then a full wrap of the
    launch tag with the rows past K left alone for 2**14 - 1 launches: a slot
    of an old launch must never pass for one of the current launch."""
    V, R = 32000, 128
    ws = markov_argmax_chain_workspace(8)
    for i, K in enumerate((7, 3, 8, 1, 5, 8, 2)):
        base, ids, w1, w2 = make_inputs(V, R, K, 200 + i)
        rows = run(base, ids, w1, w2, ws)
        check_chain(base, ids, w1, w2, rows, f"K={K} (launch {i})")
    # The 8-step launch leaves every row of the workspace holding large maxima
    # (a stale one would win every step's reduction). Then K = 1 until the
    # next launch's tag equals that launch's.
    base, ids, w1, w2 = make_inputs(V, R, 8, 300)
    base[:, ::97] += 1000
    run(base, ids, w1, w2, ws)
    b1, i1, _, _ = make_inputs(V, R, 1, 301)
    for _ in range(TAG_PERIOD - 1):
        markov_argmax_chain(b1, i1, w1, w2, ws)
    base, ids, w1, w2 = make_inputs(V, R, 8, 302)
    rows = run(base, ids, w1, w2, ws)
    check_chain(base, ids, w1, w2, rows, "after a tag wrap")
    assert ws.err.item() == 0
    print("K changing between launches, tag wrap: passed")


def test_supported_and_checks(cus):
    ok = markov_argmax_chain_supported
    cap256 = 16 * (54 * cus - 1)
    assert ok(cap256, 256, 8) and not ok(cap256 + 16, 256, 8)
    for args, kw in [
        ((200001, 256, 7), {}),  # not whole 16-id tiles
        ((1 << 18, 128, 7), {}),  # ids past the 18-bit field
        ((200064, 64, 7), {}),  # rank not built
        ((200064, 512, 7), {}),
        ((200064, 256, 0), {}),
        ((200064, 256, MARKOV_ARGMAX_CHAIN_MAX_STEPS + 1), {}),
        ((200064, 256, 7), {"logits_dtype": torch.float16}),
        ((200064, 256, 7), {"table_dtype": torch.float32}),
        ((200064, 256, 7), {"id_dtype": torch.int16}),
        ((200064, 256, 7), {"num_workgroups": cus + 1}),
        ((200064, 256, 7), {"num_workgroups": 0}),
        ((200064, 256, 7), {"num_workgroups": 64}),  # too many tiles a workgroup
    ]:
        assert not ok(*args, **kw), (args, kw)
    # The launcher re-checks and raises instead of launching.
    V, K = 4096, 3
    base, ids, w1, w2 = make_inputs(V, 256, K, 5)
    ws = markov_argmax_chain_workspace(K)
    bad = [
        ("rank", (base, ids, w1[:, :64].contiguous(), w2[:, :64].contiguous(), ws)),
        ("ids", (base, ids[:K], w1, w2, ws)),
        ("steps", (base, ids, w1, w2, markov_argmax_chain_workspace(K - 1))),
        ("dtype", (base.half(), ids, w1, w2, ws)),
    ]
    for name, args in bad:
        try:
            markov_argmax_chain(*args)
        except RuntimeError:
            continue
        raise AssertionError(f"launcher accepted a bad {name}")
    torch.cuda.synchronize()
    print("supported() boundaries and launcher checks: passed")


def cu_masked_stream(keep):
    """A stream on the current device that may use only ``keep`` CUs, or None."""
    hip = load_hip_runtime()
    cus = torch.cuda.get_device_properties(DEV).multi_processor_count
    words = (cus + 31) // 32
    mask = (ctypes.c_uint32 * words)()
    for i in range(keep):
        mask[i // 32] |= 1 << (i % 32)
    handle = ctypes.c_void_p()
    fn = getattr(hip, "hipExtStreamCreateWithCUMask", None)
    if fn is None or fn(ctypes.byref(handle), ctypes.c_uint32(words), mask) != 0:
        return None, None
    return torch.cuda.ExternalStream(handle.value), (hip, handle)


def test_wait_gives_up(cus):
    """Half the CUs for a grid of one workgroup a CU: the waits give up, set
    the error word, and the workspace serves the next launch."""
    V, R, K = 32000, 256, 1
    base, ids, w1, w2 = make_inputs(V, R, K, 400)
    ws = markov_argmax_chain_workspace(K)
    stream, owner = cu_masked_stream(cus // 2)
    if stream is None:
        print("wait-gives-up path: not testable here (no CU-masked stream)")
        return
    try:
        with torch.cuda.stream(stream):
            markov_argmax_chain(base, ids, w1, w2, ws)
        stream.synchronize()
    finally:
        owner[0].hipStreamDestroy(owner[1])
    assert ws.err.item() & 1, "no wait gave up with half the CUs masked off"
    ws.err.zero_()
    rows = run(base, ids, w1, w2, ws)
    check_chain(base, ids, w1, w2, rows, "after a launch that gave up")
    assert ws.err.item() == 0
    print("a wait that gives up sets the error word: passed")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if (
        torch.version.hip is None
        or not torch.cuda.is_available()
        or not markov_argmax_chain_supported(32000, 128, 1)
    ):
        print("markov_argmax_chain: gfx1250 only, skipped")
        return 0
    cus = torch.cuda.get_device_properties(DEV).multi_processor_count
    test_supported_and_checks(cus)
    test_shapes(cus)
    test_strided_and_small_grid()
    test_nan_and_inf_steps()
    test_steps_change_between_launches()
    test_wait_gives_up(cus)
    print("markov_argmax_chain: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
