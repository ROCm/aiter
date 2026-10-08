# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Op test for aiter.router_gemm_topk_softmax_asm: the router GEMM, softmax
# top-k and the shared expert sigmoid in one gfx950 kernel, for 1 to 8 tokens.
#
# One @benchmark function times both paths in one table:
#   "unfused": torch.mm for the BF16 logits, then aiter.topk_softmax with the
#              sigmoid shared expert. This is the path the fused op replaces.
#   "fused"  : router_gemm_topk_softmax_asm.
# Both paths are checked against a torch reference, and the `err` columns are
# that check. CI runs this file with python3 and only sees the exit code, so
# main() raises when any `err` is not zero.
#
# Two kinds of input:
#   "exact": every value is a small multiple of a power of two, so every fp32
#            sum is exact in any order. All paths then see the same BF16
#            logits, and ids, ties and weights must match the reference exactly.
#   "randn": normal random inputs. A path that sums in another order can move
#            a logit by one BF16 step, so the check accepts an expert whose
#            reference logit is within two steps of the k-th largest one.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.moe_op import (
    get_router_topk_workspace,
    router_gemm_topk_softmax_asm,
    topk_softmax,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
DIM = 8192
NUM_EXPERTS = 512
TOPK = 10


def make_inputs(tokens, mode, seed=0, dim=DIM, num_experts=NUM_EXPERTS):
    """Return hidden_states [tokens, dim] and gate_weight [num_experts + 1, dim] in bf16."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = num_experts + 1
    if mode == "exact":
        # x is in {-3..3} / 64 and w is in {-3..3}. Every product is a multiple
        # of 1/64 and every sum stays far below 2^24 / 64, so fp32 holds every
        # partial sum exactly and the summation order does not matter.
        x = torch.randint(-3, 4, (tokens, dim), generator=gen).to(dtypes.bf16) / 64
        w = torch.randint(-3, 4, (rows, dim), generator=gen).to(dtypes.bf16)
    else:
        # Logits with a standard deviation of about 1, like a trained router.
        x = torch.randn(tokens, dim, generator=gen).to(dtypes.bf16)
        w = (torch.randn(rows, dim, generator=gen) * dim**-0.5).to(dtypes.bf16)
    return x, w


def output_buffers(tokens, topk=TOPK, num_experts=NUM_EXPERTS):
    """Output buffers in the fused shared expert layout, filled with garbage."""
    weights = torch.full((tokens, topk + 1), float("nan"), dtype=dtypes.fp32)
    ids = torch.full((tokens, topk + 1), -1, dtype=dtypes.i32)
    # Column topk holds the shared expert id. The ops never write it.
    ids[:, topk] = num_experts
    tei = torch.full((tokens, topk), -1, dtype=dtypes.i32)
    return weights, ids, tei


def reference_logits(x, w):
    """BF16 logits rounded from float64 sums, which are exact for bf16 inputs."""
    return (x.double() @ w.double().t()).to(dtypes.bf16).double()


def bf16_step(t):
    """Distance from |t| to the next larger bf16 value."""
    _, exponent = torch.frexp(t.abs().float())
    return torch.ldexp(torch.ones_like(t, dtype=torch.float64), exponent - 8)


def check(ref_logits, weights, ids, tei, topk, exact, name):
    """Return the worst error of one path, so a wrong id, index or weight
    shows up as a nonzero `err` in the table.

    The reference rule is the one of topk_softmax: experts in descending logit
    order with the lower id first on a tie, softmax weights renormalized over
    the top k, and sigmoid(last logit) for the shared expert.
    """
    tokens = ref_logits.shape[0]
    num_experts = ref_logits.shape[1] - 1
    routed = ref_logits[:, :num_experts]
    order = torch.sort(routed, dim=1, descending=True, stable=True)
    got_ids = ids[:, :topk].long()
    in_range = ((got_ids >= 0) & (got_ids < num_experts)).all(dim=1, keepdim=True)
    got_ids = got_ids.clamp(0, num_experts - 1)

    errs = []
    if exact:
        errs.append(
            checkAllclose(
                got_ids.float(),
                order.indices[:, :topk].float(),
                rtol=0,
                atol=0,
                msg=f"{name} ids",
            )
        )
    else:
        # Each logit of a path can be one bf16 step away from the reference,
        # so a swap at the k-th place can cost up to two steps. A selected
        # expert must lie within two steps of the k-th reference logit, the
        # reference logits must not rise by more than two steps from one
        # column to the next, and no id may repeat.
        got = routed.gather(1, got_ids)
        kth = order.values[:, topk - 1 : topk]
        ok = got >= kth - 2 * bf16_step(kth)
        ok[:, 1:] &= got[:, 1:] <= got[:, :-1] + 2 * bf16_step(got[:, :-1])
        distinct = (torch.sort(got_ids, dim=1).values.diff(dim=1) != 0).all(
            dim=1, keepdim=True
        )
        ok &= distinct & in_range
        errs.append(
            checkAllclose(
                ok.float(),
                torch.ones_like(ok, dtype=torch.float32),
                rtol=0,
                atol=0,
                msg=f"{name} ids within two bf16 steps",
            )
        )

    # Reference weights of the ids the path picked.
    top = routed.gather(1, got_ids)
    numer = torch.exp(top - top.max(dim=1, keepdim=True).values)
    ref_w = (numer / numer.sum(dim=1, keepdim=True)).float()
    ref_shared = torch.sigmoid(ref_logits[:, num_experts:]).float()
    # With the same BF16 logits only fp32 rounding differs. Otherwise one bf16
    # step in a logit near 3 moves its weight by up to 1.6 percent.
    rtol, atol = (1e-5, 1e-5) if exact else (2e-2, 1e-2)
    errs.append(
        checkAllclose(
            weights[:, :topk], ref_w, rtol=rtol, atol=atol, msg=f"{name} routed weights"
        )
    )
    errs.append(
        checkAllclose(
            weights[:, topk:],
            ref_shared,
            rtol=rtol,
            atol=atol,
            msg=f"{name} shared weight",
        )
    )
    ref_tei = torch.arange(topk).view(1, -1) * tokens + torch.arange(tokens).view(-1, 1)
    errs.append(
        checkAllclose(
            tei.float(),
            ref_tei.float(),
            rtol=0,
            atol=0,
            msg=f"{name} token_expert_indices",
        )
    )
    shared_id_kept = (ids[:, topk] == num_experts).float()
    errs.append(
        checkAllclose(
            shared_id_kept,
            torch.ones_like(shared_id_kept),
            rtol=0,
            atol=0,
            msg=f"{name} shared expert id column untouched",
        )
    )
    return max(errs)


def run_unfused(x, w, weights, ids, tei, logits):
    torch.mm(x, w.t(), out=logits)
    topk_softmax(weights, ids[:, :TOPK], tei, logits, True, 1, "sigmoid")


def run_fused(x, w, weights, ids, tei, workspace=None):
    router_gemm_topk_softmax_asm(x, w, weights, ids[:, :TOPK], tei, workspace)


@benchmark()
def test_router_gemm_topk_softmax(tokens, mode):
    x, w = make_inputs(tokens, mode)
    ref_logits = reference_logits(x, w)
    exact = mode == "exact"

    w_u, ids_u, tei_u = output_buffers(tokens)
    logits = torch.empty(tokens, NUM_EXPERTS + 1, dtype=dtypes.bf16)
    w_f, ids_f, tei_f = output_buffers(tokens)
    # The tensors are passed as arguments, so run_perftest rotates through
    # copies of them and the gate weight is read from memory, not from cache,
    # as it is in a real decode step.
    candidates = {
        "unfused": (run_unfused, (x, w, w_u, ids_u, tei_u, logits)),
        "fused": (run_fused, (x, w, w_f, ids_f, tei_f)),
    }

    # The gate weight is most of the traffic.
    nbytes = w.numel() * w.element_size() + x.numel() * x.element_size()
    ret = {"gfx": get_gfx()}
    for name, (fn, args) in candidates.items():
        # run_perftest launches the op many times first, so the check below also
        # shows that the kernel resets its counter after every launch.
        _, us = run_perftest(fn, *args)
        torch.cuda.synchronize()
        weights, ids, tei = args[2:5]
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = check(ref_logits, weights, ids, tei, TOPK, exact, name)
    ret["speedup"] = ret["unfused us"] / ret["fused us"]
    return ret


def _expect_raises(fn, needle, desc):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), f"{desc}: unexpected error message: {e}"
        return
    raise AssertionError(f"{desc}: expected RuntimeError containing {needle!r}")


def _graph_replay_check():
    """Capture eight launches in a CUDA graph and replay it with new inputs.

    The kernel keeps a counter in the workspace, and the last workgroup of
    every launch sets it back to zero. All eight launches share one workspace,
    so this checks that back-to-back launches inside one replay and repeated
    replays all start from a zero counter.
    """
    tokens = 4
    x, w = make_inputs(tokens, "randn", seed=1)
    weights, ids, tei = output_buffers(tokens)
    workspace = get_router_topk_workspace()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # The first call loads the kernel, which must happen before the
        # capture starts.
        run_fused(x, w, weights, ids, tei, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(8):
            run_fused(x, w, weights, ids, tei, workspace)
    errs = []
    for seed in (2, 3, 4):
        x_new, _ = make_inputs(tokens, "randn", seed=seed)
        x.copy_(x_new)
        weights.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        errs.append(
            check(
                reference_logits(x, w),
                weights,
                ids,
                tei,
                TOPK,
                False,
                f"graph replay seed {seed}",
            )
        )
    assert max(errs) == 0, f"graph replay results differ from the reference: {errs}"
    aiter.logger.info("graph replay check passed")


def _concurrent_graph_check(launches=8, wanted=100, max_tries=1000):
    """Capture two graphs on one stream, each with its own workspace, and
    replay them at the same time on two other streams.

    A CUDA graph replays on the stream that is current at replay time, not on
    the stream it was captured on. Two graphs that share one workspace count
    into one counter and read each other's logits, and on MI355X about half of
    their launches then give wrong results. Both replay streams wait for one
    event behind a GPU sleep, so that the two replays start together. A busy
    host can still queue the second replay after the first one has finished,
    and on MI355X some pairs of PyTorch pool streams never ran at the same
    time. So GPU timestamps around every replay show whether the two
    overlapped, only pairs of replays that overlapped count toward `wanted`,
    and after ten tries without overlap the check takes new streams. Every
    launch writes its own outputs, which must match the same launch outside a
    graph bit for bit.
    """
    tokens = 8
    capture_stream = torch.cuda.Stream()
    graphs, cases = [], []
    for seed in (5, 6):
        x, w = make_inputs(tokens, "randn", seed=seed)
        expected = output_buffers(tokens)
        run_fused(x, w, *expected)
        workspace = get_router_topk_workspace()
        outs = [output_buffers(tokens) for _ in range(launches)]
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            # The first call loads the kernel, which must happen before the
            # capture starts.
            run_fused(x, w, *outs[0], workspace)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for out in outs:
                run_fused(x, w, *out, workspace)
        graphs.append(graph)
        # The workspace must stay allocated for as long as its graph replays.
        cases.append((outs, expected, workspace))
    torch.cuda.synchronize()

    bad = overlapped = tries = misses = pairs = 0
    while overlapped < wanted and tries < max_tries:
        if tries == 0 or misses == 10:
            replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            gate_stream = torch.cuda.Stream()
            pairs += 1
            misses = 0
        tries += 1
        for (outs, _, _), stream in zip(cases, replay_streams):
            with torch.cuda.stream(stream):
                # Garbage in the outputs shows a launch that wrote nothing.
                for weights, ids, tei in outs:
                    weights.fill_(float("nan"))
                    ids[:, :TOPK].fill_(-1)
                    tei.fill_(-1)
        start = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(gate_stream):
            torch.cuda._sleep(2_000_000)
            start.record()
        spans = []
        for graph, stream in zip(graphs, replay_streams):
            with torch.cuda.stream(stream):
                stream.wait_event(start)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                spans.append((begin, end))
        torch.cuda.synchronize()
        (a0, a1), (b0, b1) = [
            (start.elapsed_time(begin), start.elapsed_time(end)) for begin, end in spans
        ]
        hit = min(a1, b1) > max(a0, b0)
        overlapped += hit
        misses = 0 if hit else misses + 1
        for outs, (exp_w, exp_ids, exp_tei), _ in cases:
            for weights, ids, tei in outs:
                same = (
                    torch.equal(weights, exp_w)
                    and torch.equal(ids, exp_ids)
                    and torch.equal(tei, exp_tei)
                )
                bad += not same
    total = tries * len(graphs) * launches
    streams_used = f"{pairs} pair{'s' if pairs > 1 else ''} of streams"
    assert overlapped >= wanted, (
        f"the two graph replays overlapped in only {overlapped} of {tries} tries "
        f"on {streams_used}, so this check could not run them at the same time"
    )
    assert bad == 0, (
        f"{bad} of {total} launches in two graphs that replayed at the same time "
        "differ from the same launch outside a graph"
    )
    aiter.logger.info(
        "concurrent graph replay check passed: %d launches, the replays overlapped "
        "in %d of %d tries on %s",
        total,
        overlapped,
        tries,
        streams_used,
    )


def _capture(fn):
    """Call fn while a CUDA graph captures on a new stream."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        fn()


def _run_guard_checks():
    """Shapes the kernels do not cover must raise instead of writing garbage."""
    x, w = make_inputs(9, "randn")
    weights, ids, tei = output_buffers(9)
    _expect_raises(lambda: run_fused(x, w, weights, ids, tei), "no kernel", "9 tokens")

    x, w = make_inputs(2, "randn", dim=4096)
    weights, ids, tei = output_buffers(2)
    _expect_raises(lambda: run_fused(x, w, weights, ids, tei), "no kernel", "dim 4096")

    x, w = make_inputs(2, "randn")
    weights, _, tei = output_buffers(2)
    contiguous_ids = torch.empty(2, TOPK, dtype=dtypes.i32)
    _expect_raises(
        lambda: router_gemm_topk_softmax_asm(x, w, weights, contiguous_ids, tei),
        "row stride",
        "topk_ids without room for the shared expert column",
    )

    weights, ids, tei = output_buffers(2)
    _expect_raises(
        lambda: run_fused(x.half(), w, weights, ids, tei),
        "must be bf16",
        "fp16 hidden_states",
    )

    # A captured call must get its workspace from the caller, and the caller
    # must allocate it before the capture.
    weights, ids, tei = output_buffers(2)
    _expect_raises(
        lambda: _capture(lambda: run_fused(x, w, weights, ids, tei)),
        "pass a workspace",
        "a captured call without a workspace",
    )
    _expect_raises(
        lambda: _capture(get_router_topk_workspace),
        "before CUDA graph capture",
        "get_router_topk_workspace() during a capture",
    )

    x, w = make_inputs(0, "randn")
    weights, ids, tei = output_buffers(0)
    run_fused(x, w, weights, ids, tei)

    aiter.logger.info(
        "guard checks passed: 9 tokens, dim 4096, a contiguous [M, 10] topk_ids, "
        "fp16 input, a captured call without a workspace and a workspace allocated "
        "during a capture raise, and 0 tokens is a no-op"
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "router_gemm_topk_softmax_asm unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="number of tokens, 1 to 8",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="*",
        choices=["exact", "randn"],
        default=["exact", "randn"],
    )
    args = parser.parse_args()

    df = pd.DataFrame(
        [
            test_router_gemm_topk_softmax(tokens, mode)
            for mode in args.mode
            for tokens in args.tokens
        ]
    )
    aiter.logger.info(
        "router_gemm_topk_softmax_asm vs torch.mm + topk_softmax:\n%s",
        df.to_markdown(index=False),
    )

    _graph_replay_check()
    _concurrent_graph_check()
    _run_guard_checks()

    err_cols = [c for c in df.columns if c.endswith(" err")]
    bad = df[(df[err_cols] != 0).any(axis=1)]
    assert bad.empty, f"results differ from the reference:\n{bad.to_markdown()}"


if __name__ == "__main__":
    main()
