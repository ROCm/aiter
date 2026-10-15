# SPDX-License-Identifier: MIT
"""MiniMax M3 post-attention norm, router, BM16 metadata and MoE zeroing.

Source-only gfx950 Gluon port of the corresponding ATOM FlyDSL K4 stages.
The input is the already reduced attention output, so sparse attention, o_proj
and its TP collective remain upstream. One 256-CTA launch prepares everything
that the selected two-stage AITER expert path consumes. No GPU validation yet.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from .sync import poll_local


@g.jit
def _route_and_sort(
    Scores,
    Sigmoids,
    RouteIDs,
    RouteWeights,
    SortIDs,
    SortWeights,
    SortExperts,
    SortCounts,
    SortM,
    SortReverse,
    Ready,
    Errors,
    Epoch,
    B: gl.constexpr,
    SHARED_WEIGHT: gl.constexpr,
    ROUTE_SCALE: gl.constexpr,
):
    """CTA 128: acquire score tiles, select top-4 and build AITER BM16 rows."""
    flat: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    tile = gl.arange(0, 128, layout=flat)
    # Readiness words are initialized once; each invocation has a fresh epoch.
    flags = gl.arange(0, 128, layout=flat)
    active = flags < B * 8
    failed = poll_local(Ready + flags, Epoch, active)
    if gl.sum((active & (failed != 0)).to(gl.int32), 0) != 0:
        gl.store(Errors, 1)
        gl.store(SortCounts, 0)
        gl.store(SortCounts + 1, B)
    else:
        for token in range(B):
            key = gl.load(Scores + token * 128 + tile, cache_modifier=".cv")
            sig = gl.load(Sigmoids + token * 128 + tile, cache_modifier=".cv")
            # ATOM's route_key: signed ordered score, low seven bits replaced
            # by 127 - expert ID, so an exact key tie picks the lower ID.
            for slot in gl.static_range(4):
                best = gl.max(key, 0)
                eid = 127 - (best & 127)
                gl.store(RouteIDs + token * 5 + slot, eid)
                picked = gl.sum(gl.where(tile == eid, sig, 0.0), 0)
                gl.store(RouteWeights + token * 5 + slot, picked)
                key = gl.where(tile == eid, -2147483648, key)
            gl.store(RouteIDs + token * 5 + 4, 128)
            gl.store(RouteWeights + token * 5 + 4, SHARED_WEIGHT)
            slots = gl.arange(0, 4, layout=flat)
            raw = gl.load(RouteWeights + token * 5 + slots)
            total = gl.sum(raw, 0)
            gl.store(
                RouteWeights + token * 5 + slots,
                raw * (ROUTE_SCALE / gl.maximum(total, 1.0e-20)),
            )

        # Five routes/token include the always-on shared expert. Each expert
        # occurs at most once per token, so B<=16 gives one BM16 block/expert.
        pairs: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [8, 1], [1, 0])
        ir = gl.arange(0, 128, layout=gl.SliceLayout(1, pairs))
        jr = gl.arange(0, 128, layout=gl.SliceLayout(0, pairs))
        valid_i = ir < B * 5
        valid_j = jr < B * 5
        ei = gl.load(RouteIDs + ir // 5 * 5 + ir % 5, mask=valid_i, other=129)
        ej = gl.load(RouteIDs + jr // 5 * 5 + jr % 5, mask=valid_j, other=129)
        same = (ei[:, None] == ej[None, :]) & valid_i[:, None] & valid_j[None, :]
        row = gl.sum((same & (jr[None, :] < ir[:, None])).to(gl.int32), 1)
        count = gl.sum(same.to(gl.int32), 1)
        leader = valid_i & (row == 0)
        leader_j = gl.convert_layout(leader, gl.SliceLayout(0, pairs))
        block = gl.sum(
            ((ej[None, :] < ei[:, None]) & leader_j[None, :]).to(gl.int32), 1
        )
        n_blocks = gl.sum(leader.to(gl.int32), 0)
        pad = gl.arange(0, 16, layout=gl.SliceLayout(0, pairs))
        padding = leader[:, None] & (pad[None, :] >= count[:, None])
        padding_offset = block[:, None] * 16 + pad[None, :]
        gl.store(SortIDs + padding_offset, B, mask=padding)
        gl.store(SortWeights + padding_offset, 0.0, mask=padding)
        gl.store(SortM + padding_offset, B, mask=padding)
        gl.store(SortExperts + block, ei, mask=leader)
        row = gl.convert_layout(row, flat)
        block = gl.convert_layout(block, flat)
        rid = gl.arange(0, 128, layout=flat)
        dst = block * 16 + row
        wt = gl.load(RouteWeights + rid, mask=rid < B * 5, other=0.0)
        gl.store(SortIDs + dst, (rid // 5) | ((rid % 5) << 24), mask=rid < B * 5)
        gl.store(SortWeights + dst, wt, mask=rid < B * 5)
        gl.store(SortM + dst, rid // 5, mask=rid < B * 5)
        gl.store(SortReverse + rid, dst, mask=rid < B * 5)
        gl.store(SortCounts, n_blocks * 16)
        gl.store(SortCounts + 1, B)


@g.jit
def front(
    AttnReduced,
    Residual,
    Gain,
    Gate,
    Bias,
    HMid,
    XNorm,
    Scores,
    Sigmoids,
    RouteIDs,
    RouteWeights,
    SortIDs,
    SortWeights,
    SortExperts,
    SortCounts,
    SortM,
    SortReverse,
    SortOut,
    Ready,
    Epochs,
    Errors,
    B: gl.constexpr,
    EPS: gl.constexpr = 1.0e-6,
    ROUTE_SCALE: gl.constexpr = 2.0,
    SHARED_WEIGHT: gl.constexpr = 1.0,
):
    """256 CTA resident front; 8 score tiles/token, route CTA 128, zero CTAs."""
    gl.static_assert(B >= 1 and B <= 16)
    pid = gl.program_id(0)
    epoch = gl.load(Epochs + pid) + 1
    gl.store(Epochs + pid, epoch)
    flat: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    if pid < B * 8:
        token = pid // 8
        tile = pid % 8
        k = gl.arange(0, 8192, layout=flat)
        valid = k < 6144
        ar = gl.load(AttnReduced + token * 6144 + k, mask=valid, other=0).to(gl.float32)
        res = gl.load(Residual + token * 6144 + k, mask=valid, other=0).to(gl.float32)
        acc = ar + res
        ss = gl.sum(gl.where(valid, acc * acc, 0.0), 0)
        gain = gl.load(Gain + k, mask=valid, other=0).to(gl.float32)
        x = (acc * gl.rsqrt(ss / 6144.0 + EPS) * (gain + 1.0)).to(gl.bfloat16)
        if tile == 0:
            gl.store(HMid + token * 6144 + k, acc, mask=valid)
            gl.store(XNorm + token * 6144 + k, x, mask=valid)
        for slot in range(16):
            expert = tile * 16 + slot
            w = gl.load(Gate + expert * 6144 + k, mask=valid, other=0).to(gl.float32)
            logit = gl.sum(gl.where(valid, x.to(gl.float32) * w, 0.0), 0)
            # ATOM rounds the gate GEMV to BF16 before sigmoid.
            logit = logit.to(gl.bfloat16).to(gl.float32)
            sig = 1.0 / (1.0 + gl.exp2(-logit * 1.4426950408889634))
            biased = sig + gl.load(Bias + expert)
            bits = biased.to(gl.int32, bitcast=True)
            ordered = gl.where(bits < 0, bits ^ 2147483647, bits)
            key = (ordered & -128) | (127 - expert)
            gl.store(Scores + token * 128 + expert, key)
            gl.store(Sigmoids + token * 128 + expert, sig)
        gl.atomic_store(
            Ready + pid, epoch.to(gl.uint64) << 32 | 1, sem="release", scope="gpu"
        )
    elif pid == 128:
        _route_and_sort(
            Scores,
            Sigmoids,
            RouteIDs,
            RouteWeights,
            SortIDs,
            SortWeights,
            SortExperts,
            SortCounts,
            SortM,
            SortReverse,
            Ready,
            Errors,
            epoch,
            B,
            SHARED_WEIGHT,
            ROUTE_SCALE,
        )
    elif pid > 128:
        z = (pid - 129) * 1024 + gl.arange(0, 1024, layout=flat)
        gl.store(SortOut + z, 0, mask=z < B * 6144)
