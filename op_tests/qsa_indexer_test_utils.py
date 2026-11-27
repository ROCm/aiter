# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Fixtures and independent references from tingqli/pyhip (MIT), revision 8027faad2.

"""Direct tensor fixtures and independent PyTorch references for the QSA indexer.

``source.inputs`` is the raw operator keyword mapping; ``source.state`` is an
independent initial-state snapshot. Projection, fixture planning and references
are outside the operator boundary. No model, backend, pool object or hook is used.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx_runtime


def run_prefill(**inputs):
    from aiter.ops.qsa import qsa_prefill_indexer

    return qsa_prefill_indexer(**inputs)


def run_decode(*, verify=False, **inputs):
    from aiter.ops.qsa import qsa_decode_indexer

    return qsa_decode_indexer(**inputs, verify=verify)


HEADS, DIM, ROTARY, PAGE = 4, 128, 64, 64
RATIO, TOPK, WIDTH = 4, 512, 2051
# SGLang's pending ring: slots req * RING + position % RING (a verify window plus the members before it).
RING = 2 * RATIO
STATE_NAMES = ("key_state", "rope_state", "compressed")


def _gpu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm gfx942")
    torch.cuda.set_device(int(os.environ.get("QSA_REPLAY_GPU", "0")))
    if get_gfx_runtime() != "gfx942":
        pytest.skip("requires gfx942")
    return torch.device("cuda", torch.cuda.current_device())


def clone_state(state):
    return {name: value.clone() for name, value in state.items()}


def reset_state(source):
    for name, value in source.state.items():
        source.inputs[name].copy_(value)


def _state(device, ring_rows, slots, generator):
    return {
        "key_state": torch.randn(
            (ring_rows, 1, DIM), generator=generator, dtype=torch.float32, device=device
        ).to(torch.bfloat16),
        "rope_state": torch.zeros((ring_rows, 3), dtype=torch.int64, device=device),
        "compressed": torch.randn(
            (slots, 1, DIM), generator=generator, dtype=torch.float32, device=device
        ).to(torch.bfloat16),
    }


def _axis_map(section, interleaved, device):
    half = ROTARY // 2
    axes = torch.zeros(half, dtype=torch.int32, device=device)
    if not section:
        return axes
    if len(section) != 3 or min(section) < 0 or sum(section) != half:
        raise ValueError("MRoPE sections must partition the 32 rotary pairs")
    s0, s1, s2 = section
    if interleaved:
        pairs = torch.arange(half, device=device)
        axes[(pairs % 3 == 1) & (pairs < 3 * s1)] = 1
        axes[(pairs % 3 == 2) & (pairs < 3 * s2)] = 2
    else:
        axes[s0 : s0 + s1] = 1
        axes[s0 + s1 :] = 2
    return axes


def _rope_matrix(positions):
    if positions.ndim == 1:
        return positions[:, None].expand(-1, 3).contiguous()
    return positions.T.contiguous()


def _parameters(
    generator,
    device,
    context,
    *,
    section=(11, 11, 10),
    interleaved=True,
    cache_dtype=torch.bfloat16,
):
    return {
        "q_weight": (
            torch.randn(DIM, generator=generator, dtype=torch.float32, device=device)
            / 4
        ).to(torch.bfloat16),
        "k_weight": (
            torch.randn(DIM, generator=generator, dtype=torch.float32, device=device)
            / 4
        ).to(torch.bfloat16),
        "cos_sin_cache": torch.randn(
            (context + PAGE, ROTARY),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ).to(cache_dtype),
        "axis_map": _axis_map(section, interleaved, device),
        "q_eps": 1e-6,
        "k_eps": 1e-6,
    }


def _host_lengths(seq_lens, extend_lens):
    seq_lens, extend_lens = tuple(seq_lens), tuple(extend_lens)
    if (
        not seq_lens
        or len(seq_lens) != len(extend_lens)
        or not sum(extend_lens)
        or any(e < 0 or s < e or (s - e) % RATIO for s, e in zip(seq_lens, extend_lens))
    ):
        raise ValueError(
            "Require nonempty rows and group-aligned, nonnegative prefixes"
        )
    return seq_lens, extend_lens, tuple(s - e for s, e in zip(seq_lens, extend_lens))


@torch.no_grad()
def synthetic(
    seq_lens,
    extend_lens,
    device,
    seed=11,
    *,
    section=(11, 11, 10),
    interleaved=True,
    position_axes=3,
    cache_dtype=torch.bfloat16,
):
    """Host-planned, page-64 requests; qk is BF16 [rows, (4 + 1) * 128], not attention Q."""
    seq_lens, extend_lens, prefixes = _host_lengths(seq_lens, extend_lens)
    if position_axes not in (1, 3):
        raise ValueError("positions must have one or three axes")
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, batch = sum(extend_lens), len(seq_lens)
    width = math.ceil(max(seq_lens) / PAGE) * PAGE
    starts = [PAGE * (1 + request) + request * width for request in range(batch)]
    table = torch.stack(
        [
            torch.arange(start, start + width, device=device, dtype=torch.int32)
            for start in starts
        ]
    )
    logical_host, slots_host, groups = [], [], []
    row_start = 0
    for request, (length, extend, prefix) in enumerate(
        zip(seq_lens, extend_lens, prefixes)
    ):
        for position in range(prefix, length):
            logical_host.append(position)
            pending = position >= length // RATIO * RATIO
            slots_host.append(
                (request + 1) * RING + position % RING if pending else position % RING
            )
        for block in range(prefix // RATIO, length // RATIO):
            groups.append(
                (
                    (starts[request] + block * RATIO) // RATIO,
                    row_start + block * RATIO - prefix,
                    request,
                    block * RATIO + RATIO - 1,
                )
            )
        row_start += extend
    capacity = rows // RATIO + batch
    groups.extend([(0, 0, 0, RATIO - 1)] * (capacity - len(groups)))
    plan = torch.tensor(groups, dtype=torch.int64, device=device)
    logical = torch.tensor(logical_host, dtype=torch.int64, device=device)
    positions = (
        logical.clone()
        if position_axes == 1
        else torch.stack((logical, logical + 3, logical + 7))
    )
    state = _state(device, (batch + 1) * RING, (starts[-1] + width) // RATIO, generator)
    inputs = dict(
        qk=torch.randn(
            (rows, (HEADS + 1) * DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ).to(torch.bfloat16),
        heads=HEADS,
        positions=positions,
        logical_positions=logical,
        state_slots=torch.tensor(slots_host, dtype=torch.int64, device=device),
        write_locs=plan[:, 0].to(torch.int32).contiguous(),
        member_rows=plan[:, 1].clone(),
        group_sequences=plan[:, 2].clone(),
        group_ends=plan[:, 3].clone(),
        rope_matrix=_rope_matrix(positions),
        token_slot_table=table,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        **clone_state(state),
        **_parameters(
            generator,
            device,
            max(seq_lens),
            section=section,
            interleaved=interleaved,
            cache_dtype=cache_dtype,
        ),
    )
    return SimpleNamespace(
        name=f"synthetic_s{'-'.join(map(str, seq_lens))}_e{'-'.join(map(str, extend_lens))}",
        inputs=inputs,
        state=state,
        rows=rows,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        host={"prefixes": prefixes, "request_ids": tuple(range(1, batch + 1))},
    )


@triton.jit
def _sqrt_fp32_reference(X, Y, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < size, other=0.0)
    tl.store(Y + offsets, tl.sqrt(value), offsets < size)


def norm_rope_reference(x, coordinates, weight, cache, axis_map, eps):
    """Independent FP32 RMS tree and per-operation BF16 norm/NeoX RoPE."""
    shape = x.shape
    values = x.reshape(-1, DIM).float()
    squares = (values * values).reshape(-1, 2, 2, 2, 2, 2, 2, 2)
    for _ in range(4):
        squares = squares.sum(dim=4)
    total = squares.sum(dim=3).sum(dim=2).sum(dim=1)
    variance = total / DIM + eps
    root = torch.empty_like(variance)
    # Gemma's Triton path uses v_sqrt_f32, not Torch's corrected sqrtf.
    # This scalar primitive differs by an FP32 ULP at BF16 rounding boundaries.
    _sqrt_fp32_reference[(triton.cdiv(root.numel(), 256),)](
        variance, root, root.numel(), BLOCK=256
    )
    rstd = 1.0 / root
    normalized = (values * rstd[:, None] * (1.0 + weight.float())[None]).to(
        torch.bfloat16
    )
    half = cache.shape[1] // 2
    pairs = torch.arange(half, device=x.device)
    position = coordinates.reshape(-1, 3)[:, axis_map.long()]
    cosine = cache[position, pairs].to(torch.bfloat16).float()
    sine = cache[position, pairs + half].to(torch.bfloat16).float()
    left, right = normalized[:, :half].float(), normalized[:, half : 2 * half].float()
    left_cos, right_sin = (left * cosine).to(torch.bfloat16), (right * sine).to(
        torch.bfloat16
    )
    right_cos, left_sin = (right * cosine).to(torch.bfloat16), (left * sine).to(
        torch.bfloat16
    )
    rotated_left = (left_cos.float() - right_sin.float()).to(torch.bfloat16)
    rotated_right = (right_cos.float() + left_sin.float()).to(torch.bfloat16)
    return torch.cat(
        (rotated_left, rotated_right, normalized[:, 2 * half :]), dim=-1
    ).reshape(shape)


def _mean_reference(members):
    total = members[:, 0].float()
    for member in range(1, RATIO):
        total = total + members[:, member].float()
    return (total * (1.0 / RATIO)).to(torch.bfloat16)


@torch.no_grad()
def reference_prep(inputs, initial_state, *, decode=False):
    """Q, valid state writes and packed keys, without any implementation helper.

    Reserved ring rows [0, RING) and compressed slot 0 are inert racing destinations;
    they have no defined final value and are never used as reference data.
    """
    state = clone_state(initial_state)
    rows = inputs["qk"].shape[0]
    raw = inputs["qk"].reshape(rows, HEADS + 1, DIM)
    coordinates = _rope_matrix(inputs["positions"])
    q = norm_rope_reference(
        raw[:, :HEADS],
        coordinates[:, None].expand(-1, HEADS, -1),
        inputs["q_weight"],
        inputs["cos_sin_cache"],
        inputs["axis_map"],
        inputs["q_eps"],
    )
    live = inputs["state_slots"] >= RING
    slots = inputs["state_slots"][live].long()
    assert (
        slots.unique().numel() == slots.numel()
    ), "valid ring writes must be request-local and unique"
    state["key_state"][slots, 0] = raw[live, HEADS]
    state["rope_state"][slots] = coordinates[live]
    live = inputs["write_locs"] != 0
    destinations = inputs["write_locs"][live].long()
    assert (
        destinations.unique().numel() == destinations.numel()
    ), "duplicate compressed destinations"
    if destinations.numel():
        if decode:
            members = inputs["group_locs"][live].long()
            keys = state["key_state"][members, 0]
            group_coordinates = state["rope_state"][members[:, 0]]
        else:
            first = inputs["member_rows"][live].long()
            members = first[:, None] + torch.arange(RATIO, device=raw.device)
            keys = raw[members, HEADS]
            group_coordinates = inputs["rope_matrix"][first]
        compressed = norm_rope_reference(
            _mean_reference(keys),
            group_coordinates,
            inputs["k_weight"],
            inputs["cos_sin_cache"],
            inputs["axis_map"],
            inputs["k_eps"],
        )
        state["compressed"][destinations, 0] = compressed
    packed = None
    if not decode:
        keys = [
            state["compressed"][
                inputs["token_slot_table"][
                    request, : length // RATIO * RATIO : RATIO
                ].long()
                // RATIO,
                0,
            ]
            for request, length in enumerate(inputs["seq_lens"])
        ]
        packed = torch.cat(keys)
    return SimpleNamespace(q=q, state=state, packed=packed)


def assert_exact(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), label


def assert_state(source, expected):
    report = {}
    for name in STATE_NAMES:
        lo = 1 if name == "compressed" else RING
        actual, wanted = source.inputs[name][lo:], expected.state[name][lo:]
        assert_exact(actual, wanted, name)
        report[f"{name}_rows_written"] = int(
            (wanted != source.state[name][lo:]).flatten(1).any(1).sum()
        )
    return report


def _counts(positions, sequences):
    return (
        torch.minimum(
            (positions.long() + 1).clamp_min(0), sequences.long().clamp_min(0)
        )
        // RATIO
    )


def _blocks(output, counts):
    live = (
        torch.arange(TOPK, device=output.device)[None] < counts.clamp_max(TOPK)[:, None]
    )
    return torch.where(live, output[:, : TOPK * RATIO : RATIO] // RATIO, -1)


def expand_reference(blocks, positions, sequences):
    """Quartets first, then the visible incomplete group, then every remaining -1."""
    visible = torch.minimum(
        (positions.long() + 1).clamp_min(0), sequences.long().clamp_min(0)
    )
    selected = (visible // RATIO).clamp_max(TOPK)
    column = torch.arange(WIDTH, device=blocks.device)[None].expand(blocks.shape[0], -1)
    block = blocks.gather(1, (column // RATIO).clamp_max(TOPK - 1))
    quartets = block * RATIO + column % RATIO
    tail_offset = column - selected[:, None] * RATIO
    tail = (tail_offset >= 0) & (tail_offset < (visible % RATIO)[:, None])
    values = torch.where(
        column < selected[:, None] * RATIO,
        quartets,
        torch.where(tail, (visible // RATIO * RATIO)[:, None] + tail_offset, -1),
    )
    return values.to(torch.int32)


def assert_token_abi(actual, positions, sequences):
    rows = positions.numel()
    assert (
        actual.shape == (rows, WIDTH) and actual.dtype == torch.int32
    ), "int32 [rows,2051] ABI"
    counts = _counts(positions, sequences)
    blocks = _blocks(actual, counts)
    live = (
        torch.arange(TOPK, device=actual.device)[None] < counts.clamp_max(TOPK)[:, None]
    )
    assert torch.equal(blocks >= 0, live), "missing selected block"
    assert bool((blocks < counts[:, None]).all()), "noncausal block"
    ordered = torch.where(live, blocks, 1 << 30).sort(dim=1).values
    assert not bool(
        ((ordered[:, 1:] == ordered[:, :-1]) & live[:, 1:]).any()
    ), "duplicate block"
    assert torch.equal(
        torch.where(live, blocks, 1 << 30), ordered
    ), "blocks must ascend"
    assert torch.equal(
        actual, expand_reference(blocks, positions, sequences)
    ), "token quartet/tail/-1 ABI"
    return blocks, counts, live, ordered


def assert_topk(actual, positions, sequences, exact, *, equal_ties=False):
    blocks, counts, live, ordered = assert_token_abi(actual, positions, sequences)
    if equal_ties:
        lowest = torch.arange(TOPK, device=actual.device)[None].expand_as(blocks)
        assert torch.equal(
            ordered[live], lowest[live]
        ), "equal ties must keep lowest block ids"
    ranked = counts > TOPK
    worst = 0.0
    if bool(ranked.any()):
        scores, selected_blocks, limits = (
            exact[ranked],
            blocks[ranked].long(),
            counts[ranked],
        )
        columns = torch.arange(scores.shape[1], device=actual.device)[None]
        causal = columns < limits[:, None]
        assert bool(torch.isfinite(scores[causal]).all()), "nonfinite reference logits"
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            1, selected_blocks, True
        )
        high = scores.masked_fill(~causal | selected, -math.inf).amax(1)
        low = scores.masked_fill(~selected, math.inf).amin(1)
        scale = scores.abs().masked_fill(~causal, 0).amax(1).clamp_min(1e-30)
        gap = (high - low) / scale
        assert bool(torch.isfinite(gap).all()), "nonfinite top-k boundary"
        worst = max(0.0, float(gap.max()))
        assert worst <= 1e-5, {"worst_relative_boundary_violation": worst}
    return {
        "worst_relative_boundary_violation": worst,
        "tolerance": 1e-5,
        "checked_rows": actual.shape[0],
    }


def _scores(q, keys):
    # Separate heads bound reference memory even at 65536 keys; no MFMA/helper reuse.
    values = keys.double().T
    result = torch.zeros(
        (q.shape[0], keys.shape[0]), dtype=torch.float64, device=q.device
    )
    for head in range(HEADS):
        result = result + (q[:, head].double() @ values).relu()
    return result / math.sqrt(DIM)


def _selected_reference(exact, counts):
    blocks = torch.full(
        (counts.numel(), TOPK), -1, dtype=torch.int32, device=exact.device
    )
    take = min(TOPK, exact.shape[1])
    if take:
        causal = (
            torch.arange(exact.shape[1], device=exact.device)[None] < counts[:, None]
        )
        order = exact.masked_fill(~causal, -math.inf).argsort(
            dim=1, descending=True, stable=True
        )[:, :take]
        blocks[:, :take] = torch.where(
            torch.arange(take, device=exact.device)[None] < counts[:, None], order, -1
        )
    return blocks


@torch.no_grad()
def check(source, *, actual=None, q=None, packed=None, expected=None):
    """Check a prefill invocation against independent prep, state and all-row FP64 selection."""
    inputs = source.inputs
    expected = reference_prep(inputs, source.state) if expected is None else expected
    if actual is None:
        q = torch.empty(
            (source.rows, HEADS, DIM), dtype=torch.bfloat16, device=inputs["qk"].device
        )
        actual = run_prefill(**inputs, q_out=q)
        # The keys the logits read, request by request: a group's key sits at its first token's slot // RATIO.
        table, compressed = inputs["token_slot_table"], inputs["compressed"]
        packed = torch.cat(
            [
                compressed[table[s, : n // RATIO * RATIO : RATIO].long() // RATIO, 0]
                for s, n in enumerate(source.seq_lens)
            ]
        )
    assert (
        actual.shape == (source.rows, WIDTH) and actual.dtype == torch.int32
    ), "prefill token frame"
    if q is not None:
        assert_exact(q, expected.q, "index Q must be bitexact")
    if packed is not None:
        assert_exact(
            packed[: expected.packed.shape[0]],
            expected.packed,
            "packed keys must be bitexact",
        )
    report = dict(
        case=source.name,
        rows=source.rows,
        q_checked=q is not None,
        packed_checked=packed is not None,
        **assert_state(source, expected),
        different_token_sets=0,
        worst_relative_boundary_violation=0.0,
    )
    report.update(check_prefill_selection(source, actual, expected))
    return report


@torch.no_grad()
def check_prefill_selection(source, actual, expected):
    """Validate selection alone against independent Q and logical compressed keys."""
    inputs = source.inputs
    report = {"different_token_sets": 0, "worst_relative_boundary_violation": 0.0}
    row_base, key_base = 0, 0
    for length, extend in zip(source.seq_lens, source.extend_lens):
        key_count = length // RATIO
        for offset in range(0, extend, 64):
            first, last = row_base + offset, row_base + min(offset + 64, extend)
            positions = inputs["logical_positions"][first:last]
            sequences = torch.full_like(positions, length)
            counts = _counts(positions, sequences)
            visible_keys = min(
                key_count, (length - extend + min(offset + 64, extend)) // RATIO
            )
            exact = _scores(
                expected.q[first:last],
                expected.packed[key_base : key_base + visible_keys],
            )
            checked = assert_topk(actual[first:last], positions, sequences, exact)
            report["worst_relative_boundary_violation"] = max(
                report["worst_relative_boundary_violation"],
                checked["worst_relative_boundary_violation"],
            )
            tokens = expand_reference(
                _selected_reference(exact, counts), positions, sequences
            )
            report["different_token_sets"] += int(
                (actual[first:last].sort(1).values != tokens.sort(1).values)
                .any(1)
                .sum()
            )
        row_base, key_base = row_base + extend, key_base + key_count
    return report


@torch.no_grad()
def _check_selection(source, actual, *, q, compressed, logits=None):
    """Check a decode selection (and its logits when given) against FP64 scores of the reference Q and keys."""
    inputs = source.inputs
    assert (
        actual.shape == (q.shape[0], WIDTH) and actual.dtype == torch.int32
    ), "decode token frame"
    if logits is not None:
        assert (
            logits.shape == (q.shape[0], inputs["page_table"].shape[1] * 16)
            and logits.dtype == torch.float32
        )
    report = {
        "rows": q.shape[0],
        "width": inputs["page_table"].shape[1] * 16,
        "heads": q.shape[1],
        "different_token_sets": 0,
        "max_relative_logit_error": 0.0,
        "worst_relative_boundary_violation": 0.0,
    }
    for row, count in enumerate(inputs["lengths"].cpu().tolist()):
        pages = inputs["page_table"][row, : math.ceil(count / 16)].long()
        slots = (pages[:, None] * 16 + torch.arange(16, device=q.device)).flatten()[
            :count
        ]
        exact = _scores(q[row : row + 1, :HEADS], compressed.reshape(-1, DIM)[slots])
        positions, sequences = (
            inputs["logical_positions"][row : row + 1],
            inputs["seq_lens"][row : row + 1],
        )
        counts = _counts(positions, sequences)
        assert (
            int(counts[0]) == count
        ), "decode compressed length disagrees with the token frame"
        checked = assert_topk(actual[row : row + 1], positions, sequences, exact)
        report["worst_relative_boundary_violation"] = max(
            report["worst_relative_boundary_violation"],
            checked["worst_relative_boundary_violation"],
        )
        tokens = expand_reference(
            _selected_reference(exact, counts), positions, sequences
        )
        report["different_token_sets"] += int(
            not torch.equal(actual[row].sort().values, tokens[0].sort().values)
        )
        if logits is not None:
            assert bool(
                torch.isneginf(logits[row, count : math.ceil(count / 16) * 16]).all()
            ), "partial-page padding"
            if count:
                scale = exact.abs().max().clamp_min(1e-30)
                error = float(
                    (logits[row, :count].double() - exact[0]).abs().max() / scale
                )
                assert math.isfinite(error) and error <= 1e-6, {
                    "relative_logit_error": error
                }
                report["max_relative_logit_error"] = max(
                    report["max_relative_logit_error"], error
                )
    report["logits_checked"] = logits is not None
    return report


@torch.no_grad()
def decode_case(lengths, device, *, window=1, padding=0, seed=17, context=65536):
    """Static tensor buffers for after-projection decode, with page-64 request allocation.

    ``window`` > 1 builds a TARGET_VERIFY layout: each request owns ``window`` consecutive rows whose
    first row has sequence length ``lengths[i]``."""
    if (
        not lengths
        or min(lengths) < 1
        or max(lengths) + window - 1 > context
        or context % PAGE
        or padding < 0
        or not 1 <= window <= RATIO
    ):
        raise ValueError(
            "Require positive lengths whose windows fit a page-64 context, 1..4-row windows and "
            "nonnegative padding"
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    requests, rows, pages = (
        len(lengths),
        len(lengths) * window + padding,
        context // PAGE,
    )
    table = torch.zeros((requests + 1, context), dtype=torch.int32, device=device)
    for request in range(1, requests + 1):
        table[request] = PAGE * (1 + (request - 1) * pages) + torch.arange(
            context, device=device
        )
    slots = (1 + requests * pages) * 16
    state = _state(device, (requests + 1) * RING, slots, generator)
    for request, length in enumerate(lengths, 1):
        for position in range(max(0, length - 1 - RATIO), length - 1):
            state["rope_state"][request * RING + position % RING] = torch.tensor(
                (position, position + 3, position + 7), device=device
            )
    parameters = _parameters(generator, device, context)
    inputs = dict(
        qk=torch.empty((rows, (HEADS + 1) * DIM), dtype=torch.bfloat16, device=device),
        positions=torch.empty((3, rows), dtype=torch.int64, device=device),
        state_slots=torch.empty(rows, dtype=torch.int64, device=device),
        group_locs=torch.empty((rows, RATIO), dtype=torch.int32, device=device),
        write_locs=torch.empty(rows, dtype=torch.int32, device=device),
        page_table=torch.empty((rows, pages), dtype=torch.int32, device=device),
        lengths=torch.empty(rows, dtype=torch.int32, device=device),
        logical_positions=torch.empty(rows, dtype=torch.int32, device=device),
        seq_lens=torch.empty(rows, dtype=torch.int32, device=device),
        **clone_state(state),
        **parameters,
    )
    name = f"decode_n{'-'.join(map(str, lengths))}{f'_w{window}' if window > 1 else ''}_pad{padding}"
    source = SimpleNamespace(
        name=name,
        inputs=inputs,
        state=state,
        table=table,
        requests=requests,
        rows=rows,
        window=window,
        padding=padding,
        context=context,
        host={
            "request_ids": tuple(
                request for request in range(1, requests + 1) for _ in range(window)
            )
            + (0,) * padding
        },
        generator=generator,
    )
    decode_step(source, lengths)
    return source


@torch.no_grad()
def decode_step(source, lengths):
    """Refresh host metadata and existing tensor contents, never graph-visible addresses."""
    if (
        len(lengths) != source.requests
        or min(lengths) < 1
        or max(lengths) + source.window - 1 > source.context
    ):
        raise ValueError(
            "decode step must preserve the request frame and fit the context"
        )
    inputs, device = source.inputs, source.inputs["qk"].device
    if max(lengths) + source.window + 5 >= inputs["cos_sin_cache"].shape[0]:
        raise ValueError("RoPE cache does not cover this step's three position axes")
    source.host["sequence_lengths"] = (
        tuple(n + j for n in lengths for j in range(source.window))
        + (1,) * source.padding
    )
    source.host["compressed_lengths"] = tuple(
        n // RATIO for n in source.host["sequence_lengths"]
    )
    source.seq_lens = source.host["sequence_lengths"]
    sequences = torch.tensor(source.seq_lens, dtype=torch.int64, device=device)
    requests = torch.tensor(
        source.host["request_ids"], dtype=torch.int64, device=device
    )
    positions = sequences - 1
    inputs["seq_lens"].copy_(sequences)
    inputs["logical_positions"].copy_(positions)
    inputs["lengths"].copy_(sequences // RATIO)
    inputs["positions"].copy_(torch.stack((positions, positions + 3, positions + 7)))
    inputs["state_slots"].copy_(requests * RING + positions % RING)
    members = (
        positions[:, None] - torch.arange(RATIO - 1, -1, -1, device=device)
    ).clamp_min(0)
    inputs["group_locs"].copy_(requests[:, None] * RING + members % RING)
    last_slots = source.table[requests, positions].long()
    inputs["write_locs"].copy_(
        torch.where(sequences % RATIO == 0, last_slots // RATIO, 0)
    )
    inputs["page_table"].copy_(source.table[requests, ::PAGE].long() // PAGE)
    inputs["qk"].copy_(
        torch.randn(
            inputs["qk"].shape,
            generator=source.generator,
            dtype=torch.float32,
            device=device,
        )
    )


@torch.no_grad()
def check_decode(source, actual=None, *, q=None, logits=None, expected=None):
    """Check a decode (or verify) invocation; a supplied result is never replaced by another launch."""
    expected = (
        reference_prep(source.inputs, source.state, decode=True)
        if expected is None
        else expected
    )
    if actual is None:
        rows, width = source.rows, source.inputs["page_table"].shape[1] * 16
        device = source.inputs["qk"].device
        q = torch.empty((rows, HEADS, DIM), dtype=torch.bfloat16, device=device)
        logits = torch.empty(rows * width + 512, dtype=torch.float32, device=device)
        logits = logits[: rows * width].view(rows, width)
        actual = run_decode(
            **source.inputs, verify=source.window > 1, q_out=q, logits_out=logits
        )
    if q is not None:
        assert_exact(q, expected.q, "decode Q must be bitexact, including padding rows")
    report = dict(
        case=source.name, q_checked=q is not None, **assert_state(source, expected)
    )
    report["selection"] = _check_selection(
        source,
        actual,
        q=expected.q,
        compressed=expected.state["compressed"],
        logits=logits,
    )
    return report


def mask_decode_padding(source):
    """Mask padded rows without changing graph input addresses."""
    if source.padding:
        source.inputs["logical_positions"][-source.padding :] = -1
        source.inputs["seq_lens"][-source.padding :] = 0
        source.inputs["lengths"][-source.padding :] = 0
