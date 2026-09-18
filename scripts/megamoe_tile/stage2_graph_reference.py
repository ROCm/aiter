"""CPU preparation for an eager reference with repeated expert selections."""

import math


def skewed_duplicate_routes(tokens, source_rank):
    """K3 EP16 stress input with three unequal active ranks and exact repeats.

    Per eight source tokens, ranks 0/9/3 receive 91/20/17 route slots;
    the other 13 ranks receive no work. Some tokens visit only one node.
    A token can put all 16 slots on one rank and repeat an expert four times.
    Swapping expert nodes moves every active rank to a previously empty rank,
    so graph routing-replay checks also exercise clearing stale partials.
    This is a correctness fixture, not an EPLB performance workload.
    """
    if tokens < 1 or not 0 <= source_rank < 16:
        raise ValueError("skewed K3 fixture requires tokens > 0 and rank in [0,16)")
    ids, weights = [], []
    for token in range(tokens):
        global_token = source_rank * tokens + token
        kind = global_token % 8
        row = []
        for slot in range(16):
            if kind < 4:
                owner = 0
            elif kind == 4:
                owner = 0 if slot < 12 else 9
            elif kind == 5:
                owner = 9
            elif kind == 6:
                owner = 0 if slot < 15 else 3
            else:
                owner = 3
            row.append(owner * 56 + (global_token * 3 + slot % 4) % 56)
        ids.append(row)
        weights.append([((slot + global_token) % 16 + 1) / 136.0 for slot in range(16)])
    return ids, weights


def coalesce_reference_routes(ids, weights, *, experts_per_rank, num_experts):
    """Make expert IDs unique while preserving weighted expert contributions.

    This is valid when weights are applied after the expert MLP (as in this
    benchmark): sum_s w_s * F_e(x) == sum_s(w_s) * F_e(x). Keep the first slot
    for each expert, and give its summed weight to that slot. Replace duplicate
    IDs with unused experts on the SAME rank, with zero weight. Keeping every
    slot's owner unchanged preserves MORI's rank/node deduplication behavior.

    This is an eager correctness oracle, never a candidate input rewrite or a
    timed baseline optimization. BF16 accumulation order can still differ.
    """
    if experts_per_rank < 1 or num_experts % experts_per_rank:
        raise ValueError("invalid expert partition")
    if len(ids) != len(weights):
        raise ValueError("route IDs and weights need the same row count")
    result_ids, result_weights = [], []
    duplicate_slots = 0
    for row_ids, row_weights in zip(ids, weights):
        if len(row_ids) != len(row_weights):
            raise ValueError("route IDs and weights need the same slot count")
        if any(not 0 <= expert < num_experts for expert in row_ids):
            raise ValueError("expert ID outside the partition")
        if any(not math.isfinite(weight) for weight in row_weights):
            raise ValueError("nonfinite route weight")
        out_ids, out_weights = list(row_ids), list(row_weights)
        used = set(row_ids)
        slots_by_expert = {}
        for slot, expert in enumerate(row_ids):
            slots_by_expert.setdefault(expert, []).append(slot)
        for expert, slots in slots_by_expert.items():
            out_weights[slots[0]] = math.fsum(row_weights[slot] for slot in slots)
            for slot in slots[1:]:
                begin = expert // experts_per_rank * experts_per_rank
                spare = next((e for e in range(begin, begin + experts_per_rank)
                              if e not in used), None)
                if spare is None:
                    raise ValueError("no unused same-rank expert for zero-weight slot")
                used.add(spare)
                out_ids[slot], out_weights[slot] = spare, 0.0
                duplicate_slots += 1
        result_ids.append(out_ids)
        result_weights.append(out_weights)
    return result_ids, result_weights, duplicate_slots
