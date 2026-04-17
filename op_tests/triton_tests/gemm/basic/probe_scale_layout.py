# SPDX-License-Identifier: MIT
# Probe script: discover and visualize how MX scales must be shuffled
# for the gfx1250 WMMA_SCALED unshuffle pattern.
#
# The kernel loads scales in packed (N//PF, K*PF) format via TDM, then
# applies an unshuffle (reshape+permute) to recover logical (N, K_GROUPS).
# This script applies that unshuffle to an identity tensor on the HOST
# to reveal exactly how elements move — no kernel needed.
#
# Usage:
#   python3 op_tests/triton_tests/gemm/basic/probe_scale_layout.py
#   python3 op_tests/triton_tests/gemm/basic/probe_scale_layout.py 128 256

import torch
from itertools import permutations


SCALE_GROUP_ELEMS = 32


def run_probe(BLOCK_N, BLOCK_K, preshuffle_factor=32):
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS
    PF = preshuffle_factor
    KW = 4 if K_GROUPS >= 4 else K_GROUPS
    PACKED_ROWS = BLOCK_N // PF
    PACKED_COLS = K_GROUPS * PF

    print(f"\n{'='*70}")
    print(f"BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, K_GROUPS={K_GROUPS}")
    print(f"PRESHUFFLE_FACTOR={PF}, SCALE_KWIDTH={KW}")
    print(f"Packed shape: ({PACKED_ROWS}, {PACKED_COLS})")
    print(f"Unshuffle 5D shape: ({PACKED_ROWS}, {K_GROUPS // KW}, {PF // 4}, 4, {KW})")
    print(f"Unshuffle permute:  (0, 3, 2, 1, 4)")
    print(f"{'='*70}")

    # Identity tensor: element [n, k] = n * K_GROUPS + k
    # This represents scales in LOGICAL order (what we want after unshuffle)
    identity = torch.arange(BLOCK_N * K_GROUPS, dtype=torch.int32)
    identity = identity.reshape(BLOCK_N, K_GROUPS)

    # --- What the UNSHUFFLE does ---
    # The kernel receives data in packed (PACKED_ROWS, PACKED_COLS) LDS format
    # and applies: reshape(5D).permute(0,3,2,1,4).reshape(N, K_GROUPS)
    #
    # If we feed the identity (logical order) through the unshuffle,
    # the output shows SCRAMBLED data — this is what happens when you
    # DON'T shuffle on the host but the kernel still unshuffles.
    unshuffle_of_identity = (
        identity.reshape(PACKED_ROWS, PACKED_COLS)
        .reshape(PACKED_ROWS, K_GROUPS // KW, PF // 4, 4, KW)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(BLOCK_N, K_GROUPS)
    )

    # Verify round-trip: shuffle then unshuffle should give identity
    shuffled_packed = (
        identity
        .reshape(PACKED_ROWS, 4, PF // 4, K_GROUPS // KW, KW)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(PACKED_ROWS, PACKED_COLS)
    )
    roundtrip = (
        shuffled_packed
        .reshape(PACKED_ROWS, K_GROUPS // KW, PF // 4, 4, KW)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(BLOCK_N, K_GROUPS)
    )
    assert torch.equal(identity, roundtrip), "Round-trip failed!"
    print(f"\nRound-trip verified: shuffle -> unshuffle = identity")

    # --- Show the scramble ---
    # "If I store scales in logical order and the kernel unshuffles, what do I get?"
    n_moved = (identity != unshuffle_of_identity).sum().item()
    total = BLOCK_N * K_GROUPS
    print(f"\n  {n_moved} / {total} elements moved\n")

    print(f"  Scramble mapping (first {min(BLOCK_N, 16)} rows):")
    print(f"  {'[n, k]':<10} {'got':<8} {'from (n,k)':<14} {'expected':<10}")
    print(f"  {'-'*50}")
    for n in range(min(BLOCK_N, 16)):
        for k in range(K_GROUPS):
            exp = identity[n, k].item()
            got = unshuffle_of_identity[n, k].item()
            src_n = got // K_GROUPS
            src_k = got % K_GROUPS
            marker = "\u2713" if exp == got else "\u2717"
            print(f"  [{n:2d},{k:2d}]    {got:4d}    ({src_n:2d}, {src_k:2d})     {exp:4d}  {marker}")

    # --- Stride analysis ---
    print(f"\n  Stride analysis on scrambled output:")
    flat = unshuffle_of_identity.flatten().float()
    for label, step in [("k+1 (within row)", 1), ("n+1 (next row)", K_GROUPS)]:
        diffs = flat[step:] - flat[:-step]
        unique = diffs.unique()
        if len(unique) <= 8:
            print(f"    step={step} ({label}): diffs = {[int(x) for x in unique.tolist()]}")
        else:
            print(f"    step={step} ({label}): {len(unique)} unique diffs")

    # --- Explain the pattern ---
    print(f"\n  === PATTERN EXPLANATION ===")
    print(f"  The unshuffle splits each packed row ({PACKED_COLS} elements) into:")
    print(f"    ({K_GROUPS // KW} k-chunks) x ({PF // 4} n-inner) x (4 n-sub) x ({KW} k-inner)")
    print(f"  and swaps n-sub with k-chunks via permute(0, 3, 2, 1, 4).")
    print()
    print(f"  In plain English:")
    print(f"    - Groups of {KW} consecutive k-scales stay together (SCALE_KWIDTH)")
    print(f"    - Within each packed row, the {PF} n-rows are split into")
    print(f"      4 sub-groups of {PF // 4} rows each")
    print(f"    - The permute interleaves these sub-groups with k-chunks")
    print()
    print(f"  Concretely for scale at logical [n, k]:")
    print(f"    n_chunk = n // {PF}")
    print(f"    n_sub   = (n % {PF}) // {PF // 4}     (which of the 4 sub-groups)")
    print(f"    n_inner = (n % {PF}) % {PF // 4}      (position within sub-group)")
    print(f"    k_outer = k // {KW}")
    print(f"    k_inner = k % {KW}")
    print()
    print(f"    Packed position: row = n_chunk")
    print(f"                     col = k_outer*{(PF//4)*4*KW} + n_inner*{4*KW} + n_sub*{KW} + k_inner")

    # Show a concrete example
    for n_ex, k_ex in [(0, 0), (0, KW), (1, 0), (5, 3)]:
        if n_ex < BLOCK_N and k_ex < K_GROUPS:
            nc = n_ex // PF
            ns = (n_ex % PF) // (PF // 4)
            ni = (n_ex % PF) % (PF // 4)
            ko = k_ex // KW
            ki = k_ex % KW
            col = ko * (PF // 4) * 4 * KW + ni * 4 * KW + ns * KW + ki
            print(f"\n    Example: [{n_ex}, {k_ex}] (val={n_ex * K_GROUPS + k_ex})")
            print(f"      n_chunk={nc}, n_sub={ns}, n_inner={ni}, k_outer={ko}, k_inner={ki}")
            print(f"      -> packed[{nc}, {col}]")

    # --- Code output ---
    print(f"\n  === CODE ===\n")
    print(f"  # Host-side shuffle (run once when loading model weights):")
    print(f"  def shuffle_scales(scales, PF={PF}, KW={KW}):")
    print(f"      N, K = scales.shape")
    print(f"      return (scales")
    print(f"          .reshape(N // PF, 4, PF // 4, K // KW, KW)")
    print(f"          .permute(0, 3, 2, 1, 4)")
    print(f"          .contiguous()")
    print(f"          .reshape(N // PF, K * PF))")
    print()
    print(f"  # Kernel-side unshuffle (in LDS, every K-tile):")
    print(f"  def unshuffle_scales(packed, N, K, PF={PF}, KW={KW}):")
    print(f"      return (packed")
    print(f"          .reshape(N // PF, K // KW, PF // 4, 4, KW)")
    print(f"          .permute(0, 3, 2, 1, 4)")
    print(f"          .reshape(N, K))")


def find_unshuffle(BLOCK_N, BLOCK_K, preshuffle_factor=32):
    """
    Brute-force: given the unshuffle is unknown, try all 5D reshapes
    and permutations to find what produces the observed scramble.
    Use this when porting to a new architecture.
    """
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS
    PF = preshuffle_factor
    PACKED_ROWS = BLOCK_N // PF
    PACKED_COLS = K_GROUPS * PF
    total = BLOCK_N * K_GROUPS

    # You would replace this with actual kernel output
    print(f"\n  To use find_unshuffle, replace 'target' with actual kernel output.")
    print(f"  For now, showing the search mechanism.\n")

    identity = torch.arange(total, dtype=torch.int32).reshape(BLOCK_N, K_GROUPS)

    def factors(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    def factor_triples(n):
        result = []
        for a in factors(n):
            for b in factors(n // a):
                c = n // (a * b)
                result.append((a, b, c))
        return result

    def factor_pairs(n):
        return [(a, n // a) for a in factors(n)]

    # Search over 5D reshapes of (PACKED_ROWS, PACKED_COLS)
    packed = identity.reshape(PACKED_ROWS, PACKED_COLS)
    results = []

    for (d0, d1, d2) in factor_triples(PACKED_ROWS):
        for (d3, d4) in factor_pairs(PACKED_COLS):
            shape = (d0, d1, d2, d3, d4)
            if d0 * d1 * d2 * d3 * d4 != total:
                continue
            try:
                reshaped = packed.reshape(shape)
            except:
                continue
            for perm in permutations(range(5)):
                try:
                    candidate = reshaped.permute(perm).contiguous().reshape(BLOCK_N, K_GROUPS)
                    # Check: candidate should NOT be identity (it's a scramble)
                    # and shuffle(candidate) via inverse should give identity
                    if not torch.equal(candidate, identity) and candidate.max() < total:
                        results.append((shape, list(perm), candidate))
                except:
                    continue

    print(f"  Found {len(results)} candidate reshapes+permutes")
    return results


if __name__ == "__main__":
    import sys

    configs = [
        # (BLOCK_N, BLOCK_K)
        (32,  256),
        (64,  256),
        (128, 256),
        (32,  128),
    ]

    if len(sys.argv) == 3:
        configs = [(int(sys.argv[1]), int(sys.argv[2]))]

    for BLOCK_N, BLOCK_K in configs:
        run_probe(BLOCK_N, BLOCK_K)
