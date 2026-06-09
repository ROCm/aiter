"""Pure-Python model of jagged_dense_bmm_xcd._xcd_remap. Verify it is a
permutation (every output tile computed exactly once) for the sweep configs and
the real B1024_D512 dims, plus a small Mi correctness-test dim."""

NXCD = 8


def remap(xy, num_rows, num_cols, C, W, nXCD=NXCD):
    total = num_rows * num_cols
    period = nXCD * C
    prefix = total - (total % period)
    xcd = xy % nXCD
    local = xy // nXCD
    chunk_idx = local // C
    pos = local % C
    xy_g_remap = chunk_idx * (nXCD * C) + xcd * C + pos
    xy_g = xy_g_remap if xy < prefix else xy
    tids_per_grp = W * num_cols
    group_id = xy_g // tids_per_grp
    first_row = group_id * W
    remaining = num_rows - first_row
    win_h = remaining if remaining < W else W
    l = xy_g % tids_per_grp
    row = first_row + (l % win_h)
    col = l // win_h
    return row, col


def check(num_rows, num_cols, C, W):
    total = num_rows * num_cols
    seen = {}
    bad = 0
    for xy in range(total):
        row, col = remap(xy, num_rows, num_cols, C, W)
        if not (0 <= row < num_rows and 0 <= col < num_cols):
            bad += 1
            if bad <= 3:
                print(f"  OOB xy={xy} -> row={row} col={col}")
            continue
        key = (row, col)
        seen[key] = seen.get(key, 0) + 1
    dups = sum(1 for v in seen.values() if v > 1)
    missing = total - len(seen)
    ok = (bad == 0 and dups == 0 and missing == 0 and len(seen) == total)
    return ok, bad, dups, missing, len(seen), total


def xcd_of(xy):
    return xy % NXCD


def report_grouping(num_rows, num_cols, bm, C, W):
    """For each group (row // bm), which XCDs serve its tiles? Fewer XCDs/group
    is better (more L2 reuse). Report avg #distinct XCDs per group."""
    total = num_rows * num_cols
    n_groups = num_rows // bm
    # invert: for each raw xy we know its hw XCD; find which (group) it lands on
    grp_xcds = {g: set() for g in range(n_groups)}
    for xy in range(total):
        row, col = remap(xy, num_rows, num_cols, C, W)
        g = row // bm
        grp_xcds[g].add(xcd_of(xy))
    avgx = sum(len(s) for s in grp_xcds.values()) / n_groups
    return avgx


CONFIGS = [(4, 16), (8, 16), (8, 32), (5, 25), (8, 64)]

print("=== B1024_D512 real dims: bm=60, N_BLOCKS=4, n_groups=1024 ===")
bm, ncols, ng = 60, 4, 1024
nrows = bm * ng
for W, C in CONFIGS:
    ok, bad, dups, missing, nseen, total = check(nrows, ncols, C, W)
    avgx = report_grouping(nrows, ncols, bm, C, W)
    print(f"  W={W:>2} C={C:>2}: perm={ok}  (bad={bad} dups={dups} missing={missing} "
          f"seen={nseen}/{total})  avg_XCDs_per_group={avgx:.2f}/8")

# baseline (no remap) grouping reference
def baseline_avgx(nrows, ncols, bm, ng):
    grp_xcds = {g: set() for g in range(ng)}
    total = nrows * ncols
    # baseline: hw linear id == launched (off_b*gx + pid_mn), gx = bm*ncols
    gx = bm * ncols
    for off_b in range(ng):
        for pid_mn in range(gx):
            xy = off_b * gx + pid_mn
            grp_xcds[off_b].add(xy % NXCD)
    return sum(len(s) for s in grp_xcds.values()) / ng
print(f"  baseline (round-robin) avg_XCDs_per_group={baseline_avgx(nrows,ncols,bm,ng):.2f}/8")

print("\n=== correctness-test small dim: Mi=512 -> bm=4, N_BLOCKS=4, n_groups=4 ===")
bm2, ncols2, ng2 = 4, 4, 4
nrows2 = bm2 * ng2
for W, C in CONFIGS:
    ok, bad, dups, missing, nseen, total = check(nrows2, ncols2, C, W)
    print(f"  W={W:>2} C={C:>2}: perm={ok}  (bad={bad} dups={dups} missing={missing} "
          f"seen={nseen}/{total})")
