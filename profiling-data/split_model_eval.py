"""Fit and evaluate the gfx942 decode split policy using measured sweep totals (CPU only)."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/tmp/ua_repro")
CU_COUNT = 304
BATCHES = (1, 2, 4, 5, 6, 8, 12, 16, 20, 24, 32, 36, 40, 44, 48, 56, 64, 80, 96, 128, 256)


def caps(page, max_k):
    length = min(max_k, 1024)
    return length, min(16, length // 64, (length + page - 1) // page)


def head(batch, page, max_k=32768):
    length, maximum = caps(page, max_k)
    if length < 512:
        return 1
    target = max(1, (CU_COUNT + batch * 16 - 1) // (batch * 16))
    if batch <= 4:
        target = 1 << (target - 1).bit_length()
    elif batch >= 32:
        target = 4
    return max(1, min(target, maximum))


def piecewise(batch, page, max_k=32768):
    length, maximum = caps(page, max_k)
    if length < 512:
        return 1
    if batch <= 4:
        target = max(1, (CU_COUNT + batch * 16 - 1) // (batch * 16))
        target = 1 << (target - 1).bit_length()
    elif batch < 32:
        target = max(2, (5 * CU_COUNT + 16 * batch) // (32 * batch))
    elif batch < 64:
        target = max(2, (5 * CU_COUNT + 8 * batch) // (16 * batch))
    else:
        target = 4
    return max(1, min(target, maximum))


def features(batch, split, max_k=32768):
    tiles = (min(max_k, 1024) + 31) // 32
    slots = CU_COUNT * 3
    full, rem = divmod(batch * 16 * split, slots)
    waves = full + (rem > 0)
    tile_count = (tiles + split - 1) // split
    return np.array((waves * tile_count,
                     tile_count * (full + rem / slots),
                     waves, split > 1), dtype=float)


def cost(batch, split, params, max_k=32768):
    return float(features(batch, split, max_k) @ params)


def model(batch, page, params, max_k=32768):
    length, maximum = caps(page, max_k)
    if length < 512:
        return 1
    return min(range(1, maximum + 1), key=lambda s: cost(batch, s, params, max_k))


def samples(path, small_only=False):
    data = pd.read_csv(DATA / path)
    if "context" in data:
        data = data.loc[data.context.eq(32768)]
    if small_only:
        data = data.loc[data.batch.le(4)]
    return data[["page", "batch", "S", "total_us"]].groupby(
        ["page", "batch", "S"], as_index=False
    ).total_us.min()


def cells_from(data):
    return {(page, batch): dict(zip(group.S, group.total_us))
            for (page, batch), group in data.groupby(["page", "batch"])}


def evaluate(cells, variants, params):
    print("| Page | Batch | Oracle S/us | HEAD S/loss | Piecewise S/loss | Model S/loss | Model tie (<2%) |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    losses = {name: [] for name in variants}
    missing = {name: [] for name in variants}
    failures = []
    for (page, batch), times in sorted(cells.items()):
        oracle = min(times, key=times.get)
        best = times[oracle]
        picks = []
        for name, choose in variants.items():
            pick = choose(batch, page)
            if pick not in times:
                missing[name].append((page, batch, pick))
                picks.append(f"{pick}/unmeasured")
            else:
                loss = 100 * (times[pick] / best - 1)
                losses[name].append(loss)
                picks.append(f"{pick}/{loss:.2f}%")
                if name == "model" and loss > 5:
                    failures.append((page, batch, pick, oracle, loss))
        pick = variants["model"](batch, page)
        near = [s for s in times if s != pick and
                cost(batch, s, params) <= cost(batch, pick, params) * 1.02]
        print(f"| {page} | {batch} | {oracle}/{best:.2f} | " + " | ".join(picks) +
              f" | {','.join(map(str, sorted(near))) or '-'} |")
    for name in variants:
        loss = losses[name]
        print(f"{name}: measured {len(loss)}/{len(cells)}; mean {np.mean(loss) if loss else float('nan'):.2f}%; max {max(loss, default=float('nan')):.2f}%; unmeasured {missing[name]}")
    print("model >5% cells:", failures)
    return losses, missing


def main():
    primary = samples("split_fill.csv")
    x = np.stack([features(row.batch, row.S) for row in primary.itertuples()])
    y = primary.total_us.to_numpy()
    params, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    error = y - x @ params
    print("fit a,b,c,d:", params, "normalized:", params / params[0])
    print(f"residual mean={np.mean(error):.3f} us, MAE={np.mean(abs(error)):.3f} us, RMS={np.sqrt(np.mean(error**2)):.3f} us, max_abs={max(abs(error)):.3f} us")
    for page, batch, split in ((32, 44, 1), (32, 44, 5), (32, 44, 2),
                               (32, 64, 1), (32, 64, 4)):
        measured = primary.loc[primary.page.eq(page) & primary.batch.eq(batch) & primary.S.eq(split), "total_us"].iloc[0]
        print(f"spot {page}/{batch}/{split}: hand={cost(batch, split, np.array((1.9,1.7,0,3))):.2f} fit={cost(batch, split, params):.2f} measured={measured:.2f}")
    for page, group in primary.groupby("page"):
        residual = group.total_us.to_numpy() - np.stack([features(r.batch, r.S) for r in group.itertuples()]) @ params
        print(f"page {page} residual MAE={np.mean(abs(residual)):.3f} RMS={np.sqrt(np.mean(residual**2)):.3f} range=({min(residual):.3f},{max(residual):.3f}) us")
    variants = {"HEAD": head, "piecewise": piecewise,
                "model": lambda b, p: model(b, p, params)}
    print("Unconstrained primary grid")
    losses, missing = evaluate(cells_from(primary), variants, params)
    chosen = params
    if max(losses["model"]) > 5:
        for name, cols in (("drop c", (0, 1, 3)), ("drop d", (0, 1, 2)),
                           ("fix b/a=1.7/1.9", (0, 2, 3))):
            xx = x[:, cols] if name != "fix b/a=1.7/1.9" else np.column_stack((x[:, 0] + (1.7 / 1.9) * x[:, 1], x[:, 2], x[:, 3]))
            fitted, _, _, _ = np.linalg.lstsq(xx, y, rcond=None)
            candidate = np.zeros(4)
            if name == "fix b/a=1.7/1.9":
                candidate[:] = fitted[0], fitted[0] * (1.7 / 1.9), fitted[1], fitted[2]
            else:
                candidate[list(cols)] = fitted
            variants_alt = {"HEAD": head, "piecewise": piecewise,
                            "model": lambda b, p, v=candidate: model(b, p, v)}
            print(name, "parameters", candidate, "residual RMS", np.sqrt(np.mean((y - x @ candidate)**2)))
            alt_losses, alt_missing = evaluate(cells_from(primary), variants_alt, candidate)
            if name == "fix b/a=1.7/1.9" and max(alt_losses["model"]) <= 5 and not alt_missing["model"]:
                chosen = candidate
    params = chosen
    print("Selected parameters:", params, "normalized:", params / params[0])
    variants["model"] = lambda b, p: model(b, p, params)
    print("Additional prefetch B8")
    prefetch = samples("split_sweep_prefetch.csv")
    evaluate(cells_from(prefetch.loc[prefetch.batch.eq(8)]), variants, params)
    print("Older HEAD small-batch ranking (not used in fit)")
    evaluate(cells_from(samples("split_sweep_wide.csv", small_only=True)), variants, params)
    print("| Batch | p32 HEAD | p32 piecewise | p32 model | p64 HEAD | p64 piecewise | p64 model |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for batch in BATCHES:
        picks = [choose(batch, page) for page in (32, 64) for choose in variants.values()]
        print(f"| {batch} | " + " | ".join(map(str, picks)) + " |")
    for batch in (16, 64):
        picks = [choose(batch, page, 600) if choose != variants["model"] else model(batch, page, params, 600)
                 for page in (32, 64) for choose in variants.values()]
        print(f"| {batch} (600) | " + " | ".join(map(str, picks)) + " |")


if __name__ == "__main__":
    main()
