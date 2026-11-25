"""Best-effort helper: do a PR's tuned and untuned CSV row changes move together?

For every tuned/untuned CSV pair touched by the PR (paired by file name only), it
compares the rows the PR *adds or removes* and reports four kinds of mismatch:

  TUNED_NOT_IN_UNTUNED        added tuned workload, no matching untuned row
  UNTUNED_NOT_IN_TUNED        added untuned workload, no matching tuned row
  TUNED_REMOVED_UNTUNED_KEPT  tuned workload removed, untuned row left behind
  UNTUNED_REMOVED_TUNED_KEPT  untuned workload removed, tuned row left behind

Informational lines: CHECKED (pair compared, nothing to report), UNPAIRED (tuned
rows changed but no untuned CSV matches the name), NOT_CHECKED (format or pairing
the helper does not handle). `SKIPPED:` means a Git or parse failure.

This is deliberately not exhaustive. It does not model runtime-merged tuned files,
shared untuned inputs, or tuner-specific key layouts; legacy drift that the PR does
not touch is never reported. When a key appears in another CSV changed by the same
PR, the finding says so (often a move). The reviewer decides from the diff and these
hints; a finding here is a question to check, not a verdict.
"""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

TUNED = re.compile(r"(^|[_-])tuned(?=[_-]|$)", re.IGNORECASE)
UNTUNED = re.compile(r"(^|[_-])untuned(?=[_-]|$)", re.IGNORECASE)
ARCH_TOKEN = re.compile(r"^gfx\d+[a-z0-9]*$", re.IGNORECASE)
HW_COLS = ("gfx", "cu_num")
# Tuner outputs; never part of a workload key even if an untuned file carries them.
OUTPUT_COLS = frozenset(
    c.lower()
    for c in (
        "kernelId kernelName kernelName1 kernelName2 solidx us us1 us2 tflops bw "
        "err_ratio errRatio err1 err2 splitK libtype ksplit block_m run_1stage"
    ).split()
)
# Tuned rows serialise these keys differently from the untuned input.
UNSUPPORTED = ("chunk_gdn_h_opt", "paged_mqa_logits")
SAMPLE = 10


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def tracked_csvs(repo, rev):
    data = git(repo, "ls-tree", "-r", "--name-only", "-z", rev)
    names = (p.decode("utf-8", "surrogateescape") for p in data.split(b"\0") if p)
    return {n for n in names if n.lower().endswith(".csv")}


def changed_csvs(repo, base, head, renames=None):
    """(base_path_or_None, head_path_or_None) for every changed CSV; renames split.

    Renames are also recorded in `renames` as {new_path: old_path}.
    """
    data = git(
        repo, "diff", "--no-ext-diff", "--no-textconv", "--name-status", "-z",
        "--find-renames", base, head, "--",
    )
    fields = iter(data.decode("utf-8", "surrogateescape").split("\0")[:-1])
    for status in fields:
        a = next(fields)
        b = next(fields) if status[0] in "RC" else a
        if not (a.lower().endswith(".csv") or b.lower().endswith(".csv")):
            continue
        if status[0] in "RC":
            # A rename is a removal at the old path plus an addition at the new one;
            # the cross-file hints below make the move visible to the reviewer.
            if status[0] == "R":
                if renames is not None:
                    renames[b] = a
                yield a, None
            yield None, b
        elif status[0] == "A":
            yield None, a
        elif status[0] == "D":
            yield a, None
        else:
            yield a, a


def kind(path):
    name = PurePosixPath(path).stem
    if UNTUNED.search(name):
        return "untuned"
    if TUNED.search(name):
        return "tuned"
    return None


def tokens(path, marker):
    p = PurePosixPath(path)
    stem = marker.sub(lambda m: m.group(1), p.stem)
    return str(p.parent), sorted(t for t in re.split(r"[_-]+", stem.lower()) if t)


def untuned_for(tuned, tracked):
    """Untuned CSVs in the same directory whose name matches the tuned name.

    Exact token match first (any order: `x_bf16_tuned` ~ `x_untuned_bf16`), then the
    tuned name without an arch token against an arch-neutral untuned name.
    """
    folder, want = tokens(tuned, TUNED)
    cands = [(u, tokens(u, UNTUNED)) for u in tracked if kind(u) == "untuned"]
    cands = [(u, t) for u, (d, t) in cands if d == folder]
    exact = [u for u, t in cands if t == want]
    if exact:
        return exact
    no_arch = [t for t in want if not ARCH_TOKEN.match(t)]
    if no_arch == want:
        return []
    return [
        u for u, t in cands if t == no_arch and not any(ARCH_TOKEN.match(x) for x in t)
    ]


def tuned_for(untuned, tracked):
    return sorted(
        t for t in tracked if kind(t) == "tuned" and untuned_for(t, tracked) == [untuned]
    )


def read_csv(repo, rev, path):
    """Header and [(line, row)]; callers only ask for paths tracked at rev."""
    raw = git(repo, "show", f"{rev}:{path}")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    header = [h.strip() for h in next(reader, [])]
    if not header or not all(header) or len(header) != len(set(header)):
        raise ValueError(f"{path}: empty or duplicate column names")
    rows = []
    for row in reader:
        if not row or not any(v.strip() for v in row):
            continue
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{reader.line_num}: {len(row)} fields, expected {len(header)}"
            )
        rows.append((reader.line_num, dict(zip(header, (v.strip() for v in row)))))
    return header, rows


def norm(value, column):
    """Undo the spellings the tuners are known to rewrite; nothing more."""
    low = value.lower()
    if low in ("true", "false"):
        return low
    if column == "q_type" and value == "QuantType.per_128x128":
        return "QuantType.per_1x128"
    if value in ("torch.float8_e4m3fn", "torch.float8_e4m3fnuz"):
        return "torch.float8_e4m3"  # arch-native flavour; best-effort, not per-arch
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return number.normalize() if number.is_finite() else value


def key(row, cols):
    if any(row.get(c, "") == "" for c in cols):
        return None
    return tuple(norm(row[c], c) for c in cols)


def show(row, cols):
    return json.dumps({c: row[c] for c in cols}, ensure_ascii=True)


class Scan:
    def __init__(self, repo, base, head):
        self.repo, self.base, self.head = repo, base, head
        self.base_csvs = tracked_csvs(repo, base)
        self.head_csvs = tracked_csvs(repo, head)
        self.renames = {}
        self.changed = list(changed_csvs(repo, base, head, self.renames))
        self.cache = {}
        self.out = []
        self.failed = False

    def rows(self, rev, path):
        tracked = self.base_csvs if rev == self.base else self.head_csvs
        if path is None or path not in tracked:
            return None, []
        if (rev, path) not in self.cache:
            self.cache[(rev, path)] = read_csv(self.repo, rev, path)
        return self.cache[(rev, path)]

    def elsewhere(self, k, cols, want, exclude):
        """Other CSVs of kind `want` changed by this PR whose head has key k."""
        hits = []
        for _, path in self.changed:
            if path is None or path in exclude or kind(path) != want:
                continue
            header, rows = self.rows(self.head, path)
            if header and set(cols) <= set(header):
                if any(key(r, cols) == k for _, r in rows):
                    hits.append(path)
        return f"  [also at head in {', '.join(sorted(set(hits)))}]" if hits else ""

    def report(self, tag, title, items, cols, want, exclude):
        if not items:
            return 0
        self.out.append(f"{tag} {title}")
        for i, (path, line, row, k) in enumerate(items):
            if i == SAMPLE:
                self.out.append(f"  ... {len(items) - SAMPLE} more")
                break
            hint = self.elsewhere(k, cols, want, exclude)
            self.out.append(f"  {path}:{line}: {show(row, cols)}{hint}")
        return len(items)

    def check_group(self, untuned):
        """Compare one untuned CSV with every tuned CSV name-paired to it."""
        # Name-pair at each revision. A deleted input keeps the tuned files nothing else
        # claims; a tuned file whose name now pairs with a different input leaves this
        # group entirely. Rows are then compared as the union over the group's files, so
        # renames and moves between files of one pair are not row changes.
        def members(csvs):
            if untuned in csvs:
                return set(tuned_for(untuned, csvs))
            return {t for t in tuned_for(untuned, csvs | {untuned}) if not untuned_for(t, csvs)}

        def claimed_elsewhere(t, csvs):
            other = untuned_for(t, csvs)
            return len(other) == 1 and other[0] != untuned

        group = members(self.base_csvs) | members(self.head_csvs)
        head_files = [
            t for t in group
            if t in self.head_csvs and not claimed_elsewhere(t, self.head_csvs)
        ]
        tuned = sorted(t for t in group if t in head_files or t not in self.head_csvs)
        if not tuned:
            return  # input with no name-paired tuned file (e.g. shared input): skip
        # A renamed input keeps its base rows, so its legacy rows are not "added".
        u_hdr_b, u_base = self.rows(self.base, untuned)
        if u_hdr_b is None and untuned in self.renames:
            u_hdr_b, u_base = self.rows(self.base, self.renames[untuned])
        u_hdr_h, u_head = self.rows(self.head, untuned)
        u_hdr = u_hdr_h or u_hdr_b
        base_t = [(t, *self.rows(self.base, t)) for t in tuned if t in self.base_csvs]
        head_t = [(t, *self.rows(self.head, t)) for t in tuned if t in head_files]
        headers = [h for _, h, _ in base_t + head_t]
        pair = f"{', '.join(tuned)} <-> {untuned}"
        # Key: untuned input columns the tuned files also carry and the input always
        # fills. Anything else is ignored (and named), not a reason to skip the pair.
        inputs = [c for c in u_hdr if c.lower() not in OUTPUT_COLS]
        cols = [
            c for c in inputs
            if all(c in h for h in headers)
            and all(r.get(c, "") != "" for _, r in u_base + u_head)
        ]
        ignored = [c for c in inputs if c not in cols]
        if not cols:
            self.out.append(f"NOT_CHECKED {pair}: no input column shared with tuned rows")
            return
        pair += f" [key: {', '.join(cols)}" + (
            f"; ignored: {', '.join(ignored)}]" if ignored else "]")
        # Per-GPU rows are distinct tuned workloads when every tuned header has them.
        ident = cols + [c for c in HW_COLS if c not in cols and all(c in h for h in headers)]

        def keys(files, columns):
            return {key(r, columns) for _, _, rows in files for _, r in rows} - {None}

        tk_base, tk_head = keys(base_t, cols), keys(head_t, cols)
        base_ids = keys(base_t, ident)
        uk_base = {key(r, cols) for _, r in u_base} - {None}
        uk_head = {key(r, cols) for _, r in u_head} - {None}

        def first(files, keep):
            found = {}
            for t, _, rows in files:
                for line, r in rows:
                    k = key(r, cols)
                    if k is not None and k not in found and keep(r, k):
                        found[k] = (t, line, r, k)
            return list(found.values())

        added_tuned = first(
            head_t, lambda r, k: key(r, ident) not in base_ids and k not in uk_head)
        removed_tuned = first(
            base_t, lambda r, k: k not in tk_head and k in uk_head)

        added_untuned = [
            (untuned, line, r, k)
            for line, r in u_head
            if (k := key(r, cols)) is not None and k not in uk_base and k not in tk_head
        ]
        removed_untuned = [
            (untuned, line, r, k)
            for line, r in u_base
            if (k := key(r, cols)) is not None and k not in uk_head and k in tk_head
        ]

        own = set(tuned) | {untuned}
        n = self.report(
            "TUNED_NOT_IN_UNTUNED", f"{pair}: {len(added_tuned)} added tuned workloads "
            f"have no row in {untuned}", added_tuned, cols, "untuned", own)
        n += self.report(
            "UNTUNED_NOT_IN_TUNED", f"{pair}: {len(added_untuned)} added untuned "
            "workloads have no tuned row", added_untuned, cols, "tuned", own)
        n += self.report(
            "TUNED_REMOVED_UNTUNED_KEPT", f"{pair}: {len(removed_tuned)} removed tuned "
            f"workloads are still in {untuned}", removed_tuned, cols, "tuned", own)
        n += self.report(
            "UNTUNED_REMOVED_TUNED_KEPT", f"{pair}: {len(removed_untuned)} removed "
            "untuned workloads still have tuned rows", removed_untuned, cols, "untuned", own)
        if not n:
            # Rows count per GPU (a new gfx942 row for a shape tuned on gfx950 is a
            # new row but not a new workload), so a per-arch PR does not read as +0.
            head_ids = keys(head_t, ident)
            counts = (len(head_ids - base_ids), len(base_ids - head_ids),
                      len(tk_head - tk_base), len(tk_base - tk_head),
                      len(uk_head - uk_base), len(uk_base - uk_head))
            self.out.append(
                f"CHECKED {pair}: tuned +{{}}/-{{}} rows (+{{}}/-{{}} workloads), "
                "untuned +{}/-{} workloads, all matched".format(*counts)
            )

    def tuned_rows_changed(self, base_path, head_path):
        hdr_h, rows_h = self.rows(self.head, head_path)
        hdr_b, rows_b = self.rows(self.base, base_path)
        hdr = hdr_h or hdr_b
        cols = [c for c in hdr if c.lower() not in OUTPUT_COLS]
        if hdr_h and hdr_b and cols != [c for c in hdr_b if c.lower() not in OUTPUT_COLS]:
            return True
        return {key(r, cols) for _, r in rows_h} != {key(r, cols) for _, r in rows_b}

    def run(self):
        groups, seen_unpaired = set(), set()
        for base_path, head_path in self.changed:
            for path, tracked in ((base_path, self.base_csvs), (head_path, self.head_csvs)):
                if path is None:
                    continue
                k = kind(path)
                if k is None:
                    continue
                if any(f in path.lower() for f in UNSUPPORTED):
                    if path not in seen_unpaired:
                        seen_unpaired.add(path)
                        self.out.append(f"NOT_CHECKED {path}: key layout differs from its input")
                    continue
                if k == "untuned":
                    groups.add(path)
                    continue
                cands = untuned_for(path, tracked)
                if len(cands) > 1 and tracked is self.head_csvs:
                    # Ambiguous only at head: keep the pairing the base had, if any.
                    before = untuned_for(path, self.base_csvs) if path in self.base_csvs else []
                    if len(before) == 1:
                        cands = before
                if len(cands) == 1:
                    groups.add(cands[0])
                elif path not in seen_unpaired:
                    seen_unpaired.add(path)
                    if cands:
                        self.out.append(
                            f"NOT_CHECKED {path}: several untuned CSVs match the name "
                            f"({', '.join(sorted(cands))})")
                    else:
                        try:
                            changed_rows = self.tuned_rows_changed(base_path, head_path)
                        except (csv.Error, UnicodeDecodeError, ValueError) as exc:
                            self.fail(path, exc)
                            continue
                        if changed_rows:
                            self.out.append(
                                f"UNPAIRED {path}: workload rows changed but no untuned "
                                "CSV matches the name; not checked (flag only if this "
                                "family normally has an untuned input)")
        for untuned in sorted(groups):
            try:
                self.check_group(untuned)
            except (csv.Error, UnicodeDecodeError, ValueError) as exc:
                self.fail(untuned, exc)
        if not self.out:
            self.out.append("No paired tuning CSVs changed.")

    def fail(self, path, exc):
        self.failed = True
        self.out.append(
            f"SKIPPED: tuning coverage could not parse {path} ({exc}); review it manually")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="repository containing the PR commits")
    parser.add_argument("base", help="target branch tip")
    parser.add_argument("head", help="PR head")
    args = parser.parse_args()
    try:
        merge_base = git(args.repo, "merge-base", args.base, args.head).decode().strip()
        scan = Scan(args.repo, merge_base, args.head)
        scan.run()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip() or (
            f"git exited with status {exc.returncode} (no common ancestor or missing object)"
        )
        print(f"SKIPPED: tuning coverage could not read the PR commits: {detail}; "
              "review these files manually")
        return 1
    except OSError as exc:
        print(f"SKIPPED: tuning coverage could not access repository: {exc}; "
              "review these files manually")
        return 1
    print("\n".join(scan.out))
    return 1 if scan.failed else 0


if __name__ == "__main__":
    sys.exit(main())
