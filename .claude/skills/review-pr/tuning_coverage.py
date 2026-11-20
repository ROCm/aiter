"""Report new tuned CSV workloads missing from their untuned input."""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

TUNED = re.compile(r"(^|_)tuned(?=_|(?:\.csv)?$)", re.IGNORECASE)
UNTUNED = re.compile(r"(^|_)untuned(?=_|(?:\.csv)?$)", re.IGNORECASE)
CONFIG_DIRS = {"config", "configs", "model_configs", "tuning"}


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def paths_at(repo, revision):
    data = git(repo, "ls-tree", "-r", "--name-only", "-z", revision)
    return {p.decode("utf-8", "surrogateescape") for p in data.split(b"\0") if p}


def changes(repo, base, head):
    data = git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-status",
        "-z",
        "--find-renames",
        base,
        head,
        "--",
    )
    fields = iter(data.decode("utf-8", "surrogateescape").split("\0")[:-1])
    for status in fields:
        old = next(fields)
        if status[0] in "RC":
            yield status[0], old, next(fields)
        else:
            yield status[0], old if status[0] != "A" else None, old


def source_for(path, tracked):
    """Prefer a same-name input; model files can also use the canonical family."""
    p = PurePosixPath(path)
    sibling = str(p.with_name(TUNED.sub(r"\1untuned", p.name, count=1)))
    if sibling in tracked:
        return sibling
    if p.parent != PurePosixPath("aiter/configs/model_configs"):
        return None

    candidates = []
    for name in tracked:
        source = PurePosixPath(name)
        if source.parent != PurePosixPath("aiter/configs") or not UNTUNED.search(
            source.name
        ):
            continue
        family = UNTUNED.sub(r"\1tuned", source.stem, count=1)
        if re.search(r"(^|[_-])" + re.escape(family) + r"($|[_-])", p.stem):
            candidates.append((len(family), name))
    if not candidates:
        return None
    longest = max(length for length, _ in candidates)
    matches = [name for length, name in candidates if length == longest]
    return matches[0] if len(matches) == 1 else None


def read_csv(repo, revision, path):
    text = git(repo, "show", f"{revision}:{path}").decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, None)
    if not header:
        raise ValueError(f"{path}: empty CSV header")
    header = [name.strip() for name in header]
    if not all(header) or len(header) != len(set(header)):
        raise ValueError(f"{path}: empty or duplicate column names")
    rows = []
    while True:
        line = reader.line_num + 1
        row = next(reader, None)
        if row is None:
            break
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{line}: row has {len(row)} fields, expected {len(header)}"
            )
        rows.append((line, dict(zip(header, (value.strip() for value in row)))))
    return header, rows


def value_key(value):
    # CSV readers infer numeric columns; preserve equality between 1 and 1.0.
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return number if number.is_finite() else value


def workload(row, columns):
    return tuple(value_key(row[column]) for column in columns)


def looks_like_tuning(path):
    p = PurePosixPath(path.lower())
    if p.suffix == ".csv":
        return bool(TUNED.search(p.name)) or (
            bool(CONFIG_DIRS.intersection(p.parts)) and not UNTUNED.search(p.name)
        )
    return p.suffix in {".json", ".yaml", ".yml", ".toml"} and (
        "tuned" in p.name or bool(CONFIG_DIRS.intersection(p.parts))
    )


def check_csv(repo, base, head, old_path, path, source):
    columns, inputs = read_csv(repo, head, source)
    tuned_columns, tuned = read_csv(repo, head, path)
    missing_columns = set(columns) - set(tuned_columns)
    if missing_columns:
        raise ValueError(f"{path}: missing input columns {sorted(missing_columns)}")
    old_keys = set()
    if old_path is not None:
        old_columns, old_rows = read_csv(repo, base, old_path)
        missing_columns = set(columns) - set(old_columns)
        if missing_columns:
            raise ValueError(
                f"{old_path} at merge base: missing input columns {sorted(missing_columns)}"
            )
        old_keys = {workload(row, columns) for _, row in old_rows}
    input_keys = {workload(row, columns) for _, row in inputs}
    new = {}
    for line, row in tuned:
        key = workload(row, columns)
        if key not in old_keys:
            new.setdefault(key, (line, row))
    missing = [(line, row) for key, (line, row) in new.items() if key not in input_keys]
    if not missing:
        print(f"COVERED {path}: {len(new)} new workload keys checked against {source}")
        return
    print(f"MISSING {path}: {len(missing)} of {len(new)} new keys absent from {source}")
    for line, row in missing:
        key = {column: row[column] for column in columns}
        print(f"  {path}:{line}: {json.dumps(key, ensure_ascii=True)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="repository containing the PR commits")
    parser.add_argument("base", help="target branch tip")
    parser.add_argument("head", help="PR head")
    args = parser.parse_args()

    try:
        merge_base = git(args.repo, "merge-base", args.base, args.head).decode().strip()
        tracked = paths_at(args.repo, args.head)
        changed = list(changes(args.repo, merge_base, args.head))
        found = False
        for status, old_path, path in changed:
            if status == "D" or not looks_like_tuning(path):
                continue
            found = True
            if not path.lower().endswith(".csv") or not TUNED.search(
                PurePosixPath(path).name
            ):
                print(
                    f"UNVERIFIED {path}: no automatic workload comparison; review manually"
                )
                continue
            source = source_for(path, tracked)
            if source is None:
                print(
                    f"UNVERIFIED {path}: no unambiguous untuned CSV found; review manually"
                )
                continue
            try:
                check_csv(args.repo, merge_base, args.head, old_path, path, source)
            except (csv.Error, UnicodeDecodeError, ValueError) as exc:
                print(f"UNVERIFIED {path}: {exc}")
        if not found:
            print("No changed tuning files identified.")
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        print(f"SKIPPED: tuning coverage could not read the PR commits: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
