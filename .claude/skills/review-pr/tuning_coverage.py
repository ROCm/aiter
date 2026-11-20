"""Best-effort check for new tuned CSV workloads missing from paired untuned inputs."""

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

TUNED = re.compile(r"(^|[_-])tuned(?=[_-]|(?:\.csv)?$)", re.IGNORECASE)
UNTUNED = re.compile(r"(^|[_-])untuned(?=[_-]|(?:\.csv)?$)", re.IGNORECASE)
ARCH_TOKEN = re.compile(r"^gfx\d+[a-z0-9-]*$", re.IGNORECASE)

# From aiter/utility/dtypes.py defaultDtypes + normalize_fp8_dtype.
ARCH_FP8_E4M3 = {
    "gfx942": "torch.float8_e4m3fnuz",
    "gfx950": "torch.float8_e4m3fn",
    "gfx1100": "torch.float8_e4m3fn",
    "gfx1101": "torch.float8_e4m3fn",
    "gfx1102": "torch.float8_e4m3fn",
    "gfx1150": "torch.float8_e4m3fn",
    "gfx1151": "torch.float8_e4m3fn",
    "gfx1200": "torch.float8_e4m3fn",
    "gfx1201": "torch.float8_e4m3fn",
    "gfx1250": "torch.float8_e4m3fn",
    "gfx1250-strict": "torch.float8_e4m3fn",
}
E4M3_FLAVORS = frozenset({"torch.float8_e4m3fn", "torch.float8_e4m3fnuz"})
FP8_QUANT_COLS = frozenset({"q_dtype_a", "q_dtype_w"})
UNSUPPORTED_FAMILIES = ("chunk_gdn_h_opt", "paged_mqa_logits")
OUTPUT_PLACEHOLDER_COLS = frozenset(
    {
        "kernelid",
        "kernelname",
        "kernelname1",
        "kernelname2",
        "solidx",
        "us",
        "us1",
        "us2",
        "tflops",
        "bw",
        "err_ratio",
        "err1",
        "err2",
    }
)


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


def _stem_tokens(stem, marker_re):
    cleaned = marker_re.sub(lambda m: m.group(1), stem)
    return [tok for tok in re.split(r"[_-]+", cleaned.lower()) if tok]


def _unique_match(candidates):
    return candidates[0] if len(candidates) == 1 else None


def source_for(path, tracked):
    """Find an authoritative sibling or arch-compatible model untuned CSV."""
    p = PurePosixPath(path)
    sibling = str(p.with_name(TUNED.sub(r"\1untuned", p.name, count=1)))
    if sibling in tracked:
        return sibling
    if p.parent != PurePosixPath("aiter/configs/model_configs"):
        return None

    model_untuned = [
        name
        for name in tracked
        if PurePosixPath(name).parent == PurePosixPath("aiter/configs/model_configs")
        and PurePosixPath(name).suffix.lower() == ".csv"
        and UNTUNED.search(PurePosixPath(name).name)
    ]
    tuned_tokens = _stem_tokens(p.stem, TUNED)
    exact_matches = [
        name
        for name in model_untuned
        if sorted(_stem_tokens(PurePosixPath(name).stem, UNTUNED))
        == sorted(tuned_tokens)
    ]
    if exact_matches:
        return _unique_match(exact_matches)

    # Allow an arch-suffixed tuned file to pair only with an arch-neutral model input.
    tuned_no_arch = [t for t in tuned_tokens if not ARCH_TOKEN.match(t)]
    arch_neutral_matches = [
        name
        for name in model_untuned
        if not any(
            ARCH_TOKEN.match(t) for t in _stem_tokens(PurePosixPath(name).stem, UNTUNED)
        )
        and sorted(_stem_tokens(PurePosixPath(name).stem, UNTUNED))
        == sorted(tuned_no_arch)
    ]
    return _unique_match(arch_neutral_matches)


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


def value_key(value, column=None, fp8_target=None):
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower
    if column == "q_type" and value == "QuantType.per_128x128":
        return "QuantType.per_1x128"
    if fp8_target is not None and column in FP8_QUANT_COLS and value in E4M3_FLAVORS:
        return fp8_target
    # CSV readers infer numeric columns; preserve equality between 1 and 1.0.
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return number if number.is_finite() else value


def workload(row, columns, fp8_target=None):
    return tuple(
        value_key(row[column], column=column, fp8_target=fp8_target)
        for column in columns
    )


def _is_paired_tuned_csv(path):
    name = PurePosixPath(path).name.lower()
    return (
        name.endswith(".csv")
        and bool(TUNED.search(name))
        and not any(family in name for family in UNSUPPORTED_FAMILIES)
    )


def _uses_e4m3(rows, cols):
    return any(row.get(col) in E4M3_FLAVORS for _, row in rows for col in cols)


def check_csv(repo, base, head, old_path, path, source, moved_old_pairs=()):
    columns, inputs = read_csv(repo, head, source)
    tuned_columns, tuned = read_csv(repo, head, path)
    if set(columns) - set(tuned_columns):
        return []
    # Skip untuned CSVs that carry output/placeholder columns (e.g. empty kernelId).
    if any(col.lower() in OUTPUT_PLACEHOLDER_COLS for col in columns) or any(
        row[col] == "" for _, row in inputs for col in columns
    ):
        return []

    fp8_cols = FP8_QUANT_COLS.intersection(columns)
    has_gfx = "gfx" in tuned_columns
    # Gfx-less FP8 tables (e.g. cu_num-keyed files) are ambiguous across e4m3fn/e4m3fnuz; skip.
    if (
        fp8_cols
        and not has_gfx
        and (_uses_e4m3(inputs, fp8_cols) or _uses_e4m3(tuned, fp8_cols))
    ):
        return []

    def _old_key(row):
        key = workload(row, columns)
        return (row.get("gfx", ""), key) if fp8_cols else key

    old_keys = set()
    if old_path is not None:
        old_columns, old_rows = read_csv(repo, base, old_path)
        if set(columns) - set(old_columns):
            return []
        if fp8_cols and "gfx" not in old_columns:
            return []
        old_keys.update(_old_key(row) for _, row in old_rows)

    for sibling_old, sibling_new, sibling_status in moved_old_pairs:
        sib_old_cols, sib_old_rows = read_csv(repo, base, sibling_old)
        if set(columns) - set(sib_old_cols):
            continue
        if fp8_cols and "gfx" not in sib_old_cols:
            return []
        sib_old_keys = {_old_key(row) for _, row in sib_old_rows}
        if sibling_status == "D":
            old_keys.update(sib_old_keys)
        else:
            sib_new_cols, sib_new_rows = read_csv(repo, head, sibling_new)
            if set(columns) - set(sib_new_cols):
                continue
            if fp8_cols and "gfx" not in sib_new_cols:
                return []
            sib_head_keys = {_old_key(row) for _, row in sib_new_rows}
            old_keys.update(sib_old_keys - sib_head_keys)

    input_keys_by_arch = {
        None: {workload(row, columns) for _, row in inputs},
    }
    for target in set(ARCH_FP8_E4M3.values()):
        input_keys_by_arch[target] = {
            workload(row, columns, fp8_target=target) for _, row in inputs
        }

    new = {}
    for line, row in tuned:
        key = workload(row, columns)
        old_k = (row.get("gfx", ""), key) if fp8_cols else key
        if old_k not in old_keys:
            gfx = row.get("gfx", "")
            new.setdefault((gfx, key), (line, row))

    missing = []
    for (gfx, key), (line, row) in new.items():
        fp8_target = ARCH_FP8_E4M3.get(gfx)
        if key not in input_keys_by_arch[fp8_target]:
            missing.append((line, row))

    if not missing:
        return [
            f"COVERED {path}: {len(new)} new workload keys checked against {source}"
        ]
    lines = [
        f"MISSING {path}: {len(missing)} of {len(new)} new keys absent from {source}"
    ]
    for line, row in missing:
        key = {column: row[column] for column in columns}
        lines.append(f"  {path}:{line}: {json.dumps(key, ensure_ascii=True)}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="repository containing the PR commits")
    parser.add_argument("base", help="target branch tip")
    parser.add_argument("head", help="PR head")
    args = parser.parse_args()

    out_lines = []
    had_error = False
    try:
        merge_base = git(args.repo, "merge-base", args.base, args.head).decode().strip()
        base_tracked = paths_at(args.repo, merge_base)
        head_tracked = paths_at(args.repo, args.head)
        changed = list(changes(args.repo, merge_base, args.head))

        rename_map = {old_p: new_p for status, old_p, new_p in changed if status == "R"}
        base_tuned_by_source = {}
        for status, old_p, new_p in changed:
            if old_p is None or not _is_paired_tuned_csv(old_p):
                continue
            old_src = source_for(old_p, base_tracked)
            if old_src is not None:
                canon_src = rename_map.get(old_src, old_src)
                base_tuned_by_source.setdefault(canon_src, []).append(
                    (old_p, new_p, status)
                )

        checked = False
        for status, old_path, path in changed:
            if status == "D" or not _is_paired_tuned_csv(path):
                continue
            source = source_for(path, head_tracked)
            if source is None:
                continue

            effective_old_path = None
            if old_path is not None and _is_paired_tuned_csv(old_path):
                old_src = source_for(old_path, base_tracked)
                if old_src is not None and (
                    old_src == source or rename_map.get(old_src) == source
                ):
                    effective_old_path = old_path

            moved_old_pairs = [
                pair
                for pair in base_tuned_by_source.get(source, ())
                if pair[0] != effective_old_path
            ]

            try:
                lines = check_csv(
                    args.repo,
                    merge_base,
                    args.head,
                    effective_old_path,
                    path,
                    source,
                    moved_old_pairs=moved_old_pairs,
                )
            except (csv.Error, UnicodeDecodeError, ValueError) as exc:
                had_error = True
                out_lines.append(
                    f"SKIPPED: tuning coverage could not parse {path} ({exc}); "
                    "review these files manually"
                )
                continue
            if lines:
                checked = True
                out_lines.extend(lines)
        if not checked and not had_error:
            out_lines.append("No paired tuning CSVs to check.")
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        if not detail:
            cmd = " ".join(str(a) for a in exc.cmd)
            detail = (
                f"{cmd} exited with status {exc.returncode} "
                "(no common ancestor or missing object)"
            )
        print(
            f"SKIPPED: tuning coverage could not read the PR commits: {detail}; "
            "review these files manually"
        )
        return 1
    except OSError as exc:
        print(
            f"SKIPPED: tuning coverage could not access repository: {exc}; "
            "review these files manually"
        )
        return 1

    for line in out_lines:
        print(line)
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
