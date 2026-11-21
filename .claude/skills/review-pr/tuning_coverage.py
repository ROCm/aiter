"""Best-effort bidirectional check for paired tuned/untuned tuning CSVs."""

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
        "errratio",
        "err1",
        "err2",
        "splitk",
        "libtype",
        "ksplit",
        "block_m",
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


def resolve_source(path, tracked):
    """Return ('matched', src), ('ambiguous', None), or ('missing', expected_sibling)."""
    p = PurePosixPath(path)
    sibling = str(p.with_name(TUNED.sub(r"\1untuned", p.name, count=1)))
    if sibling in tracked:
        return "matched", sibling
    if p.parent != PurePosixPath("aiter/configs/model_configs"):
        return "missing", sibling

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
    if len(exact_matches) == 1:
        return "matched", exact_matches[0]
    if len(exact_matches) > 1:
        return "ambiguous", None

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
    if len(arch_neutral_matches) == 1:
        return "matched", arch_neutral_matches[0]
    if len(arch_neutral_matches) > 1:
        return "ambiguous", None

    # If an arch-mismatched or alternate-named model untuned file plausibly matches
    # (e.g. qwen3_5_397b_untuned_gemm.csv for qwen3_5_397b_bf16_tuned_gemm.csv),
    # treat as ambiguous rather than falsely reporting missing untuned input.
    tuned_no_arch_set = set(tuned_no_arch)
    for name in model_untuned:
        u_no_arch = [
            t
            for t in _stem_tokens(PurePosixPath(name).stem, UNTUNED)
            if not ARCH_TOKEN.match(t)
        ]
        if sorted(u_no_arch) == sorted(tuned_no_arch) or (
            len(u_no_arch) >= 2 and set(u_no_arch).issubset(tuned_no_arch_set)
        ):
            return "ambiguous", None
    return "missing", sibling


def source_for(path, tracked):
    kind, val = resolve_source(path, tracked)
    return val if kind == "matched" else None


def tuned_for(untuned_path, tracked):
    """Exact inverse of source_for: tuned CSVs whose resolved source is untuned_path."""
    return sorted(
        name
        for name in tracked
        if _is_paired_tuned_csv(name) and source_for(name, tracked) == untuned_path
    )


def runtime_tuned_for(untuned_path, tracked):
    paired = set(tuned_for(untuned_path, tracked))
    associated = set(paired)
    config_dir = PurePosixPath("aiter/configs")
    model_dir = config_dir / "model_configs"
    source = PurePosixPath(untuned_path)
    if source.parent == config_dir:
        canonical = UNTUNED.sub(r"\1tuned", source.stem, count=1)
        families = {canonical}
    elif source.parent == model_dir:
        model_names = [
            PurePosixPath(name).stem
            for name in associated
            if PurePosixPath(name).parent == model_dir
        ]
        families = {
            PurePosixPath(name).stem
            for name in tracked
            if PurePosixPath(name).parent == config_dir
            and _is_paired_tuned_csv(name)
            and any(PurePosixPath(name).stem in model for model in model_names)
        }
    else:
        return sorted(associated), paired

    for name in tracked:
        p = PurePosixPath(name)
        if not _is_paired_tuned_csv(name):
            continue
        if p.parent == config_dir and p.stem in families:
            associated.add(name)
        elif p.parent == model_dir and any(family in p.stem for family in families):
            associated.add(name)
    return sorted(associated), paired


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


def _is_paired_untuned_csv(path):
    name = PurePosixPath(path).name.lower()
    return (
        name.endswith(".csv")
        and bool(UNTUNED.search(name))
        and not any(family in name for family in UNSUPPORTED_FAMILIES)
    )


def _uses_e4m3(rows, cols):
    return any(row.get(col) in E4M3_FLAVORS for _, row in rows for col in cols)


def _has_placeholder_or_output(columns, rows):
    return any(col.lower() in OUTPUT_PLACEHOLDER_COLS for col in columns) or any(
        row[col] == "" for _, row in rows for col in columns
    )


def _add_partial_keys(partial, columns, sibling_columns, rows, fp8_cols):
    shared = tuple(column for column in columns if column in sibling_columns)
    for _, row in rows:
        gfx = row.get("gfx") if fp8_cols and "gfx" in sibling_columns else None
        key = workload(row, shared, fp8_target="torch.float8_e4m3fn")
        partial.setdefault((shared, gfx), set()).add(key)


def _possibly_covered(row, partial, gfx=None):
    return any(
        (gfx is None or sibling_gfx is None or gfx == sibling_gfx)
        and workload(row, columns, fp8_target="torch.float8_e4m3fn") in keys
        for (columns, sibling_gfx), keys in partial.items()
    )


def _other_tuned_head_keys(
    repo, head, source, exclude_path, columns, fp8_cols, head_tracked
):
    keys = set()
    partial = {}
    siblings, paired = runtime_tuned_for(source, head_tracked)
    for sibling_tuned in siblings:
        if sibling_tuned == exclude_path:
            continue
        s_cols, s_rows = read_csv(repo, head, sibling_tuned)
        if set(columns) - set(s_cols) or (fp8_cols and "gfx" not in s_cols):
            if sibling_tuned in paired:
                return None
            _add_partial_keys(partial, columns, s_cols, s_rows, fp8_cols)
            continue
        for _, row in s_rows:
            k = workload(row, columns)
            keys.add((row.get("gfx", ""), k) if fp8_cols else k)
    return keys, partial


def check_no_untuned(repo, base, head, old_path, path, old_source, expected_source):
    key_cols, inputs = read_csv(repo, base, old_source)
    tuned_cols, tuned = read_csv(repo, head, path)
    old_cols, old_rows = read_csv(repo, base, old_path)
    if (
        _has_placeholder_or_output(key_cols, inputs)
        or set(key_cols) - set(tuned_cols)
        or set(key_cols) - set(old_cols)
    ):
        return []
    fp8_cols = FP8_QUANT_COLS.intersection(key_cols)
    if fp8_cols and ("gfx" not in tuned_cols or "gfx" not in old_cols):
        return []

    def key(row):
        value = workload(row, key_cols)
        return (row.get("gfx", ""), value) if fp8_cols else value

    old_keys = {key(row) for _, row in old_rows}
    new_rows = [
        (line, row) for line, row in tuned if key(row) not in old_keys
    ]
    if not new_rows:
        return []
    lines = [
        f"NO_UNTUNED_INPUT {path}: {len(new_rows)} new workload rows have no untuned input (expected {expected_source})"
    ]
    for line, row in new_rows[:10]:
        sample = {c: row[c] for c in key_cols}
        lines.append(f"  {path}:{line}: {json.dumps(sample, ensure_ascii=True)}")
    return lines


def check_deleted_tuned(repo, base, head, old_path, source, head_tracked):
    if source not in head_tracked:
        return []
    columns, inputs = read_csv(repo, head, source)
    if _has_placeholder_or_output(columns, inputs):
        return []
    old_cols, old_rows = read_csv(repo, base, old_path)
    if set(columns) - set(old_cols):
        return []
    fp8_cols = FP8_QUANT_COLS.intersection(columns)
    if fp8_cols and "gfx" not in old_cols and _uses_e4m3(old_rows, fp8_cols):
        return []
    other = _other_tuned_head_keys(
        repo, head, source, None, columns, fp8_cols, head_tracked
    )
    if other is None:
        return []
    other_keys, partial = other
    input_keys_by_arch = {None: {workload(row, columns) for _, row in inputs}}
    for target in set(ARCH_FP8_E4M3.values()):
        input_keys_by_arch[target] = {
            workload(row, columns, fp8_target=target) for _, row in inputs
        }
    removed = {}
    for line, row in old_rows:
        key = workload(row, columns)
        gfx = row.get("gfx", "")
        full_k = (gfx, key) if fp8_cols else key
        if full_k in other_keys or _possibly_covered(row, partial, gfx if fp8_cols else None):
            continue
        fp8_target = ARCH_FP8_E4M3.get(gfx)
        if key in input_keys_by_arch[fp8_target]:
            removed.setdefault(full_k, (line, row))
    if not removed:
        return []
    lines = [
        f"REMOVED_TUNED {old_path}: {len(removed)} removed keys still present in {source}"
    ]
    for line, row in removed.values():
        key = {column: row[column] for column in columns}
        lines.append(f"  {old_path}:{line}: {json.dumps(key, ensure_ascii=True)}")
    return lines


def check_csv(
    repo, base, head, old_path, path, source, head_tracked, moved_old_pairs=()
):
    columns, inputs = read_csv(repo, head, source)
    tuned_columns, tuned = read_csv(repo, head, path)
    if set(columns) - set(tuned_columns):
        return []
    if _has_placeholder_or_output(columns, inputs):
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
    old_rows = []
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
    head_full_keys = set()
    for line, row in tuned:
        key = workload(row, columns)
        old_k = (row.get("gfx", ""), key) if fp8_cols else key
        head_full_keys.add(old_k)
        if old_k not in old_keys:
            gfx = row.get("gfx", "")
            new.setdefault((gfx, key), (line, row))

    missing = []
    for (gfx, key), (line, row) in new.items():
        fp8_target = ARCH_FP8_E4M3.get(gfx)
        if key not in input_keys_by_arch[fp8_target]:
            missing.append((line, row))

    lines = []
    if missing:
        lines.append(
            f"MISSING {path}: {len(missing)} of {len(new)} new keys absent from {source}"
        )
        for line, row in missing:
            key = {column: row[column] for column in columns}
            lines.append(f"  {path}:{line}: {json.dumps(key, ensure_ascii=True)}")

    # Check tuned workloads removed from this file while remaining in the untuned input.
    if old_path is not None:
        other = _other_tuned_head_keys(
            repo, head, source, path, columns, fp8_cols, head_tracked
        )
        if other is not None:
            other_keys, partial = other
            removed = {}
            for line, row in old_rows:
                key = workload(row, columns)
                gfx = row.get("gfx", "")
                full_k = (gfx, key) if fp8_cols else key
                if (
                    full_k in head_full_keys
                    or full_k in other_keys
                    or _possibly_covered(row, partial, gfx if fp8_cols else None)
                ):
                    continue
                fp8_target = ARCH_FP8_E4M3.get(gfx)
                if key in input_keys_by_arch[fp8_target]:
                    removed.setdefault(full_k, (line, row))
            if removed:
                lines.append(
                    f"REMOVED_TUNED {path}: {len(removed)} removed keys still present in {source}"
                )
                for line, row in removed.values():
                    key = {column: row[column] for column in columns}
                    lines.append(
                        f"  {old_path}:{line}: {json.dumps(key, ensure_ascii=True)}"
                    )

    if not lines:
        return [
            f"COVERED {path}: {len(new)} new workload keys checked against {source}"
        ]
    return lines


def _has_ambiguous_tuned_candidate(untuned_path, tracked):
    p = PurePosixPath(untuned_path)
    if p.parent != PurePosixPath("aiter/configs/model_configs"):
        return False
    u_no_arch = {t for t in _stem_tokens(p.stem, UNTUNED) if not ARCH_TOKEN.match(t)}
    if len(u_no_arch) < 2:
        return False
    for name in tracked:
        np = PurePosixPath(name)
        if np.parent != PurePosixPath(
            "aiter/configs/model_configs"
        ) or not _is_paired_tuned_csv(name):
            continue
        if resolve_source(name, tracked)[0] != "ambiguous":
            continue
        t_no_arch = {t for t in _stem_tokens(np.stem, TUNED) if not ARCH_TOKEN.match(t)}
        if u_no_arch.issubset(t_no_arch):
            return True
    return False


def check_untuned_additions(repo, base, head, old_path, path, head_tracked):
    columns, inputs = read_csv(repo, head, path)
    if _has_placeholder_or_output(columns, inputs):
        return []

    old_keys = set()
    if old_path is not None:
        old_cols, old_rows = read_csv(repo, base, old_path)
        if set(columns) - set(old_cols):
            return []
        old_keys = {workload(row, columns) for _, row in old_rows}

    new_inputs = [
        (line, row) for line, row in inputs if workload(row, columns) not in old_keys
    ]
    if not new_inputs:
        return []

    associated_tuned, paired = runtime_tuned_for(path, head_tracked)
    if not associated_tuned:
        if _has_ambiguous_tuned_candidate(path, head_tracked):
            return []
        p = PurePosixPath(path)
        expected = str(p.with_name(UNTUNED.sub(r"\1tuned", p.name, count=1)))
        lines = [
            f"UNTUNED_MISSING {path}: {len(new_inputs)} of {len(new_inputs)} new untuned keys have no tuned CSV (expected {expected})"
        ]
        for line, row in new_inputs[:10]:
            key = {column: row[column] for column in columns}
            lines.append(f"  {path}:{line}: {json.dumps(key, ensure_ascii=True)}")
        return lines

    fp8_cols = FP8_QUANT_COLS.intersection(columns)
    usable_tuned = []
    tuned_arch_sets = []
    partial = {}
    for t_path in associated_tuned:
        t_cols, t_rows = read_csv(repo, head, t_path)
        if set(columns) - set(t_cols) or (
            fp8_cols
            and "gfx" not in t_cols
            and (_uses_e4m3(inputs, fp8_cols) or _uses_e4m3(t_rows, fp8_cols))
        ):
            if t_path in paired:
                return []
            _add_partial_keys(partial, columns, t_cols, t_rows, fp8_cols)
            continue
        usable_tuned.append(t_path)
        by_arch = {}
        for _, t_row in t_rows:
            fp8_target = ARCH_FP8_E4M3.get(t_row.get("gfx", ""))
            by_arch.setdefault(fp8_target, set()).add(workload(t_row, columns))
        tuned_arch_sets.append(by_arch)

    if not usable_tuned:
        return []

    missing_u = []
    for line, row in new_inputs:
        covered = any(
            workload(row, columns, fp8_target=fp8_target) in t_keys
            for by_arch in tuned_arch_sets
            for fp8_target, t_keys in by_arch.items()
        ) or _possibly_covered(row, partial)
        if not covered:
            missing_u.append((line, row))

    if not missing_u:
        return []
    lines = [
        f"UNTUNED_MISSING {path}: {len(missing_u)} of {len(new_inputs)} new untuned keys absent from {', '.join(usable_tuned)}"
    ]
    for line, row in missing_u:
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
            if status == "D":
                if _is_paired_tuned_csv(old_path):
                    old_src = source_for(old_path, base_tracked)
                    if old_src is not None:
                        canon_src = rename_map.get(old_src, old_src)
                        try:
                            lines = check_deleted_tuned(
                                args.repo,
                                merge_base,
                                args.head,
                                old_path,
                                canon_src,
                                head_tracked,
                            )
                        except (csv.Error, UnicodeDecodeError, ValueError) as exc:
                            had_error = True
                            out_lines.append(
                                f"SKIPPED: tuning coverage could not parse {old_path} ({exc}); "
                                "review these files manually"
                            )
                            continue
                        if lines:
                            checked = True
                            out_lines.extend(lines)
                continue

            if _is_paired_tuned_csv(path):
                kind, val = resolve_source(path, head_tracked)
                if kind == "ambiguous":
                    continue
                if kind == "missing":
                    old_src = (
                        source_for(old_path, base_tracked)
                        if old_path is not None and _is_paired_tuned_csv(old_path)
                        else None
                    )
                    if old_src is None or rename_map.get(old_src, old_src) in head_tracked:
                        continue
                    try:
                        lines = check_no_untuned(
                            args.repo, merge_base, args.head, old_path, path, old_src, val
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
                    continue

                source = val
                effective_old_path = None
                if old_path is not None and _is_paired_tuned_csv(old_path):
                    old_src = source_for(old_path, base_tracked)
                    if (old_src is None and old_path == path) or (
                        old_src is not None
                        and (old_src == source or rename_map.get(old_src) == source)
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
                        head_tracked,
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
            elif _is_paired_untuned_csv(path):
                try:
                    lines = check_untuned_additions(
                        args.repo, merge_base, args.head, old_path, path, head_tracked
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
