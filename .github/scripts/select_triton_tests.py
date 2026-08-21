#!/usr/bin/env python
"""Select Triton unit tests to run for a PR, from its git diff, by OP category."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC = "aiter/ops/triton/"
KERNELS = SRC + "_triton_kernels/"
GLUON_KERNELS = SRC + "_gluon_kernels/"
CONFIGS = SRC + "configs/"
TESTS = "op_tests/triton_tests/"
BENCH = "op_tests/op_benchmarks/triton/"

# Any change here invalidates everything: run the full suite.
GLOBAL_PREFIXES = (
    ".github/",
    SRC + "utils/",
    KERNELS + "common/",
    TESTS + "utils/",
)

# ---------------------------------------------------------------------------
# THE MAP — edit here.
#
# category -> pytest targets (folders or single test files). List the
# category's own test folder AND every fused test elsewhere that exercises
# its kernels. Seeded from an import-graph dump of the tree (2026-08); keep
# it current as fusions are added.
# ---------------------------------------------------------------------------

TEST_TARGETS = {
    "attention": [TESTS + "attention/"],
    "chunk_delta_attn": [TESTS + "chunk_delta_attn/"],
    "comms": [],  # no dedicated UTs yet -> empty selection falls through to full suite
    "conv": [TESTS + "conv/"],
    "fusions": [TESTS + "fusions/"],
    "gated_delta_net": [
        TESTS + "chunk_delta_attn/",
        TESTS + "test_fused_rearrange_sigmoid_gdr.py",
    ],
    "gemm": [
        TESTS + "gemm/",
        TESTS + "fusions/test_fused_bmm_rope_kv_cache.py",
    ],
    # Gluon and triton backends share test files, so gluon changes run the
    # category folders with gluon-backed tests. Kernels under
    # _gluon_kernels/<arch>/<cat>/ resolve to <cat> directly and never reach
    # this entry — it covers the flat gluon/ wrapper modules.
    "gluon": [
        TESTS + "attention/",
        TESTS + "fusions/",
        TESTS + "gemm/",
        TESTS + "moe/",
        TESTS + "normalization/",
        TESTS + "quant/",
        TESTS + "test_pa_decode_gluon.py",
    ],
    "kimi_delta_attn": [TESTS + "chunk_delta_attn/"],
    "moe": [
        TESTS + "moe/",
        TESTS + "attention/test_fav3_sage.py",
        TESTS + "attention/test_fav3_sage_compile.py",
    ],
    "normalization": [TESTS + "normalization/"],
    "quant": [
        TESTS + "quant/",
        TESTS + "attention/test_fav3_sage.py",
        TESTS + "attention/test_fav3_sage_compile.py",
        TESTS + "fusions/test_fused_bmm_rope_kv_cache.py",
        TESTS + "fusions/test_fused_clamp_act_mul.py",
        TESTS + "fusions/test_fused_kv_cache.py",
        TESTS + "gemm/basic/test_gemm_a16w8_blockscale.py",
        TESTS + "gemm/basic/test_gemm_a16wfp4.py",
        TESTS + "gemm/batched/test_batched_gemm_a16wfp4.py",
        TESTS + "gemm/fused/test_fused_gemm_a16w16_quant_x.py",
        TESTS + "moe/test_moe.py",
        TESTS + "moe/test_moe_gemm_a4w4.py",
        TESTS + "moe/test_moe_gemm_a8w4.py",
        TESTS + "test_kv_cache.py",
    ],
    "rope": [
        TESTS + "rope/",
        TESTS + "fusions/test_fused_bmm_rope_kv_cache.py",
        TESTS + "fusions/test_fused_kv_cache.py",
        TESTS + "fusions/test_fused_qk_concat.py",
        TESTS + "fusions/test_fused_reduce_qk_norm_rope_swa_write.py",
    ],
    # Test-only folders (no source dir of the same name):
    "torch_compile": [TESTS + "torch_compile/"],
    "triton_metadata_redirect": [TESTS + "triton_metadata_redirect/"],
}

# Source dirs whose name is not the category it belongs to.
CATEGORY_ALIASES = {
    "flash_attn_triton_amd": "attention",  # vendored FA under _triton_kernels/
    "gated_delta_rule": "gated_delta_net",
}

# Loose modules at the top of aiter/ops/triton/ -> their tests.
LOOSE_MODULE_TESTS = {
    SRC
    + "activation.py": [
        TESTS + "fusions/test_fused_silu_mul.py",
        TESTS + "test_activation.py",
    ],
    SRC + "gather_kv_b_proj.py": [TESTS + "test_gather_kv_b_proj.py"],
    SRC + "gmm.py": [TESTS + "test_gmm.py"],
    SRC + "kv_cache.py": [TESTS + "test_kv_cache.py"],
    SRC + "softmax.py": [TESTS + "test_softmax.py"],
    SRC + "topk.py": [TESTS + "test_topk.py"],
}


def find_root():
    # Normally two levels above .github/scripts/; fall back to the current
    # directory so the script also works when run from a repo checkout.
    here = Path(__file__).resolve().parent.parent.parent
    return here if (here / TESTS).is_dir() else Path.cwd()


ROOT = find_root()


def log(msg):
    print(msg, file=sys.stderr)


def basename(path):
    return path.rsplit("/", 1)[-1]


def all_test_files():
    return sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / TESTS).rglob("test_*.py")
    )


def category_of(path):
    """Folder-derived category of a source/test file, or None for loose files."""
    for base in (KERNELS, GLUON_KERNELS, SRC, TESTS):
        if not path.startswith(base):
            continue
        parts = path[len(base) :].split("/")
        # Gluon is a backend, not an op: _gluon_kernels/<arch>/<cat>/...
        # resolves to <cat>. A file directly under the arch dir falls back
        # to the generic 'gluon' map entry.
        if base == GLUON_KERNELS and parts[0].startswith("gfx"):
            parts = parts[1:]
            if len(parts) < 2:
                return "gluon"
        if len(parts) < 2:  # the file sits directly in the base dir
            return None
        return CATEGORY_ALIASES.get(parts[0], parts[0])
    return None


def changed_files(args):
    if args.merge_ref:
        # A PR merge ref: diff against its first parent (the base branch).
        cmd = ["git", "diff", "--name-only", args.merge_ref + "^1", args.merge_ref]
    else:
        cmd = ["git", "diff", "--name-only", f"{args.target}...{args.source}"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def select(diff):
    """Map changed files to test files. Raises when it is not safe to pick
    a subset — the caller falls back to the full suite."""
    tests = all_test_files()
    reasons = []
    targets = set()  # map targets (folders or single test files)
    selected = set()  # concrete test files
    relevant = False

    def add_category(cat, changed_file):
        if cat not in TEST_TARGETS:
            raise RuntimeError(
                f"{changed_file}: category '{cat}' is not in TEST_TARGETS — add it"
            )
        targets.update(TEST_TARGETS[cat])
        reasons.append(f"{changed_file}: category '{cat}'")

    for f in diff:
        # Docs and placeholders never need tests.
        if f.endswith(".md") or basename(f) == ".gitkeep":
            continue

        if any(f.startswith(p) for p in GLOBAL_PREFIXES):
            raise RuntimeError(f"{f} is shared machinery/CI infra")

        if f.startswith(BENCH):
            reasons.append(f"{f}: benchmark change — no unit tests selected")
            continue

        if f.startswith(TESTS):
            relevant = True
            if basename(f).startswith("test_") and f.endswith(".py"):
                selected.add(f)
                reasons.append(f"{f}: changed test file — runs itself")
                continue
            cat = category_of(f)  # a test helper runs its whole folder
            if not cat:
                raise RuntimeError(f"{f} is a shared test helper")
            add_category(cat, f)
            continue

        if f.startswith(CONFIGS):
            relevant = True
            # Only the nested layout maps to a category:
            # configs/<arch>/<backend>/<op>/<d_type>/...
            parts = f[len(CONFIGS) :].split("/")
            nested = len(parts) >= 4 and parts[1] in ("triton", "gluon")
            if not f.endswith(".json") or not nested:
                raise RuntimeError(f"{f}: config outside the nested layout")
            add_category(parts[2], f)
            continue

        if f.startswith(SRC):
            relevant = True
            if basename(f) == "__init__.py":
                raise RuntimeError(f"{f}: package __init__ changed")
            if not f.endswith(".py"):
                raise RuntimeError(f"{f}: non-Python file under triton sources")
            cat = category_of(f)
            if cat:
                add_category(cat, f)
            elif f in LOOSE_MODULE_TESTS:
                targets.update(LOOSE_MODULE_TESTS[f])
                reasons.append(f"{f}: loose module — mapped tests")
            else:
                raise RuntimeError(f"{f}: loose module not in LOOSE_MODULE_TESTS")
            continue

        # Anything else (csrc/, other aiter/, ...) is covered by other CI jobs.

    # Turn map targets into concrete test files.
    for target in sorted(targets):
        if target.endswith("/"):
            found = [t for t in tests if t.startswith(target)]
        else:
            found = [target] if target in tests else []
        if not found:
            raise RuntimeError(f"map entry '{target}' matches nothing — fix the map")
        selected.update(found)

    if relevant and not selected:
        raise RuntimeError("relevant files changed but nothing was selected")
    return sorted(selected), reasons


def write_outputs(tests, reasons, is_full, output):
    Path(output).write_text("".join(t + "\n" for t in tests), encoding="utf-8")
    if is_full:
        header = f"Triton test selection: FULL SUITE ({len(tests)} files)"
    else:
        header = f"Triton test selection: {len(tests)} test file(s)"
    log(header)
    for r in reasons:
        log(f"  - {r}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {header}\n\n")
            fh.writelines(f"- {r}\n" for r in reasons)
            if not is_full:
                fh.write("\n<details><summary>Selected tests</summary>\n\n")
                fh.writelines(f"- `{t}`\n" for t in tests)
                fh.write("\n</details>\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--merge-ref", help="PR merge ref; diff is taken against its first parent"
    )
    mode.add_argument("--source", help="source ref (with --target)")
    mode.add_argument("--all", action="store_true", help="select the full suite")
    ap.add_argument("--target", help="target ref for --source mode")
    ap.add_argument("--output", default="selected_triton_tests.list")
    args = ap.parse_args()
    if args.source and not args.target:
        ap.error("--source requires --target")
    return args


def main():
    args = parse_args()
    if args.all:
        write_outputs(all_test_files(), ["full suite requested"], True, args.output)
        return
    try:
        diff = changed_files(args)
        log(f"Changed files ({len(diff)}):")
        for f in diff:
            log(f"  {f}")
        tests, reasons = select(diff)
        is_full = False
    except Exception as why:  # noqa: BLE001 -- any failure falls open to a full run
        tests, reasons, is_full = all_test_files(), [f"FULL SUITE: {why}"], True
    write_outputs(tests, reasons, is_full, args.output)


if __name__ == "__main__":
    main()
