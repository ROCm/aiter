#!/usr/bin/env python3
"""Structural pre-filter for rule D9 (int32 overflow in index x stride arithmetic).

WHY THIS IS AN AST PASS AND NOT A REGEX
---------------------------------------
The previous implementation matched `([\\w\\.\\[\\]]+)\\s*\\*\\s*([\\w\\.\\[\\]]+)` on raw diff
text. That character class contains no comma and no space, so it cannot span a broadcast
subscript -- and `ptr[:, None] * stride` is the standard Triton pointer-arithmetic idiom.
Measured on ROCm/aiter#4978, the PR that introduced the moe_wgrad overflow later fixed by
#5132: both real defect lines were missed, the single moe_wgrad candidate reported was an
already-int64 line, and 390 candidates were emitted overall. 0 true positives, 1 false
positive, 390 candidates. The defect lived in the syntax tree; the scanner was reading text.

It also matched operand NAMES against two hard-coded lists (INDEXY / STRIDEY). A controlled
14-PR run moved that name dependence from D9's trigger into the scanner's operands and the
family still scored 0/3. Both lists are gone. The trigger below is derived from the code's
structure only.

THE TRIGGER
-----------
Inside a GPU kernel function, a multiplication is a candidate when ALL hold:

  1. it is a `BinOp(op=Mult)` that feeds pointer/index arithmetic -- it is an operand of an
     addition chain, which is how a buffer address is built;
  2. at least one operand derives from a **non-constexpr parameter of the enclosing kernel
     function**. Runtime-supplied parameters (strides, token counts, sequence extents) are the
     only values that can grow past 2^31; `tl.constexpr` parameters are compile-time tile
     constants and are excluded. This replaces the STRIDEY name list with provenance.
  3. no operand is 64-bit widened, counting widening applied on an EARLIER line and carried in
     through a local name. Without that propagation the scanner fires on the fixed code:
     #5132's fix hoists `token_row = (offs_token // top_k).to(tl.int64)` to its own line.

Only sites on lines the diff ADDS are reported.

EVIDENCE, NOT VERDICT
---------------------
Output is a candidate list. D9 still requires the reviewer to name the production scale at
which the product exceeds 2^31. What changed is that the list now contains the defect.

usage: scan_index_width.py --diff <file> [--source-root DIR] [--json]
       scan_index_width.py <repo> <pr> [--source-root DIR] [--json]
"""

import argparse
import ast
import json
import re
import subprocess
import sys

# Recognised 64-bit widening spellings. This is a list of WIDENING FORMS, not of variable
# names -- it says how a programmer writes "make this int64", not what they call their loop
# counter. Missing a spelling here costs a false positive, never a miss.
# Compared case-insensitively: FlyDSL spells it `fx.Int64(...)`, Triton `tl.int64`, C++
# `int64_t`. An exact-case set reported explicitly widened FlyDSL code as a candidate.
# This IS a spelling list, and it is the one remaining place where the scanner depends on how
# a widening is written rather than on what it does. A missing spelling costs a false
# positive, never a miss, which is the safe direction -- but add new spellings here rather
# than working around the false positive somewhere else.
WIDEN_ATTRS = {"int64", "int64_t", "long"}


def _is_widen_name(name):
    return name.lower() in WIDEN_ATTRS


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
INDEX_RE = re.compile(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)")
PY_EXT = (".py",)


# --------------------------------------------------------------------------- diff parsing
class FileDiff:
    def __init__(self, path):
        self.path = path
        self.added_lines = set()  # line numbers in the POST image
        self.post_blob = None
        self.is_new = False
        self.new_text_from_diff = []


def parse_diff(diff_text):
    """Collect, per file, the post-image line numbers the diff adds."""
    files, cur, new_lineno = {}, None, 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            cur = None
            continue
        if line.startswith("--- "):
            if line == "--- /dev/null" and cur is not None:
                cur.is_new = True
            continue
        m = INDEX_RE.match(line)
        if m:
            pending_blob = m.group(2)
            if cur is not None:
                cur.post_blob = pending_blob
            else:
                # index line precedes +++ ; stash it on the next file created
                parse_diff._pending = pending_blob
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                cur = None
                continue
            path = target[2:] if target.startswith("b/") else target
            cur = files.setdefault(path, FileDiff(path))
            blob = getattr(parse_diff, "_pending", None)
            if blob:
                cur.post_blob = blob
                parse_diff._pending = None
            continue
        if cur is None:
            continue
        m = HUNK_RE.match(line)
        if m:
            new_lineno = int(m.group(1))
            continue
        if line.startswith("+"):
            cur.added_lines.add(new_lineno)
            cur.new_text_from_diff.append((new_lineno, line[1:]))
            new_lineno += 1
        elif line.startswith("-") or line.startswith("\\"):
            continue
        else:
            new_lineno += 1
    return files


def parse_diff_for_new_file_check(diff_text, files):
    """A second pass: mark files whose old side is /dev/null (whole file is in the diff)."""
    cur = None
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            cur = "/dev/null" if line.strip() == "--- /dev/null" else "existing"
        elif line.startswith("+++ ") and cur is not None:
            target = line[4:].strip()
            if target != "/dev/null":
                path = target[2:] if target.startswith("b/") else target
                if path in files and cur == "/dev/null":
                    files[path].is_new = True
            cur = None


# ------------------------------------------------------------------- post-image retrieval
def read_post_image(fd, source_root):
    """Return (text, reason_if_unavailable). Never guesses: an unavailable post image is
    reported, not silently skipped, because a scanner that drops files quietly is
    indistinguishable from one that found nothing."""
    if source_root:
        try:
            with open(f"{source_root}/{fd.path}", encoding="utf-8") as handle:
                return handle.read(), None
        except OSError as exc:
            return None, f"--source-root has no readable {fd.path}: {exc.strerror}"
    if fd.post_blob:
        result = subprocess.run(
            ["git", "cat-file", "-p", fd.post_blob],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout, None
        blob_reason = (
            f"post-image blob {fd.post_blob} not in the local object store "
            f"(fetch the PR head: git fetch origin refs/pull/N/head)"
        )
    else:
        blob_reason = "diff carries no index line, so the post-image blob is unknown"
    if fd.is_new:
        # A file added by the PR is fully contained in the diff.
        lines, expected = [], 1
        for lineno, text in fd.new_text_from_diff:
            while expected < lineno:
                lines.append("")
                expected += 1
            lines.append(text)
            expected += 1
        return "\n".join(lines), None
    return None, blob_reason


# ------------------------------------------------------------------------- the AST pass
class KernelScopeScanner(ast.NodeVisitor):
    """One pass per function definition, so parameter provenance is scoped correctly."""

    def __init__(self, path, added_lines):
        self.path = path
        self.added = added_lines
        self.hits = []
        self.untyped_params = []

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _is_constexpr(annotation):
        """`x: tl.constexpr` / `gl.constexpr` / bare `constexpr`."""
        if annotation is None:
            return False
        node = annotation
        if isinstance(node, ast.Subscript):
            node = node.value
        name = getattr(node, "attr", None) or getattr(node, "id", None)
        return name == "constexpr"

    @staticmethod
    def _widens(node):
        """True if this subtree explicitly produces a 64-bit value."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and _is_widen_name(sub.attr):
                return True
            if isinstance(sub, ast.Name) and _is_widen_name(sub.id):
                return True
            if isinstance(sub, ast.Call):
                func = sub.func
                if getattr(func, "attr", None) == "to" and sub.args:
                    if KernelScopeScanner._widens(sub.args[0]):
                        return True
                if getattr(func, "attr", None) in {"astype", "cast"} and sub.args:
                    if KernelScopeScanner._widens(sub.args[0]):
                        return True
        return False

    @staticmethod
    def _own_nodes(func):
        """Walk this function's body without crossing into a nested function.

        `ast.walk` descends through nested defs, so every expression inside a nested helper was
        scanned in the ENCLOSING function's scope: with the outer function's parameters, the
        outer function's widened locals, and the outer function's device/host verdict. Four
        real candidates inside a `@flyc.kernel` body were classified host-side that way.
        """
        stack = list(func.body)
        while stack:
            node = stack.pop()
            yield node
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ):
                    continue
                stack.append(child)

    @staticmethod
    def _names_in(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    @staticmethod
    def _is_compile_time(node, constexpr_params):
        """True when this operand is built only from compile-time constants.

        `k_ptrs += BLOCK_N * stride_kn` is a pointer advance by one tile. It matches
        "runtime parameter inside a multiplication inside pointer arithmetic" exactly, and it
        cannot overflow, because the tile side is a `tl.constexpr`. Excluding operands with no
        data-dependent name removed the largest false-positive family without naming any
        variable: the test is where the value comes from, not what it is called.
        """
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if not names:
            return True  # pure literal arithmetic
        return names <= constexpr_params

    # -- the visit --------------------------------------------------------------
    @staticmethod
    def _device_scope(node):
        """Is this function compiled for the device?

        int32 overflow in index arithmetic is a DEVICE concern. Scanning every Python function
        put host-side bookkeeping in the candidate list: on a held-out PR all four candidates
        were `float` FLOP accounting in a `_flops_bytes()` helper and none were in the 916-line
        kernel. The docstring and this class's own name already claimed kernel scope; the code
        did not implement it.

        Two structural signals, no variable names: a `constexpr`-annotated parameter (only a
        device function has compile-time tile parameters), or a decorator that names a JIT or
        kernel entry point. The decorator test IS a spelling test -- it is the one place this
        scanner asks how something is written -- so a function matching neither is still
        scanned and reported separately rather than dropped, and the reviewer sees both lists.
        """
        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            if KernelScopeScanner._is_constexpr(getattr(arg, "annotation", None)):
                return True
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            parts = []
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            spelling = ".".join(reversed(parts)).lower()
            if "jit" in spelling or "kernel" in spelling:
                return True
        return False

    def visit_FunctionDef(self, node):
        args = node.args
        params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        if args.vararg:
            params.append(args.vararg)
        if args.kwarg:
            params.append(args.kwarg)

        runtime_params = {
            a.arg
            for a in params
            if not self._is_constexpr(getattr(a, "annotation", None))
        }
        constexpr_params = {
            a.arg for a in params if self._is_constexpr(getattr(a, "annotation", None))
        }
        # A stride-like parameter with no width annotation is reported separately, but the
        # judgement of which parameters are "stride-like" is left to the reviewer: we report
        # every runtime scalar parameter the DIFF introduced, not every one that matches a
        # name pattern.
        # Inherited, not re-derived. A device function's nested helpers carry no decorator and
        # no constexpr parameter of their own; deriving scope per function demoted four real
        # candidates inside a `@flyc.kernel` body to host scope. Code that runs on the device
        # because its enclosing function does is device code.
        outer_scope = getattr(self, "_scope", "host")
        outer_runtime = set(getattr(self, "_enclosing_runtime", set()))
        outer_constexpr = set(getattr(self, "_enclosing_constexpr", set()))
        outer_widened = set(getattr(self, "_enclosing_widened", set()))
        self._scope = (
            "device"
            if (outer_scope == "device" or self._device_scope(node))
            else "host"
        )
        # A nested helper closes over the enclosing kernel's parameters, so provenance and
        # widening both follow the scope chain.
        runtime_params = runtime_params | outer_runtime
        constexpr_params = constexpr_params | outer_constexpr
        self._enclosing_widened = outer_widened
        widened_locals = self._collect_widened_locals(node)
        self._enclosing_runtime = runtime_params
        self._enclosing_constexpr = constexpr_params
        self._enclosing_widened = widened_locals
        before = len(self.hits)
        self._scan_body(node, runtime_params, constexpr_params, widened_locals)

        # A runtime parameter is reported as "unannotated" only when this function actually
        # uses it as a multiplicand in pointer arithmetic. The old scanner reported every
        # parameter whose NAME looked stride-ish (244 on aiter#4978); reporting only the ones
        # the code multiplies keeps the same signal without the name list.
        multiplicands = set()
        for _, _, provenance, _scope in self.hits[before:]:
            multiplicands.update(provenance)
        for a in params:
            if (
                a.lineno in self.added
                and a.arg in multiplicands
                and getattr(a, "annotation", None) is None
            ):
                self.untyped_params.append((a.arg, a.lineno))
        # `visit`, not `generic_visit`: generic_visit descends to a statement's CHILDREN, so a
        # nested `def` that IS a statement of this body was never dispatched to this method.
        # While _scan_body walked the whole subtree that went unnoticed; once it stopped at
        # function boundaries, those bodies would have gone unscanned entirely.
        for child in node.body:
            self.visit(child)
        self._scope = outer_scope
        self._enclosing_runtime = outer_runtime
        self._enclosing_constexpr = outer_constexpr
        self._enclosing_widened = outer_widened

    visit_AsyncFunctionDef = visit_FunctionDef

    def _collect_widened_locals(self, func):
        """Names bound, anywhere in this function, to an explicitly widened expression.

        This is what makes the scanner quiet on #5132's fix, which hoists the widening onto
        its own statement (`token_row = (...).to(tl.int64)`) and then multiplies `token_row`.
        Flow-insensitive on purpose: assuming widened-somewhere means widened-here costs a
        miss only in code that widens a name and then rebinds it narrower, which would be a
        separate defect worth its own rule.
        """
        widened = set(self._enclosing_widened)
        for sub in self._own_nodes(func):
            targets = []
            if isinstance(sub, ast.Assign):
                targets, value = sub.targets, sub.value
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                targets, value = [sub.target], sub.value
            elif isinstance(sub, ast.AugAssign):
                targets, value = [sub.target], sub.value
            else:
                continue
            if not self._widens(value):
                continue
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        widened.add(name.id)
        return widened

    def _scan_body(self, func, runtime_params, constexpr_params, widened_locals):
        for sub in self._own_nodes(func):
            # `a_ptrs += BLOCK_K * stride_ak` builds the same address as
            # `a_ptrs = a_ptrs + BLOCK_K * stride_ak`, but an AugAssign is not a BinOp, so
            # matching only BinOp(Add) left the standard Triton pointer-advance idiom
            # unscanned. The constexpr filter in _is_compile_time was written for exactly
            # this shape and could never reach it.
            if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add):
                self._check_mult(
                    sub.value, runtime_params, constexpr_params, widened_locals
                )
                continue
            if not isinstance(sub, ast.BinOp) or not isinstance(sub.op, ast.Add):
                continue
            # Each addend of an address-building chain.
            for operand in (sub.left, sub.right):
                self._check_mult(
                    operand, runtime_params, constexpr_params, widened_locals
                )

    def _check_mult(self, node, runtime_params, constexpr_params, widened_locals):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            return
        lineno = getattr(node, "lineno", None)
        if lineno not in self.added:
            return
        # (3) already 64-bit, on this line or carried in from an earlier one
        if self._widens(node):
            return
        names = self._names_in(node)
        if names & widened_locals:
            return
        # (2) provenance: at least one operand is a runtime (non-constexpr) kernel parameter
        if not (names & runtime_params):
            return
        # (2b) and BOTH operands must be data-dependent. A tile-constant operand bounds the
        # product at compile time, so it cannot be the overflow.
        if self._is_compile_time(node.left, constexpr_params) or self._is_compile_time(
            node.right, constexpr_params
        ):
            return
        self.hits.append(
            (
                lineno,
                ast.unparse(node),
                sorted(names & runtime_params),
                self._scope,
            )
        )


def scan_file(path, text, added_lines):
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return None, None, f"{path} does not parse: {exc}"
    scanner = KernelScopeScanner(path, added_lines)
    scanner.visit(tree)
    return scanner.hits, scanner.untyped_params, None


# ----------------------------------------------------------------------------- interface
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?")
    parser.add_argument("pr", nargs="?")
    parser.add_argument("--diff")
    parser.add_argument(
        "--source-root",
        help="directory holding the PR's post-image files; when omitted the post image is "
        "recovered from the diff's index blobs via git cat-file",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.diff:
        if args.repo or args.pr:
            parser.error("--diff cannot be combined with repo/pr")
    elif not args.repo or not args.pr:
        parser.error("provide either --diff FILE or REPO PR")
    return args


def get_diff(args):
    if args.diff:
        with open(args.diff, encoding="utf-8") as handle:
            return handle.read()
    result = subprocess.run(
        ["gh", "pr", "diff", args.pr, "--repo", args.repo],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main():
    args = parse_args()
    diff_text = get_diff(args)
    files = parse_diff(diff_text)
    parse_diff_for_new_file_check(diff_text, files)

    candidates, params, unscanned = [], [], []
    for path, fd in sorted(files.items()):
        if not path.endswith(PY_EXT):
            continue
        if not fd.added_lines:
            continue
        text, reason = read_post_image(fd, args.source_root)
        if text is None:
            unscanned.append({"path": path, "reason": reason})
            continue
        hits, untyped, err = scan_file(path, text, fd.added_lines)
        if err:
            unscanned.append({"path": path, "reason": err})
            continue
        # One reasoning step clears one distinct expression, however many times a generated
        # kernel family repeats it, so identical expressions in a file collapse to one row
        # that carries its own occurrence count and every line it appears on.
        by_expr = {}
        for lineno, expr, provenance, scope in hits:
            entry = by_expr.setdefault(
                expr,
                {
                    "path": path,
                    "line": lineno,
                    "expression": expr,
                    "runtime_params": provenance,
                    "scope": scope,
                    "occurrences": 0,
                    "lines": [],
                },
            )
            entry["occurrences"] += 1
            entry["lines"].append(lineno)
            entry["line"] = min(entry["line"], lineno)
        for entry in by_expr.values():
            entry["lines"] = sorted(entry["lines"])[:8]
            candidates.append(entry)
        for name, lineno in untyped:
            params.append({"path": path, "line": lineno, "name": name})

    device = [c for c in candidates if c.get("scope") == "device"]
    host = [c for c in candidates if c.get("scope") != "device"]
    payload = {
        "index_stride_candidates": len(device),
        "host_scope_candidates": len(host),
        "unannotated_runtime_parameters": len(params),
        "total_candidates": len(device) + len(params),
        "candidates": device,
        "host_candidates": host,
        "parameters": params,
        "unscanned": unscanned,
    }
    if args.json:
        print(json.dumps(payload))
        # Always 0: an unscanned file is reported in the `unscanned` field, and a --json
        # caller is expected to read it. Turning it into a non-zero exit would be the
        # honest signal for the plain-text caller, which has nothing but the exit status --
        # but validate_pr.sh reads this JSON and does not inspect `unscanned` yet, so
        # raising here would only convert a silent gap into a skipped stage. Wire
        # --source-root and an `unscanned` check into validate_pr.sh first.
        return 0

    # Per-file rollup first. A reviewer clears index arithmetic a file at a time, and a PR
    # that adds a generated kernel family can carry a hundred distinct sites in one file;
    # printing all of them crowds the rule pass out of context. The rollup is always complete
    # and --json always carries every row, so nothing is discarded -- only deferred.
    per_file = {}
    for c in device:
        per_file.setdefault(c["path"], []).append(c)
    print(
        f"== index x stride in DEVICE code, reaching pointer arithmetic, no 64-bit "
        f"widening: {len(device)} distinct expressions in {len(per_file)} files =="
    )
    for path, rows in sorted(per_file.items(), key=lambda kv: -len(kv[1])):
        sites = sum(r["occurrences"] for r in rows)
        print(f"  {path}: {len(rows)} distinct ({sites} sites)")
    print()
    ROWS_PER_FILE = 5
    shown = []
    for path, rows in sorted(per_file.items(), key=lambda kv: -len(kv[1])):
        shown.extend(rows[:ROWS_PER_FILE])
        if len(rows) > ROWS_PER_FILE:
            shown.append(
                {
                    "path": path,
                    "line": 0,
                    "expression": (
                        f"... {len(rows) - ROWS_PER_FILE} more distinct expressions in this "
                        f"file; all rows are in --json output"
                    ),
                    "runtime_params": [],
                    "occurrences": 1,
                    "lines": [],
                }
            )
    for c in shown:
        provenance = ", ".join(c["runtime_params"])
        repeat = (
            f"  (x{c['occurrences']} at lines {c['lines']})"
            if c["occurrences"] > 1
            else ""
        )
        print(f"  {c['path']}:{c['line']}{repeat}")
        print(f"      {c['expression']}")
        print(f"      runtime kernel params in this expression: {provenance}")
    if host:
        # Listed, not merged: int32 overflow is a device concern, and host-side arithmetic in
        # this shape is usually FLOP accounting rather than an index. Kept visible because the
        # device/host test is a decorator spelling, and a wrong answer must cost a glance
        # rather than a miss.
        print(
            f"\n== same shape in HOST code ({len(host)}) -- normally not D9; check only if "
            f"one of these computes a device offset =="
        )
        for c in host[:5]:
            print(f"  {c['path']}:{c['line']}  {c['expression']}")
        if len(host) > 5:
            print(f"  ... {len(host) - 5} more; all rows are in --json output")
    print(
        f"\n== runtime kernel parameters added with no width annotation: {len(params)} =="
    )
    params_by_file = {}
    for p in params:
        params_by_file.setdefault(p["path"], []).append(p)
    for path, rows in sorted(params_by_file.items(), key=lambda kv: -len(kv[1])):
        names = sorted({r["name"] for r in rows})
        head = ", ".join(names[:8])
        tail = f", +{len(names) - 8} more" if len(names) > 8 else ""
        print(f"  {path}: {head}{tail}")
    if unscanned:
        # Loud on purpose. A file the scanner could not read is not a file with no defects,
        # and an empty candidate list that silently excluded files is the exact failure this
        # skill argues against elsewhere.
        print(f"\n== NOT SCANNED -- these files were not checked: {len(unscanned)} ==")
        for u in unscanned:
            print(f"  {u['path']}: {u['reason']}")
        print(
            "  D9 CANNOT be cleared from this run. Recover the post images "
            "(git fetch origin refs/pull/<N>/head, or pass --source-root) and re-scan."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
