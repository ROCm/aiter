# SPDX-License-Identifier: MIT
"""ROUND-NATIVE64 test support: the query_positions load seam.

The structural CPU gates replay builder helpers / wrapper statements from the
kernel source in hand-built namespaces. ROUND-NATIVE64 routed every
``query_positions`` access through one build-time seam (``_pos_loaders``):

    ld_pos(qp_rsrc, r)          == _ld_i32(qp_rsrc, r * 4)           (pos64=False)
    ld_uniform_pos(qp_rsrc, r)  == _ld_uniform_i32(qp_rsrc, r * 4)   (pos64=False)
    _plane_bytes(rows, qp_bytes) with qp_bytes == 4, kernel-name suffix qp_tag == ""

``bind_i32`` installs the real seam source plus the int32 (pos64=False)
bindings into such a namespace, so every gate keeps replaying exactly the
int32 specialization it replayed before. ``unseam`` maps the new source text
back to the pre-NATIVE64 text: a source-identity hash taken after ``unseam``
proves that the seam is the *only* change in a pinned function.
"""
from __future__ import annotations

import ast
import pathlib

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
SEAM_FUNCTIONS = ("_sat_pos64", "_ld_pos_i32", "_ld_pos_i64", "_ld_uniform_pos_i32",
                  "_ld_uniform_pos_i64", "_pos_loaders", "_qp_pad_i32")
SEAM_CONSTANTS = ("_POS_SAT_HI", "_POS_SAT_LO", "_POS64_KW", "_NO_KW")


def seam_nodes(tree=None):
    tree = tree if tree is not None else ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in SEAM_FUNCTIONS:
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in SEAM_CONSTANTS for t in targets):
                nodes.append(node)
    found = {getattr(n, "name", None) for n in nodes}
    missing = set(SEAM_FUNCTIONS) - found
    assert not missing, missing
    return nodes


def bind_i32(ns, tree=None):
    """Install the real seam source and the pos64=False bindings into ``ns``."""
    module = ast.Module(body=seam_nodes(tree), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), "exec"), ns)  # noqa: S102
    ns.update(ld_pos=ns["_ld_pos_i32"], ld_uniform_pos=ns["_ld_uniform_pos_i32"],
              qp_bytes=4, qp_tag="", qp_kw={}, pos64=False)
    return ns


def ensure_i32(ns, tree=None):
    """``bind_i32`` unless ``ns`` already holds seam functions bound to itself.

    Harnesses copy namespaces (``dict(base)``) and then override loaders such
    as ``_ld_i32``; the seam functions must resolve those names in the very
    namespace the replayed helpers run in, so (re)bind on every exec.
    """
    fn = ns.get("_ld_pos_i32")
    if fn is None or getattr(fn, "__globals__", None) is not ns:
        bind_i32(ns, tree)
    return ns


# New text -> pre-NATIVE64 text (exact, ordered; longest/most specific first).
_UNSEAM = (
    ("def _build_kernel(is_fp8: bool, threads: int, divisor: float, pos64: bool = False):\n"
     '    """One specialization per (dtype, block size, score divisor[, int64 positions])."""\n'
     "    ld_pos, _, qp_bytes, qp_tag = _pos_loaders(pos64)\n",
     "def _build_kernel(is_fp8: bool, threads: int, divisor: float):\n"
     '    """One specialization per (dtype, block size, score divisor)."""\n'),
    ('_t{threads}_flydsl"\n    ) + qp_tag\n', '_t{threads}_flydsl"\n    )\n'),
    ("qp_rsrc = _plane(QP, _plane_bytes(rows, qp_bytes))", "qp_rsrc = _plane(QP, _plane_bytes(rows, 4))"),
    ("pos = ld_pos(qp_rsrc, row)\n", "pos = _ld_i32(qp_rsrc, row * 4)\n"),
    ("pos = ld_pos(qp_rsrc, safe_row)", "pos = _ld_i32(qp_rsrc, safe_row * 4)"),
    ("pos = ld_uniform_pos(qp_rsrc, safe_row)", "pos = _ld_uniform_i32(qp_rsrc, safe_row * 4)"),
    ("pos = ld_uniform_pos(qp_rsrc, t0 + i)", "pos = _ld_uniform_i32(qp_rsrc, (t0 + i) * 4)"),
)


def unseam(text):
    for new, old in _UNSEAM:
        text = text.replace(new, old)
    return text
