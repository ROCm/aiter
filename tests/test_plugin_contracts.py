# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The routed op contracts, and the gate that stops them changing silently.

A routed op's public signature *is* its contract: a provider implements against
the parameter names, order, kinds and defaults it sees. ``route()`` captures that
signature at decoration, so it always agrees with itself -- which means nothing
in the interface notices when the signature changes. The contract would move
under every installed provider without an ``op_version`` bump.

This gate reads the signatures out of the source and compares them with a
committed baseline, so a change has to be deliberate. It is a static scan rather
than an import: no GPU, no toolchain, and no list of "modules that route ops"
that somebody has to remember to extend.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_BASELINE = Path(__file__).with_name("plugin_contracts.json")


def _route_spellings(tree: ast.Module) -> set[str]:
    """The names this file can call ``route()`` by.

    Resolved from the imports rather than guessed from the name, so an alias
    (``from ..plugin import route as _route``, which is what the routed ops
    use) is found, and an unrelated function whose name happens to end in
    "route" is not.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("plugin"):
                names |= {a.asname or a.name for a in node.names if a.name == "route"}
            if module.endswith("aiter") or module == "":
                names |= {
                    f"{a.asname or a.name}.route"
                    for a in node.names
                    if a.name == "plugin"
                }
        elif isinstance(node, ast.Import):
            names |= {
                f"{a.asname or a.name}.route"
                for a in node.names
                if a.name in {"aiter.plugin", "aiter"}
            }
    return names


def _is_route(decorator: ast.expr, spellings: set[str]) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    target = decorator.func
    if isinstance(target, ast.Attribute):
        name = f"{ast.unparse(target.value)}.{target.attr}"
    else:
        name = getattr(target, "id", "")
    if name not in spellings:
        return False
    # ``op`` alone: ``version`` defaults to 1 and is written only once it has
    # moved, so requiring it here would make every unversioned route invisible
    # to this gate -- unprotected in exactly the common case.
    return "op" in {keyword.arg for keyword in decorator.keywords}


def _literal(node: ast.expr | None) -> str | None:
    """A default rendered so a change to it moves the fingerprint.

    A literal is rendered by value. Anything else -- a module constant, an
    attribute, a call -- is rendered by the source that spells it, because this
    gate reads the tree and never imports it, so it has no value to render.

    That is a real gap, and it is stated rather than hidden: a routed op whose
    default is a module constant keeps the same fingerprint when the constant's
    *value* changes, which is the silent contract change this gate exists to
    catch. Closing it means either importing the module here -- trading a gate
    that runs in milliseconds without a build for one that needs the package to
    load -- or forbidding non-literal defaults on routed ops outright. Neither
    is obviously right while no routed op has such a default; what is wrong is
    leaving a reader to assume the coverage is total.
    """

    if node is None:
        return None
    try:
        return repr(ast.literal_eval(node))
    except ValueError:
        return ast.unparse(node)


def _signature(function: ast.FunctionDef) -> str:
    """The public parameter surface, rendered so any change to it shows."""

    args = function.args
    defaults: dict[str, str] = {}
    positional = args.posonlyargs + args.args
    for arg, default in zip(
        positional[len(positional) - len(args.defaults) :], args.defaults
    ):
        defaults[arg.arg] = _literal(default) or ""
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            defaults[arg.arg] = _literal(default) or ""

    def render(arg: ast.arg, kind: str) -> str:
        part = f"{arg.arg}:{kind}"
        if arg.annotation is not None:
            # A provider implements against the declared types too, so a change
            # from Tensor to list[Tensor] has to move the fingerprint.
            part += f":{ast.unparse(arg.annotation)}"
        if arg.arg in defaults:
            part += f"={defaults[arg.arg]}"
        return part

    parts = [render(a, "positional-only") for a in args.posonlyargs]
    parts += [render(a, "positional") for a in args.args]
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    parts += [render(a, "keyword-only") for a in args.kwonlyargs]
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    if function.returns is not None:
        parts.append(f"-> {ast.unparse(function.returns)}")
    return ", ".join(parts)


def routed_contracts(root: Path) -> dict[str, dict[str, object]]:
    """Every routed op in the tree, by op name."""

    found: dict[str, dict[str, object]] = {}
    for path in sorted(root.glob("aiter/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        spellings = _route_spellings(tree)
        if not spellings:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not _is_route(decorator, spellings):
                    continue
                keywords = {k.arg: k.value for k in decorator.keywords}
                op = ast.literal_eval(keywords["op"])
                version = (
                    ast.literal_eval(keywords["version"])
                    if "version" in keywords
                    else 1
                )
                signature = _signature(node)
                # Which older versions this contract still accepts is part of
                # the contract: it decides whether a Provider built for an
                # earlier shape is offered this call at all. Leaving one out is
                # a deliberate act -- rmsnorm v1 was left out because a v1
                # kernel handed a v2 call writes the caller's input -- so
                # adding one back has to be visible here.
                supersedes = (
                    sorted(ast.literal_eval(keywords["supersedes"]))
                    if "supersedes" in keywords
                    else []
                )
                found[op] = {
                    "version": version,
                    "signature": signature,
                    "fingerprint": hashlib.sha256(signature.encode()).hexdigest()[:16],
                    "source": str(path.relative_to(root)),
                    "supersedes": supersedes,
                }
    return found


def test_routed_contracts_match_the_committed_baseline():
    current = routed_contracts(_REPO)
    baseline = json.loads(_BASELINE.read_text()) if _BASELINE.exists() else {}

    for op, entry in sorted(current.items()):
        recorded = baseline.get(op)
        assert recorded is not None, (
            f"{op} is routed at {entry['source']} but is not in "
            f"{_BASELINE.name}. Add it: a routed signature is a contract, and "
            f"the baseline is where it is recorded."
        )
        assert recorded.get("supersedes", []) == entry["supersedes"], (
            f"The versions {op} still accepts changed, from "
            f"{recorded.get('supersedes', [])} to {entry['supersedes']}. That "
            f"decides whether a Provider built for an older shape is offered "
            f"this call, so it is a contract change: record it in "
            f"{_BASELINE.name} in the same change."
        )
        if recorded["fingerprint"] == entry["fingerprint"]:
            assert recorded["version"] == entry["version"], (
                f"The op_version of {op} moved but its signature did not. Either "
                f"the bump is spurious, or {_BASELINE.name} was not rewritten "
                f"with the change it describes."
            )
            continue
        # A bump must move forward, and the baseline must be rewritten in the
        # same change. Accepting any difference would let one bump license every
        # later signature change for that op -- the hole this guards.
        assert recorded["version"] < entry["version"], (
            f"The public signature of {op} changed without an op_version bump.\n"
            f"  was: v{recorded['version']}  {recorded['signature']}\n"
            f"  now: v{entry['version']}  {entry['signature']}\n"
            f"Providers implement against this signature. Bump version= on the "
            f"route, record the new contract in {_BASELINE.name}, and declare "
            f"the old version in supersedes= -- or revert the signature change."
        )
        pytest.fail(
            f"{op} was bumped to v{entry['version']} but {_BASELINE.name} still "
            f"records v{recorded['version']}. Rewrite the baseline in the same "
            f"change, so the next signature edit is checked against the new "
            f"contract rather than the old one."
        )

    for op in sorted(set(baseline) - set(current)):
        pytest.fail(
            f"{op} is in {_BASELINE.name} but is no longer routed. Removing a "
            f"routed op is a contract removal: it needs a deprecation window, "
            f"not a deletion."
        )


SAMPLE = """
from aiter.plugin import route

@route(op="demo.one", version=1)
def one(a, b, c=3, *, d=4): ...

@route(op="demo.two", version=2)
def two(x, /, y, *rest, **options): ...

def not_routed(z): ...
"""


def test_the_scanner_sees_what_a_provider_would_see(tmp_path):
    (tmp_path / "aiter").mkdir()
    (tmp_path / "aiter" / "sample.py").write_text(SAMPLE)

    found = routed_contracts(tmp_path)

    assert set(found) == {"demo.one", "demo.two"}
    assert (
        found["demo.one"]["signature"]
        == "a:positional, b:positional, c:positional=3, d:keyword-only=4"
    )
    assert (
        found["demo.two"]["signature"]
        == "x:positional-only, y:positional, *rest, **options"
    )


@pytest.mark.parametrize(
    "changed",
    [
        "def one(a, b, c=3, *, d=5): ...",  # a default moved
        "def one(a, b, c=3): ...",  # a parameter went away
        "def one(a, b, cc=3, *, d=4): ...",  # a parameter was renamed
        "def one(a, b, *, c=3, d=4): ...",  # a parameter changed kind
    ],
)
def test_every_kind_of_signature_change_moves_the_fingerprint(tmp_path, changed):
    (tmp_path / "aiter").mkdir()
    original = routed_contracts_for(tmp_path, SAMPLE)["demo.one"]["fingerprint"]
    edited = SAMPLE.replace("def one(a, b, c=3, *, d=4): ...", changed)

    assert routed_contracts_for(tmp_path, edited)["demo.one"]["fingerprint"] != original


def routed_contracts_for(tmp_path: Path, source: str) -> dict:
    (tmp_path / "aiter" / "sample.py").write_text(source)
    return routed_contracts(tmp_path)


UNVERSIONED = """
from aiter.plugin import route

@route(op="demo.plain")
def plain(a, b=2): ...
"""


def test_a_route_without_an_explicit_version_is_still_seen(tmp_path):
    """version defaults to 1, so the common route omits it.

    Requiring the keyword here would have made every such route invisible to
    this gate: unprotected in exactly the case that is recommended.
    """

    (tmp_path / "aiter").mkdir()
    (tmp_path / "aiter" / "sample.py").write_text(UNVERSIONED)

    found = routed_contracts(tmp_path)

    assert set(found) == {"demo.plain"}
    assert found["demo.plain"]["version"] == 1


SUPERSEDING = """
from aiter.plugin import route

@route(op="demo.bumped", version=2, supersedes={1: "a later aiter release"})
def bumped(a, b=2): ...
"""


def test_the_versions_a_contract_still_accepts_are_part_of_the_baseline(tmp_path):
    """Adding a version back to supersedes= must not be a silent change.

    It decides whether a Provider built for the older shape is offered the call
    at all. rmsnorm's v1 was left out of supersedes= on purpose, because a v1
    kernel handed a v2 call treats the caller's input as its output buffer; a
    later edit putting it back would undo that with no signature change to show
    for it, so the fingerprint alone would never notice.
    """

    (tmp_path / "aiter").mkdir()

    plain = routed_contracts_for(tmp_path, UNVERSIONED)["demo.plain"]
    assert plain["supersedes"] == [], "a route with no supersedes= accepts only its own"

    bumped = routed_contracts_for(tmp_path, SUPERSEDING)["demo.bumped"]
    assert bumped["supersedes"] == [1]
    assert bumped["version"] == 2


def _routed_docstrings(root: Path) -> dict[str, tuple[str, str | None]]:
    """Every routed op's source file and the docstring on its function."""

    found: dict[str, tuple[str, str | None]] = {}
    for path in sorted(root.glob("aiter/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        spellings = _route_spellings(tree)
        if not spellings:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not _is_route(decorator, spellings):
                    continue
                keywords = {k.arg: k.value for k in decorator.keywords}
                op = ast.literal_eval(keywords["op"])
                found[op] = (
                    str(path.relative_to(root)),
                    ast.get_docstring(node),
                )
    return found


def test_every_routed_op_documents_its_contract():
    """A routed op's docstring is the only contract an out-of-tree provider gets.

    `plugin_contracts.json` pins the signature, which is names and arity. It
    cannot say what a parameter MEANS, and the two ops whose names differ by
    one suffix are routinely the pair whose meanings differ -- so the prose is
    what a provider implements against, and a routed op without it ships a
    contract nobody outside this tree can read.
    """

    missing = sorted(
        op for op, (_, doc) in _routed_docstrings(_REPO).items() if not doc
    )
    assert not missing, (
        "routed ops with no docstring: "
        + ", ".join(missing)
        + " -- an out-of-tree provider has only the signature to go on"
    )


# Returning a new tensor and writing into a caller's buffer are the same
# signature to every check that exists: `plugin_contracts.json` records names
# and arity, and a provider that gets this backwards returns a plausible wrong
# answer rather than failing. So the docstring has to say which one it is.
_OUTPUT_DISCIPLINE = ("return", "in place", "in-place", "writes", "written")


def test_every_routed_op_says_whether_it_returns_or_writes():
    """The one fact a provider cannot infer from the signature.

    `gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out, bias=None, splitK=0)` and
    `gemm_a8w8(XQ, WQ, x_scale, w_scale, bias=None, dtype=bf16, splitK=None)`
    are the shape of the mistake: seven parameters each, and the fifth is an
    output buffer in one and a bias in the other. Nothing about the names
    catches that, and neither does arity.
    """

    silent = sorted(
        op
        for op, (_, doc) in _routed_docstrings(_REPO).items()
        if doc and not any(word in doc.lower() for word in _OUTPUT_DISCIPLINE)
    )
    assert (
        not silent
    ), "routed ops whose docstring never says what they return or write: " + ", ".join(
        silent
    )


def test_the_documentation_gate_is_not_vacuous():
    """Both gates above pass trivially if nothing is routed. Pin that it is."""

    documented = _routed_docstrings(_REPO)
    assert documented, (
        "the scanner found no routed ops; the gates above would pass without "
        "checking anything"
    )
    assert set(documented) == set(json.loads(_BASELINE.read_text())), (
        "the docstring scan and the contract baseline disagree about which "
        "ops are routed"
    )
