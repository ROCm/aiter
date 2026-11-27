# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-translation-unit build failures for JIT modules.

A tuner module (``module_*_tune``) is one shared library that holds every
candidate kernel as its own generated translation unit. A single candidate that
cannot compile (for example a CK Tile tile whose LDS need exceeds the arch
limit) used to abort ninja and, with it, the whole module: every candidate in
the run was reported dead and the real error sat in a multi-thousand-line log.

This module is the pure-Python part of the fix. It needs no GPU, torch or
compiler:

* :func:`parse_ninja_failures` turns ninja's output into one
  :class:`CompileFailure` per failed translation unit, with its compiler errors.
* :class:`JitCompileError` carries those failures to the caller.
* :func:`isolate_failed_instances` rewrites the failed candidate's translation
  unit so that it still defines the symbol the dispatch table references, but
  throws a ``std::runtime_error`` naming the candidate and the compiler error.
  The module then links, the other candidates are measured, and only the broken
  candidate fails when it is called.
"""

import os
import re
from dataclasses import dataclass, field

# Substring of the stub's error text. Tuner infrastructure keys on it to label
# the candidate and to print the end-of-run summary.
COMPILE_FAILED_MARKER = "failed to compile:"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FAILED_LINE = re.compile(r"^FAILED:(?: \[code=-?\d+\])? (?P<target>\S.*?)\s*$")
# Progress lines ("[12/40] <command>") and ninja's own messages end a block.
_BLOCK_END = re.compile(r"^(?:\[\d+/\d+\] |ninja: )")
_ERROR_LINE = re.compile(r"\berror\s*:", re.IGNORECASE)
_WARNING_LINE = re.compile(r"\bwarning\s*:", re.IGNORECASE)
_IMPL_INCLUDE = re.compile(r'#\s*include\s+"impl/(?P<name>[^"/]+)\.cuh"')
# An explicit instantiation *definition*: ``template RET NAME<ARGS>(PARAMS);``.
# ``template <`` (a template definition) is excluded by the lookahead.
_EXPLICIT_INSTANTIATION = re.compile(
    r"^template\s+(?!<)(?P<decl>[^;{}]*?\))\s*;", re.MULTILINE | re.DOTALL
)

_MAX_ERRORS = 5
_MAX_ERROR_CHARS = 300
_STUB_ERROR_CHARS = 160


@dataclass
class CompileFailure:
    """One translation unit that ninja could not build."""

    target: str
    errors: list[str] = field(default_factory=list)
    log: str = ""

    @property
    def stem(self) -> str:
        name = os.path.basename(self.target)
        for suffix in (".cuda.o", ".o", ".obj"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return os.path.splitext(name)[0]


class JitCompileError(RuntimeError):
    """A JIT build failed; ``failures`` lists each translation unit that did."""

    def __init__(self, message, failures=()):
        super().__init__(message)
        self.failures = list(failures)


def _short(line: str, limit: int = _MAX_ERROR_CHARS) -> str:
    line = " ".join(_ANSI.sub("", line).split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _extract_errors(lines):
    errors = []
    for line in lines:
        if _ERROR_LINE.search(line) and not line.lstrip().startswith("FAILED:"):
            text = _short(line)
            # Device passes repeat one diagnostic once per --offload-arch.
            if text not in errors:
                errors.append(text)
    if errors:
        return errors[:_MAX_ERRORS]
    # A compiler killed by a signal or the OOM killer prints no "error:" line.
    tail = [
        _short(line)
        for line in lines
        if line.strip() and not _WARNING_LINE.search(line)
    ]
    return tail[-3:]


def parse_ninja_failures(output) -> list[CompileFailure]:
    """Return one :class:`CompileFailure` per ``FAILED:`` block of a ninja log."""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    failures = []
    block = None
    for raw in output.splitlines():
        line = _ANSI.sub("", raw)
        match = _FAILED_LINE.match(line)
        if match:
            block = (match.group("target"), [])
            failures.append(block)
        elif block is not None and _BLOCK_END.match(line):
            block = None
        elif block is not None:
            block[1].append(raw)
    result = []
    for target, body in failures:
        # body[0] is the echoed compile command (ninja -v), not compiler output.
        output_lines = body[1:]
        result.append(
            CompileFailure(
                target, _extract_errors(output_lines), "\n".join(output_lines)
            )
        )
    return result


def format_failure_summary(failures, names=None) -> str:
    """Short block for the *end* of an error message or a log record.

    ``names`` optionally maps ``failure.target`` to a friendlier name.
    """
    if not failures:
        return ""
    lines = [f"[aiter] {len(failures)} translation unit(s) failed to compile:"]
    for failure in failures:
        name = (names or {}).get(failure.target, failure.stem)
        lines.append(f"  - {name}")
        for error in failure.errors or ["(no compiler error captured)"]:
            lines.append(f"      {error}")
    return "\n".join(lines)


def source_for_target(target, sources):
    """The source path whose object file is ``target``, or ``None``."""
    stem = CompileFailure(target).stem
    matches = [s for s in sources if os.path.splitext(os.path.basename(s))[0] == stem]
    return matches[0] if len(matches) == 1 else None


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def instance_name(source_text):
    """Candidate name of a generated instance TU, or ``None`` if it is not one."""
    match = _IMPL_INCLUDE.search(source_text)
    return match.group("name") if match else None


def _c_string_literal(text: str) -> str:
    text = text.encode("ascii", "replace").decode("ascii")
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + re.sub(r"\s+", " ", text).strip() + '"'


def stub_message(name, errors) -> str:
    error = errors[0] if errors else "see the JIT build log"
    if len(error) > _STUB_ERROR_CHARS:
        error = error[: _STUB_ERROR_CHARS - 3] + "..."
    return f"[aiter] tune candidate '{name}' {COMPILE_FAILED_MARKER} {error}"


def _stubbed_text(source_text, message):
    """``source_text`` with each explicit instantiation turned into a throwing
    explicit specialization, or ``None`` if it has no such statement.

    The ``#include "impl/NAME.cuh"`` stays, so the primary template and every
    type are still in scope; the specialization keeps the failing body from
    being instantiated, and defines the same symbol the dispatch table names.
    """
    literal = _c_string_literal(message)
    text, count = _EXPLICIT_INSTANTIATION.subn(
        lambda m: f"template <> {m.group('decl')} {{ throw std::runtime_error({literal}); }}",
        source_text,
    )
    if not count:
        return None
    return "#include <stdexcept>\n#include <string>\n" + text


@dataclass
class IsolatedInstance:
    name: str
    source: str
    errors: list[str]


def isolate_failed_instances(failures, sources):
    """Rewrite the failed candidates' translation units into throwing stubs.

    Returns the :class:`IsolatedInstance` list, or ``None`` when the failure
    cannot be isolated, in which case no file is touched and the caller must
    fail the build:

    * a failed unit is not a generated instance (a pybind or dispatcher unit),
    * its source is not in the form this rewrite understands, or
    * every instance failed, which points at the toolchain, not a candidate.
    """
    if not failures:
        return None
    planned = []
    for failure in failures:
        source = source_for_target(failure.target, sources)
        if source is None or not os.path.isfile(source):
            return None
        text = _read(source)
        name = instance_name(text)
        stubbed = (
            _stubbed_text(text, stub_message(name, failure.errors)) if name else None
        )
        if stubbed is None:
            return None
        planned.append((IsolatedInstance(name, source, failure.errors), stubbed))

    failed_sources = {item.source for item, _ in planned}
    instance_sources = [
        s for s in sources if os.path.isfile(s) and instance_name(_read(s))
    ]
    if all(s in failed_sources for s in instance_sources):
        return None
    for item, stubbed in planned:
        with open(item.source, "w", encoding="utf-8") as handle:
            handle.write(stubbed)
    return [item for item, _ in planned]


def summarize_compile_failures(details) -> str:
    """End-of-run summary from candidate failure details (see mp_tuner)."""
    seen = []
    for detail in details:
        if COMPILE_FAILED_MARKER in detail and detail not in seen:
            seen.append(detail)
    if not seen:
        return ""
    lines = [f"{len(seen)} candidate(s) failed to compile and were skipped:"]
    lines += [f"  - {detail}" for detail in seen]
    return "\n".join(lines)
