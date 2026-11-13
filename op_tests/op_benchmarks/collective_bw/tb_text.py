# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run the stock TransferBench binary and parse its text output.

TransferBench is an optional binary found at run time
(``$TRANSFERBENCH``, then ``PATH``, then the ROCmValidationSuite paths), and
callers treat its absence as "no roofline" rather than as an error.

The parser reads the human-readable table, because ``OUTPUT_TO_CSV`` prints
a stale header and drops the ``Test N:`` delimiters. It is checked against
captured output in ``testdata/``. A version it has not seen is accepted with a
warning; the parser then fails loudly if the test count does not match.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("aiter")

ENV_BIN = "TRANSFERBENCH"
_RVS_PATHS = ("/opt/rocm/bin/TransferBench", "/opt/rocm/libexec/rvs/TransferBench")
# Versions whose output the parser has been checked against (testdata/).
KNOWN_VERSIONS = ("1.71",)

# Settings every run uses, on top of each test's own environment.
BASE_ENV = {
    "USE_HIP_EVENTS": "1",
    # Validation walks every element on the host, which costs far more than
    # the transfers.
    "ALWAYS_VALIDATE": "-1",
    "OUTPUT_TO_CSV": "0",
    "HIDE_ENV": "1",
}

_SEP = r"\s*[│|]\s*"
_VERSION_RE = re.compile(r"TransferBench v(\d+\.\d+)(\.\d+)?")
_TEST_RE = re.compile(r"^Test\s+(\d+):")
_EXEC_RE = re.compile(
    rf"Executor:\s+(?:Rank\s+\d+\s+)?(\S+)\s+(\d+){_SEP}([\d.]+)\s*GB/s"
    rf"{_SEP}([\d.]+)\s*ms{_SEP}(\d+)\s*bytes"
)
_ERR_RE = re.compile(r"^\s*\[ERROR\]\s*(.*)", re.MULTILINE)


@dataclass
class TestResult:
    """One TransferBench test: milliseconds per executor, per sub-iteration."""

    num: int
    exec_ms: dict[str, float] = field(default_factory=dict)

    @property
    def slowest_ms(self) -> float:
        """The executor a collective would wait on."""
        if not self.exec_ms:
            return math.nan
        return max(self.exec_ms.values())


def find_binary(explicit: str | None = None) -> str | None:
    """The TransferBench executable, or None if there is none."""
    for cand in (explicit, os.environ.get(ENV_BIN)):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
            logger.warning("TransferBench: %r is not an executable file", cand)
            return None
    found = shutil.which("TransferBench")
    if found:
        return found
    return next((p for p in _RVS_PATHS if os.access(p, os.X_OK)), None)


def _duration_ms(gbps: float, ms: float, nbytes: int) -> float:
    """The row's duration."""
    if gbps > 0.0 and nbytes > 0:
        return nbytes / 1e6 / gbps
    return ms if ms > 0.0 else math.nan


def version(text: str) -> str | None:
    m = _VERSION_RE.search(text)
    return m.group(1) + (m.group(2) or "") if m else None


def parse(text: str) -> list[TestResult]:
    """Every test in a TransferBench run, in output order."""
    tests: list[TestResult] = []
    for line in text.splitlines():
        m = _TEST_RE.match(line)
        if m:
            tests.append(TestResult(num=int(m.group(1))))
            continue
        m = _EXEC_RE.search(line)
        if m and tests:
            name = f"{m.group(1)} {int(m.group(2))}"
            tests[-1].exec_ms[name] = _duration_ms(
                float(m.group(3)), float(m.group(4)), int(m.group(5))
            )
    return tests


def run(
    binary: str,
    lines: list[str],
    env: dict[str, str],
    *,
    iters: int,
    warmup: int,
    timeout: float = 900.0,
) -> tuple[list[TestResult], str]:
    """Run *lines* as one config file. Returns the tests and the version.

    Raises RuntimeError if TransferBench fails or the number of parsed tests
    does not match the number of lines.
    """
    full_env = dict(os.environ)
    full_env.update(BASE_ENV)
    full_env.update(env)
    full_env.update({"NUM_ITERATIONS": str(iters), "NUM_WARMUPS": str(warmup)})
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cfg", prefix="collective_bw_", delete=False
    ) as fh:
        fh.write("\n".join(lines) + "\n")
        cfg = fh.name
    try:
        proc = subprocess.run(
            [binary, cfg],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        os.unlink(cfg)
    out = proc.stdout + proc.stderr
    errors = _ERR_RE.findall(out)
    if proc.returncode != 0 or errors:
        detail = "; ".join(errors) or out[-400:].strip()
        raise RuntimeError(f"TransferBench exited {proc.returncode}: {detail}")
    ver = version(out)
    if ver is None or not ver.startswith(KNOWN_VERSIONS):
        logger.warning(
            "TransferBench %s: output format not checked against this version "
            "(checked: %s)",
            ver,
            ", ".join(KNOWN_VERSIONS),
        )
    tests = parse(out)
    if len(tests) != len(lines) or any(not t.exec_ms for t in tests):
        raise RuntimeError(
            f"TransferBench {ver}: parsed {len(tests)} test(s) for {len(lines)} "
            "config line(s); the output format may have changed"
        )
    return tests, ver
