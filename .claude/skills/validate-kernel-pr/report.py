#!/usr/bin/env python3
"""Sole writer of the validation report's verdict, exit code, and stage ledger.

WHY THE VERDICT IS NOT COMPUTED IN THE SHELL
--------------------------------------------
The list of stages a PASS depends on existed in four independent copies: the schema's
``required``, a tuple that back-filled missing stages, a hand-written ten-term boolean that
decided completeness, and a set in the test file. Nothing compared them. Any two drifting apart
does not raise -- it silently changes what a PASS means, in whichever direction the drift went,
and the report keeps claiming the same word.

So the list is read from ``report_schema.json``, which is the copy the schema already enforces,
and completeness is folded over it. Adding a stage to the schema now changes the verdict; adding
one anywhere else cannot.

WHY THE SCHEMA IS VALIDATED HERE AND NOT ONLY IN TESTS
-------------------------------------------------------
The schema was never checked by the thing that produces reports -- only by tests, and only when
``jsonschema`` happened to be importable. A report that violated its own contract was therefore
publishable, and a consumer reading a field the producer had stopped writing would see a missing
key rather than a failed run. Validation happens here, before the report is copied to its
published path, and a report that does not validate cannot be PASS.

WHY EVERY WRITE COMES THROUGH HERE
----------------------------------
The report had five writers, each a separate heredoc in the shell, and one of them --
``mark_runtime_coverage`` -- was not boilerplate at all: it carried three early-exit rules
deciding when a run may claim it exercised an architecture. Policy embedded in a heredoc is
policy no test can reach, and those three rules had none. They are functions here, and tested.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import sys

SKILL_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = SKILL_DIR / "report_schema.json"

# A required stage is satisfied by "pass", except the scan, which is informational by design and
# reports "info" -- it contributes candidates, never a verdict. Any required stage absent from
# this mapping takes the default, so adding one to the schema does not silently take a status
# nobody chose for it.
SATISFYING_STATUS = {"index_width_scan": "info"}
DEFAULT_SATISFYING_STATUS = "pass"

MISSING_STAGE_NOTE = "validator internal error: stage did not record a result"
MISSING_STAGE_DETAIL = "stage result was missing; validation is inconclusive"


def required_stages(schema: dict) -> list[str]:
    """The stages a complete run must record, as the schema declares them."""
    stages = schema.get("properties", {}).get("stages", {})
    names = stages.get("required")
    if not names:
        raise SystemExit(
            "report_schema.json declares no required stages; refusing to compute a verdict "
            "against an empty contract"
        )
    return list(names)


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _write(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _schema_violation(data: dict, schema: dict) -> str | None:
    """The first way this report breaks its own contract, or None.

    An absent ``jsonschema`` is reported as a violation rather than skipped. The alternative --
    treating "could not check" as "checked and fine" -- is the exact substitution this skill
    exists to refuse everywhere else.
    """
    try:
        import jsonschema
    except ImportError:
        return "jsonschema is not installed, so the report could not be validated"
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        return f"report does not match its schema at {location}: {error.message}"
    return None


def backfill_missing_stages(data: dict, required: list[str]) -> None:
    """Record an absent stage as a skip, so it cannot vanish from the ledger.

    A stage that never ran and never appears is indistinguishable from one that passed, to
    anyone reading the report rather than the code that produced it.
    """
    for name in required:
        if name not in data["stages"]:
            data["stages"][name] = {"status": "skip", "note": MISSING_STAGE_NOTE}
            data["findings"].append(
                {"severity": "note", "stage": name, "detail": MISSING_STAGE_DETAIL}
            )


def assign(data: dict, dotted_key: str, value) -> None:
    """Set ``a.b.c`` on the report, creating the intermediate objects."""
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def runtime_credit_refusal(stats: dict, runner: str, log_size: int) -> str | None:
    """Why this run may not claim it exercised an architecture, or None if it may.

    A script that exits 0 with output has proved that a process ran, not that an architecture
    was exercised: aiter#4538's own target returns silently with exit 0 and a log line when the
    arch is unsupported or an optional package is missing. When a route WAS named and the run's
    receipt observed no call to it, there is positive evidence that no work reached the device.
    With no route named -- ``observed_work`` absent rather than zero -- nothing was observed
    either way, so this does not refuse; the basis string says so instead of implying a
    measurement.
    """
    if stats.get("executed", 0) < 1:
        return "nothing executed"
    if runner == "script":
        if log_size == 0:
            return "the script produced no output"
        if stats.get("observed_work") == 0:
            return "a route was named and the receipt observed no call to it"
    return None


def runtime_credit_basis(stats: dict, runner: str) -> str:
    """How this run knows the architecture was exercised.

    "script-exit-zero-with-output" described the process, not the work: a target that printed
    one line and returned earned the same credit as one that graded 56 cases. The basis names
    the count the stats carry and where it came from, so a reader can see whether an
    architecture was exercised or merely visited.
    """
    if runner == "pytest":
        return f"pytest-junit-executed:{stats['executed']}"
    basis = stats.get("basis", "unknown basis")
    if stats.get("failures", 0) != 0:
        return f"script-nonzero-with-output ({basis})"
    return f"script-observed-work:{stats.get('observed_work')} ({basis})"


def compute_verdict(data: dict, required: list[str]) -> str:
    """The verdict the recorded evidence supports -- the one judgement call no one narrates.

    Kept a pure function of the report so it can be exercised directly. Reached through the
    shell only, its every branch needed a full validation run to observe, which is why the
    completeness rule went years without one.
    """
    identity = data.get("runtime_identity")
    complete = isinstance(identity, dict) and bool(identity.get("module_path"))
    for name in required:
        wanted = SATISFYING_STATUS.get(name, DEFAULT_SATISFYING_STATUS)
        complete = complete and data["stages"].get(name, {}).get("status") == wanted

    severities = {finding["severity"] for finding in data["findings"]}
    if "blocker" in severities:
        return "BLOCK"
    if "should-fix" in severities:
        return "NEEDS_WORK"
    if not complete:
        return "INCONCLUSIVE"
    return "PASS"


def exit_code_for(verdict: str) -> int:
    return 0 if verdict == "PASS" else (2 if verdict == "INCONCLUSIVE" else 1)


def cmd_init(args: argparse.Namespace) -> int:
    _write(
        pathlib.Path(args.report),
        {"label": args.label, "stages": {}, "findings": []},
    )
    return 0


def _edit(path_str: str, mutate) -> int:
    path = pathlib.Path(path_str)
    data = json.loads(path.read_text())
    mutate(data)
    _write(path, data)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    value = json.loads(args.value) if args.json else args.value
    return _edit(args.report, lambda data: assign(data, args.key, value))


def cmd_stage(args: argparse.Namespace) -> int:
    return _edit(
        args.report,
        lambda data: data["stages"].__setitem__(
            args.name, {"status": args.status, "note": args.note}
        ),
    )


def cmd_finding(args: argparse.Namespace) -> int:
    return _edit(
        args.report,
        lambda data: data["findings"].append(
            {"severity": args.severity, "stage": args.stage, "detail": args.detail}
        ),
    )


def cmd_coverage(args: argparse.Namespace) -> int:
    stats = json.loads(args.stats)
    log = pathlib.Path(args.log)
    log_size = log.stat().st_size if log.exists() else 0
    if runtime_credit_refusal(stats, args.runner, log_size) is not None:
        return 0

    def mutate(data: dict) -> None:
        gpu = data["stages"].get("gpu_claim", {})
        arch = gpu.get("arch")
        if gpu.get("status") != "pass" or not arch:
            return
        data["arch_coverage"][arch] = "runtime"
        data.setdefault("arch_coverage_basis", {})[arch] = runtime_credit_basis(
            stats, args.runner
        )

    return _edit(args.report, mutate)


def cmd_finish(args: argparse.Namespace) -> int:
    schema = _load_schema()
    required = required_stages(schema)
    source = pathlib.Path(args.report)
    data = json.loads(source.read_text())

    backfill_missing_stages(data, required)
    verdict = compute_verdict(data, required)
    data["verdict"] = verdict
    data["process_exit_code"] = exit_code_for(verdict)
    data["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # Validated after the verdict is set, because the verdict is part of what the schema
    # constrains. A violation downgrades rather than raises: the run's own evidence is still
    # worth publishing, and losing it would leave a reviewer with nothing to inspect.
    violation = _schema_violation(data, schema)
    if violation is not None and verdict == "PASS":
        data["findings"].append(
            {"severity": "note", "stage": "report", "detail": violation}
        )
        data["verdict"] = verdict = "INCONCLUSIVE"
        data["process_exit_code"] = 2

    _write(source, data)
    shutil.copyfile(source, args.out)
    # The exit code is derived from THIS run's verdict, recorded here, next to the write that
    # earned it. Reading it back out of the published path made the caller's exit status depend
    # on a file any earlier run could have left behind.
    source.with_name("verdict").write_text(verdict + "\n")

    print(f"verdict={verdict}  findings={len(data['findings'])}  -> {args.out}")
    for item in data["findings"]:
        print(f"  [{item['severity']}] {item['stage']}: {item['detail'][:150]}")
    if violation is not None:
        print(f"  report validation: {violation}", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create an empty report")
    init.add_argument("report")
    init.add_argument("label")
    init.set_defaults(func=cmd_init)

    setter = sub.add_parser("set", help="set a dotted key on the report")
    setter.add_argument("report")
    setter.add_argument("key")
    setter.add_argument("value")
    setter.add_argument(
        "--json", action="store_true", help="parse VALUE as JSON rather than a string"
    )
    setter.set_defaults(func=cmd_set)

    stage = sub.add_parser("stage", help="record a stage's status and note")
    stage.add_argument("report")
    stage.add_argument("name")
    stage.add_argument("status")
    stage.add_argument("note")
    stage.set_defaults(func=cmd_stage)

    found = sub.add_parser("finding", help="append a finding")
    found.add_argument("report")
    found.add_argument("severity")
    found.add_argument("stage")
    found.add_argument("detail")
    found.set_defaults(func=cmd_finding)

    coverage = sub.add_parser(
        "coverage",
        help="credit an architecture as runtime-exercised, if the run earned it",
    )
    coverage.add_argument("report")
    coverage.add_argument("stats")
    coverage.add_argument("runner")
    coverage.add_argument("log")
    coverage.set_defaults(func=cmd_coverage)

    finish = sub.add_parser("finish", help="compute the verdict and publish the report")
    finish.add_argument("report")
    finish.add_argument("out")
    finish.set_defaults(func=cmd_finish)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
