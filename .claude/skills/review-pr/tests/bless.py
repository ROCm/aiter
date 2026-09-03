#!/usr/bin/env python3
"""Regenerate tests/expected.json from the pinned corpus.

Run this ONLY when a deriver change is intended, and commit the resulting diff with the
change: that diff is the review, showing which real PRs would now be triaged differently.
"""
import json, pathlib, subprocess, sys, tarfile, tempfile

HERE = pathlib.Path(__file__).resolve().parent
TRIAGE = HERE.parent / "triage.py"
ROOT = HERE.parents[3]

with tempfile.TemporaryDirectory() as td:
    with tarfile.open(HERE / "corpus.tgz") as t:
        t.extractall(td)
    d = pathlib.Path(td)
    titles = json.loads((d / "titles.json").read_text())
    out = {}
    for f in sorted(d.glob("*.diff")):
        r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(f),
                            titles.get(f.stem, "")], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{f.name}: {r.stderr}")
        e = subprocess.run([sys.executable, str(TRIAGE), "symbols", str(f), str(ROOT)],
                           capture_output=True, text=True)
        out[f.stem] = {"rules": r.stdout,
                       "symbols": sorted(l for l in e.stdout.splitlines()
                                         if l.startswith("UNRESOLVED-IMPORT"))}
(HERE / "expected.json").write_text(
    json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
print(f"blessed {len(out)} PRs")
