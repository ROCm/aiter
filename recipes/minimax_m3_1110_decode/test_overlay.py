# SPDX-License-Identifier: MIT
import csv
import importlib.util
import io
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "overlay", pathlib.Path(__file__).with_name("make_overlay.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample():
    row = dict(
        mod.KEY,
        **mod.EXPECTED,
        us="31.2030",
        kernelName1="grouped_gemm1_fp4_gugu",
        kernelName2="grouped_gemm2_fp4",
    )
    f = io.StringIO()
    wr = csv.DictWriter(f, fieldnames=list(row))
    wr.writeheader()
    wr.writerow(row)
    return f.getvalue()


def test_appends_tp_row_and_preserves_wildcard():
    text = sample()
    old = list(csv.DictReader(io.StringIO(text)))
    new = list(csv.DictReader(io.StringIO(mod.make_overlay(text))))
    assert new[:-1] == old
    assert new[-1] == dict(old[0], **mod.UPDATE)
    assert new[-1]["ep_fused"] == "0"
    assert new[0]["ep_fused"] == ""


def test_duplicate_rejected():
    with pytest.raises(ValueError, match="expected one"):
        mod.make_overlay(mod.make_overlay(sample()))


@pytest.mark.parametrize("field", list(mod.EXPECTED))
def test_changed_baseline_rejected(field):
    row = next(csv.DictReader(io.StringIO(sample())))
    row[field] = "not-validated"
    f = io.StringIO()
    wr = csv.DictWriter(f, fieldnames=list(row))
    wr.writeheader()
    wr.writerow(row)
    with pytest.raises(ValueError, match="unmeasured"):
        mod.make_overlay(f.getvalue())


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="required"):
        mod.make_overlay("gfx,token\ngfx1250,16\n")


def test_wrong_commit_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="requires recipe"):
        mod.check_source(tmp_path)


def test_changed_source_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: mod.PINNED_HEAD)
    for f in mod.PINNED_FILES:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("changed")
    with pytest.raises(ValueError, match="source mismatch"):
        mod.check_source(tmp_path)
