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


def sample(token=16):
    key, expected, _ = mod.profile(token)
    row = dict(
        key,
        **expected,
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


@pytest.mark.parametrize("token", [16, 32])
def test_bucket_and_duplicate_guards(token):
    before = list(csv.DictReader(io.StringIO(sample(token))))
    text = mod.make_overlay(sample(token), token)
    after = list(csv.DictReader(io.StringIO(text)))
    assert after[:-1] == before
    assert after[-1] == dict(before[0], **mod.profile(token)[2])
    with pytest.raises(ValueError, match="expected one"):
        mod.make_overlay(text, token)


@pytest.mark.parametrize("field", list(mod.M32_EXPECTED))
def test_m32_changed_baseline_rejected(field):
    row = next(csv.DictReader(io.StringIO(sample(32))))
    row[field] = "not-validated"
    f = io.StringIO()
    wr = csv.DictWriter(f, fieldnames=list(row))
    wr.writeheader()
    wr.writerow(row)
    with pytest.raises(ValueError, match="unmeasured"):
        mod.make_overlay(f.getvalue(), 32)


def test_composes_without_changing_other_rows():
    text = sample() + sample(32).split("\n", 1)[1]
    before = list(csv.DictReader(io.StringIO(text)))
    combined = mod.make_overlay(mod.make_overlay(text), 32)
    after = list(csv.DictReader(io.StringIO(combined)))
    assert after[:2] == before
    assert after[2] == dict(before[0], **mod.UPDATE)
    assert after[3] == dict(before[1], **mod.M32_UPDATE)
    reverse = mod.make_overlay(mod.make_overlay(text, 32))
    rev = list(csv.DictReader(io.StringIO(reverse)))
    assert rev[:2] == before
    assert rev[2:] == list(reversed(after[2:]))


def test_unmeasured_bucket_rejected():
    with pytest.raises(ValueError, match="unmeasured token"):
        mod.make_overlay(sample(), 64)


@pytest.mark.parametrize("token", [16, 32])
def test_cli_creates_output_without_modifying_baseline(monkeypatch, tmp_path, token):
    baseline = tmp_path / "base.csv"
    out = tmp_path / "out.csv"
    text = sample(token)
    baseline.write_text(text)
    monkeypatch.setattr(mod, "check_source", lambda root: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_overlay.py",
            "--aiter-root",
            str(tmp_path),
            "--baseline",
            str(baseline),
            "--output",
            str(out),
            "--token",
            str(token),
        ],
    )
    mod.main()
    assert baseline.read_bytes() == text.encode()
    assert out.read_bytes() == mod.make_overlay(text, token).encode()
    with pytest.raises(FileExistsError):
        mod.main()
    assert out.read_bytes() == mod.make_overlay(text, token).encode()
