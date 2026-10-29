"""setup.py's metadata-only classification, which decides whether to stage aiter_meta."""

import ast
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SETUP = _ROOT / "setup.py"


@pytest.fixture(scope="module")
def is_metadata_only(tmp_path_factory):
    # Importing setup.py would run setup(); load only the classifier.
    source = _SETUP.read_text()
    (fn,) = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_is_metadata_only"
    ]
    path = tmp_path_factory.mktemp("setup_classifier") / "classifier.py"
    path.write_text("import sys\n\n\n" + ast.get_source_segment(source, fn) + "\n")
    spec = importlib.util.spec_from_file_location("setup_classifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._is_metadata_only


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["setup.py"], True),
        (["setup.py", "egg_info"], True),
        (["setup.py", "dist_info", "--output-dir", "/tmp/x", "--keep-egg-info"], True),
        # An option value that happens to name a build command is not a command.
        (["setup.py", "dist_info", "--output-dir", "build"], True),
        (["setup.py", "dist_info", "--output-dir=build"], True),
        (["setup.py", "egg_info", "--egg-base", "build"], True),
        (["setup.py", "egg_info", "--egg-base=build"], True),
        (["setup.py", "egg_info", "--egg-base", "build", "bdist_wheel"], False),
        (["setup.py", "clean", "--all"], True),
        (["setup.py", "--version"], True),
        (["setup.py", "clean", "--all", "bdist_wheel"], False),
        # setup() prints display options and then runs no command.
        (["setup.py", "--version", "bdist_wheel"], True),
        # Multi-word display options reach the parser with dashes.
        (["setup.py", "--author-email"], True),
        (["setup.py", "--long-description"], True),
        (["setup.py", "--help-commands"], True),
        (["setup.py", "--help"], True),
        # Help on a command prints it and runs nothing.
        (["setup.py", "bdist_wheel", "--help"], True),
        (["setup.py", "clean", "--all", "bdist_wheel", "--help"], True),
        (["setup.py", "bdist_wheel", "--no-such-option"], False),
        (["setup.py", "egg_info", "develop"], False),
        (["setup.py", "bdist_wheel"], False),
        (["setup.py", "develop"], False),
    ],
)
def test_metadata_only_classifies_every_command(is_metadata_only, argv, expected):
    assert is_metadata_only(argv) is expected


def test_classifier_falls_back_without_setuptools_base_error(
    is_metadata_only, monkeypatch
):
    """setuptools < 59.0.1 has no BaseError; an unknown option still means a build."""
    import setuptools.dist  # noqa: F401  loaded before setuptools.errors is hidden

    monkeypatch.setitem(
        sys.modules, "setuptools.errors", types.ModuleType("setuptools.errors")
    )
    assert is_metadata_only(["setup.py", "bdist_wheel", "--no-such-option"]) is False
    assert is_metadata_only(["setup.py", "egg_info"]) is True


def test_metadata_only_run_keeps_an_existing_aiter_meta(tmp_path):
    """egg_info stages nothing, so it must not delete an aiter_meta/ it did not create."""
    for name in (
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "aiter_worker_limits.py",
    ):
        shutil.copy(_ROOT / name, tmp_path / name)
    (tmp_path / "aiter").mkdir()
    (tmp_path / "aiter" / "__init__.py").touch()
    (tmp_path / "aiter_meta").mkdir()
    (tmp_path / "aiter_meta" / "keep").touch()

    run = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / "aiter_meta" / "keep").exists()
