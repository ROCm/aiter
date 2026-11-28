from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_MAX_SECONDS = 10 * 60
CI_MARKERS = frozenset({"smoke", "full", "perf"})

STANDARD_TUNING_TESTS = frozenset(
    {
        "op_tests/tuning_tests/test_config_shape_collision.py",
        "op_tests/tuning_tests/test_csv_validation.py",
        "op_tests/tuning_tests/test_fast_tuning.py",
        "op_tests/tuning_tests/test_fhmoe_tuner.py",
        "op_tests/tuning_tests/test_mixed_mxfp_tuning.py",
    }
)


def _load_aiter_test_times() -> dict[str, int]:
    times: dict[str, int] = {}
    split_file = REPO_ROOT / ".github" / "split-test-times" / "aiter.tsv"
    if not split_file.is_file():
        return times

    for line in split_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "\t" not in line:
            continue
        path, seconds_text = line.split("\t", 1)
        try:
            times[path] = int(seconds_text)
        except ValueError:
            continue
    return times


AITER_TEST_TIMES = _load_aiter_test_times()


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _is_standard_aiter_test(relpath: str) -> bool:
    path = Path(relpath)
    if path.parent == Path("op_tests") and path.name.startswith("test_"):
        return path.suffix == ".py"
    return relpath in STANDARD_TUNING_TESTS


def _has_ci_marker(item: pytest.Item) -> bool:
    iter_markers = getattr(item, "iter_markers", None)
    if iter_markers is not None:
        return any(marker.name in CI_MARKERS for marker in iter_markers())
    return any(marker in item.keywords for marker in CI_MARKERS)


def ci_markers_for_path(path: Path) -> tuple[str, ...]:
    relpath = _repo_relative_path(path)

    if not _is_standard_aiter_test(relpath):
        return ()

    seconds = AITER_TEST_TIMES.get(relpath)
    if seconds is None:
        return ("full",)
    if seconds <= SMOKE_MAX_SECONDS:
        return ("smoke",)
    return ("full",)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    for item in items:
        if _has_ci_marker(item):
            continue
        item_path = getattr(item, "path", None)
        if item_path is None:
            item_path = Path(str(item.fspath))
        for marker in ci_markers_for_path(Path(item_path)):
            item.add_marker(getattr(pytest.mark, marker))
