from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent
SMOKE_MAX_SECONDS = 10 * 60
CI_MARKERS = frozenset({"smoke", "full", "perf", "nightly", "slow"})

FOCUSED_SMOKE_TESTS = frozenset(
    {
        "tests/test_aot_drivers.py",
        "tests/test_chip_info_xcds.py",
        "tests/test_ci_marker_classification.py",
        "tests/test_custom_all_reduce_layout.py",
        "tests/test_custom_ar_fused_rms_row_options.py",
        "tests/test_custom_ar_residual_guard.py",
        "tests/test_group_coordinator_all_gather.py",
        "tests/test_import_without_gpu.py",
        "tests/test_mega_moe_gfx1250_recipe.py",
        "tests/test_worker_entrypoints.py",
        "tests/test_worker_limits.py",
    }
)


def _load_split_test_times() -> dict[str, int]:
    times: dict[str, int] = {}
    for split_file in (
        REPO_ROOT / ".github" / "split-test-times" / "aiter.tsv",
        REPO_ROOT / ".github" / "split-test-times" / "triton.tsv",
    ):
        if not split_file.is_file():
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            path, seconds_text = line.split("\t", 1)
            try:
                times[path] = int(seconds_text)
            except ValueError:
                continue
    return times


SPLIT_TEST_TIMES = _load_split_test_times()


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _has_ci_marker(item: pytest.Item) -> bool:
    iter_markers = getattr(item, "iter_markers", None)
    if iter_markers is not None:
        return any(marker.name in CI_MARKERS for marker in iter_markers())
    return any(marker in item.keywords for marker in CI_MARKERS)


def ci_markers_for_path(path: Path) -> tuple[str, ...]:
    relpath = _repo_relative_path(path)

    if relpath in FOCUSED_SMOKE_TESTS:
        return ("smoke",)

    if relpath.startswith("tests/"):
        return ("full",)

    if relpath.startswith("op_tests/op_benchmarks/"):
        return ("perf",)

    if relpath.startswith("op_tests/multigpu_tests/"):
        return ("full", "slow")

    if relpath.startswith("op_tests/"):
        seconds = SPLIT_TEST_TIMES.get(relpath)
        if seconds is None:
            return ("full",)
        if seconds <= SMOKE_MAX_SECONDS:
            return ("smoke",)
        return ("full", "slow")

    return ()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        if _has_ci_marker(item):
            continue
        item_path = getattr(item, "path", None)
        if item_path is None:
            item_path = Path(str(item.fspath))
        for marker in ci_markers_for_path(Path(item_path)):
            item.add_marker(getattr(pytest.mark, marker))
