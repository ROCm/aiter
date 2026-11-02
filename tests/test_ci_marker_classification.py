from __future__ import annotations

import conftest


class DummyMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class DummyItem:
    def __init__(self, relpath: str, marker_names: list[str] | None = None) -> None:
        self.path = conftest.REPO_ROOT / relpath
        self.keywords: dict[str, object] = {}
        self.markers = [DummyMarker(name) for name in marker_names or []]
        self.added_markers: list[str] = []

    def add_marker(self, marker: object) -> None:
        self.added_markers.append(getattr(marker, "name"))

    def iter_markers(self) -> list[DummyMarker]:
        return self.markers


def test_known_fast_aiter_test_is_smoke() -> None:
    assert conftest.ci_markers_for_path(
        conftest.REPO_ROOT / "op_tests/test_aiter_sigmoid.py"
    ) == ("smoke",)


def test_known_long_aiter_test_is_full_and_slow() -> None:
    assert conftest.ci_markers_for_path(
        conftest.REPO_ROOT / "op_tests/test_topk_per_row.py"
    ) == ("full", "slow")


def test_unknown_aiter_test_defaults_to_full() -> None:
    assert conftest.ci_markers_for_path(
        conftest.REPO_ROOT / "op_tests/test_new_untracked_case.py"
    ) == ("full",)


def test_focused_regression_test_is_smoke() -> None:
    assert conftest.ci_markers_for_path(
        conftest.REPO_ROOT / "tests/test_import_without_gpu.py"
    ) == ("smoke",)


def test_explicit_ci_marker_is_preserved() -> None:
    item = DummyItem("op_tests/test_aiter_sigmoid.py", ["perf"])

    conftest.pytest_collection_modifyitems(object(), [item])  # type: ignore[list-item]

    assert item.added_markers == []
