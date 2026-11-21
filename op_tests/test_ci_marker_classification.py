from __future__ import annotations

import conftest as ci_markers


class DummyMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class DummyItem:
    def __init__(self, relpath: str, marker_names: list[str] | None = None) -> None:
        self.path = ci_markers.REPO_ROOT / relpath
        self.keywords: dict[str, object] = {}
        self.markers = [DummyMarker(name) for name in marker_names or []]
        self.added_markers: list[str] = []

    def add_marker(self, marker: DummyMarker) -> None:
        self.added_markers.append(marker.name)

    def iter_markers(self) -> list[DummyMarker]:
        return self.markers


def test_known_fast_standard_aiter_test_is_smoke() -> None:
    assert ci_markers.ci_markers_for_path(
        ci_markers.REPO_ROOT / "op_tests/test_aiter_sigmoid.py"
    ) == ("smoke",)


def test_known_long_standard_aiter_test_is_full() -> None:
    assert ci_markers.ci_markers_for_path(
        ci_markers.REPO_ROOT / "op_tests/test_topk_per_row.py"
    ) == ("full",)


def test_unknown_standard_aiter_test_defaults_to_full() -> None:
    assert ci_markers.ci_markers_for_path(
        ci_markers.REPO_ROOT / "op_tests/test_new_untracked_case.py"
    ) == ("full",)


def test_non_standard_op_tests_are_not_auto_classified() -> None:
    assert (
        ci_markers.ci_markers_for_path(
            ci_markers.REPO_ROOT / "op_tests/multigpu_tests/test_mori_smoke.py"
        )
        == ()
    )


def test_explicit_ci_marker_is_preserved() -> None:
    item = DummyItem("op_tests/test_aiter_sigmoid.py", ["perf"])

    ci_markers.pytest_collection_modifyitems(object(), [item])  # type: ignore[list-item]

    assert item.added_markers == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
