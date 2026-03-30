"""Integration tests that exercise multiple tuning-agent modules together."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from .config import load_config
from .machine_pool import MachinePool
from .notifications import Notifier, NotificationLevel
from .artifacts import ArtifactManager
from .types import ShapeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_manager(tmp_path: Path, kernel_name: str = "a8w8") -> ArtifactManager:
    executor = MagicMock()
    return ArtifactManager(
        executor=executor,
        kernel_name=kernel_name,
        run_id="integration_run",
        local_base_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Test 1 – config → pool flow
# ---------------------------------------------------------------------------


class TestConfigToPoolFlow:
    """Load a YAML config, build a MachinePool from it, and drive the pool."""

    def test_config_to_pool_flow(self) -> None:
        config = load_config(str(_FIXTURES_DIR / "valid_config.yaml"))

        pool = MachinePool(config.machines)

        # Two machines are declared in valid_config.yaml.
        assert pool.available_count == 2

        # Allocate for the "a8w8" kernel; the pool prefers the machine with
        # more GPUs (gpu-node-01 has 4 GPUs, gpu-node-02 has 1 GPU).
        machine = pool.allocate("a8w8")

        assert machine.gpu_count == 4  # must be the richer machine
        assert pool.available_count == 1  # one machine is now allocated

        # Release and confirm it is returned to the idle pool.
        pool.release(machine.host)
        assert pool.available_count == 2


# ---------------------------------------------------------------------------
# Test 2 – results round-trip through ArtifactManager
# ---------------------------------------------------------------------------


class TestResultsRoundtrip:
    """Save ShapeResults to disk and load them back, verifying every field."""

    def test_results_roundtrip(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.setup_local()

        original = ShapeResult(m=512, n=1024, k=256, main_ns=4200.0, reduce_ns=300.0)
        mgr.save_results("benchmark", [original])

        loaded = mgr.load_results("benchmark")
        assert len(loaded) == 1

        result = loaded[0]
        assert result.m == original.m
        assert result.n == original.n
        assert result.k == original.k
        assert result.main_ns == pytest.approx(original.main_ns)
        assert result.reduce_ns == pytest.approx(original.reduce_ns)
        assert result.total_ns == pytest.approx(original.total_ns)


# ---------------------------------------------------------------------------
# Test 3 – notification flow
# ---------------------------------------------------------------------------


class TestNotificationFlow:
    """Send notifications and exercise the auto-approval path."""

    def test_notification_flow(self) -> None:
        notifier = Notifier(auto_approve=True)

        notifier.notify(NotificationLevel.INFO, "Phase started", "Beginning phase 1")
        notifier.notify(NotificationLevel.WARNING, "Slow shape", "m=16 is slow")

        assert len(notifier.history) == 2

        approved = notifier.request_approval("Proceed with phase 2?")
        assert approved is True


# ---------------------------------------------------------------------------
# Test 4 – checkpoint lifecycle
# ---------------------------------------------------------------------------


class TestCheckpointLifecycle:
    """Verify phase completion markers are written and read back correctly."""

    def test_checkpoint_lifecycle(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.setup_local()

        # Phase 2 should not be marked complete at the start.
        assert mgr.is_phase_complete(2) is False

        summary = {"shapes_tuned": 42, "best_config": "x128_y4"}
        mgr.mark_phase_complete(2, summary)

        assert mgr.is_phase_complete(2) is True

        stored = mgr.get_phase_summary(2)
        assert stored is not None
        assert stored["shapes_tuned"] == summary["shapes_tuned"]
        assert stored["best_config"] == summary["best_config"]
        # The manager always adds a timestamp alongside the caller's fields.
        assert "timestamp" in stored
