"""Tests for the MachinePool class.

Run with:
    python -m pytest aiter/ops/triton/utils/_triton/tunning/tuning_agent/test_machine_pool.py -v
"""

from __future__ import annotations

import subprocess
import threading
import unittest
from unittest.mock import MagicMock, call, patch

from .machine_pool import MachinePool, NoMachineAvailable
from .types import MachineInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_machine(host: str, gpu_count: int = 1) -> MachineInfo:
    return MachineInfo(
        host=host,
        user="testuser",
        ssh_key="/tmp/test_key",
        gpus=list(range(gpu_count)),
    )


def _make_pool(*specs: tuple) -> MachinePool:
    """Build a pool from (host, gpu_count) tuples."""
    machines = [_make_machine(host, gpus) for host, gpus in specs]
    return MachinePool(machines)


# ---------------------------------------------------------------------------
# Allocation tests
# ---------------------------------------------------------------------------

class TestAllocate(unittest.TestCase):
    """Tests for MachinePool.allocate()."""

    def test_allocate_prefers_more_gpus(self):
        """allocate() should return the machine with the most GPUs."""
        pool = _make_pool(("host-a", 2), ("host-b", 4), ("host-c", 1))
        machine = pool.allocate("gemm_kernel")
        self.assertEqual(machine.host, "host-b")
        self.assertEqual(machine.gpu_count, 4)

    def test_allocate_second_call_gives_different_machine(self):
        """Two consecutive allocations must yield distinct machines."""
        pool = _make_pool(("host-a", 4), ("host-b", 4))
        first = pool.allocate("kernel1")
        second = pool.allocate("kernel2")
        self.assertNotEqual(first.host, second.host)

    def test_allocate_all_busy_raises(self):
        """NoMachineAvailable is raised when every machine is already allocated."""
        pool = _make_pool(("host-a", 2))
        pool.allocate("kernel1")
        with self.assertRaises(NoMachineAvailable):
            pool.allocate("kernel2")

    def test_allocate_single_machine_pool(self):
        """A pool with one machine can be allocated exactly once."""
        pool = _make_pool(("solo", 1))
        m = pool.allocate("k")
        self.assertEqual(m.host, "solo")

    def test_allocate_tiebreak_by_host_name(self):
        """On equal GPU count the lexicographically largest host name wins."""
        # max() with key=(gpu_count, host) means the highest host name wins.
        pool = _make_pool(("b-host", 2), ("a-host", 2))
        m = pool.allocate("k")
        self.assertEqual(m.host, "b-host")

    def test_allocate_records_kernel_name(self):
        """The allocated kernel name must appear in status()."""
        pool = _make_pool(("host-x", 2))
        pool.allocate("my_kernel")
        statuses = {s["host"]: s for s in pool.status()}
        self.assertEqual(statuses["host-x"]["kernel"], "my_kernel")
        self.assertEqual(statuses["host-x"]["state"], "allocated")


# ---------------------------------------------------------------------------
# Release tests
# ---------------------------------------------------------------------------

class TestRelease(unittest.TestCase):
    """Tests for MachinePool.release()."""

    def test_release_makes_machine_available_again(self):
        """After release() a machine should be allocatable again."""
        pool = _make_pool(("host-a", 2))
        pool.allocate("kernel1")
        with self.assertRaises(NoMachineAvailable):
            pool.allocate("kernel2")
        pool.release("host-a")
        m = pool.allocate("kernel3")
        self.assertEqual(m.host, "host-a")

    def test_release_unknown_host_is_noop(self):
        """release() on a host that was never allocated should not raise."""
        pool = _make_pool(("host-a", 1))
        pool.release("not-a-real-host")  # Should not raise

    def test_release_already_idle_is_noop(self):
        """release() on an idle host should leave it idle."""
        pool = _make_pool(("host-a", 1))
        pool.release("host-a")  # Not allocated — should be a no-op.
        self.assertEqual(pool.available_count, 1)


# ---------------------------------------------------------------------------
# mark_dead tests
# ---------------------------------------------------------------------------

class TestMarkDead(unittest.TestCase):
    """Tests for MachinePool.mark_dead()."""

    def test_mark_dead_removes_from_available(self):
        """After mark_dead() the machine must not be returned by allocate()."""
        pool = _make_pool(("dead-host", 4), ("live-host", 1))
        pool.mark_dead("dead-host")
        m = pool.allocate("k")
        self.assertEqual(m.host, "live-host")

    def test_mark_dead_removes_from_allocated(self):
        """mark_dead() on an allocated machine should also free the allocation."""
        pool = _make_pool(("host-a", 2))
        pool.allocate("k")
        pool.mark_dead("host-a")
        statuses = {s["host"]: s for s in pool.status()}
        self.assertEqual(statuses["host-a"]["state"], "dead")

    def test_mark_dead_shows_in_status(self):
        """Dead machines must have state='dead' in status()."""
        pool = _make_pool(("host-a", 1))
        pool.mark_dead("host-a")
        statuses = {s["host"]: s for s in pool.status()}
        self.assertEqual(statuses["host-a"]["state"], "dead")
        self.assertIsNone(statuses["host-a"]["kernel"])

    def test_mark_dead_all_machines_raises_on_allocate(self):
        """If all machines are dead allocate() must raise NoMachineAvailable."""
        pool = _make_pool(("host-a", 1), ("host-b", 2))
        pool.mark_dead("host-a")
        pool.mark_dead("host-b")
        with self.assertRaises(NoMachineAvailable):
            pool.allocate("k")

    def test_mark_dead_unknown_host_is_noop(self):
        """mark_dead() for an unknown host should not raise."""
        pool = _make_pool(("host-a", 1))
        pool.mark_dead("completely-unknown")  # Should not raise.
        self.assertEqual(pool.available_count, 1)


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------

class TestStatus(unittest.TestCase):
    """Tests for MachinePool.status()."""

    def test_status_initial_all_idle(self):
        """All machines should start as idle."""
        pool = _make_pool(("a", 2), ("b", 4))
        for entry in pool.status():
            self.assertEqual(entry["state"], "idle")
            self.assertIsNone(entry["kernel"])

    def test_status_reflects_allocation(self):
        """status() must reflect allocations correctly."""
        pool = _make_pool(("host-a", 2), ("host-b", 4))
        pool.allocate("gemm")
        statuses = {s["host"]: s for s in pool.status()}
        # host-b has more GPUs so should be allocated.
        self.assertEqual(statuses["host-b"]["state"], "allocated")
        self.assertEqual(statuses["host-b"]["kernel"], "gemm")
        self.assertEqual(statuses["host-a"]["state"], "idle")

    def test_status_gpu_info_correct(self):
        """status() entries must carry the correct gpus list and gpu_count."""
        pool = _make_pool(("host-a", 3))
        statuses = pool.status()
        self.assertEqual(len(statuses), 1)
        entry = statuses[0]
        self.assertEqual(entry["gpus"], [0, 1, 2])
        self.assertEqual(entry["gpu_count"], 3)

    def test_status_all_states_present(self):
        """status() should show idle, allocated, and dead states simultaneously."""
        pool = _make_pool(("idle-host", 1), ("alloc-host", 2), ("dead-host", 1))
        pool.allocate("k")   # Allocates alloc-host (most GPUs)
        pool.mark_dead("dead-host")
        statuses = {s["host"]: s for s in pool.status()}
        self.assertEqual(statuses["idle-host"]["state"], "idle")
        self.assertEqual(statuses["alloc-host"]["state"], "allocated")
        self.assertEqual(statuses["dead-host"]["state"], "dead")


# ---------------------------------------------------------------------------
# available_count tests
# ---------------------------------------------------------------------------

class TestAvailableCount(unittest.TestCase):
    """Tests for MachinePool.available_count property."""

    def test_available_count_initial(self):
        """All machines are available at start."""
        pool = _make_pool(("a", 1), ("b", 2), ("c", 1))
        self.assertEqual(pool.available_count, 3)

    def test_available_count_after_allocation(self):
        """Each allocation should decrease available_count by 1."""
        pool = _make_pool(("a", 1), ("b", 2))
        self.assertEqual(pool.available_count, 2)
        pool.allocate("k1")
        self.assertEqual(pool.available_count, 1)
        pool.allocate("k2")
        self.assertEqual(pool.available_count, 0)

    def test_available_count_after_release(self):
        """Releasing a machine should increase available_count by 1."""
        pool = _make_pool(("a", 1))
        pool.allocate("k")
        self.assertEqual(pool.available_count, 0)
        pool.release("a")
        self.assertEqual(pool.available_count, 1)

    def test_available_count_dead_machines_excluded(self):
        """Dead machines must not be counted as available."""
        pool = _make_pool(("a", 1), ("b", 2))
        pool.mark_dead("a")
        self.assertEqual(pool.available_count, 1)
        pool.mark_dead("b")
        self.assertEqual(pool.available_count, 0)

    def test_available_count_empty_pool(self):
        """An empty pool should have available_count of 0."""
        pool = MachinePool([])
        self.assertEqual(pool.available_count, 0)


# ---------------------------------------------------------------------------
# get_machine tests
# ---------------------------------------------------------------------------

class TestGetMachine(unittest.TestCase):
    """Tests for MachinePool.get_machine()."""

    def test_get_machine_known_host(self):
        """get_machine() should return the MachineInfo for a known host."""
        pool = _make_pool(("host-a", 2))
        m = pool.get_machine("host-a")
        self.assertIsNotNone(m)
        self.assertEqual(m.host, "host-a")

    def test_get_machine_unknown_host_returns_none(self):
        """get_machine() should return None for an unknown host."""
        pool = _make_pool(("host-a", 2))
        self.assertIsNone(pool.get_machine("no-such-host"))


# ---------------------------------------------------------------------------
# validate_connectivity tests
# ---------------------------------------------------------------------------

class TestValidateConnectivity(unittest.TestCase):
    """Tests for MachinePool.validate_connectivity()."""

    def _make_completed_process(self, returncode: int, stdout: str = "") -> subprocess.CompletedProcess:
        cp = subprocess.CompletedProcess(args=[], returncode=returncode)
        cp.stdout = stdout
        cp.stderr = ""
        return cp

    @patch("tuning_agent.machine_pool.RemoteExecutor")
    def test_reachable_with_rocm_smi(self, MockExecutor):
        """A machine that is reachable and has rocm-smi should remain live."""
        mock_exec = MagicMock()
        mock_exec.check_ssh_connectivity.return_value = True
        mock_exec.ssh_run.return_value = self._make_completed_process(0, "/usr/bin/rocm-smi")
        MockExecutor.return_value = mock_exec

        pool = _make_pool(("host-a", 2))
        results = pool.validate_connectivity()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["reachable"])
        self.assertTrue(results[0]["rocm_smi_ok"])
        self.assertIsNone(results[0]["error"])
        # Machine should NOT be dead.
        statuses = {s["host"]: s for s in pool.status()}
        self.assertNotEqual(statuses["host-a"]["state"], "dead")

    @patch("tuning_agent.machine_pool.RemoteExecutor")
    def test_unreachable_machine_marked_dead(self, MockExecutor):
        """An unreachable machine should be marked dead."""
        mock_exec = MagicMock()
        mock_exec.check_ssh_connectivity.return_value = False
        MockExecutor.return_value = mock_exec

        pool = _make_pool(("dead-host", 1))
        results = pool.validate_connectivity()

        self.assertFalse(results[0]["reachable"])
        statuses = {s["host"]: s for s in pool.status()}
        self.assertEqual(statuses["dead-host"]["state"], "dead")

    @patch("tuning_agent.machine_pool.RemoteExecutor")
    def test_reachable_but_no_rocm_smi_marked_dead(self, MockExecutor):
        """Reachable but missing rocm-smi should be marked dead."""
        mock_exec = MagicMock()
        mock_exec.check_ssh_connectivity.return_value = True
        mock_exec.ssh_run.return_value = self._make_completed_process(1, "")
        MockExecutor.return_value = mock_exec

        pool = _make_pool(("host-b", 2))
        results = pool.validate_connectivity()

        self.assertTrue(results[0]["reachable"])
        self.assertFalse(results[0]["rocm_smi_ok"])
        self.assertIsNotNone(results[0]["error"])
        statuses = {s["host"]: s for s in pool.status()}
        self.assertEqual(statuses["host-b"]["state"], "dead")

    @patch("tuning_agent.machine_pool.RemoteExecutor")
    def test_validate_multiple_machines_independently(self, MockExecutor):
        """Validate checks each machine independently."""
        results_map = {
            "live-host": (True, True),
            "dead-host": (False, False),
        }

        def executor_factory(machine):
            mock = MagicMock()
            reachable, rocm_ok = results_map[machine.host]
            mock.check_ssh_connectivity.return_value = reachable
            if reachable:
                rc = 0 if rocm_ok else 1
                cp = subprocess.CompletedProcess(args=[], returncode=rc)
                cp.stdout = "/usr/bin/rocm-smi" if rocm_ok else ""
                cp.stderr = ""
                mock.ssh_run.return_value = cp
            return mock

        MockExecutor.side_effect = executor_factory

        pool = _make_pool(("live-host", 4), ("dead-host", 2))
        results = pool.validate_connectivity()
        by_host = {r["host"]: r for r in results}

        self.assertTrue(by_host["live-host"]["reachable"])
        self.assertTrue(by_host["live-host"]["rocm_smi_ok"])
        self.assertFalse(by_host["dead-host"]["reachable"])

        statuses = {s["host"]: s for s in pool.status()}
        self.assertNotEqual(statuses["live-host"]["state"], "dead")
        self.assertEqual(statuses["dead-host"]["state"], "dead")

    @patch("tuning_agent.machine_pool.RemoteExecutor")
    def test_validate_returns_result_for_every_machine(self, MockExecutor):
        """validate_connectivity() must return one entry per machine."""
        mock_exec = MagicMock()
        mock_exec.check_ssh_connectivity.return_value = True
        cp = subprocess.CompletedProcess(args=[], returncode=0)
        cp.stdout = "/usr/bin/rocm-smi"
        cp.stderr = ""
        mock_exec.ssh_run.return_value = cp
        MockExecutor.return_value = mock_exec

        machines = [_make_machine(f"host-{i}", i + 1) for i in range(5)]
        pool = MachinePool(machines)
        results = pool.validate_connectivity()
        self.assertEqual(len(results), 5)


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):
    """Smoke-tests for concurrent allocation and release."""

    def test_concurrent_allocate_release(self):
        """Concurrent threads should each allocate a distinct machine."""
        n = 8
        machines = [_make_machine(f"host-{i}", i + 1) for i in range(n)]
        pool = MachinePool(machines)

        allocated: list = []
        errors: list = []
        lock = threading.Lock()

        def worker():
            try:
                m = pool.allocate("concurrent_kernel")
                with lock:
                    allocated.append(m.host)
                # Simulate some work then release.
                pool.release(m.host)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        # All hosts should have been allocated at least once across all threads.
        self.assertGreater(len(allocated), 0)


if __name__ == "__main__":
    unittest.main()
