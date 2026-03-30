"""Tests for the terminal dashboard."""

import time
import threading
from unittest.mock import patch

import pytest

from aiter.ops.triton.utils._triton.tunning.tuning_agent.dashboard import Dashboard


class TestRenderEmpty:
    def test_render_empty_contains_header(self):
        d = Dashboard()
        output = d.render()
        assert "TRITON KERNEL TUNING PIPELINE" in output

    def test_render_empty_contains_separator(self):
        d = Dashboard()
        output = d.render()
        assert "═" in output

    def test_render_empty_contains_sections(self):
        d = Dashboard()
        output = d.render()
        assert "MACHINES:" in output
        assert "KERNELS:" in output
        assert "RECENT LOGS:" in output
        assert "NOTIFICATIONS:" in output

    def test_render_returns_string(self):
        d = Dashboard()
        output = d.render()
        assert isinstance(output, str)

    def test_render_empty_no_machine_entries(self):
        d = Dashboard()
        output = d.render()
        # No machine entries should appear beyond header
        lines = output.splitlines()
        machine_section = False
        machine_entries = 0
        for line in lines:
            if "MACHINES:" in line:
                machine_section = True
                continue
            if machine_section and line.strip() and not line.startswith("═"):
                if "KERNELS:" in line:
                    break
                if line.strip():
                    machine_entries += 1
        assert machine_entries == 0


class TestUpdateMachine:
    def test_machine_appears_in_render(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "idle", gpu_count=8)
        output = d.render()
        assert "gpu-machine-1" in output

    def test_machine_state_idle_appears(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "idle", gpu_count=4)
        output = d.render()
        assert "IDLE" in output

    def test_machine_state_busy_appears(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "busy", kernel="a8w8", gpu_count=8)
        output = d.render()
        assert "BUSY" in output

    def test_machine_state_dead_appears(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "dead", gpu_count=4)
        output = d.render()
        assert "DEAD" in output

    def test_machine_kernel_appears_when_busy(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "busy", kernel="a8w8", gpu_count=8)
        output = d.render()
        assert "a8w8" in output

    def test_machine_gpu_count_appears(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "idle", gpu_count=8)
        output = d.render()
        assert "8" in output

    def test_multiple_machines_appear(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "busy", kernel="a8w8", gpu_count=8)
        d.update_machine("gpu-machine-2", "idle", gpu_count=4)
        output = d.render()
        assert "gpu-machine-1" in output
        assert "gpu-machine-2" in output

    def test_update_machine_overwrites_previous(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "idle", gpu_count=4)
        d.update_machine("gpu-machine-1", "busy", kernel="a8w8", gpu_count=4)
        output = d.render()
        assert "BUSY" in output

    def test_machines_stored_in_dict(self):
        d = Dashboard()
        d.update_machine("host1", "idle", gpu_count=2)
        assert "host1" in d.machines
        assert d.machines["host1"]["state"] == "idle"
        assert d.machines["host1"]["gpu_count"] == 2

    def test_machine_kernel_stored(self):
        d = Dashboard()
        d.update_machine("host1", "busy", kernel="mykernel", gpu_count=4)
        assert d.machines["host1"]["kernel"] == "mykernel"

    def test_machine_kernel_none_when_not_provided(self):
        d = Dashboard()
        d.update_machine("host1", "idle")
        assert d.machines["host1"]["kernel"] is None


class TestUpdateKernel:
    def test_kernel_appears_in_render(self):
        d = Dashboard()
        d.update_kernel("a8w8", 4, "TUNING", progress="12/56", elapsed=720)
        output = d.render()
        assert "a8w8" in output

    def test_kernel_phase_name_appears(self):
        d = Dashboard()
        d.update_kernel("a8w8", 4, "TUNING", progress="12/56", elapsed=720)
        output = d.render()
        assert "TUNING" in output

    def test_kernel_phase_number_appears(self):
        d = Dashboard()
        d.update_kernel("a8w8", 4, "TUNING", progress="12/56", elapsed=720)
        output = d.render()
        assert "4" in output

    def test_kernel_elapsed_appears(self):
        d = Dashboard()
        d.update_kernel("a8w8", 4, "TUNING", progress="", elapsed=720)
        output = d.render()
        # 720 seconds = 12m
        assert "12m" in output

    def test_kernel_status_appears(self):
        d = Dashboard()
        d.update_kernel("a8w8", 4, "TUNING", status="running")
        output = d.render()
        assert "running" in output

    def test_kernel_status_done(self):
        d = Dashboard()
        d.update_kernel("a8w8", 6, "DONE", status="done")
        output = d.render()
        assert "done" in output

    def test_multiple_kernels_appear(self):
        d = Dashboard()
        d.update_kernel("a8w8", 4, "TUNING", elapsed=720)
        d.update_kernel("afp4wfp4", 2, "BASELINE", elapsed=180)
        output = d.render()
        assert "a8w8" in output
        assert "afp4wfp4" in output

    def test_kernel_stored_in_dict(self):
        d = Dashboard()
        d.update_kernel("mykernel", 3, "COLLECTING", elapsed=60, status="running")
        assert "mykernel" in d.kernels
        assert d.kernels["mykernel"]["phase"] == 3
        assert d.kernels["mykernel"]["phase_name"] == "COLLECTING"
        assert d.kernels["mykernel"]["elapsed"] == 60
        assert d.kernels["mykernel"]["status"] == "running"

    def test_kernel_default_status_is_running(self):
        d = Dashboard()
        d.update_kernel("mykernel", 1, "SETUP")
        assert d.kernels["mykernel"]["status"] == "running"

    def test_kernel_progress_stored(self):
        d = Dashboard()
        d.update_kernel("mykernel", 4, "TUNING", progress="45/56")
        assert d.kernels["mykernel"]["progress"] == "45/56"


class TestAddLog:
    def test_log_appears_in_render(self):
        d = Dashboard()
        d.add_log("[14:23:01] a8w8: 45/56 shapes tuned")
        output = d.render()
        assert "a8w8: 45/56 shapes tuned" in output

    def test_multiple_logs_appear(self):
        d = Dashboard()
        d.add_log("First log message")
        d.add_log("Second log message")
        output = d.render()
        assert "First log message" in output
        assert "Second log message" in output

    def test_log_stored_in_list(self):
        d = Dashboard()
        d.add_log("test message")
        assert "test message" in d.logs

    def test_log_buffer_max_20(self):
        d = Dashboard()
        for i in range(25):
            d.add_log(f"log line {i}")
        assert len(d.logs) == 20

    def test_log_buffer_keeps_last_20(self):
        d = Dashboard()
        for i in range(25):
            d.add_log(f"log line {i}")
        # Should have lines 5..24 (last 20)
        assert "log line 5" in d.logs
        assert "log line 24" in d.logs
        assert "log line 4" not in d.logs

    def test_log_buffer_max_20_in_render(self):
        d = Dashboard()
        for i in range(25):
            d.add_log(f"log line {i}")
        output = d.render()
        assert "log line 24" in output
        assert "log line 4" not in output


class TestAddNotification:
    def test_notification_appears_in_render(self):
        d = Dashboard()
        d.add_notification("[14:20:00] a16w16 tuning complete: 1.8x geomean")
        output = d.render()
        assert "a16w16 tuning complete" in output

    def test_multiple_notifications_appear(self):
        d = Dashboard()
        d.add_notification("First notification")
        d.add_notification("Second notification")
        output = d.render()
        assert "First notification" in output
        assert "Second notification" in output

    def test_notification_stored_in_list(self):
        d = Dashboard()
        d.add_notification("test notification")
        assert "test notification" in d.notifications


class TestRenderAnsiColors:
    def test_busy_shows_yellow_ansi(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "busy", kernel="a8w8", gpu_count=8)
        output = d.render()
        # Yellow ANSI code
        assert "\033[33m" in output

    def test_idle_shows_green_ansi(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "idle", gpu_count=4)
        output = d.render()
        # Green ANSI code
        assert "\033[32m" in output

    def test_dead_shows_red_ansi(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "dead", gpu_count=4)
        output = d.render()
        # Red ANSI code
        assert "\033[31m" in output

    def test_ansi_reset_present(self):
        d = Dashboard()
        d.update_machine("gpu-machine-1", "idle", gpu_count=4)
        output = d.render()
        assert "\033[0m" in output

    def test_busy_yellow_wraps_state_label(self):
        d = Dashboard()
        d.update_machine("myhost", "busy", gpu_count=2)
        output = d.render()
        # Yellow should appear before BUSY in the same vicinity
        yellow_pos = output.find("\033[33m")
        busy_pos = output.find("BUSY")
        assert yellow_pos != -1 and busy_pos != -1
        assert yellow_pos < busy_pos

    def test_idle_green_wraps_state_label(self):
        d = Dashboard()
        d.update_machine("myhost", "idle", gpu_count=2)
        output = d.render()
        green_pos = output.find("\033[32m")
        idle_pos = output.find("IDLE")
        assert green_pos != -1 and idle_pos != -1
        assert green_pos < idle_pos


class TestAutoRefresh:
    def test_start_and_stop(self):
        d = Dashboard(refresh_interval=0.1)
        d.start_auto_refresh()
        time.sleep(0.25)
        d.stop()
        # Should not raise or hang

    def test_stop_without_start(self):
        d = Dashboard()
        # Should not raise
        d.stop()

    def test_auto_refresh_calls_render(self):
        render_calls = []
        d = Dashboard(refresh_interval=0.05)
        original_render = d.render

        def counting_render():
            result = original_render()
            render_calls.append(1)
            return result

        d.render = counting_render
        with patch("builtins.print"):
            d.start_auto_refresh()
            time.sleep(0.2)
            d.stop()
        assert len(render_calls) >= 2

    def test_refresh_thread_is_daemon(self):
        d = Dashboard(refresh_interval=10.0)
        with patch("builtins.print"):
            d.start_auto_refresh()
        # Grab the thread before stopping
        thread = d._thread
        d.stop()
        assert thread is not None
        assert thread.daemon is True


class TestDashboardInit:
    def test_default_refresh_interval(self):
        d = Dashboard()
        assert d.refresh_interval == 2.0

    def test_custom_refresh_interval(self):
        d = Dashboard(refresh_interval=5.0)
        assert d.refresh_interval == 5.0

    def test_initial_machines_empty(self):
        d = Dashboard()
        assert d.machines == {}

    def test_initial_kernels_empty(self):
        d = Dashboard()
        assert d.kernels == {}

    def test_initial_logs_empty(self):
        d = Dashboard()
        assert d.logs == []

    def test_initial_notifications_empty(self):
        d = Dashboard()
        assert d.notifications == []
