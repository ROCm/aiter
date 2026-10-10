# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Operation timeout ownership under controlled signal/timer boundaries."""

import signal
from unittest import mock

from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import (
    KEYS,
    moe,
)


def test_staged_operation_restores_existing_handler_and_timer():
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    tuner.keys = KEYS
    owned_handler = object()
    restored = []
    with (
        mock.patch.object(signal, "getsignal", return_value=owned_handler),
        mock.patch.object(signal, "getitimer", return_value=(25.0, 3.0)),
        mock.patch.object(
            signal, "signal", side_effect=lambda sig, handler: restored.append(handler)
        ),
        mock.patch.object(signal, "setitimer") as timer,
        tuner._staged_timeout(4),
    ):
        pass
    assert restored[-1] is owned_handler
    assert timer.call_args_list[-1] == mock.call(signal.ITIMER_REAL, 25.0, 3.0)


def test_staged_soft_timeout_is_returned_as_screen_failure_and_restores_timer():
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    installed = []
    owned_handler = object()

    def interrupted():
        installed[0](signal.SIGALRM, None)

    with (
        mock.patch.object(signal, "getsignal", return_value=owned_handler),
        mock.patch.object(signal, "getitimer", return_value=(13.0, 0.0)),
        mock.patch.object(
            signal, "signal", side_effect=lambda sig, handler: installed.append(handler)
        ),
        mock.patch.object(signal, "setitimer") as timer,
    ):
        assert tuner._screen_candidate(interrupted, "fixture", 4) < 0
    assert installed[-1] is owned_handler
    assert timer.call_args_list[-1] == mock.call(signal.ITIMER_REAL, 13.0, 0.0)


def test_staged_soft_timeout_unavailable_handler_runs_without_touching_timer():
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    with (
        mock.patch.object(signal, "getsignal", return_value=None),
        mock.patch.object(signal, "setitimer") as timer,
    ):
        assert tuner._screen_candidate(lambda: 3.0, "fixture", 4) == 3.0
    timer.assert_not_called()


def test_staged_soft_timeout_outside_main_thread_runs_without_touching_timer():
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    with (
        mock.patch.object(signal, "signal", side_effect=ValueError("main thread")),
        mock.patch.object(signal, "setitimer") as timer,
    ):
        assert tuner._screen_candidate(lambda: 3.0, "fixture", 4) == 3.0
    timer.assert_not_called()


def test_mxfp4_help_reports_soft_timeout_and_preserves_other_tuner_help():
    from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import RESULTS, devices

    with devices(("gfx950", 256)):
        dedicated = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        ordinary = moe.FmoeTuner("fmoe", KEYS, RESULTS)
    help_text = " ".join(dedicated.parser.format_help().split())
    assert "best-effort per-operation SIGALRM" in help_text
    assert "still-live permanent hang" in help_text
    assert "per-task watchdog" in ordinary.parser.format_help()
