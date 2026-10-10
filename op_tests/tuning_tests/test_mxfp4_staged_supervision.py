# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only process ownership checks for the staged private-Pipe supervisor.

Imports use the normal ROCm/FlyDSL toolchain with synthetic discovery and empty
GPU visibility. The only admitted worker targets are the CPU functions below.
"""

import multiprocessing
import os
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from types import SimpleNamespace

import pandas as pd
import pytest

from op_tests.tuning_tests import test_mxfp4_tuner_lifecycle as lifecycle

moe = lifecycle.moe
_GET_CONTEXT = multiprocessing.get_context
_PROCESS_START = BaseProcess.start


def cpu_result_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    keys, row, args, gpu = payload
    del keys, args
    try:
        result_tx.send({**row, "gpu": gpu, "pid": os.getpid(), "us": 3.0})
    finally:
        result_tx.close()


def cpu_coordinated_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    keys, row, args, gpu = payload
    del keys
    try:
        if row.get("barrier"):
            args.barrier.wait(timeout=15)
        result_tx.send({**row, "gpu": gpu, "pid": os.getpid(), "us": 3.0})
        if row.get("pause") and not args.allow_exit.wait(timeout=15):
            raise RuntimeError("test did not release CPU worker")
    finally:
        result_tx.close()


def cpu_outcome_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    keys, row, args, gpu = payload
    del keys, args
    try:
        if row.get("outcome") == "fatal":
            os._exit(23)
        if row.get("outcome") == "eof":
            return
        result = {**row, "gpu": gpu, "pid": os.getpid(), "us": 3.0}
        if row.get("outcome") == "python-error":
            result.update(us=-1.0, kernelName1="FAILED")
        result_tx.send(result)
        if row.get("outcome") == "result-fatal":
            os._exit(23)
    finally:
        result_tx.close()


CPU_WORKERS = {cpu_result_worker, cpu_coordinated_worker, cpu_outcome_worker}


@pytest.fixture(autouse=True)
def allow_only_explicit_cpu_shape_workers(monkeypatch: "pytest.MonkeyPatch") -> "None":
    lifecycle.prohibit_real_shape_workers_and_gpu_data.__wrapped__(monkeypatch)

    def cpu_only_start(process: "BaseProcess") -> "None":
        if process._target not in CPU_WORKERS:
            pytest.fail("supervisor test attempted a non-CPU worker target")
        return _PROCESS_START(process)

    monkeypatch.setattr(BaseProcess, "start", cpu_only_start)


def payloads(
    *rows: "object", args: "SimpleNamespace | None" = None
) -> "list[tuple[tuple[str, ...], dict[str, object], SimpleNamespace, None]]":
    return [(tuple(row), row, args or SimpleNamespace(), None) for row in rows]


def test_staged_supervisor_empty_batch_launches_none_and_one_shape_gets_worker_pid() -> (
    "None"
):
    ctx = _GET_CONTEXT("spawn")
    assert moe._run_staged_shapes_isolated([], 1, ctx, entry=cpu_result_worker) == []
    results = moe._run_staged_shapes_isolated(
        payloads({"token": 4}), 1, ctx, entry=cpu_result_worker
    )
    assert results[0]["token"] == 4 and results[0]["gpu"] == 0
    assert results[0]["pid"] != os.getpid()


class TrackingProcess:
    def __init__(
        self, process: "BaseProcess", gpu: "int", context: "TrackingContext"
    ) -> "None":
        self.process, self.gpu, self.context = process, gpu, context
        self.closed = False
        self.joined = False
        self.saved_exitcode = None

    def start(self) -> "None":
        outstanding = [proc for proc in self.context.processes if not proc.closed]
        self.context.handle_counts.append(len(outstanding))
        self.context.starts.append(self.gpu)
        assert all(proc.gpu != self.gpu for proc in outstanding if proc is not self)
        if self.context.start_failure is not None:
            raise self.context.start_failure
        self.process.start()

    def join(self, *args: "object", **kwargs: "object") -> "None":
        self.process.join(*args, **kwargs)
        self.joined = True

    def close(self) -> "None":
        self.saved_exitcode = self.process.exitcode
        self.process.close()
        self.closed = True

    @property
    def exitcode(self) -> "int | None":
        return self.saved_exitcode if self.closed else self.process.exitcode

    def __getattr__(self, name: "str") -> "object":
        return getattr(self.process, name)


class TrackingConnection:
    def __init__(self, connection: "Connection", context: "TrackingContext") -> "None":
        self.connection, self.context = connection, context

    def recv_bytes(self) -> "bytes":
        if self.context.receive_failure is not None:
            raise self.context.receive_failure
        payload = self.connection.recv_bytes()
        with self.context.received:
            self.context.receive_count += 1
            self.context.received.notify_all()
        return b"invalid-pickle" if self.context.invalid_payload else payload

    def close(self) -> "None":
        self.connection.close()
        if self.context.close_failure is not None:
            raise self.context.close_failure

    def __getattr__(self, name: "str") -> "object":
        return getattr(self.connection, name)


class TrackingContext:
    def __init__(self, base: "BaseContext") -> "None":
        self.base = base
        self.processes = []
        self.pipes = []
        self.starts = []
        self.handle_counts = []
        self.received = threading.Condition()
        self.receive_count = 0
        self.receive_failure = None
        self.start_failure = None
        self.close_failure = None
        self.invalid_payload = False

    def Pipe(
        self, *args: "object", **kwargs: "object"
    ) -> "tuple[TrackingConnection, Connection]":
        rx, tx = self.base.Pipe(*args, **kwargs)
        result = TrackingConnection(rx, self), tx
        self.pipes.append(result)
        return result

    def Process(self, *args: "object", **kwargs: "object") -> "TrackingProcess":
        process = TrackingProcess(
            self.base.Process(*args, **kwargs), kwargs["args"][0][3], self
        )
        self.processes.append(process)
        return process

    def wait_until_received(self, count: "int") -> "bool":
        with self.received:
            return self.received.wait_for(
                lambda: self.receive_count >= count, timeout=20
            )

    def assert_closed(self) -> "None":
        assert all(process.closed for process in self.processes)
        assert all(endpoint.closed for pipe in self.pipes for endpoint in pipe)


def run_in_thread(
    operation: "Callable[[], object]",
) -> "tuple[threading.Thread, list[object], list[BaseException]]":
    result = []
    errors = []

    def run() -> "None":
        try:
            result.append(operation())
        except BaseException as error:  # noqa: BLE001 - transport test-thread failures
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result, errors


def test_result_arrival_keeps_gpu_and_process_owned_until_exit() -> "None":
    base = _GET_CONTEXT("spawn")
    args = SimpleNamespace(allow_exit=base.Event())
    ctx = TrackingContext(base)
    thread, results, errors = run_in_thread(
        lambda: moe._run_staged_shapes_isolated(
            payloads({"token": 1, "pause": True}, {"token": 2}, args=args),
            1,
            ctx,
            entry=cpu_coordinated_worker,
        )
    )
    try:
        assert ctx.wait_until_received(1)
        assert ctx.starts == [0]
        assert not ctx.processes[0].closed and not ctx.processes[0].joined
    finally:
        args.allow_exit.set()
        thread.join(timeout=20)
    assert not thread.is_alive() and errors == []
    assert [row["token"] for row in results[0]] == [1, 2]
    assert all(process.joined for process in ctx.processes)
    ctx.assert_closed()


def test_different_gpu_workers_overlap_and_all_completed_handles_close() -> "None":
    base = _GET_CONTEXT("spawn")
    args = SimpleNamespace(allow_exit=base.Event(), barrier=base.Barrier(2))
    ctx = TrackingContext(base)
    rows = [{"token": i, "pause": i < 2, "barrier": i < 2} for i in range(6)]
    thread, results, errors = run_in_thread(
        lambda: moe._run_staged_shapes_isolated(
            payloads(*rows, args=args), 2, ctx, entry=cpu_coordinated_worker
        )
    )
    try:
        assert ctx.wait_until_received(2)
        assert ctx.starts == [0, 1]
        assert all(process.process.is_alive() for process in ctx.processes)
    finally:
        args.allow_exit.set()
        thread.join(timeout=30)
    assert not thread.is_alive() and errors == []
    assert len(results[0]) == 6
    assert len({row["pid"] for row in results[0]}) == 6
    assert max(ctx.handle_counts) <= 2
    assert all(process.joined and process.exitcode == 0 for process in ctx.processes)
    ctx.assert_closed()


def test_one_slot_uses_fresh_worker_and_bounded_handles_for_every_shape() -> "None":
    ctx = TrackingContext(_GET_CONTEXT("spawn"))
    results = moe._run_staged_shapes_isolated(
        payloads(*[{"token": i} for i in range(4)]), 1, ctx, entry=cpu_result_worker
    )
    assert len({row["pid"] for row in results}) == 4
    assert ctx.starts == [0, 0, 0, 0] and ctx.handle_counts == [1, 1, 1, 1]
    assert all(process.joined for process in ctx.processes)
    ctx.assert_closed()


def test_fatal_clean_eof_python_failure_and_result_before_fatal_continue_later_rows() -> (
    "None"
):
    ctx = TrackingContext(_GET_CONTEXT("spawn"))
    results = moe._run_staged_shapes_isolated(
        payloads(
            *[
                {"token": i, "outcome": outcome}
                for i, outcome in enumerate(
                    ("fatal", "eof", "python-error", "result-fatal", "success")
                )
            ]
        ),
        1,
        ctx,
        entry=cpu_outcome_worker,
    )
    assert results[0] is results[1] is results[3] is None
    assert results[2]["us"] < 0 and results[2]["kernelName1"] == "FAILED"
    assert results[4]["token"] == 4 and results[4]["us"] > 0
    assert [proc.exitcode for proc in ctx.processes] == [23, 0, 0, 23, 0]
    ctx.assert_closed()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("receive failed"),
        KeyboardInterrupt("receive cancelled"),
        "deserialize",
    ],
)
def test_unexpected_receive_and_deserialization_failure_clean_owned_workers(
    failure: "Exception | str",
) -> "None":
    base = _GET_CONTEXT("spawn")
    args = SimpleNamespace(allow_exit=base.Event())
    ctx = TrackingContext(base)
    if failure == "deserialize":
        ctx.invalid_payload = True
        expected = Exception
    else:
        ctx.receive_failure = failure
        expected = type(failure)
    with pytest.raises(expected) as actual:
        moe._run_staged_shapes_isolated(
            payloads({"token": 1, "pause": True}, {"token": 2}, args=args),
            1,
            ctx,
            entry=cpu_coordinated_worker,
        )
    if failure != "deserialize":
        assert actual.value is failure
    assert ctx.starts == [0]
    assert ctx.processes[0].joined and ctx.processes[0].exitcode is not None
    ctx.assert_closed()


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt("parent cancelled"), RuntimeError("wait failed")]
)
def test_parent_cancellation_or_wait_failure_stops_launches_and_preserves_exception(
    monkeypatch: "pytest.MonkeyPatch", failure: "Exception | str"
) -> "None":
    base = _GET_CONTEXT("spawn")
    ctx = TrackingContext(base)
    args = SimpleNamespace(allow_exit=base.Event())

    def fail_wait(*args: "object", **kwargs: "object") -> "None":
        raise failure

    monkeypatch.setattr(moe, "wait_for_connections", fail_wait)
    with pytest.raises(type(failure)) as actual:
        moe._run_staged_shapes_isolated(
            payloads({"token": 1, "pause": True}, {"token": 2}, args=args),
            1,
            ctx,
            entry=cpu_coordinated_worker,
        )
    assert actual.value is failure and ctx.starts == [0]
    assert ctx.processes[0].joined and ctx.processes[0].exitcode is not None
    ctx.assert_closed()


def test_start_failure_closes_created_process_and_both_pipe_endpoints() -> "None":
    ctx = TrackingContext(_GET_CONTEXT("spawn"))
    ctx.start_failure = RuntimeError("CPU spawn failed")
    with pytest.raises(RuntimeError) as actual:
        moe._run_staged_shapes_isolated(
            payloads({"token": 1}, {"token": 2}), 1, ctx, entry=cpu_result_worker
        )
    assert actual.value is ctx.start_failure and ctx.starts == [0]
    ctx.assert_closed()


def test_start_failure_survives_pipe_close_failure_during_cleanup() -> "None":
    ctx = TrackingContext(_GET_CONTEXT("spawn"))
    ctx.start_failure = RuntimeError("CPU spawn failed")
    ctx.close_failure = ValueError("receive endpoint close failed")
    with pytest.raises(RuntimeError) as actual:
        moe._run_staged_shapes_isolated(
            payloads({"token": 1}, {"token": 2}), 1, ctx, entry=cpu_result_worker
        )
    assert actual.value is ctx.start_failure and ctx.starts == [0]
    ctx.assert_closed()


@pytest.mark.parametrize(
    "mp,shape_count,gpu_count,expected",
    [(0, 3, 3, 1), (-1, 3, 3, 1), (8, 1, 3, 1), (8, 3, 2, 2), (2, 3, 3, 2)],
)
def test_staged_scheduler_preserves_main_mp_cap_and_uses_preflighted_first_visible_gpus(
    monkeypatch: "pytest.MonkeyPatch",
    mp: "int",
    shape_count: "int",
    gpu_count: "int",
    expected: "object",
) -> "None":
    identities = [("gfx950", 256)] * expected + [("gfx942", 304)] * (
        gpu_count - expected
    )
    with lifecycle.devices(*identities):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", lifecycle.KEYS, lifecycle.RESULTS)
        observed = []

        def schedule(
            batch: "list[dict[str, object]]", slots: "int", ctx: "BaseContext"
        ) -> "list[dict[str, object]]":
            observed.append((slots, len(batch)))
            return [dict(row, us=3.0) for _, row, _, _ in batch]

        monkeypatch.setattr(moe, "_run_staged_shapes_isolated", schedule)
        results = tuner.tune(
            pd.DataFrame([lifecycle.shape(i + 1) for i in range(shape_count)]),
            None,
            SimpleNamespace(mp=mp, timeout=0, mxfp4_search_mode="staged"),
        )
    assert observed == [(expected, shape_count)] and len(results) == shape_count


def test_staged_single_slot_uses_first_visible_gpu_and_empty_batch_never_schedules(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with lifecycle.devices(("gfx950", 224), ("gfx942", 304), current=1):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", lifecycle.KEYS, lifecycle.RESULTS)
        launches = []

        def schedule(
            batch: "list[dict[str, object]]", slots: "int", ctx: "BaseContext"
        ) -> "list[dict[str, object]]":
            launches.append(slots)
            return [
                dict(row, gfx="gfx950", cu_num=224, us=3.0) for _, row, _, _ in batch
            ]

        monkeypatch.setattr(moe, "_run_staged_shapes_isolated", schedule)
        args = SimpleNamespace(mp=1, timeout=0, mxfp4_search_mode="staged")
        assert tuner.tune(pd.DataFrame(columns=lifecycle.KEYS), None, args) == []
        results = tuner.tune(pd.DataFrame([lifecycle.shape()]), None, args)
    assert launches == [1] and results[0]["cu_num"] == 224


def test_staged_rejects_unsupported_participant_before_scheduler(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with lifecycle.devices(("gfx950", 256), ("gfx942", 304)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", lifecycle.KEYS, lifecycle.RESULTS)

        def forbidden(*args: "object", **kwargs: "object") -> "None":
            pytest.fail("unsupported staged participant reached scheduling")

        monkeypatch.setattr(moe, "_run_staged_shapes_isolated", forbidden)
        with pytest.raises(ValueError, match="GPU1.*gfx942"):
            tuner.tune(
                pd.DataFrame([lifecycle.shape(), lifecycle.shape(8)]),
                None,
                SimpleNamespace(mp=2, timeout=0, mxfp4_search_mode="staged"),
            )


def test_staged_result_remains_readable_when_process_exit_precedes_receive(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    ctx = TrackingContext(_GET_CONTEXT("spawn"))
    observed = []

    def wait_after_exit(waitables: "list[object]") -> "list[object]":
        from multiprocessing.connection import wait

        process = next(proc for proc in ctx.processes if not proc.closed)
        wait([process.sentinel])
        process.process.join()
        assert process.process.exitcode == 0
        observed.append(ctx.receive_count)
        return wait(waitables)

    monkeypatch.setattr(moe, "wait_for_connections", wait_after_exit)
    results = moe._run_staged_shapes_isolated(
        payloads({"token": 1}, {"token": 2}), 1, ctx, entry=cpu_result_worker
    )
    assert observed == [0, 1]
    assert [row["token"] for row in results] == [1, 2]
    assert all(process.joined for process in ctx.processes)
    ctx.assert_closed()
