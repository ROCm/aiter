# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from time import sleep

from aiter.ops.flydsl.kernels import tensor_shim


def test_run_compiled_traces_once_when_threads_race(monkeypatch):
    class FakeExecutable:
        pass

    executable = FakeExecutable()
    start = Barrier(3)
    compile_started = Event()
    allow_compile_to_finish = Event()
    state_lock = Lock()
    compile_calls = 0
    executions = []

    class FakeCompiledFunction:
        def __call__(self, value):
            with state_lock:
                executions.append(("cached", value))

    def fake_compile(exe, value):
        nonlocal compile_calls
        assert exe is executable
        with state_lock:
            compile_calls += 1
        compile_started.set()
        assert allow_compile_to_finish.wait(timeout=5)
        with state_lock:
            executions.append(("compile", value))
        return FakeCompiledFunction()

    monkeypatch.setattr(tensor_shim.flyc, "compile", fake_compile)

    def invoke(value):
        start.wait()
        tensor_shim._run_compiled(executable, value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke, "first")
        second = pool.submit(invoke, "second")
        start.wait()
        assert compile_started.wait(timeout=5)
        sleep(0.05)  # Let the losing thread reach the cold-path lock.
        allow_compile_to_finish.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert compile_calls == 1
    assert sorted(value for _, value in executions) == ["first", "second"]
    assert sorted(kind for kind, _ in executions) == ["cached", "compile"]
