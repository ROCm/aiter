# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional runner role and phase for GEMM dispatch.

Use both contexts around target prefill calls to keep full and cache-reuse
prefills in the same GEMM family. Capture a separate GPU graph per context;
replay uses the kernels selected at capture time.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

ExecutionRole = Literal["target", "draft"]
ExecutionPhase = Literal["prefill", "decode", "verify"]
_execution_role: ContextVar[ExecutionRole | None] = ContextVar(
    "aiter_execution_role", default=None
)
_execution_phase: ContextVar[ExecutionPhase | None] = ContextVar(
    "aiter_execution_phase", default=None
)


def get_execution_role() -> ExecutionRole | None:
    """Return the active runner role, or ``None`` outside a runner forward."""
    return _execution_role.get()


def get_execution_phase() -> ExecutionPhase | None:
    """Return the active forward phase, or ``None`` outside a runner forward."""
    return _execution_phase.get()


@contextmanager
def execution_role(role: ExecutionRole) -> Iterator[None]:
    """Scope dispatch to a target or speculative-draft runner forward."""
    token = _execution_role.set(role)
    try:
        yield
    finally:
        _execution_role.reset(token)


@contextmanager
def execution_phase(phase: ExecutionPhase) -> Iterator[None]:
    """Scope dispatch to a prefill, decode, or teacher-verification forward."""
    token = _execution_phase.set(phase)
    try:
        yield
    finally:
        _execution_phase.reset(token)
