# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Deprecated alias for :mod:`aiter.ops.flydsl.ipc_heap`.

This shim keeps the old import path and class name working for external
callers such as vLLM. New code should import ``IpcHeap`` from
``aiter.ops.flydsl.ipc_heap``.
"""

from .ipc_heap import IpcHeap, UncachedIpcHeap

__all__ = ["IpcHeap", "UncachedIpcHeap"]
