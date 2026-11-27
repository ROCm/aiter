# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Old name of :mod:`aiter.ops.flydsl.ipc_heap`, kept for external importers.

vLLM imports ``UncachedIpcHeap`` from here. Import ``IpcHeap`` from
``ipc_heap`` in new code. Remove this module once vLLM imports ``ipc_heap``.
"""

from .ipc_heap import IpcHeap, UncachedIpcHeap

__all__ = ["IpcHeap", "UncachedIpcHeap"]
