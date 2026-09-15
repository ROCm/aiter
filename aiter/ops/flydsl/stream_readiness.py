# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-stream readiness tracking for asynchronously produced GPU tensors."""

import threading
import weakref

import torch

_LOCK = threading.RLock()
_EVENTS = {}


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    device = tensor.device
    return (device.type, device.index, storage.data_ptr(), storage.nbytes())


def _event_complete(event):
    try:
        return event.query()
    except Exception:  # noqa: BLE001 - a live event must remain registered
        return False


def _prune_completed():
    for key, (event, _refs) in list(_EVENTS.items()):
        if _event_complete(event):
            _EVENTS.pop(key, None)


def register_ready(tensors, stream=None):
    """Record producer completion and associate it with tensor allocations."""
    tensors = tuple(tensors)
    if not tensors:
        return tensors
    if stream is None:
        stream = torch.cuda.current_stream(tensors[0].device)
    with _LOCK:
        _prune_completed()
        tensors_by_key = {}
        for tensor in tensors:
            tensors_by_key.setdefault(_storage_key(tensor), []).append(tensor)

        # Serialize this producer after every previously published producer for
        # the same allocations. Holding the host lock through publication makes
        # the newest event cover all earlier work even when registrations race.
        seen = set()
        for key in tensors_by_key:
            current = _EVENTS.get(key)
            if current is not None and id(current[0]) not in seen:
                stream.wait_event(current[0])
                seen.add(id(current[0]))

        event = torch.cuda.Event()
        event.record(stream)

        for key, key_tensors in tensors_by_key.items():
            refs = []
            for tensor in key_tensors:

                def _released(ref, key=key, event=event):
                    # An unregistered alias may still own this allocation. Only
                    # discard the entry once producer work has actually completed.
                    with _LOCK:
                        current = _EVENTS.get(key)
                        if (
                            current is not None
                            and current[0] is event
                            and _event_complete(event)
                        ):
                            _EVENTS.pop(key, None)

                refs.append(weakref.ref(tensor, _released))
                tensor.record_stream(stream)
            _EVENTS[key] = (event, refs)
    return tensors


def wait_ready(stream, tensors):
    """Make ``stream`` wait for registered producers of tensors or aliases."""
    events = []
    with _LOCK:
        _prune_completed()
        for tensor in tensors:
            entry = _EVENTS.get(_storage_key(tensor))
            if entry is not None:
                events.append(entry[0])
    seen = set()
    for event in events:
        key = id(event)
        if key not in seen:
            stream.wait_event(event)
            seen.add(key)
