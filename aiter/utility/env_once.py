# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Environment switches that per-call host code consults, read once per process."""

import functools
import os


@functools.cache
def env_once(name: str, default=None):
    """``os.environ.get(name, default)``, kept until ``os.environ`` changes.

    For switches the per-call dispatch of an op reads: one ``os.environ``
    lookup costs about 0.4 us, and the gfx1250 grouped MoE asks about 35 per
    call. Any change made through ``os.environ`` (``monkeypatch.setenv``,
    ``mock.patch.dict``, a tuning sweep) clears the cache, so the result is
    always what ``os.environ.get`` would return now.
    """
    return os.environ.get(name, default)


_CLEARED_WITH_ENV_ONCE = []


def clear_with_env_once(cached_reader):
    """Clear ``cached_reader`` (a ``functools.cache`` function that reads the
    environment) whenever ``env_once`` is cleared, so a named switch helper
    follows ``os.environ`` the same way. Returns it, so it can decorate."""
    _CLEARED_WITH_ENV_ONCE.append(cached_reader)
    return cached_reader


def _clear_all():
    env_once.cache_clear()
    for cached_reader in _CLEARED_WITH_ENV_ONCE:
        cached_reader.cache_clear()


def _clear_on_environ_change():
    environ_type = type(os.environ)
    if getattr(environ_type, "_aiter_env_once_hooked", False):
        return
    set_item, del_item = environ_type.__setitem__, environ_type.__delitem__

    def __setitem__(self, key, value):
        set_item(self, key, value)
        _clear_all()

    def __delitem__(self, key):
        del_item(self, key)
        _clear_all()

    environ_type.__setitem__ = __setitem__
    environ_type.__delitem__ = __delitem__
    environ_type._aiter_env_once_hooked = True


_clear_on_environ_change()
