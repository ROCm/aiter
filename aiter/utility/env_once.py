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


def _clear_on_environ_change():
    environ_type = type(os.environ)
    if getattr(environ_type, "_aiter_env_once_hooked", False):
        return
    set_item, del_item = environ_type.__setitem__, environ_type.__delitem__

    def __setitem__(self, key, value):
        set_item(self, key, value)
        env_once.cache_clear()

    def __delitem__(self, key):
        del_item(self, key)
        env_once.cache_clear()

    environ_type.__setitem__ = __setitem__
    environ_type.__delitem__ = __delitem__
    environ_type._aiter_env_once_hooked = True


_clear_on_environ_change()
