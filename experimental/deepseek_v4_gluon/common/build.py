# SPDX-License-Identifier: MIT
"""Build switches. ``DSV4G_NOINLINE=1``: the stages as noinline functions (each
compiled once: a fraction of the compile time of one inlined kernel, a call's
register save / restore and waitcnt a stage hop at run time)."""

import os

NOINLINE = os.environ.get("DSV4G_NOINLINE", "0") == "1"
