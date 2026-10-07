# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_MONO_DEBUG`` builds of the V4.1 kernels: every mailbox wait bounded
(``atom.mono.device.sync._spin_bounded``), a wait that gives up recording its
region, pair index, tags and CTA in the scratch words at the build's
``diag_off``; the runner raises with the records after the step."""

import flydsl.expr as fx

from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.sync import Mailbox
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.runtime.debug import region_ids, region_namer

# the peer regions and the merged K2's readiness flags, besides the scratch
# layouts' regions
OTHER_REGIONS = ("attn", "ffn", "xrdy")


def mailbox(layer, scratch, diag_off, regions) -> Mailbox:
    """The kernel's mailbox for tag ``layer``: bounded waits recording into
    ``scratch`` at ``diag_off`` in a debug build (``diag_off`` >= 0)."""
    if diag_off < 0:
        return Mailbox(layer)
    ids = region_ids((*regions, *OTHER_REGIONS))
    return Mailbox(layer, scratch + fx.Int64(diag_off), ids)


def region_names(regions):
    """``region_id`` -> name, for ``atom.mono.runtime.debug.given_up_waits``."""
    return region_namer((*regions, *OTHER_REGIONS))
