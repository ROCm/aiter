# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A runner's mailboxes and where a step begins with them.

A mailbox tag is a layer's, so a pair left from one step would pass for the
next step's: every buffer holding pairs is zeroed before a step's first
kernel, then the peer fence holds every rank until all of them zeroed, so no
peer's first push of the step lands on an unzeroed buffer or is wiped by a
late zeroing (``step_begin``, one launch). At the step's start, not its end:
a step that stopped half-way leaves nothing for the next one to trip on.
"""

import torch

from atom.mono.runtime.peer_memory import PeerBuffer
from atom.mono.runtime.step_begin import MISSING, STATE_WORDS, step_begin


class StepMailboxes:
    """``peers``' data region and this rank's ``scratch`` buffers, zeroed
    together by ``begin_step``. ``debug``: a bounded fence wait, a rank that
    never arrived reported by ``missing_ranks``."""

    def __init__(
        self, peers: PeerBuffer, *scratch: torch.Tensor, debug: bool = False
    ) -> None:
        self.peers = peers
        self._scratch = scratch
        self._debug = debug
        self._state = torch.zeros(
            STATE_WORDS, dtype=torch.int64, device=peers.bytes.device
        )

    def begin_step(self) -> None:
        """Before the step's first kernel, on its stream (a graph records it)."""
        step_begin(self._scratch, self.peers, self._state, self._debug)

    def missing_ranks(self) -> list[int]:
        """The ranks the fences since the last call gave up on (a debug build;
        synchronizes: eager only)."""
        missing = self._state[MISSING : MISSING + self.peers.addresses.numel()]
        ranks = [r for r, m in enumerate(missing.tolist()) if m]
        missing.zero_()
        return ranks
