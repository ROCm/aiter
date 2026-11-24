# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Batch specs: the requests of one attention call, as text.

Underscore-separated segments, each `(<count>) q<q_len>(k) (s<seq_len>(k))`:

  - count: identical requests (default 1)
  - q_len: query tokens (the new ones)
  - seq_len: sequence length, the query tokens included (default q_len)
  - a `k` suffix multiplies by 1024

    q2k            one prefill of 2048 tokens
    q1s1k          one decode at a 1024-token sequence
    8q1s1k         eight of them
    q4s1k          four tokens after 1020 cached ones
    2q1k_32q1s1k   two 1024-token prefills and 32 decodes
"""

import re
from dataclasses import dataclass

_SEGMENT = re.compile(r"^(\d+)?q(\d+)(k?)(?:s(\d+)(k?))?$")


@dataclass(frozen=True)
class BatchRequest:
    """One request of a batch."""

    q_len: int  # query tokens
    kv_len: int  # sequence length, the query tokens included


def parse_batch_spec(spec: str) -> list[BatchRequest]:
    """The requests of a batch spec, in order.

    Raises:
        ValueError: A segment does not follow the grammar.

    """
    requests = []
    for segment in spec.split("_"):
        m = _SEGMENT.match(segment)
        if not m:
            raise ValueError(f"invalid batch spec segment: {segment!r}")
        q_len = int(m[2]) * (1024 if m[3] else 1)
        kv_len = int(m[4]) * (1024 if m[5] else 1) if m[4] else q_len
        requests += [BatchRequest(q_len, kv_len)] * int(m[1] or 1)
    return requests
