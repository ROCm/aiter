# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU harness for MoE dispatch: simulated archs, dispatch keys, snapshot records.

Shared by the legacy dispatch snapshot (tests/test_moe_dispatch_snapshot.py) and the
legacy-vs-moev2 differential test (tests/test_moev2_differential.py).

NOTE: tests/moe_dispatch belongs to the aiter.moev2 migration (under active development,
not graduated). Please coordinate changes with the moev2 owners (@qli88). See
aiter/moev2/README.md.

Golden snapshots (golden/*.jsonl.gz, one per simulated arch) record legacy dispatch.
After an intended change to it, regenerate them with
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
`snapshot.py --refresh-inputs` re-freezes the tuned tables they read (golden/inputs/);
do that in a PR of its own.
"""
