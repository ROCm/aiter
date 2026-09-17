# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where a winning Triton or Gluon launch configuration gets written.

Every backend needs its winner recorded somewhere the runtime will read it
back. For the compiled backends that place is the runtime CSV's ``backend``
column alone, because their tiles are baked into the kernel symbol: naming
``asm_v3`` or ``ck`` fully determines the launch. Triton and Gluon are the
exception -- their configuration is a runtime dict with no name -- and this
module holds the two candidate answers for where that dict should live.

Both answers keep the routing row (``backend``, ``num_splits``) in the runtime
CSV, because that is what the dispatch in ``mha.py`` reads to decide where to
send the call. They differ in exactly one respect, which is the question this
seam exists to settle:

    csv   the tile dict rides along in the CSV's backend_config column as a
          JSON blob, so one file holds routing and configuration together.
    json  the tile dict is published into the backend's own config tree under
          fwd.shapes, and the CSV column is left empty, so configuration stays
          where every other Triton config already lives.

Select with ``AITER_MHA_FWD_STORE=csv|json``. The default is ``csv``, which is
the behaviour the runtime CSV already ships, so opting into the JSON store is
an explicit act and never a silent change of contract.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

from aiter.ops.mha_fwd_policy import (
    MHA_FWD_TILE_CONFIG_BACKENDS,
    MhaFwdProblem,
    canonical_backend_config,
)

MHA_FWD_STORE_ENV = "AITER_MHA_FWD_STORE"
MHA_FWD_STORE_NAMES = ("csv", "json")


class MhaFwdConfigStore(Protocol):
    """Publishes one winner's launch configuration and reports the CSV cell."""

    name: str

    def publish(
        self,
        problem: MhaFwdProblem,
        backend: str,
        config: Mapping[str, Any] | None,
    ) -> str:
        """Persist ``config`` and return the runtime CSV backend_config cell.

        Returning the cell rather than writing it keeps the runtime CSV under
        the tuner's control: the store decides whether the configuration
        travels in that column or somewhere else, and the tuner still owns the
        one atomic write of the routing table.
        """

    def describe(self) -> dict[str, Any]:
        """Evidence-file description of where configurations were written."""


class CsvBlobStore:
    """Arm B: the tile dict rides in the runtime CSV as a JSON blob.

    This is the contract the runtime CSV already implements -- ``mha.py``
    parses the column with ``parse_backend_config`` and hands the result to
    the Triton entry point as ``config=``. Publishing is therefore just
    canonicalizing the dict for the column; there is no second file.
    """

    name = "csv"

    def publish(
        self,
        problem: MhaFwdProblem,
        backend: str,
        config: Mapping[str, Any] | None,
    ) -> str:
        if backend not in MHA_FWD_TILE_CONFIG_BACKENDS:
            # Name-is-config backends have nothing to carry; an empty cell is
            # the honest record, not a missing one.
            return ""
        return canonical_backend_config(config)

    def describe(self) -> dict[str, Any]:
        return {
            "store": self.name,
            "location": "runtime CSV backend_config column",
            "config_owner": "mha runtime CSV",
        }


class JsonShapeStore:
    """Arm A: the tile dict is published into the backend's own config tree.

    The entry lands under ``fwd.shapes[<gpu>][<shape>]`` in the family file
    the kernel already reads, so a winner applies to every caller of that
    kernel rather than only to calls routed through the MHA runtime CSV. The
    CSV keeps the routing row and leaves the config column empty.
    """

    name = "json"

    def publish(
        self,
        problem: MhaFwdProblem,
        backend: str,
        config: Mapping[str, Any] | None,
    ) -> str:
        if backend not in MHA_FWD_TILE_CONFIG_BACKENDS:
            return ""
        if not config:
            raise ValueError(f"{backend} winner carries no launch configuration")

        # Imported here so the op layer does not pull the Triton config
        # machinery in at module import time.
        from aiter.ops.triton.utils.attention_config_utils import (
            MHA_TUNABLE_KEYS,
            format_mha_shape_key,
            mha_config_relpath,
        )
        from aiter.ops.triton.utils.config_utils import (
            AITER_TRITON_CONFIGS_PATH,
            format_hardware_key,
        )
        from aiter.ops.triton.utils.config_writer import (
            update_config_entry,
            validate_config_entry,
        )

        entry = validate_config_entry(
            config,
            required=(),
            optional=MHA_TUNABLE_KEYS[backend],
            where=f"{backend} MHA winner for {problem.key()}",
        )
        relpath = mha_config_relpath(backend, arch=problem.gfx)
        update_config_entry(
            f"{AITER_TRITON_CONFIGS_PATH}/{relpath}",
            (
                "fwd",
                "shapes",
                # Keyed by the GPU that was measured, not the one publishing.
                format_hardware_key(problem.cu_num, problem.gpu_model),
                format_mha_shape_key(
                    mode=problem.mode,
                    hdim_q=problem.hdim_q,
                    hdim_v=problem.hdim_v,
                    nhead_q=problem.nhead_q,
                    nhead_k=problem.nhead_k,
                    dtype=problem.dtype,
                    causal=problem.causal,
                    max_seqlen_q=problem.max_seqlen_q,
                    max_seqlen_k=problem.max_seqlen_k,
                ),
            ),
            entry,
        )
        return ""

    def describe(self) -> dict[str, Any]:
        return {
            "store": self.name,
            "location": "aiter/ops/triton/configs/<arch>/<backend>/attention/mha, fwd.shapes",
            "config_owner": "triton/gluon config tree",
        }


_STORES = {store.name: store for store in (CsvBlobStore(), JsonShapeStore())}


def get_mha_fwd_store(name: str | None = None) -> MhaFwdConfigStore:
    """Return the configured store, defaulting to the shipped CSV contract."""
    selected = (name or os.getenv(MHA_FWD_STORE_ENV) or "csv").strip().lower()
    if selected not in _STORES:
        raise ValueError(
            f"unknown MHA config store {selected!r}; "
            f"expected one of {list(MHA_FWD_STORE_NAMES)}"
        )
    return _STORES[selected]
