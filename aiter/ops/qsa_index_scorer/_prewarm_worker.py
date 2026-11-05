# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``prewarm(jobs>1)`` worker: compile-only build of a list of variants (no device).

Run by :func:`api._compile_in_workers` as ``python -B _prewarm_worker.py '<json>'`` with
``COMPILE_ONLY=1`` and ``FLYDSL_RUNTIME_CACHE_DIR`` set: loads this package standalone
(no ``import aiter``, which touches the GPU / JIT at import time) and writes the compiled
variants into the shared persistent FlyDSL cache.
"""

import importlib.util
import json
import os
import sys


def main(argv) -> int:
    req = json.loads(argv[1])
    import torch
    torch.cuda.current_stream = lambda *a, **k: None     # never touch a device
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "qsa_isc_prewarm_worker", os.path.join(here, "__init__.py"),
        submodule_search_locations=[here])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["qsa_isc_prewarm_worker"] = pkg
    spec.loader.exec_module(pkg)
    api = sys.modules["qsa_isc_prewarm_worker.api"]
    done = api._compile_work(req["profile"], req["work"], req.get("cache_dir"))
    print(json.dumps({"done": done}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
