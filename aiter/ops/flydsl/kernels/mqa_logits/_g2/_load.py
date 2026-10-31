# SPDX-License-Identifier: MIT
"""Load the v0 kernel module: direct file load by default.

``import aiter`` pulls in ``aiter/jit``, which builds ``module_aiter_core`` and
touches the GPU at import time. The v0 kernel needs neither, so the file load is
the default here; set ``QSA_IMPORT_PACKAGE=1`` to exercise the package path.
"""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "qsa_paged_mqa_logits_page4_gfx1250.py")
)


def load_kernel_module():
    if os.environ.get("QSA_IMPORT_PACKAGE") == "1":
        from aiter.ops.flydsl.kernels.mqa_logits import (  # noqa: F401
            qsa_paged_mqa_logits_page4_gfx1250 as K,
        )

        return K, "package:aiter.ops.flydsl.kernels.mqa_logits"
    spec = importlib.util.spec_from_file_location("qsa_page4_v0", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, f"file:{KERNEL_PATH}"
