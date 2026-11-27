# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device codecs of the Attention A2A transport, public for producers of its packed bytes.

These are the transport's own scale rules, rounding and packers, so a fused producer (the
pre-attention monokernel) that quantizes with them writes the bytes the A2A op would.
"""

from .attention_a2a_intranode_kernel import (
    _pack_int8_pair as pack_int8_pair,
)
from .attention_a2a_intranode_kernel import (
    _pack_transport_pair as pack_transport_pair,
)
from .attention_a2a_intranode_kernel import (
    _pack_transport_words as pack_transport_words,
)
from .attention_a2a_intranode_kernel import (
    _pack_v4_v_pair as pack_v4_v_pair,
)
from .attention_a2a_intranode_kernel import (
    _transport_scale as transport_scale,
)
from .attention_a2a_intranode_kernel import (
    _v4_fp6_scale as v4_fp6_scale,
)
from .attention_a2a_intranode_kernel import (
    _v4_fp8_scale as v4_fp8_scale,
)
from .attention_a2a_intranode_kernel import (
    fp6p_source_token,
)

__all__ = [
    "fp6p_source_token",
    "pack_int8_pair",
    "pack_transport_pair",
    "pack_transport_words",
    "pack_v4_v_pair",
    "transport_scale",
    "v4_fp6_scale",
    "v4_fp8_scale",
]
