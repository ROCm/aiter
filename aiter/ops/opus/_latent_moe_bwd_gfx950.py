# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes as ct
import os
from pathlib import Path

import torch

from ...jit.core import OPUS_GEN_CO_DIR
from ..asm.asm_utils import current_raw_stream, get_function, launch_co

_PREFIX = "opus_gemm_gfx950_bf16_"
_TILE = "256x256x256x64_2x2_16x16x32_0x0x0"
_NN = f"{_PREFIX}m_grouped_{_TILE}_nn"
_TN = f"{_PREFIX}k_grouped_{_TILE}_tn_f32"
_SHORT = _TN + "_shortk"
_PLAN = _PREFIX + "latent_moe_bwd_plan"
_DESC = _NN + "_descriptor_plan"
_SORT = _TN + "_sort_plan"
_GENERIC = _PREFIX + "latent_moe_bwd_scalar_"


def _args(name, pointers, strides="", integers=""):
    fields = [
        (key, typ)
        for typ, keys in (
            (ct.c_void_p, pointers),
            (ct.c_int64, strides),
            (ct.c_int32, integers),
        )
        for key in keys.split()
    ]
    return type(name, (ct.Structure,), {"_fields_": fields})


_MArgs = _args(
    "MGroupedArgs",
    "dy weight dx rows offsets",
    "expert_stride",
    "m n k lda ldb ldc batch_a batch_c experts",
)
_KArgs = _args(
    "KGroupedArgs",
    "dy x dw src table turn",
    "expert_stride",
    "m n lda ldb ldc write_empty",
)
_PlanArgs = _args("PlanArgs", "offsets counts table", integers="experts")
_DescArgs = _args(
    "DescriptorArgs",
    "dy weight dx offsets table descriptors",
    integers="experts n k capacity",
)
_SortArgs = _args("SortArgs", "offsets counts table", integers="experts descending")
_ScalarArgs = _args("ScalarArgs", "x weight dy offsets dx dw", integers="experts n k")
assert (ct.sizeof(_MArgs), ct.sizeof(_KArgs)) == (88, 80)


def backward(x, weight, dy, offsets):
    e, n, k = weight.shape
    r = x.shape[0]
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight, dtype=torch.float32)
    if not r or not n or not k:
        return dx.zero_(), dw.zero_()
    device = x.device.index
    stream = current_raw_stream(device)
    root = (
        Path(os.environ.get("OPUS_GEN_CO_DIR") or OPUS_GEN_CO_DIR)
        / "gfx950/a16w16_grouped"
    )

    def launch(symbol, grid, threads, args):
        func = get_function(str(root / (symbol + ".co")), symbol, device)
        launch_co(func, grid, (threads, 1, 1), args, stream)

    if (
        (n, k) not in ((3072, 3584), (3584, 3072))
        or e > 1024
        or r > 140000
        or any(t.data_ptr() % 16 for t in (x, weight, dy))
    ):
        args = _ScalarArgs(
            x.data_ptr(),
            weight.data_ptr(),
            dy.data_ptr(),
            offsets.data_ptr(),
            dx.data_ptr(),
            dw.data_ptr(),
            e,
            n,
            k,
        )
        launch(_GENERIC + "nn", (r, 1, 1), 256, args)
        launch(_GENERIC + "tn_f32", (n, e, 1), 256, args)
        return dx, dw

    counts = torch.empty(e, dtype=torch.int32, device=x.device)
    table = torch.empty((e, 4), dtype=torch.int32, device=x.device)
    plan = _PlanArgs(offsets.data_ptr(), counts.data_ptr(), table.data_ptr(), e)
    launch(_PLAN, ((e + 255) // 256, 1, 1), 256, plan)

    capacity = (r + 255 * e) // 256
    columns = k // 256
    nn = _MArgs(
        dy.data_ptr(),
        weight.data_ptr(),
        dx.data_ptr(),
        counts.data_ptr(),
        offsets.data_ptr(),
        n * k,
        r,
        k,
        n,
        n,
        k,
        k,
        0,
        0,
        e,
    )
    tn = _KArgs(
        dy.data_ptr(),
        x.data_ptr(),
        dw.data_ptr(),
        None,
        table.data_ptr(),
        None,
        n * k,
        n,
        k,
        n,
        k,
        k,
        1,
    )
    tiles = columns * (n // 256)
    launch(_TN + "_stream", (tiles, e, 1), 256, tn)
    launch(_NN + "_compact", (columns * capacity, 1, 1), 256, nn)
    return dx, dw
