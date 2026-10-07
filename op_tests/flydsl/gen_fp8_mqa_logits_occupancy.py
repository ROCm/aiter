# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regenerate ``_MEASURED_OCCUPANCY`` in ``aiter/ops/flydsl/kernel_occupancy.py``.

Builds every fp8_mqa_logits kernel instance for the current arch, records the
occupancy HIP reports, and checks the runtime's analytic model against it --
that agreement is what licenses reading occupancy off the artifact at all.

Run on the target arch::

    python3 op_tests/flydsl/gen_fp8_mqa_logits_occupancy.py --emit

Without ``--emit`` it only reports model-vs-HIP agreement.
"""

import argparse
import ctypes
import re
import sys
from collections import defaultdict

import torch

from aiter.dist.device_communicators.vmm_allocator import load_hip_runtime
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fp8_mqa_logits_kernels import (
    KERNEL_VARIANTS,
    _parse_variant,
    compile_fp8_mqa_logits,
)
from aiter.ops.flydsl.kernel_occupancy import (
    _artifact_metadata,
    _occupancy_from_metadata,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

_BIN_RE = re.compile(r'bin = "((?:[^"\\]|\\.)*)"')
_KNAME_RE = re.compile(r'kernel_metadata<"([^"]+)"')

_hip = load_hip_runtime()
_hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_hip.hipModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_char_p,
]
_hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
]
_hip.hipModuleUnload.argtypes = [ctypes.c_void_p]


def _unescape(text):
    """Decode the MLIR string-literal escaping of the embedded ELF."""
    out = bytearray()
    i = 0
    while i < len(text):
        if text[i] != "\\":
            out.append(ord(text[i]))
            i += 1
        elif text[i + 1] == "\\":
            out.append(0x5C)
            i += 2
        elif text[i + 1] == '"':
            out.append(0x22)
            i += 2
        else:
            out.append(int(text[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


def _hip_occupancy(launcher, threads):
    """``hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`` for the artifact."""
    ir_text = launcher._cf._keepalive.ir
    blob = _unescape(_BIN_RE.search(ir_text).group(1))
    kernel_name = _KNAME_RE.search(ir_text).group(1)
    module = ctypes.c_void_p()
    if _hip.hipModuleLoadData(
        ctypes.byref(module), ctypes.create_string_buffer(blob, len(blob))
    ):
        return None
    try:
        func = ctypes.c_void_p()
        if _hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode()):
            return None
        blocks = ctypes.c_int()
        if _hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks), func, threads, 0
        ):
            return None
        return blocks.value
    finally:
        _hip.hipModuleUnload(module)


def _dispatch_once(launcher, num_heads, head_size, rows_per_block, device):
    """Force compilation: FlyDSL has no artifact until the kernel has run."""
    seq_len, seq_len_kv = 512, 4096
    padded = -(-seq_len // rows_per_block) * rows_per_block
    logits = torch.empty((padded, seq_len_kv), dtype=torch.float32, device=device)
    _run_compiled(
        launcher,
        torch.randn(seq_len, num_heads, head_size, device=device).to(
            torch.float8_e4m3fnuz
        ),
        torch.randn(seq_len_kv, head_size, device=device).to(torch.float8_e4m3fnuz),
        torch.rand(seq_len_kv, device=device, dtype=torch.float32),
        torch.randn(seq_len, num_heads, device=device, dtype=torch.float32),
        torch.zeros(seq_len, device=device, dtype=torch.int32),
        torch.full((seq_len,), seq_len_kv, device=device, dtype=torch.int32),
        logits,
        int(padded),
        int(seq_len_kv),
        int(logits.stride(0)),
        1,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", action="store_true", help="print the dict body")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    arch = get_gfx()
    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    print(f"arch={arch} device={torch.cuda.get_device_properties(args.device).name}")

    measured = defaultdict(list)
    checked = mismatched = 0
    for num_heads in (16, 32, 64, 128):
        for head_size in (64, 128):
            for variant in KERNEL_VARIANTS:
                block_kv, rows_per_block = _parse_variant(variant)
                # Build flags can move the VGPR count across an allocation
                # boundary, so measure each and keep the minimum.
                for convert_q, convert_kv, clean in (
                    (False, False, True),
                    (True, True, True),
                    (False, False, False),
                ):
                    try:
                        launcher = compile_fp8_mqa_logits(
                            num_heads=num_heads,
                            head_size=head_size,
                            block_kv=block_kv,
                            paged=False,
                            variant=variant,
                            convert_q_fn=convert_q,
                            convert_kv_fn=convert_kv,
                            clean_logits=clean,
                        )
                        _dispatch_once(
                            launcher, num_heads, head_size, rows_per_block, device
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  skip H{num_heads} D{head_size} {variant}: "
                            f"{type(exc).__name__}"
                        )
                        continue
                    fields = _artifact_metadata(launcher)
                    if fields is None:
                        print(f"  no metadata: H{num_heads} D{head_size} {variant}")
                        continue
                    hip = _hip_occupancy(launcher, fields["threads_per_block"])
                    model = _occupancy_from_metadata(fields, arch, args.device)
                    if hip is None:
                        continue
                    checked += 1
                    if model != hip:
                        mismatched += 1
                        print(
                            f"  MISMATCH H{num_heads} D{head_size} {variant} "
                            f"model={model} hip={hip}"
                        )
                    measured[(variant, num_heads, head_size)].append(hip)

    print(f"\nchecked={checked} mismatched={mismatched}")
    if args.emit:
        print(f'\n    "{arch}": {{')
        for key in sorted(measured, key=lambda k: (k[0], k[1], k[2])):
            variant, num_heads, head_size = key
            print(
                f'        ("{variant}", {num_heads}, {head_size}): '
                f"{min(measured[key])},"
            )
        print("    },")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
