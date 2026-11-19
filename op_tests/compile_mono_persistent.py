# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile the native and P256 stages for gfx1250 without launching a GPU."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from aiter.ops.triton._gluon_kernels.gfx1250.quant.fp8_group32 import (
    _quantize_fp8_group32_kernel,
)
from aiter.ops.triton._gluon_kernels.gfx1250.quant.fused_mxfp8_quant import (
    _fused_dual_rmsnorm_mxfp8_quant_ps_kernel,
)
from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (
    _fused_dual_rmsnorm_mxfp8_quant_kernel,
)


def native_reference(root):
    path = root / "op_tests/triton_tests/utils/mono_native.py"
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    excerpts = {}
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name:
            start = min(
                [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
            )
            excerpts[name] = "".join(lines[start - 1 : node.end_lineno])
    manifest = json.loads(path.with_name("mono_native_manifest.json").read_text())
    for item in manifest:
        digest = hashlib.sha256(excerpts[item["symbol"]].encode()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Frozen native source changed: {item['symbol']}")
    spec = importlib.util.spec_from_file_location("mono_native_compile_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_one(fn, name, constants, signatures, output):
    source_type = GluonASTSource if fn.is_gluon() else ASTSource
    source = source_type(fn, signature=signatures, constexprs=constants)
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4}
    )
    for suffix in ("glir", "ttir", "ttgir", "llir", "amdgcn"):
        if suffix in kernel.asm:
            (output / f"{name}.{suffix}").write_text(kernel.asm[suffix])
    (output / f"{name}.hsaco").write_bytes(kernel.asm["hsaco"])
    return {
        "name": name,
        "metadata": kernel.metadata._asdict(),
        "asm_sha256": hashlib.sha256(kernel.asm["amdgcn"].encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native = native_reference(Path(__file__).resolve().parents[1])
    results = []
    for nq, nk in ((1280, 512), (96, 33)):
        for mode in (
            "native",
            "nonpersistent",
            "persistent1",
            "persistent2",
            "persistent4",
        ):
            fn = (
                native._fused_dual_rmsnorm_mxfp8_quant_kernel
                if mode == "native"
                else (
                    _fused_dual_rmsnorm_mxfp8_quant_ps_kernel
                    if mode in ("persistent2", "persistent4")
                    else _fused_dual_rmsnorm_mxfp8_quant_kernel
                )
            )
            constants = {
                "BLOCK_SIZE_KQ": triton.next_power_of_2(nq),
                "BLOCK_SIZE_KK": triton.next_power_of_2(nk),
                "QUANT_BLOCK_SIZE": 32,
                "AMAX_FLOOR": 1e-10,
            }
            if mode.startswith("persistent"):
                constants["NUM_PRGMS"] = 256
                if mode != "persistent1":
                    constants["ROWS_PER_ITER"] = int(mode[-1])
                else:
                    constants["PERSISTENT"] = True
            signature = {name: "i32" for name in fn.arg_names}
            signature.update(
                {
                    name: "*bf16"
                    for name in ("q_ptr", "k_ptr", "gq_ptr", "gk_ptr", "yk_ptr")
                }
            )
            signature.update(
                yq_ptr="*fp8e4nv", sq_ptr="*u8", eps_q="fp32", eps_k="fp32"
            )
            signature.update({key: "constexpr" for key in constants})
            name = f"a03-{nq}-{nk}-{mode}"
            results.append(compile_one(fn, name, constants, signature, args.output))
    for dequant in (False, True):
        for mode in (
            "native",
            "nonpersistent",
            "persistent1",
            "persistent2",
            "persistent4",
        ):
            fn = (
                native.quantize_fp8_kernel
                if mode == "native"
                else _quantize_fp8_group32_kernel
            )
            constants = {"DEQUANT": dequant}
            if mode != "native":
                constants["PERSISTENT"] = mode.startswith("persistent")
                constants["TILES_PER_ITER"] = (
                    int(mode[-1]) if mode.startswith("persistent") else 1
                )
            signature = {
                "X": "*bf16",
                "Y": "*bf16" if dequant else "*fp8e4nv",
                "S": "*u8",
                "SIZE": "i32",
            }
            signature.update({key: "constexpr" for key in constants})
            results.append(
                compile_one(
                    fn,
                    f"a09-qat{int(dequant)}-{mode}",
                    constants,
                    signature,
                    args.output,
                )
            )
    report = {
        "triton": triton.__version__,
        "target": "gfx1250",
        "gpu_executed": False,
        "kernels": results,
    }
    (args.output / "compile.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
