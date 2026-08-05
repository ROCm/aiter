#!/usr/bin/env python3
"""Assemble a gfx1250 .s into a loadable code object and launch it via HIP.

This bypasses LLVM IR optimisation, register allocation and machine scheduling:
LLVM MC is used only as an assembler, so what you write is what runs. Verify
that with ``inspect`` / ``--verify-order``, which disassemble the built object
and diff the instruction sequence against the source.

    python isa_runner.py build   smoke_gfx1250.s
    python isa_runner.py inspect smoke_gfx1250.s
    python isa_runner.py run     smoke_gfx1250.s --smoke
    python isa_runner.py bench   smoke_gfx1250.s --smoke --iters 200
    python isa_runner.py inspect ../isa_cmp/w1/<kernel>/21_final_isa.s --verify-order

As a library::

    mod = IsaModule.from_source("kernel.s")
    spec = KernelLaunchSpec(grid=(64, 1, 1), block=(128, 1, 1),
                            shared_mem_bytes=159744)
    mod.launch("kernel_name", [tensor, ctypes.c_int32(7)], spec)
    stats = mod.benchmark("kernel_name", [...], spec, iters=100)

Requires ROCm's LLVM and a HIP runtime; talks to libamdhip64 through ctypes so
hip-python is not needed. Run inside the gfx1250 container.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ARCH = "gfx1250"
DEFAULT_CACHE = Path(os.environ.get("ISA_RUNNER_CACHE", Path.home() / ".isa_runner_cache"))
_ROCM = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))

# The gemm1 TDM kernel's launch profile, from 19_gpu_module_to_binary.mlir
# (threads in (128,1,1), dynamic_shared_memory_size 159744, wave32). Recorded
# for reuse; this stage does not launch that 15-arg kernel.
TDM_GEMM1_BLOCK = (128, 1, 1)
TDM_GEMM1_LDS_BYTES = 159744


class IsaRunnerError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Toolchain
# --------------------------------------------------------------------------

def _find_tool(*names: str) -> str:
    for name in names:
        candidate = _ROCM / "llvm" / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        found = shutil.which(name)
        if found:
            return found
    raise IsaRunnerError(f"none of {names} found (looked in {_ROCM}/llvm/bin and PATH)")


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise IsaRunnerError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

@dataclass
class BuildResult:
    source: Path
    code_object: Path
    arch: str
    cached: bool
    kernels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": str(self.source),
            "code_object": str(self.code_object),
            "arch": self.arch,
            "cached": self.cached,
            "kernels": self.kernels,
        }


def build(source: str | Path, arch: str = DEFAULT_ARCH, *,
          cache_dir: Path | None = None, force: bool = False) -> BuildResult:
    """Assemble + link *source* into an AMDHSA code object.

    Two steps, both LLVM MC / LLD only: ``clang++ -x assembler -c`` then a
    shared link, which is what makes the result loadable by hipModuleLoadData
    (a bare .o is not).
    """
    source = Path(source).resolve()
    if not source.is_file():
        raise IsaRunnerError(f"source not found: {source}")

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(source.read_bytes() + arch.encode()).hexdigest()[:16]
    co_path = cache_dir / f"{source.stem}.{arch}.{digest}.co"

    if co_path.is_file() and not force:
        return BuildResult(source, co_path, arch, True, list_kernels(co_path))

    clangxx = _find_tool("clang++")
    obj_path = co_path.with_suffix(".o")
    _run([clangxx, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
          f"-mcpu={arch}", "-c", str(source), "-o", str(obj_path)])
    # -nostdlib: nothing to pull in, and the default libs are host-side.
    _run([clangxx, "-target", "amdgcn-amd-amdhsa", f"-mcpu={arch}",
          "-nostdlib", "-Wl,--no-undefined", "-shared",
          str(obj_path), "-o", str(co_path)])
    obj_path.unlink(missing_ok=True)

    return BuildResult(source, co_path, arch, False, list_kernels(co_path))


# --------------------------------------------------------------------------
# Inspect
# --------------------------------------------------------------------------

def list_kernels(code_object: str | Path) -> list[str]:
    """Kernel names in *code_object*, derived from the .kd descriptor symbols."""
    nm = _find_tool("llvm-nm")
    out = _run([nm, "--defined-only", str(code_object)]).stdout
    names = []
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".kd"):
            names.append(parts[-1][:-3])
    return sorted(set(names))


def disassemble(code_object: str | Path, arch: str = DEFAULT_ARCH) -> str:
    objdump = _find_tool("llvm-objdump")
    return _run([objdump, "-d", f"--mcpu={arch}", str(code_object)]).stdout


_COMMENT = re.compile(r"(;|//).*$")
_INSTR_LINE = re.compile(r"^\s*([a-z][a-z0-9_]*)\b")
# Directives, labels and .set lines are not instructions.
_SKIP_PREFIX = (".", "#")


def _source_mnemonics(text: str) -> list[str]:
    out = []
    in_metadata = False
    for raw in text.splitlines():
        line = _COMMENT.sub("", raw).strip()
        if not line:
            continue
        if line.startswith(".amdgpu_metadata"):
            in_metadata = True
            continue
        if line.startswith(".end_amdgpu_metadata"):
            in_metadata = False
            continue
        if in_metadata or line.startswith(_SKIP_PREFIX) or line.endswith(":"):
            continue
        m = _INSTR_LINE.match(line)
        if m:
            out.append(m.group(1))
    return out


def _disasm_mnemonics(text: str) -> list[str]:
    """Mnemonics from llvm-objdump output.

    Instruction lines are indented and carry the address/encoding in a trailing
    ``// <addr>: <bytes>`` comment, so the comment must be stripped first --
    splitting on ':' would cut operands like ``s[4:5]``.
    """
    out = []
    for raw in text.splitlines():
        if not raw.startswith((" ", "\t")):
            continue  # labels, section headers, "file format" line
        body = _COMMENT.sub("", raw).strip()
        if not body or body.endswith(":") or body.startswith(_SKIP_PREFIX):
            continue
        m = _INSTR_LINE.match(body)
        if m:
            out.append(m.group(1))
    return out


def verify_order(source: str | Path, code_object: str | Path,
                 arch: str = DEFAULT_ARCH) -> dict:
    """Check the assembler emitted our instructions in our order.

    Compares mnemonic sequences. Encoding-level differences (operand syntax,
    _e32/_e64 suffixes) are expected and ignored; a reorder, insertion or
    deletion is not.
    """
    src = _source_mnemonics(Path(source).read_text())
    dis = _disasm_mnemonics(disassemble(code_object, arch))

    # objdump prints the canonical form, e.g. v_add_nc_u32_e32 -> v_add_nc_u32.
    def norm(seq):
        return [re.sub(r"_e(32|64)$", "", s) for s in seq]

    nsrc, ndis = norm(src), norm(dis)

    # Trailing s_code_end comes from the source's own .p2alignl/.fill padding
    # (0xBF9F0000). It sits after the last real instruction and is required
    # encoding padding, not a scheduling change -- drop it before comparing.
    padding = 0
    while ndis and ndis[-1] == "s_code_end" and len(ndis) > len(nsrc):
        ndis.pop()
        padding += 1

    ok = nsrc == ndis
    result = {"ok": ok, "source_count": len(nsrc), "disasm_count": len(ndis)}
    if padding:
        result["trailing_s_code_end_ignored"] = padding
    if not ok:
        first = next((i for i, (a, b) in enumerate(zip(nsrc, ndis)) if a != b),
                     min(len(nsrc), len(ndis)))
        result["first_divergence"] = {
            "index": first,
            "source": nsrc[max(0, first - 3):first + 4],
            "disasm": ndis[max(0, first - 3):first + 4],
        }
    return result


# --------------------------------------------------------------------------
# HIP runtime (ctypes; hip-python is not installed in the container)
# --------------------------------------------------------------------------

class _Hip:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise IsaRunnerError("libamdhip64.so not found")

        v = ctypes.c_void_p
        sigs = {
            "hipInit": ([ctypes.c_uint], ctypes.c_int),
            "hipSetDevice": ([ctypes.c_int], ctypes.c_int),
            "hipDeviceSynchronize": ([], ctypes.c_int),
            "hipGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
            "hipModuleLoadData": ([ctypes.POINTER(v), v], ctypes.c_int),
            "hipModuleUnload": ([v], ctypes.c_int),
            "hipModuleGetFunction": ([ctypes.POINTER(v), v, ctypes.c_char_p], ctypes.c_int),
            "hipModuleLaunchKernel": (
                [v, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                 ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                 ctypes.c_uint, v, ctypes.POINTER(v), ctypes.POINTER(v)],
                ctypes.c_int),
            "hipEventCreate": ([ctypes.POINTER(v)], ctypes.c_int),
            "hipEventDestroy": ([v], ctypes.c_int),
            "hipEventRecord": ([v, v], ctypes.c_int),
            "hipEventSynchronize": ([v], ctypes.c_int),
            "hipEventElapsedTime": ([ctypes.POINTER(ctypes.c_float), v, v], ctypes.c_int),
            "hipMalloc": ([ctypes.POINTER(v), ctypes.c_size_t], ctypes.c_int),
            "hipFree": ([v], ctypes.c_int),
            "hipMemset": ([v, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
            "hipMemcpyDtoH": ([v, v, ctypes.c_size_t], ctypes.c_int),
        }
        for name, (argtypes, restype) in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, restype

        self.check(self.lib.hipInit(0))

    def check(self, err: int):
        if err != 0:
            msg = self.lib.hipGetErrorString(err)
            raise IsaRunnerError(
                f"HIP error {err}: {msg.decode() if msg else 'unknown'}")


# HIP's "extra" argument protocol: a NULL-terminated marker array pointing at a
# single packed kernarg buffer, which is why arg packing must match the
# kernel's kernarg layout byte for byte.
_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(3)


@dataclass
class KernelLaunchSpec:
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (128, 1, 1)
    shared_mem_bytes: int = 0
    stream: int = 0
    device: int = 0


def _pack_args(args: Iterable[Any]) -> ctypes.Array:
    """Pack kernel args into one kernarg buffer with natural alignment.

    Accepts torch tensors and ints (pointers/scalars) plus explicit ctypes
    values; use ctypes when the width matters, since a bare Python int is
    ambiguous and is packed as a 4-byte int32 here.
    """
    blob = bytearray()

    def put(value, align: int):
        blob.extend(b"\0" * ((-len(blob)) % align))
        blob.extend(bytes(value))

    for a in args:
        if hasattr(a, "data_ptr"):  # torch.Tensor
            put(ctypes.c_uint64(a.data_ptr()), 8)
        elif isinstance(a, ctypes._SimpleCData):
            put(a, ctypes.sizeof(a))
        elif isinstance(a, bool):
            put(ctypes.c_int32(int(a)), 4)
        elif isinstance(a, int):
            put(ctypes.c_int32(a), 4)
        elif isinstance(a, float):
            put(ctypes.c_float(a), 4)
        else:
            raise IsaRunnerError(f"unsupported kernel arg type: {type(a)}")

    if not blob:
        blob.extend(b"\0")
    return (ctypes.c_char * len(blob)).from_buffer(blob)


class IsaModule:
    """A loaded code object. Kept resident so launch cost excludes loading."""

    def __init__(self, code_object: str | Path, device: int = 0,
                 source: str | Path | None = None):
        self.code_object = Path(code_object)
        self.source = Path(source) if source else None
        self.device = device
        self._hip = _Hip()
        self._hip.check(self._hip.lib.hipSetDevice(device))

        data = self.code_object.read_bytes()
        self._blob = (ctypes.c_char * len(data)).from_buffer_copy(data)
        self._module = ctypes.c_void_p()
        self._hip.check(self._hip.lib.hipModuleLoadData(
            ctypes.byref(self._module), ctypes.cast(self._blob, ctypes.c_void_p)))
        self._functions: dict[str, ctypes.c_void_p] = {}

    @classmethod
    def from_source(cls, source: str | Path, arch: str = DEFAULT_ARCH,
                    device: int = 0, force: bool = False) -> "IsaModule":
        res = build(source, arch, force=force)
        return cls(res.code_object, device=device, source=res.source)

    def function(self, name: str) -> ctypes.c_void_p:
        if name not in self._functions:
            fn = ctypes.c_void_p()
            self._hip.check(self._hip.lib.hipModuleGetFunction(
                ctypes.byref(fn), self._module, name.encode()))
            self._functions[name] = fn
        return self._functions[name]

    def launch(self, name: str, args: Sequence[Any], spec: KernelLaunchSpec):
        fn = self.function(name)
        buf = _pack_args(args)
        size = ctypes.c_size_t(len(buf))
        extra = (ctypes.c_void_p * 5)(
            _HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(buf, ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_END,
        )
        self._hip.check(self._hip.lib.hipModuleLaunchKernel(
            fn, *spec.grid, *spec.block, spec.shared_mem_bytes,
            ctypes.c_void_p(spec.stream), None, extra))

    def synchronize(self):
        self._hip.check(self._hip.lib.hipDeviceSynchronize())

    def benchmark(self, name: str, args: Sequence[Any], spec: KernelLaunchSpec,
                  *, iters: int = 100, warmup: int = 20) -> dict:
        """Time repeated launches with HIP events (dispatch-level timing).

        Per-launch cost is the mean over one timed region of *iters* launches,
        so it includes launch overhead; that is the honest number for a
        dispatch-level API. For single-wave cycle counts use
        s_get_shader_cycles_u64 inside the ISA instead.
        """
        for _ in range(warmup):
            self.launch(name, args, spec)
        self.synchronize()

        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        self._hip.check(self._hip.lib.hipEventCreate(ctypes.byref(start)))
        self._hip.check(self._hip.lib.hipEventCreate(ctypes.byref(stop)))
        try:
            self._hip.check(self._hip.lib.hipEventRecord(
                start, ctypes.c_void_p(spec.stream)))
            for _ in range(iters):
                self.launch(name, args, spec)
            self._hip.check(self._hip.lib.hipEventRecord(
                stop, ctypes.c_void_p(spec.stream)))
            self._hip.check(self._hip.lib.hipEventSynchronize(stop))
            ms = ctypes.c_float()
            self._hip.check(self._hip.lib.hipEventElapsedTime(
                ctypes.byref(ms), start, stop))
        finally:
            self._hip.lib.hipEventDestroy(start)
            self._hip.lib.hipEventDestroy(stop)

        total_us = ms.value * 1000.0
        return {
            "kernel": name,
            "iters": iters,
            "warmup": warmup,
            "total_us": total_us,
            "per_launch_us": total_us / iters,
            "grid": list(spec.grid),
            "block": list(spec.block),
            "shared_mem_bytes": spec.shared_mem_bytes,
        }

    def close(self):
        if getattr(self, "_module", None):
            self._hip.lib.hipModuleUnload(self._module)
            self._module = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

def run_smoke(module: IsaModule, kernel: str = "isa_smoke", *,
              blocks: int = 4, block_size: int = 64, sentinel: int = 0x5A5A0000,
              iters: int = 0, warmup: int = 20) -> dict:
    """Launch the smoke kernel and check every element equals sentinel + gid.

    Allocates through HIP directly so the check does not depend on torch.
    """
    hip = _Hip()
    n = blocks * block_size
    nbytes = n * 4
    buf = ctypes.c_void_p()
    hip.check(hip.lib.hipMalloc(ctypes.byref(buf), nbytes))
    try:
        hip.check(hip.lib.hipMemset(buf, 0, nbytes))
        spec = KernelLaunchSpec(grid=(blocks, 1, 1), block=(block_size, 1, 1))
        args = [ctypes.c_uint64(buf.value), ctypes.c_uint32(sentinel),
                ctypes.c_uint32(block_size), ctypes.c_uint32(n)]

        module.launch(kernel, args, spec)
        module.synchronize()

        host = (ctypes.c_uint32 * n)()
        hip.check(hip.lib.hipMemcpyDtoH(ctypes.cast(host, ctypes.c_void_p), buf, nbytes))
        got = list(host)
        expected = [(sentinel + i) & 0xFFFFFFFF for i in range(n)]
        bad = [(i, got[i], expected[i]) for i in range(n) if got[i] != expected[i]]

        result = {
            "kernel": kernel,
            "elements": n,
            "sentinel": hex(sentinel),
            "passed": not bad,
            "mismatches": len(bad),
            "first_mismatches": bad[:8],
            "sample": [hex(x) for x in got[:4]],
        }
        if iters and not bad:
            result["benchmark"] = module.benchmark(
                kernel, args, spec, iters=iters, warmup=warmup)
        return result
    finally:
        hip.lib.hipFree(buf)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _emit(obj: dict, as_json: bool, path: str | None = None):
    text = json.dumps(obj, indent=2)
    if as_json:
        print(text)
    else:
        for k, v in obj.items():
            print(f"{k}: {v}")
    if path:
        Path(path).write_text(text + "\n")
        print(f"[isa_runner] wrote {path}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "inspect", "run", "bench"])
    p.add_argument("source", help="path to the .s file")
    p.add_argument("--arch", default=DEFAULT_ARCH)
    p.add_argument("--kernel", help="kernel name (default: the only one found)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--force", action="store_true", help="ignore the build cache")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", help="also write the result JSON here")
    p.add_argument("--verify-order", action="store_true",
                   help="diff disassembly against source mnemonics")
    p.add_argument("--disasm", help="write the disassembly to this path")
    p.add_argument("--smoke", action="store_true",
                   help="run/bench the built-in smoke check")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args(argv)

    try:
        res = build(args.source, args.arch, force=args.force)
    except IsaRunnerError as e:
        print(f"[isa_runner] build failed: {e}", file=sys.stderr)
        return 1

    report: dict[str, Any] = res.as_dict()

    if args.command in ("inspect", "build") or args.verify_order or args.disasm:
        if args.disasm:
            Path(args.disasm).write_text(disassemble(res.code_object, args.arch))
            report["disasm_path"] = args.disasm
        if args.verify_order or args.command == "inspect":
            report["verify_order"] = verify_order(
                res.source, res.code_object, args.arch)

    if args.command in ("build", "inspect"):
        _emit(report, args.json, args.out)
        ok = report.get("verify_order", {"ok": True})["ok"]
        return 0 if ok else 2

    kernel = args.kernel
    if not kernel:
        if len(res.kernels) != 1:
            print(f"[isa_runner] --kernel required; found {res.kernels}",
                  file=sys.stderr)
            return 1
        kernel = res.kernels[0]

    try:
        with IsaModule(res.code_object, device=args.device, source=res.source) as mod:
            report["loaded"] = True
            mod.function(kernel)
            report["symbol_found"] = kernel
            if args.smoke:
                report["smoke"] = run_smoke(
                    mod, kernel, blocks=args.blocks, block_size=args.block_size,
                    iters=args.iters if args.command == "bench" else 0,
                    warmup=args.warmup)
            elif args.command == "bench":
                print("[isa_runner] bench without --smoke needs explicit args; "
                      "use the IsaModule API", file=sys.stderr)
                return 1
    except IsaRunnerError as e:
        report["error"] = str(e)
        _emit(report, args.json, args.out)
        return 1

    _emit(report, args.json, args.out)
    smoke = report.get("smoke")
    if smoke and not smoke["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
