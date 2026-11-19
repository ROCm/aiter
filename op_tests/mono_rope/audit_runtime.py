"""Audit archived A05 binaries or an actual A07 JIT shared library."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import yaml

from .check_isa import select_kernel, violations


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a05-artifacts", type=Path)
    parser.add_argument("--a07-library", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.a05_artifacts) == bool(args.a07_library):
        parser.error("select exactly one input")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    llvm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin"
    objects = []
    if args.a05_artifacts:
        for meta in sorted(args.a05_artifacts.glob("A05_persistent_*/launch.json")):
            record = json.loads(meta.read_text())
            src = meta.parent / "kernel.hsaco"
            if sha(src) != record["sha256"]:
                raise ValueError(f"binary hash mismatch: {src}")
            target = out / meta.parent.name
            target.mkdir(exist_ok=True)
            dst = target / "kernel.hsaco"
            shutil.copy2(src, dst)
            shutil.copy2(meta, target / "launch.json")
            objects.append((dst, "_rope_window", record))
    else:
        library = out / args.a07_library.name
        shutil.copy2(args.a07_library, library)
        fatbin = out / "module.fatbin"
        subprocess.run(
            [
                str(llvm / "llvm-objcopy"),
                "--dump-section",
                f".hip_fatbin={fatbin}",
                str(library),
                str(out / "host-copy.so"),
            ],
            check=True,
        )
        bundler = str(llvm / "clang-offload-bundler")
        targets = subprocess.check_output(
            [bundler, "--list", "--type=o", f"--input={fatbin}"], text=True
        ).splitlines()
        selected = [x for x in targets if x.startswith("hip") and x.endswith("gfx1250")]
        if len(selected) != 1:
            raise ValueError(f"ambiguous gfx1250 target: {targets}")
        binary = out / "kernel.hsaco"
        subprocess.run(
            [
                bundler,
                "--unbundle",
                "--type=o",
                f"--targets={selected[0]}",
                f"--input={fatbin}",
                f"--output={binary}",
            ],
            check=True,
        )
        objects.append(
            (
                binary,
                "inverse_rope_group_quant_persistent_kernel",
                dict(library=str(args.a07_library), library_sha256=sha(library)),
            )
        )
    if not objects:
        raise ValueError("no runtime code objects found")
    reports = []
    for binary, fragment, origin in objects:
        asm = subprocess.check_output(
            [str(llvm / "llvm-objdump"), "-d", "--mcpu=gfx1250", str(binary)], text=True
        )
        assembly = binary.with_suffix(".disasm")
        assembly.write_text(asm)
        selected_asm = select_kernel(asm, fragment)
        binary.with_suffix(".selected.disasm").write_text(selected_asm)
        errors = violations(selected_asm)
        notes = subprocess.check_output(
            [str(llvm / "llvm-readelf"), "--notes", str(binary)], text=True
        )
        binary.with_suffix(".notes").write_text(notes)
        metadata = yaml.safe_load(notes[notes.index("amdhsa.kernels:") :])
        kernels = [k for k in metadata["amdhsa.kernels"] if fragment in k[".name"]]
        if not kernels:
            errors.append("no matching kernel metadata")
        for k in kernels:
            if k[".wavefront_size"] != 32:
                errors.append("kernel is not wave32")
        report = dict(
            binary=str(binary),
            sha256=sha(binary),
            origin=origin,
            kernels=kernels,
            errors=errors,
            passed=not errors,
        )
        reports.append(report)
    report = dict(
        passed=all(x["passed"] for x in reports),
        objects=reports,
        boundary_register_measurement=False,
        scope="Unmodified runtime machine code; no s104/s105 references, "
        "aliases or indirect calls, complete selected-function disassembly.",
    )
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                objects=len(reports),
                output=str(out / "audit.json"),
            ),
            indent=2,
        )
    )
    raise SystemExit(not report["passed"])


if __name__ == "__main__":
    main()
