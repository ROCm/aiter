#!/usr/bin/env python3
"""Rebuild a reassemblable, self-contained .s file from an AITER kernel object (.co).

The output contains three parts, all recovered from the .co with LLVM tools:

  1. .text   - instructions from `llvm-objdump -d`, with branch labels taken from
               the ELF symbol table (`llvm-readelf -s`)
  2. .rodata - the 64-byte kernel descriptor, decoded by
               `llvm-objdump -D -j .rodata` into `.amdhsa_*` directives
  3. .note   - the AMDHSA metadata (kernel arguments, register counts, LDS size)
               from `llvm-readelf --notes`, emitted as an `.amdgpu_metadata` block

Assembling the result with

    clang -x assembler -target amdgcn-amd-amdhsa -mcpu=<gfx> -o new.co kernel.s

produces a code object that hipModuleLoad accepts directly, and whose .text,
kernel descriptor and metadata are byte-identical to the input (see the guide
for the known exceptions).  Because the kernel descriptor is now source, you can
change register counts / LDS size / code size freely.

Usage:
  python3 extract_asm.py kernel.co                      # auto-detects --mcpu
  python3 extract_asm.py kernel.co --mcpu gfx942 -o kernel.s
  python3 extract_asm.py kernel.co --list               # list kernels and labels
  python3 extract_asm.py kernel.co --text-only          # .text only (objcopy flow)

Notes on the llvm-objdump format for AMDGPU (it differs from x86):

    0000000000002200 <_ZN5aiter...E>:
    <TAB>s_load_dwordx2 s[8:9], s[0:1], 0x0        // 000000002208: C0060200 00000000
    <TAB>s_cbranch_scc1 label_0CF5                 // 000000002B54: BF850A9F
    <blank line>
    0000000000002b60 <label_0258>:

i.e. the instruction comes first and address/encoding are a trailing comment,
so stripping the comment already yields assembler input.  `label_XXXX` are
ordinary local symbols present in the .co's .symtab; several names can alias
one address and objdump prints only one of them as a header, which is why the
labels are taken from the symbol table instead.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

# e_flags & 0xff -> gfx name, for toolchains whose llvm-readobj does not decode it
EF_AMDGPU_MACH = {0x3F: "gfx90a", 0x40: "gfx940", 0x4C: "gfx942", 0x4F: "gfx950"}

# Trailing comment llvm-objdump appends to every AMDGPU instruction line
ADDR_COMMENT = re.compile(r"\s*// [0-9A-Fa-f]{12}:.*$")

# Highest SGPR count .amdhsa_next_free_sgpr accepts on GFX9; the assembler adds the
# implicitly reserved VCC / FLAT_SCRATCH / XNACK_MASK registers on top of it.
GFX9_MAX_USER_SGPRS = 102


def default_llvm_bin() -> str:
    return os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm", "bin")


class Tools:
    def __init__(self, llvm_bin: str):
        self.bin = llvm_bin

    def run(self, tool: str, *args: str) -> str:
        exe = os.path.join(self.bin, tool)
        if not os.path.exists(exe):
            sys.exit(f"error: {exe} not found (set ROCM_PATH or pass --llvm-bin)")
        res = subprocess.run([exe, *args], capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"error: {tool} {' '.join(args)} failed:\n{res.stderr}")
        return res.stdout


def detect_mcpu(tools: Tools, co: str) -> str:
    hdr = tools.run("llvm-readobj", "--file-header", co)
    m = re.search(r"EF_AMDGPU_MACH_AMDGCN_(\w+)\s", hdr)
    if m:
        return m.group(1).lower()
    m = re.search(r"^\s*Flags:\s*0x([0-9A-Fa-f]+)", hdr, re.M)
    if m and (int(m.group(1), 16) & 0xFF) in EF_AMDGPU_MACH:
        return EF_AMDGPU_MACH[int(m.group(1), 16) & 0xFF]
    sys.exit("error: could not detect the GPU architecture, pass --mcpu")


def read_symbols(tools: Tools, co: str):
    """Return ([kernel entry symbols], {addr: [label, ...]}) for the .text section."""
    text_ndx = None
    for line in tools.run("llvm-readelf", "-SW", co).splitlines():
        m = re.search(r"\[\s*(\d+)\]\s+\.text\s", line)
        if m:
            text_ndx = m.group(1)
    if text_ndx is None:
        sys.exit("error: no .text section")

    kernels, labels, seen = [], defaultdict(list), set()
    for line in tools.run("llvm-readelf", "-sW", co).splitlines():
        f = line.split()
        # Num: Value Size Type Bind Vis Ndx Name
        if len(f) < 8 or not f[0].endswith(":") or f[6] != text_ndx:
            continue
        addr, typ, name = int(f[1], 16), f[3], f[7]
        if (addr, name) in seen:  # .dynsym and .symtab both list the kernel
            continue
        seen.add((addr, name))
        if typ == "FUNC":
            kernels.append((addr, name))
        elif typ == "NOTYPE":
            labels[addr].append(name)
    kernels.sort()
    return kernels, labels


def extract_text(tools: Tools, co: str, mcpu: str, kernel: str, labels) -> list:
    """Instructions of `kernel` as assembler source lines, labels included."""
    out, in_kernel, count = [], False, 0
    dis = tools.run("llvm-objdump", "-d", f"--mcpu={mcpu}", co)
    for line in dis.splitlines():
        m = re.match(r"^([0-9a-fA-F]+) <(.+)>:$", line)
        if m:
            addr, name = int(m.group(1), 16), m.group(2)
            if name == kernel:
                in_kernel = True
                out.append(f"{kernel}:")
            elif in_kernel and name not in labels.get(addr, []):
                break  # next kernel of a multi-kernel .co
            if in_kernel:
                out.extend(f"{lab}:" for lab in labels.get(addr, []))
            continue
        if in_kernel and line.startswith("\t") and "//" in line:
            # Strip the trailing "// ADDR: ENCODING" comment.  Anchor on the
            # address rather than the first "//": on some targets objdump adds
            # /*...*/ operand annotations, and on long lines the closing "*/"
            # abuts the "//" with no space in between.
            instr = ADDR_COMMENT.sub("", line).rstrip()
            if instr.lstrip().startswith((".long", ".byte")):
                print(
                    f"warning: undecodable bytes kept as data: {instr.strip()} "
                    f"(wrong --mcpu?)",
                    file=sys.stderr,
                )
            out.append(instr)
            count += 1
    if not count:
        sys.exit(f"error: no instructions found for {kernel}")
    print(
        f"{kernel}: {count} instructions, "
        f"{sum(len(v) for v in labels.values())} labels",
        file=sys.stderr,
    )
    return out


def extract_kernel_descriptor(tools: Tools, co: str, mcpu: str, kernel: str) -> list:
    """The kernel descriptor (.rodata, <kernel>.kd) as an .amdhsa_kernel block."""
    dis = tools.run("llvm-objdump", "-D", "-j", ".rodata", f"--mcpu={mcpu}", co)
    m = re.search(
        rf"^\.amdhsa_kernel {re.escape(kernel)}\n.*?^\.end_amdhsa_kernel$",
        dis,
        re.S | re.M,
    )
    if not m:
        sys.exit(
            f"error: could not decode the kernel descriptor {kernel}.kd "
            f"(needs llvm-objdump >= 12)"
        )
    block = m.group(0).splitlines()

    fixed = []
    for line in block:
        d = line.split()
        if not d:
            continue
        if d[0] == ".amdhsa_reserve_xnack_mask":
            # Must match the target-id's xnack setting; with the plain "gfxNNN"
            # target (xnack "any", which is what AITER ships) the directive is
            # rejected, and omitting it reproduces the original descriptor.
            continue
        if d[0] == ".amdhsa_next_free_sgpr":
            # The disassembler prints the *granulated total* ((field+1)*8, which
            # includes the implicitly reserved VCC/FLAT_SCRATCH/XNACK SGPRs) and
            # sets .amdhsa_reserve_vcc 0.  The assembler rejects totals > 102, so
            # re-express it as a user SGPR count with the default reservations;
            # this encodes to the same GRANULATED_WAVEFRONT_SGPR_COUNT.
            total = int(d[1])
            user = min(max(total - 8, 0), GFX9_MAX_USER_SGPRS)
            fixed.append(
                f"\t.amdhsa_next_free_sgpr {user}"
                f"\t\t// descriptor encodes {total} SGPRs incl. "
                f"VCC/FLAT_SCRATCH/XNACK_MASK"
            )
            continue
        if d[0] == ".amdhsa_reserve_vcc":
            fixed.append("\t.amdhsa_reserve_vcc 1")
            continue
        fixed.append(line)
    return ["\t.rodata", "\t.p2align 6"] + fixed


def extract_metadata(tools: Tools, co: str) -> list:
    """The NT_AMDGPU_METADATA note as an .amdgpu_metadata YAML block."""
    notes = tools.run("llvm-readelf", "--notes", co)
    m = re.search(r"^\s*---\n(.*?)^\.\.\.$", notes, re.S | re.M)
    if not m:
        sys.exit("error: no AMDGPU metadata note found")
    return [
        "\t.amdgpu_metadata",
        "---",
        m.group(1).rstrip("\n"),
        "...",
        "\t.end_amdgpu_metadata",
    ]


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild a reassemblable .s from an AMDGPU kernel object (.co)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("co", help="kernel object (.co / .hsaco)")
    ap.add_argument("--mcpu", help="GPU architecture, e.g. gfx942 (default: from ELF)")
    ap.add_argument("--symbol", help="kernel to extract (default: the first one)")
    ap.add_argument(
        "--llvm-bin",
        default=default_llvm_bin(),
        help=f"directory with llvm-objdump etc. (default: {default_llvm_bin()})",
    )
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="emit only .text (for the llvm-objcopy --update-section flow)",
    )
    ap.add_argument(
        "--list", action="store_true", help="list kernels and labels, then exit"
    )
    ap.add_argument("-o", "--output", help="output .s file (default: stdout)")
    args = ap.parse_args()

    tools = Tools(args.llvm_bin)
    mcpu = args.mcpu or detect_mcpu(tools, args.co)
    kernels, labels = read_symbols(tools, args.co)
    if not kernels:
        sys.exit("error: no kernel (FUNC) symbols in .text")

    if args.list:
        print(f"arch: {mcpu}")
        for addr, name in kernels:
            print(f"kernel 0x{addr:x} {name}")
        for addr in sorted(labels):
            print(f"label  0x{addr:x} {' = '.join(labels[addr])}")
        return

    names = [n for _, n in kernels]
    kernel = args.symbol or names[0]
    if kernel not in names:
        sys.exit(f"error: kernel {kernel} not found; available: {', '.join(names)}")
    if len(names) > 1:
        print(
            f"note: {args.co} contains {len(names)} kernels, extracting {kernel}",
            file=sys.stderr,
        )

    out = [
        f'\t.amdgcn_target "amdgcn-amd-amdhsa--{mcpu}"',
        "\t.text",
        f"\t.globl {kernel}",
        "\t.p2align 8",
        f"\t.type {kernel},@function",
    ]
    out += extract_text(tools, args.co, mcpu, kernel, labels)
    out.append(f"\t.size {kernel}, .-{kernel}")
    if not args.text_only:
        out += [""] + extract_kernel_descriptor(tools, args.co, mcpu, kernel)
        out += [""] + extract_metadata(tools, args.co)

    text = "\n".join(out) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
