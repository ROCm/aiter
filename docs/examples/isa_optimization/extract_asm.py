#!/usr/bin/env python3
"""Extract reassemblable .s from llvm-objdump -d output.

Handles branch label resolution using word-offset addressing:
  target_address = base_address + label_value * 4

Usage:
  # Step 1: Disassemble a .co file
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx942 kernel.co > kernel.isa

  # Step 2: Find the kernel symbol
  grep "^[0-9a-f]" kernel.isa | head -1
  # Example: 0000000000001000 <_ZN5aiter...E>:

  # Step 3: Extract reassemblable .s
  python3 extract_asm.py kernel.isa _ZN5aiter...E > kernel.s

  # Step 4: Recompile
  /opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa \\
      -mcpu=gfx942 -o kernel_recompiled.co kernel.s
"""
import re
import sys
import argparse


def extract(isa_path: str, kernel_symbol: str, target: str) -> str:
    """Parse llvm-objdump output and emit a reassemblable .s file."""
    with open(isa_path) as f:
        lines = f.readlines()

    # Find kernel section
    section_start = None
    for i, line in enumerate(lines):
        if f"<{kernel_symbol}>:" in line:
            section_start = i
            break
    if section_start is None:
        print(f"Error: kernel symbol '{kernel_symbol}' not found in {isa_path}",
              file=sys.stderr)
        # List available symbols
        symbols = []
        for line in lines:
            m = re.match(r"[0-9a-fA-F]+ <(.+)>:", line)
            if m:
                symbols.append(m.group(1))
        if symbols:
            print(f"Available symbols:", file=sys.stderr)
            for s in symbols:
                print(f"  {s}", file=sys.stderr)
        sys.exit(1)

    # Parse base address from first instruction
    first_instr_line = lines[section_start + 1].strip()
    base_addr = int(first_instr_line.split(":")[0].strip(), 16)

    # Collect instructions
    instructions = []
    for line in lines[section_start + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("Disassembly"):
            break
        # Match: "  addr: hex_bytes  instruction"
        m = re.match(r"\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]+\s+)+(.+)", stripped)
        if m:
            addr = int(m.group(1), 16)
            instr = m.group(2).strip()
            # Remove trailing hex comments (e.g., "// 000000001234")
            instr = re.sub(r"\s*//\s*[0-9A-Fa-f]+$", "", instr)
            instructions.append((addr, instr))

    if not instructions:
        print(f"Error: no instructions found for '{kernel_symbol}'", file=sys.stderr)
        sys.exit(1)

    # Resolve branch labels (word offset: label_val * 4 + base_addr)
    branch_targets = {}
    for addr, instr in instructions:
        m = re.search(r"label_([0-9A-Fa-f]+)", instr)
        if m:
            label_val = int(m.group(1), 16)
            target_addr = base_addr + label_val * 4
            branch_targets[target_addr] = f"label_{m.group(1)}"

    # Emit .s file
    output = []
    output.append(f'  .amdgcn_target "{target}"')
    output.append(f"  .globl {kernel_symbol}")
    output.append(f"  .type {kernel_symbol}, @function")
    output.append(f"{kernel_symbol}:")

    for addr, instr in instructions:
        if addr in branch_targets:
            output.append(f"{branch_targets[addr]}:")
        output.append(f"  {instr}")

    output.append(f"  .size {kernel_symbol}, .-{kernel_symbol}")

    # Summary stats
    n_mfma = sum(1 for _, i in instructions if "v_mfma_" in i)
    n_buf = sum(1 for _, i in instructions if "buffer_load" in i)
    n_lds = sum(1 for _, i in instructions if i.startswith("ds_"))
    n_branch = len(branch_targets)
    print(f"Extracted {len(instructions)} instructions "
          f"(MFMA={n_mfma}, buf_load={n_buf}, LDS={n_lds}, branches={n_branch})",
          file=sys.stderr)

    return "\n".join(output)


def list_symbols(isa_path: str) -> list[str]:
    """List all kernel symbols in an llvm-objdump output file."""
    symbols = []
    with open(isa_path) as f:
        for line in f:
            m = re.match(r"[0-9a-fA-F]+ <(.+)>:", line)
            if m:
                symbols.append(m.group(1))
    return symbols


def main():
    parser = argparse.ArgumentParser(
        description="Extract reassemblable .s from llvm-objdump output")
    parser.add_argument("isa_file", help="llvm-objdump -d output file")
    parser.add_argument("kernel_symbol", nargs="?",
                        help="Kernel symbol name (omit to list available symbols)")
    parser.add_argument("--target", default="amdgcn-amd-amdhsa--gfx942",
                        help="AMDGCN target triple (default: amdgcn-amd-amdhsa--gfx942)")
    parser.add_argument("-o", "--output", help="Output .s file (default: stdout)")
    args = parser.parse_args()

    if not args.kernel_symbol:
        symbols = list_symbols(args.isa_file)
        print("Available kernel symbols:")
        for s in symbols:
            print(f"  {s}")
        return

    result = extract(args.isa_file, args.kernel_symbol, args.target)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
