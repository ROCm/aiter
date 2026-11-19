# SPDX-License-Identifier: MIT
"""Verify persistent GEMM2's emitted control flow keeps remote stores in flight."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


def audit(path: Path) -> dict:
    """Check actual ISA, including whether a store wait belongs to a CFG cycle."""
    text = path.read_text()
    if "_epbuffer_" not in text or "_pw256_" not in text:
        raise ValueError("Expected the persistent256 EP-buffer kernel")
    blocks = {"entry": []}
    current = "entry"
    for line_number, line in enumerate(text.splitlines(), 1):
        label = re.match(r"^(\.LBB\w+):", line)
        if label:
            current = label[1]
            blocks[current] = []
        instruction = re.match(
            r"\s+([sv]_[a-zA-Z0-9_]+|buffer_\w+|tensor_\w+)\s*(.*)", line
        )
        if instruction:
            blocks[current].append((line_number, instruction[1], instruction[2]))
    order = list(blocks)
    edges = {block: set() for block in blocks}
    for i, block in enumerate(order):
        instructions = blocks[block]
        for _, opcode, operands in instructions:
            if opcode == "s_branch" or opcode.startswith("s_cbranch"):
                target = re.search(r"\.LBB\w+", operands)
                if target:
                    edges[block].add(target[0])
        if i + 1 < len(order) and (
            not instructions or instructions[-1][1] not in ("s_branch", "s_endpgm")
        ):
            edges[block].add(order[i + 1])

    # A wait in any strongly connected component with a cycle can execute on a
    # loop backedge. Checking text order alone cannot establish this property.
    sys.setrecursionlimit(max(10000, len(blocks) * 4))
    indices, lowlinks, stack, on_stack, cyclic = {}, {}, [], set(), set()

    def visit(block):
        indices[block] = lowlinks[block] = len(indices)
        stack.append(block)
        on_stack.add(block)
        for target in edges[block]:
            if target not in indices:
                visit(target)
                lowlinks[block] = min(lowlinks[block], lowlinks[target])
            elif target in on_stack:
                lowlinks[block] = min(lowlinks[block], indices[target])
        if lowlinks[block] == indices[block]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == block:
                    break
            if len(component) > 1 or block in edges[block]:
                cyclic.update(component)

    for block in blocks:
        if block not in indices:
            visit(block)
    counts = Counter(opcode for rows in blocks.values() for _, opcode, _ in rows)
    waits = [
        {"line": line, "block": block, "in_cycle": block in cyclic}
        for block, rows in blocks.items()
        for line, opcode, _ in rows
        if opcode == "s_wait_storecnt"
    ]
    exits = [
        block
        for block, rows in blocks.items()
        if any(opcode == "s_endpgm" for _, opcode, _ in rows)
    ]
    result = {
        "exit_blocks": exits,
        "assembly": str(path),
        "buffer_stores": sum(
            n for op, n in counts.items() if op.startswith("buffer_store")
        ),
        "tensor_stores": sum(
            n for op, n in counts.items() if op.startswith("tensor_store")
        ),
        "store_waits": waits,
        "xcnt_waits": counts["s_wait_xcnt"],
        "xcnt_scope": "XNACK address-translation operand protection, not store completion",
        "xcnt_implicit_drain_sites": sum(
            n for op, n in counts.items()
            if op == "s_branch" or op.startswith("s_cbranch")
            or op.startswith("s_barrier") or op in ("s_endpgm", "s_sendmsg")
        ),
        "xcnt_note": "Branches and barriers also drain XCNT implicitly; static sites are not dynamic stall counts",
        "xcnt_reference": "https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp",
        "cyclic_blocks": len(cyclic),
        "wait_instructions": {
            op: n for op, n in counts.items() if op.startswith("s_wait")
        },
    }
    if not result["buffer_stores"] or result["tensor_stores"]:
        raise ValueError(f"Wrong output path: {result}")
    if len(waits) != 1 or waits[0]["in_cycle"] or exits != [waits[0]["block"]]:
        raise ValueError(
            f"Output wait blocks overlap or exit drain is missing: {result}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.assembly), indent=2))


if __name__ == "__main__":
    main()
