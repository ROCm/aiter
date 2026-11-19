"""Conservative s104/s105 gate for emitted AMD assembly or llvm-objdump text.

Reject every reference to reserved registers (including reads), ranges that
overlap them, flat-scratch aliases, and indirect calls whose clobbers cannot be
established locally. Apply to the actual code object selected for deployment.
Compiler register-count hints alone are not acceptance evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

REGISTER = re.compile(r"\bs(?:(\d+)\b|\[(\d+)\s*:\s*(\d+)\])")
ALIASES = re.compile(r"\bflat_scratch(?:_lo|_hi)?\b")
CALLS = re.compile(r"\bs_(?:swappc|setpc|call)_b64\b")


def violations(text):
    errors = []
    instructions = 0
    for number, raw in enumerate(text.splitlines(), 1):
        # An older objdump can silently print undecoded instructions as data.
        # Such bytes can include scalar writes; never call that a clean audit.
        if "<unknown>" in raw or re.search(
            r"^\s*\.(?:long|word|byte)\b.*//\s*[0-9A-Fa-f]+:", raw
        ):
            errors.append(f"line {number}: undecoded instruction: {raw.strip()}")
            continue
        line = raw.split("//", 1)[0].split(";", 1)[0].strip()
        if not line or line.startswith((".", "#")) or line.endswith(":"):
            continue
        # LLVM assembly and llvm-objdump both print the instruction mnemonic.
        if not re.search(r"\b(?:s|v|ds|buffer|global|flat|scratch|image)_\w+", line):
            continue
        instructions += 1
        for single, first, last in REGISTER.findall(line):
            lo, hi = (int(single), int(single)) if single else (int(first), int(last))
            if lo <= 105 and hi >= 104:
                errors.append(f"line {number}: reserved SGPR: {line}")
        if ALIASES.search(line) or CALLS.search(line):
            errors.append(f"line {number}: alias or indirect control flow: {line}")
    if not instructions:
        errors.append("no AMD instructions found; refusing a vacuous pass")
    return errors


def select_kernel(text, fragment):
    if fragment is None:
        return text
    # Prefer compiler assembly's function delimiters. Fall back to objdump's
    # symbol headings; a missing or ambiguous selection must never pass.
    functions = re.findall(r"(?m)^\s*\.type\s+(\S+),@function\s*$", text)
    names = [name for name in functions if fragment in name]
    if names:
        blocks = []
        for name in names:
            match = re.search(rf"(?ms)^{re.escape(name)}:.*?^\s*\.Lfunc_end\d+:", text)
            if match is None:
                raise ValueError(f"missing body for {name}")
            blocks.append(match.group())
        return "\n".join(blocks)
    headings = list(re.finditer(r"(?m)^[0-9a-fA-F]+ <([^>]+)>:\s*$", text))
    blocks = [
        text[m.end() : headings[i + 1].start() if i + 1 < len(headings) else len(text)]
        for i, m in enumerate(headings)
        if fragment in m.group(1)
    ]
    if not blocks:
        raise ValueError(f"kernel {fragment!r} not found")
    return "\n".join(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--kernel", help="audit every symbol containing this string")
    args = parser.parse_args()
    blob = args.assembly.read_bytes()
    errors = violations(select_kernel(blob.decode(), args.kernel))
    print(
        json.dumps(
            dict(
                assembly=str(args.assembly),
                kernel=args.kernel,
                sha256=hashlib.sha256(blob).hexdigest(),
                passed=not errors,
                errors=errors,
            ),
            indent=2,
        )
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
