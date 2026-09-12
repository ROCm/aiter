#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def extract_markdown_blocks(path: Path):
    """
    Extract markdown blocks from a log file.
    The blocks are defined as:
    [aiter] <operator> summary (markdown):
    | ... |
    | ... |
    ...
    """

    start_pattern = re.compile(r"^\[aiter\]\s+.*summary\s*\(markdown\):")
    table_line_pattern = re.compile(r"^\|")
    blocks = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        in_block = False
        current_block = []

        for line in f:
            stripped = line.rstrip("\n")

            if not in_block:
                if start_pattern.match(stripped):
                    in_block = True
                    current_block = [stripped]
                continue
            else:
                if table_line_pattern.match(stripped):
                    current_block.append(stripped)
                    continue
                else:
                    blocks.append(current_block)
                    in_block = False
                    current_block = []

        if in_block and current_block:
            blocks.append(current_block)

    return blocks


def main():
    if len(sys.argv) < 2:
        print("Usage: collect_logs.py <log_file>", file=sys.stderr)
        sys.exit(1)

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    blocks = extract_markdown_blocks(log_path)

    for i, block in enumerate(blocks):
        for line in block:
            print(line)
        if i != len(blocks) - 1:
            print()


if __name__ == "__main__":
    main()
