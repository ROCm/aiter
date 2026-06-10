#!/usr/bin/env python3
"""Generate a CFG graph (PNG) from an LLVM MIR dump file.

Usage:
    python mir_cfg.py <input.txt> [output.png]

If output is omitted, it defaults to <input>_cfg.png.
Requires graphviz (`dot`) to be installed.
"""

import re
import sys
import subprocess
import tempfile
from collections import OrderedDict


def parse_mir_cfg(mir_path):
    """Parse basic block labels and successor edges from a MIR file."""
    blocks = OrderedDict()  # bb_id -> {"label": str, "successors": [(bb_id, pct)], "predecessors": [bb_id]}

    bb_header_re = re.compile(
        r'^\d+B\s+bb\.(\d+)\s*(.*?):?\s*$'
        r'|'
        r'^\d+B\s+bb\.(\d+)\.([\w.]+)\s*(.*?):?\s*$'
    )
    # More robust: just look for "bb.<N>" at start of a line (with the address prefix)
    bb_start_re = re.compile(r'^\d+B\tbb\.(\d+)\s*(.*?)\s*:\s*$')
    succ_re = re.compile(r'successors:\s*(.*)')
    pred_re = re.compile(r'predecessors:\s*(.*)')
    succ_entry_re = re.compile(r'%bb\.(\d+)\(([^)]+)\)')
    succ_pct_re = re.compile(r'%bb\.(\d+)\(([\d.]+)%\)')

    current_bb = None

    with open(mir_path, 'r') as f:
        for line in f:
            line = line.rstrip()

            # Match bb header: "0B\tbb.0 (%ir-block.22):" or "560B\tbb.1.common.ret:"
            m = bb_start_re.match(line)
            if m:
                bb_id = int(m.group(1))
                rest = m.group(2).strip()
                # Extract a human-readable label
                label = ""
                if rest:
                    # Could be "(%ir-block.22)" or ".common.ret" etc
                    label = rest
                current_bb = bb_id
                blocks[bb_id] = {"label": label, "successors": [], "predecessors": []}
                continue

            # Also match the "bb.N.name" form like "560B	bb.1.common.ret:"
            m2 = re.match(r'^(\d+)B\tbb\.(\d+)\.(\S+)\s*(.*?)\s*:?\s*$', line)
            if m2:
                bb_id = int(m2.group(2))
                name = m2.group(3)
                current_bb = bb_id
                blocks[bb_id] = {"label": name, "successors": [], "predecessors": []}
                continue

            if current_bb is not None:
                # Check for successors
                sm = succ_re.search(line)
                if sm:
                    succ_str = sm.group(1)
                    # First try to get percentages from the human-readable part after ";"
                    pct_matches = succ_pct_re.findall(succ_str)
                    if pct_matches:
                        for sid, pct in pct_matches:
                            blocks[current_bb]["successors"].append((int(sid), pct + "%"))
                    else:
                        # Fall back to hex weights
                        hex_matches = succ_entry_re.findall(succ_str)
                        for sid, weight in hex_matches:
                            try:
                                w = int(weight, 16)
                                pct = f"{w / 0x80000000 * 100:.1f}%"
                            except ValueError:
                                pct = ""
                            blocks[current_bb]["successors"].append((int(sid), pct))

                # Check for predecessors
                pm = pred_re.search(line)
                if pm:
                    pred_str = pm.group(1)
                    for pid in re.findall(r'%bb\.(\d+)', pred_str):
                        blocks[current_bb]["predecessors"].append(int(pid))

    return blocks


def detect_loops(blocks):
    """Detect back-edges (bb that has itself as a successor, or successor with lower id that's reachable)."""
    loop_edges = set()
    for bb_id, info in blocks.items():
        for succ_id, _ in info["successors"]:
            if succ_id <= bb_id:
                loop_edges.add((bb_id, succ_id))
    return loop_edges


def generate_dot(blocks, title="CFG"):
    """Generate DOT source from parsed blocks."""
    loop_edges = detect_loops(blocks)

    # Identify entry and exit blocks
    entry_blocks = set()
    exit_blocks = set()
    all_bb_ids = set(blocks.keys())

    for bb_id, info in blocks.items():
        if not info["predecessors"] and bb_id == min(all_bb_ids):
            entry_blocks.add(bb_id)
        if not info["successors"]:
            exit_blocks.add(bb_id)

    lines = [
        'digraph CFG {',
        '    rankdir=TB;',
        '    node [shape=box, style="rounded,filled", fillcolor="#e8f0fe", fontname="Courier", fontsize=11];',
        '    edge [fontname="Courier", fontsize=9];',
        '',
    ]

    for bb_id, info in blocks.items():
        label_parts = [f"bb.{bb_id}"]
        if info["label"]:
            short = info["label"]
            # Truncate long labels
            if len(short) > 30:
                short = short[:27] + "..."
            label_parts.append(short)

        label = "\\n".join(label_parts)

        if bb_id in entry_blocks:
            color = "#d4edda"  # green - entry
        elif bb_id in exit_blocks:
            color = "#f8d7da"  # red - exit
        elif any((bb_id, bb_id) == e for e in loop_edges) or any(s == bb_id for s, _ in info["successors"]):
            color = "#fff3cd"  # yellow - loop
        else:
            color = "#e8f0fe"  # blue - default

        lines.append(f'    bb{bb_id} [label="{label}", fillcolor="{color}"];')

    lines.append('')

    for bb_id, info in blocks.items():
        for succ_id, pct in info["successors"]:
            attrs = []
            if pct:
                attrs.append(f'label="{pct}"')
            if (bb_id, succ_id) in loop_edges:
                attrs.append('color="red"')
                attrs.append('style="bold"')
            attr_str = f' [{", ".join(attrs)}]' if attrs else ''
            lines.append(f'    bb{bb_id} -> bb{succ_id}{attr_str};')

    lines.append('}')
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_mir.txt> [output.png]", file=sys.stderr)
        sys.exit(1)

    mir_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base = mir_path.rsplit('.', 1)[0] if '.' in mir_path else mir_path
        out_path = base + "_cfg.png"

    blocks = parse_mir_cfg(mir_path)
    print(f"Parsed {len(blocks)} basic blocks: {', '.join(f'bb.{b}' for b in blocks)}")

    dot_source = generate_dot(blocks, title=mir_path)

    # Write .dot file alongside output
    dot_path = out_path.rsplit('.', 1)[0] + ".dot"
    with open(dot_path, 'w') as f:
        f.write(dot_source)
    print(f"Wrote DOT: {dot_path}")

    # Render with graphviz
    try:
        subprocess.run(['dot', '-Tpng', dot_path, '-o', out_path], check=True)
        print(f"Wrote PNG: {out_path}")
    except FileNotFoundError:
        print("Error: 'dot' (graphviz) not found. Install with: apt install graphviz", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running dot: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
