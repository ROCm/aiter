# SPDX-License-Identifier: MIT
# WS2 helper: enumerate the Python dispatch sites that compare an architecture
# string against the exact literal "gfx1250".
#
# On MI450 A0 the runtime identity is "gfx1250-strict", so every one of these
# comparisons is False and the gfx1250 path is silently skipped.  That is a
# capability cliff, not an error: nothing is raised, the kernel selection just
# changes.  Each site needs an explicit A0 decision from its owner.
#
#   python3 ws2_gfx1250_dispatch_sites.py [--update-md]
#
# --update-md rewrites the generated section of
# docs/gfx1250-strict-kernel-inventory.md in place.

import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(REPO, "docs/gfx1250-strict-kernel-inventory.md")
BEGIN = "<!-- BEGIN generated: gfx1250 dispatch sites -->"
END = "<!-- END generated: gfx1250 dispatch sites -->"
TABLE_BEGIN = "<!-- BEGIN generated: gfx1250 table keys -->"
TABLE_END = "<!-- END generated: gfx1250 table keys -->"

# A comparison against the bare literal, excluding the strict spelling.
_COMPARE = re.compile(
    r"""(?P<lhs>[A-Za-z_][A-Za-z0-9_.()\[\]"']*)\s*(?:==|!=)\s*["']gfx1250["']"""
    r"""|["']gfx1250["']\s*(?:==|!=)"""
    r"""|\bin\s*\(\s*["']gfx1250["']"""
    r"""|["']gfx1250["']\s*,?\s*\)\s*$"""
)
_ASSIGN_SOURCES = (
    ("get_gfx", re.compile(r"get_gfx\s*\(")),
    ("triton-arch", re.compile(r"get_arch\s*\(|arch_info|get_current_target")),
    ("gcnArchName", re.compile(r"gcnArchName")),
    ("get_gfx_runtime", re.compile(r"get_gfx_runtime\s*\(")),
)


def _kind_of(line):
    """A runtime comparison, or a constant arch allowlist the literal sits in.

    Both need an A0 decision, but they are different edits: a comparison gains
    a strict spelling, an allowlist gains a strict entry.
    """
    return "compare" if re.search(r"==|!=|\bin\b", line) else "allowlist"


def _source_of(lines, idx, line):
    """Where the architecture string on this line came from."""
    for tag, pat in _ASSIGN_SOURCES:
        if pat.search(line):
            return tag
    # Compared through a local; walk back for its assignment.
    m = re.match(
        r"\s*(?:if|elif|assert|return|and|or)?\s*\(?\s*([A-Za-z_][A-Za-z0-9_]*)", line
    )
    if m:
        var = m.group(1)
        assign = re.compile(rf"\b{re.escape(var)}\s*=[^=]")
        for prev in range(idx - 1, max(idx - 60, -1), -1):
            if assign.search(lines[prev]):
                for tag, pat in _ASSIGN_SOURCES:
                    if pat.search(lines[prev]):
                        return tag
                return "local"
    return "unresolved"


_TABLE_KEY = re.compile(r"""^\s*["']gfx1250["']\s*:""")
_STRICT_KEY = re.compile(r"""^\s*["']gfx1250-strict["']\s*:""")
_TABLE_NAME = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\{\(]|=\s*\{\s*$")


def _enclosing_table(lines, idx):
    """Name of the dict literal this key sits in, and how it is read."""
    for prev in range(idx - 1, max(idx - 40, -1), -1):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{", lines[prev])
        if m:
            return m.group(1)
        if re.search(r"=\s*\{\s*$", lines[prev]):
            m2 = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", lines[prev])
            if m2:
                return m2.group(1)
    return "<inline>"


def _closing_lookup(lines, idx):
    """A lookup chained straight onto the dict literal, e.g. `}.get(arch)`."""
    for nxt in range(idx + 1, min(idx + 40, len(lines))):
        line = lines[nxt]
        if not line.lstrip().startswith("}"):
            continue
        rest = line.lstrip()[1:]
        if re.match(r"\.get\([^)]*,", rest):
            return "get+default"
        if rest.startswith(".get("):
            return "get -> None"
        if rest.startswith("["):
            return "[] -> KeyError"
        return None
    return None


def _lookup_style(text, lines, idx, name):
    """How a missing key behaves: a default, None, or a KeyError."""
    chained = _closing_lookup(lines, idx)
    if chained:
        return chained
    if name == "<inline>":
        return "unresolved"
    if re.search(rf"{re.escape(name)}\.get\([^)]*,", text):
        return "get+default"
    if re.search(rf"{re.escape(name)}\.get\(", text):
        return "get -> None"
    if re.search(rf"{re.escape(name)}\s*\[", text):
        return "[] -> KeyError"
    return "unresolved"


def collect_table_keys():
    """Arch-keyed tables that carry a gfx1250 entry but no strict entry."""
    found = []
    for root, dirs, files in os.walk(os.path.join(REPO, "aiter")):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if not _TABLE_KEY.match(line):
                    continue
                near = lines[max(0, i - 30) : i + 31]
                if any(_STRICT_KEY.match(x) for x in near):
                    continue
                name = _enclosing_table(lines, i)
                found.append(
                    (
                        os.path.relpath(path, REPO),
                        i + 1,
                        name,
                        _lookup_style(text, lines, i, name),
                    )
                )
    return found


def render_table_keys(found):
    rows = [
        "| file | line | table | missing key behaviour | A0 decision | owner |",
        "|------|------|-------|-----------------------|-------------|-------|",
    ]
    for path, line, name, style in sorted(found):
        rows.append(f"| {path} | {line} | `{name}` | {style} | pending | TBD |")
    head = (
        f"Generated by `ws2_gfx1250_dispatch_sites.py`: {len(found)} arch-keyed "
        "tables carry a `gfx1250` entry and no `gfx1250-strict` entry, so A0 "
        "takes whatever the lookup does with a missing key.  `get+default` is "
        "the quiet one: the gfx1250 tuning is dropped for a generic default "
        "with nothing logged."
    )
    return head + "\n\n" + "\n".join(rows)


def collect():
    found = {}
    for root, dirs, files in os.walk(os.path.join(REPO, "aiter")):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
            hits = []
            for i, line in enumerate(lines):
                if "gfx1250-strict" in line or not _COMPARE.search(line):
                    continue
                hits.append((i + 1, _source_of(lines, i, line), _kind_of(line)))
            if hits:
                found[os.path.relpath(path, REPO)] = hits
    return found


def render(found):
    rows = [
        "| file | sites | kind | arch source | lines | A0 decision | owner |",
        "|------|-------|------|-------------|-------|-------------|-------|",
    ]
    for path, hits in sorted(found.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sources = sorted({tag for _, tag, _ in hits})
        kinds = sorted({kind for _, _, kind in hits})
        lines = ", ".join(str(n) for n, _, _ in hits)
        rows.append(
            f"| {path} | {len(hits)} | {'/'.join(kinds)} | {'/'.join(sources)} | "
            f"{lines} | pending | TBD |"
        )
    total = sum(len(h) for h in found.values())
    compares = sum(1 for h in found.values() for _, _, k in h if k == "compare")
    head = (
        f"Generated by `ws2_gfx1250_dispatch_sites.py`: {total} uses of the exact "
        f"literal `gfx1250` across {len(found)} files -- {compares} runtime "
        f"comparisons and {total - compares} entries in constant arch allowlists.  "
        "On A0 the arch string is `gfx1250-strict`, so a comparison is simply "
        "False and an allowlist simply does not contain it: the gfx1250 path is "
        "skipped silently, with no error raised.  Each site needs an explicit "
        "owner decision -- reuse the gfx1250 path on A0, or keep the fallback.\n\n"
        "`arch source` says where the compared string comes from.  `get_gfx` and "
        "`get_gfx_runtime` keep the hyphenated spelling, so those sites are known "
        "to miss.  `triton-arch` reads Triton's own target name, whose spelling on "
        "a strict target is unverified until the toolchain lands; `local` and "
        "`unresolved` need a per-site read."
    )
    return head + "\n\n" + "\n".join(rows)


def _splice(md, begin, end, body, heading):
    block = f"{begin}\n\n{body}\n\n{end}"
    if begin in md and end in md:
        return re.sub(
            re.escape(begin) + r".*?" + re.escape(end), block, md, flags=re.DOTALL
        )
    return md.rstrip("\n") + f"\n\n{heading}\n\n" + block + "\n"


def main():
    body = render(collect())
    table_body = render_table_keys(collect_table_keys())
    if "--update-md" not in sys.argv:
        print(body)
        print()
        print(table_body)
        return
    with open(MD, encoding="utf-8") as f:
        md = f.read()
    md = _splice(
        md, BEGIN, END, body, "## gfx1250 dispatch sites (A0 decision pending)"
    )
    md = _splice(
        md,
        TABLE_BEGIN,
        TABLE_END,
        table_body,
        "## gfx1250 arch-keyed tables (A0 decision pending)",
    )
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"updated {MD}")


if __name__ == "__main__":
    main()
