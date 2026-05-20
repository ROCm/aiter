#!/usr/bin/env python3
"""Build payload for Confluence update from confluence_page.md (use with MCP confluence_update_page)."""

from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent
PAGE_ID = "1690335375"
TITLE = "Benchmark CK_PR_6764"


def main() -> None:
    content = (BENCH / "confluence_page.md").read_text(encoding="utf-8")
    payload = {
        "page_id": PAGE_ID,
        "title": TITLE,
        "content": content,
        "content_format": "markdown",
        "version_comment": "ck_pr_6764 + jax_unfused tables; graph screenshot headings",
    }
    out = BENCH / "confluence_update_payload.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out} ({len(content)} chars)")


if __name__ == "__main__":
    main()
