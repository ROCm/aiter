# AGENTS.md -- LLM Wiki Schema for aiter + Composable Kernel

This file tells LLM agents how to maintain the aiter developer wiki.
The wiki is an LLM-generated, LLM-maintained knowledge base that serves
as a quick reference for APIs, configs, backend choices, and operator behavior.

## Architecture

```
aiter/
  wiki/                     # LLM-generated markdown (this is the wiki)
    index.md                # Master catalog of all pages
    log.md                  # Chronological record of ingests/updates
    overview.md             # High-level architecture map
    operators/              # One page per operator family
    concepts/               # Cross-cutting topics (backends, tuning, etc.)
    integration/            # Framework integration (vLLM, etc.)
    ck/                     # Composable Kernel internals
  AGENTS.md                 # This file (schema + conventions)
```

The **raw sources** are the codebase itself: `aiter/`, `csrc/`, `3rdparty/composable_kernel/`,
and `docs/`. The LLM reads from these but never modifies them as part of wiki operations.

## Page Format

Every wiki page uses this structure:

```markdown
---
title: "Page Title"
last_verified: YYYY-MM-DD
source_files:
  - path/relative/to/aiter/root
tags: [tag1, tag2]
---

# Page Title

## Overview
Brief description.

## [Content sections]
Detailed content with [[wikilinks]] to related pages.

## Source Files
- `path/to/file.py` -- brief note on what it contains
```

### Conventions

- **Frontmatter** is YAML with `title`, `last_verified`, `source_files`, and `tags`.
- **`last_verified`** is the date the page was last checked against source code.
- **`source_files`** lists the code files this page synthesizes from (relative to repo root).
- **Cross-references** use `[[wikilinks]]` (Obsidian-compatible): `[[operators/gemm]]`, `[[concepts/backend-selection]]`.
- **Do not duplicate code.** Reference file paths and function names instead.
- **Tags** use lowercase, hyphenated: `attention`, `triton`, `ck`, `gemm`, `fp8`, `moe`, etc.

## Workflows

### Ingest (adding or updating a wiki page from source)

1. Read the relevant source files (Python API, C++ headers, tests, docs).
2. Write or update the wiki page following the page format above.
3. Update `wiki/index.md` -- add/update the entry with a one-line summary.
4. Append an entry to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] ingest | Page Title
   Updated wiki/path/to/page.md from [list of source files].
   Summary of changes.
   ```

### Query (answering a developer question)

1. Read `wiki/index.md` to find relevant pages.
2. Read those pages for the answer.
3. If the answer requires synthesis across pages, provide it directly.
4. If the answer is valuable and reusable, offer to save it as a new wiki page.

### Lint (health-check the wiki)

1. For each wiki page, check `last_verified` -- flag pages older than 30 days.
2. Verify `source_files` still exist and haven't changed significantly.
3. Look for orphan pages (no inbound `[[wikilinks]]` from other pages).
4. Look for missing pages (concepts mentioned but lacking their own page).
5. Report findings and offer to fix.

## Operator Page Guidelines

Each operator page in `wiki/operators/` should cover:

- **What it does** (one paragraph).
- **Python API** -- key function signatures and where they live.
- **Backend variants** -- which backends (Triton, CK, ASM) implement this op and when each is used.
- **Configuration** -- tuning configs, env flags, compile-time parameters.
- **Data types** -- supported input/output dtypes.
- **Related operators** -- wikilinks to fused variants or dependent ops.

## Concept Page Guidelines

Concept pages in `wiki/concepts/` explain cross-cutting concerns:

- **What it is** and why it matters.
- **How it works** in aiter specifically.
- **Which operators** are affected (wikilinks).
- **Configuration knobs** -- env vars, build flags, config files.

## CK Page Guidelines

Pages in `wiki/ck/` explain Composable Kernel internals relevant to aiter:

- **Architecture concepts** (tile model, template layers).
- **Specific pipelines** (attention, GEMM) with references to CK header files.
- Keep focused on what aiter developers need to know, not full CK documentation.

## Environment Variables Reference

These env vars affect aiter behavior and should be mentioned in relevant pages:

- `AITER_LOG_LEVEL` -- logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `AITER_LOG_MORE` -- enable detailed log formatting
- `AITER_LOG_TUNED_CONFIG` -- log which tuned config was selected for GEMM ops
- `VLLM_ROCM_USE_AITER` -- enable aiter operators in vLLM (integration flag)
