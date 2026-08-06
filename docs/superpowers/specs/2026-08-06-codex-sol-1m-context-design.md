# Codex GPT-5.6 Sol 1M Context Override Design

Date: 2026-08-06

## Objective

Keep `gpt-5.6-sol` and the `custom-gateway` provider while raising the raw
context window used by new Codex sessions from 272,000 tokens to 1,000,000
tokens. Codex reserves 5% of the raw window, so the expected runtime-reported
usable context is 950,000 tokens.

## Current Behavior and Root Cause

The active user configuration already requests:

- `model = "gpt-5.6-sol"`
- `model_context_window = 1000000`
- `model_auto_compact_token_limit = 950000`

The active Codex model catalog declares both `context_window` and
`max_context_window` as 272,000 for `gpt-5.6-sol`. Codex resolves a configured
window as `min(model_context_window, max_context_window)`, which clamps the raw
window to 272,000. It then applies the catalog's 95% usable-window factor, so
the current session reports 258,400 tokens.

The current session rollout confirms `model_context_window: 258400`.

## Selected Design

Create an authoritative local model catalog at:

`/root/.codex/model-catalogs/gpt-5.6-sol-1m.json`

The file will be a mechanically generated snapshot of the complete active
Codex model catalog. Every model and every field will remain unchanged except
for these two fields on the `gpt-5.6-sol` entry:

```json
"context_window": 1000000,
"max_context_window": 1000000
```

Using the complete catalog avoids removing other model-picker entries or
dropping Sol-specific tool, reasoning, instruction, and service-tier metadata.

Update `/root/.codex/config.toml` to load that catalog at startup:

```toml
model_catalog_json = "/root/.codex/model-catalogs/gpt-5.6-sol-1m.json"
```

Keep the existing model, provider, and raw-window selection. Set
`model_auto_compact_token_limit` to 900,000 so the configuration states the
effective behavior directly. Codex clamps auto-compaction to 90% of the raw
window even if a larger value is configured.

The custom catalog is startup-only. It cannot expand the currently running
thread; the change applies to a newly started Codex process and new thread.

## Scope

In scope:

- A global Codex model-catalog override under `/root/.codex`.
- The minimum related edits to `/root/.codex/config.toml`.
- Local catalog validation, new-process runtime validation, and a staged
  gateway request above the previous 272,000-token raw limit.
- Clear rollback if catalog parsing or gateway validation fails.

Out of scope:

- Switching away from `gpt-5.6-sol`.
- Modifying the AMD gateway or its deployed model.
- Patching the installed Codex binary or bundled resources.
- Claiming that the gateway supports the full 1M boundary without a
  near-boundary request.
- Expanding the already-running thread in place.

## Validation

Validation proceeds in stages:

1. Parse the generated JSON and assert that it contains the same model slugs
   as the active catalog.
2. Compare the source and generated catalogs and assert that only Sol's
   `context_window` and `max_context_window` changed.
3. Run Codex configuration diagnostics and render the effective raw catalog to
   detect schema or startup errors.
4. Start an isolated new Codex process and confirm its runtime event reports
   `model_context_window: 950000` while the model remains `gpt-5.6-sol` and the
   provider remains `custom-gateway`.
5. Run a normal small gateway request as a smoke test.
6. Run one controlled request targeted at 300,000 input tokens. Confirm from
   the resulting runtime token-usage event that the request contained more
   than 272,000 input tokens and completed without a context-length error.
   Abort the test before submission if its local size estimate exceeds 500,000
   tokens. Success proves that the client and gateway can operate beyond the
   previous limit, but it is not a full proof of the gateway's 1M boundary.

The large-input validation will avoid printing its generated payload or any
credential material. It will record only size, exit status, response marker,
and any redacted error category.

## Failure Handling and Rollback

Before editing, record a hash of the configuration and the exact affected
values without duplicating unrelated configuration or credential material.
If the catalog fails to parse, Codex fails to start, the selected model/provider
changes, or the gateway rejects the above-272k validation with a context-length
error:

1. Remove the `model_catalog_json` selection from the active configuration.
2. Restore `model_auto_compact_token_limit = 950000`.
3. Re-run configuration diagnostics and a small request.
4. Report that the client-side override could not establish a larger usable
   backend window; do not claim that 1M is active.

The generated catalog may remain as an inert diagnostic artifact after
rollback, but it will not affect Codex unless selected by configuration.

## Operational Caveats

- A local catalog changes what Codex permits and attempts to send; it cannot
  increase the AMD gateway's physical model capacity.
- New upstream catalog metadata will not automatically merge into this static
  snapshot. After a Codex upgrade, regenerate and revalidate the catalog before
  relying on new model metadata.
- The expected displayed usable capacity is 950,000 tokens, while automatic
  compaction begins at 900,000 tokens.
- Repeated compaction remains lossy for old conversational detail even with the
  larger window.

## Success Criteria

- The active default remains `gpt-5.6-sol` through `custom-gateway`.
- A fresh Codex session reports a 950,000-token usable context window.
- A controlled request above the old 272,000-token raw limit succeeds without
  a context-length error.
- No project source files or unrelated user changes are modified.
- If any validation stage fails, the active configuration is restored to the
  original values and a small post-rollback request succeeds.
