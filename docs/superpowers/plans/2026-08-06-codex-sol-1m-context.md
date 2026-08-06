# Codex GPT-5.6 Sol 1M Context Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make new Codex sessions using `gpt-5.6-sol` through `custom-gateway` use a 1,000,000-token raw context declaration, report 950,000 usable tokens, and verify operation beyond the previous 272,000-token raw limit.

**Architecture:** Generate an authoritative full model-catalog snapshot outside the repository, changing only Sol's raw and maximum context fields. Select that startup-only catalog from the global Codex configuration, retain the existing model and provider, and validate both local resolution and a staged gateway request. Restore the original active configuration immediately if parsing, model selection, startup, or context-length validation fails.

**Tech Stack:** Codex CLI 0.146.1, JSON model catalog, TOML configuration, Node.js 24 for mechanical JSON generation and assertions, Bash for isolated verification commands.

---

## File Structure

- Create: `/root/.codex/model-catalogs/gpt-5.6-sol-1m.json` — complete authoritative model catalog with only Sol's two context fields changed.
- Modify: `/root/.codex/config.toml:4-5` — select the catalog and state the effective 900,000-token auto-compaction threshold.
- Reference: `docs/superpowers/specs/2026-08-06-codex-sol-1m-context-design.md` — approved behavior, validation, and rollback contract.
- Create: `docs/superpowers/plans/2026-08-06-codex-sol-1m-context.md` — this execution plan; no project source file is modified.

### Task 1: Establish the Failing Baseline and Generate the Catalog

**Files:**
- Create: `/root/.codex/model-catalogs/gpt-5.6-sol-1m.json`
- Read: `/root/.codex/config.toml:2-5`

- [ ] **Step 1: Verify the current catalog fails the desired 1M assertion**

Run:

```bash
codex debug models | node -e '
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const catalog = JSON.parse(raw);
  const sol = catalog.models.find(model => model.slug === "gpt-5.6-sol");
  if (!sol) throw new Error("gpt-5.6-sol is missing");
  console.log(JSON.stringify({
    context_window: sol.context_window,
    max_context_window: sol.max_context_window,
    effective_context_window_percent: sol.effective_context_window_percent
  }));
  if (sol.max_context_window !== 1000000) process.exit(1);
});
'
```

Expected: exit 1 after printing `context_window: 272000`,
`max_context_window: 272000`, and `effective_context_window_percent: 95`.

- [ ] **Step 2: Refuse to overwrite an existing target**

Run:

```bash
test ! -e /root/.codex/model-catalogs/gpt-5.6-sol-1m.json
```

Expected: exit 0. If the file exists, stop and inspect it instead of replacing
it.

- [ ] **Step 3: Generate a full catalog with an internal two-field diff assertion**

Run:

```bash
mkdir -p /root/.codex/model-catalogs
codex debug models | node -e '
const assert = require("assert");
const crypto = require("crypto");
const fs = require("fs");
const output = "/root/.codex/model-catalogs/gpt-5.6-sol-1m.json";
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const source = JSON.parse(raw);
  const edited = structuredClone(source);
  const sourceMatches = source.models.filter(model => model.slug === "gpt-5.6-sol");
  const editedMatches = edited.models.filter(model => model.slug === "gpt-5.6-sol");
  assert.strictEqual(sourceMatches.length, 1, "source must contain exactly one Sol entry");
  assert.strictEqual(editedMatches.length, 1, "output must contain exactly one Sol entry");
  const sourceSol = sourceMatches[0];
  const editedSol = editedMatches[0];
  assert.strictEqual(sourceSol.context_window, 272000, "unexpected source context_window");
  assert.strictEqual(sourceSol.max_context_window, 272000, "unexpected source max_context_window");
  editedSol.context_window = 1000000;
  editedSol.max_context_window = 1000000;

  const restored = structuredClone(edited);
  const restoredSol = restored.models.find(model => model.slug === "gpt-5.6-sol");
  restoredSol.context_window = sourceSol.context_window;
  restoredSol.max_context_window = sourceSol.max_context_window;
  assert.deepStrictEqual(restored, source, "catalog changed outside the two approved fields");

  const rendered = JSON.stringify(edited, null, 2) + "\n";
  fs.writeFileSync(output, rendered, { flag: "wx", mode: 0o600 });
  console.log(JSON.stringify({
    output,
    models: edited.models.length,
    context_window: editedSol.context_window,
    max_context_window: editedSol.max_context_window,
    sha256: crypto.createHash("sha256").update(rendered).digest("hex")
  }));
});
'
```

Expected: exit 0 and a summary showing both Sol fields as `1000000` without
printing the catalog contents.

- [ ] **Step 4: Validate JSON shape, Sol fields, model count, and file mode**

Run:

```bash
node -e '
const assert = require("assert");
const fs = require("fs");
const path = "/root/.codex/model-catalogs/gpt-5.6-sol-1m.json";
const catalog = JSON.parse(fs.readFileSync(path, "utf8"));
assert.ok(Array.isArray(catalog.models) && catalog.models.length > 1, "full model catalog required");
const sol = catalog.models.find(model => model.slug === "gpt-5.6-sol");
assert.strictEqual(sol.context_window, 1000000);
assert.strictEqual(sol.max_context_window, 1000000);
assert.strictEqual(sol.effective_context_window_percent, 95);
const mode = fs.statSync(path).mode & 0o777;
assert.strictEqual(mode, 0o600);
console.log(JSON.stringify({models: catalog.models.length, mode: mode.toString(8)}));
'
```

Expected: exit 0 with a model count greater than one and mode `600`.

### Task 2: Activate the Catalog in Global Codex Configuration

**Files:**
- Modify: `/root/.codex/config.toml:4-5`
- Read: `/root/.codex/model-catalogs/gpt-5.6-sol-1m.json`

- [ ] **Step 1: Record the configuration preimage without exposing values outside scope**

Run:

```bash
sha256sum /root/.codex/config.toml
rg -n '^(model|model_provider|model_context_window|model_auto_compact_token_limit|model_catalog_json)\s*=' /root/.codex/config.toml
```

Expected: model `gpt-5.6-sol`, provider `custom-gateway`, window `1000000`,
auto-compaction `950000`, no `model_catalog_json`, and one SHA-256 digest.

- [ ] **Step 2: Apply the minimal configuration edit**

Use `apply_patch` with exactly this change:

```diff
-model_auto_compact_token_limit = 950000
+model_auto_compact_token_limit = 900000
+model_catalog_json = "/root/.codex/model-catalogs/gpt-5.6-sol-1m.json"
```

Expected: no model, provider, authentication, permission, or project setting
changes.

- [ ] **Step 3: Verify strict configuration loading and installation health**

Run:

```bash
codex --strict-config doctor --json
```

Expected: exit 0, top-level `overallStatus` equals `ok`, `config.load` equals
`ok`, model remains `gpt-5.6-sol`, and provider remains `custom-gateway`.

- [ ] **Step 4: Verify the active raw catalog is the 1M catalog**

Run:

```bash
codex debug models | node -e '
const assert = require("assert");
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const catalog = JSON.parse(raw);
  const sol = catalog.models.find(model => model.slug === "gpt-5.6-sol");
  assert.strictEqual(sol.context_window, 1000000);
  assert.strictEqual(sol.max_context_window, 1000000);
  assert.strictEqual(sol.effective_context_window_percent, 95);
  console.log(JSON.stringify({
    model: sol.slug,
    raw: sol.context_window,
    usable: sol.context_window * sol.effective_context_window_percent / 100
  }));
});
'
```

Expected: `{"model":"gpt-5.6-sol","raw":1000000,"usable":950000}`.

### Task 3: Verify a Fresh Runtime and Small Gateway Request

**Files:**
- Read: the exact new rollout resolved from `/root/.codex/sessions/**/*.jsonl`
- Do not modify project files.

- [ ] **Step 1: Start a fresh persisted session and assert its runtime metadata**

Run:

```bash
node -e '
const assert = require("assert");
const childProcess = require("child_process");
const fs = require("fs");
const path = require("path");

const run = childProcess.spawnSync("codex", [
  "exec",
  "--json",
  "-C",
  "/root/aiter",
  "Reply with exactly SOL_CONTEXT_SMOKE_OK and do not call tools."
], { encoding: "utf8", maxBuffer: 10 * 1024 * 1024 });
assert.strictEqual(run.status, 0, `codex exec failed: ${run.stderr.slice(0, 2000)}`);
const outputEvents = run.stdout.trim().split("\n").filter(Boolean).map(JSON.parse);
const threadStarted = outputEvents.find(event => event.type === "thread.started");
assert.ok(threadStarted, "thread.started event is missing");
assert.match(threadStarted.thread_id, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
const messages = outputEvents.filter(event => event.type === "item.completed" && event.item?.type === "agent_message");
assert.ok(messages.some(event => event.item.text === "SOL_CONTEXT_SMOKE_OK"), "smoke marker is missing");
assert.ok(outputEvents.some(event => event.type === "turn.completed"), "turn.completed event is missing");

function rolloutFiles(directory) {
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...rolloutFiles(candidate));
    if (entry.isFile() && entry.name.endsWith(".jsonl")) files.push(candidate);
  }
  return files;
}

const matches = rolloutFiles("/root/.codex/sessions").filter(file =>
  fs.readFileSync(file, "utf8").includes(threadStarted.thread_id)
);
assert.strictEqual(matches.length, 1, "smoke-test thread must resolve to one rollout");
const events = fs.readFileSync(matches[0], "utf8").trim().split("\n").map(JSON.parse);
const settings = events.find(event => event.type === "event_msg" && event.payload.type === "thread_settings_applied");
const started = events.find(event => event.type === "event_msg" && event.payload.type === "task_started");
assert.strictEqual(settings.payload.thread_settings.model, "gpt-5.6-sol");
assert.strictEqual(settings.payload.thread_settings.model_provider_id, "custom-gateway");
assert.strictEqual(started.payload.model_context_window, 950000);
console.log(JSON.stringify({
  thread_id: threadStarted.thread_id,
  rollout: matches[0],
  model: settings.payload.thread_settings.model,
  provider: settings.payload.thread_settings.model_provider_id,
  model_context_window: started.payload.model_context_window
}));
'
```

Expected: exit 0 with the validated thread ID and exact rollout path, model
`gpt-5.6-sol`, provider `custom-gateway`, and `model_context_window: 950000`.

### Task 4: Verify Operation Above the Previous Raw Limit

**Files:**
- No persistent input file; generate the payload on a pipe.
- Read only the JSONL event stream emitted by the isolated command.

- [ ] **Step 1: Submit an approximately 300,000-token controlled prompt**

Run with pipeline failure propagation:

```bash
set -o pipefail
node -e '
const marker = "Return exactly SOL_CONTEXT_300K_OK. Do not call tools.\n";
process.stdout.write(marker + "x ".repeat(300000) + "\n" + marker);
' | codex exec --ephemeral --json -C /root/aiter - | node -e '
const assert = require("assert");
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const events = raw.trim().split("\n").filter(Boolean).map(JSON.parse);
  const failed = events.find(event => event.type === "turn.failed" || event.type === "error");
  if (failed) throw new Error(JSON.stringify(failed));
  const completed = events.find(event => event.type === "turn.completed");
  assert.ok(completed, "turn.completed event is missing");
  assert.ok(completed.usage.input_tokens > 272000, "request did not exceed the old raw limit");
  assert.ok(completed.usage.input_tokens < 500000, "request exceeded the approved test bound");
  console.log(JSON.stringify({
    status: "completed",
    input_tokens: completed.usage.input_tokens,
    output_tokens: completed.usage.output_tokens
  }));
});
'
```

Expected: exit 0 with `status: completed`, `input_tokens` greater than 272,000
and less than 500,000, and no prompt payload printed.

- [ ] **Step 2: Follow the success path**

If Task 4 Step 1 succeeds, run:

```bash
rg -n '^(model|model_provider|model_context_window|model_auto_compact_token_limit|model_catalog_json)\s*=' /root/.codex/config.toml
```

Expected: Sol, custom gateway, raw window `1000000`, auto-compaction `900000`,
and the selected 1M catalog path. Record that the gateway is proven above the
old limit but not proven at the full 1M boundary.

- [ ] **Step 3: Follow the context-length failure path only if required**

If catalog parsing, startup, model/provider assertions, or the large request
fails with a context-length error, use `apply_patch` to restore exactly:

```diff
-model_auto_compact_token_limit = 900000
-model_catalog_json = "/root/.codex/model-catalogs/gpt-5.6-sol-1m.json"
+model_auto_compact_token_limit = 950000
```

Then run:

```bash
codex --strict-config doctor --json
codex exec --ephemeral --json -C /root/aiter 'Reply with exactly SOL_CONTEXT_ROLLBACK_OK and do not call tools.'
```

Expected: diagnostics return `overallStatus: ok`, the small request completes,
and no claim is made that 1M is active. Leave the catalog file inert for
inspection; do not delete it automatically.

### Task 5: Final Verification and Handoff

**Files:**
- Read: `/root/.codex/config.toml`
- Read: `/root/.codex/model-catalogs/gpt-5.6-sol-1m.json`
- Read: `docs/superpowers/specs/2026-08-06-codex-sol-1m-context-design.md`

- [ ] **Step 1: Re-run compact local assertions**

Run:

```bash
codex --strict-config doctor --summary --ascii
codex debug models | node -e '
const assert = require("assert");
let raw = "";
process.stdin.on("data", chunk => raw += chunk);
process.stdin.on("end", () => {
  const sol = JSON.parse(raw).models.find(model => model.slug === "gpt-5.6-sol");
  assert.strictEqual(sol.context_window, 1000000);
  assert.strictEqual(sol.max_context_window, 1000000);
  assert.strictEqual(sol.effective_context_window_percent, 95);
  console.log("Sol catalog assertion: PASS");
});
'
```

Expected: doctor reports no failed checks and the catalog assertion prints
`PASS`.

- [ ] **Step 2: Confirm the implementation did not add project changes**

Run:

```bash
git status --short --untracked-files=no
git diff --check
```

Expected: no new tracked project change from implementation. Existing user
changes, if any, remain untouched.

- [ ] **Step 3: Hand off the startup requirement and evidence**

Report all of the following:

- The exact catalog and configuration paths changed.
- Fresh subprocess evidence of `model_context_window: 950000`.
- The measured input-token count from the above-272k gateway request.
- The 900,000-token automatic compaction point.
- The limitation that the current thread remains at 258,400 tokens.
- The required user action: restart Codex and open a new thread to receive the
  larger context window.
