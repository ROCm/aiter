---
name: aiter-kernel-provider
description: Port an out-of-tree kernel library onto AITER's provider interface — declare the entry point, implement select()/run() against a versioned op contract, decline cheaply, and verify the port without a GPU. Use whenever adding, debugging or reviewing a kernel plugin that serves an AITER op (aiter.plugin, entry-point group aiter.plugins, RFC #7240).
argument-hint: [kernel library name, or the AITER op you want to serve]
---

# Serving an AITER op from an out-of-tree kernel library

AITER owns the op contract; your library implements it or declines. No fork, no
monkeypatch. `aiter/plugin/README.md` is the prose; this is the procedure.

**Ownership, because every rule below follows from it:** AITER owns signatures,
versions, execution rules and provenance. You own your kernels, your evidence,
and the per-call decision to serve. The framework — not you, not the
environment — decides whether you are on.

## The port, in order

**1. Read the contract from the baseline, not from the source.**
`tests/plugin_contracts.json` holds the committed signature of every routed op,
and the contract gate fails any PR that moves one without a version bump — so
it is the one description that cannot drift from what dispatch actually hands
you. **If your op is not in that file it is not routed**, and there is nothing
to serve yet; `plugin.status()['routed_ops']` says the same at runtime.

```json
"rmsnorm": {"signature": "input:positional, weight:positional, epsilon:positional, ...",
            "version": 2, "supersedes": []}
```

**2. Declare the entry point.** This one line is the whole discovery mechanism:

```toml
[project.entry-points."aiter.plugins"]
mylib = "mylib.aiter_provider:provider"
```

The target may be an instance or a zero-argument factory. Anything with a
`select` attribute is treated as an instance.

**3. One selector per contract, unpacking by name.**

```python
_SELECTORS = {"rmsnorm": _select_rmsnorm, "rmsnorm_with_add": _select_rmsnorm_with_add}

class Provider:
    name = "mylib"
    api_versions = {plugin.API_VERSION}
    op_versions = set(_SELECTORS)          # derive it; a hand-written list drifts

    def select(self, op, args, kwargs):
        selector = _SELECTORS.get(op.name)
        return selector(args) if selector else None

def _select_rmsnorm(args):
    value, weight, epsilon, _sensitive = args   # AITER applies defaults first
    if my_rms_norm.fallback_reason(value, weight, epsilon):
        return None                             # declining is normal
    return plugin.KernelCandidate(
        provider="mylib", kernel_id="rms_norm@1.0",
        artifact_digest="sha256:...", run=_run_rmsnorm)
```

AITER canonicalizes the call and applies the contract's defaults **before**
asking, so `args` always holds the full positional list. Unpack it by name. A
contract that grows a parameter then fails loudly here — recorded as a select
error, routed native — instead of reading the wrong index quietly.

**4. Bind `run` to the contract, not to the layout.** See rule 1 below.

## Hard rules

1. **Bind to the contract, not the layout.** A real case: a provider offered its
   `bpreshuffle` kernel for the `blockscale` contract, because the shuffled
   layout is a flag *inside* that contract and also a separate public entry of
   its own. Seven arguments were about to reach a six-argument kernel. Pick the
   `run` by the op name you are answering for, never by a layout flag.

2. **Arity is not meaning.** The candidate check compares parameter counts, so
   two contracts with the same arity and different semantics both pass it:

   ```
   mla_sparse_prefill_v4 :  15 parameters, 2nd = q_rope
   mla_sparse_prefill_v41:  15 parameters, 2nd = q_scale
   ```

   A kernel bound to the wrong one reads a scale tensor as a RoPE component,
   silently, with every gate green. One op name, one contract, one `run`.

3. **Match hardware on `gcnArchName`, never on the device name.** On MI355X
   under `rocm/atom-dev` with torch 2.10.0+rocm7.2.4:

   ```
   torch.cuda.get_device_name(0)  ->  'AMD Radeon Graphics'
   gcnArchName                    ->  'gfx950:sramecc+:xnack-'
   ```

   A provider matching by device name loaded zero kernels and declined every
   call — the interface looks broken while behaving exactly as specified.

4. **`select()` returns, it does not raise, and it is pure.** `explain()` runs
   `select()` without executing anything; side effects there make the dry run a
   lie. Raising is isolated and recorded, but it costs the call. `None` is the
   supported no.

5. **Once `run()` starts, errors propagate.** AITER will not fall back
   mid-flight: a retry could repeat side effects or hide a broken kernel behind
   a correct-looking result. Fail loudly inside `run`.

6. **Report only numbers you can attribute.** Offline validation timings were
   measured on shapes you chose, often as a ratio. AITER labels what it records
   with the conditions of the call in front of it, so such a number arrives
   attached to a shape it never ran on. Omit it; a number that looks sourced and
   is not is harder to catch than a missing one.

7. **Declining is not failing.** Cannot serve this dtype, shape or arch? Return
   `None`, cheaply, every time. `explain()` shows the decline and
   `decision_counts()` counts it. Neither is an error state.

## Verifying the port without a GPU

**Cross-check every selector's arity against the committed baseline.** This
catches the whole class of rule-2 defects before any hardware is involved:

```python
import ast, json
base = json.load(open("tests/plugin_contracts.json"))
arity = {k: len([p for p in v["signature"].split(" -> ")[0].split(", ") if p.strip()])
         for k, v in base.items()}
tree = ast.parse(open("mylib/aiter_provider.py").read())
# for each selector: the tuple it unpacks must have len == arity[op]
```

**Then ask the interface, not the kernel:**

```python
plugin.configure(source="port check", enable=["mylib"])
plugin.status()                      # found? enabled? which ops are routed?
plugin.explain("rmsnorm", *args)     # would-be decision + why each provider declined
```

`status()['skipped_providers']` carries the reason a wheel was rejected
(unsupported `api_versions`, bad metadata). `explain()` is a dry run of gates,
eligibility and selection that never calls `run()`.

After a real call: `last_decision()`, `decision_counts()`, `on_decision(cb)`.
`last_decision()` returns `None` while nothing is enabled or under
`AITER_PLUGINS=0` — those states record nothing, and a stale answer would be
worse than none.

## Enablement and the kill switch

Installing a provider does nothing. Only a framework calling
`plugin.configure(source=..., enable=[...])` turns one on; `enable` is
preference order and `source` is required provenance, recorded in every
decision. Nothing in the environment can enable a provider.

`AITER_PLUGINS=0` disables routing process-wide. It only subtracts — and it
does **not** reach a kernel already captured into a CUDA graph. Replay runs no
Python, so whatever was decided at capture is what the graph does for its
lifetime; rebuild the graphs. Call each routed op once outside a capture first.

## Versions

`op_versions` maps an op name to the contract versions you serve; AITER owns
those numbers. A contract change that breaks providers bumps the op version, and
AITER may keep accepting an older one for a stated window (`supersedes`), which
`status()['deprecations']` reports. Declaring a version you do not implement is
caught by the candidate check, not trusted.

`api_versions` is the provider-API level you speak (`plugin.API_VERSION`).
Declare every version you support so one wheel covers several AITER releases.

Design rationale, open questions and the deprecation policy: [RFC #7240](https://github.com/ROCm/aiter/issues/7240).
