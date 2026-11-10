---
name: aiter-kernel-plugin
description: Port an out-of-tree kernel library onto AITER as a kernel plugin — declare the entry point, implement select()/run() against a versioned op contract, decline cheaply, and verify the port without a GPU. Use whenever adding, debugging or reviewing a kernel plugin that serves an AITER op (aiter.plugin, entry-point group aiter.plugins, RFC #7240).
argument-hint: [kernel library name, or the AITER op you want to serve]
---

# Serving an AITER op from an out-of-tree kernel library

AITER owns the op contract; your library implements it or declines. No fork, no
monkeypatch. `aiter/plugin/README.md` is the prose; this is the procedure.

**Ownership, because every rule below follows from it:** AITER owns signatures,
versions, execution rules and provenance. You own your kernels, your evidence,
and the per-call decision to serve. The framework — not you, not the
environment — decides whether you are on.

**Plugin or provider?** Both, for different things. A **kernel plugin** is what
you ship and register — one wheel, one entry point in the `aiter.plugins` group.
**`KernelProvider`** is the interface that plugin implements: the object with
`select()`. Everything user-facing says plugin (`aiter.plugin`,
`AITER_PLUGINS=0`, `plugin.configure()`); the word provider survives where it
names the interface.

## The port, in order

**1. Read the contract from the baseline, not from the source.**
`tests/plugin_contracts.json` holds the committed signature of every routed op,
and the contract gate fails any PR that moves one without a version bump — so
it is the one description that cannot drift from what dispatch actually hands
you. **If your op is not in that file it is not routed** — there is nothing to
serve, and `plugin.status()['routed_ops']` says the same at runtime.

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
    op_versions = set(_SELECTORS)  # derive it; a hand-written list drifts

    def select(self, op, args, kwargs):
        selector = _SELECTORS.get(op.name)
        return selector(args) if selector else None


def _select_rmsnorm(args):
    value, weight, epsilon, _sensitive = args  # AITER applies defaults first
    if my_rms_norm.fallback_reason(value, weight, epsilon):
        return None  # declining is normal
    return plugin.KernelCandidate(
        provider="mylib",
        kernel_id="rms_norm@1.0",
        artifact_digest="sha256:...",
        run=_run_rmsnorm,
    )
```

AITER canonicalizes the call and applies the contract's defaults **before**
asking, so `args` always holds the full positional list. Unpack it by name. A
contract that grows a parameter then fails loudly here — recorded as a select
error, routed native — instead of reading the wrong index quietly.

**4. Bind `run` to the contract, not to the layout.** See rule 1 below.

## A worked example you can read

AKLR (https://github.com/AMD-AGI/AKLR) is the first library ported onto this
interface, and it is public. Its adapter lives in
[`aklr/plugins/aiter/`](https://github.com/AMD-AGI/AKLR/tree/main/aklr/plugins/aiter):

- **`aklr/plugins/aiter/__init__.py`** — one selector per contract unpacking by
  name, declines routed through the library's own `fallback_reason()`, and the
  run functions bound per contract rather than per layout (rule 1 below is the
  bug that shaped it).
- **`aklr/plugins/aiter/REPRODUCING.md`** — install, enable, serve and verify,
  written so somebody who was not involved can repeat it, with the steps that
  are not possible on a given host stated as such.

Every rule below is something that port either got wrong first or had to design
around. Read the adapter next to this file; the two were written together.

## Hard rules

1. **Bind to the contract, not the layout.** A real case: a provider offered its
   `bpreshuffle` kernel for the `blockscale` contract, because the shuffled
   layout is a flag *inside* that contract and also a separate public entry of
   its own. Seven arguments were about to reach a six-argument kernel. Pick the
   `run` by the op name you are answering for, never by a layout flag.

1. **Arity is not meaning.** The candidate check compares parameter counts, so
   two contracts with the same arity and different semantics both pass it:

   ```
   mla_sparse_prefill_v4 :  15 parameters, 2nd = q_rope
   mla_sparse_prefill_v41:  15 parameters, 2nd = q_scale
   ```

   A kernel bound to the wrong one reads a scale tensor as a RoPE component,
   silently, with every gate green. One op name, one contract, one `run`.

1. **Match hardware on `gcnArchName`, never on the device name.** On MI355X
   under `rocm/atom-dev` with torch 2.10.0+rocm7.2.4:

   ```
   torch.cuda.get_device_name(0)  ->  'AMD Radeon Graphics'
   gcnArchName                    ->  'gfx950:sramecc+:xnack-'
   ```

   A provider matching by device name loaded zero kernels and declined every
   call — the interface looks broken while behaving exactly as specified.

1. **`select()` returns, it does not raise, and it is pure.** `explain()` runs
   `select()` without executing anything; side effects there make the dry run a
   lie. Raising is isolated and recorded, but it costs the call. `None` is the
   supported no.

1. **Once `run()` starts, errors propagate.** AITER will not fall back
   mid-flight: a retry could repeat side effects or hide a broken kernel behind
   a correct-looking result. Fail loudly inside `run`.

1. **Report only numbers you can attribute.** Offline validation timings were
   measured on shapes you chose, often as a ratio. AITER labels what it records
   with the conditions of the call in front of it, so such a number arrives
   attached to a shape it never ran on. Omit it; a number that looks sourced and
   is not is harder to catch than a missing one.

1. **Declining is not failing.** Cannot serve this dtype, shape or arch? Return
   `None`, cheaply, every time. `explain()` shows the decline and
   `decision_counts()` counts it. Neither is an error state.

## Verifying the port without a GPU

**Cross-check every selector's arity against the committed baseline.** This
catches the whole class of rule-2 defects before any hardware is involved:

```python
import ast, json

base = json.load(open("tests/plugin_contracts.json"))
arity = {
    op: len([p for p in c["signature"].split(" -> ")[0].split(", ") if p.strip()])
    for op, c in base.items()
}

tree = ast.parse(open("mylib/aiter_provider.py").read())
fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
table = next(
    n.value
    for n in tree.body
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_SELECTORS"
)

for op_node, fn_node in zip(table.keys, table.values):
    op, fn = op_node.value, fns[fn_node.id]
    unpack = next(
        (
            st
            for st in fn.body
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Tuple)
        ),
        None,
    )
    got = len(unpack.targets[0].elts) if unpack else None
    assert (
        got == arity[op]
    ), f"{op}: contract takes {arity[op]}, {fn.name} unpacks {got}"
print("every selector matches its contract")
```

It reads the selector table, not a list you maintain by hand, so a contract
that grows a parameter fails here on the next run rather than at a customer.

**This is not just advice.** The same check runs in-tree as
`tests/test_plugin_selector_arity.py`, with fixtures proving it catches a
selector short by one. Verified against a real port: AKLR's four selectors pass
against the committed baseline, and shortening one of them to six parameters —
exactly the arity of the *other* GEMM contract, the mistake most likely to slip
through — is reported as
`gemm_a8w8_blockscale: contract takes 7, _select_gemm unpacks 6`.

**Then ask the interface, not the kernel:**

```python
plugin.configure(source="port check", enable=["mylib"])
plugin.status()  # found? enabled? which ops are routed?
plugin.explain("rmsnorm", *args)  # would-be decision + why each provider declined
```

`status()['skipped_providers']` carries the reason a wheel was rejected
(unsupported `api_versions`, bad metadata). `explain()` is a dry run of gates,
eligibility and selection that never calls `run()`.

After a real call: `last_decision()`, `decision_counts()`, `on_decision(cb)`.
`last_decision()` returns `None` while nothing is enabled or under
`AITER_PLUGINS=0` — those states record nothing, and a stale answer would be
worse than none.

## What goes in `kernel_id`, `artifact_digest` and `qualification`

These are **identity, not quality**. AITER records them before execution and
hands them back unchanged; it never interprets them. Performance numbers,
coverage claims and status labels do not belong here — a routing decision that
carried a quality claim would be AITER arbitrating between libraries, which it
refuses to do.

- **`kernel_id`** identifies the *kernel*, not the op. `"rms_norm_bf16@3.2"`,
  not `"rmsnorm"`. Two candidates for the same contract must not share it.
- **`artifact_digest`** identifies the *built artifact* the kernel came from, so
  a bug can be traced to a binary. A content hash is ideal.
- **`qualification`** is optional provenance — a URL or id pointing at evidence
  you stand behind. **Omit it rather than invent one.**

**When your dispatcher cannot name the kernel.** This is the common case on a
first port: the library knows which kernel file served a call, but exposes no
id. Do not paper over it with the contract id and move on — the field then
says "rms\_norm@1.0" for every kernel that ever serves that contract, and the
provenance is worthless exactly when you need it. Add a read-only accessor on
your side; until then, record the contract id **and leave a TODO naming what is
missing**, so the next reader knows the digest does not distinguish kernels.

## Contracts that carry a workspace

Some contracts take a caller-owned scratch buffer as a public argument — for
example AITER's `mla_decode_fwd_v4_nm_ps`, whose `workspace` is an ordinary
parameter. Nothing special is required of you: it arrives bound like any other
argument, and the contract's own documentation fixes its lifetime (one
workspace serves GPU-ordered calls; concurrent calls need one each; allocate it
outside a graph capture). Honour that documentation — it is part of the
contract, and getting it wrong corrupts another call rather than failing yours.

**The interface offers no hook for resources of your own**
that outlive a call: no `prepare()`, no `reset()`, no `release()`. Two
consequences worth knowing before you design around them:

- **Allocating lazily inside `run()` during a CUDA-graph capture is unsafe.** A
  persistent buffer allocated there lands in the graph's private pool and dies
  with the graph, and the zero-init gets captured and replayed. Allocate before
  capture, from the framework side.
- **A failed `run()` can leave your state dirty**, and there is no reset verb to
  call before the next one. If your kernel keeps error words or windows, clear
  them on entry rather than on exit.

If your library needs those verbs, say so on
[RFC #7240](https://github.com/ROCm/aiter/issues/7240). The design exists; a
second library asking is what moves it.

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
