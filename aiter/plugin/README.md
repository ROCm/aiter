# Serving an AITER op from an out-of-tree kernel library

This interface lets a kernel library serve an AITER op without forking AITER
and without AITER knowing the library exists. AITER owns the contract; a
provider implements it or declines.

Design rationale, open questions and the deprecation policy live in
[RFC #7240](https://github.com/ROCm/aiter/issues/7240). This file is how to
write one.

## Read this first

**Installing a provider does nothing.** Discovery is lazy, enablement is
default-deny, and only a framework calling `configure()` turns anything on. A
provider wheel arriving as somebody's transitive dependency cannot change which
kernels run.

**Five ops are routed today**, and `tests/plugin_contracts.json` is the list
that cannot drift from what dispatch actually hands you:

| op | version |
|---|---|
| `rmsnorm` | 2 |
| `rmsnorm_with_add` | 1 |
| `pertoken_quant` | 1 |
| `gemm_a8w8_blockscale` | 1 |
| `gemm_a8w8_blockscale_bpreshuffle` | 1 |

If your op is not in that file it is not routed — there is nothing to serve,
and `plugin.status()['routed_ops']` says the same at runtime.

## The smallest provider that works

```python
from aiter import plugin


def my_rmsnorm(out, input, weight, epsilon, gemma_norm=False):
    ...  # your kernel; must accept AITER's contract exactly
    return out


class MyProvider:
    name = "mylib"                       # the name a framework enables
    api_versions = {plugin.API_VERSION}  # which provider API you speak
    op_versions = {"rmsnorm": {2}}       # which op contracts, at which version

    def select(self, op, args, kwargs):
        """Decide whether to serve THIS call. Pure: no side effects, no launch.

        Return a KernelCandidate to serve it, or None to decline. AITER asks the
        next enabled provider, then falls back to its own kernel.
        """
        if args[1].dtype is not torch.bfloat16:
            return None                  # declining is normal, not a failure
        return plugin.KernelCandidate(
            provider="mylib",
            kernel_id="rmsnorm_bf16@3.2",   # identifies the kernel, not the op
            artifact_digest="sha256:...",   # identifies the built artifact
            qualification="https://...",    # optional: evidence you stand behind
            run=my_rmsnorm,
        )
```

Declare the entry point so AITER can find it:

```toml
[project.entry-points."aiter.plugins"]
mylib = "mylib.aiter_provider:MyProvider"
```

The target may be a provider instance or a zero-argument factory. Anything with
a `select` attribute is treated as an instance, so a class object is never
called.

Then a framework — not you, and not the environment — turns it on:

```python
plugin.configure(source="vllm rocm defaults 2026.11", enable=["mylib"])
```

`enable` is **preference order**. `source` is required provenance: every routing
decision records who enabled it, which is why enablement cannot live in an
environment variable.

## What `select()` must honour

- **Return, do not raise.** Raising is isolated and recorded, but it costs the
  call. `None` is the supported way to say no.
- **Be pure.** `explain()` calls `select()` without running anything, so a
  `select()` with side effects makes the dry run a lie.
- **Take the contract exactly.** AITER checks your `run` against the contract's
  arity before handing it the call, and refuses the candidate with
  `reason="contract-mismatch"` if it does not fit. That check compares argument
  counts, not meaning — see *Bind to the contract* below for why that
  distinction matters.
- **Once `run()` starts, errors propagate.** AITER does not fall back mid-flight:
  retrying could repeat side effects or hide a broken kernel behind a correct
  result.

## "I installed it and nothing happens"

Four questions, four answers, none of which require running a kernel:

```python
plugin.status()
# {'enabled': (), 'providers': [...], 'skipped_providers': [...], 'routed_ops': [...]}
```

- *Was I found?* → `providers`. If you are in `skipped_providers`, the reason is
  there (bad wheel, unsupported `api_versions`).
- *Am I enabled?* → `enabled`. Empty means no framework called `configure()`,
  and nothing will route.
- *Is this op even routed?* → `routed_ops`.

```python
plugin.explain("rmsnorm", out, x, w, 1e-6)
# Explanation(decision=Decision(route='native', reason='provider-op-unsupported'),
#             declines={'mylib': 'serves rmsnorm version 1, not 2'})
```

`explain()` is a dry run of dispatch — gates, eligibility, selection — that
never calls `run()`. It is the fastest way to find out why a call went native,
and it answers after the fact too:

```python
plugin.last_decision()   # the last routing decision in this context
plugin.decision_counts() # per (op, route, reason, provider)
plugin.on_decision(cb)   # called inline on every decision; for telemetry
```

`last_decision()` returns `None` while nothing is enabled or under
`AITER_PLUGINS=0` — in those states nothing is recorded, and a stale answer
would be worse than none.

## Four things that have actually gone wrong

Each of these cost real debugging time on the first provider to use this
interface.

**Bind to the contract, not to the layout.** A provider offered its
`bpreshuffle` kernel for the `blockscale` contract — the shuffled layout is a
flag *inside* that contract and also a separate public function of its own.
Seven arguments were about to reach a six-argument kernel.

That one was caught by the arity check — `reason="contract-mismatch"`, the
provider named in the log — and nothing was corrupted. But do not read that as
a safety net:

**arity is not meaning.** Two contracts with the same number of parameters and
different semantics pass the check. `mla_sparse_prefill_v4` and `_v41` both take
15 arguments, and their second parameter is `q_rope` in one and `q_scale` in the
other. A kernel bound to the wrong one reads a scale tensor as a RoPE component
— silently, with every gate green.

One op name, one contract, one `run`. Nothing downstream can recover a
contract bound wrong.

**Match on `gcnArchName`, not on the device name.** On MI355X under
`rocm/atom-dev` with torch 2.10.0+rocm7.2.4:

```
torch.cuda.get_device_name(0)  →  'AMD Radeon Graphics'
gcnArchName                    →  'gfx950:sramecc+:xnack-'
```

A provider matching hardware by device name loaded zero kernels and declined
every call — the interface looked broken while behaving exactly as specified.

**Having timings is not having a number you can report.** A provider's offline
validation timings are measured on shapes it chose, often as a ratio against a
baseline. AITER labels what it records with the conditions of the call in front
of it, so such a number arrives attached to a shape it never ran on. Omit it
instead; a number that looks sourced and is not is harder to catch than a
missing one.

**Declining is not failing.** A provider that cannot serve a dtype, a shape or
an arch should return `None` every time, cheaply. `explain()` will show the
decline and `decision_counts()` will count it; neither is an error state.

## Telling AITER what a kernel costs

`KernelCandidate` takes an optional `latency_us`: how long **this kernel takes
for this call**, in microseconds. Omitting it is the default and stays
supported — AITER records `None` and nothing else changes.

```python
plugin.KernelCandidate(..., latency_us=12.4, run=my_kernel)
plugin.last_decision().latency_us    # 12.4, recorded unchanged
```

Three rules, and the first is the one that is easy to get wrong:

- **it describes the call in hand, not a past measurement.** AITER labels what
  it records with the conditions of the call the candidate was offered for. A
  number measured on some other shape arrives wearing conditions it was never
  measured under — worse than no number, because it looks sourced.
- **no number is better than an extrapolated one.** With nothing for the shape
  in front of you, omit the field. `0`, a negative, `inf` and `nan` are refused
  at construction: each looks like data and carries none.
- **it never influences routing.** Preference is the enable list, which names
  an owner for that decision. A number a provider picks for itself must not be
  able to outrank it.

If what you have is offline validation timing — a ratio against a baseline, on
shapes your own harness picked — that is not this field. Put a reference to the
evidence in `qualification` instead, and leave `latency_us` unset.

## Versions

`op_versions` maps an op name to the contract versions you serve. AITER owns
those versions: a contract change that breaks providers bumps the op version,
and AITER may keep accepting an older one for a stated window (`supersedes`),
which `status()['deprecations']` reports. Declaring a version you do not
actually implement is caught by the candidate check, not trusted.

`api_versions` is the provider-API level you speak — currently
`plugin.API_VERSION == 1`. Declare every version you support so one wheel can
cover several AITER releases.

## The kill switch

`AITER_PLUGINS=0` disables routing process-wide. It only subtracts: the
environment can never enable a provider, because enablement has to name an
owner and an environment variable cannot. Under it, dispatch goes straight to
native and records nothing.

It does not reach a kernel already captured into a CUDA graph — replay runs no
Python, so whatever was decided at capture is what the graph does for its
lifetime. Rebuild the graphs.
