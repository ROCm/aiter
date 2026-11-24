# MiniMax-M3.1 TP4 perf reproduction: aiter branch `p1-integration`

This is the aiter half of a two-branch pair; the ATOM half is
`ganyi1996ppo/ATOM` branch `p1-c17-reproducible` (see its HANDOFF.md for
the full recipe, env vars and the companion e2e verification this
document reports the aiter side of).

## What this branch is

`origin/main` (ROCm/aiter) + PRs #8465, #8472, #8473, applied clean, zero
conflicts. Confirmed this pass: the production serving container's
baked-in aiter commit (`b9e08daca`, "mega_mhc: dispatch raw FP32 fn to
the ASM seam on gfx942") **is a direct ancestor** of this branch's HEAD
-- 114 commits behind, zero divergence. This branch is a strict superset
of what the container ships today: nothing it currently relies on is
reverted or replaced, only added to (PR #8465/#8472/#8473's fused
qknorm/index-score and MiniMax-M3 prefill-scorer/sparse-attention launch
plan work, plus everything else that landed on aiter `main` in between).

This branch does **not** touch amd_kernels in any way (checked this
pass): the only `amd_kernels` string match in the whole tree is two
docstring examples in `aiter/plugin/__init__.py` illustrating the
generic plugin-provider naming scheme (`"amd_kernels:rmsnorm"`), not an
import or a runtime dependency.

## The known prior gap, and why this branch needed re-verifying from scratch

A previous pass's import/unit-level checks (real JIT build + the PRs'
own test scripts, all passing on real hardware) were real but
incomplete: the production serving recipe's container has aiter baked
into its image (`/app/aiter-test`, pip/image-installed, not loaded from
a checkout), so **no actual server run had ever exercised this branch's
code** -- only standalone test scripts imported it. A colleague trusting
this branch's `HANDOFF.md` could reasonably believe it was "production
tested" when it had only been "unit tested in isolation."

## This pass: wiring it into a real, full-recipe e2e server run

Procedure (full detail, including the ATOM side, in `ganyi1996ppo/ATOM`
branch `p1-c17-reproducible`'s `HANDOFF.md` and
`~/.superagent/tickets/034-final-e2e-reproduction.md` on the cluster):

1. Fresh clone of this branch into
   `/shared_nfs/ygan/p1/grid2/final-e2e-repro/aiter` (not a cluster
   working tree) **with submodules initialized**
   (`git submodule update --init --recursive` -- see the gotcha below).
2. A new, isolated container (`p1final35`, node 035 MI355X).
3. Verified directly that, with no PYTHONPATH override, `import aiter`
   (from a neutral cwd) resolves to the image's baked-in
   `/app/aiter-test/aiter/__init__.py`; with this checkout's path
   prepended to `PYTHONPATH` ahead of everything else, it resolves to
   this checkout instead, and performs a real JIT kernel build
   (`module_aiter_core`, ~11s) on real MI355X hardware.
4. Ran the real MiniMax-M3.1 TP4 production recipe (arm C17, byte-exact
   env/flags) with only `PYTHONPATH` changed this way (and `REPO`
   pointed at the companion fresh ATOM checkout) -- confirmed from the
   live server process's own startup log that
   `aiter.__file__=/shared_nfs/ygan/p1/grid2/final-e2e-repro/aiter/
   aiter/__init__.py`, i.e. this branch's real code, not the image's.
5. Ran one real AgentX-style point (c1, concurrency 1, 900s, real
   `semianalysis_cc_traces_weka_062126` dataset, seed 42, 393 traces) --
   the server served real decode/prefill traffic through this branch's
   `aiter` for the whole run (confirmed: `module_aiter_core` JIT-built
   from this checkout at server startup, before any request was served).

### Result (actual; n=1, 900s, c1 conc=1, node 035 GPUs 0-3, seed 42)

| | RS tok/s/GPU | ix_p90 | TTFT p50/p90 ms | n/err |
|---|---|---|---|---|
| **this pass (wired, fresh checkouts, 900s)** | 6817 | 579.8 | 90/450 | 148/0 |
| known baseline (ATOM C17, quiet, official 3600s) | 8029 | 655.9 | 144/205 | 393/0 |
| vLLM published c1 | 7744 | 621.1 | --/431 | -- |

**Not a clean numeric match** (RS -15.1%, ix_p90 -11.6% vs the 3600s
baseline) -- but the ISL mean (167K vs 292K) and n (148 vs 393) show this
is a materially different, shorter trace-mix sample, an already-known
property of c1 (concurrency 1, single-session, think-time-limited) at
short duration, not evidence of a regression from this branch. What the
run does establish cleanly: zero errors, zero stalls/OOM over the full
900s window, with real traffic served through this aiter checkout's
code the whole time. See the ATOM branch's HANDOFF.md for the full
breakdown and the honest statement of what would be needed to turn this
into a load-bearing performance comparison (a same-duration,
same-condition control run, not attempted this pass).

## Gotcha found this pass: aiter's composable_kernel submodule

A plain (non-recursive) `git clone` of this branch leaves
`3rdparty/composable_kernel/` empty. Several JIT-compiled kernels (e.g.
`csrc/cpp_itfs/pa/pa_ps.py`) hard-fail at runtime with
`FileNotFoundError: .../3rdparty/composable_kernel/include` the first
time they're built, inside a live server process, well after the server
appeared to start loading normally -- not at import time, so a shallow
import-only smoke check (as the previous pass's verification did) would
not have caught this. Always run
`git submodule update --init --recursive` immediately after cloning.

## amd_kernels status: not applicable to this branch

This branch (aiter) never depended on amd_kernels and still doesn't. The
project's broader amd_kernels-elimination effort (replacing amd_kernels'
MoE/attention/etc. kernels with aiter-native equivalents, tracked under
ticket 022/029/033 and the `aiter-moe-gap` knowledge topic) is a
separate, NOT-yet-completed effort that lives on the **ATOM** side (see
that branch's HANDOFF.md): the ATOM tree still imports amd_kernels
directly in 19 files. This aiter branch being clean of amd_kernels is
necessary but not sufficient for that project goal -- the dependency is
on the ATOM side, and remains unresolved.
