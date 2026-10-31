# FIRST TOUCH — page-4 QSA logits, minimal-first-touch protocol (next GPU window)

Status: **offline only.** No GPU run happened for this document. The previous three attempts
ended in (a) a `GPU Hang` during `import aiter` (a08-1, before our kernel was compiled),
(b) 40 min of **zero output** from `run_gpu.py --stage smoke` on d07-2, followed by
`amdgpu 0001:04:00.0: [0] id=4 (VM_PAGE_FAULT) size=0x1000 start=0x16bf5fff000` and a wedged
device, and (c) **the first real dispatch of the fixed kernel (`6c272c3c`) on d01-3 hung the
GPU**: dispatch returned in 0.8 s, `torch.cuda.synchronize()` never returned, and `dmesg`
showed `MES(0, 0) failed to respond to msg=REMOVE_QUEUE` / `Failed to evict queue 1` /
`Failed to quiesce KFD` — the node needs a reset. This protocol replaces "one big run, no
output" with a ladder where every step names itself, has its own log and JSON, and is killed
at 60 s.

**The ladder in this file is no longer the first GPU step.** `_g2/bisect.py` (18 minimal
rungs, 10 s each, one construct per rung) runs **first** and names the construct that hangs;
this 9-step ladder runs only if *every* bisect rung is OK. See §1b and `BISECT.md`.

Read `records/g2/builder2-log.md` (predicated stores instead of the `0x7FFFFFFF` "dropped"
sentinel, i64 descriptor extents, host guard rails) and `records/g2/builder3-log.md` (the
d01-3 hang, the access-layer change, the bisect ladder) before touching a node.

---

## 0. Preconditions (all must be true)

| # | check | why |
|---|---|---|
| 1 | the human/root has **declared a node healthy** (a07-3, a08-1 and d07-2 were all left wedged/reachable-unknown) | a wedged device makes every step meaningless |
| 2 | `gpu-lock` is installed on that node and no other tenant holds it | shared node |
| 3 | the worktree on the node is at the **builder3 commit** (`git log --oneline -1`; it must contain the access-layer change and `_g2/bisect.py`) | the fix must be the thing under test |
| 4 | `python3 -c "import flydsl; print(flydsl.__version__)"` prints `0.4.1.dev1004` **with** `PYTHONPATH=/work/pylibs:/work/work/qsa-v0` | the image's flydsl 0.4.1 is not the pinned one |
| 5 | nobody has run anything on that GPU since the last `dmesg` check | the fault may be pre-existing |

If any is false, **stop** and report; do not improvise.

## 1. The command (one line, one gpu-lock section, bounded)

```bash
# on the GPU host, from the worktree copy:
bash /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2/host_first_touch.sh ft1 60
```

`host_first_touch.sh` wraps exactly the approved container form
(`gpu-lock docker run --rm --ipc=host --device=/dev/kfd --device=/dev/dri --group-add
video/render -v <root>:/work -e PYTHONPATH=/work/pylibs:/work/work/qsa-v0 ...`), and for a
persistent container: `MODE=exec CONTAINER=qsa_m455_blyu bash host_first_touch.sh ft1 60`.
For a node whose task copy lives elsewhere: `QSA_ROOT=/home/blyu/qsa IMG=<image>`.

Add `DMESG=1` to print the `amdgpu` fault lines after the ladder (never fails the run).

Outputs land in `$QSA_ROOT/results/g2/builder/first_touch_ft1/`:

```
ft_<step>.log                 # that step's whole stdout/stderr (streamed while it runs)
ft_<step>.json                # tiny per-step record (status, timings, numbers)
first_touch_summary.json      # every step, status + seconds + worst
```

## 1b. Order of the window: env -> dev -> compile -> **bisect** -> this ladder

Do not start with §2. The order is fixed, and each arrow is a gate:

| # | step | command | gate |
|---|---|---|---|
| 1 | `env` | `bash …/_g2/first_touch.sh 60 ft1` runs it, or `python3 -u first_touch.py --step env` | `cuda_available: true`, arch `gfx1250` |
| 2 | `dev` | `first_touch.py --step dev` (`torch.zeros(64).sum()`) | must be OK in ~1 s; if it hangs the device was already bad — stop, reset, not our code |
| 3 | `compile` | `first_touch.py --step compile` (`COMPILE_ONLY=1`, no device) | both dtypes `[COMPILED]`; a FAIL here is a lowering regression, no GPU needed |
| 4 | **`bisect`** | `bash …/_g2/bisect.sh 10 b1` | **every rung OK/SKIP** (10 s kernel-phase timeout each, 18 rungs). Any `TIMEOUT` in `phase=kernel` ⇒ STOP RULE in `BISECT.md` §4: capture dmesg, ask for a reset, do not retry. |
| 5 | this ladder | `bash …/_g2/host_first_touch.sh ft1 60` | only if step 4 was all OK |

Step 4 is the one that converts "the kernel hangs" into "**this** construct hangs": it runs
the two access layers (raw / copy-atom), predicated and unconditional stores, scalar and
16 B loads and a body-less kernel as separate processes, checking exact result values. It is
also the cheapest way to spend a GPU window: ~18 rungs x 10 s + imports, and the first hang
ends it.

```bash
# step 4, the exact command (inside the container, under gpu-lock):
bash /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2/bisect.sh 10 b1
#   -> $QSA_RESULTS/bisect_b1/{bs_*.log,bs_*.json,bisect_summary.json}
#   add --include-orig to append the frozen v0 (6c272c3c) control, last
```

Before any of it, on the node and with no GPU touched:

```bash
python3 -u bisect.py --dry-run     # must print "dry-run OK"
```

## 2. The ladder (each step: own process, `PYTHONFAULTHANDLER=1`, hard 60 s timeout)

| order | step | what it proves |
|---|---|---|
| 1 | `env` | python / torch / **flydsl version** / `torch.cuda.is_available()` / device name+arch. No tensor op. If this hangs, the device or the driver is already bad. |
| 2 | `dev` | one trivial GPU tensor op (`torch.zeros(64).sum()`) — **the exact probe that hung after the d07-2 wedge**. Separates "device is wedged" from "our kernel is broken". |
| 3 | `compile` | `COMPILE_ONLY=1` build of **both** dtype specializations, no dispatch, no device. Catches a tracing/lowering regression from the fix before anything touches the GPU. **Caveat:** under `COMPILE_ONLY=1` the LLVM backend stage is skipped (`code_objects: []`), so this does *not* prove a `.hsaco` builds; bisect rung `empty` is the first real end-to-end compile. |
| 4 | `a_1row_1req_4col_bf16` | **1 row × 1 request × one 4-column page, bf16.** Grid = (1,1,1), one workgroup, 4 active lanes out of 256. The smallest thing that exercises the descriptor, the page/row split, the Q/K loads, the `-inf` mask and the store. |
| 5 | `b_1row_1req_8col_bf16` | 2 pages / 8 columns: the `page = c//4`, `off = c%4` split across two pages. |
| 6 | `c_4rows_1req_16col_bf16` | 4 rows: grid.y > 1, the per-row metadata path (`token_to_request`, `context_lens`, `query_positions`) and the single-writer `visible_groups` store. |
| 7 | `d0_fp8_scale0` | fp8 cache + per-token scale **0**: `k = E4M3 * 0 = 0` ⇒ every in-window logit is exactly `0.0`. Proves the fp8 decode path and the scale index `[pages,4,1]` without depending on numerics. |
| 8 | `d1_fp8_scale1` | fp8 cache + scale **1**: compares against the public golden (E4M3 values must match exactly). |
| 9 | `e_I5_min_decode_bf16` | the existing I5 smoke case (`rows=4, requests=1, width=16, cache_pages=64`), i.e. **the first case `run_gpu.py --stage smoke` would have run**. A PASS here means the v0 smoke path is alive again. |

Each dispatch step builds its fixture **on the CPU**, computes the golden **on the CPU**,
launches, calls `torch.cuda.synchronize()`, then compares on the CPU — so a failure is
attributed to the launch, the sync, or the numbers, never to the fixture.

The `compile` step is also the **only** step that can be run on a node with no free GPU:
`_g2/compile_only.py` (needs no `/dev/kfd`).

## 3. How to read the result

| observation | conclusion | next action |
|---|---|---|
| `env` SKIP/`cuda_available: false` | no GPU in the container (device passthrough / `HIP_VISIBLE_DEVICES`) | fix the launch form; keep the same ladder |
| `env` OK, `dev` TIMEOUT | **device is wedged before we do anything** | stop, report, node reset. Not our kernel. |
| `compile` FAIL | the fix does not build on this toolchain (see `ft_compile.log`, `compile_only.json`) | send the log; `QSA_G2_ACCESS=cop` is the one-variable revert of the builder3 access layer |
| **bisect**: any `TIMEOUT phase=kernel` | **a specific minimal construct hangs** — do not go on to step 5 | `BISECT.md` §3 names the construct from the first non-OK rung, §4 is the STOP RULE |
| `a` FAIL/TIMEOUT but `dev` OK | a real kernel bug in the smallest case: descriptor/address/launch | stop after capturing `ft_a_*.log` + `dmesg`; send both. (Only reachable when bisect `full_v1` was OK, so the hang is in a construct the module adds on top of the rungs: the 16x4 Q/KV loop or the `-inf` select.) |
| `a` OK, `b` TIMEOUT/FAIL | the column/page arithmetic beyond one page | same |
| `a`,`b` OK, `c` FAIL | the row metadata path (`grid.y`, `visible_groups`) | same |
| `a`–`c` OK, `d0`/`d1` FAIL | the fp8 decode / scale plane (the software E4M3 path was never GPU-validated) | same |
| `a`–`d1` OK, `e` FAIL | the golden-fixture path (int64 metadata, `max_columns`, random page table) | same |
| every step OK | v0 is dispatchable; correctness numbers are in `first_touch_summary.json` | then, and only then, `run_gpu.sh smoke smoke1` and the full correctness stage |

**Any TIMEOUT or `VM_PAGE_FAULT`/`GPU Hang` in `dmesg` means: stop. Do not retry, do not
start another container, do not "try the other dtype".** Capture
`first_touch_summary.json` + all `ft_*.log` + `dmesg` and report.

## 4. What this protocol does NOT establish

* **That the d01-3 hang is fixed.** Bisect rung `full_v1` only proves the tiny bf16 case
  (1 row x 4 columns). The fp8 path, the 16-step Q/KV loop at full depth and every larger
  shape are still first-contact here.
* **That `QSA_G2_ACCESS=raw` is correct in the limit.** The access-layer change is
  behavioural-equivalent by construction (same planes, extents, byte addresses, predicates
  and values), but it is compile-unverified offline: bisect rung `empty`/`store_*` prove the
  primitives, rung `full_v1` proves the module, and `full_v1_cop` runs the *old* layer for
  comparison in the same window. If `raw` misbehaves, `QSA_G2_ACCESS=cop` reverts it without
  an edit.
* **Performance.** Nothing here is timed beyond "it finished". `run_gpu.py --stage timing`
  is the first-look path and it stays exploratory/not-evidence.
* **The full contract.** Only 9 cases, ≤ 64 columns, ≤ 64 pages. The I1/I2/I3 e2e shapes
  (738 964-page pool, 65 536 columns), the 512 × 65536 prefill chunk, `columns`
  truncation to 32768 and the `page >= num_pages` / `token_to_request = -1` edge cases
  are covered by `run_gpu.py --stage correctness` (19 cases) **after** this ladder passes.
* **A code-object hash.** FlyDSL materialises the `.hsaco` at dispatch, not at compile, so
  the hash only exists after step 4; take it from `$FLYDSL_RUNTIME_CACHE_DIR` then.

## 5. Offline evidence that already exists (no GPU needed)

```bash
python3 _g2/test_contract_cpu.py            # contract + guard rails, CPU only
python3 _g2/audit_addressing_cpu.py --variant v0   # every address the kernel computes
python3 _g2/bisect.py --dry-run             # the 18 bisect rungs + their expected values
```

`audit_addressing_cpu.py` is the offline half of "cannot fault": it enumerates (or
bounds in closed form) every load/store address for 18 shapes, checks each against its
plane's extent and against int32, and — with `--variant v0` — reports how many stores the
old code steered at the constant offset `0x7FFFFFFF` (762 of them in the very first
first-touch case) plus whether the V# would have had to drop them. Its `fault_check()`
re-derives the d07-2 dmesg address from such a store: a page-aligned allocation base
`B` puts the fault page start at `B + 0x7FFFF000`, and `0x16bf5fff000` is exactly
`0x1FF000` mod 2 MiB with an implied base of `0x16b76000000` (2 MiB aligned).
