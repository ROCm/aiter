# BISECT - which FlyDSL construct hangs the gfx1250 queue?

Status: **offline only.** Nothing in this document was run on a GPU. It is the protocol for
the **next** GPU window, after the one that hung d01-3.

What we know, from exactly one healthy node (d01-3, single gfx1250, 256 CU,
`flydsl==0.4.1.dev1004` from `/work/pylibs`):

| step | result |
|---|---|
| `env` | OK 1.0 s |
| `dev` (`torch.zeros(64).sum()`) | OK 1.1 s — **the device was healthy** |
| `compile` (both dtypes, `COMPILE_ONLY=1`) | OK 3.1 s — `[COMPILED]` bf16 + fp8 |
| `a_1row_1req_4col_bf16` | fixture-cpu → golden-cpu → kernel-module → **dispatch 0.8 s** → `dispatched (0.8s) -> synchronize` → **never returns** (TIMEOUT 62 s, GPU 100 % busy) |
| after | device wedged: `MES(0, 0) failed to respond to msg=REMOVE_QUEUE`, `Failed to evict queue 1`, `Failed to evict process queues`, `Failed to quiesce KFD`. **No new `VM_PAGE_FAULT`.** |

Two facts drive everything here:

1. **The launch returned.** So the kernel was compiled, its kernargs were accepted, and it
   was enqueued — the failure is inside the queue, not in the JIT or the API.
2. **There was no VM fault.** Every load/store address the kernel computes is inside its
   plane (`_g2/audit_addressing_cpu.py`, 18 shapes), so the address arithmetic is not the
   suspect: something makes a **wave never reach `s_endpgm`**.

A hang costs the node a reset, so the next window must not "run the kernel and see". It runs
a ladder of **minimal kernels**, smallest first, one process each, and stops at the first
hang.

---

## 0. Preconditions

Same as `FIRST_TOUCH.md` §0 (healthy node declared, `gpu-lock` held, worktree at the commit
under test, `flydsl 0.4.1.dev1004` first on `PYTHONPATH`), **plus**:

| # | check | why |
|---|---|---|
| 6 | `dmesg \| tail -40` shows no `amdgpu`/`MES`/`VM_PAGE_FAULT` line newer than the boot | you must be able to attribute the next hang |
| 7 | the ladder has been dry-run **on the node**: `python3 -u bisect.py --dry-run` prints `dry-run OK` | verifies the driver, the fixtures and the expected values before any dispatch |
| 8 | you accept that **one hang ends the session** | the STOP RULE below |

## 1. The command

```bash
# on the GPU host, inside the approved container form (see host_first_touch.sh):
bash /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2/bisect.sh 10 b1
#   arg1 = kernel-phase timeout in seconds (default 10)
#   arg2 = tag (output dir suffix)
#   anything after it is forwarded to bisect.py, e.g. --include-orig
```

`bisect.sh` is a thin wrapper: it exports `PYTHONPATH`/`QSA_GOLDEN_DIR`/`QSA_RESULTS`/
`FLYDSL_RUNTIME_CACHE_DIR`, `cd`s to `_g2`, and runs `python3 -u bisect.py --timeout <t>
--out $QSA_RESULTS/bisect_<tag> "$@"`. To go through `gpu-lock` + a container, reuse the
`host_first_touch.sh` form with `first_touch.sh` replaced by `bisect.sh`.

Outputs, in `$QSA_RESULTS/bisect_b1/`:

```
bs_<variant>.log              # that rung's whole stdout/stderr, streamed while it runs
bs_<variant>.json             # status, phase marks, the exact result values
bisect_summary.json           # every rung: label, seconds, timeout_phase, device_suspect
```

The parent prints, with `flush=True`:

```
VARIANT <name> START
VARIANT <name> OK | FAIL | TIMEOUT | SKIP <seconds>[ phase=ready|compile|kernel]
```

### The three timers (why the 10 s is meaningful)

Each rung is a **fresh process**, so a single wall-clock budget would either be dominated by
`import torch` (~3-7 s) or would fire during an import and look like a device hang. The
child therefore prints two markers and the parent budgets each phase:

| marker | covers | budget | a timeout here means |
|---|---|---|---|
| `CHILD <name> READY` | import torch + flydsl, allocate fixtures, trace the kernel | `--ready-timeout`, 180 s | import/trace stall — **not** a device event |
| `CHILD <name> DISPATCHED` | `flyc.compile` + the launch call | `--compile-timeout`, 60 s | JIT/launch stall — **not** a device event |
| process exit | `torch.cuda.synchronize()` + the exact-value readback | `--timeout`, **10 s** | **kernel-phase hang: the queue never completed. Device suspect. STOP.** |

Only a `phase=kernel` timeout sets `"device_suspect": true` in `bisect_summary.json`.

## 2. The ladder (order is the design)

The ladder climbs from "nothing" to "the whole kernel". Because the **first TIMEOUT stops
it**, the rung that hangs is the *smallest* construct that hangs, and every rung above it is
already proven good on this hardware/toolchain.

| # | variant | the one construct under test | checked result (exact) |
|---|---|---|---|
| 1 | `env` | versions + device enumeration, no tensor op | — |
| 2 | `dev` | one trivial torch GPU op | `sum == 0.0` |
| 3 | `empty` | body-less `@flyc.kernel`: module load + kernarg pack + launch, `known_block_size=[256,1,1]`, `grid=(Int64(1),Int64(1),1)` | — |
| 4 | `store_uncond_raw` | **raw** layer: one unconditional `buffer_store_dword` per lane, 256 lanes, all in bounds | `A[i] == 0x1234` for all 256 |
| 5 | `store_uncond_gridint` | same store, `grid=(1,1,1)` as **Python ints** instead of `fx.Int64` | same |
| 6 | `store_pred_raw` | raw store inside a **lane-varying** `if col < 4:` | `A[0..3] == 7.0`, `A[4:]` untouched |
| 7 | `store_uncond_cop` | **copy-atom** layer (`make_view` + `memref_store_vec` + `copy_atom_call`), unconditional | `A[i] == 0x1234` |
| 8 | `store_pred_cop` | copy-atom store under the lane-varying `if` — **exactly what `6c272c3c` did** | `A[0..3] == 7.0` |
| 9 | `meta_store_raw` | the `writer`-predicated int32 metadata stores (`if (tid==0)&(bid==0)`) | `A[0]==5`, `B[0]==0` |
| 10 | `meta_store_cop` | same through the copy-atom layer | same |
| 11 | `load_scalar_raw` | one 4 B int32 load + store-back | `B[0] == 7` |
| 12 | `load_vec_raw` | two 16 B loads (offsets 0, 16) summed + store-back | `B[0] == 28` |
| 13 | `load_vec_cop` | same two loads through `BufferCopy(128)`/`memref_load_vec` | `B[0] == 28` |
| 14 | `nostore` | loads + an 8-deep integer FMA chain, **no store at all** | `B[0] == 0` (untouched) |
| 15 | `barrier_probe` | `fx.gpu.barrier()` in uniform control flow + one store (SKIP if absent) | `A[0] == 0x5A5A` |
| 16 | `full_v1` | the **module at HEAD** (fixed access layer), tiny bf16 case | logits `[0, -inf, -inf, -inf]`, `visible == 1`, `row_starts == 0` |
| 17 | `full_v1_cop` | same module, `QSA_G2_ACCESS=cop` (the `6c272c3c` layer) | same |
| 18 | `v0_orig` | frozen `6c272c3c` module via `git show`, tiny case — **only with `--include-orig`, always last** | same |

`full_v1`'s fixture is golden-free: `q = 0`, `cache = 0` ⇒ every in-window logit is exactly
`0.0`; `pos = 3`, `ctx = 16` ⇒ `visible = 1` ⇒ only column 0 is `0.0` and columns 1-3 are
`-inf`; `logits` is **NaN-prefilled**, so the exact comparison also proves the kernel wrote
*every* column (a leftover NaN fails). `--dry-run` re-derives each expected value from the
fixture on the CPU, so a wrong expectation cannot reach the GPU window.

## 3. Decision table - read the FIRST non-OK rung

Rows 1-3 are about the harness; rows 4-15 name a construct; 16-18 are the module itself.

| first non-OK rung | conclusion | action |
|---|---|---|
| `env` FAIL/SKIP (`cuda_available: false`) | device not passed into the container | fix the launch form; rerun |
| `dev` TIMEOUT | the device was **already wedged** before any FlyDSL kernel | stop, report, node reset. Not our code. |
| `dev` OK, `empty` TIMEOUT/FAIL | the **launch path** (kernarg packing / `fx.Int64` grid / block / stream / code object), not a memory op | rerun `--only empty` once; then `--only empty --no-stream` (splits the torch stream out) and `--only empty,store_uncond_gridint` (splits `fx.Int64` grid out) |
| `empty` OK, `store_uncond_raw` TIMEOUT | a plain in-bounds **unconditional dword store** hangs | suspicion (b)/(c): the store instruction or its descriptor. Report `bs_store_uncond_raw.log`; the kernel cannot work on this toolchain |
| `store_uncond_raw` OK, `store_uncond_gridint` TIMEOUT | **`fx.Int64` grid dims** are the difference (suspicion (d)) | launch with Python-int grid values in the `@flyc.jit` wrapper (one line) |
| `store_uncond_raw` OK, `store_pred_raw` TIMEOUT | the **lane-varying `if` around a store** hangs (suspicion (a)) | replace both store predicates with the in-tree `_guarded` idiom (`mqa_logits/fp8_paged_mqa_logits_gfx1250.py:93-99` + `:247-255`), i.e. `@flyc.jit`-wrapped region with the predicate as a default argument; revert `QSA_G2_ACCESS` if needed |
| `store_pred_raw` OK, `store_uncond_cop`/`store_pred_cop` TIMEOUT | the **copy-atom layer** is the hang, with or without a predicate | the fix already in HEAD (`raw` is the module default) is the answer; nothing to change |
| `store_*_raw` OK, `meta_store_raw` TIMEOUT | the **int32 / 1-lane-predicated** store path | report; the metadata stores must be re-shaped (e.g. `_guarded`) |
| `meta_store_raw` OK, `meta_store_cop` TIMEOUT | copy-atom on the metadata store only | same as the copy-atom rows above |
| `load_scalar_raw` TIMEOUT | the scalar dword **load** path | report; the metadata loads are the problem |
| `load_vec_raw` TIMEOUT | the **16 B vector load** path (suspicion (c)) | report; Q/KV loads are the problem |
| `load_vec_raw` OK, `load_vec_cop` TIMEOUT | copy-atom **loads** too | the access-layer fix is the answer |
| `nostore` TIMEOUT | loads/compute alone hang with **no store** | pure load-path hang; report |
| `barrier_probe` TIMEOUT | barriers hang on this toolchain | do not add one to this kernel |
| every rung up to `full_v1_cop` OK, `full_v1` TIMEOUT/FAIL | the fix is **not** the fix | the ladder has already proven which layer is safe: bisect the module against that layer (e.g. move the Q/KV loop or the `-inf` select into a rung) |
| `full_v1` OK, `full_v1_cop` TIMEOUT | reproduces d01-3 **only with the old layer** | the access-layer change *is* the fix; continue to `FIRST_TOUCH.md` |
| `full_v1` OK, `full_v1_cop` OK | the access layer is not the hang | re-read rows 4-15: the module differs from every rung in the Q/KV loop depth (16 steps x 4 heads) and the `-inf` select; add rungs for those |
| all OK, `v0_orig` OK | the d01-3 hang was **not** this kernel (environment) | stop; re-read the d01-3 evidence with the root agent |

If a rung **FAILs** (exception/traceback or a wrong value), the device is fine: read
`bs_<rung>.log` — a wrong value means the construct works but the *address or descriptor* is
wrong, which is a correctness bug, not a hang.

## 4. STOP RULE

> **Any `TIMEOUT` with `phase=kernel` ⇒ STOP.**
> Do not retry that rung, do not run the next rung, do not run `FIRST_TOUCH.md`, do not
> "try the other dtype", do not start another container.
> Capture, in this order: `bisect_summary.json`, all `bs_*.log`, `dmesg | tail -40`,
> `rocm-smi --showpids`, `ps -ef | grep -c python3`.
> Then ask the human/root for a **node reset** and report which rung hung.
>
> One hang per session is the budget: the queue cannot be evicted
> (`MES ... failed to respond to msg=REMOVE_QUEUE`), so a second attempt can only cost
> another reset.

A `TIMEOUT` with `phase=ready`/`phase=compile` is **not** a device event: the queue was
probably never touched. Capture the log, and retry that single rung **once** with
`--compile-timeout 300` before drawing any conclusion (a cold LLVM/JIT can be slow).

## 5. What a passing ladder proves - and what it does not

Proves: on this image/toolchain, each listed construct (both access layers, predicated and
unconditional stores, scalar and 16 B loads, a body-less kernel) completes and produces the
exact expected values, and the module's smallest bf16 case is correct.

Does **not** prove: the fp8 specialization, any shape beyond 1 row x 4 columns, the Q/KV loop
at full depth (16 steps x 4 heads) as a *separate* construct, performance, or the full
contract. Those stay with `FIRST_TOUCH.md` (rows `a`-`e`) and `run_gpu.py --stage
correctness`, in that order.

## 6. Offline checks that already exist (no GPU)

```bash
python3 _g2/bisect.py --dry-run            # every expected value re-derived on the CPU
python3 _g2/bisect.py --list               # the ladder, with what each rung isolates
python3 _g2/test_contract_cpu.py           # contract + guard rails (63 checks)
python3 _g2/audit_addressing_cpu.py --variant v0   # every address the kernel computes
```
