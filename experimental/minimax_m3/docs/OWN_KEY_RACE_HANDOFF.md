# Own-key stale-row race in the Gluon monokernel: handoff (2026-10-22)

Status: FIXED in main (`m3mono/gluon/kernel.py`, knob `VLLM_M3_GLUON_TAIL_ACQ`, default 8).
Nothing committed to git; the fix lives in the working tree. Backup of the pre-fix kernel:
`m3mono/gluon/kernel_pre_tailacq.py.bak`.

## 1. Symptom

`bench/bench_agentx.py --check` (Gluon vs FlyDSL, TP4, AgentX shape ql=4) reported nRMSE 0 at 4 layers
but large divergence at 57 layers. Once `--check` restored pristine KV / index caches before EACH
backend's first step (it now does), the 4-layer check also diverged everywhere: c=1 0.32 / 0.073 / 0.066
and c=2 0.34 / 0.28 / 0.067 at ctx 1024 / 2048 / 320K. The old check was masked because FlyDSL ran first
and left the correct own-key rows in the KV cache for Gluon to read.

`bench/debug_layer.py 4 2048 1` localized the first divergence to the split attention: the dense path
(keys <= 2048) was 4-5% off on "a (sum)". That error equals `DBG_NOTAIL` (own key not attended at all).

## 2. Root cause (two parts, both required)

Each decode token's own K / V row (and each new index key's row) is written by one CTA in the same launch
and re-read by the split-attention / scorer CTAs after a tagged flag.

1. **Wave-0-only flag poll, no CTA barrier.** `land()` on the own-key flags (`R_OWN`, tail split,
   lanes 0..QL-1 / QL..2QL-1) and on the new-index-key flag (`R_IKF`, `score_body`, lanes < S) is executed
   by wave 0's lanes only. Waves 1..7 did not wait and re-read the cache rows before the flag had landed.
2. **Stale rows in the CU's vector L1.** The partition's early page loads brought the tail page into the
   CU's L1 (TCP). `.cv` (sc0 sc1) buffer loads do NOT bypass an L1 hit on gfx950, so even a wave that did
   wait re-read stale bytes when the speculative rows share the tail page (sparse / spec case).

Mode experiments on `debug_layer.py 4 2048 1` (dense rows=4) and the shared-tail-page spec case
(`M3_NB=25 debug_layer_spec.py 1 2048 1 2 4`):

| TAIL_ACQ | what | dense rows=4 | shared tail page |
|---|---|---|---|
| 0 | off | 4.2e-2 (wrong) | wrong |
| 6 | gl.barrier() only | exact | wrong (0.386) |
| 4 / 5 | buffer_inv sc0 (+vmcnt), no barrier | 4.2e-2 (wrong) | wrong |
| 2 | buffer_inv sc0 ; vmcnt(0) ; barrier | exact | timing dependent (passed on worker copy, failed on main) |
| 1 | buffer_inv sc1 (L2) ; vmcnt(0) ; barrier | exact | exact, ~1 us/layer |
| **8** | **vmcnt(0) ; barrier ; buffer_inv sc0 ; vmcnt(0) ; barrier** | **exact** | **exact** |

Mode 2 fails intermittently because a wave's early page loads can return after another wave's invalidate
and refill the L1 stale. Mode 8 drains every wave first, then invalidates, then releases.

## 3. The fix in main

`kernel.py` near line 292: knob + helper `l2_acquire(x)` (the modes above). Call sites:

- `stage_split`, after `land(rgn(sc, R_OWN), fidx, tag, fpk, errors, bid, E_QKV)` / `stamp(tr_, bid, 37)`:
  `if TAIL_ACQ: l2_acquire(ft)` (kernel.py ~line 2625).
- `score_body`, after `land(rgn(sc, R_IKF), tl_, tag, fpk, errors, bid, E_QKV)`:
  `if TAIL_ACQ: l2_acquire(tl_)` (kernel.py ~line 1931).

Compiles S=4 226 VGPRs / S=8 197, no spills. Other raw `land()` sites were audited: they consume the
polled words themselves (no re-read of other memory), so they are safe. Rule for future code: any
`land`/`peek` wait whose predicate covers only some lanes needs a CTA barrier (and, if the data could be in
L1 from an earlier load, the drain + `buffer_inv sc0`) before other waves consume the data. `gate_poll`
already does this.

Cost of the fix (57 layers, min of 10 steps, knob 8 vs 0): c=1 +32 / +22 us/step, c=2 +25 / +27 us/step
at ctx 2048 / 320K.

## 4. Verification (main, TAIL_ACQ=8), logs in `bench/logs/`

- `fin5_chk2.log`: c=2, 4 layers, pristine caches: nRMSE h 0 / r 0 at 1024, 2048, 320K; tables EQUAL.
- `fin5_chk1.log`: c=1, 4 layers: 3.2e-2 / 0.386 / 5.6e-3. This residual is FlyDSL's (section 5), not Gluon's.
- `fin5_c2.log`: 57 layers c=2: gluon 4659 / 5206 vs flydsl 4618 / 5128 us/step (1.009 / 1.015).
- `fin5_c1.log`: 57 layers c=1: gluon 3233 / 3498 vs flydsl 3245 / 3506 us/step (0.996 / 0.998).
- `fin5_ab{1,2}_{a,b}.log`: knob 0 vs 8 cost A/B.
- `/home/vigokhal/agentx/run_dbg17.out`: debug_layer(+spec) with mode 8 == TAIL_SELF numbers
  (L=1 4.556e-4, L=2 2.608e-2, L=4 ar 0.2556; dense rows=4 2.5e-5 / 0.0).

## 5. FlyDSL defect found on the way (NOT fixed, not reported to its owners yet)

`bench/ref_attn.py` computes a per-row attention reference for rank 0's 16 heads in three flavours:
plain float, fp8-emulated, and "kern" (faithful to both kernels' pipeline: per-partition 256-key max,
exp2, fp8 P with 448 scale, bf16 partials, merge by l*exp2(m-gm), per-token fp8 output quant). Only "kern"
discriminates: the plain float reference is ~4.7% from BOTH kernels.

`/home/vigokhal/agentx/run_ref4.out` (reuse layer L=2, c=1 spec ql=4, M3_NB=25, ctx 2048):

| row | gluon vs kern | flydsl vs kern | gluon vs flydsl |
|---|---|---|---|
| 0 | 1.8e-3 | 1.8e-3 | 0 |
| 1 | 9.7e-4 | 9.7e-4 | 0 |
| 2 | 5.0e-3 | **5.1e-2** | 5.1e-2 |
| 3 | 3.3e-3 | 6.0e-3 | 5.1e-3 |

FlyDSL mishandles speculative row jq=2 at reuse layers (weakly at layer 0 too). Because a 5e-2 error on
one row flips top-16 index rankings in later layers, the 57-layer c=1 bench can never match FlyDSL bit for
bit; the c=1 4-layer residual (3.2e-2 / 0.386 / 5.6e-3) is this defect chaotically amplified.
Validation of Gluon changes must therefore use `debug_layer*.py` / `ref_attn.py`, or the c=2 check
(which is exact), not c=1 nRMSE vs FlyDSL.

## 6. Reproduce

All commands run inside container `ix_dsv41` on a free TP4 set (GPUs 4-7 default; GPUs 0-3 only when
`bench/set0_idle.sh` passes). Wrappers serialize per set with flock.

```bash
cd /home/vigokhal/m3_monokernel/bench
# pristine-cache 4-layer check, both concurrencies (expect c=2 exact, c=1 residual per section 5)
GPUSET=1 ./bax2.sh chk2 "" --backends flydsl,gluon --reqs 2 --ctx 1024,2048,327680 --layers 4 --check --steps 3
GPUSET=1 ./bax2.sh chk1 "" --backends flydsl,gluon --reqs 1 --ctx 1024,2048,327680 --layers 4 --check --steps 3
# show the bug again
GPUSET=1 ./bax2.sh bug2 "VLLM_M3_GLUON_TAIL_ACQ=0" --backends flydsl,gluon --reqs 2 --ctx 2048 --layers 4 --check --steps 3
# cost A/B (knob off vs on), 57 layers
GPUSET=1 ./ab2.sh cost2 "VLLM_M3_GLUON_TAIL_ACQ=0" "" --backends gluon --reqs 2 --ctx 2048,327680 --layers 57 --steps 10
```

Per-stage debugging (see `/home/vigokhal/agentx/run_dbg17.sh` for the exact docker exec / env line):

```bash
# dense rows=4, one layer; prints "a (sum)" nRMSE per stage (expect ~2.5e-5 / 0)
torchrun --nproc_per_node 4 bench/debug_layer.py 4 2048 1
# spec ql=4, c=1, L layers, last layer's regions compared; M3_NB fixes the page permutation
M3_NB=25 torchrun --nproc_per_node 4 bench/debug_layer_spec.py 1 2048 1 2 4     # argv: reqs ctx topk L ql
# per-row attention reference (section 5); see /home/vigokhal/agentx/run_ref.sh
M3_NB=25 torchrun --nproc_per_node 4 bench/ref_attn.py 1 2048 1 2 4
```

Notes: at 320K pass `max_pos=ctx+ql+16` to `make_layers` (the debug tools already do; otherwise FlyDSL
produces NaN). `bench_agentx.py` with `M3_OUT_SAVE=<prefix>` saves first-step h / r per backend and ctx
on rank 0 (saved examples: `/home/vigokhal/agentx/det{1,2}_{gluon,flydsl}_{2048,8192}.pt`).

## 7. Open items

- Report the FlyDSL row-2 defect to its owners with `bench/ref_attn.py` + `run_ref4.out` as evidence.
- Run an end-to-end accuracy eval (GSM8K / GPQA / AIME, harness in `/home/vigokhal/herd_eval/`) with the
  Gluon monokernel now that the own-key path is correct, compared with FlyDSL. Note: neither monokernel uses
  HERD; HERD exists only in AITER's Triton `routing()` on the `aiter_triton_mxfp4_bf16` backend. Both
  monokernels do plain per-token top-k (`route_top4` in kernels/stages/router.py).
- Optional: cheaper acquire. Mode 8 costs ~25 us/step; the drain could be narrowed to the tail-page
  loads only, or the early load of the tail partition skipped (TAIL_LATE_LOAD in the r25 worker copy,
  `/home/vigokhal/m3_work/r25/m3mono/gluon/kernel.py`) so that only the barrier is needed.
- The r25 worker copy also holds the rejected MERGE_LAST / OQ_TAG experiments (all slower); not merged.
