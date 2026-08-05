# gfx1250 A8W4 grouped-MoE TDM GEMM — session handoff

Paste this into a new session. Everything below is either measured or cited;
inferences are marked as such.

---

## 1. Environment

### Local (this machine)
- Windows, **no ROCm / HIP / GPU / torch**. Only edit code, statically analyse,
  and parse already-dumped artifacts (ISA, ATT trace JSON, sqlite).
  Never propose "let me run it to verify".
- Repo: `C:\Users\yanguahe\code\wk_sp2\aiter` (branch `hyg_gfx1250_gemm`,
  remote `git@github.com:ROCm/aiter.git`).
- FlyDSL source for reference: `C:\Users\yanguahe\code\wk_sp2\FlyDSL`
- MI400 ISA doc (cite file+line for every HW claim):
  `C:\Users\yanguahe\code\mi400_hw_wiki\raw\papers\mi400_hd_txt\architecture\subsystem\SH\MI400_Shader_Programming#65.txt`

### Remote GPU boxes
| host | container | repo path | notes |
|---|---|---|---|
| `heliosr-2b805-b8-2.aus-b200.dcgpu` | `hyg_fyd1` | `/data/yanguahe/code/wk_sp1` | current box; has `llvm-project` source |
| `heliosr-2b805-d7-3.aus-b200.dcgpu` | `hyg_fyd2` | `/data/yanguahe/code/wk_sp1/aiter` | earlier box |

```bash
ssh -i ~/.ssh/id_rsa_carhuang carhuang@heliosr-2b805-b8-2.aus-b200.dcgpu
# ConnectTimeout>=45: Conductor auth handshake is slow
# git on the HOST needs:
export GIT_SSH_COMMAND='ssh -i /data/yanguahe/code/id_rsa.hyg -o IdentitiesOnly=yes'
```

Code transfer: **edit locally → git push → git pull on the box.**

### Hard-won operational rules
- **NEVER `kill -9` a process that has touched the GPU.** SIGKILL leaves the
  amdgpu MES unable to reclaim queues → `MES failed to respond to REMOVE_QUEUE`
  → ASIC reset fails → machine reboot. I did this once and killed two GPUs.
  Send `SIGTERM` only, or let the script's `timeout` expire. `timeout`/`bash`/
  `tee` wrappers that never opened `/dev/kfd` are safe to `-9`.
- `subprocess.run(timeout=)` in Python sends SIGKILL on expiry — beware in
  harness scripts that drive GPU work.
- The user's `run_vdi_dump_att.sh --git/--am` does a **force-push**; it has
  silently reverted my commits twice. Check `git log` after they use it.
- Never quote-nest `ssh 'docker exec ... bash -lc "..."'` with inner escapes;
  write a script file, `scp` it, `docker cp`, then run it.

---

## 2. Workload under study

```
tokens T=4096, topk=6 → 24576 routes, experts E=384 → exactly 64 routes/expert
model_dim H=7168, inter_dim I=768, contiguous_M=30720
GEMM1 (silu, GUGU): N,K = 1536,7168   GEMM2 (noact): N,K = 7168,768
A = MXFP8 E4M3, W = MXFP4 E2M1, per-1x32 E8M0 scales, FP32 acc, BF16 out
```

`64 routes/expert` is the key number — it is why `tile_m=64` wins, and it does
**not** generalise to other token counts.

Test command:
```bash
python op_tests/test_flydsl_grouped_gemm_gfx1250.py --scenario bench \
  --data-format a8w4 --layout gugu --experts 384 --tokens 4096 --topk 6 \
  --iters 100 --model-dim 7168 --inter-dim 768 --act silu --real-gemm --no-check-aot
```
Env: `ENABLE_CK=0 AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1
AITER_FORCE_GFX1250=1 HIP_VISIBLE_DEVICES=<n>`

Metric = the torch-profiler table's `device_time_avg` per kernel (NOT the CUDA
Event number, NOT e2e alone). Correctness gate = `rel_l2`, must be
**2.8725e-03**; `nan` means the run aborted on `logits_diff gate 0.01`.
**Always check `rel_l2` before reporting any timing** — I once reported a 2.9%
regression from runs that were producing NaN.

---

## 3. Tuning result (DONE, shipped)

Sweep driver: `my_code/sweep_tdm.sh <tag ...>` (tags defined inside; appends to
`my_code/sweep_tdm/summary.tsv`). Config is injected via env overrides I added
to `aiter/ops/flydsl/grouped_moe_gfx1250.py`:

```
AITER_TDM_TILE_M / _N / _K / _NUM_BUFFERS       (gemm1)
AITER_TDM_TILE_M2 / _N2 / _K2 / _NUM_BUFFERS2   (gemm2)
```

**Winner `g2_m64_nb3`: gemm1 `64x256x256_b3`, gemm2 `64x512x128_b3`.**

Same-batch medians (baseline = `16x256x256_b2` / `16x512x128_b2`):

| | gemm1 | gemm2 | e2e |
|---|---:|---:|---:|
| baseline | 894.0 | 621.9 | 2072.9 |
| g2_m64_nb3 | 380.5 (−57%) | 282.0 (−55%) | 1255.7 (−39%) |

On b8-2 later (different machine state): baseline 916.8 / 634.2, g2_m64_nb3
**205.4 / 189.8**, e2e 1002. Absolute numbers drift a lot between boxes/runs —
**always re-measure a same-batch baseline; never compare across batches.**

Why it works:
1. `tile_m=64` == rows/expert → one tile covers a whole expert → the weight slab
   is read once instead of 4× (per-unit-M TDM traffic 39040 → 12928 B/K-tile).
   It also turns on `WAVE_SPEC` (`tile_m>=64 and tile_n>=64`), so `TDM_PER`
   drops 4→2, staying inside the HW limit of 3 TDM in flight per wave (:14743).
2. `tile_m2` **must match** `tile_m`: `align_m = max(tile_m, tile_m2)`, so a
   16-row stage-2 tile re-reads its weights 4× over an already-64-aligned
   buffer. Fixing this took gemm2 534 → 282 us. I had predicted the opposite in
   a code comment; the measurement corrected me.
3. `num_buffers=3` deepens the pipeline. Note it *wins* despite halving
   occupancy to 1 block/CU — pipeline depth beats wave-slot concurrency here,
   because two co-resident waves just stall on `s_wait_tensorcnt` together.

Traps found: `tile_m=128` speeds gemm1 47% but wrecks gemm2 (+80%) via
`align_m`, net e2e **worse**. Selecting on one GEMM alone picks the wrong config
— the two are coupled through `align_m`, so **optimise on e2e**.

Not yet done: writing the winner into `aiter/configs/tuned_grouped_fmoe.csv`,
and checking whether it holds for other token counts.

---

## 4. Blacklisted config: `g1_n128_nb6`

`16x128x256_b6` **wedges the GPU** (MES stops answering REMOVE_QUEUE; ASIC reset
then fails; reboot required). Reproduced twice, the second time with a clean
`pkill -TERM`, so it is the kernel, not the kill.

Root cause **UNKNOWN**. Ruled out by experiment:
- not compile-time blowup — compiles in 7.7 s, *faster* than baseline's 8.4 s;
- not TDM over-subscription per se — a standalone repro kernel issuing 20
  `tensor_load_to_lds` + `s_wait_tensorcnt 0x10` runs clean in 6.9 s
  (this disproved my main hypothesis);
- LDS in range, control flow isomorphic to baseline, drain waits well-formed.

It has no performance value (its tile_n=128 line tops out at 496 us vs 378 for
`g1_m64_nb3`), so it is commented out in `sweep_tdm.sh`. **Do not re-run it.**

---

## 5. Current open task: the "wide-KSL" hot-loop rewrite (FAILING)

### Goal the user specified
Restructure the gemm1 K256 hot loop to:
```
1. load_b_and_scales(ksl=0)     ~12 physical DS (8 b128 + 4 ds_load_2addr_b32)
2. load all A0                  16 ds_load_b128
3. load_b_and_scales(ksl=1)     ~12 DS
4. issue next K-tile TDM        TENSORcnt, does not touch DScnt
5. s_wait_dscnt(12)             B0/SB0/SA0 + A0 ready; B1/SB1/SA1 may stay pending
6. load all A1                  16 DS, DScnt 12 → 28
7. execute KSL0                 16 × WMMAScale_16x16x128, hiding B1+A1 latency
8. s_wait_dscnt(0)
9. execute KSL1                 16 × WMMAScale_16x16x128
```

### Baseline pipeline it replaces (measured, correct)
```
1. load B0/SB0/SA0   12 DS   2. issue next TDM   3. load all A0   16 DS
4. s_wait_dscnt 0    avg 93.9 cyc
5. first 8 KSL0 WMMA + load B1 (8 DS)
6. load SB1/SA1 (4 DS) + A1 wm0 (4 DS)
7. remaining 8 KSL0 WMMA      8. load A1 wm1..wm3 (12 DS)
9. s_wait_dscnt 0    avg 74.1 cyc
10. 16 KSL1 WMMA
```

### Implementation
`compute_ktile_wide` in `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
(~line 403), selected by `USE_WIDE` (auto: `KWS==2 and _wide_vgpr<=512`;
override with `AITER_TDM_WIDE_KSL=0/1`). A leftover probe
`AITER_TDM_WIDE_WAIT` also exists.

### Status: **generates the right ISA but computes `rel_l2 = nan`**

The emitted ISA matches all 9 steps — verified line by line, both files are in
the repo:
- `my_code/gemm1_hotloop_annotated.s` — WIDE=1, 346 VGPR, **nan**
- `my_code/gemm1_hotloop_baseline_annotated.s` — WIDE=0, 281 VGPR, correct

Physical DS per K256 tile is 56 in both. `s_wait_dscnt 0xc` (=12) is emitted as
planned. Activation split confirms the two K-slices (WMMA 1–16 read v0..v63,
17–32 read v64..v127).

**Static checks that all came back CLEAN (so none of them is the bug):**
no VGPR/SGPR spill; 346 < the 1024 wave32 limit (:2250); A1's destinations are
read by WMMA 17–32 (not dead, not clobbered); no WMMA reads a register a later
load overwrites; accumulators overlap no load destination; wait immediates are
exactly 12 and 0.

**Deviation found (explains slowness, NOT the NaN):** LLVM hoists
`s_wait_dscnt 0` to after the *first* WMMA and inserts an extra
`s_wait_dscnt 0x13`, so the hiding window collapses from 16 WMMA wide to 1.

### Measured (b8-2, alternating runs)
| | gemm1 | gemm2 | e2e | rel_l2 |
|---|---:|---:|---:|---|
| WIDE=0 | 204.3 / 203.1 | 190.4 / 190.7 | 993.7 / 996.3 | 2.8725e-03 |
| WIDE=1 | 208.9 / 208.7 | 207.1 / 209.2 | 1020.5 / 1024.8 | **nan** |

WIDE timings come from wrong-result runs — they bound the schedule's cost, not
its value.

### Hypotheses I proposed and that were DISPROVEN — do not repeat these
1. "`mma_rows` scale args swapped" — they were already correct; my "fix" broke
   it and was reverted.
2. "mixed DS types make the partial wait unsafe" — separating them still NaN.
3. "sched_group order lets A1 clobber A0" — interleaving still NaN.
4. "accumulators overlap A1 registers" — **my analysis script dropped the
   VGPR-MSB comment.** In this ISA `v[66:73] /*v[322:329]*/` really means
   v322–329 (v256+ goes through VGPR-MSB indexing, doc §3.3.2.3 :2249). Any
   register analysis MUST parse the comment, not the 8-bit encoding.

I also left `s_wait_dscnt(8)` in the code for a while after hypothesis 2 was
disproven — the user caught it. When a hypothesis dies, revert its code.

### Suggested next step (not yet run)
Bisect rather than hypothesise: set both waits to `s_wait_dscnt(0)`. If accuracy
returns, the bug is in the partial-wait semantics; if not, it is in the reordered
loads. One run, binary answer.

**Worth knowing before investing more:** the ceiling here is small. The two
baseline waits cost 92.9 + 73.1 = 166 cyc/K-tile against 168 cyc of WMMA issue;
WMMA is 35.9% of wave cycles and LDS waits 17.0%. Even perfect hiding wins ~17%.
`BARRIER` at 14.5% (177.8 cyc/iter) may be the better target.

---

## 6. ATT trace analysis

Original bottleneck finding (baseline `t16x256x256_b2`, 144 valid waves):
`s_wait_tensorcnt` = **59% of all wave cycles**; WMMA only 6.3%. TDM completion
latency ~1270 cyc but `num_buffers=2` left only ~324 cyc of overlappable
compute → ~950 cyc/iter of pure stall. This diagnosis drove the whole retune and
was correct.

After the retune (`gemm_a8w4_m64nb3_att.wv1`, waves-per-eu=1,1, 36 waves):
WMMA 35.9%, WAIT_ds 17.0%, BARRIER 14.5%, WAIT_tensor 8.2%. The TDM problem is
solved; the profile is now compute-led.

Tools:
- `my_code/run_vdi_dump_att.sh <outdir>` — host-side; defaults to the
  `g2_m64_nb3` tiles, passes `AITER_TDM_*` and `AITER_TDM_WIDE_KSL` into the
  container, discovers kernel names **by pattern** (not hardcoded), clears the
  flydsl cache, logs the tile config into `logs/environment.log`.
- `cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/trace_segment_cycles.py` —
  the rule-sanctioned segment/compare tool; anchor sample points from
  `stats_ui_output_agent_*.csv` (trace ISA is ground truth, not a `.s` dump).
- ATT wall-clock is unreliable (ATT slows execution) — use it for instruction
  *cycle patterns*, and `--kernel-trace` for real timing.

Trace-parsing gotchas:
- `occupancy.json` reuses slots; match start/end pairs **by time**, keyed on
  physical `(SE,CU,SIMD,slot)`. The rule's stock script assumes one lifetime per
  slot and merges them wrongly.
- `code.json` row = `[ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency,
  Stall, Idle]`; wave JSON instruction = `[ts, type, stall, latency, code_idx]`.
- rocprof DB `arch_vgpr`/`sgpr` and ELF metadata VGPR/SGPR are different
  accountings — do not mix them.

---

## 7. Dumping ISA without running the kernel

`COMPILE_ONLY=1` returns right after codegen (`jit_function.py:1582`) while the
IR/ISA dump happens inside `_compile()` (`:895-906`), so they compose — full ISA,
kernel never reaches the GPU. `FLYDSL_DUMP_IR=1` also bypasses the disk cache
(`:1413`).

```bash
# on the box, inside the container, from the aiter root
for w in 1 0; do
  rm -rf ~/.flydsl/cache/*
  COMPILE_ONLY=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa_cmp/w$w \
  ENABLE_CK=0 AITER_FORCE_GFX1250=1 AITER_TDM_WIDE_KSL=$w \
  AITER_TDM_TILE_M=64 AITER_TDM_TILE_N=256 AITER_TDM_TILE_K=256 AITER_TDM_NUM_BUFFERS=3 \
  timeout 900 python my_code/compile_only_timing.py --child 64x256x256x3
done
tar czf /tmp/isa_cmp.tar.gz -C /tmp/isa_cmp .
```
Output: `<dir>/<kernel_name>/{00_origin.mlir … 21_final_isa.s}`.
Locally unpacked at `my_code/isa_cmp/{w0,w1}/` (not committed).

`my_code/compile_only_timing.py` also times cold compiles per config.

---

## 8. Files

Committed under `my_code/`:
| file | what |
|---|---|
| `sweep_tdm.sh` | tile/buffer sweep driver, all tags |
| `run_vdi_dump_att.sh` | ATT capture, host-side |
| `compile_only_timing.py` | cold-compile timing / `--child` compile of one config |
| `gemm1_hotloop_annotated.s` | WIDE=1 hot loop verbatim + plan as comments |
| `gemm1_hotloop_baseline_annotated.s` | WIDE=0 hot loop verbatim + plan as comments |
| `grouped_gemm_a8w4_gfx1250.md` | workload/layout/preshuffle reference (575 lines) |
| `gemm1_a8w4_tdm_silu_gfx1250_deep_dive.md` | GEMM1 deep dive (1778 lines) |
| `gemm2_a8w4_tdm_noact_gfx1250_deep_dive.md` | GEMM2 deep dive (1381 lines) |

Kernel source: `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
Dispatch/config: `aiter/ops/flydsl/grouped_moe_gfx1250.py`
Wrapper: `aiter/ops/flydsl/batched_gemm_mxfp4.py`
Preshuffle: `aiter/ops/shuffle.py`
Tuned CSV: `aiter/configs/tuned_grouped_fmoe.csv`

HEAD at handoff: `764b78ad9`.

---

## 9. Working preferences

- Reply in **Chinese**.
- Git commit messages ≤ 2 lines; detail goes in docs, not the commit body.
- Code comments ≤ 3 lines, never past 5.
- No approval needed before each commit.
- Cite file+line for HW claims; label **measured** vs **inferred** explicitly and
  retract when disproven.
