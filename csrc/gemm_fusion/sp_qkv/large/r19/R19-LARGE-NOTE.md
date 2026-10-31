# large/r19 — the r13/large path extended to local m ≤ 16384 (ABI 7; INTERNAL-ONLY)

Status: **source complete; device + host builds green in the campaign env and pinned-env Gate 1 (ISA/resources) PASS (§9.2). GPU Gate 2/3 pending (Python/profile wiring).** Tree = a copy of `large/r13` (r13g text: the item epilogue and its call sites are r12 text again, 0 spills) with an **ABI-generation bump** and the chunk-count extension. r13 and r18-put stay untouched as the rollback. Route A of the user decision: extend the fused large path natively to local m ≤ 16384 (T ≤ 65536 at world 4), not route B. The extended band ships as **24 bucket rows at 1024 stride** (§5.1).

Everything that is **not** in §2/§3 is byte-for-byte r13 semantics: numerics (`bf16_rne(fp32(fp32(acc*Ws)*As))`), the R12 protocol (credit tag + credit release → bounded credit wait with tag check → early arrival tag → sticky ERR → quietQueue → arrival release), the R16 data/geometry split (the kernel gets `m_real`; the plan's bucket M supplies tiles/chunks/grid/windows/K counts), the r13g S/R layout (S slab = bucket stride M, R = `[4·m_real, F]` compact, self tile straight into R for a full bucket, S[rank] + quiet_ready compaction for pad-M), the six GEMM entries (n1/n2/n4/n2h/n4h/n3q), the host entry points and the attribution knobs (except `LG_R13_ARGS`, see §3.4), and the order proofs.

## 1. Why a new tree and a new ABI generation

* A new tree keeps `large/r13` (shipped rows m 1024..8192) and `large/r18-put` (route B buckets 512/768) as rollback and as the A/B baseline; nothing in either is edited.
* The new image **cannot** share ABI 5/6: `LG_MAX_CHUNKS` is compiled into the device guard and the C-window layout, and the tag field widths change. r14 already owns `LG_COMM_ABI_VERSION 6` (split-K `sk` in the old pad word), so r19 uses **generation 7** and the header file `interface/lg_comm_abi_v6.h` (file name = previous struct generation, as r13's `lg_comm_abi_v4.h` = ABI 5).
* The launch-args struct is **unchanged in size and offset** (`LgCommArgsV6`, 360 B, `m_real` @352, `pad4` @356): only the 32-bit tag word and the C-window image change. A runtime can keep one ctypes struct layout and pick the ABI generation per source.

## 2. ABI 7 table (old ABI 5 → new ABI 7)

| item | r13 (ABI 5) | r19 (ABI 7) |
|---|---|---|
| `LG_COMM_ABI_VERSION` | 5 | **7** |
| header file | `interface/lg_comm_abi_v4.h` | **`interface/lg_comm_abi_v6.h`** |
| args struct | `LgCommArgsV4` 360 B | **`LgCommArgsV6` 360 B (same layout/offsets)** |
| `LG_MAX_CHUNKS` | 32 (M ≤ 8192) | **64 (M ≤ 16384)** |
| tag low word | `call_id[18:0] | (m_real-1)[31:19]` (13-bit m_real) | **`call_id[17:0] | (m_real-1)[31:18]` (14-bit m_real)** |
| `LG_TAG_CALLID_BITS` / `MASK` / `M_SHIFT` | 19 / 0x7FFFF / 19 | **18 / 0x3FFFF / 18** |
| C-window arrays | `[4][32]` | **`[4][64]`** |
| C-window layout | `DONE 512 / STATE 1024 / LOCK 1536 / REPORT 2048 / POSTCNT 2112 / CTAG 3072 / ATAG 3104` | **`DONE 512 / STATE 1536 / LOCK 2560 / FOLD 2620 / REPORT 2624 / POSTCNT 2688 / CTAG 3712 / ATAG 3744`** |
| `LG_R13_ARGS` knob | 1 = 360 B / 0 = 352 B | **removed** (the 352 B V3 layout cannot exist in ABI 7: m_real is unconditionally in the struct and the tag) |
| everything else | — | unchanged (frames, entries, protocol, K=6144, N/F, negctrl seams, `LG_QR_WGS`, `LG_TAG_EARLY`, `LG_POLL_BATCH`, …) |

## 3. Tag packing (E) and the call-id obligation

```
E = (STEP << 32) | ((m_real - 1) << 18) | (call_id & 0x3FFFF)          // 64-bit word written to ctag[]/atag[]
low 32 bits:  bits  0..17 = call_id + 1 byte boundary: 18 bits          (mask 0x3FFFF)
              bits 18..31 = m_real - 1:                 14 bits          (0..16383  => m_real 1..16384)
```

* `m_real` 1..16384 → all 14-bit codes; the 18-bit `call_id` keeps LOW bits only. The pair is injective over `(m_real, call_id mod 2^18)`; **the runtime's static call-id uniqueness gate must hold on 18 bits** (`callid.check_call_ids(..., tag_bits=18)`), not 19.
* Ranks that disagree on `m_real` present different tags (credit-tag check → `0x31`, arrival-tag check → `0x32`, as r13).
* Worked extremes: `m_real=1, call_id=0` → low word `0x00000000`; `m_real=16384, call_id=0x3FFFF` → `0xFFFFFFFF`; `m_real=8193, call_id=0` → `0x20000000` (bit 29 set), which is exactly what a 13-bit field could not represent — the new width is the point of the bump.

## 4. Control-window re-layout (the only place with real design content)

The r11..r13 protocol words (`EP 0`, `POISON 8`, `MAGIC 16`, `CREDIT 64`, `ARRIVAL 128`) keep their offsets — `ARRIVAL 128` deliberately, so a receiver written against the r13 contract (the small/r27 fused receiver's `SM_R27_ARRIVAL_OFF`) still finds its word if the two are ever composed. Everything sized by the chunk count moves; the new image is packed and every relation is a `static_assert` in the header:

| bytes | area | bytes | note |
|---|---|---|---|
| [0,8) | `EP` u64 | 8 | unmoved |
| [8,12) | `POISON` u32 (diagnostic mirror) | 4 | unmoved |
| [16,24) | `MAGIC` u64 | 8 | unmoved |
| [64,96) | `CREDIT[4]` u64 | 32 | **unmoved** (written by d into this window) |
| [128,160) | `ARRIVAL[4]` u64 | 32 | **unmoved** (r27 contract offset) |
| [512,1536) | `chunk_done[4][64]` u32 (r6x poster group counters) | 1024 | production `done[4][64]` — the lead's collision-free requirement |
| [1536,2560) | chunk record `state[4][64]` u32 (r5b-combining path only) | 1024 | debug/control path |
| [2560,2576) | submit `lock[4]` u32 (r5b path only) | 16 | |
| [2620,2624) | fold counter u32 (`LG_R8_FOLD` path only; `LOCK_OFF+60`) | 4 | |
| [2624,2688) | `report u64[8]` | 64 | moved from 2048 |
| [2688,3712) | `post_count[4][64]` u32 (`LG_DEBUG_POSTCOUNT=1`) | 1024 | cumulative debug counter |
| [3712,3744) | `CTAG[4]` u64 | 32 | moved from 3072 |
| [3744,3776) | `ATAG[4]` u64 | 32 | moved from 3104 |
| [4096,4288) | quiet-ready stamps u64[3][8] (`LG_STAMPS` only) | 192 | clear of the tags |
| [4352,4384) / [4416,4424) / [5120,5504) | XCCQ / first-put / poster stamps (debug) | 32/8/384 | clear |
| [65536,…) | per-WG stamp records (`LG_STAMPS` only; needs C ≥ 65536+64·grid) | — | grid at M=16384 ≤ 2625 blocks ⇒ 233,536 B ≤ the 4 MiB C view |
| `LGC_BYTES` | image initialized by `lg_comm_init_control` | 8192 | all production offsets < 3776 |

`static_assert` set (header, enforced at device + host compile): `DONE==512`, `STATE==DONE+4·4·64`, `LOCK==STATE+4·4·64`, `LOCK+16≤FOLD`, `FOLD+4≤REPORT`, `REPORT+64≤POSTCNT`, `POSTCNT+4·4·64==CTAG`, `CTAG+32==ATAG`, `ATAG+32≤QSTAMP`, stamp areas in order, `LGC_BYTES≤STAMP_OFF`. `LG_DEBUG_POSTCOUNT` builds read/write `4·LG_MAX_CHUNKS` entries (the host readback size is updated accordingly). Runtime consumers must take `CTAG`/`ATAG` from `lg_abi_info_v1` `out[5]/out[6]` (3712/3744 for ABI 7), never hardcode r13's 3072/3104.

## 5. Chunk geometry at the new range (M multiple of 256; chunks = M/256 ≤ 64)

The r13 GEMM is M-linear and needs no new tiles/entries: one 512-thread WG = one 256×256 output tile over the full K=6144 (48 K-steps of 128, LDS 135168 B, VGPR 253–255 → 1 WG/CU). One HSACO covers every M (chunks is a runtime argument); the 8192 cap lived only in the ABI/host/profile, which this tree raises.

| local m | chunks | dense items / grid / 256-CTA rounds | indexer items / grid / rounds |
|---:|---:|---|---|
| 1024 | 4 | 144 / 145 / 1 | 164 / 165 / 1 |
| 2048 | 8 | 288 / 289 / 2 | 328 / 329 / 2 |
| 4096 | 16 | 576 / 577 / 3 | 656 / 657 / 3 |
| 8192 | 32 | 1152 / 1153 / 5 | 1312 / 1313 / 6 |
| 10240 | 40 | 1440 / 1441 / 6 | 1640 / 1641 / 7 |
| 12288 | 48 | 1728 / 1729 / 7 | 1968 / 1969 / 8 |
| **16384** | **64** | **2304 / 2305 / 10** | **2624 / 2625 / 11** |

Plan legality is unchanged: n ∈ {1,2,4} with `chunks % n == 0`; half_first=1 needs n≥2; quarter-first (n=3, half_first=2) only dense M=2048 (chunks 8). chunk counts that are not divisible by 4 can only run n=1 (or 2), exactly as in the 1024..8192 grid. half_first=1 at chunks 64 adds 27·(64/n) half items (n=4 → +432 ⇒ 11 rounds dense / 12 indexer).

### 5.1 Shipped Route-A extended band: **1024-stride bucket rows** (Lead ruling)
The extended band is profiled as **8 bucket rows per mode = 24 rows total**: local m **9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384** (T = 4·m, so 36864..65536 step 4096). The band floor is **8193**: a real local m 8193..9215 pads into the 9216 bucket (a row's own `pad_min` / the profile's `m_pad_min_large` gate applies as for the 1024..8192 grid); m ≤ 8192 stays on the r13 grid and m > 16384 is native. Profile/window accounting: 24 large owners' rows, arena 788 MiB, S = R = 324 MiB (indexer M=16384 is the largest window), C = 4 MiB, gf.global = 4 MiB (see §6).

Why 1024 and not 256 stride (the architect's decisive argument): with pad-M the GEMM computes only the live rows, so a 1024-wide gap costs little; but the plan's puts-per-peer `n` is legal only when `chunks % n == 0` (`chunks = M/256`). At 1024 stride `chunks ∈ {36,40,44,48,52,56,60,64}` — **always a multiple of 4, so n = 4 stays legal and the in-GEMM put/GEMM overlap is preserved**; at 256 stride the odd-chunk buckets (33, 35, 37, …) force n = 1 and lose that overlap. The much coarser row set is therefore a protocol/overlap decision, not a coverage compromise: coverage is carried by pad-M buckets, not by an exact row per multiple of 256.

Gate-3 sweep scope follows from this: measure **each of the 24 bucket rows** at a few real m per bucket — at least the exact bucket M and one pad value near the bucket floor (preferably one just above the band floor, e.g. 8193/8449, and one mid-bucket pad) ⇒ **≤ 48–72 timing points** total (not a 96-cell × candidates grid). Keep **n = 4 as the default where legal** (it is, on all 24 buckets); only test n = 2 / n = 1 plus `half_first = 1` if those numbers suggest a win. Correctness (Gate 2) is independent of the sweep and additionally includes a pad case just above the band floor (m 8193 or 8449) to exercise the 8193 floor.

## 6. Memory (unchanged formulas; the new maximum)

* S window (staging) = `[4][M][F]` bf16 = **8·M·F B** with the **bucket** stride (pad-M launches keep it; the dead tail rows stay inside their slab and are never sent).
* R window = `[4·m_real][F]` bf16 = **8·m_real·F B** (≤ 8·M·F), compact.
* Window bytes = `round_up(max(8·M·F + 4096, 327680), 4 MiB)` (lg_window_bytes_v1 / variants_large helper).
* M=16384: dense S=R=301,989,888 B → 292 MiB registered; indexer S=R=335,544,320 B → **324 MiB registered each**. Current largest plan (M=8192 indexer) was 164 MiB each; the shared-arena profile grows by **+320 MiB ⇒ 468 → 788 MiB** (`S=R=324 MiB, C=4 MiB, gf.global=4 MiB, small windows 132 MiB`). Per-rank VMM heap stays 4 GiB (arena+256 MiB slack < 4 GiB). Reuse-view boundary at T=65536: last byte 301,990,400 ≤ R+pad 301,993,984 (fits, 3.5 KiB slack) — the same LG_R_PAD_BYTES mechanism as r13.
* Nothing in the large kernels needs the arena policy changed; the Python side must add a T=65536 row to `specs.LARGE_CELLS` (window budget) and regenerate the profile.

## 7. Proof strategy (why the chunk-count generality is still safe at 64)

`device/lg_r6x.hpp` carries three independent layers; the static checker `tests/check_r19_static.py` is the independent L3 mirror:

* **L1 (compile time, every chunks 1..64, every `<NP,HG>`):** structural identities — the put groups tile `[0,chunks)` consecutively and non-empty; each group's arrival target equals `group_nchunks·LG_ITEMS_PER_GROUP` (+ the r7 half-first `NPD` per group-0 chunk); the r13 live-row sum/prefix law at sample rows of the pad-M tail (the general m follows because the live sets are clamped intersections of consecutive ascending ranges).
* **L2 (compile time, exhaustive per item):** the historical r13 assert sets `{8,16,32}` are unchanged; the new maximum `64` is added for the r6x n=1 plan (legal at every chunk count) and the r7 n=2 half-first plan, plus the n1/n2/n4/n2h/n4h live-row checks at 64. The real `select_item_*` / `r13_live_rows` code is enumerated exhaustively at the old and the new boundary.
* **L3 (host mirror, exhaustive):** `tests/check_r19_static.py` re-implements the device formulas independently and enumerates **every** (chunks 1..64 × every legal plan) × every `m_real` 1..chunks·256, plus the r6x/r7/r9 bijection/coverage/arrival properties for **both builds** (dense and indexer), the tag packing over the full 1..16384 × 2^18 domain, the C-window layout partition and the address model. It passed at source freeze (see §9).
* **Safety argument:** the device's only runtime geometry input is `args.chunks`; the entry guard admits exactly `chunks == M/256`, `chunks ≤ LG_MAX_CHUNKS` and the plan-shape leg (`chunks % NP == 0`, or `chunks == 8` for n3q). The plan template `<NP,HG>` is compile-time. L1+the guard prove the structural laws for every admissible chunks, L2 checks the item mapping exhaustively at both boundaries, L3 covers the intermediate counts independently.
* **Constexpr budget:** L1 is cheap but L2 is a heavy template evaluation; `build_device.sh` raises clang's default 1,048,576 constexpr steps with `-fconstexpr-steps=20000000`. The device build command in `jit.large_device_cmd` (Python) must carry the same flag for src r19 — otherwise the JIT build of the same source fails where the recipe succeeds. (The r19 device build in §9.2 needed it; the flag is in the frozen recipe and its `build-command.txt`.)

## 8. Host surface (unchanged; the M cap moves)

Same exports as r13: `lg_attach_v1`, `lg_launch_v2` (m_real = M), `lg_launch_v3(..., m_real, ...)`, `lg_gpage_attach_v1`, `lg_abi_info_v1/v2`, `lg_info_v1`, `lg_window_bytes_v1`, `lg_init_control_v1`, `lg_report_v1`, `lg_drain_v1`, `lg_free_v1`, `lg_comm_R_ptr`, debug readers. No put entry, no `lg_swin_bytes_v1`, no `lg_launch_v4` (those are r14/ABI-6 only; the shared `host/lg_fusion_driver.cpp` compiles the r14 split-K block only for `LG_COMM_ABI_VERSION == 6`, and maps `LgCommArgsCurrent = LgCommArgsV6` for `>= 7`).
`lg_attach_v1` accepts `M` multiple of 256, `M ≤ LG_MAX_CHUNKS·LG_CHUNK_ROWS = 16384`; `lg_launch_v3` accepts `1 ≤ m_real ≤ M`; the device guard additionally rejects `m_real == 0`, `m_real > M`, `pad4 != 0`, `chunks > 64`, `chunks != M/256`, plan-code mismatch and shape mismatches (sticky CONFIG).
Error codes unchanged: `-100` chunks%n, `-101` bad n, `-102` half-first n=1, `-103` n3q shape/dense, `-104` call_id ≥ 2^31, `-105` m_real outside [1,M], `-110` no ERR/STEP page, `-106/-111/-112` do not exist here (put-only).

## 9. Build + checks

Recipes: `build_device.sh` (gfx950 code object + HSACO + resource usage + `.s`), `host/build_host.sh` (plain `clang++ -x c++`, libmori_cco-linked `.so`). Same `-D` set as r13 except `-DLG_R13_ARGS` removed and `-fconstexpr-steps=20000000` added to the device command. `lg_quiet_ready_v2` resets `4·64` counters and fills at most `8·m_real·F/16` vectors under ERR.

### 9.1 Source identity
`SOURCE_SHA256.txt` = sha256 of every tree file (15 entries at freeze; see the Lead report for the exact hash). `tests/check_r19_static.py --write-manifest` regenerates it; the build recipes verify it with `sha256sum -c` before compiling. The static checker passes (`R19 STATIC ALL PASS`).

### 9.2 First build (development host, ROCm 7.2.0 clang 22.0.0git) — **BUILD GREEN**
* Device: `LG_DEVICE_BUILD_OK target=hipv4-amdgcn-amd-amdhsa--gfx950`.
  * First dev build (dir `r19-dev-20261029-2005`): `lg_gemm_quad_fp8_comm.hsaco` = `172c4fddf36ce7bf7ef6bb15840ca4d5506824ebd2da1ed99ad1190adfaeb6ce` (205,024 B), `.so` = `9f04cb99366bc0c8c19293f1da0517801526a7b59cdf0926fb2af45b0a7c93a5`.
  * Frozen-tree re-verification (dir `r19-final-20261029-2000`, same source, manifest verified with `sha256sum -c`): HSACO = `d89d522d41228d4cafcf90f662a75d692bd86b2037e02123c21cb608aaee6004`, `.so` = `4760fd7200c9352b09fa76001a77fca2b022252b2fb166eda11fcc1eed97b3be`. The two directories' artifacts differ ONLY through clang's path-derived `__hip_cuid` CU id: the emitted `.s` diff contains nothing but those lines and the per-entry resource table is identical (`RESOURCE_IDENTICAL`), so the code is the same. Each build's `gen/lg_expected_hashes.h` pins its own artifact for the runtime's H-3 exact-image gate (the Python JIT build generates the same pair it loads).
  * per-entry resource usage (identical to an r13 baseline built with the same host toolchain, which was built in the same session for attribution):
    n1 = VGPR 253 / AGPR 0 / SGPR 91 / VGPR-spill 0 / scratch 0 / LDS 135168; n2, n4, n2h, n4h = VGPR 254 / AGPR 0 / SGPR 106 / VGPR-spill 0 / scratch 0 / LDS 135168; n3q (indexer trap stub) VGPR 0; quiet_ready VGPR 26 / LDS 4; drain VGPR 10. **0 VGPR spills, 0 scratch, 0 private segment** on every entry; the extension adds no register/LDS cost vs r13.
  * `s_swappc_b64`: 2 in each of n2/n4/n2h/n4h = **8 total in the indexer build** (a dense build would be 10: n3q is real there; r19 matches r13 in both families). This is the **accepted r13-shipped baseline**, not an r19 artifact: r12 had 0, and r13's live-row poster code pushed MORI's *non-forced* `ccoSdmaDispatchPut<ccoCoopThread,…>` over the inline threshold (callee resolved in `records/evidence-owner/stage3-evidence/r16-r13-isa/r13c.call-targets.txt`; accepted with a measured ≤ +1.1 % timing cost at the anchors — `R16-PAD-M-REPORT.md` §4: "the r13 object contains 10 s_swappc_b64 (dense) / 8 (indexer) where r12 has none"). The r13d "expected: no s_swappc_b64" line was a hypothesis, not the shipped object. Evidence's pinned 7.2.4 Gate 1 confirms 8/8 r19 vs r13 ⇒ no count change and no new instruction class; the ISA clause means "no regression vs the accepted r13 baseline".
  * SGPR spills 13/20 on n2/n4/n2h/n4h are also byte-identical to the r13 baseline (r13's own property, pinned build included); VGPR spill/scratch gates are clean.
* Host: `LG_HOST_BUILD_OK`; `liblg_comm.so` sha256 = `948580569faa82da85d7fa01a04f6735b5f70ca39c4f1a2e0a05807d4bc8b39b` (18,456 B), one pre-existing `-Wunused-value` warning in the r13 debug readback (`lg_comm_read_stamps`, unchanged).
* Shared `host/lg_fusion_driver.cpp` smoke-compiled against **r19 (ABI 7), r13 (ABI 5), r14 (ABI 6) and r18-put (ABI 5 + put)**: all rc=0, each exports the 13 expected symbols; r14 still exports `lg_launch_v4`/`lg_swin_bytes_v1`, r18-put still exports the four `lg_put_launch*` entries, r19 exports no put/split-K entries. So the `== 6` split-K branch narrowing and the new `>= 7` mapping did not disturb the older generations.
* Artifacts left on the dev host: `/data/blyu/sessions/sp4-opus-mori-20261012-091456/build/r19-dev-20261029-2005/` (build-device/, build-host/, build-r13/, shared/).
* **Pinned-environment Gate 1 (Evidence, commit `e8c196b6`, campaign pinned container `atom-dev@sha256:566aa3dc…`, ROCm 7.2.4 / HIP 7.2.53211 / clang roc-7.2.4 26084): PASS.** Manifest `841b1948…` verified in-container; r19 vs r13 baseline built in the same session: all 8 entries resource-identical (VGPR 253/254, SGPR 91/106, 0 VGPR spills, scratch 0 B/lane, private segment 0, LDS 135168, AGPR 0); ISA structural counts identical (`v_mfma_scale_f32_16x16x128_f8f6f4` 1216, `ds_read_b128` 1200, `buffer_load_dwordx4` 496, `s_swappc_b64` 8, `s_sleep` 51, `s_memrealtime` 32); the `.s` diff begins with exactly the designed constants (`LG_COMM_ABI_VERSION 7 vs 5`, `LG_MAX_CHUNKS 64 vs 32`, the moved control-window store offset) and is otherwise label renumbering — no new instruction class. Shared driver pinned check: ABI 5/6/4 surfaces unchanged (26/30/30 exports incl. r14 split-K and r18-put entries), r19 ABI 7 = 26 exports = exactly the r13 surface. Record: `records/evidence-owner/stage3-evidence/R19-GATE1-ISA-BUILD.md`.

## 10. What the Python/profile side and Evidence owe this tree

* Python (aiter-integrator, his scope): `jit.LARGE_ABI["r19"] = 6`; `large_rt` gen 6 → `(LG_COMM_ABI_VERSION 7, 360 B, LgCommArgsV6)`, `LG_TAG_CALL_BITS = 18`, `LG_M_REAL_MAX = 16384`, `LG_CTAG_OFF/LG_ATAG_OFF` per generation (3712/3744 for ABI 7); `LARGE_SRCS`/`LARGE_PAD_M`/`LARGE_SRC_SHA["r19"]` = this manifest; `variants_large` M range [256,16384]; `specs.LARGE_CELLS` T=65536 row (arena 788 MiB) + the **24 extended-band rows (§5.1: 1024-stride M 9216..16384 × 3 modes, band floor 8193, n = 4 default)** with placeholder (n,half_first) until the sweep; `callid` tag_bits=18; profile-rule tests' M universes.
* Evidence: (a) a fresh 4-rank native-vs-fused correctness pass at m 8448/10240/12288/16384 **plus a pad case just above the band floor (m 8193 or 8449)** (byte-exact `--exact`-equivalent, `.02`, caches, pad-M odd m, eager+graph, dead-chunk/`m_real<M` self-compaction path); (b) the **§5.1 Gate-3 sweep**: the 24 bucket rows × a few m per bucket (exact M + ≥1 pad near the floor) ⇒ ≤48–72 timing points, n = 4 default, n = 2/1 + half_first only if suggestive — not a 96-cell grid; (c) native baseline at T = 33792..65536 (current anchors stop at T=32768); (d) the wire/receiver model re-anchor (3·m·F·2 at the measured rate; receiver 8+0.0117·T is [inf] beyond T=32768). No new kernel entries/tiles are needed, so the compile budget is 2 device builds (dense/reuse + indexer) for the new source.

## 11. Risks / open points

* ~~The device build needs the pinned campaign toolchain (ROCm 7.2.4 container) for certifiable ISA numbers~~ — **DONE**: Evidence's pinned-env Gate 1 PASS (§9.2), r19 == r13 baseline on resources/ISA; the dev-host run stays as the pre-gate smoke record.
* Gate 2/3 are GPU-only and blocked only on the Python/profile wiring commit (aiter-integrator's files): correct fused-vs-native numerics at m 8448..16384 incl. pad-M, the band-floor pad case (8193/8449) and the `m_real<M` self-compaction; the §5.1 24-bucket / ≤48–72-point (n, half_first) sweep (n = 4 default); and native B at T 33792..65536.
* `LG_DEBUG_POSTCOUNT` debug builds now write `post_count[4][64]`; the host readback size changed in `lg_comm_read_postcount`. Non-debug builds unchanged.
* The `LG_R8_FOLD` path is dead code in production (`build_device.sh` uses LG_R8_FOLD=0 and the kernel static_asserts `!LG_R8_FOLD`); its fold counter address moved to 2620 but stays valid.
* `-fconstexpr-steps=20000000` is a build-recipe addition (device only). It is harmless for clang, but the Python JIT device command must match (see §7).
* GPU-only unknowns (cannot be closed on CPU): the ISA gate numbers on the 7.2.4 container toolchain, the quiet_ready reset-loop cost at 64 counters (expected negligible, 256 stores), and everything in §10.
