# HIP `kn_mla_reduce_v1` — full design dissection

Source: `csrc/kernels/mla/reduce.cu` (host entry `mla_reduce_v1`, line 1024).
Target: MI300X / gfx942 (CDNA3), warp size **64**. This is the stage-2 epilogue of
split-KV MLA decode: an LSE-weighted online-softmax combine of per-split partials.
**Pure reduction — no MFMA, memory-bandwidth bound.**

---

## Variable / symbol glossary

### Compile-time Traits (`MlaReduceKernelV1Traits`, reduce.cu:13)


| symbol                 | value (gfx942)                     | meaning                                                                           |
| ---------------------- | ---------------------------------- | --------------------------------------------------------------------------------- |
| `kSizeDV`              | e.g. 512                           | head value-dim `Dv` — width of one output row in floats                           |
| `kNumHeadQ`            | e.g. 128                           | number of Q attention heads `H`                                                   |
| `kNumWarps`            | `2`                                | **wavefronts per workgroup** (block geometry, NOT waves/EU)                       |
| `kNumThreads`          | `kNumWarps·64 = 128`               | threads per workgroup                                                             |
| `kOccupancy`           | `8`                                | **min resident workgroups per CU** (2nd arg of `__launch_bounds__`) → ~4 waves/EU |
| `kNumThreadGroupPerBh` | 1/2/4/8/16/64/256                  | WGs fanned out per (batch,head) to fill CUs on small batches                      |
| `kMassiveThreshold`    | `4`                                | `num_splits ≥ this` → massive path; else simple path                              |
| `kVecWidth`            | `kSizeDV / kNumThreads` = 4 @Dv512 | **floats each thread owns** of the output row (lane partition)                    |


### Runtime / loop variables


| symbol                                | meaning                                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `num_splits`                          | `reduce_indptr[tile+1] − reduce_indptr[tile]` — partials to merge for this tile                |
| `reduce_tile_start/end`               | `[start,end)` slice into `reduce_partial_map` for this tile (held in SGPR via `readfirstlane`) |
| `head_idx` / `block_idx` / `tile_idx` | the 3 logical work dims (head, q-pos-group, reduce-tile)                                       |
| `seq_idx`                             | q-position the WG is currently combining (WG strides these by `kNumThreadGroupPerBh`)          |
| `local_seqlen_idx`                    | `seq_idx − q_start`, offset into the partial buffers                                           |
| `work_idx`                            | flat work index in the persistent kernel (`blockIdx.x`, grid-strided)                          |
| `tot_work`                            | `H · kNumThreadGroupPerBh · num_reduce_tile` — total work items                                |
| `lane_idx`                            | `opus::lane_id()` = `threadIdx.x % 64`, lane within a wavefront                                |
| `num_lse_per_thr`                     | LSEs each lane of warp 0 handles (split-count / 64, rounded up)                                |


### Per-thread compute state


| symbol                    | meaning                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------- |
| `reg_out`                 | `vector_t<float,kVecWidth>` — this thread's slice of the output accumulator (VGPR) |
| `max_lse`                 | running max LSE across splits (online-softmax)                                     |
| `sum_e` / `sum_e_lse`     | running denominator `Σ exp(LSE_i − max)`                                           |
| `old_scale` / `new_scale` | rescale factor for running accum / weight for the new partial                      |
| `global_lse`              | final merged LSE for the (seq,head): `log(sum) + max`                              |
| `lse_scale[s]`            | **pre-normalized** per-split weight `exp(LSE_s − global_lse)` (massive, in LDS)    |


### Buffers / pointers (`MlaReduceKernelV1Params`, reduce.cu:95)


| symbol                 | shape / dtype                | meaning                                           |
| ---------------------- | ---------------------------- | ------------------------------------------------- |
| `p_partial_output`     | fp32 `[row,H,Dv]` contiguous | stage-1 partial outputs `O_i`                     |
| `p_partial_lse`        | fp32 `[row,H]` contiguous    | stage-1 partial LSEs                              |
| `p_reduce_indptr`      | int32 `[#work+1]`            | CSR offsets into `reduce_partial_map`             |
| `p_reduce_partial_map` | int32 `[indptr[-1]]`         | split → partial-row **gather index**              |
| `p_reduce_final_map`   | int32 `[#work,2]` opt        | `{q_start,q_end}` per tile (`MlaPartialTileInfo`) |
| `p_final_output`       | bf16/fp16 `[bs,H,Dv]`        | merged output (runtime strides `stride_s_o/h_o`)  |
| `p_final_lse`          | fp32 `[bs,H]` opt            | merged LSE                                        |
| `max_splits`           | = `num_cu`                   | sizes the LDS scratch arrays                      |


### LDS scratch (reduce.cu:425)


| symbol                     | meaning                                                               |
| -------------------------- | --------------------------------------------------------------------- |
| `p_lds_reduce_partial_map` | the tile's gather indices, staged gmem→LDS once, reused everywhere    |
| `p_lds_lse_scale`          | the `lse_scale[s]` array warp 0 publishes to both warps               |
| `p_lds_local_lse`          | overflow store for LSEs beyond 256 splits (`kUpToLdsLimit` tier only) |


---

## 0. TL;DR of the design

- One **workgroup = 128 threads = 2 wavefronts**, `kOccupancy=8` WGs resident per CU.
- A **"work item" = (head, q-position-group, reduce-tile)**. Each work item combines
`num_splits` partials for one (seq, head) into one final output row of `Dv` floats.
- The `Dv`-wide output vector is **lane-partitioned**: 128 threads each own
`kVecWidth = Dv/128` contiguous floats (=4 for Dv=512). No cross-thread comms on the
output axis — the only reductions are along the *split/LSE* axis, done by warp 0.
- **Two algorithms** chosen at runtime by `num_splits`:
  - `num_splits ∈ {2,3}` → **simple**: sequential register online-softmax merge.
  - `num_splits ≥ 4` → **massive**: precompute normalized per-split scales in LDS, then
  a **software-pipelined weighted accumulate** (double-buffered loads).
- **Two launch shapes** chosen at host: 3-D **grid-launch** (work fits HW) or 1-D
**persistent** (`_ps`, grid-stride over work items).

---

## 1. The work decomposition (logical → grid)

### 1.1 Indexing scheme (CSR over splits)

- `reduce_indptr[t .. t+1]` gives `[start,end)` into `reduce_partial_map`; the slice is the
list of partial-row indices belonging to reduce-tile `t`. `num_splits = end − start`.
- `reduce_partial_map[i]` is a **gather index** into `partial_output`/`partial_lse` rows.
- Final q-range for the tile comes from `reduce_final_map[t] = {q_start,q_end}` (a 2-int
union `MlaPartialTileInfo`, mla.h:26), or — when absent — is derived as
`qo_len = map[1]−map[0]; q_start=t·qo_len`. This handles both uniform and varlen qo.

### 1.2 Three nested logical dims

Total work = `kNumHeadQ × kNumThreadGroupPerBh × num_reduce_tile`:

- **head_idx** ∈ `[0, H)` — independent per attention head.
- **block_idx** ∈ `[0, kNumThreadGroupPerBh)` — splits the q-positions of a tile across
multiple WGs so short batches still fill the machine (see §5, `get_num_work_group_per_bh`).
- **tile_idx** ∈ `[0, num_reduce_tile)` — which reduce-tile.

Each WG then **strides over seq positions** inside its assigned (tile, block):
`for seq_idx = q_start + block_idx; seq_idx < q_end; seq_idx += kNumThreadGroupPerBh`.

---

## 2. Block & grid layout

### 2.1 Block (workgroup) — fixed


| qty            | value                                           | where                 |
| -------------- | ----------------------------------------------- | --------------------- |
| threads/WG     | `kNumThreads = kNumWarps·64 = 128`              | Traits:18-19          |
| warps/WG       | `kNumWarps = 2`                                 | Traits:18             |
| occupancy      | `kOccupancy = 8` (`__launch_bounds__(128,8)`)   | Traits:20, kernel:670 |
| lane partition | `kVecWidth = Dv/128` floats/thread (=4 @ Dv512) | Traits:23             |


`static_assert(Dv % 128 == 0)` guarantees the output vector divides evenly across lanes.

**Wave cooperation (important — not one wave per block).** The 2 wavefronts of a WG
**cooperate on the *same* work item**; they do NOT each take a separate block. Proof in
code: the output partition `threadIdx.x * kVecWidth` (reduce.cu:320,598) runs `threadIdx.x`
0..127 across *both* warps, slicing one `Dv` row into 128 pieces. In the massive path,
**warp 0 alone** computes `lse_scale[]` (gated `threadIdx.x/64 == 0`, reduce.cu:210), an
`s_barrier` publishes it (reduce.cu:521), then **all 128 threads** of both warps do the
wide output accumulate. The split/LSE-axis reduction needs only one 64-lane warp; the
second wave exists to halve `kVecWidth` (4 not 8 @Dv512) for better latency hiding on the
output axis. Axis split: **LSE axis → warp 0 only; output (`Dv`) axis → all 128 lanes.**

### 2.2 Grid — two modes (host picks)

**Grid-launch** `kn_mla_reduce_v1` (line 773), used when total work ≤ `num_cu·8·2`:

```
grid = dim3(kNumHeadQ, kNumThreadGroupPerBh, num_reduce_tile)   // (x,y,z)
blockIdx.x = head, blockIdx.y = block_idx, blockIdx.z = tile_idx
```

One WG per work item; no internal work loop. Simplest mapping.

**Persistent** `kn_mla_reduce_v1_ps` (line 669), used when work exceeds that:

```
grid = dim3(num_cu · kOccupancy · 2)        // 1-D, sized to the machine
work_idx = blockIdx.x;  then  work_idx += gridDim.x   (grid-stride)
// unflatten: head = w % H; tmp = w / H;
//            block_idx = tmp % NTG; tile_idx = tmp / NTG
```

Early-exit sentinel: if `reduce_indptr[tile] == reduce_indptr[num_reduce_tile]` (the last
index), the work item is past the end → `return false`, loop breaks. `readfirstlane` on the
indptr loads keeps `tile_start/end` in SGPRs (scalar, uniform across the WG).

The persistent form amortizes launch overhead and keeps all CUs busy on large
`num_reduce_tile` without oversubscription; there's a `s_barrier` between iterations
(line 761) since the LDS scratch is reused each work item.

---

## 3. Data layout & staging across the memory hierarchy

### 3.1 Global memory layouts (assumed/used)

- `partial_output` fp32 `[row, H, Dv]` **contiguous** → strides d=1, h=Dv, row=Dv·H.
- `partial_lse` fp32 `[row, H]` **contiguous** → stride h=1, row=H.
- `final_output` bf16/fp16 `[bs,H,Dv]` — **runtime strides** `stride_s_o, stride_h_o`
(NOT assumed contiguous; passed from `final_output.stride()`).
- `final_lse` fp32 `[bs,H]` → stride h=1, row=H.

All gmem access goes through `opus::make_gmem<T>(ptr)` → **buffer_load/store with an SGPR
base descriptor** built once from the kernel-arg pointer (line 467+). Because the pointer
is a uniform kernel arg, there is **no waterfall loop** — the address is scalar, per-thread
offset is added as a vector byte-offset. `buf_load_vec`/`buf_store_vec` (lines 34/65) split
a `kVecWidth` access into ≤16B buffer ops (`kMaxBufVec<T> = 16/sizeof(T)`), so a 4-float
load is one 16B `buffer_load_dwordx4`.

### 3.2 LDS (shared) staging

Allocated dynamically (`extern __shared__ int32_t p_lds[]`), partitioned (line 425):

```
[ reduce_partial_map : max_splits × i32 ]
[ lse_scale          : max_splits × f32 ]
[ local_lse overflow : max(0, max_splits−256) × f32 ]   // massive ≤LdsLimit tier only
```

`max_splits = num_cu` (host, line 1070). LDS is **tiny (a few KB)** — this is what lets
`kOccupancy=8` coexist. Host validates `lds_size ≤ maxSharedMemoryPerMultiProcessor` and
warns if it would cut occupancy (lines 935-941).

Two things are staged into LDS:

1. `**reduce_partial_map`** for the tile: cooperatively copied gmem→LDS by all 128 threads
  (`for i = tid; i < num_splits; i += 128`), then `s_waitcnt(0); s_barrier()` (line 437).
   Reason: the gather indices are reused `num_splits` times across both the LSE pass and
   the output pass, and by every seq position — read once from gmem, reuse from LDS.
2. `**lse_scale[s]**` = normalized per-split weight `exp(LSE_s − global_LSE)`, computed by
  warp 0 and broadcast to all lanes via LDS (massive path only).

### 3.3 Register staging

- Output accumulator `reg_out` = `vector_t<float, kVecWidth>` lives entirely in VGPRs;
each thread owns its `kVecWidth` slice of the `Dv` output for the whole merge.
- In the massive ≤256-split tiers, the **per-split LSEs live in VGPR** (`LocalLse`,
line 128): `kUpTo64` keeps a single scalar, `kUpTo256` a `float[4]`; only the
`>256` tier spills the tail to LDS. This keeps the hot LSE reduction register-resident.

---

## 4. The two compute paths

### 4.1 Simple path (`mla_reduce_v1_impl_simple`, line 538) — `num_splits ∈ {2,3}`

Classic **online-softmax merge in registers**, one pass over splits:

```
reg_out = load(split0);  max_lse = lse0;  sum_e = 1
for s in 1..num_splits:
    o_s = load(split_s);  lse = lse_s
    new_max   = max(max_lse, lse)
    old_scale = exp(max_lse − new_max)        // rescale running accum
    new_scale = exp(lse    − new_max)         // weight new partial
    reg_out   = old_scale·reg_out + new_scale·o_s     // FMA over kVecWidth
    max_lse   = new_max
    sum_e     = sum_e·old_scale + new_scale
reg_out *= 1/sum_e                             // single normalize at end
store(cast<out_t>(reg_out));  optionally store final_lse = log(sum_e)+max_lse
```

No LDS scales, no warp reduction — every thread does the full scalar LSE bookkeeping
redundantly but on tiny `num_splits`, which is cheaper than synchronizing. Output FMAs are
`static_for<kVecWidth>` unrolled.

### 4.2 Massive path (`mla_reduce_v1_impl_massive`, line 416) — `num_splits ≥ 4`

Split into **two phases** with a barrier between, because the per-split scale needs a global
reduction over all splits before the weighted accumulate can start.

**Phase A — `reduce_lse_massive` (line 193), warp 0 only:**

- Each lane loads its strided subset of LSEs (`num_lse_per_thr` of them, stride 64).
- `warpReduce<Max>` → `max_lse`; `Σ exp(lse−max)` then `warpReduce<Add>` → `sum_lse`.
- `global_lse = log(sum_lse)+max_lse` (with NaN/zero → INF guard); lane0 writes final_lse.
- Each lane writes `lse_scale[split] = exp(lse − global_lse)` to LDS.
- Restricting to warp 0 means a single 64-lane warp reduction (no cross-wave LDS reduce)
handles up to 64 splits/pass; larger counts loop `num_lse_per_thr` times.

`s_barrier()` (line 521) — publish `lse_scale` to all 128 threads.

**Phase B — `reduce_output_massive` (line 305), all 128 threads:**
Weighted accumulate `reg_out = Σ_s lse_scale[s]·O_s`, already-normalized so **no final
divide**. The loop is **2-way software pipelined** (lines 340-365) to hide buffer-load
latency behind FMA compute:

```
prefetch map[2]; load O[1]; compute O[0];  prefetch map[3]; load O[2]; compute O[1]; ...
```

i.e. while computing tile k's FMA, the loads for tile k+1 are already in flight and the
gather index for tile k+2 is being read from LDS. Tail handled by two cleanup blocks
(lines 367, 399) for odd counts. This is the classic load/compute decoupling — the
*prefetch distance is 2* here, which the FlyDSL port exposes as `prefetch_depth`.

### 4.3 Split-count sub-tiers (massive)

Dispatched by `num_splits` (kernel lines 699-740): `≤64` (`kUpTo64Splits`, LSE in 1 VGPR),
`≤256` (`kUpTo256Splits`, `float[4]`), else `kUpToLdsLimit` (VGPR[4] + LDS spill). Smaller
tiers avoid the LDS overflow path entirely — a compile-time specialization to minimize
register/LDS pressure for the common case.

---

## 5. Host-side launch heuristics (`mla_reduce_v1` → `dispatch_mla_reduce_v1`)

1. `**get_num_work_group_per_bh*`* (line 981): picks `kNumThreadGroupPerBh` to fill the
  machine on small batches. If `num_cu·8·1.3 > num_reduce_tile·H`, it scales WGs-per-(b,h)
   up to spread q-positions, clamped to `max_seqlen_q`, rounded to next pow2, snapped to a
   supported value `{1,2,4,8,16,64,256}`. Empirical `1.3` over-subscription factor.
2. **Mode select** (line 944): if total work ≤ `num_cu·8·2` → 3-D grid-launch; else 1-D
  persistent (grid sized `num_cu·8·2`).
3. **LDS budget check** (line 935): `lds_size = max_splits·(4+4) + max(0,max_splits−256)·4`;
  error if it exceeds shared-mem-per-CU; warn if it would drop occupancy below `kOccupancy`.
4. **Static dispatch macros** (line 870 `MLA_REDUCE_ROUTER`): instantiate templates for the
  enumerated `(H, Dv)` combos and `NUM_WG_PER_BH`, and for `lse_t × out_t` (fp32 LSE ×
   {bf16,fp16} out). Supported `(H,Dv)` include **(128,512),(16,512),(128,128)** — the
   exact set the FlyDSL port must verify.

---

## 6. Optimizations applied (catalog)


| #   | Optimization                                           | Where                      | Why it helps                                                                              |
| --- | ------------------------------------------------------ | -------------------------- | ----------------------------------------------------------------------------------------- |
| 1   | **Lane-partitioned output** (`Dv/128` floats/thread)   | Traits:23                  | output axis needs zero cross-thread comm; perfectly coalesced 16B buffer ops              |
| 2   | **16B vectorized buffer load/store** (`dwordx4`)       | buf_*_vec:34/65            | max HBM transaction size; fewest instructions per byte                                    |
| 3   | **SGPR buffer descriptors, no waterfall**              | make_gmem:467              | base ptr is uniform kernel arg → scalar address, no per-lane addr divergence              |
| 4   | `**readfirstlane` on indptr**                          | kernel:676,688             | forces tile bounds into SGPRs (uniform control flow, scalar compares)                     |
| 5   | **Gather indices staged in LDS once**                  | impl:433                   | `reduce_partial_map` reused across LSE pass + output pass + all seq → 1 gmem read         |
| 6   | **Per-split LSE scale precomputed in LDS** (massive)   | reduce_lse_massive         | decouples the global softmax-norm from the wide output FMA; computed by 1 warp            |
| 7   | **2-way software pipelining of loads vs FMA**          | reduce_output_massive:340  | hides buffer-load latency; prefetch map[k+2] + data[k+1] while FMA-ing tile k             |
| 8   | **Pre-normalized weights → no final divide** (massive) | phase B                    | divide folded into `lse_scale`; saves a per-element reciprocal on the wide path           |
| 9   | **Online-softmax single-pass merge** (simple)          | impl_simple                | small `num_splits`: avoid barrier/LDS entirely, scalar bookkeeping in regs                |
| 10  | **Split-count sub-tiering** (64/256/LDS)               | dispatch:699               | keep LSEs in VGPR for common counts; spill to LDS only when forced                        |
| 11  | `**static_for<kVecWidth>` unrolled FMAs**              | throughout                 | no loop overhead on the inner Dv slice                                                    |
| 12  | `**__launch_bounds__(128,8)`**                         | kernel:670                 | tells compiler to cap VGPRs for 8-way occupancy → latency hiding via WG-level parallelism |
| 13  | **Persistent grid-stride kernel**                      | _ps:669                    | amortize launch cost, exact-fit the CU count, no oversubscription                         |
| 14  | **WG-per-(b,h) fan-out for small batches**             | get_num_work_group_per_bh  | fills all CUs when `num_reduce_tile·H` is small (decode's typical regime)                 |
| 15  | `**sched_barrier(0)` fences**                          | impl:439,520               | prevent the scheduler from hoisting loads across the LDS-publish barrier                  |
| 16  | **NaN/zero LSE → INF guard**                           | reduce_lse:275, simple:658 | numerically robust combine when a split contributed nothing                               |
| 17  | **Tiny LDS footprint by design**                       | dispatch:932               | a few KB keeps occupancy 8; LDS holds only indices+scales, never the O data               |


### Hierarchy summary

- **HBM ↔ VGPR**: partial O streamed in 16B vectors straight to the register accumulator;
it is **never staged in LDS** (it's read exactly once — the byte floor for a reduction).
- **HBM → LDS**: only the small metadata (`reduce_partial_map`) and derived `lse_scale`.
- **VGPR**: output accumulator + (in small tiers) the per-split LSEs.
- **Cross-thread**: only along the split/LSE axis, only in warp 0, only via warp shuffle +
one LDS broadcast. The output axis is embarrassingly parallel across lanes.

---

## 7. Implications for the FlyDSL port

- The byte floor (read each partial O once, write final once) is already hit → expect
**HBM-bound parity, not a speedup** (confirm with rocprofv3 before tuning).
- Port-critical correctness items: lane partition `Dv/128`, the simple-vs-massive
threshold at 4, the no-final-divide pre-normalized massive weights, the NaN→INF guard,
runtime `final_output` strides, and the LDS-staged gather map.
- Optimization knobs map directly: `prefetch_depth` ↔ optim #7 (currently 2),
`waves_per_eu` ↔ #12, `use_exp2` ↔ exp in #6/#9, `use_packed_cvt` ↔ store cast in #2,
`use_packed_f32_fma` ↔ the `static_for` FMAs in #11.

