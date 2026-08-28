# qr_int4 on MI350P: why it collapses, and how to fix it

Analysis of the 76x regression of the FlyDSL INT4 two-shot all-reduce
(`aiter/ops/flydsl/kernels/qr_int4*.py`) between MI350X and MI350P, and a plan
for an MI350P-specific kernel.

- Date: 2026-08-28
- Baselines: `op_tests/dump_data/mi350x-comm-bench-baseline.md`,
  `op_tests/dump_data/mi350p-comm-bench-baseline.md`
- Producer: `op_tests/multigpu_tests/bench_comm_allreduce.py`
- Background on the candidate: `docs/communication_kernels.md` §8.8

---

## 1. The observation

TP4, bf16, hidden 7168, `us` is the slowest rank (what the model waits on).

| tokens | KiB | kernel | MI350X qr_int4 | MI350P qr_int4 | ratio | MI350P baseline | MI350P rccl |
|-------:|----:|:-------|---------------:|---------------:|------:|----------------:|------------:|
| 1      | 14      | 1stage | 20.60 us       | 30.94 us       | 1.5x  | 22.15 us        | 39.00 us    |
| 8      | 112     | 1stage | 20.07 us       | 56.95 us       | 2.8x  | 54.30 us        | 50.58 us    |
| 1024   | 14336   | 2stage | 73.71 us       | **5603 us**    | **76x** | 833.90 us     | 584.60 us   |

SQNR is 19.1-19.2 dB on both machines, so this is purely a performance
regression -- the codec is doing the right thing.

The key framing: `aiter_cross_device_reduce` degrades **5.3x** across the two
machines (157.4 -> 833.9 us) and RCCL degrades **3.1x** (186.7 -> 584.6 us).
That much is the fabric. qr_int4 degrades **76x**. The extra ~15x is a kernel
pathology specific to this box, and it is fully explained below.

---

## 2. The machine

`rocm-smi --showtopo` on this host:

```
Link Type between two GPUs
       GPU0    GPU1    GPU2    GPU3    GPU4
GPU0   0       PCIE    PCIE    PCIE    PCIE
GPU1   PCIE    0       PCIE    PCIE    PCIE
...
Hops:  2 within a NUMA node, 3 across   (GPU0-2 on NUMA0, GPU3-4 on NUMA1)
```

| | MI350X (reference report) | MI350P (this box) |
|---|---|---|
| CUs per GPU | 256 | 128 |
| HBM | 288 GiB | 144 GiB |
| GPU-GPU link | xGMI | **PCIe Gen5 x16, no xGMI** |
| Partitioning | -- | NPS1 / SPX |

Measured `hipMemcpyPeer` bandwidth, all pairs: **53.8-55.5 GB/s**, uniform.
That is PCIe Gen5 x16 at ~86% efficiency. NUMA distance does not matter for
bulk copies. There is no xGMI on this system at all.

This single fact drives everything: qr_int4 was designed around a fabric whose
native transfer unit is a 64 B xGMI sector, and it is running on PCIe.

---

## 3. Root cause: uncached IPC memory + writes to multiple peers

`UncachedIpcHeap.alloc_uncached` (`aiter/ops/flydsl/kernels/qr_int4_ipc.py:121`)
allocates the **entire** IPC inbox -- handshake flags *and* payload -- with:

```python
hipExtMallocWithFlags(&buf, size, hipDeviceMallocUncached)   # flag 0x3
```

Peer writes into uncached memory over PCIe collapse as the number of distinct
peer destinations grows. Measured with a standalone HIP microbenchmark
reproducing `_fanout_nt`'s exact access pattern (16 B `global_store_dwordx4 nt`
per lane, 4 lanes per 64 B sector, consecutive sectors round-robined across
peers), 32 MiB total, one GPU writing:

| allocation | 1 peer | 2 peers | 3 peers |
|---|---:|---:|---:|
| `hipMalloc` (coarse-grained) | 52.4 GB/s | 52.96 GB/s | **32.65 GB/s** |
| `hipDeviceMallocFinegrained` (0x1) | 52.3 GB/s | 53.64 GB/s | **33.45 GB/s** |
| **`hipDeviceMallocUncached` (0x3)** | 55.4 GB/s | **4.45 GB/s** | **1.44 GB/s** |

A single destination is fine. Two destinations cost 12x. Three cost **38x**.

### This accounts for the entire regression

Per-rank remote egress for the 1024x7168 case (`num_tiles` = 448, TP4,
`rank_atoms` = 2, 1152 B rank-tile = 18 x 64 B sectors, two phases, 3 remote
peers):

```
per tile:  2 phases x 3 peers x 18 sectors x 2 rank_atoms x 64 B  = 13824 B
           + publish 2 x 3 x 64 B                                 =   384 B
total:     14208 B x 448 tiles                                    =  6.36 MB
```

6.36 MB / 5603 us = **1.13 GB/s**, sitting exactly on the measured 1.44 GB/s
uncached-3-peer wall. There is no residual left to explain.

### Why it was never caught

On xGMI the peer aperture has no such penalty, and 64 B *is* the native fabric
packet -- so on MI350X this design is not merely acceptable, it is optimal.
The uncached flag is what makes the handshake work without cache-maintenance
gymnastics, and on xGMI it is free.

Note the precedent in the in-tree HIP path: `custom_all_reduce.py` allocates
uncached memory **only for the meta/signal buffer** (line 1053,
`self._pool.create("meta", meta_size, uncached=True)`) and uses ordinary cached
memory for data. qr_int4 conflates the two into one allocation.

---

## 4. Secondary cause: 64 B destination interleave

Even on cached memory, round-robining destinations every 64 B costs a further
1.5-1.65x. Same microbenchmark, 3 peers, varying the per-peer contiguous chunk:

| per-peer chunk | cached | uncached |
|---:|---:|---:|
| 64 B   | 36.25 GB/s | 1.58 GB/s |
| 256 B  | 54.03 GB/s | 2.27 GB/s |
| 1024 B | 54.37 GB/s | 2.29 GB/s |
| 4096 B | 55.00 GB/s | 2.42 GB/s |
| 64 KiB | 54.88 GB/s | 2.41 GB/s |

Two readings:

1. On cached memory, going from 64 B to >=256 B per-peer runs recovers full
   line rate. This is worth ~1.65x on top of the allocation fix.
2. On uncached memory, coarsening does **not** rescue it (1.58 -> 2.42 GB/s).
   The penalty is per-destination-switch in the write path, not a granularity
   effect. Confirmed separately: assigning each block a single fixed peer
   ("per-block peer specialization") lifts uncached 3-peer from 1.44 to only
   2.31 GB/s. **The allocation flag must change; no access-pattern tuning can
   work around it.**

---

## 5. Ruled out (measured, not assumed)

**Grid oversubscription / super-tile choice.** `DEFAULT_GRID_CAP = 304 * 4 =
1216` is sized for MI300X's 304 CUs; this box has 128. The natural hypothesis
is that blocks are not co-resident across GPUs and the `_wait_release` spin
serializes. Sweeping the actual knobs at 1024x7168 says otherwise:

| `grid_cap` | ST=1 | ST=8 |
|---:|---:|---:|
| 128  | 6143.1 us | 5997.0 us |
| 256  | 6507.9 us | 6271.9 us |
| 448  | 7008.3 us | 5967.2 us |
| 1216 | 7001.7 us | 7044.5 us |

Everything lands in 5967-7045 us. `grid_cap=128` fits trivially on 128 CUs and
changes nothing. Not the problem. (Worth re-checking *after* the allocation fix,
when the kernel is no longer wire-starved -- at that point occupancy may start
to matter.)

**The `nt` cache modifier.** Byte-identical timings with and without `nt` in
every configuration tested. Not a factor either way.

**Receive-path read bandwidth.** Local `nt` read of the inbox: 2824 GB/s
uncached vs 3806 GB/s for `hipMalloc` -- a real 26% penalty, but two orders of
magnitude away from mattering here. Secondary; it comes along for free with the
allocation fix.

---

## 6. The release problem, and how it resolves

The allocation change alone is not sufficient, but not for the reason first
suspected. Sweeping the receiver's cache modifiers in a 2-GPU harness
(bounded spin, 300 iterations, per-block waiters):

| alloc | load policy | invalidate | spin timeouts | bad words |
|---|---|---|---:|---:|
| `hipMalloc` | `nt` | `sc1` | 0 | 3 007 248 |
| `hipMalloc` | `sc0 sc1 nt` | `sc0 sc1` | 0 | **0** |
| finegrained | `nt` (current kernel) | `sc1` | 0 | **0** |
| uncached | `nt` (current kernel) | `sc1` | 0 | **0** |

`spin_timeouts = 0` everywhere: the flag handshake always converges. So the
receiver was never the problem, and **fine-grained needs no cache-modifier
change at all**. Coarse-grained would need `sc0 sc1` loads, and buys nothing
over fine-grained on the wire (32.65 vs 33.45 GB/s), so it is not offered.

The real problem is on the **writer**, and it is a release problem:

- `nt` is a non-temporal *hint*. It does not write through.
- On a cacheable inbox a peer store can therefore sit in the writer's L2 after
  `vmcnt(0)` retires, while the peer spins on a flag it cannot see. The stall
  clears only when unrelated traffic evicts the line, so its cost scales
  **inversely with how busy the kernel is**: 227 us at 448 blocks, 50 ms at 4,
  5.3 s at 1.
- The earlier 2-GPU harness missed this entirely because the producer *kernel
  ended* between iterations and the implicit end-of-kernel writeback hid it.
  qr_int4 publishes mid-kernel.

The obvious fix -- make every payload store `sc0 sc1` -- works, and is wrong:

| 1024 tokens (14 MiB) | payload `nt` | payload `sc0 sc1 nt` |
|---|---:|---:|
| fine-grained | 226.76 us | 6707.60 us |

Write-through defeats the very thing that makes fine-grained fast. Letting the
payload land in L2 is not incidental; **the L2 is acting as a write-combining
buffer**, batching this kernel's 64 B destination-interleaved stores into large
PCIe bursts. That is where the 30x comes from.

So the correct structure is a release fence, not a store policy:

1. payload stores stay `nt` and cacheable, so L2 keeps coalescing them;
2. `s_waitcnt vmcnt(0)` + workgroup barrier, as today;
3. `buffer_wbl2 sc1` writes the coalesced lines back, `vmcnt(0)` waits for it;
4. the flag store alone goes out `sc0 sc1 nt`, so the peer's spin sees it at once.

Every workgroup issues its own writeback: L2 is per-XCD, so one workgroup's
writeback says nothing about a workgroup on another die.

## 7. Results

Implemented and measured. TP4, bf16, hidden 7168, max over ranks, 101 timed
iterations. `finegrained` is the release-fenced kernel of §6; `uncached` is the
shipped behaviour, unchanged.

| tokens | KiB | uncached (was) | fine-grained (now) | speedup | SQNR dB |
|-------:|----:|---------------:|-------------------:|--------:|--------:|
| 1     | 14      | 20.20 us   | **13.44 us**  | 1.5x  | 19.13 |
| 8     | 112     | 49.56 us   | **15.03 us**  | 3.3x  | 19.17 |
| 16    | 224     | 80.54 us   | **17.78 us**  | 4.5x  | 19.16 |
| 32    | 448     | 172.06 us  | **23.98 us**  | 7.2x  | 19.16 |
| 64    | 896     | 364.41 us  | **37.88 us**  | 9.6x  | 19.17 |
| 128   | 1792    | 751.37 us  | **68.77 us**  | 10.9x | 19.18 |
| 256   | 3584    | 1579.60 us | **136.01 us** | 11.6x | 19.19 |
| 512   | 7168    | 3199.49 us | **274.17 us** | 11.7x | 19.19 |
| 1024  | 14336   | 6039.33 us | **575.99 us** | 10.5x | 19.19 |
| 4096  | 57344   | 25297 us   | **1859.65 us**| 13.6x | 19.19 |

SQNR is unchanged at every size, so the win is not bought with accuracy. Scaling
is now clean and monotonic; the old curve was not.

Against the other candidates at 1024x7168 (from the baseline report): 575.99 us
versus 833.90 for `cross_device_reduce` (1.45x) and 584.60 for RCCL (1.00x).

**Known headroom left on the table.** The unfenced run hit 226.76 us at this
shape -- correct output, but only because lazy eviction happened to be prompt at
448 blocks. The fence costs 2.5x there because `buffer_wbl2` writes back the
whole L2 and every workgroup issues one, twice per tile: 896 full-L2 writebacks
for one collective. Amortizing them is P2, which is now a real lever rather than
the no-op it measured as in §5.

## 8. Plan

Ordered by measured value. P0 is the whole regression; everything else is
sharpening.

### P0 -- Fine-grained inbox + explicit release fence  **[DONE]**

**Worth 1.5x-13.6x depending on size** (§7). Implemented across:

- `qr_int4_ipc.py` -- `alloc(size, flags)` replaces the hardcoded uncached
  allocation; `alloc_uncached` kept as a wrapper.
- `qr_int4.py` -- `QRInt4(inbox_memory=...)`, resolved by `_has_xgmi_peer_links()`
  reading the **KFD topology** (`p2p_links/*/properties`, type 11 = xGMI, 2 =
  PCIe). Arch is not a usable signal: MI350X and MI350P both report `gfx950` and
  want opposite answers. Unreadable topology assumes xGMI, so the MI350X path is
  bit-for-bit unchanged.
- `qr_int4_kernel.py` -- `_INBOX_POLICY` per memory type, the `buffer_wbl2`
  release fence of §6, and the inbox mode in the JIT symbol name so two variants
  differing only in cache bits cannot collide in the cache.

The inbox is the only allocation that changed type; the peer-pointer/colour
table stays uncached, as no peer writes it.

Deliberately **not** done: splitting flags and payload into two allocations, and
supporting a coarse-grained inbox. The split buys nothing once the payload is
cacheable, and coarse-grained needs `sc0 sc1` receive-side loads (§6) for
32.65 GB/s against fine-grained's 33.45.

Also added: `MIN_PAYLOAD_BYTES` / `QRInt4(min_bytes=...)` and `is_beneficial()`,
so small messages are refused rather than run. Post-fix this is an **accuracy**
policy, not a performance one -- the kernel is now faster than
`cross_device_reduce` at every size measured, so the only thing it costs at
decode sizes is ~36 dB of SQNR.

### P1 -- Coarsen the peer-store granularity

**Worth ~1.5-1.65x on top of P0.**

`_fanout_nt` (`qr_int4_kernel.py:483`) currently maps consecutive quads to
consecutive *peers* (`fanout_int4_stripe = make_layout((world_size, 8), (1, world_size))`)
so that one NT store instruction hits every GPU -- the right call on xGMI, where
64 B is the native sector.

For MI350P, transpose it: make consecutive quads cover consecutive *sectors* of
one peer, so each peer receives a >=256 B contiguous run per store group. A
whole 1152 B rank-tile per peer per group is the natural unit and is well past
the 256 B knee. This is a layout swap, not a rewrite -- `fanout_int4_stripe` and
`fanout_scale_stripe` change stride order, and the `quad_id < n_quads` guard
adjusts.

### P2 -- Amortize the release fence / re-tune `grid_cap`

**Now the top remaining item, worth up to ~2.5x at prefill sizes.** §5 measured
this as a no-op; that verdict is void, because the kernel was wire-starved then
and is not now.

`buffer_wbl2` writes back the whole L2, and every workgroup issues one per phase
per tile -- 896 full-L2 writebacks for a single 14 MiB collective. The unfenced
run reached 226.76 us at that shape against 575.99 fenced, and that gap is the
prize. Levers, in order of expected value:

1. Fewer publishes per collective: shrink `grid_cap` so each block owns several
   tiles, and let `_pick_st` choose ST=8 so one publish covers 8 tiles.
   `DEFAULT_GRID_CAP = 304 * 4` is an MI300X constant (304 CUs) on a 128-CU
   part; at 1024x7168 it yields 448 blocks x 1 tile, the worst possible ratio.
2. Issue the writeback from one workgroup per XCD rather than all of them, if
   the cross-workgroup ordering can be made sound.
3. Re-run the §5 grid_cap x ST sweep once the above land.

### P3 -- Decode-shape latency (1-8 tokens)

**Separate problem, do not let it block P0-P2.** At 14-112 KiB the payload is
too small for compression to pay and the cost is launch overhead plus two
handshake round trips over PCIe. qr_int4 is 0.72x / 0.95x the baseline here and
P0 will not change that much. Options if this matters: fuse the two phases into
a one-shot for sub-tile messages (the baseline already dispatches 1stage below
160 KiB), or skip quantization entirely below a byte threshold and ship bf16.
Measure before building.

### P4 -- Dispatch and reporting

- The MI350P kernel needs a selection gate. Arch alone is insufficient --
  both machines report `gfx950`. Key on the actual interconnect (absence of
  xGMI peer links, or CU count as a proxy) rather than on `get_gfx()`.
- `bench_comm_allreduce.py`'s provenance header records arch and CU count but
  not link type. Add it; the two baseline reports in `op_tests/dump_data/` are
  otherwise indistinguishable on the axis that actually explains their
  difference.

---

## 9. Reproducing the measurements

The microbenchmarks behind §3, §4 and §6 are standalone HIP files; the sweep in
§5 drives `QRInt4` directly. None are checked in. Rebuilding them:

- **P2P bandwidth matrix (§2):** `torch` `copy_()` between devices, 64 MiB, 10
  iterations, all ordered pairs.
- **Fanout collapse (§3, §4):** one kernel doing `__builtin_nontemporal_store`
  of a 16 B vector, with a `chunk16` parameter controlling the per-peer
  contiguous run and destinations round-robined across `np` peer buffers
  allocated with `hipMalloc` / `hipExtMallocWithFlags(0x1)` /
  `hipExtMallocWithFlags(0x3)`. Compile `--offload-arch=gfx950 -O3`.
  Do not write the stores as inline asm with a `"v"` constraint on a struct
  pointer -- clang rejects it; use `__builtin_nontemporal_store` on an
  `ext_vector_type(4)`.
- **Handshake visibility (§6):** two GPUs, peer access enabled, producer
  nt-stores payload then flag, consumer spins per-block on the flag. Every
  block's thread 0 must spin -- gating the spin on `blockIdx.x == 0` makes all
  three allocation modes "fail" and is a harness bug, not a hardware result.
- **End-to-end allocation swap (§6):** temporarily make the flag passed to
  `hipExtMallocWithFlags` in `UncachedIpcHeap.alloc_uncached`
  (`qr_int4_ipc.py:121`) read from an env var, then run the 4-rank benchmark
  once per mode. This edit was reverted after measuring; the tree is clean.
- **`grid_cap` / ST sweep (§5):** 4 spawned ranks, `QRInt4(super_tile=..., grid_cap=...)`,
  `run_perftest(..., use_cuda_event=True)`, max over ranks. Note
  `fly.allreduce` returns `None`; the timed thunk must return the output buffer
  or `run_perftest`'s accuracy path raises.
