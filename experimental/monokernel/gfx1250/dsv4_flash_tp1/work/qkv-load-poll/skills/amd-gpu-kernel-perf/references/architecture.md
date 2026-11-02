# CDNA3 / CDNA4 numbers that drive kernel design

Spec values are approximate public figures; "measured" values come from MI355X (gfx950) work on an 8-GPU node with
ROCm 7 / Triton 3.8 Gluon and should be re-probed on a new SKU or software stack before relying on them.

## Contents
1. Topology
2. Per-CU resources and residency
3. Achievable rates (measured)
4. Latencies and fixed costs (measured)
5. Consequences for kernel design

## 1. Topology

| | MI300X (gfx942) | MI355X (gfx950) |
|---|---|---|
| CUs | 304 (8 XCDs x 38) | 256 (8 XCDs x 32) |
| L2 | 4 MB per XCD, private to the XCD | 4 MB per XCD, private to the XCD |
| Infinity Cache (MALL) | 256 MB shared | 256 MB shared |
| HBM | 192 GB, ~5.3 TB/s spec | 288 GB, ~8 TB/s spec |
| LDS per CU | 64 KB | 160 KB |
| Wavefront | 64 lanes | 64 lanes |
| Peer GPUs | XGMI full mesh in an 8-GPU node | XGMI full mesh in an 8-GPU node |

Facts that matter more than the table:
- The L2 is per XCD and is not kept coherent with other XCDs while a kernel runs. A CTA on XCD 3 can read a stale
  L2 line that a CTA on XCD 5 already overwrote. Data shared across CTAs within one launch must be written and read
  at system or device scope (cache-modifier bits, see `memory-and-sync.md`) or handed over at a kernel boundary.
  A hybrid design used this deliberately: data written once by one workgroup and re-read by many (normalized
  activations, activation outputs, a reduced slice) was placed across a kernel boundary instead of polled.
- CTAs are dispatched round-robin across XCDs; consecutive block ids land on different XCDs. "Nearby" block ids do
  not share an L2.
- Peer memory over XGMI is never cached locally on the reader's L2 when it is allocated uncached; every access is a
  round trip.

## 2. Per-CU resources and residency

- A persistent ("resident") kernel needs every CTA resident at once, typically one CTA per CU. A CTA of 512 threads
  (8 waves) at one CTA per CU leaves a budget of 256 VGPRs per lane (measured config); any spill to the private
  segment or LDS growth that reduces residency below one CTA per CU makes a spin-waiting kernel deadlock, because
  waiters occupy the slots producers need. Always check VGPR / spill / LDS numbers of every specialization before
  launching (compile-only check in `codegen.md`).
- Typical budgets seen: 229-254 VGPRs with 0 VGPR spills; SGPR spills are cheap (they go to VGPR lanes); 136-139 KB
  of LDS on gfx950 (limit 160 KB).
- A library kernel picks its own occupancy per op (e.g. 2 CTAs/CU, 97-111 VGPRs, 64 KB LDS for a 512-CTA design);
  a resident megakernel cannot, which is one reason big GEMMs leave the megakernel as rows grow.

## 3. Achievable rates (measured on MI355X)

| Quantity | Measured | Note |
|---|---|---|
| HBM read, pure streaming probe, >= 256 KB in flight per CTA | 7.0-7.5 TB/s | upper bound; real kernels rarely keep that much in flight |
| HBM read, library GEMM / MoE at decode sizes | ~5.5-6 TB/s | use this for floors; 22 MB in ~6.5 us, 51 MB in ~11.6 us including launch |
| Latency-bound persistent kernel without deep prefetch | ~4.3 TB/s chip (17-20 GB/s per CU) | the gap prefetch closes |
| Dense bf16 MFMA under load | ~1.5-1.7 PF | vs ~2.5 PF peak |
| Per-CU streaming, persistent kernel | ~28 GB/s per CU with ~4.4 us fixed latency | 128 KB per CTA in ~6.6 us median |
| Per-CU reads in flight | ~32 KB (about L1 size) | caps a GEMM's A re-reads competing with the weight stream |
| Dot-operand-layout global loads | ~37 GB/s per CU | stage through LDS with async `buffer_load_to_shared`: ~110 GB/s per CU |
| Remote (XGMI) plain stores to uncached peer memory | 110-300 GB/s per rank | |
| Remote write-through (`.wt`) stores to uncached peer memory | ~30 GB/s per rank | use only for flags |
| Remote reads | 128 B contiguous segments needed | 64 B segments ~2x slower |

## 4. Latencies and fixed costs (measured)

| Event | Cost |
|---|---|
| Empty kernel launch (graph replay) | ~2.4 us |
| Cross-CTA hop through a mailbox (producer store -> consumer sees it), idle GPU | ~0.7-1 us |
| One XGMI round trip | ~1-5 us depending on load |
| 3.4 MB exact bf16 all-reduce over 8 GPUs, pull design | ~33 us |
| Same, INT4-quantized library all-reduce | ~35-39 us (lossy) |
| fp32 partial stores for split-K | 4-5 us + 4-7 us reduce kernel |
| fp32 global atomics for a split-K reduction | 187 us (dead end) |

## 5. Consequences for kernel design

- Decode-time weight streaming is the floor for most layers: plan around bytes, not FLOPs. A layer's weights at
  ~5.5 TB/s set the minimum; everything else is latency to hide behind that stream.
- Issue weight loads as early as possible and before polling for activations: per-CU streaming has ~4 us of fixed
  latency, and an activation poll issued after a big load waits for it (in-order completion).
- Keep remote traffic in large, contiguous, plain stores or (better) let the consumer pull; reserve coherent /
  write-through operations for small flags.
- The MFMA rate is rarely the limit at decode sizes: in a fused MoE expert-up at ~24 experts, the MMA was 0.6 us of
  a 15 us stage; setup (decoding experts, building route tables, issuing loads) was the rest.
