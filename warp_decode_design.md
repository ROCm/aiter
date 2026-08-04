# Warp Decode: Output-Centric MoE Inference for Small-Batch Decode

## 1. Overview

Warp decode is a GPU kernel design for Mixture-of-Experts (MoE) layers that reorganizes the parallelism axis from **expert-centric** to **output-centric**. Instead of grouping tokens by expert and running grouped GEMMs, each hardware wavefront (warp) independently computes exactly one scalar output element.

This eliminates all token-sorting, padding, intermediate buffering, and cross-wavefront synchronization — at the cost of low compute utilization per wavefront. The design is optimal for the **autoregressive decode step** at small batch sizes (B=1..32), where the MoE layer is memory-bandwidth-bound and the traditional expert-centric overhead dominates wall-clock time.

**Key results (reported on NVIDIA B200):**
- 1.84× throughput improvement over the traditional grouped-GEMM path
- 1.4× closer to FP32 ground truth (by eliminating intermediate requantization)
- 3.95 TB/s sustained memory bandwidth (58% of measured peak)

---

## 2. Context: MoE Decode Is Memory-Bandwidth-Bound

### 2.1 The MoE Layer

A standard MoE layer with gated activation (SwiGLU) computes, for each token:

```
1. Router selects TOP_K experts out of E total (e.g., 8 out of 128)
2. For each selected expert e with routing weight w_e:
     gate_e = X @ W_gate[e].T        # [HIDDEN] × [HIDDEN, INTER] → [INTER]
     up_e   = X @ W_up[e].T          # [HIDDEN] × [HIDDEN, INTER] → [INTER]
     mid_e  = SiLU(gate_e) * up_e    # [INTER]
     out_e  = mid_e @ W_down[e].T    # [INTER] × [INTER, HIDDEN] → [HIDDEN]
3. Y = Σ_e  w_e * out_e              # weighted sum across experts
```

### 2.2 Arithmetic Intensity at Decode

During autoregressive decode, batch size is small (often B=1). Each token routes to TOP_K experts. For a single token and one expert, the gate+up projection is:

```
FLOPs:  2 × HIDDEN × INTER × 2  =  4 × HIDDEN × INTER   (two dot products)
Bytes:  (INTER × HIDDEN + INTER × HIDDEN) × bytes_per_weight  (two weight rows streamed)
      + HIDDEN × bytes_per_activation  (input vector, likely cached/reused)
```

For typical dimensions (HIDDEN=7168, INTER=2048, FP8 weights):

```
FLOPs:  4 × 7168 × 2048 ≈ 58.7 MFLOP
Bytes:  2 × 2048 × 7168 × 1 ≈ 29.4 MB  (across all INTER neurons for one expert)
```

Arithmetic intensity ≈ 2 FLOP/byte. Modern GPUs deliver 100+ TFLOP/s compute but only 4-8 TB/s memory bandwidth. The crossover (where compute = bandwidth) is at ~15-25 FLOP/byte. At 2 FLOP/byte, the kernel is **purely memory-bandwidth-bound** — the compute units are idle most of the time regardless of how you organize the work.

### 2.3 Implication

Since compute is free and memory bandwidth is the bottleneck, the optimal kernel design is one that:
1. **Maximizes memory bandwidth utilization** (keeps the memory pipeline saturated)
2. **Minimizes non-weight memory traffic** (sorting buffers, intermediate activations, padding)
3. **Minimizes kernel launch and synchronization overhead**

Warp decode achieves all three by trading compute efficiency for memory efficiency.

---

## 3. The Traditional Expert-Centric Pipeline

The standard MoE inference path used in frameworks like vLLM, SGLang, and AITER follows this structure:

```
Step 1: topk_softmax         — route each token to TOP_K experts, get weights
Step 2: moe_sorting           — sort tokens into expert-major order, pad to block boundaries
Step 3: [optional] quant      — quantize activations for reduced-precision GEMM
Step 4: grouped_gemm (gate)   — batched matrix multiply, one tile per expert
Step 5: grouped_gemm (up)     — second projection (or fused with gate)
Step 6: activation            — SiLU/GeLU applied to gate, multiply with up
Step 7: [optional] re-quant   — re-quantize intermediate activations
Step 8: grouped_gemm (down)   — down projection per expert
Step 9: unpermute + combine   — scatter results back to token order, weighted sum
```

### 3.1 Overhead at Small Batch

At B=1 with TOP_K=8 and E=128:
- **Sorting**: 8 tokens (one token routed to 8 experts) must be sorted into expert-major layout
- **Padding**: Each expert's token list is padded to a block boundary (e.g., 32). With 8 active experts, that is up to 8 × 31 = 248 padded slots — 31× the actual work
- **Intermediate buffer**: The gate+up output is `[B × TOP_K, INTER_DIM]` = `[8, 2048]` in BF16 = 32 KB per token, written to global memory, then read back for the down projection
- **Requantization**: If using FP8 GEMMs, intermediates are quantized (BF16 → FP8) and dequantized, adding both latency and rounding error
- **Combine step**: A separate kernel reads 8 partial outputs and computes the weighted sum

Five of these nine steps perform zero useful compute — they exist solely to arrange data for the expert-centric grouped GEMM.

---

## 4. Warp Decode: Output-Centric Design

### 4.1 Core Principle

Assign each hardware wavefront (warp) to compute **one scalar output value**. The wavefront independently:
1. Looks up which expert(s) it needs
2. Streams the relevant weight row(s) from memory
3. Computes dot product(s) using lane-parallel reduction
4. Writes one scalar to the output tensor

No sorting. No padding. No intermediate buffers. No cross-wavefront communication.

### 4.2 Architecture

The MoE layer is implemented as **two sequential kernel launches**:

```
Kernel 1: gate_up_fused
  - Computes the gated activation: SiLU(X @ W_gate[e]) × (X @ W_up[e])
  - One wavefront per (token, expert, intermediate_neuron) triple
  - Output: intermediate tensor [B, TOP_K, INTER_DIM]

Kernel 2: down_reduce
  - Computes the down projection and weighted expert reduction
  - One wavefront per (token, output_neuron) pair
  - Output: final tensor [B, HIDDEN_DIM]
```

The intermediate tensor between the two kernels is the **only** data that touches global memory beyond the final output write.

### 4.3 Why Two Kernels, Not One

A single fused kernel would need to synchronize all wavefronts computing the intermediate dimension for a given (token, expert) before starting the down projection — requiring either:
- A global barrier (not supported within a single kernel on most GPUs)
- Atomic accumulation (serialization, poor performance)
- Device-wide cooperative groups (occupancy constraints)

Two sequential kernels use the implicit barrier of the kernel launch to synchronize. The intermediate tensor is small (B × TOP_K × INTER_DIM × element_size) and likely stays in L2 cache between launches.

### 4.4 Why Not Fuse Across Experts in the Down Kernel

The down kernel loops over TOP_K experts **within a single wavefront**, folding each expert's routing weight into a running accumulator. This eliminates the separate combine/reduce step entirely — the weighted sum happens in registers, and only the final scalar is written to memory.

This is possible precisely because we assigned the wavefront to an **output element** rather than to an **expert**. If the wavefront were assigned to an expert, it could only compute one expert's contribution and would need a separate reduction step.

---

## 5. Detailed Pseudo-Code

### 5.1 Notation

```
B           Number of tokens in the batch
E           Total number of experts
TOP_K       Number of experts each token routes to (e.g., 8)
HIDDEN      Hidden dimension / model dimension (e.g., 7168)
INTER       Intermediate dimension per expert (e.g., 2048)
WAVE_SIZE   Wavefront width (e.g., 32 for NVIDIA, 64 for AMD)

X           [B, HIDDEN]            Input activations (BF16, read-only)
router_ids  [B, TOP_K]             Selected expert indices per token
router_wts  [B, TOP_K]             Routing weights per token (FP32)
W_gate[e]   [INTER, HIDDEN]        Gate projection weights for expert e
W_up[e]     [INTER, HIDDEN]        Up projection weights for expert e
W_down[e]   [HIDDEN, INTER]        Down projection weights for expert e
Y           [B, HIDDEN]            Output activations
```

Weight storage format is implementation-defined (FP8, FP4, BF16, etc.). The pseudo-code shows logical indexing; actual implementations will handle dequantization inline.

### 5.2 Kernel 1: Gate + Up + Activation (Fused)

```
KERNEL gate_up_fused:
  Grid dimensions:  (B, TOP_K, INTER)
  Each wavefront is identified by (token_b, expert_k, neuron_j)

  // ---- Setup ----
  e = router_ids[token_b, expert_k]    // which expert

  // ---- Dot products over HIDDEN dimension ----
  // Each lane handles a strided subset of the inner dimension
  gate_acc = 0.0    // FP32, private register
  up_acc   = 0.0    // FP32, private register

  lane = thread_id_within_wavefront     // 0..WAVE_SIZE-1

  for i in range(lane, HIDDEN, WAVE_SIZE):
      x_val = to_fp32(X[token_b, i])                    // load activation element
      g_val = to_fp32(W_gate[e][neuron_j, i])            // load gate weight element
      u_val = to_fp32(W_up[e][neuron_j, i])              // load up weight element

      gate_acc += x_val * g_val    // fused multiply-add
      up_acc   += x_val * u_val    // reuse x_val (same cache line)

  // ---- Wavefront-level reduction ----
  // Butterfly shuffle: log2(WAVE_SIZE) steps, no shared memory
  gate_val = wavefront_reduce_sum(gate_acc)
  up_val   = wavefront_reduce_sum(up_acc)

  // ---- Gated activation (SwiGLU) ----
  if lane == 0:
      intermediate[token_b, expert_k, neuron_j] = silu(gate_val) * up_val
```

**Memory access pattern per wavefront:**
- Reads: 1 row of W_gate (HIDDEN elements), 1 row of W_up (HIDDEN elements), activation vector X[token_b] (HIDDEN elements, shared across wavefronts for same token)
- Writes: 1 scalar

**Total wavefronts launched:** B × TOP_K × INTER

### 5.3 Kernel 2: Down Projection + Weighted Expert Reduction

```
KERNEL down_reduce:
  Grid dimensions:  (B, HIDDEN)
  Each wavefront is identified by (token_b, out_j)

  // ---- Accumulate across all TOP_K experts ----
  acc = 0.0    // FP32, private register — single running total

  lane = thread_id_within_wavefront

  for k in range(TOP_K):
      e = router_ids[token_b, k]       // which expert
      w = router_wts[token_b, k]       // routing weight (FP32)

      for i in range(lane, INTER, WAVE_SIZE):
          act_val = to_fp32(intermediate[token_b, k, i])       // intermediate activation
          d_val   = to_fp32(W_down[e][out_j, i])               // down weight element

          acc += w * act_val * d_val    // fold routing weight into accumulator

  // ---- Wavefront-level reduction ----
  result = wavefront_reduce_sum(acc)

  if lane == 0:
      Y[token_b, out_j] = to_output_dtype(result)
```

**Memory access pattern per wavefront:**
- Reads: TOP_K rows of W_down (each INTER elements), TOP_K slices of intermediate (each INTER elements)
- Writes: 1 scalar

**Total wavefronts launched:** B × HIDDEN

### 5.4 Wavefront Reduction Primitive

The butterfly reduction compiles to a single hardware instruction on both NVIDIA (`shfl.sync.bfly`) and AMD (`ds_permute` / `dpp`):

```
FUNCTION wavefront_reduce_sum(val):
    // For WAVE_SIZE = 32: 5 iterations
    // For WAVE_SIZE = 64: 6 iterations
    for offset in [WAVE_SIZE/2, WAVE_SIZE/4, ..., 1]:
        val += shuffle_xor(val, offset)
    return val
```

This uses **no shared memory** — values stay in registers and are exchanged directly between lanes via the register file crossbar.

---

## 6. Why "One Wavefront Per Scalar" Makes Sense

### 6.1 The Intuition

At first glance, using 32 (or 64) threads to produce a single floating-point value seems wasteful. But the relevant metric is not *compute utilization* — it is *memory bandwidth utilization*.

Consider what happens inside one wavefront computing a dot product of length HIDDEN=7168:

```
Per lane:
  - Loop iterations:       HIDDEN / WAVE_SIZE = 7168 / 32 = 224  (or 112 for wave64)
  - Multiply-adds:         224 × 2 = 448 FMA operations (gate + up)
  - Weight bytes loaded:   224 × 2 × sizeof(weight) per projection

Per wavefront:
  - Total weight bytes:    2 × HIDDEN × sizeof(weight) = 2 × 7168 × 1 = 14 KB (FP8)
  - Total FLOPs:           2 × 2 × HIDDEN = 28,672 FLOPs
  - Arithmetic intensity:  28672 / 14336 ≈ 2 FLOP/byte
```

The GPU's memory subsystem needs many concurrent outstanding loads to saturate bandwidth. Each lane issues independent load instructions, and with WAVE_SIZE lanes per wavefront and thousands of wavefronts in flight (across all compute units), the memory pipeline stays full.

### 6.2 Occupancy and Latency Hiding

A modern GPU has hundreds of compute units, each capable of scheduling multiple wavefronts:

```
Example GPU: 304 CUs × 4 SIMD units × ~8-16 wavefronts per SIMD
           = ~10,000 - 20,000 wavefronts concurrently in-flight

Warp decode launches:
  Kernel 1: B × TOP_K × INTER = 1 × 8 × 2048 = 16,384 wavefronts  (B=1)
  Kernel 2: B × HIDDEN         = 1 × 7168     = 7,168 wavefronts   (B=1)
```

Even at B=1, the kernel launches enough wavefronts to keep most compute units busy. At B=8 or B=32, the wavefront count exceeds 100K, far more than the hardware can execute simultaneously — which is exactly what enables latency hiding. When one wavefront stalls on a memory load, the scheduler switches to another ready wavefront at zero cost.

### 6.3 Resource Usage Per Wavefront

The design is minimal in register and shared memory pressure:

```
Kernel 1 (gate_up_fused):
  Registers per lane:  ~6 (gate_acc, up_acc, x_val, g_val, u_val, loop counter)
  Shared memory:       0 bytes

Kernel 2 (down_reduce):
  Registers per lane:  ~5 (acc, act_val, d_val, w, loop counters)
  Shared memory:       0 bytes
```

Low register pressure maximizes occupancy (wavefronts per SIMD unit). Zero shared memory means no bank conflicts and no barrier synchronization.

### 6.4 Comparison: What Would More Work Per Wavefront Cost?

If each wavefront computed, say, 8 output scalars instead of 1:
- **8× more registers** for accumulators (gate_acc × 8, up_acc × 8)
- **8× more weight data** to load per wavefront (8 rows instead of 1)
- **Register spilling** likely at higher counts, pushing data to local memory (slow)
- **Reduced occupancy**: fewer wavefronts fit per SIMD → less latency hiding
- **Diminishing returns**: you need fewer wavefronts total, but each one is slower

In a bandwidth-bound regime, reducing wavefront count hurts because you have fewer opportunities to hide memory latency. The scheduler works best with a deep pool of independent work items.

---

## 7. Memory Traffic Analysis

### 7.1 Weight Traffic (Dominant Cost)

For one token, one MoE layer (all operations combined):

```
Gate weights:  TOP_K × INTER × HIDDEN × sizeof(weight)
Up weights:    TOP_K × INTER × HIDDEN × sizeof(weight)
Down weights:  TOP_K × HIDDEN × INTER × sizeof(weight)

Total weights: 3 × TOP_K × INTER × HIDDEN × sizeof(weight)
             = 3 × 8 × 2048 × 7168 × 1   (FP8)
             = 352 MB per token per layer
```

This is the irreducible minimum — every approach must read these weights.

### 7.2 Activation and Intermediate Traffic

**Warp decode:**
```
Input X read:          HIDDEN × sizeof(BF16) × (effectively 1, cached across wavefronts)
Intermediate write:    B × TOP_K × INTER × sizeof(BF16) = 1 × 8 × 2048 × 2 = 32 KB
Intermediate read:     same 32 KB (consumed by kernel 2, likely in L2)
Output Y write:        HIDDEN × sizeof(BF16) = 14 KB
Sorting buffers:       0
Padding overhead:      0
```

**Traditional expert-centric:**
```
Input X read:          HIDDEN × sizeof(BF16) (original)
Gather buffer:         HIDDEN × sizeof(quant_type) × TOP_K  (copy into expert-major layout)
Intermediate write:    B × TOP_K × INTER × sizeof(quant_type) (requantized)
Intermediate read:     same (for stage2 GEMM)
Sorting metadata:      sorted_ids + sorted_weights + sorted_expert_ids (O(B × TOP_K) + O(E))
Padding:               up to E × block_size × HIDDEN bytes of wasted reads
Output gather+reduce:  TOP_K × HIDDEN × sizeof(BF16) (8 partial results → 1)
```

At B=1, the traditional path's non-weight overhead (sorting, padding, gather, combine) can exceed the useful intermediate data by 10-30×.

### 7.3 Bandwidth Utilization

Warp decode sustains 58% of peak memory bandwidth (3.95 TB/s out of 6.8 TB/s measured peak on B200). The gap from 100% comes from:

1. **Non-contiguous access**: Expert routing sends wavefronts to non-adjacent experts (e.g., experts 5, 8, 14, 19...), so weight rows are scattered across HBM pages
2. **Activation vector reuse**: X[token_b] is read by many wavefronts (INTER × TOP_K wavefronts for kernel 1), but the first access is a cold miss
3. **Instruction overhead**: loop control, address calculation, type conversion

58% of peak is excellent for a non-contiguous access pattern. Traditional grouped GEMM achieves higher peak utilization *per GEMM call*, but the total wall-clock time includes the sorting, padding, and combine steps which add latency without contributing useful bandwidth.

---

## 8. Correctness and Numerical Properties

### 8.1 Accumulation Precision

Both kernels accumulate in **FP32** registers. The only precision reduction is in the weight storage (FP8/FP4/etc.) and the intermediate tensor between kernels (BF16).

The traditional path typically:
1. Quantizes activations to FP8 before stage 1
2. Runs the GEMM in FP8 × FP8 → FP32 → truncate to BF16
3. Re-quantizes intermediate activations to FP8 before stage 2
4. Runs stage 2 GEMM in FP8 × FP8 → FP32 → truncate to BF16

Each quantization step introduces rounding error. Warp decode eliminates steps 1 and 3 — the input activations stay in BF16, and the intermediate is written in BF16 without requantization.

### 8.2 Numerical Equivalence

The warp decode output is mathematically identical to a naive FP32 reference implementation (modulo weight dequantization order). There is no approximate reduction, no reordering of accumulation, and no lossy intermediate format.

### 8.3 Verification Metrics

Reported correctness against a reference implementation:
- Minimum cosine similarity > 0.999996
- Maximum absolute difference < 0.002

---

## 9. Relationship to Real-World Inference

### 9.1 Where Warp Decode Sits in the Inference Pipeline

```
┌─────────────────────────────────────────────────────┐
│                  Inference Server                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Prefill  │  │  Decode   │  │  Scheduling /    │  │
│  │ (large   │  │  (small   │  │  Batching        │  │
│  │  batch)  │  │   batch)  │  │                  │  │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘  │
│       │              │                               │
│       ▼              ▼                               │
│  Expert-centric   Warp decode                        │
│  grouped GEMM     output-centric                     │
│                                                      │
│  ◄── uses the same weight tensors, same routing ──►  │
└─────────────────────────────────────────────────────┘
```

Warp decode replaces **only** the MoE layer's compute kernels during the decode phase. Everything else is unchanged:
- The router (gating network) still runs and produces `topk_ids` and `topk_weights`
- Attention layers are unaffected
- Multi-GPU parallelism (tensor parallel, expert parallel, pipeline parallel) operates above this level — warp decode is the **local compute kernel** that runs on each GPU after inter-GPU dispatch

### 9.2 Dispatch Criteria

The serving framework chooses between warp decode and grouped GEMM based on batch size:

```
if batch_size <= WARP_DECODE_THRESHOLD:    # e.g., 32-64
    use warp_decode kernels
else:
    use grouped_gemm kernels (expert-centric)
```

The threshold depends on model dimensions, number of experts, TOP_K, and hardware. It can be determined empirically by benchmarking both paths at various batch sizes and finding the crossover.

### 9.3 Chunked Prefill Interaction

Modern inference engines use chunked prefill, where prefill tokens are interleaved with decode tokens. In a mixed batch:
- Prefill tokens (high count) → expert-centric grouped GEMM
- Decode tokens (low count) → warp decode

Some implementations may route all tokens through one path for simplicity, accepting suboptimal performance for the minority workload.

### 9.4 External Corroboration from a Public Frontier-Scale Serving Report

**Provenance.** Everything in this subsection is read off a publicly released technical report for a large open MoE model (extracted text at [`kimi-k3/refs/k3_tech_report.txt`](kimi-k3/refs/k3_tech_report.txt); the serving/kernel material is §5.4.2, pp. 24–25). It is a *model* report, not a kernel report: it states structure and motivation, but no tile shapes, instruction sequences, occupancy figures, or kernel-level measurements. None of it is measured on our kernels. It therefore sets design direction and priority; it does not substitute for the benchmarking and profiling tracked in the companion optimization docs (§12).

**The core bet is independently validated.** The report describes routed-expert MoE *decode* exactly as §2.2 does: at small batch sizes "the group GEMMs reduce to memory-bound streaming of weight matrices — a regime for which conventional tile-centric kernels are poorly suited due to their compute-oriented design and preprocessing overheads" [K3 report p.25]. Their MoE decoding kernel is built on a **token-centric** design "in which each warp is responsible for one output neuron and streams the associated weights directly from memory" [K3 report p.25]. That is the same parallelism axis as §4.1, the same explicit rejection of the expert-centric grouped-GEMM path as §3.1, and the same batch-dependent applicability as §11.1 and §11.2 — arrived at independently, for a production serving stack rather than a microbenchmark. Two details make the corroboration stronger rather than merely rhetorical:

- The scale is frontier-scale and the sparsity is extreme (16 of 896 routed experts activated per token [K3 report pp. 1, 6]), and the report names the growth in expert count and per-token expert count as precisely what makes "conventional MoE kernels" unable to sustain utilization [K3 report p.24]. Our §11.2 crossover heuristic (`B × TOP_K / E`) predicts that this shape stays in the warp-decode regime out to larger batch than the E=128, TOP_K=8 case we usually quote.
- The MoE expert weights are quantized (MXFP4 weights with MXFP8 activations, non-expert modules left in higher precision) [K3 report p.14]. So the corroborated design point is specifically a *quantized* weight-streaming decode kernel with inline dequantization — the regime §10.5 and §12.1 already target.

The report also cites, as its reference [12], the same publicly published upstream write-up that this design doc is derived from (the source of the B200 numbers in §1 and the kernel names in §10.3). The lineage is therefore explicit: this doc and their MoE decode kernel share an upstream design, and they have taken it further in two ways we have not.

**Technique we do not have (1): lane teams over disjoint expert subsets.** The report states that "to further increase parallelism, we subdivide each warp into finer-grained lane teams, each processing a disjoint subset of experts, followed by a warp-wide reduction of the partial results" [K3 report p.25]. The natural mapping onto our kernel is `down_reduce` (§5.3), which today walks `TOP_K` experts *serially* inside one wavefront while all 64 lanes split the `INTER` axis. Partitioning the wave into `T` teams of `64/T` lanes, with each team taking `TOP_K/T` experts, shortens that serial loop by `T`:

```
today:      64 lanes × (INTER / 64) elements, looped TOP_K times
lane teams: T teams × (64/T) lanes × (INTER / (64/T)) elements, looped TOP_K/T times
```

Two properties of the existing design make this cheap to attempt: the full-wave butterfly reduction (§5.4) already yields the cross-team sum with no extra code, and the routing weight is already folded into each expert's accumulator (§4.4), so partial sums originating from *different* experts are directly summable. The costs are equally concrete: per-lane element count grows by `T`, tightening the vectorized-load divisibility constraint to `INTER % (lanes_per_team × kVector) == 0` — the same alignment problem already tracked for non-1024-aligned `INTER` — and the expert index and scale metadata stop being warp-uniform, which is in tension with the landed per-workgroup scale-broadcast work (a lane-group scale broadcast in `down_reduce` was already prototyped and came out neutral-to-negative; see the v2 ticket list).

Be careful about *why* it would help here. The report's stated motivation is increasing parallelism, and it gives no team size, does not say whether teams split only the expert axis or the reduction axis as well, does not say whether it applies to the gate/up half at all (our gate/up kernel has no expert loop to shorten — it is already one wavefront per `(token, expert, neuron)` triple), and reports no measurement. The v2 profile says we are L2-capacity-bound at 45–61 % occupancy with very deep in-flight VMEM, i.e. not starved for exposed parallelism. So the mechanism by which lane teams would pay off *for us* would have to be a shorter per-wave critical path and more independent loads per wave — a different claim from the one the report makes, and one we would have to establish by measurement rather than inherit.

**Technique we do not have (2): offline weight permutation to reduce runtime dequantization.** The report states that "the weight layout is permuted offline at a one-time preprocessing cost, substantially reducing the runtime dequantization overhead" [K3 report p.25]. That is the entire description — no permutation granularity, no target instruction sequence, no quantified saving. What we can say confidently is that the *cost it attacks* is real and first-order in our kernel: dequantization happens per element in the inner loop (§10.5), and the packed-conversion experiments in the v2 optimization docs showed that the choice of conversion instruction sequence alone was the difference between a regression and a win, with VALU utilization at 60–78 %. Any change that removes shift/mask/lane-shuffle work from that loop is aimed at the right target.

What it would mean for us, stated as a hypothesis rather than as the report's mechanism: pre-interleave the packed low-precision elements at model-load time so the bytes/nibbles arrive in the order the packed-conversion builtins and the lane mapping want, and co-locate scale metadata with the block it applies to so the scale index need not be recomputed in the loop. Structurally this is a **weight-prepack step outside the kernel**, which means it lands in the dispatch/integration layer we have not built yet (§12.2), and it invalidates the assumption running through §10.3, §10.4 and §12.1 that weights are a plain `[E, N, K]` tensor with a separately described scale layout — a permuted layout has to be a declared, versioned property of the weight tensor that the kernel and the prepack step agree on, not an implicit convention. We should not assert that our guessed permutation is theirs; the report does not describe one.

**Adjacent scheduling and fusion items (context, not scope).** The same passage lists optimizations that sit outside this kernel but change how much of the MoE layer's latency is actually exposed, and therefore how much end-to-end benefit a fast routed-expert decode kernel delivers: shared-expert computation is used as the overlap partner for the latent all-gather communication rather than being serialized against it, the latent down-projection is fused with the MoE router into a single GEMM, and the latent weight matrices are sharded with the output all-gather fused into the GEMM epilogue [K3 report p.25]. (Dispatching shared-expert GEMMs to a separate stream is described on the training side [K3 report p.20], not in the decode path.) None of this is in scope here — §11.4 already places shared experts outside this op and Appendix B places EP/TP plumbing above it — but it is the right context for whoever wires warp decode into a serving stack (§9.1, §12.2), and it is a reminder that a kernel-level win can be partly or wholly hidden by scheduling decisions one level up.

---

## 10. Implementation Roadmap

### 10.1 Kernel Signatures

```
// Kernel 1: Gate + Up + Activation
void gate_up_fused(
    const activation_t*  X,              // [B, HIDDEN]
    const weight_t*      W_gate,         // [E, INTER, HIDDEN]
    const weight_t*      W_up,           // [E, INTER, HIDDEN]
    const int32_t*       router_ids,     // [B, TOP_K]
    intermediate_t*      intermediate,   // [B, TOP_K, INTER]  (output)
    int B, int HIDDEN, int INTER, int TOP_K,
    // Optional: weight scales for dequantization
    const scale_t*       w_gate_scale,   // layout depends on quant scheme
    const scale_t*       w_up_scale
);

// Kernel 2: Down + Reduce
void down_reduce(
    const intermediate_t*  intermediate,   // [B, TOP_K, INTER]
    const weight_t*        W_down,         // [E, HIDDEN, INTER]
    const int32_t*         router_ids,     // [B, TOP_K]
    const float*           router_wts,     // [B, TOP_K]
    output_t*              Y,              // [B, HIDDEN]  (output)
    int B, int HIDDEN, int INTER, int TOP_K,
    const scale_t*         w_down_scale
);
```

### 10.2 Grid and Block Configuration

```
// Kernel 1
grid  = (B, TOP_K, INTER)                      // one wavefront per output scalar
block = (WAVE_SIZE,)                            // 32 or 64 threads

// Or equivalently, flatten:
grid  = (B * TOP_K * INTER,)
block = (WAVE_SIZE,)

// Kernel 2
grid  = (B, HIDDEN)
block = (WAVE_SIZE,)
```

For very large grids, the 3D grid for kernel 1 may exceed hardware grid dimension limits. In that case, flatten to 1D and reconstruct `(token_b, expert_k, neuron_j)` from the linear index.

### 10.3 Implementation Considerations

**Weight layout**: Expert weights should be stored such that consecutive lanes access consecutive memory addresses. For `W_gate[e][neuron_j, :]`, the HIDDEN dimension should be the fastest-varying (innermost) dimension, enabling coalesced loads across lanes.

```
Preferred layout:   W_gate[expert][neuron][hidden]   — hidden is contiguous
                    so lane i loads W_gate[e][j][lane * stride + offset]
                    with stride = 1 for coalescing
```

**Activation vector caching**: X[token_b] is read by INTER × TOP_K wavefronts in kernel 1. Hardware L1/L2 caching handles this naturally for small B, but explicit use of read-only / texture cache hints may help:
- Mark X as `__restrict__ const`
- Use read-only cache load intrinsics where available

**Intermediate tensor placement**: Between kernel 1 and kernel 2, the intermediate tensor `[B, TOP_K, INTER]` is written and then read. At B=1, this is 32 KB (BF16) — small enough to remain in L2 cache. No explicit management needed; the kernel launch barrier ensures visibility.

**Vectorized loads**: Where the weight format allows (e.g., FP8 × 4 packed into 32 bits, or BF16 × 2 packed into 32 bits), use vector load instructions to increase bytes per load and reduce instruction count:

```
// Instead of loading 1 FP8 per instruction:
for i in range(lane, HIDDEN, WAVE_SIZE):
    w = load_fp8(W_gate[e][j][i])

// Load 4 FP8 values as a single 32-bit word:
for i in range(lane * 4, HIDDEN, WAVE_SIZE * 4):
    w4 = load_u32(W_gate[e][j][i:i+4])
    w0, w1, w2, w3 = unpack_fp8x4(w4)
    // accumulate all four
```

This reduces the total number of load instructions by 4× and improves instruction-level parallelism.

**Batched weight layout (3D tensor)**: The blog post names the kernels `moe_gate_up_3d_batched` and `moe_down_3d_batched`. The "3D" refers to the weight tensors being indexed as `[expert, output_dim, input_dim]`. Pre-arranging weights in this layout (rather than gathering them per-expert at runtime) avoids pointer indirection.

### 10.4 Handling Fused vs. Split Gate/Up Weights

Some models store gate and up weights as a single fused tensor `W1[e]` with shape `[2 × INTER, HIDDEN]` (gate in the first half, up in the second half). Others store them separately. The kernel should handle both:

```
// Fused: W1[e] has shape [2 * INTER, HIDDEN]
//   gate row = W1[e][neuron_j, :]
//   up row   = W1[e][INTER + neuron_j, :]

// Split: W_gate[e] and W_up[e] each have shape [INTER, HIDDEN]
//   gate row = W_gate[e][neuron_j, :]
//   up row   = W_up[e][neuron_j, :]
```

The fused layout has a locality advantage: both rows for the same neuron are HIDDEN elements apart, potentially sharing cache lines if HIDDEN is small.

### 10.5 Dequantization

Weight dequantization (FP8→FP32, FP4→FP32, MXFP→FP32, etc.) happens inline in the inner loop, converting each loaded element to FP32 before the multiply-add:

```
raw = load_weight(W_gate[e][j][i])           // e.g., FP8 E4M3
scale = load_scale(w_gate_scale, e, j, i)    // per-tensor, per-channel, or per-block
val = to_fp32(raw) * scale                   // dequantize
gate_acc += x_val * val                      // accumulate in FP32
```

The scale lookup pattern depends on the quantization scheme:
- **Per-tensor**: one scale per expert → trivial
- **Per-channel**: one scale per (expert, neuron) → loaded once per wavefront
- **Per-block (e.g., per-128)**: one scale per 128 elements of HIDDEN → loaded every 128/WAVE_SIZE iterations

### 10.6 Supporting Multiple Activation Functions

The activation between kernel 1's output and kernel 2's input is applied in kernel 1 before writing the intermediate. Parameterize it:

```
// After reduction in kernel 1:
switch (activation_type):
    case SILU:    result = silu(gate_val) * up_val        // SwiGLU
    case GELU:    result = gelu(gate_val) * up_val        // GeGLU
    case RELU:    result = relu(gate_val) * up_val
    case NONE:    result = gate_val                       // no gating (single projection)
```

For non-gated experts (single projection, no up path), kernel 1 simplifies to a single dot product per wavefront.

---

## 11. Limitations and When NOT to Use Warp Decode

### 11.1 Large Batch / Prefill

When B × TOP_K is large enough that multiple tokens share the same expert, expert-centric grouped GEMM wins because:
- The GEMM tiles fill up, achieving high compute utilization
- Sorting cost is amortized over many tokens
- Weight data loaded for one expert serves many tokens (better reuse)

Warp decode at large batch wastes bandwidth: every wavefront independently loads its weight row, with no reuse across tokens that happen to route to the same expert.

### 11.2 The Crossover Point

The crossover depends on model dimensions, but roughly:
- B × TOP_K / E < 1 (fewer than 1 token per expert on average): warp decode wins
- B × TOP_K / E > 2-4 (multiple tokens per expert): grouped GEMM wins

For E=128, TOP_K=8, this means:
- B ≤ 16: warp decode is likely faster
- B ≥ 32-64: grouped GEMM is likely faster

Benchmark both on your target hardware to find the exact crossover.

### 11.3 Dense Models

Warp decode is specific to MoE. Dense transformer layers (where every token uses the same FFN weights) should use standard GEMM kernels, which achieve near-peak compute utilization at any batch size because there is no expert routing overhead.

### 11.4 Shared Expert Patterns

Some MoE architectures (e.g., DeepSeek-V2/V3) have shared experts that process all tokens plus routed experts that process selected tokens. Warp decode applies only to the routed expert portion. The shared expert FFN should use a standard dense GEMM.

---

## 12. Implementation Status

> **Status (2026-06-03).** This section describes the original kernel family. The optimization/measurement work since then lives in a chain of companion docs:
> - [`warp_decode_optimization.md`](warp_decode_optimization.md) (v1) — Tier 1–4 optimization catalog + `WD-OPT-N` tickets.
> - [`warp_decode_optimization_v2.md`](warp_decode_optimization_v2.md) (v2) — profiling-informed re-grade (L2-capacity-bound, not HBM-bound).
> - [`warp_decode_optimization_v3.md`](warp_decode_optimization_v3.md) (v3) — **current state-of-the-world**: what landed, what was tested and dropped, the persistent-single-stage closure, the ASM-fmoe crossover, and borrowable ASM design ideas.
> - Per-experiment profiling/decision notes: [`issues/warp_decode_profiling/`](issues/warp_decode_profiling/).
>
> Headline current verdict: **split bf16 remains the production decode default** (wins e2e at B≤2 on all measured shapes); the persistent producer-owned single-stage family (V2/V3/V4) is closed; the AITER ASM all-fused single-launch kernel crosses over to a win at B≥4 on Qwen3Next-class shapes.

The warp-decode design is currently implemented as a CKTile operator on the `warp-decode` branch of `composable_kernel`. The goal at this stage is a complete, correctness-validated kernel family that can later be wired into higher-level dispatch layers (AITER, vLLM).

### 12.1 What is implemented

**Kernels** (`include/ck_tile/ops/warp_decode/`):
- `WarpDecodeGateUpKernel` — fused gate + up + SwiGLU, producing `intermediate[B, TOP_K, INTER]`.
- `WarpDecodeDownReduceKernel` — down projection with in-register weighted reduction across `TOP_K` experts.
- Grid is one wavefront per output scalar (flattened 1D), with `kVector`-wide vectorized loads across the HIDDEN / INTER reduction axis.
- Activation is pluggable via `Problem::Activation` (currently exercised with SiLU/SwiGLU).

**Data types:**
- Activations / outputs: `bf16_t`.
- Weights: `bf16_t`, `fp8_t` (OCP E4M3, guarded by `CK_USE_OCP_FP8`), and packed `pk_fp4_t` (MXFP4).
- Compute / accumulators: FP32 throughout.
- Scales: `float` and `e8m0_t` (MXFP-style).

**Scale layouts** (`WarpDecodeScaleLayout` in `pipeline/warp_decode_problem.hpp`):
- `PerTensor`, `PerToken`, and `Block2D<Block_N, Block_K>` (including `Block_N > 1`, e.g. `Block2D<128, 128>`).
- Independent `XScaleLayout` / `WScaleLayout` selection for the gate/up kernel.

**Host API** (`include/ck_tile/ops/warp_decode.hpp`):
- `launch_warp_decode_gate_up<Kernel>(args, stream_config)` and `launch_warp_decode_down_reduce<Kernel>(args, stream_config)` wrappers.
- Both wrappers call `Kernel::IsSupportedArgument(args)` before launch and throw `std::invalid_argument` on rejection.
- `IsSupportedArgument` enforces: non-null tensor pointers, positive tensor dimensions, row strides ≥ inner extent, `HIDDEN % (warp_size * kVector) == 0` (gate/up) / `INTER % (warp_size * kVector) == 0` (down), and `Block2D` divisibility constraints on `B`, `HIDDEN`, `INTER`, and `E * {inter, hidden}`.

**Tests** (`test/ck_tile/warp_decode/`, target `test_ck_tile_warp_decode`):
- GoogleTest-based, integrated into the standard CK CTest flow via `add_gtest_executable`.
- Positive coverage across dtype × scale-layout combinations, including BF16/BF16, FP8/FP8 with per-token × per-tensor, MXFP4 weights with `e8m0_t` scales, and Block2D scales with `Block_N > 1`.
- Negative coverage asserts that `IsSupportedArgument` returns `false` and the launch wrapper throws for: null pointers, non-divisible `HIDDEN`/`INTER`, invalid strides, and Block2D dimensions that violate divisibility.
- One realistic DeepSeek-V3-like shape (`B=1, HIDDEN=7168, INTER=2048, TOP_K=8, E=8`, FP8 weights + Block2D scales) with `E` bounded to keep unit-test memory footprint manageable.
- Optional verbose perf mode, gated by `CK_WARP_DECODE_VERBOSE=1`: enables `stream_config` timing and prints per-test latency, TFLOP/s, and GB/s for `gate_up`, `down_reduce`, and the combined layer.

### 12.2 What is not yet implemented

- **Dispatch integration**: no wiring into AITER / vLLM MoE dispatch yet; the kernels are exercised only through the CK unit test at this point.
- **Activation variants**: only SwiGLU (SiLU-gated) is exercised; GeGLU / ReLU-gated / non-gated paths described in §10.6 are not instantiated in tests.
- **Fused `W1[e] = [gate|up]` weight layout** (§10.4) is not supported yet; the host API currently takes split `W_gate` / `W_up` tensors.
- **Shared-expert / non-MoE fallbacks** (§11.4) are out of scope for this op and remain the caller's responsibility.
- **Multi-GPU (EP / TP) plumbing** (Appendix B) is orthogonal and not part of this branch.
- **Performance characterization** on AMD hardware (B200 numbers in §1 are from the upstream reference); no internal benchmark report has been produced yet. The verbose perf mode is the tool intended to drive that work.

---

## 13. Summary

| Property | Value |
|---|---|
| **Target workload** | MoE decode, B=1..32 |
| **Parallelism axis** | Output-centric (one wavefront per output scalar) |
| **Number of kernels** | 2 sequential (gate_up_fused, down_reduce) |
| **Sorting required** | No |
| **Padding required** | No |
| **Intermediate buffers** | One: [B, TOP_K, INTER] between the two kernels |
| **Shared memory usage** | Zero |
| **Cross-wavefront sync** | None (embarrassingly parallel) |
| **Accumulation precision** | FP32 throughout |
| **Weight precision** | Any (dequantized inline to FP32) |
| **Activation precision** | BF16 input, BF16 intermediate, FP32 accumulators |
| **Bandwidth utilization** | ~58% of peak (measured on B200) |
| **Speedup** | 1.84× over traditional path at small batch |
| **Accuracy** | 1.4× closer to FP32 ground truth |
| **Applicable to** | Any GPU with wavefront/warp execution model |

---

## Appendix A: Comparison with Expert-Centric Fused Kernels

Expert-centric fused kernels (e.g., AITER's `fmoe_g1u1`) represent a middle ground — they fuse the sorting, GEMM, activation, and combine steps into fewer kernel launches while retaining the expert-centric parallelism axis. The key differences from warp decode:

1. **Still requires sorting**: `moe_sorting()` produces `sorted_ids`, `sorted_expert_ids` — the kernel iterates over expert blocks
2. **Still pads**: Token lists are padded to `block_size_M` (typically 32) per expert
3. **Still uses tiled GEMM**: Work is organized as M×N tiles where M is the token dimension within one expert — these tiles are partially empty at small B
4. **Still has intermediate requantization**: Between stage 1 and stage 2 GEMMs in the 2-stage variant

These fused kernels are more efficient than the fully disaggregated 8-stage pipeline but less efficient than warp decode at B=1 because they retain the expert-centric data layout assumptions.

## Appendix B: Multi-GPU Integration

Warp decode is orthogonal to multi-GPU parallelism strategies. It serves as the **local compute kernel** within any of these configurations:

- **Expert Parallelism (EP)**: Each GPU holds a subset of experts. An all-to-all dispatch sends tokens to the GPU holding their routed expert. Warp decode runs locally on each GPU for the experts it owns.
- **Tensor Parallelism (TP)**: Each expert's weights are sharded across GPUs along one dimension. Each GPU runs warp decode on its weight shard, then an all-reduce combines partial results.
- **EP + TP hybrid**: Common at scale. EP across nodes, TP within a node. Warp decode runs at the innermost level on each GPU.

The kernel signatures accept `router_ids` that reference **local** expert indices (0..E_local-1), where E_local is the number of experts on this GPU. The multi-GPU dispatch layer is responsible for remapping global expert IDs to local IDs and handling the inter-GPU communication.
