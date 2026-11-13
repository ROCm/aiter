# Opt-in gfx950 HGEMM candidates

These extensions add tuning choices to `flydsl_hgemm`. Existing CSV rows,
legacy kernel names and default choices are unchanged. **Capability support
is not a speedup claim.** Benchmark the complete invocation before selecting
a new configuration.

## Wide panels

Small-M BF16 tuning appends full-tile panels of N width112/160/192/224 after
the historical candidate/pruning pass. The FT kernel already expresses these
panels. New enumeration bypasses the old N-repeat cutoff but still validates
alignment, LDS capacity and DMA limits. The historical pruning population is
unchanged, so the additional candidates do not prune existing choices.

Tuning takes longer with a larger catalog. Numeric indices in the tuner's
sorted catalog may shift. Persisted FlyDSL rows execute by the stable
`kernelName`, not that index; no existing CSV is rewritten.

## MFMA32

Use `mma_m=32, mma_n=32, mma_k=16` with the existing `flydsl_hgemm` API.
For example:

    flydsl_hgemm(
        a, w, out=out, block_m=64, block_n=64, block_k=64, stages=3,
        m_waves=2, n_waves=2, k_waves=1, split_k=1, policy="ft",
        mma_m=32, mma_n=32, mma_k=16,
    )

A/W are BF16 [M,K]/[N,K]. Default MFMA remains16x16x32. MFMA32 is
gfx950 BF16 NT full-tile only, with `k_waves=1` and block_k divisible by64
for the current LDS swizzle. Existing FT bias, BF16/FP32 outputs and global
split-K mechanisms are reused. BF16 atomic split-K retains its original
intermediate-rounding contract; this extension does not repair it.

MFMA32 local slice-K is rejected: its BF16 partial rounding failed strict
random-reference qualification. New names add `_mi32x32x16` before the
policy suffix. Legacy names omit it and retain their original parse results.

## Balanced persistent Stream-K

Positive `streamk_blocks` explicitly selects a new balanced scheduler:

    flydsl_hgemm(
        a, w, out=out, block_m=64, block_n=64, block_k=64, stages=3,
        m_waves=2, n_waves=2, k_waves=1, split_k=1, group_m=0,
        policy="ft", streamk_blocks=224,
    )

Zero remains the default. This is not a port of the complete hipBLASLt
Stream-K heuristic, nor a claim of matching its speed. Restrictions:

- gfx950 BF16 input/output, NT full-tile, block_k divisible by64;
- no bias, scaling, stagger or B non-temporal hint;
- split_k=k_waves=1, group_m=0;
- 1..1024 workers, at most the total number of K tiles;
- positive M/N/K, K divisible by block_k, standard FT N/alignment limits;
- addressed tensor/scratch byte spans and schedule products fit signed i32.

M/N edges and legal padded leading strides are supported. Public wrapper
contiguous copies for the extensions are enqueued on the explicit launch
stream. Direct NT calls keep runtime leading strides.

### Work assignment

For T output tiles, Q K tiles per output tile and W workers, worker w owns:

    [floor(w*T*Q/W), floor((w+1)*T*Q/W))

It walks the output tiles intersecting that interval. Short segments use a
guarded one-tile LDS pipeline rather than loading beyond the interval.
A wholly owned output tile writes directly. Shared tiles use FP32 scratch
with layout [tile, contributor_slot, row//4, column, row%4].
Contributors are consecutive worker IDs; storage is conservatively bounded:

    slots = ceil(W/T) + 1
    partial_bytes = T * slots * block_m * block_n * sizeof(float)
    counter_bytes = T * sizeof(int32_t)

All lanes release their scratch stores before a workgroup barrier; the leader
publishes an agent-scope acq_rel counter RMW. Another workgroup barrier
distributes the arrival ticket. The last publisher executes an agent acquire
fence and reads/sums the partials in increasing worker/K order, then rounds
once to BF16. No workgroup spins awaiting another workgroup's scheduling.
Barriers protect LDS union reuse between persistent tiles.

Fixed inputs and worker partitions are deterministic. Changing worker count
changes floating-point association; byte equality to sequential accumulation
or a different partition is not promised.

### Workspace and cost

Each invocation owns scratch and counters allocated on the launch stream.
Counters are zeroed each call, including every graph replay. There is no
global mutable scratch cache. Independent captured invocations own separate
allocations. Compiled dispatch is keyed by device/dtypes/shapes/strides,
constexpr config and scratch slot count, never by tensor data pointers.

Execution includes **counter initialization plus one GEMM kernel with fused
final reduction**. It is not an initialization-free single-kernel operation.
Time allocations, initialization and the GEMM/reduction together. Standard
agent-scope fences are intentionally conservative and can be expensive.
This initial capability is not selected as a faster runtime default.

New names add `_skg<W>` and optionally `_mi32x32x16` before it. The tuner and
tuned-config wrapper forward both extensions. New cache identities distinguish
them from old binaries.

## Validation

The existing HGEMM test covers new names, rejection, cache-warmed invalid
types, interval/slot coverage, M/N edges, short segments, both MMA shapes,
bias/output modes, poisoned output, input immutability, workspace/output
canaries, padded strides, cold capture and concurrent stream replay.

    python op_tests/test_flydsl_hgemm.py -s 65,120,320 128,256,256 128,3584,7168 128,7168,3584
    python op_tests/test_gemm_a16w16_latent.py --seed 2417
    python -m pytest -q op_tests/tuning_tests/test_a16w16_tuned_config_lookup.py op_tests/tuning_tests/test_config_shape_collision.py

Sparse dyadic tests for legacy BF16 atomic reductions verify ownership without
loosening strict random-reference tolerances elsewhere. Existing strict
random-input limitations are not claimed fixed by MFMA32 alone.

Independent fresh-cache checks of representative old paths produced identical
ELF objects. Existing CSV names parse identically. These checks do not promise
universal end-to-end latency under arbitrary host/GPU contention.
