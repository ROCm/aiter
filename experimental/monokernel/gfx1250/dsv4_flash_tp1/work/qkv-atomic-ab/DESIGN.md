# Isolated Q-A/KV split-K and RMSNorm A/B

C1 Flash layer0: input4096, Q latent1024, KV512. Packed FP8/E8M0 checkpoint weights, BF16 WMMA operands, FP32 accumulation. BN32, splitK8, K512/task; 48 output tiles x8=384 producer CTAs,4waves/CTA. No Q-B, attention, MoE, quantization-of-input timing, or serving claims. Norm is weighted RMS over1024 Q latent or512 KV. Preserve BF16 rounding after split reduction, float32 norm output. No RoPE in this ablation: isolate the split-K/partial-squares proposal.

A: scratch partial projection then separate explicit reduction+norm (2launches).
B: zero accumulator/counters then atomic projection+last-arrival tile sumsq+last-arrival group norm (2launches).
C: zero then atomic projection then separate norm (3launches), an attribution control.
D: zero then scratch projection+last-arrival explicit reduction+sumsq+group norm (2launches), same completion protocol as B, no float atomics.

Edges: each (tile,split) owns scratch[split,channel] for A/D; B/C atomic-add Acc[channel]. B/D each producer signals Count[tile] only after payload completion. Last of8 owns completed tile, rounds BF16, stores Acc tile and SS[tile], then signals Count[48+group]. Last of32 Q tiles /16 KV tiles reduces SS and normalizes the completed Acc group. All synchronization GPU scope; payload buffer operations DEV scope. No polling, no residency requirement; no inter-CTA scheduling deadlock. Init is a prior ordered kernel in every invocation/captured replay. No overlapping invocations share scratch. Counters reset each invocation, so no epoch wrap.

Budget:4waves, WMMA16x32x128 operand chunks; inspect compiled VGPR/LDS/spills, float atomic ISA and counter ordering. Fixed disjoint scratch sizes:8*1536 FP32 partials,1536 FP32 accumulators,48 FP32 sums,50 int32 counters. Test sequential runs and changing-input graph replays including poisoned scratch/reset. Time complete graph sequences with initialization included, warmup +two rounds, report round2 only with alternating A/B order across samples. Also retain per-kernel profiler timings.

E/F repeat B/D with explicit completion + relaxed GPU counters. G repeats E, but final consumer zeroes its completed group accumulator/counters after loading the completed output. All group producers have finished before reset. Sequential same-stream invocations only, initial zero before first use.
