# C1 complete Q/KV preparation ablation

Scope: BF16 input4096 -> block128 FP8 quant -> combined QA1024/KV512 WMMA -> split16 atomic reduction -> BF16 round -> first weighted RMSNorm -> Qlatent block128 FP8 quant -> QB32768 WMMA -> BF16 round -> per-head RMSNorm512 -> interleaved RoPE last64 channels. KV weighted norm -> same RoPE. Output BF16. All quantization is timed. No attention/cache write/MoE/serving.

Variants: separate stages; two launches (QA input quant/projection/weighted norm/latent quant/KV RoPE, then QB/norm/Q RoPE); single launch with one resident grid barrier between these regions. BF16 WMMA throughout. No native FP8 WMMA due to earlier accuracy rejection.

Ownership: QA producers192 CTAs (12 N128 tiles x split16) atomically update1536FP32. Two padded group counters elect final Q/KV producers after explicit device drains and CTA barrier. Q final producer publishes1024FP8 +8E8M0; KV final producer publishes512BF16 rotated output. Counters and accumulators reset within invocation. QB256 N128 tiles each stores128BF16-rounded FP32 outputs; four producers per512-channel head use a counter to elect its norm/RoPE producer, which reads full head and writes512BF16. Head counters reset. No polling in two-launch path.

Single launch:192 resident CTAs, each QA task then bounded device-scope grid barrier, then QB tasks assigned round-robin. Requires at least192 CUs and sufficient occupancy; validate compiled resources before timing. Grid counter monotonically advances; timeout health flag must stayzero. Independent state per stream, sequential reuse only. All writable device loads/stores device scope; readonly weights use.ca.

Validation: independent FP64 GEMMs using exact quantized operands, model precision boundaries, Q/KV outputs separately; also stagewise checks to distinguish propagated quantization differences. Same0.1% relative-RMS acceptance threshold. Two complete timing rounds, report onlysecond. Saved fixture includes exact QB weights and rotary table; benchmark is isolated, not serving.
