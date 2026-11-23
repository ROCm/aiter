# SiTU-and-mul backward for Kimi-K3 / K3.1

Owner: @xiangM99. Draft reservation for https://github.com/ZhangDanyang-AMD/Lumen/issues/35 (section 1.4).

Status: scope placeholder only; kernel implementation, correctness tests and GPU timings are pending.

Planned scope:
- Implement gradients for both gate and up inputs of SiTU-and-mul.
- Cover Kimi routed-expert inputs shaped [T, 3072], with beta=4 and lambda=25.
- Match the Kimi SiTU forward semantics; coordinate with the separate forward work in ROCm/aiter#7167.
- Validate BF16 gradients against an FP32 PyTorch reference on gfx950 and report tolerances and timings.

This draft reserves the SiTU activation backward task only. Grouped-GEMM backward, router gradients and complete LatentMoE integration remain separate tasks.
