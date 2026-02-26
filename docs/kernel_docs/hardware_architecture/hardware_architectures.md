# 🧱 Hardware Architectures

[← Back to Index](../index.md)

---

## 🧭 Introduction

AMD GPUs are built around Compute Units (CUs), each containing multiple wavefronts (64-thread SIMD units) that execute instructions in lockstep. Key components include:

Vector General Purpose Registers (VGPRs): Store per-thread data
Local Data Store (LDS): On-chip shared memory for fast inter-thread communication
Global Memory & L2 Cache: High-capacity memory with higher latency than LDS
SIMD and ALU pipelines: Execute arithmetic and logic instructions efficiently
This architecture allows AMD GPUs to achieve high throughput by balancing compute, memory, and parallelism, making them well-suited for AI, HPC, and graphics workloads.

Hardware architecture documentation can be found here...

[AMD GPU architecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html)


---

## 🧮 Sections

- [Overview](overview.md)
- [XGMI](xgmi.md)
- [Inspect installed hardware (rocminfo)](gpu.md)
- [Hardware architectures comparisons](hardware_comparisons.md)
---

[← Back to Index](../index.md)
