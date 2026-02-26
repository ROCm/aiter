# 🧩 Instruction Pipelining on AMD GPUs

[← Kernel Optimization](kernel_optimization.md)

---

Instruction pipelining on AMD GPUs involves not only overlapping arithmetic and memory instructions within individual threads but also coordinating **wavefront-level pipelining** across groups of threads. Effective pipelining ensures that the GPU’s compute units (CUs) remain busy and that latency from memory or synchronization is hidden by other active wavefronts.

This guide expands on pipelining concepts — both **intra-thread** (instruction-level) and **inter-thread** (wavefront-level) — and explains how to design kernels to maximize throughput and minimize stalls.

---

## ⚙️ What Is Instruction and Wavefront Pipelining?

Each AMD GPU compute unit (CU) supports multiple **wavefronts** (groups of 32–64 threads executing in SIMD). While one wavefront executes, others can wait on memory, synchronization, or dependencies. The hardware scheduler interleaves wavefronts to keep pipelines active.

| Level                       | Description                                                   | Example                                              |
| --------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- |
| **Instruction-level (ILP)** | Overlaps independent instructions from a single thread.       | Arithmetic and memory ops executing in parallel.     |
| **Wavefront-level (WLP)**   | Interleaves execution of multiple wavefronts to hide latency. | One wavefront executes while another waits for data. |

This dual-level pipelining model allows AMD GPUs to sustain high throughput by always having work available for execution.

---

## 🧠 Pipeline Stages and Scheduling

AMD GPUs use multiple pipelines per CU:

| Pipeline                   | Function                                                                        |
| -------------------------- | ------------------------------------------------------------------------------- |
| **Scalar (SALU)**          | Executes per-wavefront scalar operations, such as loop indices or conditionals. |
| **Vector (VALU)**          | Handles per-thread arithmetic instructions in SIMD fashion.                     |
| **Memory (VMEM)**          | Manages global and texture memory access.                                       |
| **LDS (Local Data Share)** | Handles fast shared-memory reads/writes between threads.                        |
| **Control / Branch**       | Manages synchronization and branching flow.                                     |

Each CU scheduler can issue instructions from different wavefronts to different pipelines each cycle — as long as there are no dependencies or synchronization barriers.

---

## 🔄 Wavefront Pipelining in Practice

When one wavefront stalls (e.g., waiting for memory), another can take over the CU’s pipelines. This is the primary mechanism for **latency hiding**.

Example:

* **Wavefront A**: Waiting for global memory load.
* **Wavefront B**: Executing arithmetic instructions.
* **Wavefront C**: Performing LDS writes.

By rotating through active wavefronts, the GPU keeps all pipelines busy and minimizes idle cycles.

---

## 🔁 Synchronization and Pipeline Stalls

Synchronization primitives (e.g., `__syncthreads()` in HIP) can **pause all threads in a workgroup**, creating potential pipeline stalls. While synchronization is necessary for correctness, it must be placed carefully to avoid unnecessary blocking.

### Example:

```cpp
__shared__ float buf[64];
int tid = threadIdx.x;

// Load phase
buf[tid] = global_in[tid];
__syncthreads();  // Synchronization barrier

// Compute phase
float res = buf[tid] * coeff;
```

If synchronization is too frequent, it interrupts wavefront pipelining by forcing threads to wait. Use barriers only when threads truly depend on each other’s data.

### Tips:

* Reduce the number of `__syncthreads()` per kernel.
* Combine compute and memory operations between barriers.
* Exploit **wavefront-level independence** — each wavefront can progress independently unless explicitly synchronized.

---

## 🧩 Techniques to Improve Thread and Wavefront Pipelining

### 1. **Overlap Computation and Memory Loads**

Prefetch or load data in advance to hide memory latency:

```cpp
float next = global[i + stride]; // Prefetch
float current = process(global[i]); // Compute
```

While one instruction waits for memory, another executes.

### 2. **Balance Work Across Threads**

Ensure all threads in a wavefront do similar work to avoid divergence. Divergent branches (e.g., `if`/`else` where only some threads execute) reduce pipeline efficiency because inactive lanes stall the wavefront.

### 3. **Use LDS (Shared Memory) for Cross-Thread Pipelining**

Threads can share data through the **Local Data Share (LDS)** memory to enable multi-phase pipelines within a block:

```cpp
buf[tid] = stage1(data[tid]);
__syncthreads();
float res = stage2(buf[tid]);
```

Here, stage 1 and stage 2 can be pipelined across wavefronts, as one group computes stage 1 while another executes stage 2.

### 4. **Exploit Asynchronous Operations**

Use asynchronous copies (`hipMemcpyAsync` or `__builtin_amdgcn_s_waitcnt`) to overlap computation with data transfer. This enables true pipeline overlap between host-device or global-local stages.

---

## ⚖️ Pipelining vs. Occupancy

While pipelining hides latency, it increases **register and resource usage** since multiple instructions and wavefronts are active at once. More registers per thread can reduce **occupancy** (the number of active wavefronts per CU).

The optimal configuration balances:

* Enough wavefronts to cover latency.
* Enough registers to support ILP.
* Avoiding oversubscription that causes spills.

Use **Radeon GPU Profiler (RGP)** to examine occupancy and stall reasons (e.g., memory, synchronization, dependencies).

---

## ⚠️ Common Pitfalls

* **Over-synchronization:** Too many barriers block inter-wavefront pipelining.
* **Imbalanced Workload:** Divergent threads cause underutilized pipelines.
* **Dependency Chains:** Reduce ILP and force serial execution.
* **High Register Pressure:** Limits wavefront concurrency.

---
[← Kernel Optimization](kernel_optimization.md)