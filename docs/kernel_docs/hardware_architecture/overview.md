# 🧩 High-Level Overview of AMD GPU Hardware Components

[← Hardware Architectures](hardware_architectures.md)

---
AMD GPUs are built around a highly parallel, hierarchical design that allows thousands of lightweight threads to run simultaneously.  
The architecture can be broadly divided into **compute**, **memory**, and **control** components.

---


## 🧮 1. Compute Units (CUs)

**Definition:**  
The **Compute Unit (CU)** is the fundamental building block of an AMD GPU.  
Each CU is roughly analogous to a CPU core but designed for throughput instead of single-thread performance.

**Purpose:**  
A CU executes many small parallel tasks (called *work-items*) in groups (*wavefronts*).  
It contains the hardware needed to perform arithmetic, logic, and control flow operations.

---

## 🧠 2. SIMD Units

**Definition:**  
Each CU contains multiple **SIMD (Single Instruction, Multiple Data)** units — vector processors that execute the same instruction across multiple data elements simultaneously.

**Purpose:**  
They provide the GPU’s massive data-parallel compute power, executing vector instructions across dozens of threads in lockstep.

---

## 🧵 3. Wavefronts

**Definition:**  
A **wavefront** is a group of 64 threads that execute together on a SIMD unit.

**Purpose:**  
It’s the basic scheduling unit of execution — all threads in a wavefront execute the same instruction at the same time, maximizing efficiency for uniform workloads.

---

## 🧮 4. Scalar Unit

**Definition:**  
A **scalar unit** in each CU handles operations that are uniform across all threads in a wavefront (e.g., loop counters or constants).

**Purpose:**  
This avoids redundant computations across threads and improves efficiency by executing shared control logic once per wavefront.

---

## 💾 5. Registers

**Definition:**  
Fast on-chip storage locations for active threads.

**Types:**
- **VGPRs (Vector General Purpose Registers):** Per-thread registers used by SIMD units.
- **SGPRs (Scalar General Purpose Registers):** Shared across a wavefront, used by the scalar unit.

**Purpose:**  
Hold temporary data and operands close to the execution units for fastest access.

---

## ⚡ 6. Local Data Share (LDS)

**Definition:**  
A small block of fast on-chip memory (typically 64KB per CU) that all threads in a workgroup can access.

**Purpose:**  
Used for data sharing and reuse between threads, reducing trips to slower global memory.

---

## 🧱 7. Cache Hierarchy

AMD GPUs use a **multi-level cache** structure to reduce memory latency and improve data reuse.

| Level | Location | Scope | Purpose |
|--------|-----------|--------|----------|
| **L0 / L1 Cache** | Inside CU | Per CU | Stores recently used instructions/data |
| **L2 Cache** | Shared across GPU | All CUs | Buffers large data transfers, reduces global memory pressure |

**Purpose:**  
The cache hierarchy hides memory latency and improves bandwidth utilization by keeping frequently accessed data on-chip.

---

## 🧩 8. Global Memory (VRAM / HBM)

**Definition:**  
The large off-chip memory (e.g., HBM2e) connected to the GPU.

**Purpose:**  
Holds program data, textures, and kernel inputs/outputs.  
Offers high bandwidth (hundreds of GB/s) but higher latency than on-chip memories.

---

## 🧭 9. Memory Controllers

**Definition:**  
Hardware interfaces that connect the GPU core to the external memory (VRAM/HBM).

**Purpose:**  
Manage the flow of data between compute units and memory.  
They optimize access patterns to sustain high throughput.

---

## 🕹️ 10. Command Processor (CP)

**Definition:**  
The GPU’s control unit that interprets commands from the host (CPU).

**Purpose:**  
Dispatches work (kernels) to the compute units, handles synchronization, and manages queues of pending operations.

---

## 🔄 11. Scheduler and Dispatcher

**Definition:**  
Internal hardware that decides which wavefronts to run and when.

**Purpose:**  
Keeps the GPU busy by switching between wavefronts to hide memory latency.  
If one wavefront stalls waiting for data, another can execute immediately.

---

## 🧩 12. Infinity Fabric / XGMI Links

**Definition:**  
High-speed interconnects used to connect multiple GPU dies (chiplets) or multiple GPUs in a node.

**Purpose:**  
Enables **peer-to-peer (P2P)** data transfers and unified memory access between GPUs.  
Critical for scaling performance in multi-GPU HPC systems (e.g., AMD Instinct MI250/MI300).

---

## 🔌 13. DMA Engines (SDMA)

**Definition:**  
Specialized units for **Direct Memory Access (DMA)** transfers that move data without involving compute units.

**Purpose:**  
Handle background data movement (like host↔device copies) while the compute units execute kernels.

---

## 📦 14. Hardware Queues

**Definition:**  
Command submission queues managed by the host and processed by the GPU.

**Purpose:**  
Allow multiple kernels or command streams to run concurrently, supporting multi-tenant or asynchronous workloads.

---

## 🧰 15. Matrix / Tensor Cores (CDNA & RDNA3+)

**Definition:**  
Specialized hardware units for matrix operations (FP16/BF16/INT8, etc.) on newer architectures.

**Purpose:**  
Accelerate AI, ML, and HPC workloads that rely on dense linear algebra.

---

# 🏁 Summary Table

| Component | Description | Purpose |
|------------|--------------|----------|
| **Compute Unit (CU)** | Main execution block | Runs kernels and manages SIMD units |
| **SIMD Units** | Vector ALUs | Perform parallel math across many threads |
| **Wavefronts** | 64-thread execution groups | Simplify scheduling and parallel execution |
| **Scalar Unit** | Executes uniform instructions | Avoid redundant operations across threads |
| **Registers (VGPR/SGPR)** | Fast on-chip storage | Hold per-thread and per-wave data |
| **LDS** | Shared on-chip memory | Enable thread cooperation and data reuse |
| **Caches (L0/L1/L2)** | Multi-level on-chip memory | Reduce latency and bandwidth pressure |
| **Global Memory (HBM)** | External GPU memory | Store large datasets |
| **Memory Controllers** | Interface to HBM | Manage high-speed data transfers |
| **Command Processor** | Control unit | Dispatch and manage GPU workloads |
| **Scheduler/Dispatcher** | Wavefront manager | Keep compute units fully utilized |
| **Infinity Fabric / XGMI** | Interconnect fabric | Multi-GPU data sharing |
| **DMA Engines** | Copy hardware | Move data independently of compute |
| **Matrix/Tensor Cores** | Specialized compute units | Accelerate matrix/AI operations |


[← Hardware Architectures](hardware_architectures.md)

---
