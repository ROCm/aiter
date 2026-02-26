# Register Spilling

[← Kernel Optimization](kernel_optimization.md)

---

## 📊 GPU Register Spilling and Occupancy

GPUs are throughput-oriented machines, well-suited for highly parallel workloads. Parallelism allows GPUs to hide the high cost of memory accesses with computation. The performance of GPU kernels is strongly influenced by **occupancy**, which refers to the number of wavefronts running in parallel.

Occupancy depends on the resources used by each wavefront, such as registers and local data storage (LDS). Registers are a limited resource, so the number of registers used per thread is a key factor in determining occupancy. If a single wavefront uses fewer registers, the hardware scheduler can run more wavefronts concurrently.

The number of registers a wavefront uses is determined during the compiler’s **register allocation** phase. Register allocation, in turn, depends on the **instruction order** produced by the instruction scheduler. The instruction order affects **register pressure (RP)**, and minimizing RP is essential to maximize occupancy.


## 📊 GPU Occupancy vs Register Usage

This table illustrates how **register pressure** (number of vector general-purpose registers, VGPRs) affects **occupancy**, i.e., the number of warps (wavefronts) that can run per Execution Unit (EU) and per Compute Unit (CU) on a GPU.

| Number of VGPRs | Occupancy per EU | Occupancy per CU |
|----------------|-----------------|----------------|
| <= 64          | 32 waves        | <= 648 waves   |
| <= 72          | 28 waves        | <= 727 waves   |
| <= 80          | 24 waves        | <= 806 waves   |
| <= 96          | 20 waves        | <= 965 waves   |
| <= 128         | 16 waves        | <= 1284 waves  |
| <= 168         | 12 waves        | <= 1683 waves  |
| <= 256         | 8 waves         | <= 2562 waves  |
| > 256 (with spilling) | 1 wave  | 4 waves        |

> **Notes:**
> - Fewer registers per thread allow more wavefronts to run concurrently, increasing occupancy.
> - When register usage exceeds the hardware limit (>256 VGPRs per thread), spilling occurs, reducing occupancy drastically.

This table can be used to understand and optimize GPU kernel performance by balancing **register usage** and **wavefront occupancy**.


# ⚙ Compiler Register Allocation and Trade-offs

Register allocation is a critical step in GPU compilation, but it is an **NP-Hard problem**. NP-Hard problems cannot generally be solved efficiently for all cases, so compilers must make compromises in terms of speed, generality, or correctness (optimality) to find a feasible solution.

Compilers often compromise on correctness by applying **heuristics**, which produce mostly correct solutions. These heuristics aim to range from:
- Finding the best solution in most cases, to
- Finding a solution close to the best in most cases.

Helping the compiler can involve transforming a "bad" case into one that the compiler can optimize more effectively. Compilers optimize not only for **occupancy** but also for **instruction-level parallelism (ILP)**, which minimizes schedule length. However, these two goals can conflict:

- Higher ILP often requires more registers.
- More registers per thread decrease occupancy.

In codes with high register pressure, obtaining good ILP can significantly impact final performance.

# 📂 Scratch Use on AMD GPUs

**Scratch** is an address space on AMD GPUs roughly equivalent to **local memory** in CUDA. It is thread-local global memory with interleaved addressing and is primarily used for **register spills** and **stack space**.

## Sources of Scratch Use

### 1. Register Spills
- During **register allocation** at compile/link time, if there are not enough registers available, variables will be **spilled** to scratch memory.
- The compiler tracks **variable liveness** and tries to maximize register reuse and minimize spilling, which is an **NP-Hard problem**.
- Spilling is not always detrimental; sufficient **occupancy** can hide the extra cost of memory access.

### 2. Memory Objects
- Any variable is inherently a **memory object**.
- The compiler may attempt to place it in **registers** or **local data storage (LDS)**, but this is not always possible, especially for large objects.

Scratch usage is thus a combination of compiler decisions and hardware resource limitations, impacting performance depending on occupancy and memory access patterns.


## 🛠 How to Identify Scratch Use and Register Spilling on AMD GPUs

Understanding scratch use and register spilling can help optimize GPU performance. Here are the main methods to identify them.

### 1. Using ISA (Assembly Code)
- Looking at the **ISA** is the simplest approach.
- Tools like `roc-obj / disassembly` can provide insights but may lack metadata to differentiate between a spill and regular scratch use.
- Using the `--save-temps` option  during compliation generates ISA files with detailed statistics on register and scratch usage, including spills.
- Note: In rare cases, `--save-temps` may alter the generated ISA slightly.
  
The following example compiles the HIP code `vector_add` and dumps the ISA to the terminal.
```bash
hipcc -O2 --save-temps vector_add.cpp -o vector_add
roc-obj -o output vector_add
llvm-objdump -d output/vector_add\:1.hipv4-amdgcn-amd-amdhsa--gfx942 
```

### 2. Direct Indicators of Spills (with `--save-temps`)
If the kernel is compiledusing the `--save-temps` option the intermediate ISA and metadata is maintained.
- In the `hip-amdgcn-amd-amdhsa-gfxXXX.s` file, check for:
  - `.sgpr_spill_count`
  - `.vgpr_spill_count`
- When compiling with debugging symbols (`-g --ggdb`), look for comments such as `; 4-Byte Folded Spill`.

### 3. Direct Indicators of Scratch Use (with `--save-temps`)
- Non-zero `;ScratchSizeXX` values in the metadata after the kernel indicate scratch usage.

### 4. Indirect Indicators of Scratch Use (also in disassembly)
- Instructions that access memory indirectly, e.g.,
```
buffer_store_dword v18, off, s[0:3], 0 offset:160
```
  may indicate spill-related scratch usage.

These methods allow developers to pinpoint register spills and scratch usage, which can then guide optimizations for occupancy and performance.


## 🛠 How to Reduce Register Spilling on AMD GPUs

Reducing register spilling can improve GPU performance and occupancy. Here are some practical strategies:

### 1. Avoid Allocating Data on the Stack in a Kernel
- Memory allocated on the stack resides in **scratch memory**.
- Where possible, data may be optimized into registers, reducing scratch usage.

### 2. Avoid Passing Large Objects as Kernel Arguments
- Function arguments are allocated on the stack.
- Large objects may cause spills; passing smaller arguments or using pointers can help.

### 3. Avoid Writing Very Large Kernels
- Kernels with many function calls (including math functions and assertions) can become too big.
- All device functions are currently **inlined**, which increases register usage.

### 4. Keep Loop Unrolling Under Control
- Excessive unrolling increases instruction-level parallelism but also register usage.
- Balance unrolling to minimize spills while maintaining performance.

### 5. Move Variable Declarations/Assignments Close to Their Use
- Declaring variables just before they are used reduces their live range.
- Shorter live ranges help the compiler allocate registers efficiently.

### 6. Manually Spill to Local Data Storage (LDS)
- For very large temporary variables, consider manually storing them in **LDS** to relieve register pressure.
- This can help maintain higher occupancy and reduce automatic spills to scratch.

### 7. Use `__launch_bounds__` to Suggest Compiler Hints
- Example:
  ```cpp
  __global__ void __launch_bounds__(256, 1) my_kernel(...) {
      // kernel code
  }
  ```
- First argument: **MAX_THREADS_PER_BLOCK** (default 1024)
- Second argument: **MIN_WARPS_PER_EU** (default 1)

### 8. Understand Register Classes and Address Root Causes
- **Vector registers (VGPRs):** Store variables unique to each thread in a wave; most local variables go here.
- **Scalar registers (SGPRs):** Store variables uniform across threads in a wave (e.g., kernel arguments, pointers, constants).
- **SGPRs can only spill into VGPRs.** High SGPR usage can increase VGPR-based scratch spilling.
- Try to identify which register class is causing spills and address the root cause in your code.
---
[← Kernel Optimization](kernel_optimization.md)