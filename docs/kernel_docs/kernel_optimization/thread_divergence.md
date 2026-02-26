# 🛠 How To: Understanding and Avoiding Thread Divergence on AMD GPUs

Thread divergence occurs when threads in the **same wavefront (warp)** follow **different execution paths** due to conditional branches. This can lead to **serialized execution**, reducing GPU throughput.

---

## 📌 1. What is Thread Divergence?

- On AMD GPUs, a **wavefront** is typically **64 threads**.  
- When threads in a wavefront evaluate a branch differently (`if` / `else`), the hardware executes **all paths serially**, masking inactive threads.  
- Divergence reduces **SIMD efficiency**, meaning fewer threads are active simultaneously.

### Example: Divergence in HIP
```cpp
__global__ void divergent_kernel(int *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Divergent branch based on thread id
    if (tid % 2 == 0) {
        data[tid] += 1;
    } else {
        data[tid] -= 1;
    }
}
```
- Half the threads execute one path, half the other.  
- GPU executes **even threads**, then **odd threads**, effectively **halving parallel efficiency**.

---

## ⚡ 2. Performance Impact

- **Reduced occupancy** – fewer threads actively computing in a wavefront.  
- **Serialization of execution** – divergent paths run one after another.  
- **Increased register pressure** – compilers may allocate extra registers for different execution paths.  
- **Memory latency amplification** – divergent paths may access memory differently, reducing coalescing.

> On AMD GPUs, divergence in a wavefront can **double execution time** in worst-case scenarios if half the threads take one path and half take another.

---

## 🛡 3. How to Avoid or Minimize Divergence

### 3.1. Use Predication

- Replace `if-else` with **predicated operations** when possible.
```cpp
int flag = (tid % 2 == 0);
data[tid] += flag ? 1 : -1;
```
- Both outcomes are calculated, but inactive threads are masked efficiently.

### 3.2. Group Threads by Condition

- Reorganize threads so threads **within the same wavefront follow the same path**.
```cpp
// Sort or partition data such that similar conditions are in the same warp
```

### 3.3. Avoid Divergent Loops

- Ensure loop bounds or conditions are **uniform across threads in a wavefront**.

### 3.4. Use Inline Functions and Templates

- Let the compiler **specialize code** per condition, reducing runtime branches.

### 3.5. Branchless Algorithms

- Use **mathematical operations** instead of conditional branching.
```cpp
data[tid] += 1 - 2*(tid % 2);  // equivalent to if-else without branches
```

---

## 📝 4. Profiling Divergence Using `rocprofv3`

ROCm provides `rocprofv3` for GPU performance profiling, including **wavefront efficiency** metrics.

### 4.1. Install and Setup
```bash
# Ensure ROCm 5.x or later is installed
rocprofv3 --version
```

### 4.2. Collect Metrics
```bash
rocprofv3 --stats -i ./my_hip_app
```
- Use metrics:
  - `SQ_WAVES_ACTIVE` – Active waves executing.
  - `SQ_WAVES_EFFICIENCY` – Ratio of active threads vs total threads.
  - `SIMD_DIVERGENCE` – Amount of thread divergence in wavefronts.

### 4.3. Example Output
```
Metric Name                 Value
--------------------------  -----
SQ_WAVES_ACTIVE             128
SQ_WAVES_EFFICIENCY         50%  <-- indicates divergence / underutilization
SIMD_DIVERGENCE             48%  <-- high divergence in kernel
```

### 4.4. Analyze Results

- **High divergence metrics** → consider restructuring kernel code.
- **Low wavefront efficiency** → may indicate branching, non-coalesced memory, or unbalanced workloads.
- Combine with **rocprofv3 timeline** for per-kernel and per-wavefront analysis.

---

## ⚡ 5. Summary

| Aspect | Details |
|--------|---------|
| Definition | Threads in a wavefront take different paths |
| Impact | Reduced occupancy, serialized execution, extra latency |
| Avoidance | Predication, branchless algorithms, thread grouping |
| Profiling | `rocprofv3` metrics like `SIMD_DIVERGENCE` and `SQ_WAVES_EFFICIENCY` |

**Key Takeaways:**

- Divergence can halve performance in extreme cases.  
- Minimizing divergence improves **GPU throughput** and **latency hiding**.  
- Profiling with `rocprofv3` identifies divergence hotspots for optimization.  

---

**References**
---

[← Kernel Optimization](kernel_optimization.md)
