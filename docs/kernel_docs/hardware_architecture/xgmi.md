# 🧭 How to Check AMD XGMI Link Status and Health

[← Hardware Architectures](hardware_architectures.md)

---
This guide explains how to **check the status and health of AMD XGMI (Infinity Fabric) links** between GPUs on AMD hardware. It covers command-line tools, sysfs paths, and best practices for link verification and diagnostics.

---

## 🧠 What is XGMI?

**XGMI (Infinity Fabric Interconnect)** is AMD’s high-speed GPU-to-GPU interconnect used in multi-GPU systems (e.g., MI200/MI300 series).  
It enables **high-bandwidth, low-latency communication** between GPUs in a single system, forming what AMD calls an **XGMI hive**.

Checking XGMI link health is critical for:
- ROCm multi-GPU workloads (RCCL, PyTorch DDP, etc.)
- Cluster communication reliability
- Debugging bandwidth or synchronization bottlenecks

---

## 🧰 Prerequisites

1. **Install ROCm and AMD SMI Tools**
   ```bash
   sudo apt install rocm-smi
   ```
   or on RHEL/CentOS:
   ```bash
   sudo dnf install rocm-smi
   ```

2. **Ensure the AMDGPU driver is active**  
   ```bash
   lsmod | grep amdgpu
   ```

3. **Run with root or elevated privileges** for full access to GPU telemetry.

4. **Verify your GPU supports XGMI** (Instinct-series GPUs or those with Infinity Fabric links).

---

## ⚙️ Step 1 — Using `amd-smi` CLI (ROCm 6.0+)

### Check GPU and XGMI Information
```bash
amd-smi
```

This provides an overview of GPU utilization, temperature, and memory. To specifically check **XGMI link status**, use:

```bash
amd-smi xgmi
```

This command displays the **state of XGMI links** between GPUs in the system, showing which GPUs are interconnected and if the links are active.

If supported, you can also view detailed **XGMI metrics**, including error counters and bandwidth usage.

---

## 🧩 Step 2 — Using `rocm-smi` (Legacy CLI)

In older ROCm versions or distributions, the tool is called `rocm-smi`.  
While it lacks dedicated XGMI subcommands, it provides topology and connectivity insights.

### View GPU Topology and XGMI Hive Membership

```bash
rocm-smi --showtopo
```

This displays the **topology map** for all detected GPUs, including:
- XGMI-connected peers
- NUMA distances
- PCIe relationships

**Example output:**
```
GPU[0]	: XGMI connection to GPU[1]
GPU[1]	: XGMI connection to GPU[0]
```

📊 **Weight Between Two GPUs**

|          | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 |
| :------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **GPU0** |   0  |  15  |  15  |  15  |  15  |  15  |  15  |  15  |
| **GPU1** |  15  |   0  |  15  |  15  |  15  |  15  |  15  |  15  |
| **GPU2** |  15  |  15  |   0  |  15  |  15  |  15  |  15  |  15  |
| **GPU3** |  15  |  15  |  15  |   0  |  15  |  15  |  15  |  15  |
| **GPU4** |  15  |  15  |  15  |  15  |   0  |  15  |  15  |  15  |
| **GPU5** |  15  |  15  |  15  |  15  |  15  |   0  |  15  |  15  |
| **GPU6** |  15  |  15  |  15  |  15  |  15  |  15  |   0  |  15  |
| **GPU7** |  15  |  15  |  15  |  15  |  15  |  15  |  15  |   0  |

🧮 **Hops Between Two GPUs**
|          | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 |
| :------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **GPU0** |   0  |   1  |   1  |   1  |   1  |   1  |   1  |   1  |
| **GPU1** |   1  |   0  |   1  |   1  |   1  |   1  |   1  |   1  |
| **GPU2** |   1  |   1  |   0  |   1  |   1  |   1  |   1  |   1  |
| **GPU3** |   1  |   1  |   1  |   0  |   1  |   1  |   1  |   1  |
| **GPU4** |   1  |   1  |   1  |   1  |   0  |   1  |   1  |   1  |
| **GPU5** |   1  |   1  |   1  |   1  |   1  |   0  |   1  |   1  |
| **GPU6** |   1  |   1  |   1  |   1  |   1  |   1  |   0  |   1  |
| **GPU7** |   1  |   1  |   1  |   1  |   1  |   1  |   1  |   0  |


🔗 **Link Type Between Two GPUs**

|          | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 |
| :------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **GPU0** |   0  | XGMI | XGMI | XGMI | XGMI | XGMI | XGMI | XGMI |
| **GPU1** | XGMI |   0  | XGMI | XGMI | XGMI | XGMI | XGMI | XGMI |
| **GPU2** | XGMI | XGMI |   0  | XGMI | XGMI | XGMI | XGMI | XGMI |
| **GPU3** | XGMI | XGMI | XGMI |   0  | XGMI | XGMI | XGMI | XGMI |
| **GPU4** | XGMI | XGMI | XGMI | XGMI |   0  | XGMI | XGMI | XGMI |
| **GPU5** | XGMI | XGMI | XGMI | XGMI | XGMI |   0  | XGMI | XGMI |
| **GPU6** | XGMI | XGMI | XGMI | XGMI | XGMI | XGMI |   0  | XGMI |
| **GPU7** | XGMI | XGMI | XGMI | XGMI | XGMI | XGMI | XGMI |   0  |

🧠 **NUMA Nodes**

| GPU    | NUMA Node | NUMA Affinity |
| :----- | :-------: | :-----------: |
| GPU[0] |     0     |       0       |
| GPU[1] |     0     |       0       |
| GPU[2] |     0     |       0       |
| GPU[3] |     0     |       0       |
| GPU[4] |     1     |       1       |
| GPU[5] |     1     |       1       |
| GPU[6] |     1     |       1       |
| GPU[7] |     1     |       1       |



This confirms which GPUs are linked via XGMI fabric.  
If GPUs are not connected, XGMI may be disabled in BIOS or not supported on your SKU.

---

## 🧮 Step 3 — Checking Link Errors with `amdsmi` API

For advanced diagnostics or automated scripts, use the **AMDSMI library**:

```c
amdsmi_get_gpu_xgmi_link_status(handle, &link_status);
amdsmi_gpu_xgmi_error_status(handle, &err_status);
amdsmi_reset_gpu_xgmi_error(handle);
```

These APIs can programmatically query link states and detect CRC or retry errors on XGMI connections.

---

## 🧪 Example Diagnostic Session

```bash
# Basic GPU info
amd-smi

# Check XGMI link status
amd-smi xgmi 

# Check topology and connectivity
rocm-smi --showtopo
```

---

## ⚠️ Troubleshooting Tips

- **`amd-smi xgmi` not supported?** → Upgrade to ROCm 6.0+.
- **Links missing in topology?** → Ensure XGMI is enabled in BIOS and GPUs share the same hive.
- **Low bandwidth or high errors?** → Run `amdsmi_gpu_xgmi_error_status()` or check `dmesg` for XGMI-related driver logs.
- **Virtualized environments** (e.g., SR-IOV) may mask XGMI details.

---

## ✅ Summary

| Method | Tool | Purpose |
|---------|------|----------|
| `amd-smi xgmi --link-status` | Modern CLI | Check XGMI link state |
| `rocm-smi --showtopo` | Legacy CLI | Show GPU interconnect topology |
| `amdsmi` API | Programmatic | Query and reset XGMI error counters |

---

**In summary:**  
Use `amd-smi xgmi` for direct link state, `rocm-smi --showtopo` for GPU topology visualization and `amdsmi` APIs for full diagnostic coverage of AMD’s XGMI interconnect.

---

## 📚 References

[Description of XGMI](https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html)

[← Hardware Architectures](hardware_architectures.md)
