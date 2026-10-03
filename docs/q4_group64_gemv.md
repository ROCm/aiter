# Experimental Q4 group-64 GEMV (gfx1201)

The public API `q4_group64_gemv(x, packed_weight)` is a low-level,
experimental operator for inference code that already owns weights in its
prepacked format. It multiplies one FP32 activation vector by one signed-INT4
matrix with one FP16 scale per row and group of 64 columns:

```text
x:             contiguous FP32  [K]
packed_weight: contiguous uint8  [N / 32, K / 64, 1088]
result:        contiguous FP32  [N]
```

`N` must be positive and divisible by 32, and `K` must be positive and
divisible by 64. The input tensors must be on the same HIP device. `x` must be
naturally aligned for FP32 (4 bytes), and the packed tensor must be at least
2-byte aligned for its FP16 scales; the returned output is an AITER-allocated,
naturally aligned FP32 tensor. The implementation currently supports `gfx1201`
only. Set `AITER_ENABLE_EXPERIMENTAL=1` while calling the operator (and before an
all-ops build). Both the Python wrapper and the directly callable C++ entry
check this opt-in on every call. Removing or setting the variable to zero
immediately disables an already-loaded module.

## Packed tile format

For row tile `rt` and column group `g`, let
`tile = packed_weight[rt, g]`. Each tile describes 32 rows by 64 columns and
contains exactly 1088 bytes:

| Byte offset | Size | Meaning |
| --- | ---: | --- |
| `0 + 2*r` | 2 bytes | Little-endian FP16 scale for row `r`, where `0 <= r < 32` |
| `64 + 32*p + r` | 1 byte | Two signed INT4 values for pair `p` and row `r`, where `0 <= p < 32` |

For `b = tile[64 + 32*p + r]`, decode a nibble with
`signed4(v) = v if v < 8 else v - 16`. Then:

```text
q[32*rt + r, 64*g + 2*p    ] = signed4(b & 0x0f)
q[32*rt + r, 64*g + 2*p + 1] = signed4(b >> 4)
```

Thus values are pair-major and row-interleaved: the low nibble is the even
column and the high nibble is the following odd column. The operator computes

```text
y[n] = sum_g scale[n, g] * sum_j=0..63 q[n, 64*g + j] * x[64*g + j]
```

## Producing and loading weights

Quantization and packing are intentionally outside the runtime operator. An
integration is responsible for the following offline/loader flow:

1. Quantize each matrix to values in `[-8, 7]` and produce one scale per
   `(row, 64-column group)`. Scale selection is a quantizer policy, not part of
   this operator.
2. Pack those values and FP16-rounded scales into the byte layout above.
   [`op_tests/q4_group64_reference.py`](../op_tests/q4_group64_reference.py)
   contains the CPU reference `pack_group64` implementation shared by the
   correctness test and benchmark. It is a conformance reference for an
   offline exporter, not a public runtime packing API.
3. Store the packed bytes in a sidecar/checkpoint together with at least its
   format version, `N`, and `K`. At model load, validate that metadata and move
   the tensor to the target GPU as contiguous `torch.uint8` storage.
4. Keep the packed GPU tensor with the layer and call the operator for each
   FP32 activation vector. Do not quantize or repack weights per token.

A minimal source-tree example is:

```bash
export AITER_ENABLE_EXPERIMENTAL=1
python - <<'PY'
import torch

from aiter import q4_group64_gemv
from op_tests.q4_group64_reference import pack_group64

N, K = 32, 64
q = torch.randint(-8, 8, (N, K), dtype=torch.int8)  # offline quantizer output
scales = torch.full((N, K // 64), 0.02, dtype=torch.float32)
packed_cpu = pack_group64(q, scales)

# A real integration stores packed_cpu plus format/N/K metadata in a sidecar,
# then performs this transfer once while loading the layer.
packed_weight = packed_cpu.contiguous().to("cuda")
x = torch.randn(K, dtype=torch.float32, device="cuda")
y = q4_group64_gemv(x, packed_weight)
print(y.shape)
PY
```

There is currently no vLLM call site or AITER model-level call site for this
format. The present consumer is direct operator integration and benchmarking;
a framework adapter must supply the offline exporter and sidecar loader before
an end-to-end model can use it. Quantizer and packer time is therefore not
included in the kernel benchmark and is not on its inference hot path.

## Dispatch and benchmarking

The public API has no mapping knob. `auto` selects tuned mappings only when the
runtime identity matches `gfx1201`, PCI chip ID `0x7550`, and 32 HIP-reported
multiprocessors (WGP reporting on this stack). The runtime name must be either
exactly `AMD Radeon RX 9070 XT` or empty. The empty-name case is a narrowly
scoped ROCm 7.2 compatibility path because that runtime leaves both
`hipDeviceProp_t.name` and PyTorch's device name blank on this GPU; the numeric
chip and multiprocessor checks remain mandatory. Any other nonempty name fails
closed. Identity queries run after selecting the tensor's HIP device and are
cached by device id. Other `gfx1201` identities and every unseen `(N, K)` shape
use the conservative `old` mapping; non-`gfx1201` devices are rejected. Explicit
private benchmark mappings remain available for controlled ablation and are not
changed by the SKU guard. Dispatch is based on device identity and shape, never
on model name.

Run the measured-shape sweep on a `gfx1201` GPU with:

```bash
AITER_ENABLE_EXPERIMENTAL=1 python \
  op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py \
  --sweep --mappings old auto selected \
  --cache rotating --rotate 0 \
  --warmup 100 --samples 30 --timing both \
  --calibration-iterations 100 --target-sample-ms 100 \
  -o q4_group64_summary.csv --json q4_group64_raw.json
```

The sweep requires PCI chip ID `0x7550` via
`aiter.jit.utils.chip_info._get_pci_chip_id(0)`, 32 HIP-reported
multiprocessors, and either an exact `AMD Radeon RX 9070 XT` name or the blank
ROCm 7.2 compatibility case. It records the actual arch, chip ID,
multiprocessor count, name, and fallback status in the raw configuration. This
prevents a different `gfx1201` SKU from producing results labeled as the tuned
14-shape protocol.

The existing AITER helper hard-codes HIP attribute `10019`; in the ROCm 7.2
container that query returns the unrelated value `0x100`. Both the container's
ROCm 7.2 headers and the host ROCm 7.14 headers compile
`hipDeviceAttributePciChipId` as `10020`. If the helper returns a value outside
the benchmark's plausible AMD PCI-ID range, the benchmark alone retries
`libamdhip64` with attribute `10020` and still requires `0x7550`. Raw output
records the helper value, attribute number, fallback value, and whether the
fallback was used. Runtime dispatch does not use this workaround: its C++ guard
uses the HIP symbolic enum.

The benchmark treats synchronized host wall-clock latency as the primary public
API metric and records HIP-event elapsed time as a device-timeline supplement.
For `auto`, it calls the public `q4_group64_gemv` function object directly.
`old`, `selected`, and other explicit mappings are allocation-equivalent private
controls: they call `_q4_group64_gemv(..., out=None)` and therefore perform the
same output allocation, validation, runtime gate, and C++ entry work. They are
not public API call paths. Every raw row records this distinction in
`call_path`, and every sample records both `wall_us` and `event_us`.

The benchmark reports two batching boundaries:

- `integration` synchronizes and measures one call, including output allocation.
- `batched` synchronizes and measures calibrated repeated calls, including one
  output allocation per call, and divides by the call count. It amortizes the
  first submission gap, but can still contain host queue starvation.

Both boundaries report host wall and HIP-event views. The correctness check also
records maximum absolute error and relative L2 error for every requested mapping.

### Roofline context

Each packed tile stores 2,048 signed-INT4 weights in 1,088 bytes (1,024 value
bytes plus 64 FP16-scale bytes). Counting one multiply and one add per weight,
the weight-only arithmetic intensity is therefore 3.765 FLOP/byte
(`4096 / 1088`). Including the benchmark's activation read and output write gives
`3.702-3.756 FLOP/byte` over the 14 measured shapes. This is far below the RX
9070 XT ridge point: AMD specifies up to 48.7 FP32 TFLOP/s and 640 GB/s of
memory bandwidth, or about 76 FLOP/byte. The corresponding DRAM roof for this
operator is approximately 2.41 TFLOP/s before activation and output traffic.

For the calibrated batched public-entry boundary, `auto` reaches
58.9-696.7 GB/s of timing-derived logical bandwidth (498.8 GB/s median). This
quantity is packed bytes plus activation and output bytes divided by
synchronized host-wall time; it is cache-sensitive and is not a hardware DRAM
counter. Values above the nominal 640 GB/s can therefore occur and must not be
reported as greater-than-100% physical DRAM utilization. No physical bandwidth
counter was collected, so this benchmark makes no measured percent-of-peak
claim. RX 9070 XT specifications: https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html

A strict kernel-event result requires a native direct-launch harness with HIP
events around a batch of kernel launches, divided by the launch count. Such a
result excludes Python, public input validation, dispatch-cache lookup, and
pybind overhead and must be labeled separately from the command above.
