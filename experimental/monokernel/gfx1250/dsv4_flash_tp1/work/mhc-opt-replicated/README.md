# Flash C1: mHC through attention reduction

See REPORT.md for second-round results and DESIGN.md for ownership/synchronization.

Reproduce on an idle gfx1250 GPU with 256 CUs, in the existing container:

```sh
docker exec vigokhal_dsv4_tp4 bash -c 'cd /work/mhc-through-attention && bash run.sh'
```

The included fixture.pt, mhc-fixture.pt, and sink.pt make model checkpoint access
unnecessary for replay. run.sh overwrites result files: copy this directory first.
Python State.run supports unfused (17 kernels), three_regions (3), and fused (1).
The default context is 128, attention split-KV=4. Inputs are four BF16 residual
streams, shape[4,4096]. This is a standalone region benchmark, not ATOM serving.

Sources derive from attention-direct-kv and the prior all-WMMA mHC algorithm.
TDM now stages the mHC input and FP32 weight tiles; two BF16 weight components
preserve its arithmetic. Split and fused paths use identical learned GEMMs.
