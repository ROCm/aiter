# Qwen3.5 / 3.6 verifier monokernel (gfx1250, FlyDSL) — work in progress

Research code for fusing a Qwen3.6-35B-A3B decoder layer's small kernels into one persistent launch on gfx1250
(256 CTAs x 256 threads, cluster of 16, 320 KB of LDS a CTA). Not wired into aiter's op registry; nothing here is
imported by the package. The scripts expect FlyDSL 0.4.0 and the serving image's layout (`/data/<checkpoint>`).

## What is here

| File | What it is |
|---|---|
| `vmoe2.py` | The verifier's sparse MoE block as one dataflow kernel: router, top-8, MXFP4 routed experts (A4W4, as ATOM serves them), bf16 shared expert, gated add. 1 to 16 rows. |
| `vmoe_atom.py` | ATOM-facing wrapper: packs a layer's weights into the kernel's layouts, owns the scratch buffers, `run(layer, x)`. |
| `vroute.py`, `bench_vroute.py` | Routing only (router GEMM, rank, publish, consumers), built from `vmoe2`'s own functions, with full timestamps and a torch check of the selection every wave holds. Use it to iterate the handoff. |
| `vgdn1.py`, `bench_vgdn1.py` | GDN mixer step 1: fused in_proj (N 12352, K 2048) and out_proj (N 2048, K 4096), MXFP8 1x32 on both operands, with a stand-in for the verify. Checked against torch. |
| `check_vmoe_atom.py` | Numerics of `vmoe2` against a torch model of ATOM's formats. Run this after every change. |
| `cold_vmoe_atom.py` | 40 layers back to back in one captured graph, new inputs every replay; times a launch and flags garbage rows. `REALX=1` uses real served rows. |
| `coldst_vmoe_atom.py`, `coldrole_vmoe_atom.py` | The kernel's internal timestamps, overall and by CTA role. |
| `mbox.py`, `stcost.py` | Micro-benches: handoff latency (counter vs tagged word vs plain 16-byte store), cost of issuing stores. |
| `hxdev.py`, `post.py`, `head.py`, `stage.py`, `vmoe.py`, `draft_mono.py`, `pull.py` | Shared helpers (raw addresses, scoped loads / stores, TDM, WMMA wrappers) and the earlier drafter kernels they came from. |
| `atom_hook/qwen3_5_vmoe.py` | The ATOM-side hook (`ATOM_VMOE=1`), with shadow / mix diagnostic modes. `atom_hook/qwen3_next.diff` is the uncommitted edit of ATOM's `qwen3_next.py` it needs (the diff also carries the drafter-kernel edits of the same file). |
| `tools/` | Run scripts and report scripts used for the serving checks. |

## Running

```
python check_vmoe_atom.py                       # numerics, several row counts
REALX=1 R=16 REPS=300 python cold_vmoe_atom.py  # time a launch on real rows, 40 layers
REALX=1 R=16 python coldrole_vmoe_atom.py       # timeline by CTA role
python bench_vroute.py                          # routing chain with timestamps
STAMPS=1 ALLOW_STAMPS=1 python bench_vgdn1.py   # GDN projections
```

`vmoe2.py` reads its options from the environment at import; the ones that matter:

| Flag (default) | Meaning |
|---|---|
| `S128=1` | Down outputs as two plain 16-byte stores a tile, not four device-scope 8-byte ones. |
| `BANK=1` | Two counter banks selected by the launch tag's parity; the kernel zeroes the next launch's bank, so no counter-zero kernel runs before it. |
| `XB2=1` | Split gate/up units: the partner's partial sums are read before a second CTA barrier. Do not turn off. |
| `EARLY=0` | An owner CTA starts its expert's units when any token's top-8 names it. Helps only at 16 rows. |
| `MB=0` | Routing handoffs as tagged 64-bit words. Correct but slower, and it stalls with timestamps on. |
| `RT3=0` | Routing with packed stores and one collector wave a CTA. Correct, not faster. |
| `A4=1`, `SHB=1` | ATOM's formats: MXFP4 activations for the routed experts, bf16 shared expert. |

## Rules learned (read before changing the kernel)

- Check numerics with `check_vmoe_atom.py`. `cold_vmoe_atom.py` only flags huge values.
- Time on real rows cycling all 40 layers. One layer with synthetic inputs selects few experts and flatters the kernel.
- LDS shared between waves needs a barrier on both sides of the read (the `XB2` bug: a partner's next weight load landed
  on the partial sums).
- One wave a CTA polls a counter, with a sleep; the others park at the CTA barrier. Many polling lanes starve other
  CTAs' loads, and can starve the very producer they wait for.
- A store instruction costs the wave time to issue; the store wait is cheap. Pack data into few, wide stores.
- Device-scope (`scope="agent"`) loads are serial: a dozen a lane from every wave costs microseconds. Plain vector loads
  of the same data do not.
- An atomic add blocks its wave until the old value returns. Keep it off the critical path.
- Kernel outputs another kernel reads: plain stores, waited for. Every wave drains its counters before it exits.
- Use TDM for bulk loads and batch a group's LDS operand reads behind a block boundary.

## Status

The MoE kernel serves correctly inside ATOM and is at about the same time as ATOM's own MoE chain; it is not yet a win
end to end. The GDN projections are correct but miss their timing gates; the verify is not ported. The intended next step
is a chained kernel (GDN stages + MoE, several GDN layers a launch) in which every stage's fixed weights are loaded one
stage ahead.
