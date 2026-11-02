# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Smoke test for the mori install that aiter's multi-GPU tests depend on.

Each case checks one piece of mori that aiter uses. A broken install then fails
here under the piece's name, instead of deep inside a MoE test:

  import      mori.{cco,shmem,ops,allocator} import, and the cco device bitcode
              carries the SDMA entry points that SdmaComm links
  shmem       shmem_torch_process_group_init, then every rank stores to and
              loads from every peer's symmetric buffer (MoriAll2AllManager,
              MegaMoE v2)
  ep          IntraNode dispatch + combine round trip (test_dispatch_combine,
              test_mori_all2all)
  cco_sdma    cco communicator + devComm with SDMA queues and no GDA, created
              as SdmaComm does (test_mega_moe_tp); no SDMA transfer is posted
  cco_window  a torch.symm_mem buffer registered as a cco external window;
              every peer's buffer is then read and written through the flat
              VA (test_comm_fused_moe)

Only cco_sdma sets MORI_ENABLE_SDMA, as aiter does, so the other cases run on
the P2P transport aiter uses for them.

Every multi-rank case runs in fresh processes with its own deadline, so a hang
is reported as one failed case, not a stuck job.
"""

import argparse
import os
import socket
import sys
import time
import traceback
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SDMA_SYMBOLS = ("cco_sdma_put_ptr_rsig__thread", "cco_sdma_signal_slot")
FLAT_VA_RANK_STRIDE = 1 << 32  # as comm_fused_moe_host


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_gfx125x() -> bool:
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx125")


def _view(ptr: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
    """A torch view (no copy) of device memory at ``ptr``."""

    buf = types.SimpleNamespace(
        __cuda_array_interface__={
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "<u1",
            "version": 3,
        }
    )
    t = torch.as_tensor(buf, device="cuda")
    assert t.data_ptr() == ptr, f"view of {ptr:#x} is a copy at {t.data_ptr():#x}"
    return t.view(dtype)


def check_import() -> None:
    import mori
    import mori.allocator
    import mori.cco
    import mori.ops
    import mori.shmem

    print(f"mori {mori.__version__} from {os.path.dirname(mori.__file__)}")
    for name in ("Communicator", "CCODevCommRequirements", "GDA_CONNECTION_NONE"):
        assert hasattr(mori.cco, name), f"mori.cco has no {name}"
    for name in (
        "get_unique_id",
        "init",
        "create_dev_comm",
        "register_external_window",
    ):
        assert hasattr(mori.cco.Communicator, name), f"Communicator has no {name}"
    assert hasattr(mori.ops, "EpDispatchCombineOp")
    assert hasattr(mori.shmem, "shmem_torch_process_group_init")

    from mori.cco.device.bitcode import find_cco_bitcode

    path = find_cco_bitcode(6)
    with open(path, "rb") as f:
        bc = f.read()
    missing = [s for s in SDMA_SYMBOLS if s.encode() not in bc]
    # the SDMA entry points exist only in a BUILD_CCO_SDMA=ON mori build
    assert not missing, f"{path} lacks {missing}; is mori built with BUILD_CCO_SDMA?"
    print(f"cco bitcode {path} has {', '.join(SDMA_SYMBOLS)}")


def _shmem_init(rank: int, world: int) -> None:
    import mori

    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("default")
    assert mori.shmem.shmem_mype() == rank and mori.shmem.shmem_npes() == world


def case_shmem(rank: int, world: int) -> None:
    from mori import shmem

    _shmem_init(rank, world)
    n = 4096
    buf = shmem.mori_shmem_create_tensor((world, n), torch.int32)
    buf.fill_(-1)
    torch.cuda.synchronize()
    shmem.shmem_barrier_all()
    # store: row `rank` of every peer's buffer
    for peer in range(world):
        shmem.symm_mori_shmem_tensor(buf, peer)[rank].fill_(rank)
    torch.cuda.synchronize()
    shmem.shmem_barrier_all()
    want = torch.arange(world, dtype=torch.int32, device="cuda")[:, None].expand(-1, n)
    assert torch.equal(buf, want), "peer stores did not land"
    # load: every peer's buffer
    for peer in range(world):
        got = shmem.symm_mori_shmem_tensor(buf, peer)
        assert torch.equal(got, want), f"load from rank {peer} differs"
    torch.cuda.synchronize()
    shmem.shmem_barrier_all()
    shmem.mori_shmem_free_tensor(buf)
    shmem.shmem_finalize()


def case_ep(rank: int, world: int) -> None:
    import mori

    _shmem_init(rank, world)
    tokens, hidden, experts_per_rank, topk = 64, 2048, 4, 4
    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=experts_per_rank,
        num_experts_per_token=topk,
        warp_num_per_block=16,
        block_num=80,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=world,
    )
    op = mori.ops.EpDispatchCombineOp(cfg)

    gen = torch.Generator().manual_seed(rank)
    x = torch.randn(tokens, hidden, generator=gen).to("cuda", torch.bfloat16)
    idx = torch.stack(
        [
            torch.randperm(world * experts_per_rank, generator=gen)[:topk]
            for _ in range(tokens)
        ]
    ).to("cuda", torch.int32)
    w = torch.rand(tokens, topk, generator=gen).cuda()

    out, out_w, _, _, recv = op.dispatch(x, w, None, idx)
    torch.cuda.synchronize()
    dist.barrier()
    # no expert compute: each token comes back once per distinct destination rank
    comb, _ = op.combine(out, out_w, idx)
    torch.cuda.synchronize()
    dist.barrier()
    pes = idx.long().cpu() // experts_per_rank
    copies = torch.tensor([p.unique().numel() for p in pes], device="cuda")
    want = x.float() * copies[:, None]
    got = comb[:tokens].float()
    assert torch.allclose(got, want, atol=1e-2, rtol=1e-2), (
        f"combine mismatch, max |diff| {(got - want).abs().max().item()}, "
        f"recv {recv[0].item()}"
    )
    del op
    dist.barrier()
    mori.shmem.shmem_finalize()


def _cco_init(rank: int, world: int, per_rank_vmm: int):
    from mori.cco import Communicator

    uid = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    return Communicator.init(world, rank, uid[0], per_rank_vmm=per_rank_vmm)


def case_cco_sdma(rank: int, world: int) -> None:
    # same setup as aiter.ops.communication.SdmaComm.commit (and its _mori_env)
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements

    comm = _cco_init(rank, world, 1 << 30)
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_NONE
    reqs.gda_signal_count = 0
    reqs.gda_counter_count = 0
    reqs.sdma_queue_count = 8
    dev_comm = comm.create_dev_comm(reqs)
    assert (dev_comm.rank, dev_comm.world_size) == (rank, world), dev_comm
    assert dev_comm.lsa_size == world, f"lsa_size {dev_comm.lsa_size}, want {world}"
    # torch must still work after cco init (cco used to leave a HIP error latched)
    assert torch.ones(16, device="cuda").sum().item() == 16
    comm.barrier()
    dev_comm.close()
    comm.destroy()


def case_cco_window(rank: int, world: int) -> None:
    import torch.distributed._symmetric_memory as symm_mem

    # as comm_fused_moe_host: allocate, then create the communicator, then register
    nbytes = 6 << 20
    buf = symm_mem.empty((nbytes,), dtype=torch.uint8, device="cuda")
    own = buf.view(torch.int32)
    own.fill_(rank)
    torch.cuda.synchronize()
    comm = _cco_init(rank, world, FLAT_VA_RANK_STRIDE)
    # one window per comm, as aiter: a 2nd can fail hipMemSetAccess on ROCm 7.0-7.2
    window = comm.register_external_window(buf.data_ptr(), nbytes)
    base = window.local_ptr - rank * FLAT_VA_RANK_STRIDE
    peers = [
        _view(base + peer * FLAT_VA_RANK_STRIDE, nbytes, torch.int32)
        for peer in range(world)
    ]
    torch.cuda.synchronize()
    comm.barrier()
    # load: every peer's buffer
    for peer, got in enumerate(peers):
        assert torch.all(got == peer), f"load from rank {peer} differs"
    torch.cuda.synchronize()
    comm.barrier()
    # store: slot `rank` of every peer's buffer
    slot = own.numel() // world
    for dst in peers:
        dst[rank * slot : (rank + 1) * slot].fill_(-1 - rank)
    torch.cuda.synchronize()
    comm.barrier()
    want = -1 - torch.arange(world, dtype=torch.int32, device="cuda")[:, None]
    got = own[: world * slot].view(world, slot)
    assert torch.equal(got, want.expand(-1, slot)), "peer stores did not land"
    comm.barrier()
    comm.destroy()


CASES = {
    "shmem": case_shmem,
    "ep": case_ep,
    "cco_sdma": case_cco_sdma,
    "cco_window": case_cco_window,
}


def _worker(rank: int, world: int, case: str, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", rank),
    )
    try:
        CASES[case](rank, world)
    finally:
        dist.destroy_process_group()


def run_case(case: str, world: int, timeout: float) -> str:
    ctx = mp.start_processes(
        _worker,
        args=(world, case, _free_port()),
        nprocs=world,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + timeout
    try:
        while not ctx.join(timeout=5):
            if time.monotonic() > deadline:
                for p in ctx.processes:
                    p.kill()
                for p in ctx.processes:
                    p.join(timeout=30)
                return f"TIMEOUT after {timeout:.0f}s"
    except mp.ProcessRaisedException as e:
        print(e, file=sys.stderr)
        return "FAIL"
    except mp.ProcessExitedException as e:
        print(e, file=sys.stderr)
        return "FAIL"
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--world-size",
        type=int,
        default=min(torch.cuda.device_count(), 8),
        help="ranks per multi-rank case (default: visible GPUs, at most 8)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["import", *CASES],
        default=["import", *CASES],
    )
    parser.add_argument("--timeout", type=float, default=300, help="seconds per case")
    args = parser.parse_args()

    results = {}
    for case in args.cases:
        print(f"==== mori smoke: {case}", flush=True)
        t0 = time.monotonic()
        if case == "import":
            try:
                check_import()
                status = "PASS"
            except Exception:  # noqa: BLE001 (reported as the case result)
                traceback.print_exc()
                status = "FAIL"
        elif args.world_size < 2:
            status = f"SKIP: needs 2 GPUs, have {args.world_size}"
        elif case in ("shmem", "ep") and _is_gfx125x():
            # MoriAll2AllManager skips shmem there; its EP is cco-backed instead
            status = f"SKIP: {case} runs on mori shmem, which gfx125x does not use"
        else:
            status = run_case(case, args.world_size, args.timeout)
        results[case] = (status, time.monotonic() - t0)

    print("==== mori smoke summary")
    for case, (status, secs) in results.items():
        print(f"  {case:<11} {status:<40} {secs:6.1f}s")
    failed = [c for c, (s, _) in results.items() if not s.startswith(("PASS", "SKIP"))]
    if failed:
        print(f"mori smoke FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
