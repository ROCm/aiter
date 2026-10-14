# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402, F401
from .coherent_memory import coherent_load

"""Single-version KDA projection/TP sum/AttnRes adapter for live model inputs."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .attnres import attnres_chunk
from .projection import output_projection


def allocate_workspace(batch):
    import torch

    def zeros(shape, dtype):
        return torch.zeros(shape, device="cuda", dtype=dtype)

    return (
        zeros((batch, 7168), torch.bfloat16),
        zeros((batch, 7168), torch.bfloat16),
        zeros((batch, 7168), torch.uint8),
        zeros(7168, torch.uint8),
        zeros(batch * 4 * 19, torch.uint64),
        zeros(batch * 4, torch.uint64),
        zeros(256, torch.int32),
        zeros(256 * 512, torch.int32),
    )


@g.jit
def projection_attnres_live(
    X,
    W,
    Peers,
    Local,
    Delta,
    Attention,
    Prefix,
    Blocks,
    Norm,
    Query,
    OutNorm,
    Updated,
    Moe,
    Quant,
    Scales,
    Stats,
    Ready,
    Epochs,
    Errors,
    B: gl.constexpr,
    RANK: gl.constexpr,
    N_BLOCKS: gl.constexpr,
    BOUNDARY: gl.constexpr,
    BLOCK_STRIDE_M: gl.constexpr,
    WORLD: gl.constexpr = 8,
    WATCHDOG_TICKS: gl.constexpr = 0,
    DebugIPC=None,
    Epoch=None,
):
    pid = gl.program_id(0)
    if Epoch is None:
        epoch = gl.load(Epochs + pid) + 1
        gl.barrier()
    else:
        epoch = Epoch
    if pid < (B + 3) // 4 * 112:
        output_projection(
            X,
            W,
            Peers,
            Local,
            Delta,
            Epochs,
            Errors,
            B,
            RANK,
            Attention=Attention,
            Epoch=epoch,
            WORLD=WORLD,
            INPUT_DOUBLE_BUFFERED=False,
            WATCHDOG_TICKS=WATCHDOG_TICKS,
            DebugIPC=DebugIPC,
        )
    if pid < B * 4:
        attnres_chunk(
            Attention,
            Delta,
            Prefix,
            Blocks,
            Norm,
            Query,
            OutNorm,
            Updated,
            Moe,
            Quant,
            Scales,
            Stats,
            Ready,
            Errors,
            epoch,
            pid,
            B,
            N_BLOCKS,
            BOUNDARY,
            True,
            INPUT_DOUBLE_BUFFERED=False,
            BLOCK_STRIDE_M=BLOCK_STRIDE_M,
            WATCHDOG_TICKS=WATCHDOG_TICKS,
        )
    if Epoch is None:
        gl.atomic_store(Epochs + pid, epoch, sem="relaxed", scope="gpu")
