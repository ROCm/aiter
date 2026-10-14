# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402, F401
from .coherent_memory import coherent_load

"""Owned workspace and explicit argument mapping for complete K3 MoE tests."""
import torch
from hip_ipc import DevicePointer


def workspace(batch, world):
    specs = {
        "Scores": ((batch, 896), torch.float32),
        "Latent": ((batch, 3584), torch.bfloat16),
        "GU": ((batch, 12288 // world), torch.bfloat16),
        "IDs": ((batch, 16), torch.int32),
        "Weights": ((batch, 16), torch.float32),
        "SharedMid": ((batch, 6144 // world), torch.bfloat16),
        "RouterReady": ((batch, 56), torch.uint64),
        "LatentReady": ((batch, 56), torch.uint64),
        "SharedReady": ((batch, 192 // world), torch.uint64),
        "RouteReady": ((batch,), torch.uint64),
        "SharedMidReady": ((batch,), torch.uint64),
        "Mid": ((batch, 16, 3072 // world), torch.bfloat16),
        "MidReady": ((batch, 16, 192 // world), torch.uint64),
        "Routed": ((batch, 3584), torch.bfloat16),
        "RoutedReady": ((batch, 224), torch.uint64),
        "Normed": ((batch, 3584), torch.bfloat16),
        "NormReady": ((batch,), torch.uint64),
        "Out": ((batch, 7168), torch.bfloat16),
        "FinalPrefix": ((batch, 7168), torch.bfloat16),
        "Epochs": ((256,), torch.int32),
        "Errors": ((256 * 512,), torch.int32),
    }
    return {
        k: torch.zeros(shape, dtype=dtype, device="cuda")
        for k, (shape, dtype) in specs.items()
    }


def arguments(x, weights, state, down_ipc, final_ipc, rank, world):
    params = dict(
        weights,
        **state,
        X=x,
        PeersDown=down_ipc.addresses,
        LocalDown=DevicePointer(down_ipc.ptr.value),
        PeersFinal=final_ipc.addresses,
        LocalFinal=DevicePointer(final_ipc.ptr.value)
    )
    names = "X Router Bias Down Shared Scores Latent GU IDs Weights SharedMid\n               RouterReady LatentReady SharedReady RouteReady SharedMidReady\n               WUp SUp Mid MidReady WDown SDown Routed RoutedReady NormWeight\n               Normed NormReady SharedDown LatentUp PeersDown LocalDown\n               PeersFinal LocalFinal Out Epochs Errors".split()
    return tuple((params[n] for n in names)) + (x.shape[0], rank)
