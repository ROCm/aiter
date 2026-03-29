import numpy as np
import torch

def preshuffle_xdl4(packed, N, K, k_warp_tile=32):
    K_half = K // 2
    NLane = 16; KLane = 4
    KPerIter = k_warp_tile // 2
    KIter = K_half // KPerIter
    assert N % NLane == 0 and K_half % KPerIter == 0 and KIter % 4 == 0

    n_idx = np.arange(N); k_idx = np.arange(K_half)
    nn, kk = np.meshgrid(n_idx, k_idx, indexing="ij")
    n0 = nn // NLane; n1 = nn % NLane
    kIter = kk // KPerIter; k_within = kk % KPerIter
    k_group = k_within // 4; byte_in_group = k_within % 4
    kIter_pack = kIter // 4; kIter_in_pack = kIter % 4

    out_idx = (n0 * (KIter // 4) * KLane * NLane * 16
             + kIter_pack * KLane * NLane * 16
             + k_group * NLane * 16
             + n1 * 16
             + kIter_in_pack * 4
             + byte_in_group)

    output = np.empty(N * K_half, dtype=np.uint8)
    output[out_idx.ravel()] = packed.ravel()
    return output

def preshuffle_xdl4_torch(packed, N, K):
    device = packed.device
    return torch.from_numpy(preshuffle_xdl4(packed.cpu().numpy(), N, K)).to(device)
