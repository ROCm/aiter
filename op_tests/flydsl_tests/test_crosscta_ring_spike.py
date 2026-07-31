import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")


@pytest.mark.parametrize(
    "G,S,P_BLK,C_BLK",
    [
        (1, 1, 1, 1),      # degenerate: single task, single slot
        (64, 4, 1, 1),     # 1P/1C, 16 wraparound laps
        (256, 8, 2, 4),    # MPMC, S == 2*C_BLK, wraparound
        (300, 8, 4, 8),    # S == C_BLK boundary (tightest non-deadlock)
        (4096, 16, 3, 6),  # stress: many laps, asymmetric P/C
        (8192, 12, 6, 12), # stress: tight S==C_BLK, deep wraparound
    ],
)
def test_crosscta_ring_delivers_each_task_once_byte_exact(G, S, P_BLK, C_BLK):
    from aiter.ops.flydsl.kernels.crosscta_ring_spike_gfx1250 import (
        run_crosscta_ring_spike,
    )

    consumed = run_crosscta_ring_spike(G=G, S=S, P_BLK=P_BLK, C_BLK=C_BLK)
    assert consumed.shape[0] == G
    expected = torch.arange(G, device=consumed.device, dtype=torch.int32) * 7 + 13
    # each task consumed exactly once, payload byte-exact after cross-CTA hand-off
    torch.testing.assert_close(consumed.cpu(), expected.cpu(), atol=0, rtol=0)
