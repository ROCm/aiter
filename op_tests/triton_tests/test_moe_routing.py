import pytest
import torch
from aiter.ops.triton.moe_routing.routing import routing, routing_torch
from aiter.ops.triton.utils._triton.arch_info import get_arch


def assert_equal(ref, tri):
    if isinstance(ref, torch.Tensor):
        assert torch.all(ref == tri)
    else:
        assert ref == tri


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert (
        ref.shape == tri.shape
    ), f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(
        inf_mask_ref, inf_mask_tri
    ), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print(
            "%s maximum relative error = %s (threshold = %s)"
            % (description, max_err, maxtol)
        )
        print(
            "%s RMS relative error = %s (threshold = %s)"
            % (description, rms_err, rmstol)
        )

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print(
            "%d / %d mismatched elements (shape = %s) at coords %s"
            % (num_nonzero, rel_err.numel(), tuple(rel_err.shape), bad_idxs.tolist())
        )

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[tuple(bad_idxs)].cpu())
        print("tri values: ", tri[tuple(bad_idxs)].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


def init_data(n_tokens, n_expts_tot, dtype=torch.float16, device="cuda"):
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device)
    return logits


n_tokens = [4, 7, 8, 64, 255, 256, 371, 911, 1023, 1024, 4096, 8192]


@pytest.mark.parametrize("n_tokens", n_tokens)
@pytest.mark.parametrize(
    "n_expts_tot, n_expts_act", [(128, 4), (128, 32), (1500, 8), (256, 8), (8, 2)]
)
@pytest.mark.parametrize("use_expt_indx", [False, True])
@pytest.mark.parametrize("sm_first", [True, False])
def test_op(n_tokens, n_expts_tot, n_expts_act, sm_first, use_expt_indx):
    if get_arch() != "gfx950":
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")

    device = "cuda"
    torch.manual_seed(2)
    n_gates_raw = n_tokens * n_expts_act
    tri_logits = init_data(
        n_tokens, n_expts_tot, device=device, dtype=torch.float32
    ).detach()
    tri_logits[n_tokens:, :] = float("inf")  # should not be used
    ref_logits = tri_logits.clone().detach()

    if use_expt_indx:
        rand_idx = lambda: torch.randperm(n_expts_tot, device="cuda", dtype=torch.int64)
        tri_expt_indx = torch.stack([rand_idx()[:n_expts_act] for _ in range(n_tokens)])
        tri_expt_indx, _ = torch.sort(tri_expt_indx, dim=1)
        tri_expt_indx[n_tokens:] = -99999  # should not be used
        ref_expt_indx = tri_expt_indx[:n_tokens]
    else:
        tri_expt_indx = ref_expt_indx = None
    ref_routing_data, ref_gather, ref_scatter = routing_torch(
        ref_logits, n_expts_act, sm_first, ref_expt_indx
    )
    tri_routing_data, tri_gather, tri_scatter = routing(
        tri_logits, n_expts_act, sm_first, tri_expt_indx
    )

    def _assert_indx_equal(ref, tri):
        assert_equal(ref, tri[: len(ref)])
        assert torch.all(tri[len(ref) :] == -1)

    assert_close(
        ref_routing_data.gate_scal, tri_routing_data.gate_scal[:n_gates_raw], 2e-2, 4e-3
    )
    assert_equal(ref_routing_data.expt_hist, tri_routing_data.expt_hist)

    ref_expt_data = ref_routing_data.expt_data
    tri_expt_data = tri_routing_data.expt_data
    assert_equal(ref_expt_data.hist, tri_expt_data.hist)
    assert_equal(ref_expt_data.token_offs_raw, tri_expt_data.token_offs_raw)
    assert_equal(ref_expt_data.token_offs_pad, tri_expt_data.token_offs_pad)
    assert_equal(ref_expt_data.block_pid_map, tri_expt_data.block_pid_map)

    assert ref_routing_data.n_expts_tot == ref_routing_data.n_expts_tot
    assert ref_routing_data.n_expts_act == ref_routing_data.n_expts_act

    _assert_indx_equal(ref_gather, tri_gather)
    _assert_indx_equal(ref_scatter, tri_scatter)


def bench_routing():
    import triton.profiler as proton

    n_tokens = 8192
    n_expts_tot, n_expts_act = 128, 4
    tri_logits = init_data(n_tokens, n_expts_tot)
    proton.start("routing")
    proton.activate()
    for i in range(100):
        tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act)
    proton.finalize()
    try:
        import os

        os.system("proton-viewer -m time/ms routing.hatchet")
    except Exception:
        pass


if __name__ == "__main__":
    bench_routing()
