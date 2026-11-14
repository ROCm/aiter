# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM's fused gating entry: same slots, a state fork, and speculative verify."""

import pytest
import torch

from aiter.ops.triton.attention.kda import fused_recurrent_kda
from aiter.ops.triton.attention.kda_atom import fused_sigmoid_gating_delta_rule_update
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() not in ("gfx1250", "gfx1250-strict"), reason="gfx1250 only"
)

H = 12
K = 128


def _inputs(tokens, slots):
    qkv = {
        name: torch.randn((1, tokens, H, K), dtype=torch.bfloat16, device="cuda")
        for name in ("q", "k", "v", "a")
    }
    qkv["b"] = torch.randn((1, tokens, H), dtype=torch.bfloat16, device="cuda")
    state = torch.randn((slots, H, K, K), dtype=torch.float32, device="cuda")
    return qkv | {
        "A_log": torch.randn((H,), dtype=torch.float32, device="cuda") * 0.1,
        "dt_bias": torch.randn((H, K), dtype=torch.float32, device="cuda") * 0.1,
        "initial_state": state,
        "o": torch.full((tokens, H, K), 7, dtype=torch.bfloat16, device="cuda"),
    }


def _call(inputs, indices, indices_in, cu, accepted=None):
    args = dict(inputs)
    out = args.pop("o")
    return fused_sigmoid_gating_delta_rule_update(
        q=args.pop("q"),
        k=args.pop("k"),
        v=args.pop("v"),
        a=args.pop("a"),
        b=args.pop("b"),
        o=out,
        initial_state=args.pop("initial_state"),
        cu_seqlens=cu,
        ssm_state_indices=indices,
        ssm_state_indices_in=indices_in,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
        **args,
    )


def _recurrence(state, token, A_log, dt_bias):
    """One KDA step. ``state`` is [H, V, K] and is not written."""
    q, k, v, g, beta = (token[name].float() for name in ("q", "k", "v", "g", "beta"))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (K**-0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    decay = (-5.0 * torch.sigmoid(A_log.exp()[:, None] * (g + dt_bias))).exp()
    updated = state.float() * decay[:, None, :]
    err = (v - (updated * k[:, None, :]).sum(-1)) * beta.sigmoid()[:, None]
    updated = updated + err[:, :, None] * k[:, None, :]
    out = (updated * q[:, None, :]).sum(-1)
    return updated, out


def _reference(inputs, indices, indices_in, cu, accepted, state, out):
    """ATOM's read-slot / write-slot recurrence, including a negative write."""
    sequences = cu.numel() - 1
    accepted = (
        torch.ones(sequences, dtype=torch.int32, device=cu.device)
        if accepted is None
        else accepted
    )
    read = indices if indices_in is None else indices_in
    for seq in range(sequences):
        bos = int(cu[seq])
        length = int(cu[seq + 1]) - bos
        if length <= 0:
            continue
        if read.ndim == 1:
            origin = int(read[seq])
        else:
            origin = int(read[seq, int(accepted[seq]) - 1])
        if origin < 0:
            continue
        current = state[origin].clone()
        for step in range(length):
            token = {
                "q": inputs["q"][0, bos + step],
                "k": inputs["k"][0, bos + step],
                "v": inputs["v"][0, bos + step],
                "g": inputs["a"][0, bos + step],
                "beta": inputs["b"][0, bos + step],
            }
            current, produced = _recurrence(
                current, token, inputs["A_log"], inputs["dt_bias"]
            )
            out[bos + step] = produced.to(out.dtype)
            dest = int(indices[seq] if indices.ndim == 1 else indices[seq, step])
            if dest >= 0:
                state[dest] = current


def _assert_close(out, state, expected_out, expected_state):
    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=1e-4, rtol=1e-4)


def test_same_slots_match_single_index_kernel():
    torch.manual_seed(0)
    inputs = _inputs(4, 6)
    cu = torch.arange(5, dtype=torch.int32, device="cuda")
    indices = torch.tensor([1, 0, 4, 2], dtype=torch.int32, device="cuda")
    mirrored = indices.clone()
    original = inputs["initial_state"].clone()
    expected_state = original.clone()
    expected_out = inputs["o"].clone()
    _reference(inputs, indices, mirrored, cu, None, expected_state, expected_out)

    out, state = _call(inputs, indices, mirrored, cu)
    _assert_close(out, state, expected_out, expected_state)

    direct_out = inputs["o"].new_empty((1, 4, H, K))
    direct_state = original.clone()
    fused_out, fused_state = fused_recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=direct_state,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        out=direct_out,
    )
    torch.testing.assert_close(out, fused_out.squeeze(0), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, fused_state, atol=1e-4, rtol=1e-4)


def test_fork_reads_one_slot_and_writes_another():
    torch.manual_seed(1)
    inputs = _inputs(3, 8)
    cu = torch.arange(4, dtype=torch.int32, device="cuda")
    indices = torch.tensor([5, 0, 6], dtype=torch.int32, device="cuda")
    indices_in = torch.tensor([2, 7, 1], dtype=torch.int32, device="cuda")
    original = inputs["initial_state"].clone()
    expected_state = original.clone()
    expected_out = inputs["o"].clone()
    _reference(inputs, indices, indices_in, cu, None, expected_state, expected_out)
    out, state = _call(inputs, indices, indices_in, cu)
    _assert_close(out, state, expected_out, expected_state)
    kept = indices_in.long()
    torch.testing.assert_close(state[kept], original[kept])


def test_negative_read_skips_and_negative_write_keeps_output():
    torch.manual_seed(2)
    inputs = _inputs(4, 6)
    cu = torch.tensor([0, 1, 1, 2, 3], dtype=torch.int32, device="cuda")
    # seq0 writes slot 0, seq1 is empty, seq2 has a negative read, seq3 a
    # negative write. The prefilled 7s stay where the kernel must not store.
    indices = torch.tensor([0, 3, 1, -1], dtype=torch.int32, device="cuda")
    indices_in = torch.tensor([4, 3, -1, 2], dtype=torch.int32, device="cuda")
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["o"].clone()
    _reference(inputs, indices, indices_in, cu, None, expected_state, expected_out)
    out, state = _call(inputs, indices, indices_in, cu)
    _assert_close(out, state, expected_out, expected_state)
    untouched = inputs["o"].new_full((H, K), 7)
    assert torch.equal(out[1], untouched)


def test_speculative_verify_forks_from_the_accepted_column():
    torch.manual_seed(3)
    sequences, width = 3, 3
    inputs = _inputs(sequences * width, 16)
    cu = torch.arange(0, sequences * width + 1, width, dtype=torch.int32, device="cuda")
    accepted = torch.tensor([1, 3, 2], dtype=torch.int32, device="cuda")
    indices = torch.tensor(
        [[4, 5, -1], [6, 7, 8], [9, 10, 11]], dtype=torch.int32, device="cuda"
    )
    indices_in = torch.tensor(
        [[1, 2, 3], [12, 13, 14], [0, 15, 2]], dtype=torch.int32, device="cuda"
    )
    expected_state = inputs["initial_state"].clone()
    expected_out = inputs["o"].clone()
    _reference(
        inputs, indices, indices_in, cu, accepted, expected_state, expected_out
    )
    out, state = _call(inputs, indices, indices_in, cu, accepted)
    _assert_close(out, state, expected_out, expected_state)
    # Checkpoints named only by the read table stay put.
    torch.testing.assert_close(state[1], inputs["initial_state"][1])
    torch.testing.assert_close(state[14], inputs["initial_state"][14])
    torch.testing.assert_close(state[15], inputs["initial_state"][15])


def test_speculative_verify_without_fork_matches_single_index():
    torch.manual_seed(4)
    sequences, width = 2, 2
    inputs = _inputs(sequences * width, 8)
    cu = torch.arange(0, sequences * width + 1, width, dtype=torch.int32, device="cuda")
    accepted = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda")
    state = inputs["initial_state"].clone()
    out = torch.empty((1, sequences * width, H, K), dtype=torch.bfloat16, device="cuda")
    fused_out, fused_state = fused_recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=state,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        out=out,
    )
    atom_out, atom_state = _call(inputs, indices, None, cu, accepted)
    torch.testing.assert_close(atom_out, fused_out.squeeze(0), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(atom_state, fused_state, atol=1e-4, rtol=1e-4)


def test_cuda_graph_replay_uses_updated_slots():
    torch.manual_seed(5)
    inputs = _inputs(1, 6)
    cu = torch.arange(2, dtype=torch.int32, device="cuda")
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    indices_in = torch.tensor([2], dtype=torch.int32, device="cuda")
    _call(inputs, indices, indices_in, cu)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph):
            captured_out, _captured_state = _call(inputs, indices, indices_in, cu)
    torch.cuda.current_stream().wait_stream(stream)

    inputs["initial_state"].copy_(torch.randn_like(inputs["initial_state"]))
    indices.copy_(torch.tensor([4], dtype=torch.int32, device="cuda"))
    indices_in.copy_(torch.tensor([5], dtype=torch.int32, device="cuda"))
    baseline = inputs["initial_state"].clone()
    graph.replay()
    torch.cuda.synchronize()
    expected = baseline.clone()
    expected_out = torch.empty_like(inputs["o"])
    _reference(inputs, indices, indices_in, cu, None, expected, expected_out)
    _assert_close(captured_out, inputs["initial_state"], expected_out, expected)
    torch.testing.assert_close(inputs["initial_state"][5], baseline[5])


def test_rejected_signatures():
    inputs = _inputs(1, 2)
    cu = torch.arange(2, dtype=torch.int32, device="cuda")
    indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    common = dict(
        A_log=inputs["A_log"],
        a=inputs["a"],
        b=inputs["b"],
        dt_bias=inputs["dt_bias"],
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        initial_state=inputs["initial_state"],
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    with pytest.raises(NotImplementedError):
        fused_sigmoid_gating_delta_rule_update(
            **common, is_kda=False, lower_bound=-5.0
        )
    with pytest.raises(NotImplementedError):
        fused_sigmoid_gating_delta_rule_update(
            **common, is_kda=True, lower_bound=None
        )
    with pytest.raises(ValueError):
        fused_sigmoid_gating_delta_rule_update(
            **common,
            is_kda=True,
            lower_bound=-5.0,
            ssm_state_indices_in=indices[:0],
        )
