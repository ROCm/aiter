# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Guard the compiler barrier between the TDM wait and shared-memory access."""

import re

import pytest


def _assert_tdm_wait_barriers(asm):
    instructions, labels = [], {}
    for line in asm.splitlines():
        line = line.split("//", 1)[0].split(";", 1)[0].strip()
        if re.fullmatch(r"[.\w$]+:", line):
            labels[line[:-1]] = len(instructions)
        elif re.match(r"(?:s_|v_|ds_|tensor_)\w+\b", line):
            parts = line.split(maxsplit=1)
            instructions.append((parts[0], parts[1] if len(parts) == 2 else ""))

    # PREFETCH=2 uses completion counts 1 and 0 for this row size. Larger waits
    # can throttle the hardware queue before a copy; they need no rendezvous.
    waits = {
        i: int(operands, 0)
        for i, (op, operands) in enumerate(instructions)
        if op == "s_wait_tensorcnt" and int(operands, 0) in (0, 1)
    }
    assert set(waits.values()) == {0, 1}, "Missing steady-state or tail tensor wait"
    guarded_refills = set()
    for wait in waits:
        pending, visited = [(wait + 1, False, False)], set()
        while pending:
            index, signaled, barrier = pending.pop()
            state = (index, signaled, barrier)
            if index >= len(instructions) or state in visited:
                continue
            visited.add(state)
            op, operands = instructions[index]
            if op == "s_barrier":
                barrier = True
            elif op == "s_barrier_signal" and int(operands, 0) in (-1, 65535):
                signaled = True
            elif op == "s_barrier_wait" and operands in ("0xffff", "65535", "-1"):
                barrier = barrier or signaled
            if op.startswith(("tensor_load_to_lds", "ds_read", "ds_load")):
                assert (
                    barrier
                ), f"{op} is reachable after tensor wait {wait} without s_barrier"
                if op.startswith("tensor_load_to_lds"):
                    guarded_refills.add(index)
                continue
            if op.startswith("s_endpgm"):
                continue
            if op == "s_branch" or op.startswith("s_cbranch_"):
                target = operands.strip()
                assert target in labels, f"Unknown branch target: {target}"
                pending.append((labels[target], signaled, barrier))
                if op == "s_branch":
                    continue
            pending.append((index + 1, signaled, barrier))
    assert guarded_refills, "No TDM refill was reached after a tensor wait"


@pytest.mark.parametrize("barrier", ["s_barrier", "s_nop 0"])
def test_tdm_asm_guard_checks_both_wait_paths(barrier):
    # An unrelated barrier and a safe branch must not hide the unsafe branch.
    asm = f"""
        s_barrier
        s_wait_tensorcnt 0x1
        s_cbranch_scc0 .Ltail
        {barrier}
        s_branch .Lrefill
    .Ltail:
        s_wait_tensorcnt 0x0
        s_barrier
    .Lrefill:
        tensor_load_to_lds
        ds_load_b128
        s_endpgm
    """
    if barrier == "s_barrier":
        _assert_tdm_wait_barriers(asm)
    else:
        with pytest.raises(AssertionError, match="without s_barrier"):
            _assert_tdm_wait_barriers(asm)


def test_tdm_asm_guard_rejects_a_barrier_after_refill():
    asm = """
        s_barrier
        s_wait_tensorcnt 0x1
        tensor_load_to_lds
        s_barrier
        ds_load_b128
        s_wait_tensorcnt 0x0
        s_endpgm
    """
    with pytest.raises(AssertionError, match="without s_barrier"):
        _assert_tdm_wait_barriers(asm)


@pytest.mark.parametrize(
    "barrier",
    [
        "s_barrier_signal -1\ns_barrier_wait 0xffff",
        "s_barrier_signal -1",
        "s_barrier_wait 0xffff",
    ],
)
def test_tdm_asm_guard_requires_the_complete_workgroup_barrier(barrier):
    asm = f"""
        s_wait_tensorcnt 0xa
        tensor_load_to_lds
        s_wait_tensorcnt 0x1
        {barrier}
        tensor_load_to_lds
        s_wait_tensorcnt 0x0
        s_barrier_signal -1
        s_barrier_wait 0xffff
        ds_load_b128
        s_endpgm
    """
    if "signal" in barrier and "wait" in barrier:
        _assert_tdm_wait_barriers(asm)
    else:
        with pytest.raises(AssertionError, match="without s_barrier"):
            _assert_tdm_wait_barriers(asm)


@pytest.mark.parametrize("quant", [False, True])
def test_attn_res_gate_tdm_refill_barrier(monkeypatch, quant):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("The compiled TDM kernel needs a gfx1250 GPU")

    from aiter.ops.triton.fusions import attn_res
    from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

    kernel = attn_res._GATE_TDM_KERNEL
    if kernel is None:
        pytest.skip("The compiled TDM kernel needs gfx1250")
    compiled = []
    run = kernel.run

    def record_run(*args, **kwargs):
        result = run(*args, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(kernel, "run", record_run)
    monkeypatch.setattr(attn_res, "_LAUNCH_CACHE_ENABLED", False)
    # Eight rows reuse all three LDS slots. The wrapper loads the real config.
    prefix = torch.ones((4, 7168), dtype=torch.bfloat16, device="cuda")
    block = torch.ones((4, 8, 7168), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((7168,), dtype=torch.bfloat16, device="cuda")
    attn_res.attn_res_gate(
        prefix,
        block,
        weight,
        add_hidden=prefix,
        add_hidden2=prefix,
        output_rms_weight=weight,
        out_quant_dtype=get_fp8_e4m3_dtype() if quant else None,
        backend="gluon",
    )
    torch.cuda.synchronize()
    assert len(compiled) == 1, "The public wrapper must launch the TDM kernel"
    _assert_tdm_wait_barriers(compiled[0].asm["amdgcn"])
