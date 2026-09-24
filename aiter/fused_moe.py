"""
Extended verification for the Relu2 gate_mode guard added to get_2stage_cfgs().

Goal: confirm the new `if gate_mode != GateMode.SEPARATED: raise NotImplementedError`
check (scoped inside `if activation == ActivationType.Relu2:`) does exactly two things:
  1. Blocks Relu2 + INTERLEAVE (the bug case) -- already confirmed once.
  2. Does NOT affect any other activation's INTERLEAVE handling (Swiglu, Situv2),
     since the new check is nested under `activation == ActivationType.Relu2` and
     cannot fire for other activations.
  3. Does NOT change guard *ordering* for Relu2: use_g1u1=True should still raise
     the pre-existing "gate-only" error (checked first in source order), not the
     new gate_mode error, when both conditions are true simultaneously.

Run directly inside the container:
    python3 experiment_relu2_gate_mode_gap_v2.py
"""

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode
import aiter.fused_moe as fused_moe_mod

fused_moe_mod.get_gfx = lambda: "gfx950"


def try_case(label, **kwargs):
    try:
        metadata = fused_moe_mod.get_2stage_cfgs(**kwargs)
        stage1_repr = getattr(metadata.stage1, "func", metadata.stage1)
        print(f"{label}: NO EXCEPTION -- stage1={stage1_repr}")
        return "ok"
    except NotImplementedError as e:
        print(f"{label}: NotImplementedError -- {e}")
        return str(e)
    except Exception as e:
        print(f"{label}: OTHER EXCEPTION ({type(e).__name__}): {e}")
        return f"OTHER:{type(e).__name__}"


base = dict(
    token=512,
    model_dim=2048,
    inter_dim=2048,
    expert=32,
    topk=4,
    dtype=dtypes.bf16,
    q_type=QuantType.per_1x32,
    doweight_stage1=False,
    hidden_pad=0,
    intermediate_pad=0,
    is_shuffled=True,
    opus_weights_shuffled=True,
)

print("=== 1) Regression check: Relu2 + SEPARATED (must still succeed) ===")
try_case(
    "Relu2+SEPARATED",
    gate_mode=GateMode.SEPARATED.value,
    use_g1u1=False,
    activation=ActivationType.Relu2,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.fp4x2,
    **base,
)

print()
print("=== 2) Bug case: Relu2 + INTERLEAVE (must now raise) ===")
try_case(
    "Relu2+INTERLEAVE",
    gate_mode=GateMode.INTERLEAVE.value,
    use_g1u1=False,
    activation=ActivationType.Relu2,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.fp4x2,
    **base,
)

print()
print("=== 3) Guard ordering: Relu2 + INTERLEAVE + use_g1u1=True ===")
print("    (use_g1u1 check comes first in source; expect the g1u1 message, not gate_mode)")
result = try_case(
    "Relu2+INTERLEAVE+g1u1",
    gate_mode=GateMode.INTERLEAVE.value,
    use_g1u1=True,
    activation=ActivationType.Relu2,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.fp4x2,
    **base,
)
if "gate-only (non-gated)" in result:
    print("    -> PASS: g1u1 guard fired first, as expected")
elif "gate_mode" in result.lower() or "interleave" in result.lower():
    print("    -> UNEXPECTED: gate_mode guard fired instead of g1u1 guard (order changed?)")
else:
    print("    -> UNEXPECTED result, inspect manually")

print()
print("=== 4) Non-Relu2 unaffected: Swiglu + INTERLEAVE (must still succeed as before) ===")
try_case(
    "Swiglu+INTERLEAVE",
    gate_mode=GateMode.INTERLEAVE.value,
    use_g1u1=True,
    activation=ActivationType.Swiglu,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.fp4x2,
    **base,
)

print()
print("=== 5) Non-Relu2 unaffected: Situv2 + INTERLEAVE (must still succeed as before) ===")
try_case(
    "Situv2+INTERLEAVE",
    gate_mode=GateMode.INTERLEAVE.value,
    use_g1u1=True,
    activation=ActivationType.Situv2,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.fp4x2,
    **base,
)

print()
print("=== 6) Non-Relu2 unaffected: Silu default (SEPARATED, control) ===")
try_case(
    "Silu+SEPARATED",
    gate_mode=GateMode.SEPARATED.value,
    use_g1u1=False,
    activation=ActivationType.Silu,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.bf16,
    **base,
)