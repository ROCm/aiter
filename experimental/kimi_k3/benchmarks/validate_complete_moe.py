"""Checkpoint loading and independent FP32-dense/AITER-expert references."""

import json
import os
from pathlib import Path
from contextlib import ExitStack
import torch
import torch.distributed as dist
from k3_gluon.moe_complete import pack_dense, pack_down
from k3_gluon.expert_up import pack_expert_up


def dequant(q, s):
    nib = torch.stack((q & 15, q >> 4), -1).flatten(-2)
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=q.device,
    )
    scale = (s.int() << 23).view(torch.float32).repeat_interleave(32, -1)
    return (lut[nib.long()] * scale).bfloat16().float()


def layer_weight(layer, name):
    from safetensors import safe_open

    root = Path(os.environ["KIMI_K3_MODEL"])
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = f"language_model.model.layers.{layer}.{name}.weight"
    with safe_open(root / index[key], framework="pt") as f:
        return f.get_tensor(key).cuda()


def checkpoint_weights(rank, world, layer):
    from safetensors import safe_open

    root = Path(os.environ["KIMI_K3_MODEL"])
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe."
    with ExitStack() as stack:
        files = {}

        def read(name, selection=None, full=False):
            key = name if full else prefix + name
            filename = index[key]
            if filename not in files:
                files[filename] = stack.enter_context(
                    safe_open(root / filename, framework="pt")
                )
            tensor = (
                files[filename].get_tensor(key)
                if selection is None
                else files[filename].get_slice(key)[selection]
            )
            return tensor.contiguous().cuda()

        inter = 3072 // world
        shard = slice(rank * inter, (rank + 1) * inter)
        shared_shard = slice(rank * 2 * inter, (rank + 1) * 2 * inter)
        qu = torch.empty(896, 2 * inter, 1792, dtype=torch.uint8, device="cuda")
        su = torch.empty(896, 2 * inter, 112, dtype=torch.uint8, device="cuda")
        qd = torch.empty(896, 3584, inter // 2, dtype=torch.uint8, device="cuda")
        ss = torch.empty(896, 3584, inter // 32, dtype=torch.uint8, device="cuda")
        for expert in range(896):
            for offset, part in ((0, "w1"), (inter, "w3")):
                qu[expert, offset : offset + inter] = read(
                    f"experts.{expert}.{part}.weight_packed", shard
                )
                su[expert, offset : offset + inter] = read(
                    f"experts.{expert}.{part}.weight_scale", shard
                )
            qd[expert] = read(
                f"experts.{expert}.w2.weight_packed",
                (slice(None), slice(rank * inter // 2, (rank + 1) * inter // 2)),
            )
            ss[expert] = read(
                f"experts.{expert}.w2.weight_scale",
                (slice(None), slice(rank * inter // 32, (rank + 1) * inter // 32)),
            )
        router = read("gate.weight")
        bias = read("gate.e_score_correction_bias")
        down = read("routed_expert_down_proj.weight")
        gain = read("routed_expert_norm.weight")
        shared = torch.cat(
            [
                read("shared_experts.gate_proj.weight", shared_shard),
                read("shared_experts.up_proj.weight", shared_shard),
            ]
        )
        sd = read("shared_experts.down_proj.weight", (slice(None), shared_shard))
        up = read(
            "routed_expert_up_proj.weight",
            slice(rank * 7168 // world, (rank + 1) * 7168 // world),
        )
        input_gain = read(
            f"language_model.model.layers.{layer}.post_attention_layernorm.weight",
            full=True,
        )
    weights = dict(
        Router=pack_dense(router),
        Bias=bias,
        Down=pack_dense(down),
        Shared=pack_dense(shared),
        WUp=pack_expert_up(qu),
        SUp=su,
        WDown=pack_down(qd),
        SDown=ss,
        NormWeight=gain,
        SharedDown=pack_dense(sd),
        LatentUp=pack_dense(up),
    )
    ref = dict(
        router=router,
        bias=bias,
        down=down,
        gain=gain,
        shared=shared,
        sd=sd,
        up=up,
        qu=qu,
        su=su,
        qd=qd,
        ss=ss,
        input_gain=input_gain,
    )
    return weights, ref


def situ(gu):
    gate, up = gu.float().chunk(2, -1)
    return (
        4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    ).bfloat16()


def tp_sum(x, world):
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x)
    total = torch.zeros_like(x, dtype=torch.float32)
    for part in parts:
        total += part.float()
    return total.bfloat16()


def check(x, s, ref, rank, world, norm_materialized=True):
    errors = {}

    def close(name, actual, expected, rtol=0.03, atol=0.012):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=rtol, atol=atol
        )
        errors[name] = (actual.float() - expected.float()).abs().max().item()

    scores = (x.float() @ ref["router"].float().T).sigmoid()
    close("scores", s["Scores"], scores, 2.0e-5, 2.0e-6)
    ids = torch.argsort(scores + ref["bias"], dim=-1, descending=True, stable=True)[
        :, :16
    ]
    torch.testing.assert_close(s["IDs"], ids.int(), rtol=0, atol=0)
    weights = scores.gather(1, ids)
    weights /= weights.sum(-1, keepdim=True)
    close("weights", s["Weights"], weights, 2.0e-5, 2.0e-6)
    latent = (x.float() @ ref["down"].float().T).bfloat16()
    gu = (x.float() @ ref["shared"].float().T).bfloat16()
    close("latent", s["Latent"], latent)
    close("shared_gu", s["GU"], gu)
    close("shared_mid", s["SharedMid"], situ(s["GU"]))
    mids = []
    routed = []
    full_routed = []
    for sample in range(x.shape[0]):
        # Stage-isolated references use the actual preceding BF16 output.
        expert_gu = torch.stack(
            [
                dequant(ref["qu"][e], ref["su"][e]) @ s["Latent"][sample].float()
                for e in ids[sample]
            ]
        ).bfloat16()
        mids.append(situ(expert_gu))
        local = torch.zeros(3584, device="cuda")
        full_local = torch.zeros_like(local)
        full_gu = torch.stack(
            [
                dequant(ref["qu"][e], ref["su"][e]) @ latent[sample].float()
                for e in ids[sample]
            ]
        ).bfloat16()
        full_mid = situ(full_gu)
        for route, expert in enumerate(ids[sample]):
            dw = dequant(ref["qd"][expert], ref["ss"][expert])
            local += (dw @ s["Mid"][sample, route].float()) * s["Weights"][
                sample, route
            ]
            full_local += (dw @ full_mid[route].float()) * weights[sample, route]
        routed.append(local.bfloat16())
        full_routed.append(full_local.bfloat16())
    close("expert_mid", s["Mid"], torch.stack(mids), 0.04, 0.015)
    reduced = tp_sum(torch.stack(routed), world)
    close("routed", s["Routed"], reduced, 0.025, 0.008)
    rr = s["Routed"].float()
    norm = (
        rr
        * torch.rsqrt(rr.square().mean(-1, keepdim=True) + 1.0e-5)
        * ref["gain"].float()
    ).bfloat16()
    if norm_materialized:
        close("norm", s["Normed"], norm, 0.01, 0.001)
        norm = s["Normed"]
    local = (s["SharedMid"].float() @ ref["sd"].float().T).bfloat16()
    shard = slice(rank * (7168 // world), (rank + 1) * (7168 // world))
    local[:, shard] = (
        local[:, shard].float() + norm.float() @ ref["up"].float().T
    ).bfloat16()
    result = tp_sum(local, world)
    close("output_staged", s["Out"], result, 0.025, 0.012)
    full_latent = tp_sum(torch.stack(full_routed), world).float()
    full_norm = (
        full_latent
        * torch.rsqrt(full_latent.square().mean(-1, keepdim=True) + 1.0e-5)
        * ref["gain"].float()
    ).bfloat16()
    full_local = (situ(gu).float() @ ref["sd"].float().T).bfloat16()
    full_local[:, shard] = (
        full_local[:, shard].float() + full_norm.float() @ ref["up"].float().T
    ).bfloat16()
    full_output = tp_sum(full_local, world)
    # Different legal reduction orders can shift BF16 rounding boundaries.
    # Bound aggregate error as well as pointwise error (including near zeros).
    close("output_full", s["Out"], full_output, 0.06, 0.04)
    nrmse = (
        s["Out"].float() - full_output.float()
    ).square().mean().sqrt() / full_output.float().square().mean().sqrt().clamp_min(
        1.0e-12
    )
    assert nrmse.item() < 0.012, nrmse.item()
    errors["output_full_nrmse"] = nrmse.item()
    assert not s["Errors"].count_nonzero().item()
    assert torch.all(s["Epochs"] == s["Epochs"][0])
    return errors
