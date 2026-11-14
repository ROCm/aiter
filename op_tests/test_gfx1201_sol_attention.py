# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import unittest

import torch
from aiter.ops.gfx1201.asm_attention import launch_hip_sage_core
from aiter.ops.gfx1201.prepare import prepare_sage
from aiter.ops.gfx1201.sol_attention import sol_attention, sol_route


def make_prepared(rows, heads, seed=11):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # a few shared directions so the attention has structure the router can find
    basis = torch.randn((8, 128), device="cuda", generator=generator)
    def tensor():
        mix = torch.randn((rows, heads, 8), device="cuda", generator=generator)
        return (mix @ basis + 0.5 * torch.randn((rows, heads, 128), device="cuda", generator=generator)).bfloat16()
    query, key, value = tensor(), tensor(), tensor()
    return (query, key, value), prepare_sage(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))


def dense(prepared, rows, heads):
    out = torch.zeros(1, prepared[0].shape[1], heads, 128, device="cuda", dtype=torch.bfloat16)
    launch_hip_sage_core(prepared[0], prepared[2], prepared[4], prepared[1], prepared[3], prepared[5], out, 1,
                         prepared[0].shape[1], rows, heads)
    return out[0, :rows]


def reference_mask(prepared, rows, tau, prefix):
    """NVlabs Sol-Attn diag routing + Sol-H3 sink/band policy on the dequantised operands, unioned per 512 rows."""
    q, qs, k, ks = prepared[:4]
    heads = q.shape[2]
    deq = lambda x, s: x[0, :rows].float() * s[0].repeat_interleave(32, 1)[:, :rows].T[:, :, None]
    qf, kf = deq(q, qs), deq(k, ks)
    nb = math.ceil(rows / 64)
    pad = nb * 64 - rows
    lengths = torch.full((nb,), 64.0, device="cuda")
    lengths[-1] = rows - (nb - 1) * 64
    pool = lambda x: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad)).view(nb, 64, heads, 128).sum(1) / lengths[:, None, None]
    qc, kc = pool(qf), pool(kf)
    mu, var = kc.mean(0), (kc * kc).mean(0) - kc.mean(0) ** 2
    thr = (qc * mu).sum(-1) + tau * torch.sqrt((qc * qc * var).sum(-1).clamp_min(0) + 1e-6)  # [nb, H]
    exact = torch.einsum("qhd,khd->hqk", qc, kc) > thr.T[:, :, None]
    blocks = torch.arange(nb, device="cuda")
    sink = blocks < (prefix + 63) // 64
    exact |= ((blocks[:, None] - blocks[None, :]).abs() <= 1)[None] | sink[None, None] | sink[None, :, None]
    exact[..., -1] = True
    exact = torch.nn.functional.pad(exact, (0, 0, 0, (-nb) % 8)).view(heads, -1, 8, nb).any(2)
    return exact.reshape(-1, nb)  # [H * nqt, nb], workgroup order head * nqt + qtile


def plan_mask(plan, nb):
    words = plan["mask"].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = (words[:, :, None] >> torch.arange(32, device="cuda")) & 1
    return bits.reshape(words.shape[0], -1)[:, :nb].bool()


class SolAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm GPU required")

    def test_all_exact_matches_core_bitwise(self):
        # every block exact (whole-sequence prefix, or a threshold below every score): the list-driven core with
        # the neutral Phase-A state must reproduce the dense core
        for rows, heads in ((33, 7), (513, 14), (4097, 28), (20000, 7)):
            _, prepared = make_prepared(rows, heads)
            expected = dense(prepared, rows, heads)
            for tau, prefix in ((1.0, rows), (-1.0e30, 0)):
                out = torch.zeros(1, prepared[0].shape[1], heads, 128, device="cuda", dtype=torch.bfloat16)
                sol_attention(prepared, out, rows, tau=tau, prefix=prefix)
                self.assertTrue(torch.equal(out[0, :rows], expected), f"rows={rows} heads={heads} tau={tau}")

    def test_routing_matches_sol_attn_reference(self):
        for rows, heads, prefix in ((4097, 7, 0), (20000, 7, 1000), (30001, 4, 6804)):
            _, prepared = make_prepared(rows, heads, seed=rows)
            for tau in (0.0, 1.0, 2.5):
                plan = sol_route(prepared, rows, tau, prefix)
                got = plan_mask(plan, plan["nkb"])
                ref = reference_mask(prepared, rows, tau, prefix)
                mismatch = (got != ref).float().mean().item()
                # fp32 summation order may flip blocks whose score sits on the threshold
                self.assertLess(mismatch, 1e-4, f"rows={rows} tau={tau} mismatch={mismatch}")

    def test_tau_monotone_and_error_bounded(self):
        rows, heads = 20000, 7
        (query, key, value), _ = make_prepared(rows, heads)
        query = (query.float() * 0.5).bfloat16()
        prepared = prepare_sage(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
        sample = torch.arange(0, rows, 97, device="cuda")
        reference = torch.stack([torch.softmax(query[sample, h].float() @ key[:, h].float().T * 128 ** -0.5, -1)
                                 @ value[:, h].float() for h in range(heads)], 1)
        error = lambda o: ((o[sample].float() - reference).norm() / reference.norm()).item()
        base = error(dense(prepared, rows, heads))
        densities, errors = [], []
        for tau in (3.0, 1.0, 0.0, -3.0):
            out = torch.zeros(1, prepared[0].shape[1], heads, 128, device="cuda", dtype=torch.bfloat16)
            plan = sol_attention(prepared, out, rows, tau=tau, prefix=1024)
            self.assertFalse(out[0, :rows].isnan().any().item(), f"tau={tau}")
            densities.append(plan["tile_count"].sum().item() / (plan["n_wg"] * (rows // 32)))
            errors.append(error(out[0, :rows]))
        self.assertEqual(densities, sorted(densities), f"density not monotone in tau: {densities}")
        self.assertLess(densities[0], densities[-1])
        self.assertLess(errors[-1], 2 * base + 0.01, f"errors {errors}, dense {base}")


if __name__ == "__main__":
    unittest.main()
