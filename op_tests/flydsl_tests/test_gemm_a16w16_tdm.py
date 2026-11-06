# SPDX-License-Identifier: MIT
"""The gfx1250 TDM policy preserves source GEMM/bias/cluster rounding semantics."""

import unittest

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm,
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.flydsl.kernels.gemm_a16w16_tdm_gfx1250 import (
    validate_gemm_gfx1250_config,
)


class TestTdmCatalog(unittest.TestCase):
    def test_kernel_name_round_trip_and_arch_guard(self):
        cfg = {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "stages": 3,
            "m_waves": 2,
            "n_waves": 2,
            "group_m": 4,
            "split_k": 2,
            "k_waves": 1,
            "use_half_tile_interleaved": False,
            "policy": "tdm",
        }
        for dtype in (torch.float16, torch.bfloat16):
            for gfx in ("gfx1250", "gfx1250-strict"):
                name = flydsl_hgemm_kernel_name(
                    dtype=dtype,
                    out_dtype=dtype,
                    config=cfg,
                    has_bias=True,
                    target_gfx=gfx,
                )
                got = get_flydsl_hgemm_kernel_params(name)
                self.assertEqual({k: got[k] for k in cfg}, cfg)
                self.assertTrue(got["has_bias"])
                self.assertIsNone(
                    get_flydsl_hgemm_kernel_params(name.replace(gfx, "gfx950"))
                )
        name = flydsl_hgemm_kernel_name(
            dtype="bf16",
            out_dtype="fp32",
            config=cfg,
            has_bias=False,
            target_gfx="gfx1250",
        )
        self.assertIsNone(get_flydsl_hgemm_kernel_params(name))

    def test_validation_rejects_bad_tiles(self):
        cfg = {
            "m": 128,
            "n": 256,
            "k": 512,
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "stages": 2,
            "m_waves": 2,
            "n_waves": 2,
            "group_m": 0,
            "dtype_id": 2,
            "mma_m": 16,
            "mma_n": 16,
            "mma_k": 32,
        }
        for changed in (
            {"k": 513},
            {"stages": 1},
            {"split_k": 10},
            {"block_k": 96},
            {"m_waves": 0},
            {"group_m": -1},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_gemm_gfx1250_config(**{**cfg, **changed})

    def test_source_space_fingerprints(self):
        import hashlib
        import json
        from types import SimpleNamespace
        from unittest import mock

        from aiter.ops.flydsl.gemm_a16w16_tdm_policy import get_gfx1250_tdm_configs

        # Ordered fingerprints from the source commit, with gfx1250 properties.
        props = SimpleNamespace(
            multi_processor_count=256,
            warp_size=32,
            max_threads_per_multi_processor=2048,
            shared_memory_per_block=320 * 1024,
            shared_memory_per_multiprocessor=320 * 1024,
        )
        cases = [
            (
                [8, 4096, 4096],
                False,
                273,
                "e0c0d9c90d6dc7d9cfd9ceb503baac6a8fb74840124ffa7dc38c9bc27b72e15c",
            ),
            (
                [8, 4096, 4096],
                True,
                781,
                "f836fae4b08c25344a202f3942132a23c995c7e844191321c1f40d3242e3311d",
            ),
            (
                [8192, 8192, 8192],
                False,
                650,
                "aba4b3d3315ed0043406164137c21a99256b73f6251621a505129ce27dda138f",
            ),
            (
                [8192, 8192, 8192],
                True,
                650,
                "aba4b3d3315ed0043406164137c21a99256b73f6251621a505129ce27dda138f",
            ),
        ]
        with (
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(torch.cuda, "get_device_properties", lambda _: props),
        ):
            for shape, split, count, digest in cases:
                with self.subTest(shape=shape, split=split):
                    configs = get_gfx1250_tdm_configs(
                        *shape, torch.bfloat16, enable_split_k=split
                    )
                    self.assertEqual(len(configs), count)
                    self.assertEqual(
                        hashlib.sha256(
                            json.dumps(configs, sort_keys=True).encode()
                        ).hexdigest(),
                        digest,
                    )


class TestTdmDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or get_gfx() not in (
            "gfx1250",
            "gfx1250-strict",
        ):
            raise unittest.SkipTest("requires gfx1250")
        torch.set_num_threads(4)

    def case(self, dtype, layout, sk, bias_on, group, m=65, n=136):
        torch.manual_seed(1106)
        k = sk * 384
        # Copy full padded storages, then slice to retain nonpacked leading strides.
        ac = (torch.randint(-4, 5, (m, k + 16)).float() / 16).to(dtype)
        wc = (
            torch.randint(-4, 5, (n, k + 32) if layout == "nt" else (k, n + 16)).float()
            / 16
        ).to(dtype)
        a = ac.cuda()[:, :k]
        ws = wc.cuda()
        w = ws[:, :k] if layout == "nt" else ws[:, :n].t()
        a0 = ac[:, :k].double()
        w0 = wc[:, :k].double() if layout == "nt" else wc[:, :n].t().double()
        bc = (torch.arange(n).remainder(7).float() / 16).to(dtype) if bias_on else None
        bias = bc.cuda() if bc is not None else None
        part = k // sk
        reference = sum(
            (a0[:, i * part : (i + 1) * part] @ w0[:, i * part : (i + 1) * part].t())
            .to(dtype)
            .double()
            for i in range(sk)
        ).to(dtype)
        if bc is not None:
            reference = (reference.float() + bc.float()).to(dtype)
        storage = torch.full((m + 1, n + 16), 83, device="cuda", dtype=dtype)
        out = storage[:m, :n]
        cfg = {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "stages": 3,
            "split_k": sk,
            "m_waves": 2,
            "n_waves": 2,
            "group_m": group,
            "policy": "tdm",
        }
        self.assertIs(flydsl_hgemm(a, w, out, bias, **cfg), out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        flydsl_hgemm(a, w, out, bias, stream=stream, **cfg)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flydsl_hgemm(a, w, out, bias, **cfg)
        a.neg_()
        if bias is not None:
            bias.neg_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(), -reference, atol=0, rtol=0)
        self.assertTrue((storage[m:] == 83).all())
        self.assertTrue((storage[:m, n:] == 83).all())
        graph.reset()

    def test_layout_bias_splitk_group_and_graph(self):
        for dtype in (torch.float16, torch.bfloat16):
            for layout in ("nt", "nn"):
                for sk in (1, 2, 3):
                    for bias_on in (False, True):
                        with self.subTest(
                            dtype=dtype, layout=layout, sk=sk, bias=bias_on
                        ):
                            self.case(dtype, layout, sk, bias_on, 4 if bias_on else 0)

    def test_small_m_and_split9(self):
        for m in (1, 7, 8):
            with self.subTest(m=m):
                self.case(torch.bfloat16, "nt", 9, True, 4, m=m, n=136)

    def test_bias_rounding_is_before_add(self):
        dtype = torch.bfloat16
        a = torch.zeros((64, 128), device="cuda", dtype=dtype)
        w = torch.zeros((128, 128), device="cuda", dtype=dtype)
        a[:, 0] = 1
        a[:, 1] = 2**-8
        w[:, :2] = 1
        bias = torch.full((128,), 2**-8, device="cuda", dtype=dtype)
        out = flydsl_hgemm(
            a,
            w,
            bias=bias,
            policy="tdm",
            block_m=64,
            block_n=128,
            block_k=128,
            m_waves=2,
            n_waves=2,
        )
        torch.cuda.synchronize()
        self.assertTrue((out == 1).all())

    def test_tuned_dispatch_keeps_tdm_policy(self):
        from aiter.tuned_gemm import flydsl_gemm

        a = torch.ones((64, 256), device="cuda", dtype=torch.bfloat16)
        w = torch.ones((128, 256), device="cuda", dtype=a.dtype)
        cfg = {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "stages": 2,
            "m_waves": 2,
            "n_waves": 2,
            "k_waves": 1,
            "group_m": 4,
            "split_k": 1,
            "use_half_tile_interleaved": False,
            "policy": "tdm",
        }
        name = flydsl_hgemm_kernel_name(
            dtype=a.dtype,
            out_dtype=a.dtype,
            config=cfg,
            has_bias=False,
            target_gfx=get_gfx(),
        )
        out = flydsl_gemm(a, w, 0, otype=a.dtype, config={"kernelName": name})
        torch.cuda.synchronize()
        self.assertTrue((out == 256).all())


if __name__ == "__main__":
    unittest.main()
