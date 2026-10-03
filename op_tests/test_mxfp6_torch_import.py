# SPDX-License-Identifier: MIT
"""Exercise the complete FP6 packing module without Triton or GPU launches."""

import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_packer(*, without_torch=False):
    # Load the full module, bypassing AITER's GPU-dependent package initializer.
    source = (
        Path(__file__).resolve().parents[1]
        / "aiter/ops/triton/quant/mxfp6_fmha_pack.py"
    )
    spec = importlib.util.spec_from_file_location("fp6_packer_without_triton", source)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def limited_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("Triton intentionally unavailable")
        if without_torch and (name == "torch" or name.startswith("torch.")):
            raise ImportError("Torch intentionally unavailable")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", limited_import):
        spec.loader.exec_module(module)
    return module


class TestFP6TorchImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packer = load_packer()

    def test_lastdim_exact_representable_values(self):
        # All positive E2M3 codes in numerical order. Each block has amax=7.5,
        # so its scale is exactly 127 and quantization needs no rounding.
        positive = [i / 8 for i in range(8)]
        positive += [1 + i / 8 for i in range(8)]
        positive += [2 + i / 4 for i in range(8)]
        positive += [4 + i / 2 for i in range(8)]
        expected = []
        for sign in (0, 32):
            codes = [code + sign for i in range(16) for code in (i, i + 16)]
            for i in range(0, 32, 4):
                word = sum(codes[i + j] << (6 * j) for j in range(4))
                expected.extend(word.to_bytes(3, "little"))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([positive, [-v for v in positive]], dtype=dtype)
                packed, scale = self.packer.quantize_fp6_lastdim_torch(x)
                self.assertEqual(packed.dtype, torch.uint8)
                self.assertEqual(packed.shape, (2, 24))
                self.assertEqual(packed.flatten().tolist(), expected)
                self.assertEqual(scale.tolist(), [[127], [127]])

    def test_k_raw_and_view_layout_for_partial_tiles(self):
        for sequence in (1, 127, 128, 129):
            with self.subTest(sequence=sequence):
                batch, heads = 2, 3
                tiles = (sequence + 127) // 128
                x = torch.zeros(batch, sequence, heads, 128)
                buf, sbuf = self.packer.quantize_fp6_k_lds_order_torch(
                    x, return_raw=True
                )
                self.assertEqual(buf.numel(), batch * heads * tiles * 17408 + 256)
                self.assertEqual(sbuf.numel(), batch * sequence * heads * 4 + 64)
                records = buf[:-256].reshape(batch, heads, tiles, 17408)
                self.assertEqual(records[..., :16384].count_nonzero().item(), 0)
                self.assertEqual(buf[-256:].count_nonzero().item(), 0)
                self.assertEqual(sbuf[-64:].count_nonzero().item(), 0)
                self.assertTrue(torch.all(sbuf[:-64] == 127).item())
                # First lane in each tile reads its first token's scale.
                for tile in range(tiles):
                    self.assertTrue(torch.all(records[:, :, tile, 16384] == 127).item())
                view, scale = self.packer.quantize_fp6_k_lds_order_torch(x)
                rebuilt, rebuilt_scale = self.packer.fp6_k_lds_order_views_from_raw(
                    buf, sbuf, batch, sequence, heads
                )
                self.assertEqual(view.shape, (batch, sequence, heads, 96))
                self.assertEqual(
                    view.stride(), (heads * tiles * 17408, 136, tiles * 17408, 1)
                )
                self.assertTrue(torch.equal(view, rebuilt))
                self.assertTrue(torch.equal(scale, rebuilt_scale))

    def test_triton_entry_points_still_reject_missing_triton(self):
        self.assertFalse(self.packer._HAVE_TRITON)
        x = torch.zeros(1, 1, 1, 128)
        for name in (
            "quantize_fp6_lastdim_triton",
            "quantize_fp6_k_lds_order_triton",
            "quantize_fp6_v_clean_triton",
            "quantize_fp6_v_data_scale_triton",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(AssertionError, "triton/torch unavailable"),
            ):
                getattr(self.packer, name)(x)

    def test_torch_entry_points_reject_missing_torch(self):
        packer = load_packer(without_torch=True)
        for name in ("quantize_fp6_lastdim_torch", "quantize_fp6_k_lds_order_torch"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(AssertionError, "torch unavailable"),
            ):
                getattr(packer, name)(None)


if __name__ == "__main__":
    unittest.main()
