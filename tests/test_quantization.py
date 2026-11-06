import os
import sys
import unittest
from collections import OrderedDict

import torch

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import FcNet, SimpleCNN_header
from quantization import (
    calc_compression_stats,
    dequantize_representations,
    dequantize_state_dict,
    dequantize_tensor,
    quantize_representations,
    quantize_state_dict,
    quantize_tensor,
    simulate_quantization,
)


class TestMOONQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_dequantize_tensor_int4(self):
        for shape in [(64,), (32, 32), (16, 3, 3, 3), (7, 13, 11)]:
            tensor = torch.randn(shape, dtype=torch.float32)
            q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
            recon = dequantize_tensor(q_dict)

            self.assertEqual(recon.shape, tensor.shape)
            self.assertEqual(recon.dtype, tensor.dtype)
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor.flatten(), recon.flatten(), dim=0
            )
            self.assertGreater(cos_sim.item(), 0.98)

    def test_quantize_dequantize_tensor_int8(self):
        tensor = torch.randn(64, 64, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=8, block_size=64)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, tensor.shape)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.flatten(), recon.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)

    def test_zero_and_empty_tensor(self):
        zero_tensor = torch.zeros(128, dtype=torch.float32)
        q_dict = quantize_tensor(zero_tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertTrue(torch.equal(zero_tensor, recon))
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(recon).any())

        empty_tensor = torch.empty((0, 10), dtype=torch.float32)
        q_empty = quantize_tensor(empty_tensor, nbits=4, block_size=32)
        recon_empty = dequantize_tensor(q_empty)
        self.assertEqual(recon_empty.numel(), 0)

    def test_sub_byte_bit_packing(self):
        numel = 256
        tensor = torch.randn(numel, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=64)

        self.assertEqual(q_dict["data"].dtype, torch.uint8)
        self.assertEqual(q_dict["data"].numel(), numel // 2)

    def test_representation_quantization(self):
        # Contrastive projection representations (batch_size=8, pro_dim=256)
        reps = torch.randn(8, 256, dtype=torch.float32)
        q_reps = quantize_representations(reps, nbits=4, block_size=32)
        restored = dequantize_representations(q_reps)

        self.assertEqual(restored.shape, reps.shape)
        # Cosine similarity between original and quantized representations
        cos = torch.nn.functional.cosine_similarity(reps, restored, dim=-1)
        self.assertTrue((cos > 0.98).all().item())

    def test_quantize_state_dict(self):
        model = FcNet(input_dim=784, hidden_dims=[128, 64], output_dim=10)
        q_dict = quantize_state_dict(model, nbits=4, block_size=32)

        target_model = FcNet(input_dim=784, hidden_dims=[128, 64], output_dim=10)
        dequantize_state_dict(q_dict, target_model=target_model)

        for p_orig, p_target in zip(model.parameters(), target_model.parameters()):
            self.assertEqual(p_orig.shape, p_target.shape)
            cos = torch.nn.functional.cosine_similarity(
                p_orig.flatten(), p_target.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_error_feedback_residuals(self):
        residuals = {}
        true_accum = torch.zeros(64)
        quant_accum = torch.zeros(64)

        for _ in range(10):
            step_delta = torch.randn(64) * 0.05
            true_accum += step_delta

            p_dict = OrderedDict({"weight": step_delta})
            q_dict = quantize_state_dict(
                p_dict, nbits=4, block_size=32, residuals=residuals
            )
            recon = dequantize_state_dict(q_dict)
            quant_accum += recon["weight"]

        drift = torch.norm(true_accum - quant_accum).item()
        self.assertLess(drift, 0.05)

    def test_simulate_quantization(self):
        model = SimpleCNN_header(
            input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10
        )
        simulated = simulate_quantization(model, nbits=4, block_size=64)

        for p_orig, p_sim in zip(model.parameters(), simulated.parameters()):
            self.assertEqual(p_orig.shape, p_sim.shape)
            cos = torch.nn.functional.cosine_similarity(
                p_orig.flatten(), p_sim.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_calc_compression_stats(self):
        model = FcNet(input_dim=784, hidden_dims=[128, 64], output_dim=10)
        q_dict = quantize_state_dict(model, nbits=4, block_size=64)
        stats = calc_compression_stats(model, q_dict)

        self.assertGreater(stats["compression_ratio"], 6.0)
        self.assertGreater(stats["bandwidth_savings_pct"], 80.0)
        self.assertGreater(stats["raw_bytes"], stats["compressed_bytes"])

    def test_invalid_inputs(self):
        tensor = torch.randn(32)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=9)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=4, block_size=9)


if __name__ == "__main__":
    unittest.main()
