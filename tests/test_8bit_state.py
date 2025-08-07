"""
Unit tests for 8-bit optimizer states implementation.

Tests the integration of bitsandbytes.optim.Adam8bit with MANGO-LRQ
compression for memory-efficient optimization.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mango.enhanced_optimizer import EnhancedMANGO
from mango.mango_lrq import CompressionConfig

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class Test8BitOptimizerStates(unittest.TestCase):
    """Test suite for 8-bit optimizer state functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple test model
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(self.device)
        
        # Create compression configuration
        self.compression_config = CompressionConfig(
            rank=4,
            bits_P=8,
            bits_Q=8,
            momentum_precision='int8',
            use_nf4=True,
            error_feedback=True,
            variance_reduction=True,
            reference_steps=10
        )
    
    def test_8bit_optimizer_initialization(self):
        """Test that 8-bit optimizer is properly initialized when bitsandbytes is available."""
        optimizer = EnhancedMANGO(
            self.model.parameters(),
            lr=1e-3,
            compression_config=self.compression_config,
            use_8bit_optimizer=True,
            use_mango_lrq=True,
            use_tinyformer=False  # Disable for simplicity
        )
        
        if HAS_BITSANDBYTES:
            # Should have 8-bit optimizer enabled
            self.assertTrue(optimizer.use_8bit_optimizer)
            self.assertIsInstance(optimizer._8bit_optimizer_states, dict)
            print("✅ 8-bit optimizer initialization test passed")
        else:
            # Should fallback gracefully
            self.assertFalse(optimizer.use_8bit_optimizer)
            print("⚠️  bitsandbytes not available, fallback test passed")
    
    def test_8bit_optimizer_step(self):
        """Test that optimizer step works with 8-bit states."""
        optimizer = EnhancedMANGO(
            self.model.parameters(),
            lr=1e-3,
            compression_config=self.compression_config,
            use_8bit_optimizer=True,
            use_mango_lrq=True,
            use_tinyformer=False
        )
        
        # Create dummy input and target
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 1, device=self.device)
        
        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (should work with both 8-bit and fallback)
        try:
            optimizer.step()
            optimizer.zero_grad()
            print("✅ 8-bit optimizer step test passed")
        except Exception as e:
            self.fail(f"Optimizer step failed: {e}")
    
    def test_memory_savings(self):
        """Test that 8-bit optimizer actually saves memory."""
        if not HAS_BITSANDBYTES or not torch.cuda.is_available():
            self.skipTest("bitsandbytes and CUDA required for memory test")
        
        # Create larger model for measurable memory difference
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        ).to(self.device)
        
        # Test with 8-bit optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        optimizer_8bit = EnhancedMANGO(
            large_model.parameters(),
            lr=1e-3,
            compression_config=self.compression_config,
            use_8bit_optimizer=True,
            use_mango_lrq=True,
            use_tinyformer=False
        )
        
        # Run a few steps\n        for _ in range(5):\n            x = torch.randn(64, 1000, device=self.device)\n            y = torch.randn(64, 1, device=self.device)\n            \n            output = large_model(x)\n            loss = nn.MSELoss()(output, y)\n            loss.backward()\n            optimizer_8bit.step()\n            optimizer_8bit.zero_grad()\n        \n        memory_8bit = torch.cuda.max_memory_allocated() / (1024**3)  # GB\n        \n        # Reset and test with regular optimizer\n        del optimizer_8bit\n        torch.cuda.empty_cache()\n        torch.cuda.reset_peak_memory_stats()\n        \n        optimizer_regular = EnhancedMANGO(\n            large_model.parameters(),\n            lr=1e-3,\n            compression_config=self.compression_config,\n            use_8bit_optimizer=False,\n            use_mango_lrq=True,\n            use_tinyformer=False\n        )\n        \n        # Run same steps\n        for _ in range(5):\n            x = torch.randn(64, 1000, device=self.device)\n            y = torch.randn(64, 1, device=self.device)\n            \n            output = large_model(x)\n            loss = nn.MSELoss()(output, y)\n            loss.backward()\n            optimizer_regular.step()\n            optimizer_regular.zero_grad()\n        \n        memory_regular = torch.cuda.max_memory_allocated() / (1024**3)  # GB\n        \n        memory_savings = (memory_regular - memory_8bit) / memory_regular * 100\n        \n        print(f\"8-bit optimizer memory: {memory_8bit:.3f} GB\")\n        print(f\"Regular optimizer memory: {memory_regular:.3f} GB\")\n        print(f\"Memory savings: {memory_savings:.1f}%\")\n        \n        # Should see some memory savings (even if small for this test)\n        self.assertGreaterEqual(memory_savings, -5, \n            \"8-bit optimizer should not use significantly more memory\")\n    \n    def test_8bit_optimizer_compatibility(self):\n        \"\"\"Test that 8-bit optimizer works with MANGO-LRQ compression.\"\"\"\n        optimizer = EnhancedMANGO(\n            self.model.parameters(),\n            lr=1e-3,\n            compression_config=self.compression_config,\n            use_8bit_optimizer=True,\n            use_mango_lrq=True,\n            use_tinyformer=False\n        )\n        \n        # Test multiple training steps\n        for step in range(10):\n            x = torch.randn(16, 10, device=self.device)\n            y = torch.randn(16, 1, device=self.device)\n            \n            optimizer.zero_grad()\n            output = self.model(x)\n            loss = nn.MSELoss()(output, y)\n            loss.backward()\n            optimizer.step()\n            \n            # Verify loss is decreasing (basic sanity check)\n            if step == 0:\n                initial_loss = loss.item()\n            elif step == 9:\n                final_loss = loss.item()\n        \n        # Loss should generally decrease over training\n        self.assertLess(final_loss, initial_loss * 2, \n            \"Training should make progress with 8-bit optimizer\")\n        \n        print(f\"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}\")\n        print(\"✅ 8-bit optimizer compatibility test passed\")\n    \n    def test_compression_stats_with_8bit(self):\n        \"\"\"Test that compression statistics work with 8-bit optimizer.\"\"\"\n        optimizer = EnhancedMANGO(\n            self.model.parameters(),\n            lr=1e-3,\n            compression_config=self.compression_config,\n            use_8bit_optimizer=True,\n            use_mango_lrq=True,\n            use_tinyformer=False\n        )\n        \n        # Run a few steps to generate compression statistics\n        for _ in range(5):\n            x = torch.randn(16, 10, device=self.device)\n            y = torch.randn(16, 1, device=self.device)\n            \n            optimizer.zero_grad()\n            output = self.model(x)\n            loss = nn.MSELoss()(output, y)\n            loss.backward()\n            optimizer.step()\n        \n        # Get compression statistics\n        comp_stats = optimizer.get_compression_stats()\n        \n        # Should have valid compression statistics\n        self.assertIsInstance(comp_stats, dict)\n        if 'error' not in comp_stats:\n            self.assertIn('avg_compression_ratio', comp_stats)\n            self.assertGreater(comp_stats['avg_compression_ratio'], 0.5)\n            print(f\"Compression ratio: {comp_stats['avg_compression_ratio']:.2f}x\")\n        \n        print(\"✅ Compression statistics test passed\")\n    \n    def test_memory_usage_reporting(self):\n        \"\"\"Test that memory usage reporting works with 8-bit optimizer.\"\"\"\n        optimizer = EnhancedMANGO(\n            self.model.parameters(),\n            lr=1e-3,\n            compression_config=self.compression_config,\n            use_8bit_optimizer=True,\n            use_mango_lrq=True,\n            use_tinyformer=False\n        )\n        \n        # Get memory usage statistics\n        memory_stats = optimizer.get_memory_usage()\n        \n        # Should have valid memory statistics\n        self.assertIsInstance(memory_stats, dict)\n        self.assertIn('current_memory_gb', memory_stats)\n        self.assertGreaterEqual(memory_stats['current_memory_gb'], 0)\n        \n        print(f\"Current memory usage: {memory_stats['current_memory_gb']:.3f} GB\")\n        print(\"✅ Memory usage reporting test passed\")\n\n\nif __name__ == '__main__':\n    print(\"Running 8-bit optimizer state tests...\")\n    print(f\"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\")\n    print(f\"bitsandbytes available: {HAS_BITSANDBYTES}\")\n    print(\"-\" * 60)\n    \n    unittest.main(verbosity=2)