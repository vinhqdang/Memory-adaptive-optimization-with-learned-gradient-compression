"""
Test Power Sampler Integration with MANGO Enhanced Optimizer

Tests the integration of the PowerSampler with TinyFormer policy network
for energy-aware multi-objective optimization.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.power_sampler import PowerSampler, create_power_sampler
from mango.enhanced_optimizer import EnhancedMANGO
from mango.mango_lrq import CompressionConfig
from mango.tinyformer_policy import TinyFormerPolicyNet, TinyFormerConfig


class TestPowerSamplerIntegration(unittest.TestCase):
    """Test suite for PowerSampler integration with MANGO optimizer."""
    
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
        
        # Create TinyFormer policy configuration
        self.tinyformer_config = TinyFormerConfig(
            feature_dim=64,
            d_model=128,
            num_layers=6,
            num_heads=8,
            d_ff=256,
            max_seq_len=200,
            num_parameter_groups=1,
            dropout=0.1
        )
    
    def test_power_sampler_creation(self):
        """Test that PowerSampler can be created successfully."""
        power_sampler = create_power_sampler(
            sampling_interval=1.0,
            queue_size=100,
            gpu_device_id=0,
            enable_logging=False
        )
        
        self.assertIsInstance(power_sampler, PowerSampler)
        self.assertEqual(power_sampler.sampling_interval, 1.0)
        self.assertFalse(power_sampler.is_sampling)
        
        print("✅ PowerSampler creation test passed")
    
    def test_power_sampler_integration_with_optimizer(self):
        """Test that PowerSampler can be integrated with EnhancedMANGO."""
        # Create power sampler
        power_sampler = create_power_sampler(
            sampling_interval=2.0,  # Slower for testing
            queue_size=50,
            enable_logging=False
        )
        
        # Create optimizer
        optimizer = EnhancedMANGO(
            self.model.parameters(),
            lr=1e-3,
            compression_config=self.compression_config,
            use_8bit_optimizer=False,  # Disable for simplicity
            use_mango_lrq=True,
            use_tinyformer=True
        )
        
        # Set power sampler
        optimizer.set_power_sampler(power_sampler)
        
        self.assertTrue(hasattr(optimizer, 'power_sampler'))
        self.assertEqual(optimizer.power_sampler, power_sampler)
        
        print("✅ PowerSampler integration test passed")
    
    def test_tinyformer_with_power_features(self):
        """Test that TinyFormer can process power-related features."""
        # Create TinyFormer policy network
        policy_net = TinyFormerPolicyNet(**vars(self.tinyformer_config))
        
        # Create mock features with power data
        mock_features = {
            'step_ratio': torch.tensor([0.1]),
            'memory_usage': torch.tensor([0.5]),
            'ewma_power': torch.tensor([0.7]),  # Power feature from sampler
            'power_efficiency': torch.tensor([0.8]),  # Power feature from sampler
            'gpu_utilization': torch.tensor([0.9]),  # Power feature from sampler
            'training_phase': torch.tensor([0.2]),
            'loss_plateau': torch.tensor([0.0]),
            'group_0_grad_norms': torch.tensor([1.0, 0.5, 0.8]),
            'group_0_grad_variances': torch.tensor([0.1, 0.2, 0.15]),
            'group_0_momentum_alignments': torch.tensor([0.9, 0.8, 0.85]),
            'group_0_param_counts': torch.tensor([100, 200, 50]),
            'group_0_layer_depths': torch.tensor([0, 1, 2]),
            'group_0_grad_magnitudes': torch.tensor([0.1, 0.05, 0.08]),
            'group_0_grad_sparsities': torch.tensor([0.01, 0.02, 0.015]),
            'group_0_hessian_estimates': torch.tensor([0.001, 0.002, 0.0015])
        }
        
        # Test forward pass
        try:
            output = policy_net.forward(mock_features)
            
            # Check that output contains expected keys
            expected_keys = ['rank', 'bits_p', 'bits_q', 'momentum_precision', 'use_nf4', 
                           'value', 'energy_value', 'loss_value', 'memory_value', 'time_value']
            
            for key in expected_keys:
                self.assertIn(key, output, f"Output should contain {key}")
            
            # Check that energy_value is computed
            self.assertIsInstance(output['energy_value'], float)
            
            print("✅ TinyFormer power feature processing test passed")
            
        except Exception as e:
            self.fail(f"TinyFormer forward pass failed: {e}")
    
    def test_optimizer_with_power_features(self):
        """Test complete integration: optimizer + power sampler + TinyFormer."""
        # Create power sampler
        power_sampler = create_power_sampler(enable_logging=False)
        
        # Create optimizer with TinyFormer
        optimizer = EnhancedMANGO(
            self.model.parameters(),
            lr=1e-3,
            compression_config=self.compression_config,
            use_8bit_optimizer=False,
            use_mango_lrq=True,
            use_tinyformer=True,
            enable_profiling=False  # Disable profiling for simplicity
        )
        
        # Set power sampler
        optimizer.set_power_sampler(power_sampler)
        
        # Create dummy input and target
        x = torch.randn(16, 10, device=self.device)
        y = torch.randn(16, 1, device=self.device)
        
        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Test that optimizer step works with power features
        try:
            optimizer.step()
            optimizer.zero_grad()
            
            print("✅ Complete power integration test passed")
            
        except Exception as e:
            self.fail(f"Optimizer step with power integration failed: {e}")
    
    def test_ewma_power_computation(self):
        """Test EWMA power computation in PowerSampler."""
        power_sampler = create_power_sampler(enable_logging=False)
        
        # Simulate power updates
        test_powers = [100.0, 150.0, 120.0, 180.0, 160.0]
        
        for power in test_powers:
            power_sampler._update_ewma(power)
        
        ewma_power = power_sampler.get_ewma_power()
        
        # EWMA should be some weighted average of the powers
        self.assertGreater(ewma_power, 0.0)
        self.assertLess(ewma_power, max(test_powers))
        
        print(f"✅ EWMA power computation test passed: {ewma_power:.2f}W")
    
    def test_power_statistics_computation(self):
        """Test power statistics computation in PowerSampler."""
        power_sampler = create_power_sampler(enable_logging=False)
        
        # Add some sample history manually (simulating real data)
        from utils.power_sampler import PowerSample
        import time
        
        for i in range(5):
            sample = PowerSample(
                timestamp=time.time(),
                gpu_power_watts=100.0 + i * 10,
                gpu_temperature_c=60.0 + i * 2,
                gpu_utilization_percent=70.0 + i * 5,
                memory_usage_mb=8000 + i * 100,
                memory_total_mb=12000
            )
            power_sampler.sample_history.append(sample)
            power_sampler._update_ewma(sample.gpu_power_watts)
        
        stats = power_sampler.get_power_statistics()
        
        # Check that statistics are computed correctly
        self.assertEqual(stats['samples'], 5)
        self.assertGreater(stats['mean_power_watts'], 0.0)
        self.assertGreater(stats['ewma_power_watts'], 0.0)
        self.assertGreater(stats['mean_temperature_c'], 0.0)
        
        print("✅ Power statistics computation test passed")


if __name__ == '__main__':
    print("Running PowerSampler integration tests...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("-" * 60)
    
    unittest.main(verbosity=2)