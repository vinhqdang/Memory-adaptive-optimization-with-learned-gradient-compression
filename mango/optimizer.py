"""
MANGO Optimizer Implementation

Core optimizer class that implements the Memory-Adaptive Neural Gradient Optimizer
with learned gradient compression policies.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
from collections import defaultdict

from .compression import GradientCompressor
from .statistics import GradientStatistics
from .utils import MemoryProfiler


class MANGO(torch.optim.Optimizer):
    """
    MANGO (Memory-Adaptive Neural Gradient Optimizer)
    
    Dynamically adjusts memory allocation between gradient precision and 
    optimizer states based on learned policies from training dynamics.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
        memory_budget: Total memory budget in bytes (default: None, auto-detect)
        policy_net: Compression policy network (default: None, use fixed policy)
        compression_schedule: Fixed compression schedule if no policy network (default: None)
        error_feedback: Whether to use error feedback mechanism (default: True)
        policy_update_freq: Frequency to update compression policy (default: 100)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        memory_budget: Optional[int] = None,
        policy_net: Optional[nn.Module] = None,
        compression_schedule: Optional[Dict] = None,
        error_feedback: bool = True,
        policy_update_freq: int = 100
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
        self.memory_budget = memory_budget or self._estimate_memory_budget()
        self.policy_net = policy_net
        self.compression_schedule = compression_schedule or {}
        self.error_feedback = error_feedback
        self.policy_update_freq = policy_update_freq
        
        # Initialize components
        self.compressor = GradientCompressor(error_feedback=error_feedback)
        self.statistics = GradientStatistics()
        self.memory_profiler = MemoryProfiler()
        
        # State tracking
        self.step_count = 0
        self.error_buffers = {}
        self.compression_history = []
        self.current_compression_config = {}
        
        # Initialize error buffers for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_id = id(p)
                    self.error_buffers[param_id] = torch.zeros_like(p.data)
    
    def _estimate_memory_budget(self) -> int:
        """Estimate available memory budget based on current GPU memory."""
        if torch.cuda.is_available():
            # Get 70% of available GPU memory as budget
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - used_memory
            return int(0.7 * available_memory)
        else:
            # Conservative estimate for CPU: 2GB
            return 2 * 1024**3
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict including MANGO-specific state."""
        state_dict = super().state_dict()
        state_dict.update({
            'step_count': self.step_count,
            'compression_history': self.compression_history,
            'current_compression_config': self.current_compression_config
        })
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict including MANGO-specific state."""
        self.step_count = state_dict.pop('step_count', 0)
        self.compression_history = state_dict.pop('compression_history', [])
        self.current_compression_config = state_dict.pop('current_compression_config', {})
        super().load_state_dict(state_dict)
    
    def collect_gradient_features(self) -> Dict[str, torch.Tensor]:
        """Collect gradient statistics for policy network input."""
        features = {}
        
        for group_idx, group in enumerate(self.param_groups):
            group_features = {
                'grad_norms': [],
                'grad_variances': [],
                'momentum_alignments': [],
                'param_counts': [],
                'layer_depths': []
            }
            
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                    
                param_id = id(p)
                grad = p.grad.data
                
                # Gradient norm
                grad_norm = grad.norm().item()
                group_features['grad_norms'].append(grad_norm)
                
                # Gradient variance (approximated over recent history)
                grad_var = self.statistics.get_gradient_variance(param_id, grad)
                group_features['grad_variances'].append(grad_var)
                
                # Momentum alignment
                state = self.state[p]
                if 'exp_avg' in state:
                    momentum = state['exp_avg']
                    alignment = torch.cosine_similarity(
                        grad.flatten(), momentum.flatten(), dim=0
                    ).item()
                else:
                    alignment = 0.0
                group_features['momentum_alignments'].append(alignment)
                
                # Parameter count and layer depth (heuristic)
                group_features['param_counts'].append(p.numel())
                group_features['layer_depths'].append(param_idx)
            
            # Convert to tensors
            for key, values in group_features.items():
                if values:
                    features[f'group_{group_idx}_{key}'] = torch.tensor(values, dtype=torch.float32)
                else:
                    features[f'group_{group_idx}_{key}'] = torch.tensor([0.0], dtype=torch.float32)
        
        # Global features
        features['step_ratio'] = torch.tensor([self.step_count / 10000], dtype=torch.float32)  # Normalized step
        features['memory_usage'] = torch.tensor([self.memory_profiler.get_usage_ratio()], dtype=torch.float32)
        
        return features
    
    def get_compression_config(self) -> Dict[str, Dict]:
        """Get compression configuration from policy network or schedule."""
        if self.policy_net is not None and self.step_count % self.policy_update_freq == 0:
            # Use learned policy
            features = self.collect_gradient_features()
            with torch.no_grad():
                config = self.policy_net(features)
            self.current_compression_config = config
        elif self.compression_schedule:
            # Use fixed schedule
            self.current_compression_config = self._get_scheduled_compression()
        else:
            # Default: no compression early, moderate compression later
            progress = min(1.0, self.step_count / 1000)
            if progress < 0.1:
                # Early training: full precision
                bits_g, bits_m, sparsity = 32, 32, 0.0
            elif progress < 0.8:
                # Mid training: moderate compression
                bits_g, bits_m, sparsity = 16, 16, 0.1
            else:
                # Late training: aggressive compression
                bits_g, bits_m, sparsity = 8, 16, 0.3
            
            self.current_compression_config = {
                'gradient_bits': bits_g,
                'momentum_bits': bits_m,
                'sparsity_ratio': sparsity
            }
        
        return self.current_compression_config
    
    def _get_scheduled_compression(self) -> Dict[str, Any]:
        """Get compression config from predefined schedule."""
        # Simple linear schedule based on step count
        total_steps = self.compression_schedule.get('total_steps', 10000)
        progress = min(1.0, self.step_count / total_steps)
        
        # Interpolate compression parameters
        start_config = self.compression_schedule.get('start', {'gradient_bits': 32, 'momentum_bits': 32, 'sparsity_ratio': 0.0})
        end_config = self.compression_schedule.get('end', {'gradient_bits': 8, 'momentum_bits': 16, 'sparsity_ratio': 0.5})
        
        config = {}
        for key in start_config:
            start_val = start_config[key]
            end_val = end_config[key]
            config[key] = start_val + progress * (end_val - start_val)
        
        return config
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        # Get compression configuration
        compression_config = self.get_compression_config()
        self.compression_history.append(compression_config.copy())
        
        # Process each parameter group
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                grad = p.grad
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Get parameter state
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Apply gradient compression with error feedback
                if self.error_feedback and param_id in self.error_buffers:
                    # Add accumulated error to gradient
                    grad = grad + self.error_buffers[param_id]
                
                # Compress gradient
                compressed_grad, compression_error = self.compressor.compress(
                    grad, 
                    bits=compression_config.get('gradient_bits', 32),
                    sparsity_ratio=compression_config.get('sparsity_ratio', 0.0)
                )
                
                # Update error buffer
                if self.error_feedback and param_id in self.error_buffers:
                    self.error_buffers[param_id] = compression_error
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(compressed_grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values  
                exp_avg_sq.mul_(beta2).addcmul_(compressed_grad, compressed_grad, value=1 - beta2)
                
                # Compress momentum if specified
                momentum_bits = compression_config.get('momentum_bits', 32)
                if momentum_bits < 32:
                    exp_avg_compressed, _ = self.compressor.compress(
                        exp_avg, bits=momentum_bits, sparsity_ratio=0.0
                    )
                    exp_avg.copy_(exp_avg_compressed)
                
                # Compute bias-corrected first and second moment estimates
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Update parameters
                denom = (corrected_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                p.addcdiv_(corrected_exp_avg, denom, value=-step_size)
        
        # Update statistics
        self.statistics.update(self.step_count, compression_config, self.param_groups)
        
        return loss
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage breakdown."""
        return self.memory_profiler.get_detailed_usage()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get statistics about compression efficiency."""
        if not self.compression_history:
            return {}
        
        recent_configs = self.compression_history[-100:]  # Last 100 steps
        
        stats = {
            'avg_gradient_bits': sum(c.get('gradient_bits', 32) for c in recent_configs) / len(recent_configs),
            'avg_momentum_bits': sum(c.get('momentum_bits', 32) for c in recent_configs) / len(recent_configs),
            'avg_sparsity_ratio': sum(c.get('sparsity_ratio', 0.0) for c in recent_configs) / len(recent_configs),
            'total_steps': self.step_count,
            'compression_changes': len(set(str(c) for c in recent_configs))
        }
        
        return stats