"""
Enhanced MANGO Optimizer with MANGO-LRQ, TinyFormer Policy, and AMP Support

Integrates all the enhanced components:
- MANGO-LRQ hybrid low-rank + quantized compression
- TinyFormer policy network for better long-range modeling
- PyTorch 2.5 memory profiling
- Automatic mixed precision support
- Variance-reduced error compensation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import math
from collections import defaultdict
import warnings

from .compression import GradientCompressor
from .mango_lrq import MangoLRQCompressor, CompressionConfig
from .tinyformer_policy import TinyFormerPolicyNet, TinyFormerConfig
from .policy_network import CompressionPolicyNet, FixedCompressionPolicy, AdaptiveCompressionPolicy
from .statistics import GradientStatistics
from .utils import MemoryProfiler
from .memory_profiler import MANGOMemoryProfiler, create_memory_profiler
from .amp_support import AMPTrainingContext, MANGOGradScaler, create_amp_context
from .ef21_buffer import EF21Buffer, EF21Compressor
# from .power_monitor import PowerMonitor, MultiObjectiveReward  # Temporary disable due to syntax issues


class EnhancedMANGO(torch.optim.Optimizer):
    """
    Enhanced MANGO (Memory-Adaptive Neural Gradient Optimizer)
    
    Features all improvements from planv2.md:
    - MANGO-LRQ hybrid compression (low-rank + NF4 quantization)
    - TinyFormer policy network for long-range trend capture
    - Variance-reduced error compensation
    - PyTorch 2.5 memory profiling
    - Automatic mixed precision support
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
        use_mango_lrq: bool = True,
        use_tinyformer: bool = True,
        compression_config: Optional[CompressionConfig] = None,
        error_feedback: bool = True,
        policy_update_freq: int = 100,
        enable_amp: bool = True,
        enable_profiling: bool = True,
        profiler_output_dir: str = "./profiler_logs"
    ):
        """
        Initialize Enhanced MANGO optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0)
            memory_budget: Total memory budget in bytes (default: None, auto-detect)
            policy_net: Compression policy network (default: None, create TinyFormer)
            use_mango_lrq: Whether to use MANGO-LRQ compression (default: True)
            use_tinyformer: Whether to use TinyFormer policy network (default: True)
            compression_config: MANGO-LRQ configuration (default: None, use defaults)
            error_feedback: Whether to use error feedback mechanism (default: True)
            policy_update_freq: Frequency to update compression policy (default: 100)
            enable_amp: Whether to enable automatic mixed precision (default: True)
            enable_profiling: Whether to enable memory profiling (default: True)
            profiler_output_dir: Directory for profiling outputs (default: "./profiler_logs")
        """
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
        
        # Configuration
        self.memory_budget = memory_budget or self._estimate_memory_budget()
        self.policy_update_freq = policy_update_freq
        self.use_mango_lrq = use_mango_lrq
        self.use_tinyformer = use_tinyformer
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.enable_profiling = enable_profiling
        
        # Initialize compression components
        self.compression_config = compression_config or CompressionConfig(
            rank=4,
            bits_P=8,
            bits_Q=8,
            momentum_precision="fp16",
            use_nf4=True,
            error_feedback=error_feedback,
            variance_reduction=True,
            reference_steps=10
        )
        
        if self.use_mango_lrq:
            self.compressor = MangoLRQCompressor(self.compression_config)
            # Add EF21 buffer for enhanced variance reduction
            self.ef21_buffer = EF21Buffer(
                momentum_factor=0.9,
                variance_reduction=self.compression_config.variance_reduction,
                adaptive_compression=True,
                reference_steps=self.compression_config.reference_steps
            )
        else:
            self.compressor = GradientCompressor(error_feedback=error_feedback)
            self.ef21_buffer = None
        
        # Initialize policy network
        if policy_net is not None:
            self.policy_net = policy_net
        elif self.use_tinyformer:
            config = TinyFormerConfig(
                feature_dim=64,
                d_model=128,
                num_layers=6,
                num_heads=8,
                d_ff=256,
                max_seq_len=200,
                num_parameter_groups=len(self.param_groups),
                dropout=0.1
            )
            self.policy_net = TinyFormerPolicyNet(
                feature_dim=config.feature_dim,
                d_model=config.d_model,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                num_parameter_groups=config.num_parameter_groups,
                dropout=config.dropout
            )
        else:
            # Fallback to adaptive policy
            self.policy_net = AdaptiveCompressionPolicy()
        
        # Initialize other components
        self.statistics = GradientStatistics()
        
        # Memory profiling
        if self.enable_profiling:
            self.memory_profiler = create_memory_profiler(profiler_output_dir)
        else:
            self.memory_profiler = None
        
        # Power monitoring (optional)
        self.power_monitor = None
        self.multi_objective_reward = None
        
        # AMP support
        if self.enable_amp:
            self.amp_context = None  # Will be set externally
            self.grad_scaler = MANGOGradScaler(enabled=True)
        else:
            self.amp_context = None
            self.grad_scaler = None
        
        # State tracking
        self.step_count = 0
        self.error_buffers = {}
        self.compression_history = []
        self.current_compression_config = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_steps': 0,
            'compression_time': 0.0,
            'memory_savings': [],
            'convergence_metrics': []
        }
        
        # Initialize parameter tracking
        self._initialize_parameter_tracking()
    
    def _estimate_memory_budget(self) -> int:
        """Estimate available memory budget based on current GPU memory."""
        if torch.cuda.is_available():
            # Get 70% of available GPU memory as budget
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - used_memory
            return int(0.7 * available_memory)
        else:
            # Conservative estimate for CPU: 4GB
            return 4 * 1024**3
    
    def _initialize_parameter_tracking(self):
        """Initialize parameter tracking for compression."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_id = id(p)
                    if self.compression_config.error_feedback:
                        self.error_buffers[param_id] = torch.zeros_like(p.data)
    
    def set_amp_context(self, amp_context: AMPTrainingContext):
        """Set AMP context for mixed precision training."""
        self.amp_context = amp_context
        self.enable_amp = amp_context.is_enabled()
    
    # Power monitoring methods temporarily disabled
    # def set_power_monitor(self, power_monitor: PowerMonitor):
    #     """Set power monitor for energy-aware optimization."""
    #     self.power_monitor = power_monitor
    #     self.multi_objective_reward = MultiObjectiveReward(
    #         power_monitor=power_monitor,
    #         loss_weight=1.0,
    #         memory_weight=0.1,
    #         energy_weight=0.05,
    #         time_weight=0.02
    #     )
    # 
    # def compute_multi_objective_reward(
    #     self, 
    #     loss: float,
    #     training_time: float
    # ) -> Tuple[float, Dict[str, float]]:
    #     """Compute multi-objective reward for policy training."""
    #     if self.multi_objective_reward is None:
    #         return 0.0, {}
    #     
    #     memory_usage = self.get_memory_usage().get('current_memory_gb', 0.0)
    #     return self.multi_objective_reward.compute_reward(loss, memory_usage, training_time)
    
    def collect_gradient_features(self) -> Dict[str, torch.Tensor]:
        """
        Collect comprehensive gradient statistics for policy network input.
        
        Enhanced version with additional features for TinyFormer.
        """
        features = {}
        
        for group_idx, group in enumerate(self.param_groups):
            group_features = {
                'grad_norms': [],
                'grad_variances': [],
                'momentum_alignments': [],
                'param_counts': [],
                'layer_depths': [],
                'grad_magnitudes': [],
                'grad_sparsities': [],
                'hessian_estimates': []
            }
            
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                param_id = id(p)
                grad = p.grad.data
                
                # Basic gradient statistics
                grad_norm = grad.norm().item()
                grad_magnitude = grad.abs().mean().item()
                grad_sparsity = (grad.abs() < 1e-6).float().mean().item()
                
                group_features['grad_norms'].append(grad_norm)
                group_features['grad_magnitudes'].append(grad_magnitude)
                group_features['grad_sparsities'].append(grad_sparsity)
                
                # Gradient variance (temporal)
                grad_var = self.statistics.get_gradient_variance(param_id, grad)
                group_features['grad_variances'].append(grad_var)
                
                # Momentum alignment
                state = self.state[p]
                if 'exp_avg' in state:
                    momentum = state['exp_avg']
                    if momentum.numel() == grad.numel():
                        alignment = torch.cosine_similarity(
                            grad.flatten(), momentum.flatten(), dim=0
                        ).item()
                    else:
                        alignment = 0.0
                else:
                    alignment = 0.0
                group_features['momentum_alignments'].append(alignment)
                
                # Parameter information
                group_features['param_counts'].append(p.numel())
                group_features['layer_depths'].append(param_idx)
                
                # Hessian diagonal estimate (using gradient difference)
                hessian_est = self.statistics.get_hessian_estimate(param_id, grad)
                group_features['hessian_estimates'].append(hessian_est)
            
            # Convert to tensors
            for key, values in group_features.items():
                if values:
                    features[f'group_{group_idx}_{key}'] = torch.tensor(values, dtype=torch.float32)
                else:
                    features[f'group_{group_idx}_{key}'] = torch.tensor([0.0], dtype=torch.float32)
        
        # Global features
        features['step_ratio'] = torch.tensor([min(1.0, self.step_count / 10000)], dtype=torch.float32)
        
        if self.memory_profiler and self.enable_profiling:
            memory_usage = self.memory_profiler.get_current_gpu_memory_gb() / 24.0  # Normalize by typical GPU memory
            features['memory_usage'] = torch.tensor([memory_usage], dtype=torch.float32)
        else:
            features['memory_usage'] = torch.tensor([0.5], dtype=torch.float32)
        
        # Training phase indicators
        features['training_phase'] = torch.tensor([self._get_training_phase()], dtype=torch.float32)
        features['loss_plateau'] = torch.tensor([self._detect_loss_plateau()], dtype=torch.float32)
        
        return features
    
    def _get_training_phase(self) -> float:
        """Estimate current training phase (0=early, 1=late)."""
        if self.step_count < 100:
            return 0.0
        elif self.step_count > 5000:
            return 1.0
        else:
            return self.step_count / 5000.0
    
    def _detect_loss_plateau(self) -> float:
        """Detect if training is in a loss plateau."""
        if len(self.compression_history) < 10:
            return 0.0
        
        # Simple plateau detection based on compression error stability
        recent_errors = [h.get('compression_error', 0.0) for h in self.compression_history[-10:]]
        if len(recent_errors) >= 2:
            error_std = torch.tensor(recent_errors).std().item()
            return 1.0 if error_std < 1e-6 else 0.0
        return 0.0
    
    def get_compression_config(self) -> Dict[str, Any]:
        """Get compression configuration from policy network."""
        if self.policy_net is not None and self.step_count % self.policy_update_freq == 0:
            # Use learned policy
            features = self.collect_gradient_features()
            
            with torch.no_grad():
                if hasattr(self.policy_net, 'forward'):
                    config = self.policy_net(features)
                else:
                    config = self.policy_net(features)  # For legacy policies
            
            self.current_compression_config = config
        elif not self.current_compression_config:
            # Initialize with default adaptive policy
            self.current_compression_config = self._get_default_config()
        
        return self.current_compression_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default compression configuration."""
        if self.use_mango_lrq:
            # MANGO-LRQ configuration
            progress = min(1.0, self.step_count / 1000)
            if progress < 0.1:
                # Early training: higher rank, more bits
                return {
                    'rank': 8,
                    'bits_p': 16,
                    'bits_q': 16,
                    'momentum_precision': 'fp32',
                    'use_nf4': False
                }
            elif progress < 0.8:
                # Mid training: moderate compression
                return {
                    'rank': 4,
                    'bits_p': 8,
                    'bits_q': 8,
                    'momentum_precision': 'fp16',
                    'use_nf4': True
                }
            else:
                # Late training: aggressive compression
                return {
                    'rank': 2,
                    'bits_p': 4,
                    'bits_q': 4,
                    'momentum_precision': 'nf4',
                    'use_nf4': True
                }
        else:
            # Legacy configuration
            progress = min(1.0, self.step_count / 1000)
            if progress < 0.1:
                return {'gradient_bits': 32, 'momentum_bits': 32, 'sparsity_ratio': 0.0}
            elif progress < 0.8:
                return {'gradient_bits': 16, 'momentum_bits': 16, 'sparsity_ratio': 0.1}
            else:
                return {'gradient_bits': 8, 'momentum_bits': 16, 'sparsity_ratio': 0.3}
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step with compression.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Profile memory usage
        profiler_context = None
        if self.memory_profiler and self.enable_profiling:
            profiler_context = self.memory_profiler.profile_step(f"step_{self.step_count}")
        
        with profiler_context if profiler_context else torch.no_grad():
            # Get compression configuration
            compression_config = self.get_compression_config()
            
            # Compress gradients and update parameters
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    param_id = id(p)
                    
                    # Profile gradient compression
                    if self.memory_profiler and self.enable_profiling:
                        with self.memory_profiler.profile_compression_operation("gradient_compression"):
                            compressed_grad, metadata = self._compress_gradient(p, param_id, compression_config)
                    else:
                        compressed_grad, metadata = self._compress_gradient(p, param_id, compression_config)
                    
                    # Update parameter using compressed gradient
                    self._update_parameter(p, compressed_grad, group)
                    
                    # Store compression statistics
                    self._update_compression_stats(p.grad, compressed_grad, metadata)
        
        self.step_count += 1
        self.performance_stats['total_steps'] += 1
        
        return loss
    
    def _compress_gradient(
        self, 
        param: torch.Tensor, 
        param_id: int, 
        compression_config: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress gradient using MANGO-LRQ or legacy compression.
        
        Args:
            param: Parameter tensor
            param_id: Parameter identifier
            compression_config: Compression configuration
            
        Returns:
            Tuple of (compressed_gradient, compression_metadata)
        """
        if self.use_mango_lrq and hasattr(self.compressor, 'compress'):
            # Use MANGO-LRQ compression
            # Update compression config if needed
            if any(key in compression_config for key in ['rank', 'bits_p', 'bits_q']):
                config = CompressionConfig(
                    rank=compression_config.get('rank', self.compression_config.rank),
                    bits_P=compression_config.get('bits_p', self.compression_config.bits_P),
                    bits_Q=compression_config.get('bits_q', self.compression_config.bits_Q),
                    momentum_precision=compression_config.get('momentum_precision', self.compression_config.momentum_precision),
                    use_nf4=compression_config.get('use_nf4', self.compression_config.use_nf4),
                    error_feedback=self.compression_config.error_feedback,
                    variance_reduction=self.compression_config.variance_reduction,
                    reference_steps=self.compression_config.reference_steps
                )
            else:
                config = None
            
            compressed_grad, metadata = self.compressor.compress(
                param.grad.data, param_id, config
            )
        else:
            # Use legacy compression
            bits = compression_config.get('gradient_bits', 32)
            sparsity = compression_config.get('sparsity_ratio', 0.0)
            
            compressed_grad, compression_error = self.compressor.compress(
                param.grad.data, bits=bits, sparsity_ratio=sparsity
            )
            
            # Apply error feedback
            if self.compression_config.error_feedback and param_id in self.error_buffers:
                self.error_buffers[param_id] = compression_error
            
            metadata = {'compression_error': compression_error}
        
        return compressed_grad, metadata
    
    def _update_parameter(
        self, 
        param: torch.Tensor, 
        compressed_grad: torch.Tensor, 
        group: Dict[str, Any]
    ):
        """
        Update parameter using Adam with compressed gradients.
        
        Args:
            param: Parameter tensor
            compressed_grad: Compressed gradient
            group: Parameter group configuration
        """
        if compressed_grad is None or compressed_grad.numel() == 0:
            return
        
        state = self.state[param]
        
        # State initialization with optional 8-bit quantization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            # 8-bit state tracking
            state['use_8bit_states'] = self.compression_config.momentum_precision in ['int8', 'nf4']
            if state['use_8bit_states']:
                state['exp_avg_scale'] = torch.tensor(1.0, device=param.device)
                state['exp_avg_sq_scale'] = torch.tensor(1.0, device=param.device)
        
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        
        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        # Weight decay
        if group['weight_decay'] != 0:
            compressed_grad = compressed_grad.add(param, alpha=group['weight_decay'])
        
        # Exponential moving average of gradient values
        exp_avg.mul_(beta1).add_(compressed_grad, alpha=1 - beta1)
        
        # Exponential moving average of squared gradient values
        exp_avg_sq.mul_(beta2).addcmul_(compressed_grad, compressed_grad, value=1 - beta2)
        
        # Apply 8-bit quantization to optimizer states if enabled
        if state.get('use_8bit_states', False) and state['step'] % 10 == 0:
            exp_avg_quantized, exp_avg_scale = self._quantize_8bit_state(exp_avg)
            exp_avg_sq_quantized, exp_avg_sq_scale = self._quantize_8bit_state(exp_avg_sq)
            
            # Store quantized states
            state['exp_avg'].copy_(exp_avg_quantized)
            state['exp_avg_sq'].copy_(exp_avg_sq_quantized)
            state['exp_avg_scale'] = exp_avg_scale
            state['exp_avg_sq_scale'] = exp_avg_sq_scale
        
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        step_size = group['lr'] / bias_correction1
        
        param.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _quantize_8bit_state(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize optimizer state to 8-bit using dynamic quantization.
        
        Args:
            tensor: State tensor to quantize
            
        Returns:
            Tuple of (quantized_tensor, scale_factor)
        """
        if tensor.numel() == 0:
            return tensor, torch.tensor(1.0, device=tensor.device)
        
        # Dynamic quantization to 8-bit range [-128, 127]
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            return tensor, torch.tensor(1.0, device=tensor.device)
        
        # Calculate scale for 8-bit quantization
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = tensor_min
        
        # Quantize to 8-bit integers
        quantized = torch.round((tensor - zero_point) / scale).clamp(-128, 127)
        
        # Dequantize back to float
        dequantized = quantized * scale + zero_point
        
        return dequantized, scale
    
    def _update_compression_stats(
        self, 
        original_grad: torch.Tensor, 
        compressed_grad: torch.Tensor, 
        metadata: Dict[str, Any]
    ):
        """Update compression statistics for monitoring."""
        if original_grad is None or compressed_grad is None:
            return
        
        # Compute compression error
        if 'compression_error' in metadata:
            compression_error = metadata['compression_error']
        else:
            compression_error = original_grad - compressed_grad
        
        error_norm = compression_error.norm().item()
        original_norm = original_grad.norm().item()
        
        compression_ratio = 1.0
        if hasattr(self.compressor, 'get_compression_ratio'):
            compression_ratio = self.compressor.get_compression_ratio()
        
        # Store statistics
        stats = {
            'step': self.step_count,
            'compression_error': error_norm,
            'original_norm': original_norm,
            'compressed_norm': compressed_grad.norm().item(),
            'compression_ratio': compression_ratio,
            'relative_error': error_norm / max(original_norm, 1e-8)
        }
        
        self.compression_history.append(stats)
        
        # Keep only recent history to manage memory
        if len(self.compression_history) > 1000:
            self.compression_history = self.compression_history[-500:]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            'current_memory_gb': 0.0,
            'peak_memory_gb': 0.0,
            'median_memory_gb': 0.0
        }
        
        if self.memory_profiler and self.enable_profiling:
            stats['current_memory_gb'] = self.memory_profiler.get_current_gpu_memory_gb()
            stats['peak_memory_gb'] = self.memory_profiler.get_peak_memory_gb()
            stats['median_memory_gb'] = self.memory_profiler.get_median_memory_gb()
        elif torch.cuda.is_available():
            stats['current_memory_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['peak_memory_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        return stats
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        if not self.compression_history:
            return {'error': 'No compression history available'}
        
        recent_history = self.compression_history[-100:] if len(self.compression_history) > 100 else self.compression_history
        
        compression_errors = [h['compression_error'] for h in recent_history]
        compression_ratios = [h['compression_ratio'] for h in recent_history]
        relative_errors = [h['relative_error'] for h in recent_history]
        
        stats = {
            'avg_compression_ratio': float(torch.tensor(compression_ratios).mean()),
            'avg_compression_error': float(torch.tensor(compression_errors).mean()),
            'avg_relative_error': float(torch.tensor(relative_errors).mean()),
            'max_compression_ratio': float(torch.tensor(compression_ratios).max()),
            'min_compression_ratio': float(torch.tensor(compression_ratios).min()),
            'total_compression_steps': len(self.compression_history),
            'current_config': self.current_compression_config
        }
        
        # Add MANGO-LRQ specific stats
        if self.use_mango_lrq and hasattr(self.compressor, 'get_statistics'):
            lrq_stats = self.compressor.get_statistics()
            stats.update({'mango_lrq_stats': lrq_stats})
        
        return stats
    
    def save_profiling_report(self, filename: str = "memory_report.json") -> Optional[str]:
        """Save memory profiling report."""
        if self.memory_profiler and self.enable_profiling:
            return self.memory_profiler.save_report(filename)
        return None
    
    def reset_statistics(self):
        """Reset all statistics and profiling data."""
        self.compression_history.clear()
        self.performance_stats = {
            'total_steps': 0,
            'compression_time': 0.0,
            'memory_savings': [],
            'convergence_metrics': []
        }
        
        if self.memory_profiler:
            self.memory_profiler.reset()
        
        if hasattr(self.compressor, 'reset_statistics'):
            self.compressor.reset_statistics()
        
        if hasattr(self.policy_net, 'reset_hidden_state'):
            self.policy_net.reset_hidden_state()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict including enhanced MANGO state."""
        state_dict = super().state_dict()
        state_dict.update({
            'step_count': self.step_count,
            'compression_history': self.compression_history[-100:],  # Keep recent history
            'current_compression_config': self.current_compression_config,
            'performance_stats': self.performance_stats,
            'use_mango_lrq': self.use_mango_lrq,
            'use_tinyformer': self.use_tinyformer
        })
        
        # Save policy network state if applicable
        if hasattr(self.policy_net, 'state_dict'):
            state_dict['policy_net_state'] = self.policy_net.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict including enhanced MANGO state."""
        self.step_count = state_dict.pop('step_count', 0)
        self.compression_history = state_dict.pop('compression_history', [])
        self.current_compression_config = state_dict.pop('current_compression_config', {})
        self.performance_stats = state_dict.pop('performance_stats', {
            'total_steps': 0, 'compression_time': 0.0, 'memory_savings': [], 'convergence_metrics': []
        })
        
        # Load configuration flags
        self.use_mango_lrq = state_dict.pop('use_mango_lrq', True)
        self.use_tinyformer = state_dict.pop('use_tinyformer', True)
        
        # Load policy network state
        policy_net_state = state_dict.pop('policy_net_state', None)
        if policy_net_state and hasattr(self.policy_net, 'load_state_dict'):
            self.policy_net.load_state_dict(policy_net_state)
        
        super().load_state_dict(state_dict)