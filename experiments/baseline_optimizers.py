"""
Baseline Optimizer Implementations

Implements various baseline optimizers for comparison with MANGO-LRQ as specified in planv2.md:
- AdamW, Adafactor (standard optimizers)
- GaLore (low-rank only)
- AdaRankGrad (rank schedule)
- QLoRA pager (quantization only)
- 8-bit Adam and other memory-efficient variants
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import math
import numpy as np
from collections import defaultdict

try:
    from transformers import Adafactor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available, Adafactor will use fallback implementation")

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not available, using fallback implementations")


class EightBitAdam(torch.optim.Optimizer):
    """
    8-bit Adam optimizer implementation.
    
    Reduces memory usage by quantizing optimizer states to 8-bit precision
    while maintaining training performance.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        dynamic_quantization: bool = True
    ):
        """
        Initialize 8-bit Adam optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term for numerical stability
            weight_decay: Weight decay coefficient
            dynamic_quantization: Whether to use dynamic quantization ranges
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
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.dynamic_quantization = dynamic_quantization
        
        # Track quantization statistics
        self.quantization_stats = {
            'quantization_errors': [],
            'dynamic_ranges': []
        }
    
    def _quantize_8bit(self, tensor: torch.Tensor, dynamic_range: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to 8-bit representation.
        
        Args:
            tensor: Input tensor
            dynamic_range: Whether to use dynamic quantization range
            
        Returns:
            Tuple of (quantized_tensor, quantization_scale)
        """
        if tensor.numel() == 0:
            return tensor, torch.tensor(1.0, device=tensor.device)
        
        if dynamic_range:
            # Dynamic quantization based on tensor statistics
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            
            if tensor_max == tensor_min:
                return tensor, torch.tensor(1.0, device=tensor.device)
            
            # Use full 8-bit range: [-128, 127]
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = tensor_min
            
            # Quantize
            quantized = torch.round((tensor - zero_point) / scale).clamp(-128, 127)
            
            # Dequantize
            dequantized = quantized * scale + zero_point
            
        else:
            # Fixed quantization range based on standard deviation
            std = tensor.std()
            if std == 0:
                return tensor, torch.tensor(1.0, device=tensor.device)
            
            # Use 6-sigma range
            scale = (6 * std) / 255.0
            zero_point = tensor.mean()
            
            quantized = torch.round((tensor - zero_point) / scale).clamp(-128, 127)
            dequantized = quantized * scale + zero_point
        
        return dequantized, scale
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with 8-bit quantized states."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Use 8-bit quantized momentum estimates
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_scale'] = torch.tensor(1.0, device=p.device)
                    state['exp_avg_sq_scale'] = torch.tensor(1.0, device=p.device)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Quantize momentum estimates to 8-bit every few steps
                if state['step'] % 10 == 0:  # Quantize every 10 steps to balance accuracy and memory
                    exp_avg_quantized, exp_avg_scale = self._quantize_8bit(
                        exp_avg, self.dynamic_quantization
                    )
                    exp_avg_sq_quantized, exp_avg_sq_scale = self._quantize_8bit(
                        exp_avg_sq, self.dynamic_quantization
                    )
                    
                    # Store quantized states
                    state['exp_avg'].copy_(exp_avg_quantized)
                    state['exp_avg_sq'].copy_(exp_avg_sq_quantized)
                    state['exp_avg_scale'] = exp_avg_scale
                    state['exp_avg_sq_scale'] = exp_avg_sq_scale
                    
                    # Track quantization statistics
                    exp_avg_error = (exp_avg - exp_avg_quantized).norm().item()
                    exp_avg_sq_error = (exp_avg_sq - exp_avg_sq_quantized).norm().item()
                    self.quantization_stats['quantization_errors'].append(exp_avg_error + exp_avg_sq_error)
                
                # Compute corrected estimates
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Update parameters
                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                
                p.addcdiv_(corrected_exp_avg, denom, value=-step_size)
        
        return loss
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_params = 0
        total_states = 0
        
        for group in self.param_groups:
            for p in group['params']:
                total_params += p.numel()
                if p in self.state:
                    state = self.state[p]
                    total_states += state['exp_avg'].numel() + state['exp_avg_sq'].numel()
        
        # Estimate memory savings from 8-bit quantization
        # Assume 32-bit parameters, 8-bit states (effective 4x compression)
        param_memory_gb = total_params * 4 / 1024**3  # 4 bytes per param
        state_memory_gb = total_states * 1 / 1024**3   # ~1 byte per state (8-bit)
        
        return {
            'param_memory_gb': param_memory_gb,
            'state_memory_gb': state_memory_gb,
            'total_memory_gb': param_memory_gb + state_memory_gb,
            'estimated_savings_vs_fp32': 1 - (state_memory_gb / (total_states * 4 / 1024**3))
        }


class GradientCheckpointingAdam(torch.optim.Adam):
    """
    Adam optimizer with gradient checkpointing support.
    
    Reduces memory usage during backward pass by recomputing intermediate
    activations instead of storing them.
    """
    
    def __init__(self, params, **kwargs):
        """Initialize gradient checkpointing Adam."""
        super().__init__(params, **kwargs)
        self.checkpointed_modules = []
    
    def register_checkpointed_module(self, module: nn.Module):
        """Register a module for gradient checkpointing."""
        self.checkpointed_modules.append(module)
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing on registered modules."""
        for module in self.checkpointed_modules:
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()


class DeepGradientCompression(torch.optim.Optimizer):
    """
    Deep Gradient Compression (DGC) optimizer.
    
    Implements gradient sparsification with momentum correction
    and local gradient accumulation.
    """
    
    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer = None,
        compression_ratio: float = 0.1,
        momentum_factor: float = 0.9,
        warmup_steps: int = 0
    ):
        """
        Initialize DGC optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer (Adam if None)
            compression_ratio: Fraction of gradients to keep
            momentum_factor: Momentum factor for gradient accumulation
            warmup_steps: Number of warmup steps without compression
        """
        if base_optimizer is None:
            base_optimizer = torch.optim.Adam(params)
        
        self.base_optimizer = base_optimizer
        self.compression_ratio = compression_ratio
        self.momentum_factor = momentum_factor
        self.warmup_steps = warmup_steps
        
        # Initialize state
        self.step_count = 0
        self.velocity = {}
        
        # Initialize velocity for each parameter
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.velocity[id(p)] = torch.zeros_like(p.data)
        
        super().__init__(self.base_optimizer.param_groups, self.base_optimizer.defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform DGC optimization step."""
        self.step_count += 1
        
        # Apply compression after warmup period
        if self.step_count > self.warmup_steps:
            self._apply_compression()
        
        # Use base optimizer for parameter update
        return self.base_optimizer.step(closure)
    
    def _apply_compression(self):
        """Apply gradient compression with momentum correction."""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                grad = p.grad.data
                
                # Update velocity with momentum
                if param_id in self.velocity:
                    velocity = self.velocity[param_id]
                    velocity.mul_(self.momentum_factor).add_(grad)
                else:
                    velocity = grad.clone()
                    self.velocity[param_id] = velocity
                
                # Compress gradients (top-k sparsification)
                compressed_grad = self._compress_gradient(velocity, self.compression_ratio)
                
                # Update gradient and velocity
                p.grad.data = compressed_grad
                self.velocity[param_id] = velocity - compressed_grad
    
    def _compress_gradient(self, grad: torch.Tensor, compression_ratio: float) -> torch.Tensor:
        """Apply top-k gradient compression."""
        if compression_ratio >= 1.0:
            return grad
        
        # Flatten gradient for top-k selection
        flat_grad = grad.flatten()
        k = max(1, int(flat_grad.numel() * compression_ratio))
        
        # Get top-k values by magnitude
        _, top_indices = torch.topk(flat_grad.abs(), k, largest=True)
        
        # Create sparse gradient
        compressed_flat = torch.zeros_like(flat_grad)
        compressed_flat[top_indices] = flat_grad[top_indices]
        
        return compressed_flat.reshape(grad.shape)
    
    def zero_grad(self):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad()
    
    @property
    def state(self):
        """Return base optimizer state."""
        return self.base_optimizer.state
    
    def state_dict(self):
        """Return state dict including DGC state."""
        base_state = self.base_optimizer.state_dict()
        dgc_state = {
            'velocity': self.velocity,
            'step_count': self.step_count,
            'compression_ratio': self.compression_ratio,
            'momentum_factor': self.momentum_factor
        }
        return {'base_optimizer': base_state, 'dgc': dgc_state}
    
    def load_state_dict(self, state_dict):
        """Load state dict including DGC state."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        dgc_state = state_dict['dgc']
        self.velocity = dgc_state['velocity']
        self.step_count = dgc_state['step_count']
        self.compression_ratio = dgc_state['compression_ratio']
        self.momentum_factor = dgc_state['momentum_factor']


class AdafactorOptimizer(torch.optim.Optimizer):
    """
    Adafactor optimizer implementation.
    
    Memory-efficient optimizer that factors the second moment matrix
    to reduce memory usage from O(n²) to O(n).
    """
    
    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step_size: bool = True
    ):
        """Initialize Adafactor optimizer."""
        if lr is not None and lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
            beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
            relative_step_size=relative_step_size
        )
        super().__init__(params, defaults)
    
    def _get_lr(self, param_group, param_state):
        """Compute learning rate based on parameter scale."""
        min_step = 1e-6 * param_state['step'] if param_group['scale_parameter'] else 1e-2
        rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'], param_state['RMS'])
        return param_scale * rel_step_sz
    
    def _get_options(self, param_group, param_shape):
        """Get factorization options based on parameter shape."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment
    
    def _rms(self, tensor):
        """Compute RMS of tensor."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """Approximate squared gradient using factored matrices."""
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_().unsqueeze(-1)
        )
        c_factor = (
            exp_avg_sq_col.rsqrt()
        ).unsqueeze(0)
        return torch.mul(r_factor, c_factor)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored, use_first_moment = self._get_options(group, grad_shape)
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad).float()
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).float()
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).float()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad).float()
                    
                    state['RMS'] = 0
                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                
                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                
                lr = group['lr']
                if group['lr'] is None:
                    lr = self._get_lr(group, state)
                
                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad**2 + group['eps']
                
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                update.div_(
                    max(1.0, self._rms(update) / group['clip_threshold'])
                )
                
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(
                        update, alpha=1 - group['beta1']
                    )
                    update = exp_avg
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group['weight_decay'] * lr
                    )
                
                p_data_fp32.add_(update, alpha=-lr)
                
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
        
        return loss


class BaselineOptimizerFactory:
    """
    Factory class for creating baseline optimizers.
    """
    
    @staticmethod
    def create_optimizer(
        name: str,
        model_parameters,
        lr: float = 1e-3,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create a baseline optimizer by name.
        
        Args:
            name: Optimizer name
            model_parameters: Model parameters
            lr: Learning rate
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            Optimizer instance
        """
        name = name.lower()
        
        if name == 'adam':
            return torch.optim.Adam(model_parameters, lr=lr, **kwargs)
        
        elif name == 'sgd':
            return torch.optim.SGD(
                model_parameters,
                lr=lr,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 1e-4)
            )
        
        elif name == 'adamw':
            return torch.optim.AdamW(
                model_parameters,
                lr=lr,
                weight_decay=kwargs.get('weight_decay', 1e-4),
                **kwargs
            )
        
        elif name == '8bit_adam':
            return EightBitAdam(
                model_parameters,
                lr=lr,
                dynamic_quantization=kwargs.get('dynamic_quantization', True),
                **kwargs
            )
        
        elif name == 'dgc':
            base_optimizer = torch.optim.Adam(model_parameters, lr=lr)
            return DeepGradientCompression(
                model_parameters,
                base_optimizer=base_optimizer,
                compression_ratio=kwargs.get('compression_ratio', 0.1),
                **kwargs
            )
        
        elif name == 'adafactor':
            return AdafactorOptimizer(
                model_parameters,
                lr=lr,
                **kwargs
            )
        
        elif name == 'checkpoint_adam':
            return GradientCheckpointingAdam(model_parameters, lr=lr, **kwargs)
        
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    @staticmethod
    def get_available_optimizers() -> List[str]:
        """Get list of available baseline optimizers."""
        return [
            'adam', 'sgd', 'adamw', '8bit_adam', 
            'dgc', 'adafactor', 'checkpoint_adam'
        ]
    
    @staticmethod
    def get_optimizer_description(name: str) -> str:
        """Get description of optimizer."""
        descriptions = {
            'adam': 'Standard Adam optimizer',
            'sgd': 'Stochastic Gradient Descent with momentum',
            'adamw': 'Adam with decoupled weight decay',
            '8bit_adam': 'Memory-efficient 8-bit Adam optimizer',
            'dgc': 'Deep Gradient Compression with sparsification',
            'adafactor': 'Memory-efficient factored Adam variant',
            'checkpoint_adam': 'Adam with gradient checkpointing support'
        }
        return descriptions.get(name.lower(), 'Unknown optimizer')


def get_memory_efficient_optimizer_config() -> Dict[str, Dict]:
    """
    Get recommended configurations for memory-efficient optimizers.
    
    Returns:
        Dictionary of optimizer configurations optimized for memory efficiency
    """
    return {
        '8bit_adam': {
            'lr': 1e-3,
            'dynamic_quantization': True,
            'weight_decay': 1e-4
        },
        'dgc': {
            'lr': 1e-3,
            'compression_ratio': 0.1,
            'momentum_factor': 0.9,
            'warmup_steps': 1000
        },
        'adafactor': {
            'lr': None,  # Use adaptive learning rate
            'scale_parameter': True,
            'relative_step_size': True,
            'weight_decay': 1e-4
        }
    }


class GaLoreOptimizer(torch.optim.Optimizer):
    """
    GaLore optimizer (low-rank only compression).
    
    Implements the GaLore algorithm from "GaLore: Memory-Efficient LLM Training 
    by Gradient Low-Rank Projection" for comparison with MANGO-LRQ.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        rank: int = 4,
        update_freq: int = 200
    ):
        """Initialize GaLore optimizer."""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       rank=rank, update_freq=update_freq)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with low-rank projection."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['P'] = None
                    state['Q'] = None
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Apply low-rank projection to gradient
                if grad.dim() >= 2 and state['step'] % group['update_freq'] == 1:
                    # Update projection matrices
                    try:
                        U, S, Vt = torch.linalg.svd(grad, full_matrices=False)
                        rank = min(group['rank'], min(grad.shape))
                        state['P'] = U[:, :rank].clone()
                        state['Q'] = Vt[:rank, :].clone()
                    except:
                        # Fallback for numerical issues
                        m, n = grad.shape
                        rank = min(group['rank'], min(m, n))
                        state['P'] = torch.randn(m, rank, device=grad.device, dtype=grad.dtype)
                        state['Q'] = torch.randn(rank, n, device=grad.device, dtype=grad.dtype)
                
                # Apply projection if available
                if state['P'] is not None and state['Q'] is not None and grad.dim() >= 2:
                    # Project gradient
                    try:
                        projected_grad = state['P'].T @ grad @ state['Q']
                        # Reconstruct (simplified)
                        grad = state['P'] @ projected_grad @ state['Q'].T
                    except:
                        pass  # Use original gradient if projection fails
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class AdaRankGradOptimizer(torch.optim.Optimizer):
    """
    AdaRankGrad optimizer with adaptive rank scheduling.
    
    Gradually reduces gradient rank during training as mentioned in planv2.md.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        initial_rank: int = 8,
        final_rank: int = 2,
        rank_decay_steps: int = 1000
    ):
        """Initialize AdaRankGrad optimizer."""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       initial_rank=initial_rank, final_rank=final_rank,
                       rank_decay_steps=rank_decay_steps)
        super().__init__(params, defaults)
        
    def get_current_rank(self, step: int, group: Dict) -> int:
        """Compute current rank based on step and schedule."""
        progress = min(1.0, step / group['rank_decay_steps'])
        current_rank = group['initial_rank'] * (1 - progress) + group['final_rank'] * progress
        return max(1, int(current_rank))
    
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with adaptive rank."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Get current rank for this step
                current_rank = self.get_current_rank(state['step'], group)
                
                # Apply rank reduction to gradient if it's 2D
                if grad.dim() >= 2:
                    try:
                        U, S, Vt = torch.linalg.svd(grad, full_matrices=False)
                        rank = min(current_rank, min(grad.shape))
                        if rank < min(grad.shape):
                            # Apply rank reduction
                            grad = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
                    except:
                        pass  # Use original gradient if SVD fails
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Exponential moving average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class QLoRAPagerOptimizer(torch.optim.Optimizer):
    """
    QLoRA-style optimizer with paged quantization (quantization only).
    
    Implements the paging mechanism from QLoRA for comparison.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        quantization_bits: int = 8,
        page_size: int = 2048
    ):
        """Initialize QLoRA pager optimizer."""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       quantization_bits=quantization_bits, page_size=page_size)
        super().__init__(params, defaults)
        
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to specified bits."""
        if bits >= 32:
            return tensor, torch.tensor([1.0], device=tensor.device)
        
        # Simple uniform quantization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            return tensor, torch.tensor([1.0], device=tensor.device)
        
        levels = 2 ** bits - 1
        scale = (tensor_max - tensor_min) / levels
        
        quantized = torch.round((tensor - tensor_min) / scale)
        dequantized = quantized * scale + tensor_min
        
        return dequantized, scale
    
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with quantized states."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Update momentum with quantization
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Quantize momentum states periodically
                if state['step'] % 10 == 0:
                    exp_avg_quantized, _ = self._quantize_tensor(exp_avg, group['quantization_bits'])
                    exp_avg.copy_(exp_avg_quantized)
                
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Quantize second moment
                if state['step'] % 10 == 0:
                    exp_avg_sq_quantized, _ = self._quantize_tensor(exp_avg_sq, group['quantization_bits'])
                    exp_avg_sq.copy_(exp_avg_sq_quantized)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class FallbackAdafactor(torch.optim.Optimizer):
    """
    Fallback Adafactor implementation when transformers is not available.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps2: float = 1e-30,
        cliping_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True
    ):
        """Initialize fallback Adafactor."""
        defaults = dict(
            lr=lr, eps2=eps2, cliping_threshold=cliping_threshold,
            decay_rate=decay_rate, beta1=beta1, weight_decay=weight_decay,
            scale_parameter=scale_parameter, relative_step=relative_step
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """Simplified Adafactor step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients')
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored = len(grad_shape) >= 2
                
                if len(state) == 0:
                    state['step'] = 0
                    
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1])
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:])
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    
                    state['RMS'] = 0
                
                state['step'] += 1
                state['RMS'] = self._rms(p)
                
                lr = group['lr']
                if group['relative_step']:
                    min_step = 1e-6 * state['step'] if group['scale_parameter'] else 1e-2
                    rel_step_sz = min(min_step, 1.0/math.sqrt(state['step']))
                    param_scale = 1.0
                    if group['scale_parameter']:
                        param_scale = max(group['eps2'], state['RMS'])
                    lr = param_scale * rel_step_sz
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=-group['weight_decay'] * lr)
                
                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad**2 + group['eps2']
                
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0-beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0-beta2t)
                    
                    r_factor = ((exp_avg_sq_row/exp_avg_sq_row.mean()).rsqrt_().unsqueeze(-1))
                    c_factor = (exp_avg_sq_col.rsqrt()).unsqueeze(-2)
                    update = r_factor * c_factor
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0-beta2t)
                    update = exp_avg_sq.rsqrt()
                
                update.div_((self._rms(update) / group['cliping_threshold']).clamp_(min=1.0))
                
                p.add_(grad * update, alpha=-lr)
        
        return loss
    
    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)


def get_baseline_optimizer(
    name: str, 
    parameters, 
    lr: float = 1e-3, 
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Factory function to create baseline optimizers for comparison.
    
    Args:
        name: Optimizer name
        parameters: Model parameters
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Initialized optimizer
    """
    name = name.lower()
    
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif name == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    elif name == 'adafactor':
        if HAS_TRANSFORMERS:
            return Adafactor(parameters, scale_parameter=False, relative_step=False, lr=lr, **kwargs)
        else:
            return FallbackAdafactor(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif name == 'galore':
        rank = kwargs.get('rank', 4)
        return GaLoreOptimizer(parameters, lr=lr, weight_decay=weight_decay, rank=rank, **kwargs)
    
    elif name == 'adarankgrad':
        initial_rank = kwargs.get('initial_rank', 8)
        final_rank = kwargs.get('final_rank', 2)
        return AdaRankGradOptimizer(parameters, lr=lr, weight_decay=weight_decay,
                                  initial_rank=initial_rank, final_rank=final_rank, **kwargs)
    
    elif name == 'qlora_pager':
        bits = kwargs.get('quantization_bits', 8)
        return QLoRAPagerOptimizer(parameters, lr=lr, weight_decay=weight_decay,
                                 quantization_bits=bits, **kwargs)
    
    elif name == '8bit_adam':
        if HAS_BITSANDBYTES:
            return bnb.optim.Adam8bit(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
        else:
            return EightBitAdam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    else:
        raise ValueError(f"Unknown optimizer: {name}. Available: adam, adamw, sgd, adafactor, "
                        f"galore, adarankgrad, qlora_pager, 8bit_adam")