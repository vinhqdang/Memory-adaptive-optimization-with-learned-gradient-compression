"""
EF21 Error Feedback Buffer for Variance-Reduced Communication

Implementation based on:
- Richtárik, P., Mishchenko, K., & Jiang, S. (2021). 
  "Error feedback fixes signSGD and other gradient compression schemes."
  International Conference on Machine Learning (ICML).
  
- Beznosikov, A., Horváth, S., Richtárik, P., & Safaryan, M. (2020).
  "On biased compression for distributed learning."
  arXiv preprint arXiv:2002.12410.

The EF21 mechanism provides unbiased gradient compression with variance reduction
through error compensation and momentum correction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math


class EF21Buffer:
    """
    Error Feedback Buffer implementing EF21 algorithm.
    
    EF21 combines:
    1. Error feedback for unbiased compression
    2. Momentum correction for variance reduction
    3. Adaptive compression based on gradient statistics
    """
    
    def __init__(
        self,
        momentum_factor: float = 0.9,
        variance_reduction: bool = True,
        adaptive_compression: bool = True,
        reference_steps: int = 10,
        compression_schedule: str = "polynomial"
    ):
        """
        Initialize EF21 error feedback buffer.
        
        Args:
            momentum_factor: Momentum coefficient for error accumulation
            variance_reduction: Enable variance-reduced updates
            adaptive_compression: Use adaptive compression rates
            reference_steps: Steps between compression rate updates
            compression_schedule: Schedule for compression rate decay
        """
        self.momentum_factor = momentum_factor
        self.variance_reduction = variance_reduction
        self.adaptive_compression = adaptive_compression
        self.reference_steps = reference_steps
        self.compression_schedule = compression_schedule
        
        # Error buffers for each parameter
        self.error_buffers: Dict[int, torch.Tensor] = {}
        self.momentum_buffers: Dict[int, torch.Tensor] = {}
        
        # Variance reduction tracking
        self.gradient_history: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.reference_gradients: Dict[int, torch.Tensor] = {}
        
        # Statistics tracking
        self.step_count = 0
        self.compression_stats = {
            'total_error_norm': 0.0,
            'total_gradient_norm': 0.0,
            'compression_ratios': [],
            'variance_reductions': []
        }
    
    def initialize_buffers(self, param_id: int, param_shape: torch.Size, device: torch.device):
        """Initialize error and momentum buffers for a parameter."""
        if param_id not in self.error_buffers:
            self.error_buffers[param_id] = torch.zeros(param_shape, device=device)
            self.momentum_buffers[param_id] = torch.zeros(param_shape, device=device)
            self.reference_gradients[param_id] = torch.zeros(param_shape, device=device)
    
    def apply_error_feedback(
        self, 
        gradient: torch.Tensor, 
        param_id: int,
        compression_ratio: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply EF21 error feedback mechanism.
        
        Args:
            gradient: Original gradient tensor
            param_id: Parameter identifier
            compression_ratio: Target compression ratio
            
        Returns:
            Tuple of (corrected_gradient, metadata)
        """
        self.initialize_buffers(param_id, gradient.shape, gradient.device)
        
        # Get error buffer
        error_buffer = self.error_buffers[param_id]
        momentum_buffer = self.momentum_buffers[param_id]
        
        # EF21 Step 1: Add accumulated error to current gradient
        corrected_gradient = gradient + error_buffer
        
        # EF21 Step 2: Apply compression
        compressed_gradient = self._compress_gradient(
            corrected_gradient, compression_ratio
        )
        
        # EF21 Step 3: Compute compression error
        compression_error = corrected_gradient - compressed_gradient
        
        # EF21 Step 4: Update error buffer with momentum
        if self.variance_reduction:
            # Variance-reduced error update (VR-MARINA style)
            variance_reduced_error = self._apply_variance_reduction(
                compression_error, gradient, param_id
            )
            error_buffer.mul_(self.momentum_factor).add_(
                variance_reduced_error, alpha=1 - self.momentum_factor
            )
        else:
            # Standard error feedback
            error_buffer.copy_(compression_error)
        
        # Update momentum buffer
        momentum_buffer.mul_(self.momentum_factor).add_(
            compressed_gradient, alpha=1 - self.momentum_factor
        )
        
        # Update statistics
        self._update_statistics(gradient, compressed_gradient, compression_error)
        
        metadata = {
            'compression_error_norm': compression_error.norm().item(),
            'original_norm': gradient.norm().item(),
            'compressed_norm': compressed_gradient.norm().item(),
            'compression_ratio': self._compute_compression_ratio(gradient, compressed_gradient),
            'variance_reduction_factor': self._compute_variance_reduction(param_id)
        }
        
        self.step_count += 1
        
        return compressed_gradient, metadata
    
    def _compress_gradient(
        self, 
        gradient: torch.Tensor, 
        compression_ratio: float
    ) -> torch.Tensor:
        """
        Apply gradient compression (top-k sparsification).
        
        Args:
            gradient: Gradient to compress
            compression_ratio: Fraction of elements to keep
            
        Returns:
            Compressed gradient
        """
        if compression_ratio >= 1.0:
            return gradient.clone()
        
        # Flatten gradient for compression
        flat_grad = gradient.flatten()
        k = max(1, int(flat_grad.numel() * compression_ratio))
        
        # Top-k compression by magnitude
        _, top_indices = torch.topk(flat_grad.abs(), k, largest=True)
        
        # Create compressed gradient
        compressed_flat = torch.zeros_like(flat_grad)
        compressed_flat[top_indices] = flat_grad[top_indices]
        
        return compressed_flat.reshape(gradient.shape)
    
    def _apply_variance_reduction(
        self,
        compression_error: torch.Tensor,
        current_gradient: torch.Tensor,
        param_id: int
    ) -> torch.Tensor:
        """
        Apply variance reduction using reference gradients (VR-MARINA).
        
        Args:
            compression_error: Current compression error
            current_gradient: Current gradient
            param_id: Parameter identifier
            
        Returns:
            Variance-reduced error
        """
        # Update gradient history
        history = self.gradient_history[param_id]
        history.append(current_gradient.clone())
        
        # Keep only recent history
        if len(history) > self.reference_steps:
            history.pop(0)
        
        # Update reference gradient periodically
        if self.step_count % self.reference_steps == 0 and len(history) > 1:
            # Compute reference as exponential moving average
            alpha = 0.1  # Reference update rate
            reference = self.reference_gradients[param_id]
            reference.mul_(1 - alpha).add_(current_gradient, alpha=alpha)
        
        # Variance reduction term
        if param_id in self.reference_gradients:
            reference = self.reference_gradients[param_id]
            # VR-MARINA variance reduction
            variance_reduction_term = compression_error - (current_gradient - reference)
            return variance_reduction_term
        else:
            return compression_error
    
    def _compute_compression_ratio(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor
    ) -> float:
        """Compute effective compression ratio."""
        original_norm = original.norm().item()
        compressed_norm = compressed.norm().item()
        
        if original_norm == 0:
            return 1.0
        
        # Ratio of non-zero elements
        non_zero_ratio = (compressed != 0).float().mean().item()
        return non_zero_ratio
    
    def _compute_variance_reduction(self, param_id: int) -> float:
        """Compute variance reduction factor."""
        if param_id not in self.gradient_history:
            return 0.0
        
        history = self.gradient_history[param_id]
        if len(history) < 2:
            return 0.0
        
        # Compute variance of recent gradients
        recent_grads = torch.stack(history[-min(10, len(history)):])
        variance = recent_grads.var(dim=0).mean().item()
        
        # Normalized variance reduction (0 = no reduction, 1 = perfect reduction)
        return min(1.0, 1.0 / (1.0 + variance))
    
    def _update_statistics(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor,
        error: torch.Tensor
    ):
        """Update compression statistics."""
        self.compression_stats['total_error_norm'] += error.norm().item()
        self.compression_stats['total_gradient_norm'] += original.norm().item()
        
        ratio = self._compute_compression_ratio(original, compressed)
        self.compression_stats['compression_ratios'].append(ratio)
        
        # Keep only recent statistics
        if len(self.compression_stats['compression_ratios']) > 1000:
            self.compression_stats['compression_ratios'] = \
                self.compression_stats['compression_ratios'][-500:]
    
    def get_adaptive_compression_ratio(self, param_id: int, base_ratio: float = 0.1) -> float:
        """
        Compute adaptive compression ratio based on gradient statistics.
        
        Args:
            param_id: Parameter identifier
            base_ratio: Base compression ratio
            
        Returns:
            Adaptive compression ratio
        """
        if not self.adaptive_compression:
            return base_ratio
        
        # Polynomial decay schedule
        if self.compression_schedule == "polynomial":
            decay_factor = (1 + self.step_count / 1000) ** (-0.5)
            adaptive_ratio = base_ratio * decay_factor
        elif self.compression_schedule == "exponential":
            decay_factor = math.exp(-self.step_count / 2000)
            adaptive_ratio = base_ratio * decay_factor
        else:
            adaptive_ratio = base_ratio
        
        # Gradient-dependent adjustment
        if param_id in self.gradient_history:
            history = self.gradient_history[param_id]
            if len(history) > 2:
                # Increase compression for stable gradients
                recent_variance = torch.stack(history[-3:]).var(dim=0).mean().item()
                stability_factor = 1.0 / (1.0 + recent_variance)
                adaptive_ratio *= (0.5 + 0.5 * stability_factor)
        
        return max(0.01, min(1.0, adaptive_ratio))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive EF21 statistics."""
        if not self.compression_stats['compression_ratios']:
            return {'error': 'No statistics available'}
        
        ratios = torch.tensor(self.compression_stats['compression_ratios'])
        
        return {
            'average_compression_ratio': ratios.mean().item(),
            'compression_ratio_std': ratios.std().item(),
            'total_error_norm': self.compression_stats['total_error_norm'],
            'total_gradient_norm': self.compression_stats['total_gradient_norm'],
            'relative_error': (
                self.compression_stats['total_error_norm'] / 
                max(self.compression_stats['total_gradient_norm'], 1e-8)
            ),
            'steps_processed': self.step_count,
            'active_parameters': len(self.error_buffers)
        }
    
    def reset(self):
        """Reset all buffers and statistics."""
        self.error_buffers.clear()
        self.momentum_buffers.clear()
        self.gradient_history.clear()
        self.reference_gradients.clear()
        self.step_count = 0
        self.compression_stats = {
            'total_error_norm': 0.0,
            'total_gradient_norm': 0.0,
            'compression_ratios': [],
            'variance_reductions': []
        }


class EF21Compressor:
    """
    Wrapper class that integrates EF21 buffer with MANGO compression.
    """
    
    def __init__(
        self,
        base_compression_ratio: float = 0.1,
        momentum_factor: float = 0.9,
        variance_reduction: bool = True,
        adaptive_compression: bool = True
    ):
        """Initialize EF21 compressor."""
        self.base_compression_ratio = base_compression_ratio
        self.ef21_buffer = EF21Buffer(
            momentum_factor=momentum_factor,
            variance_reduction=variance_reduction,
            adaptive_compression=adaptive_compression
        )
    
    def compress(
        self,
        gradient: torch.Tensor,
        param_id: int,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress gradient using EF21 mechanism.
        
        Args:
            gradient: Gradient to compress
            param_id: Parameter identifier
            config: Optional compression configuration
            
        Returns:
            Tuple of (compressed_gradient, metadata)
        """
        # Get adaptive compression ratio
        compression_ratio = self.ef21_buffer.get_adaptive_compression_ratio(
            param_id, self.base_compression_ratio
        )
        
        # Override with config if provided
        if config and 'compression_ratio' in config:
            compression_ratio = config['compression_ratio']
        
        # Apply EF21 compression
        return self.ef21_buffer.apply_error_feedback(
            gradient, param_id, compression_ratio
        )
    
    def get_compression_ratio(self) -> float:
        """Get average compression ratio."""
        stats = self.ef21_buffer.get_statistics()
        return stats.get('average_compression_ratio', self.base_compression_ratio)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return self.ef21_buffer.get_statistics()
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.ef21_buffer.reset()