"""
Gradient Compression Module

Implements various gradient compression techniques including adaptive quantization,
sparsification, and error feedback mechanisms.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
import numpy as np


class GradientCompressor:
    """
    Main gradient compression class that handles various compression strategies
    including quantization, sparsification, and error feedback.
    """
    
    def __init__(self, error_feedback: bool = True, stochastic_rounding: bool = True):
        """
        Initialize gradient compressor.
        
        Args:
            error_feedback: Whether to maintain error feedback buffers
            stochastic_rounding: Whether to use stochastic rounding for quantization
        """
        self.error_feedback = error_feedback
        self.stochastic_rounding = stochastic_rounding
        
        # Compression statistics
        self.compression_stats = {
            'total_compressions': 0,
            'total_original_bits': 0,
            'total_compressed_bits': 0,
            'compression_ratios': []
        }
    
    def compress(
        self, 
        tensor: torch.Tensor, 
        bits: int = 32,
        sparsity_ratio: float = 0.0,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main compression function that applies quantization and sparsification.
        
        Args:
            tensor: Input tensor to compress
            bits: Number of bits for quantization (32, 16, 8, 4, 2)
            sparsity_ratio: Fraction of values to set to zero [0, 1)
            importance_scores: Optional importance weights for adaptive compression
            
        Returns:
            Tuple of (compressed_tensor, compression_error)
        """
        if tensor.numel() == 0:
            return tensor.clone(), torch.zeros_like(tensor)
        
        original_tensor = tensor.clone()
        compressed_tensor = tensor.clone()
        
        # Apply sparsification first
        if sparsity_ratio > 0.0:
            compressed_tensor = self._apply_sparsification(
                compressed_tensor, sparsity_ratio, importance_scores
            )
        
        # Apply quantization
        if bits < 32:
            compressed_tensor = self._apply_quantization(compressed_tensor, bits)
        
        # Compute compression error
        compression_error = original_tensor - compressed_tensor
        
        # Update statistics
        self._update_stats(original_tensor, compressed_tensor, bits, sparsity_ratio)
        
        return compressed_tensor, compression_error
    
    def _apply_sparsification(
        self, 
        tensor: torch.Tensor, 
        sparsity_ratio: float,
        importance_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply sparsification by keeping only top-k elements.
        
        Args:
            tensor: Input tensor
            sparsity_ratio: Fraction of elements to zero out
            importance_scores: Optional importance weights
            
        Returns:
            Sparsified tensor
        """
        if sparsity_ratio <= 0.0 or sparsity_ratio >= 1.0:
            return tensor
        
        # Number of elements to keep
        k = max(1, int(tensor.numel() * (1 - sparsity_ratio)))
        
        # Flatten tensor for top-k selection
        flat_tensor = tensor.flatten()
        
        if importance_scores is not None:
            # Use importance-weighted magnitude
            flat_importance = importance_scores.flatten()
            if flat_importance.numel() == flat_tensor.numel():
                weighted_magnitude = flat_tensor.abs() * flat_importance
            else:
                # Fallback to magnitude-only
                weighted_magnitude = flat_tensor.abs()
        else:
            weighted_magnitude = flat_tensor.abs()
        
        # Get top-k indices
        _, top_indices = torch.topk(weighted_magnitude, k, largest=True)
        
        # Create sparse tensor
        sparse_tensor = torch.zeros_like(flat_tensor)
        sparse_tensor[top_indices] = flat_tensor[top_indices]
        
        return sparse_tensor.reshape(tensor.shape)
    
    def _apply_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Apply quantization to reduce bit precision.
        
        Args:
            tensor: Input tensor
            bits: Number of quantization bits
            
        Returns:
            Quantized tensor
        """
        if bits >= 32:
            return tensor
        
        if bits == 16:
            return self._quantize_fp16(tensor)
        elif bits == 8:
            return self._quantize_int8(tensor)
        elif bits == 4:
            return self._quantize_int4(tensor)
        elif bits == 2:
            return self._quantize_int2(tensor)
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
    
    def _quantize_fp16(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to FP16."""
        return tensor.half().float()
    
    def _quantize_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to 8-bit integers."""
        if tensor.numel() == 0:
            return tensor
        
        # Compute scale and zero point
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            return tensor
        
        # 8-bit range: [-128, 127] for signed quantization
        qmin, qmax = -128, 127
        scale = (tensor_max - tensor_min) / (qmax - qmin)
        zero_point = qmin - tensor_min / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _quantize_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to 4-bit integers."""
        if tensor.numel() == 0:
            return tensor
        
        # 4-bit range: [-8, 7] for signed quantization
        qmin, qmax = -8, 7
        
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            return tensor
        
        scale = (tensor_max - tensor_min) / (qmax - qmin)
        zero_point = qmin - tensor_min / scale
        
        # Quantize with stochastic rounding for better approximation
        if self.stochastic_rounding:
            quantized_float = tensor / scale + zero_point
            quantized_floor = torch.floor(quantized_float)
            prob = quantized_float - quantized_floor
            quantized = quantized_floor + torch.bernoulli(prob)
        else:
            quantized = torch.round(tensor / scale + zero_point)
        
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _quantize_int2(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to 2-bit integers (extreme compression)."""
        if tensor.numel() == 0:
            return tensor
        
        # 2-bit range: [-2, 1] for signed quantization
        qmin, qmax = -2, 1
        
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            return tensor
        
        scale = (tensor_max - tensor_min) / (qmax - qmin)
        zero_point = qmin - tensor_min / scale
        
        # For 2-bit, always use stochastic rounding due to limited precision
        quantized_float = tensor / scale + zero_point
        quantized_floor = torch.floor(quantized_float)
        prob = quantized_float - quantized_floor
        quantized = quantized_floor + torch.bernoulli(prob)
        
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _update_stats(
        self, 
        original: torch.Tensor, 
        compressed: torch.Tensor, 
        bits: int, 
        sparsity_ratio: float
    ):
        """Update compression statistics."""
        self.compression_stats['total_compressions'] += 1
        
        original_bits = original.numel() * 32  # Assume FP32 original
        
        # Estimate compressed bits
        if sparsity_ratio > 0:
            effective_elements = int(original.numel() * (1 - sparsity_ratio))
            compressed_bits = effective_elements * bits
        else:
            compressed_bits = original.numel() * bits
        
        self.compression_stats['total_original_bits'] += original_bits
        self.compression_stats['total_compressed_bits'] += compressed_bits
        
        if original_bits > 0:
            ratio = original_bits / max(compressed_bits, 1)
            self.compression_stats['compression_ratios'].append(ratio)
    
    def get_compression_ratio(self) -> float:
        """Get overall compression ratio."""
        if self.compression_stats['total_compressed_bits'] == 0:
            return 1.0
        return self.compression_stats['total_original_bits'] / self.compression_stats['total_compressed_bits']
    
    def get_recent_compression_ratio(self, window: int = 100) -> float:
        """Get recent compression ratio over a window."""
        ratios = self.compression_stats['compression_ratios']
        if not ratios:
            return 1.0
        recent_ratios = ratios[-window:]
        return float(np.mean(recent_ratios))
    
    def reset_stats(self):
        """Reset compression statistics."""
        self.compression_stats = {
            'total_compressions': 0,
            'total_original_bits': 0,
            'total_compressed_bits': 0,
            'compression_ratios': []
        }


class ErrorFeedbackBuffer:
    """
    Manages error feedback buffers for maintaining compression accuracy.
    """
    
    def __init__(self, decay_rate: float = 0.9):
        """
        Initialize error feedback buffer.
        
        Args:
            decay_rate: Decay rate for error accumulation
        """
        self.decay_rate = decay_rate
        self.buffers = {}
    
    def get_error(self, param_id: int, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Get accumulated error for parameter."""
        if param_id not in self.buffers:
            self.buffers[param_id] = torch.zeros(shape, device=device)
        return self.buffers[param_id]
    
    def update_error(self, param_id: int, error: torch.Tensor):
        """Update error buffer with new compression error."""
        if param_id not in self.buffers:
            self.buffers[param_id] = error.clone()
        else:
            self.buffers[param_id] = self.decay_rate * self.buffers[param_id] + error
    
    def clear_buffer(self, param_id: int):
        """Clear error buffer for specific parameter."""
        if param_id in self.buffers:
            self.buffers[param_id].zero_()
    
    def clear_all_buffers(self):
        """Clear all error buffers."""
        for buffer in self.buffers.values():
            buffer.zero_()


class AdaptiveQuantizer:
    """
    Advanced quantizer that adapts quantization parameters based on tensor statistics.
    """
    
    def __init__(self, percentile_clipping: float = 99.9):
        """
        Initialize adaptive quantizer.
        
        Args:
            percentile_clipping: Percentile for outlier clipping
        """
        self.percentile_clipping = percentile_clipping
    
    def quantize(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Adaptive quantization with outlier handling.
        
        Args:
            tensor: Input tensor
            bits: Number of quantization bits
            
        Returns:
            Quantized tensor
        """
        if bits >= 32 or tensor.numel() == 0:
            return tensor
        
        # Clip outliers based on percentiles
        if self.percentile_clipping < 100.0:
            flat_tensor = tensor.flatten()
            lower_percentile = (100 - self.percentile_clipping) / 2
            upper_percentile = 100 - lower_percentile
            
            lower_bound = torch.quantile(flat_tensor.abs(), lower_percentile / 100)
            upper_bound = torch.quantile(flat_tensor.abs(), upper_percentile / 100)
            
            tensor = torch.clamp(tensor, -upper_bound, upper_bound)
        
        # Compute adaptive scale
        tensor_abs = tensor.abs()
        scale = self._compute_adaptive_scale(tensor_abs, bits)
        
        # Quantize
        levels = 2 ** (bits - 1) - 1  # Account for sign bit
        quantized = torch.round(tensor / scale).clamp(-levels, levels)
        
        # Dequantize
        dequantized = quantized * scale
        
        return dequantized
    
    def _compute_adaptive_scale(self, tensor_abs: torch.Tensor, bits: int) -> float:
        """
        Compute adaptive quantization scale based on tensor distribution.
        
        Args:
            tensor_abs: Absolute values of tensor
            bits: Number of quantization bits
            
        Returns:
            Quantization scale
        """
        if tensor_abs.numel() == 0:
            return 1.0
        
        levels = 2 ** (bits - 1) - 1
        
        # Use different strategies based on bit precision
        if bits >= 8:
            # For higher precision, use max value
            scale = tensor_abs.max().item() / levels
        else:
            # For lower precision, use a robust percentile to avoid outliers
            percentile = min(99.0, 95.0 + bits)  # Higher percentile for more bits
            scale = torch.quantile(tensor_abs, percentile / 100).item() / levels
        
        return max(scale, 1e-8)  # Avoid zero scale


class LayerWiseCompressor:
    """
    Applies different compression strategies to different layers.
    """
    
    def __init__(self, layer_configs: Optional[Dict[str, Dict]] = None):
        """
        Initialize layer-wise compressor.
        
        Args:
            layer_configs: Dictionary mapping layer names/types to compression configs
        """
        self.layer_configs = layer_configs or {}
        self.compressor = GradientCompressor()
        self.adaptive_quantizer = AdaptiveQuantizer()
    
    def compress_layer(
        self, 
        tensor: torch.Tensor, 
        layer_name: str, 
        default_config: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress tensor for specific layer.
        
        Args:
            tensor: Layer gradient tensor
            layer_name: Name/identifier of the layer
            default_config: Default compression configuration
            
        Returns:
            Tuple of (compressed_tensor, compression_error)
        """
        # Get layer-specific config or use default
        layer_config = self.layer_configs.get(layer_name, default_config)
        
        # Extract compression parameters
        bits = layer_config.get('gradient_bits', 32)
        sparsity_ratio = layer_config.get('sparsity_ratio', 0.0)
        use_adaptive = layer_config.get('use_adaptive_quantization', False)
        
        if use_adaptive and bits < 32:
            # Use adaptive quantization
            compressed = self.adaptive_quantizer.quantize(tensor, bits)
            
            # Apply sparsification if needed
            if sparsity_ratio > 0.0:
                compressed = self.compressor._apply_sparsification(compressed, sparsity_ratio)
            
            compression_error = tensor - compressed
        else:
            # Use standard compression
            compressed, compression_error = self.compressor.compress(
                tensor, bits=bits, sparsity_ratio=sparsity_ratio
            )
        
        return compressed, compression_error
    
    def update_layer_config(self, layer_name: str, config: Dict):
        """Update compression configuration for specific layer."""
        self.layer_configs[layer_name] = config
    
    def get_layer_config(self, layer_name: str) -> Dict:
        """Get compression configuration for specific layer."""
        return self.layer_configs.get(layer_name, {})