"""
MANGO-LRQ: Memory-Adaptive Neural Gradient Optimizer with Low-Rank Quantization

Implements hybrid low-rank + quantized compression combining:
1. GaLore-style low-rank gradient projection
2. NF4 quantization from QLoRA
3. Variance-reduced error compensation
4. Policy-controlled compression parameters
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union
import math
import numpy as np
from dataclasses import dataclass

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import LinearNF4
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not available, falling back to standard quantization")


@dataclass
class CompressionConfig:
    """Configuration for MANGO-LRQ compression."""
    rank: int = 4                    # Low-rank approximation rank
    bits_P: int = 8                  # Bits for P factor matrix
    bits_Q: int = 8                  # Bits for Q factor matrix
    momentum_precision: str = "fp16"  # "fp32", "fp16", "nf4"
    use_nf4: bool = True             # Use NF4 quantization
    error_feedback: bool = True       # Enable error feedback
    variance_reduction: bool = True   # Enable variance reduction
    reference_steps: int = 10         # Steps between full-precision references


class GaLoreLowRankProjector:
    """
    Implements GaLore-style low-rank gradient projection.
    """
    
    def __init__(self, rank: int = 4, update_freq: int = 200):
        """
        Initialize low-rank projector.
        
        Args:
            rank: Target rank for low-rank approximation
            update_freq: Frequency of projection update (in steps)
        """
        self.rank = rank
        self.update_freq = update_freq
        self.step_count = 0
        
        # Projection matrices
        self.P = None  # Left projection matrix
        self.Q = None  # Right projection matrix
        
        # Statistics
        self.projection_stats = {
            'total_projections': 0,
            'avg_rank_reduction': 0.0,
            'projection_errors': []
        }
    
    def project(self, gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project gradient to low-rank subspace.
        
        Args:
            gradient: Input gradient tensor
            
        Returns:
            Tuple of (projected_grad, P_matrix, Q_matrix)
        """
        if gradient.dim() != 2:
            # Reshape to 2D for projection
            original_shape = gradient.shape
            gradient_2d = gradient.view(-1, gradient.shape[-1])
        else:
            original_shape = None
            gradient_2d = gradient
        
        # Update projection matrices if needed
        if self.P is None or self.step_count % self.update_freq == 0:
            self._update_projection_matrices(gradient_2d)
        
        # Project gradient
        projected = self.P.T @ gradient_2d @ self.Q
        
        self.step_count += 1
        self.projection_stats['total_projections'] += 1
        
        # Reshape back if needed
        if original_shape is not None:
            # For non-2D tensors, we need to handle projection differently
            # This is a simplified approach - more sophisticated handling may be needed
            projected = projected.view(*original_shape[:-1], projected.shape[-1])
        
        return projected, self.P, self.Q
    
    def reconstruct(
        self, 
        projected: torch.Tensor, 
        P: torch.Tensor, 
        Q: torch.Tensor,
        original_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """
        Reconstruct gradient from low-rank projection.
        
        Args:
            projected: Projected gradient
            P: Left projection matrix
            Q: Right projection matrix  
            original_shape: Original gradient shape
            
        Returns:
            Reconstructed gradient tensor
        """
        if original_shape is not None and len(original_shape) > 2:
            # Handle non-2D reconstruction
            projected_2d = projected.view(-1, projected.shape[-1])
            reconstructed_2d = P @ projected_2d @ Q.T
            return reconstructed_2d.view(original_shape)
        else:
            return P @ projected @ Q.T
    
    def _update_projection_matrices(self, gradient: torch.Tensor):
        """Update projection matrices using randomized SVD."""
        try:
            # Use randomized SVD for efficiency
            U, S, Vt = torch.linalg.svd(gradient, full_matrices=False)
            
            # Keep top-r components
            rank = min(self.rank, min(gradient.shape))
            self.P = U[:, :rank].contiguous()
            self.Q = Vt[:rank, :].T.contiguous()
            
            # Update statistics
            total_energy = S.sum()
            kept_energy = S[:rank].sum() if len(S) >= rank else S.sum()
            rank_reduction = kept_energy / max(total_energy, 1e-8)
            self.projection_stats['avg_rank_reduction'] = rank_reduction.item()
            
        except Exception as e:
            print(f"Warning: SVD failed, using identity projection: {e}")
            # Fallback to identity-like projection
            m, n = gradient.shape
            rank = min(self.rank, min(m, n))
            self.P = torch.eye(m, rank, device=gradient.device, dtype=gradient.dtype)
            self.Q = torch.eye(rank, n, device=gradient.device, dtype=gradient.dtype)


class NF4Quantizer:
    """
    NF4 (Normal Float 4) quantization implementation.
    """
    
    def __init__(self):
        """Initialize NF4 quantizer."""
        # NF4 quantization levels (optimized for normal distributions)
        self.nf4_levels = torch.tensor([
            -1.0, -0.8480762113, -0.6106329113, -0.4244131815,
            -0.2615740984, -0.1226408571, -0.0156119577, 0.0,
            0.0156119577, 0.1226408571, 0.2615740984, 0.4244131815,
            0.6106329113, 0.8480762113, 1.0, float('inf')
        ])
    
    def quantize(self, tensor: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply NF4 quantization with per-block scaling.
        
        Args:
            tensor: Input tensor to quantize
            block_size: Block size for per-block quantization
            
        Returns:
            Tuple of (quantized_indices, scales)
        """
        if not HAS_BITSANDBYTES:
            # Fallback to standard 4-bit quantization
            return self._fallback_quantize(tensor)
        
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Pad to block_size multiple
        n_blocks = (len(flat_tensor) + block_size - 1) // block_size
        padded_size = n_blocks * block_size
        if len(flat_tensor) < padded_size:
            flat_tensor = F.pad(flat_tensor, (0, padded_size - len(flat_tensor)))
        
        # Reshape to blocks
        blocks = flat_tensor.view(n_blocks, block_size)
        
        # Per-block quantization
        quantized_blocks = []
        scales = []
        
        for block in blocks:
            # Compute scale
            absmax = block.abs().max()
            scale = absmax / 0.8480762113  # Max NF4 level
            scales.append(scale)
            
            if scale > 0:
                # Normalize to [-1, 1] range
                normalized = block / scale
                
                # Quantize using NF4 levels
                levels = self.nf4_levels.to(block.device)
                distances = torch.abs(normalized.unsqueeze(1) - levels.unsqueeze(0))
                indices = torch.argmin(distances, dim=1)
                quantized_blocks.append(indices)
            else:
                quantized_blocks.append(torch.zeros_like(block, dtype=torch.int8))
        
        quantized = torch.stack(quantized_blocks).flatten()[:len(tensor.flatten())]
        scales_tensor = torch.stack(scales)
        
        return quantized.view(original_shape), scales_tensor
    
    def dequantize(
        self, 
        quantized: torch.Tensor, 
        scales: torch.Tensor, 
        block_size: int = 64
    ) -> torch.Tensor:
        """
        Dequantize NF4 tensor.
        
        Args:
            quantized: Quantized indices
            scales: Per-block scales
            block_size: Block size used for quantization
            
        Returns:
            Dequantized tensor
        """
        original_shape = quantized.shape
        flat_quantized = quantized.flatten()
        
        # Pad to block_size multiple
        n_blocks = len(scales)
        padded_size = n_blocks * block_size
        if len(flat_quantized) < padded_size:
            flat_quantized = F.pad(flat_quantized, (0, padded_size - len(flat_quantized)))
        
        # Reshape to blocks
        blocks = flat_quantized.view(n_blocks, block_size)
        
        # Dequantize blocks
        dequantized_blocks = []
        levels = self.nf4_levels.to(quantized.device)
        
        for i, (block, scale) in enumerate(zip(blocks, scales)):
            # Map indices to NF4 values
            dequant_block = levels[block.long()] * scale
            dequantized_blocks.append(dequant_block)
        
        dequantized = torch.cat(dequantized_blocks)[:len(quantized.flatten())]
        
        return dequantized.view(original_shape)
    
    def _fallback_quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback quantization when bitsandbytes is not available."""
        absmax = tensor.abs().max()
        scale = absmax / 7.0  # 4-bit signed range: [-8, 7]
        
        if scale > 0:
            quantized = torch.round(tensor / scale).clamp(-8, 7)
            dequantized = quantized * scale
        else:
            dequantized = torch.zeros_like(tensor)
        
        return dequantized, torch.tensor([scale])


class VarianceReducedBuffer:
    """
    Implements variance-reduced error compensation similar to Byz-VR-MARINA.
    """
    
    def __init__(self, reference_steps: int = 10):
        """
        Initialize variance reduction buffer.
        
        Args:
            reference_steps: Steps between full-precision gradient references
        """
        self.reference_steps = reference_steps
        self.step_count = 0
        self.reference_gradients = {}
        self.accumulated_compressed = {}
        
    def update(
        self, 
        param_id: int, 
        full_gradient: torch.Tensor, 
        compressed_gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Update variance reduction with new gradient.
        
        Args:
            param_id: Parameter identifier
            full_gradient: Full-precision gradient
            compressed_gradient: Compressed gradient
            
        Returns:
            Variance-reduced gradient update
        """
        self.step_count += 1
        
        # Store reference gradient every T steps
        if self.step_count % self.reference_steps == 0:
            self.reference_gradients[param_id] = full_gradient.clone()
            self.accumulated_compressed[param_id] = torch.zeros_like(full_gradient)
            return full_gradient
        
        # Accumulate compressed gradients
        if param_id not in self.accumulated_compressed:
            self.accumulated_compressed[param_id] = torch.zeros_like(full_gradient)
        
        self.accumulated_compressed[param_id] += compressed_gradient
        
        # Compute variance-reduced update
        if param_id in self.reference_gradients:
            # VR update: compressed + (reference - accumulated_compressed/steps)
            steps_since_ref = (self.step_count - 1) % self.reference_steps + 1
            avg_compressed = self.accumulated_compressed[param_id] / steps_since_ref
            variance_reduction_term = self.reference_gradients[param_id] - avg_compressed
            
            return compressed_gradient + 0.1 * variance_reduction_term  # Small correction factor
        else:
            return compressed_gradient


class MangoLRQCompressor:
    """
    Main MANGO-LRQ compressor implementing hybrid low-rank + quantized compression.
    """
    
    def __init__(self, config: CompressionConfig):
        """
        Initialize MANGO-LRQ compressor.
        
        Args:
            config: Compression configuration
        """
        self.config = config
        
        # Initialize components
        self.low_rank_projector = GaLoreLowRankProjector(rank=config.rank)
        self.nf4_quantizer = NF4Quantizer() if config.use_nf4 else None
        self.variance_reducer = VarianceReducedBuffer(config.reference_steps) if config.variance_reduction else None
        
        # Error feedback buffers
        self.error_buffers = {} if config.error_feedback else None
        
        # Statistics
        self.compression_stats = {
            'total_compressions': 0,
            'memory_savings': [],
            'compression_errors': [],
            'rank_utilization': []
        }
    
    def compress(
        self, 
        gradient: torch.Tensor, 
        param_id: int,
        compression_config: Optional[CompressionConfig] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Main compression function implementing MANGO-LRQ algorithm.
        
        Args:
            gradient: Input gradient tensor
            param_id: Parameter identifier for error tracking
            compression_config: Optional override configuration
            
        Returns:
            Tuple of (compressed_gradient, compression_metadata)
        """
        if gradient.numel() == 0:
            return gradient, {}
        
        config = compression_config or self.config
        original_gradient = gradient.clone()
        
        # Add accumulated error if using error feedback
        if self.error_buffers and param_id in self.error_buffers:
            gradient = gradient + self.error_buffers[param_id]
        
        compression_metadata = {}
        
        # Step 1: Low-rank projection (GaLore-style)
        if gradient.numel() > config.rank and min(gradient.shape) > 1:
            try:
                projected_grad, P, Q = self.low_rank_projector.project(gradient)
                compression_metadata['P'] = P
                compression_metadata['Q'] = Q
                compression_metadata['original_shape'] = gradient.shape
                
                # Step 2: Quantize factor matrices
                if config.use_nf4 and self.nf4_quantizer:
                    # Quantize P matrix
                    if config.bits_P < 32:
                        P_quantized, P_scales = self.nf4_quantizer.quantize(P)
                        compression_metadata['P_quantized'] = P_quantized
                        compression_metadata['P_scales'] = P_scales
                    
                    # Quantize Q matrix
                    if config.bits_Q < 32:
                        Q_quantized, Q_scales = self.nf4_quantizer.quantize(Q)
                        compression_metadata['Q_quantized'] = Q_quantized
                        compression_metadata['Q_scales'] = Q_scales
                
                # Use projected gradient for next steps
                working_gradient = projected_grad
                
            except Exception as e:
                print(f"Warning: Low-rank projection failed: {e}, using original gradient")
                working_gradient = gradient
        else:
            working_gradient = gradient
        
        # Step 3: Final gradient quantization if needed
        if hasattr(config, 'gradient_bits') and config.gradient_bits < 32:
            if config.use_nf4 and self.nf4_quantizer:
                grad_quantized, grad_scales = self.nf4_quantizer.quantize(working_gradient)
                compression_metadata['grad_quantized'] = grad_quantized
                compression_metadata['grad_scales'] = grad_scales
                final_gradient = self.nf4_quantizer.dequantize(grad_quantized, grad_scales)
            else:
                final_gradient = self._apply_standard_quantization(working_gradient, config.gradient_bits)
        else:
            final_gradient = working_gradient
        
        # Step 4: Reconstruct if low-rank projection was applied
        if 'P' in compression_metadata:
            try:
                if 'P_quantized' in compression_metadata:
                    P_dequant = self.nf4_quantizer.dequantize(
                        compression_metadata['P_quantized'], 
                        compression_metadata['P_scales']
                    )
                else:
                    P_dequant = compression_metadata['P']
                
                if 'Q_quantized' in compression_metadata:
                    Q_dequant = self.nf4_quantizer.dequantize(
                        compression_metadata['Q_quantized'],
                        compression_metadata['Q_scales']
                    )
                else:
                    Q_dequant = compression_metadata['Q']
                
                final_gradient = self.low_rank_projector.reconstruct(
                    final_gradient, P_dequant, Q_dequant, compression_metadata['original_shape']
                )
            except Exception as e:
                print(f"Warning: Gradient reconstruction failed: {e}")
                # Fallback to projected gradient
                pass
        
        # Step 5: Error feedback
        compression_error = original_gradient - final_gradient
        if self.error_buffers is not None:
            if param_id not in self.error_buffers:
                self.error_buffers[param_id] = torch.zeros_like(original_gradient)
            self.error_buffers[param_id] = compression_error.clone()
        
        # Step 6: Variance reduction
        if self.variance_reducer:
            final_gradient = self.variance_reducer.update(param_id, original_gradient, final_gradient)
        
        # Update statistics
        self._update_statistics(original_gradient, final_gradient, compression_metadata)
        
        return final_gradient, compression_metadata
    
    def _apply_standard_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Apply standard quantization for fallback."""
        if bits >= 32:
            return tensor
        
        # Simple uniform quantization
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            return tensor
        
        levels = 2 ** bits - 1
        scale = (tensor_max - tensor_min) / levels
        
        quantized = torch.round((tensor - tensor_min) / scale)
        dequantized = quantized * scale + tensor_min
        
        return dequantized
    
    def _update_statistics(
        self, 
        original: torch.Tensor, 
        compressed: torch.Tensor, 
        metadata: Dict
    ):
        """Update compression statistics."""
        self.compression_stats['total_compressions'] += 1
        
        # Compute memory savings
        original_memory = original.numel() * 4  # FP32 bytes
        
        # Estimate compressed memory
        compressed_memory = 0
        if 'P_quantized' in metadata:
            compressed_memory += metadata['P_quantized'].numel() * (self.config.bits_P // 8)
        if 'Q_quantized' in metadata:
            compressed_memory += metadata['Q_quantized'].numel() * (self.config.bits_Q // 8)
        if 'grad_quantized' in metadata:
            compressed_memory += metadata['grad_quantized'].numel() * 0.5  # 4-bit = 0.5 bytes
        
        if compressed_memory > 0:
            memory_saving = 1 - (compressed_memory / original_memory)
            self.compression_stats['memory_savings'].append(memory_saving)
        
        # Compression error
        mse_error = F.mse_loss(original, compressed).item()
        self.compression_stats['compression_errors'].append(mse_error)
        
        # Rank utilization
        if hasattr(self.low_rank_projector, 'projection_stats'):
            rank_util = self.low_rank_projector.projection_stats.get('avg_rank_reduction', 0.0)
            self.compression_stats['rank_utilization'].append(rank_util)
    
    def get_compression_ratio(self) -> float:
        """Get overall memory compression ratio."""
        if not self.compression_stats['memory_savings']:
            return 1.0
        
        avg_saving = np.mean(self.compression_stats['memory_savings'])
        return 1.0 / (1.0 - avg_saving) if avg_saving < 1.0 else float('inf')
    
    def get_statistics(self) -> Dict:
        """Get comprehensive compression statistics."""
        stats = self.compression_stats.copy()
        
        # Add derived statistics
        if stats['memory_savings']:
            stats['avg_memory_saving'] = np.mean(stats['memory_savings'])
            stats['compression_ratio'] = self.get_compression_ratio()
        
        if stats['compression_errors']:
            stats['avg_compression_error'] = np.mean(stats['compression_errors'])
            stats['max_compression_error'] = np.max(stats['compression_errors'])
        
        if stats['rank_utilization']:
            stats['avg_rank_utilization'] = np.mean(stats['rank_utilization'])
        
        return stats
    
    def reset_statistics(self):
        """Reset all compression statistics."""
        self.compression_stats = {
            'total_compressions': 0,
            'memory_savings': [],
            'compression_errors': [],
            'rank_utilization': []
        }