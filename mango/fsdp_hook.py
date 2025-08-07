"""
FSDP Custom Reducer with Compression Hooks

Implements PyTorch FSDP integration with MANGO-LRQ gradient compression.
Provides compression-aware gradient reduction for distributed training.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.api import CPUOffload, MixedPrecision
from typing import Optional, Dict, Any, List, Callable
import warnings
from contextlib import contextmanager

from .mango_lrq import MangoLRQCompressor, CompressionConfig
from .enhanced_optimizer import EnhancedMANGO


class MANGOFSDPReducer:
    """
    Custom FSDP reducer that applies MANGO-LRQ compression after gradient sharding.
    
    Integrates with PyTorch FSDP to compress gradients post-sharding for 
    maximum memory efficiency in distributed training.
    """
    
    def __init__(
        self,
        compression_config: CompressionConfig,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        enable_compression: bool = True
    ):
        """
        Initialize MANGO FSDP reducer.
        
        Args:
            compression_config: MANGO-LRQ compression configuration
            world_size: Distributed world size (auto-detect if None)
            rank: Process rank (auto-detect if None)  
            enable_compression: Whether to enable gradient compression
        """
        self.compression_config = compression_config
        self.enable_compression = enable_compression
        
        # Distributed setup
        if dist.is_initialized():
            self.world_size = world_size or dist.get_world_size()
            self.rank = rank or dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
        # Initialize compressor
        if self.enable_compression:
            self.compressor = MangoLRQCompressor(compression_config)
        else:
            self.compressor = None
            
        # Compression statistics
        self.compression_stats = {
            'total_gradients_compressed': 0,
            'total_compression_ratio': 0.0,
            'compression_errors': []
        }
        
        print(f"MANGO FSDP Reducer initialized - Rank {self.rank}/{self.world_size}")
    
    def create_fsdp_wrapper(
        self,
        module: torch.nn.Module,
        device_mesh: Optional[torch.distributed.DeviceMesh] = None,
        cpu_offload: Optional[CPUOffload] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        **fsdp_kwargs
    ) -> FSDP:
        """
        Create FSDP wrapper with MANGO compression hooks.
        
        Args:
            module: PyTorch module to wrap
            device_mesh: Device mesh for distributed training
            cpu_offload: CPU offload configuration
            mixed_precision: Mixed precision configuration
            **fsdp_kwargs: Additional FSDP arguments
            
        Returns:
            FSDP-wrapped module with compression hooks
        """
        
        # Default FSDP configuration optimized for MANGO
        default_config = {
            'sharding_strategy': 'FULL_SHARD',  # Maximum memory efficiency
            'cpu_offload': cpu_offload or CPUOffload(offload_params=False),
            'mixed_precision': mixed_precision,
            'device_id': self.rank % torch.cuda.device_count() if torch.cuda.is_available() else None,
            'sync_module_states': True,
            'forward_prefetch': True,
            'backward_prefetch': 'BACKWARD_PRE',
            'param_init_fn': None,
            'ignored_modules': None
        }
        
        # Override with user-provided kwargs
        default_config.update(fsdp_kwargs)
        
        # Create FSDP wrapper
        fsdp_module = FSDP(module, **default_config)
        
        # Register compression hooks
        self._register_compression_hooks(fsdp_module)
        
        return fsdp_module
    
    def _register_compression_hooks(self, fsdp_module: FSDP):
        """Register gradient compression hooks on FSDP module."""
        if not self.enable_compression:
            return
            
        def compression_hook(grad: torch.Tensor) -> torch.Tensor:
            """Hook function to compress gradients after FSDP reduction."""
            if grad is None or not grad.requires_grad:
                return grad
                
            try:
                # Apply MANGO-LRQ compression
                compressed_grad, metadata = self.compressor.compress(
                    grad.data, 
                    param_id=id(grad),
                    config=self.compression_config
                )
                
                # Update statistics
                self.compression_stats['total_gradients_compressed'] += 1
                if 'compression_ratio' in metadata:
                    self.compression_stats['total_compression_ratio'] += metadata['compression_ratio']
                    
                return compressed_grad
                
            except Exception as e:
                warnings.warn(f"MANGO gradient compression failed: {e}, using uncompressed gradient")
                return grad
        
        # Register hooks on all parameters
        for param in fsdp_module.parameters():
            if param.requires_grad:
                param.register_hook(compression_hook)
                
        print(f"Registered MANGO compression hooks on {sum(1 for p in fsdp_module.parameters() if p.requires_grad)} parameters")
    
    def all_reduce_compressed_gradients(
        self, 
        gradients: Dict[str, torch.Tensor],
        group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform all-reduce operation on compressed gradients.
        
        Args:
            gradients: Dictionary of gradient tensors to reduce
            group: Process group for reduction (default group if None)
            
        Returns:
            Dictionary of reduced gradient tensors
        """
        if not self.enable_compression or not dist.is_initialized():
            return gradients
            
        reduced_gradients = {}
        
        for name, grad in gradients.items():
            if grad is None:
                reduced_gradients[name] = grad
                continue
                
            try:
                # Compress gradient before all-reduce
                compressed_grad, metadata = self.compressor.compress(
                    grad.data,
                    param_id=hash(name),
                    config=self.compression_config
                )
                
                # All-reduce compressed gradient
                dist.all_reduce(compressed_grad, group=group)
                compressed_grad /= self.world_size  # Average across processes
                
                # Decompress after reduction
                if hasattr(self.compressor, 'decompress'):
                    decompressed_grad = self.compressor.decompress(compressed_grad, metadata)
                    reduced_gradients[name] = decompressed_grad
                else:
                    reduced_gradients[name] = compressed_grad
                    
            except Exception as e:
                warnings.warn(f"Compressed all-reduce failed for {name}: {e}")
                # Fallback to uncompressed all-reduce
                dist.all_reduce(grad, group=group)
                grad /= self.world_size
                reduced_gradients[name] = grad
                
        return reduced_gradients
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics for monitoring."""
        if self.compression_stats['total_gradients_compressed'] > 0:
            avg_compression_ratio = (
                self.compression_stats['total_compression_ratio'] / 
                self.compression_stats['total_gradients_compressed']
            )
        else:
            avg_compression_ratio = 1.0
            
        return {
            'total_compressed': self.compression_stats['total_gradients_compressed'],
            'avg_compression_ratio': avg_compression_ratio,
            'compression_errors': len(self.compression_stats['compression_errors']),
            'world_size': self.world_size,
            'rank': self.rank,
            'compression_enabled': self.enable_compression
        }
    
    @contextmanager
    def compression_context(self, enable: bool = True):
        """Context manager to temporarily enable/disable compression."""
        original_state = self.enable_compression
        self.enable_compression = enable
        try:
            yield self
        finally:
            self.enable_compression = original_state


def create_mango_fsdp_module(
    module: torch.nn.Module,
    compression_config: Optional[CompressionConfig] = None,
    enable_compression: bool = True,
    **fsdp_kwargs
) -> FSDP:
    """
    Factory function to create FSDP module with MANGO compression.
    
    Args:
        module: PyTorch module to wrap with FSDP
        compression_config: MANGO-LRQ compression configuration
        enable_compression: Whether to enable gradient compression
        **fsdp_kwargs: Additional FSDP configuration arguments
        
    Returns:
        FSDP module with MANGO compression hooks
    """
    
    # Default compression config
    if compression_config is None:
        compression_config = CompressionConfig(
            rank=4,
            bits_P=8,
            bits_Q=8,
            momentum_precision='fp16',
            use_nf4=True,
            error_feedback=True,
            variance_reduction=True
        )
    
    # Create MANGO FSDP reducer
    reducer = MANGOFSDPReducer(
        compression_config=compression_config,
        enable_compression=enable_compression
    )
    
    # Create and return FSDP wrapper
    return reducer.create_fsdp_wrapper(module, **fsdp_kwargs)


class MANGOFSDPOptimizer:
    """
    Optimizer wrapper for FSDP + MANGO integration.
    
    Provides seamless integration between FSDP sharded parameters
    and MANGO-LRQ compression for distributed optimization.
    """
    
    def __init__(
        self,
        fsdp_module: FSDP,
        optimizer_class: type = EnhancedMANGO,
        compression_config: Optional[CompressionConfig] = None,
        **optimizer_kwargs
    ):
        """
        Initialize MANGO FSDP optimizer.
        
        Args:
            fsdp_module: FSDP-wrapped module
            optimizer_class: Optimizer class to use
            compression_config: Compression configuration
            **optimizer_kwargs: Arguments for optimizer initialization
        """
        self.fsdp_module = fsdp_module
        self.compression_config = compression_config or CompressionConfig()
        
        # Create optimizer with FSDP parameters
        self.optimizer = optimizer_class(
            fsdp_module.parameters(),
            compression_config=self.compression_config,
            **optimizer_kwargs
        )
        
        # FSDP-specific state
        self.step_count = 0
        
    def step(self, closure=None):
        """Perform optimization step with FSDP coordination."""
        
        # Ensure gradients are available after FSDP backward
        with self.fsdp_module.summon_full_params(self.fsdp_module):
            loss = self.optimizer.step(closure)
            
        self.step_count += 1
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients with FSDP coordination."""
        self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get optimizer state dict with FSDP support."""
        with self.fsdp_module.summon_full_params(self.fsdp_module):
            return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict with FSDP support."""
        with self.fsdp_module.summon_full_params(self.fsdp_module):
            self.optimizer.load_state_dict(state_dict)


# Usage example and integration helpers
def setup_mango_fsdp_training(
    model: torch.nn.Module,
    compression_config: Optional[CompressionConfig] = None,
    optimizer_kwargs: Optional[Dict] = None,
    fsdp_kwargs: Optional[Dict] = None
) -> tuple[FSDP, MANGOFSDPOptimizer]:
    """
    Complete setup for MANGO + FSDP distributed training.
    
    Args:
        model: PyTorch model to train
        compression_config: MANGO compression configuration
        optimizer_kwargs: Arguments for MANGO optimizer
        fsdp_kwargs: Arguments for FSDP setup
        
    Returns:
        Tuple of (fsdp_model, mango_optimizer)
    """
    
    # Create FSDP model with MANGO compression
    fsdp_model = create_mango_fsdp_module(
        model,
        compression_config=compression_config,
        **(fsdp_kwargs or {})
    )
    
    # Create MANGO FSDP optimizer
    optimizer = MANGOFSDPOptimizer(
        fsdp_model,
        compression_config=compression_config,
        **(optimizer_kwargs or {})
    )
    
    print(f"MANGO FSDP training setup complete - "
          f"Model sharded across {dist.get_world_size() if dist.is_initialized() else 1} processes")
    
    return fsdp_model, optimizer