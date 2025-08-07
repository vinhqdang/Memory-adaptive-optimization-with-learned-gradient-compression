"""
Automatic Mixed Precision (AMP) Support for MANGO

Integrates PyTorch's automatic mixed precision with MANGO optimizer,
ensuring proper gradient scaling and precision handling during compression.
"""

import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Callable, List
import warnings
from contextlib import contextmanager


class MANGOGradScaler(GradScaler):
    """
    Extended GradScaler that works with MANGO's compression pipeline.
    
    Handles gradient scaling before and after compression to maintain
    numerical stability with mixed precision training.
    """
    
    def __init__(
        self,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        compression_aware: bool = True
    ):
        """
        Initialize MANGO-compatible gradient scaler.
        
        Args:
            init_scale: Initial scale factor
            growth_factor: Factor by which scale is multiplied during scale growth
            backoff_factor: Factor by which scale is multiplied during scale decay
            growth_interval: Number of consecutive unskipped steps before growth
            enabled: Whether gradient scaling is enabled
            compression_aware: Whether to adapt scaling for compression
        """
        super().__init__(init_scale, growth_factor, backoff_factor, growth_interval, enabled)
        
        self.compression_aware = compression_aware
        self.compression_history = []
        self.adaptive_scale_factor = 1.0
        
        # Statistics for compression-aware scaling
        self.compression_stats = {
            'total_compressions': 0,
            'overflow_count': 0,
            'underflow_count': 0,
            'scale_adaptations': 0
        }
    
    def scale_for_compression(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply scaling before compression to prevent underflow.
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            Scaled gradients ready for compression
        """
        if not self.is_enabled():
            return gradients
        
        scaled_gradients = {}
        current_scale = self.get_scale()
        
        for param_name, grad in gradients.items():
            if grad is not None:
                # Apply current scale
                scaled_grad = grad * current_scale
                
                # Additional compression-aware scaling
                if self.compression_aware:
                    compression_scale = self._get_compression_scale(grad)
                    scaled_grad = scaled_grad * compression_scale
                
                scaled_gradients[param_name] = scaled_grad
            else:
                scaled_gradients[param_name] = None
        
        return scaled_gradients
    
    def unscale_after_compression(
        self, 
        compressed_gradients: Dict[str, torch.Tensor],
        compression_metadata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Remove scaling after compression.
        
        Args:
            compressed_gradients: Compressed gradients
            compression_metadata: Metadata from compression
            
        Returns:
            Unscaled gradients
        """
        if not self.is_enabled():
            return compressed_gradients
        
        unscaled_gradients = {}
        current_scale = self.get_scale()
        
        for param_name, grad in compressed_gradients.items():
            if grad is not None:
                # Remove compression-aware scaling
                if self.compression_aware:
                    compression_scale = compression_metadata.get(f'{param_name}_compression_scale', 1.0)
                    grad = grad / compression_scale
                
                # Remove main scale
                unscaled_grad = grad / current_scale
                unscaled_gradients[param_name] = unscaled_grad
            else:
                unscaled_gradients[param_name] = None
        
        return unscaled_gradients
    
    def _get_compression_scale(self, gradient: torch.Tensor) -> float:
        """
        Compute compression-aware scaling factor.
        
        Args:
            gradient: Input gradient tensor
            
        Returns:
            Additional scaling factor for compression
        """
        if not gradient.is_floating_point():
            return 1.0
        
        # Analyze gradient statistics
        grad_norm = gradient.norm().item()
        grad_max = gradient.abs().max().item()
        
        # Adapt scale based on gradient magnitude
        if grad_max > 0:
            # Prevent quantization errors for very small gradients
            if grad_max < 1e-6:
                compression_scale = 1000.0  # Scale up small gradients
            elif grad_max > 10.0:
                compression_scale = 0.1     # Scale down large gradients
            else:
                compression_scale = 1.0
        else:
            compression_scale = 1.0
        
        # Update adaptive scale factor (exponential moving average)
        self.adaptive_scale_factor = 0.99 * self.adaptive_scale_factor + 0.01 * compression_scale
        
        return compression_scale
    
    def check_compression_overflow(
        self, 
        original_gradients: Dict[str, torch.Tensor],
        compressed_gradients: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Check for overflow/underflow after compression.
        
        Args:
            original_gradients: Original gradients before compression
            compressed_gradients: Gradients after compression
            
        Returns:
            True if overflow/underflow detected
        """
        overflow_detected = False
        
        for param_name in original_gradients:
            orig_grad = original_gradients[param_name]
            comp_grad = compressed_gradients.get(param_name)
            
            if orig_grad is not None and comp_grad is not None:
                # Check for NaN or Inf
                if torch.isnan(comp_grad).any() or torch.isinf(comp_grad).any():
                    overflow_detected = True
                    self.compression_stats['overflow_count'] += 1
                    break
                
                # Check for significant magnitude loss (underflow)
                orig_norm = orig_grad.norm()
                comp_norm = comp_grad.norm()
                
                if orig_norm > 0 and comp_norm / orig_norm < 1e-6:
                    overflow_detected = True
                    self.compression_stats['underflow_count'] += 1
                    break
        
        if overflow_detected and self.compression_aware:
            # Adapt scaling for next iteration
            self._adapt_scale_for_compression()
        
        return overflow_detected
    
    def _adapt_scale_for_compression(self):
        """Adapt scaling based on compression behavior."""
        # Reduce scale if too many overflows
        if self.compression_stats['overflow_count'] > 10:
            self.set_scale(self.get_scale() * 0.5)
            self.compression_stats['scale_adaptations'] += 1
            self.compression_stats['overflow_count'] = 0
        
        # Increase scale if too many underflows
        elif self.compression_stats['underflow_count'] > 10:
            self.set_scale(self.get_scale() * 1.5)
            self.compression_stats['scale_adaptations'] += 1
            self.compression_stats['underflow_count'] = 0
    
    def get_compression_stats(self) -> Dict[str, int]:
        """Get compression-related scaling statistics."""
        return self.compression_stats.copy()


class AMPTrainingContext:
    """
    Context manager for AMP training with MANGO optimizer.
    """
    
    def __init__(
        self,
        optimizer,
        scaler: Optional[MANGOGradScaler] = None,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize AMP training context.
        
        Args:
            optimizer: MANGO optimizer instance
            scaler: Gradient scaler (created if None)
            enabled: Whether to enable AMP
            dtype: Mixed precision dtype
        """
        self.optimizer = optimizer
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
        if self.enabled:
            self.scaler = scaler or MANGOGradScaler()
        else:
            self.scaler = None
        
        # Store original optimizer methods for restoration
        self._original_step = None
        self._original_zero_grad = None
        
        if self.enabled:
            self._wrap_optimizer()
    
    def _wrap_optimizer(self):
        """Wrap optimizer methods to handle AMP scaling."""
        # Store original methods
        self._original_step = self.optimizer.step
        self._original_zero_grad = self.optimizer.zero_grad
        
        def amp_step(closure: Optional[Callable] = None):
            """AMP-aware optimizer step."""
            if closure is not None:
                with autocast(dtype=self.dtype):
                    closure()
            
            # Unscale gradients before compression
            self.scaler.unscale_(self.optimizer)
            
            # Check for gradient overflow
            if not self.scaler.state_dict()['scale'] == 0:
                # Perform original step (which includes compression)
                self._original_step()
                
                # Update scaler
                self.scaler.step(lambda: None)  # Empty closure since we already stepped
            
            self.scaler.update()
        
        def amp_zero_grad(set_to_none: bool = False):
            """AMP-compatible zero_grad."""
            self._original_zero_grad(set_to_none)
        
        # Replace optimizer methods
        self.optimizer.step = amp_step
        self.optimizer.zero_grad = amp_zero_grad
    
    def _restore_optimizer(self):
        """Restore original optimizer methods."""
        if self._original_step is not None:
            self.optimizer.step = self._original_step
        if self._original_zero_grad is not None:
            self.optimizer.zero_grad = self._original_zero_grad
    
    @contextmanager
    def autocast_context(self):
        """Get autocast context."""
        if self.enabled:
            with autocast(dtype=self.dtype):
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """Perform scaled backward pass."""
        if self.enabled:
            scaled_loss = self.scale_loss(loss)
            scaled_loss.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)
    
    def get_scale(self) -> float:
        """Get current gradient scale."""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def is_enabled(self) -> bool:
        """Check if AMP is enabled."""
        return self.enabled
    
    def __enter__(self):
        """Enter AMP context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit AMP context and restore optimizer."""
        if self.enabled:
            self._restore_optimizer()


def create_amp_context(
    optimizer,
    enabled: bool = True,
    init_scale: float = 2.**16,
    dtype: torch.dtype = torch.float16
) -> AMPTrainingContext:
    """
    Create AMP training context for MANGO optimizer.
    
    Args:
        optimizer: MANGO optimizer instance
        enabled: Whether to enable AMP
        init_scale: Initial gradient scale
        dtype: Mixed precision dtype
        
    Returns:
        AMP training context
    """
    scaler = MANGOGradScaler(init_scale=init_scale, enabled=enabled)
    return AMPTrainingContext(optimizer, scaler, enabled, dtype)


@contextmanager
def amp_training_step(
    amp_context: AMPTrainingContext,
    model: torch.nn.Module,
    loss_fn: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor
):
    """
    Context manager for a complete AMP training step.
    
    Args:
        amp_context: AMP training context
        model: PyTorch model
        loss_fn: Loss function
        inputs: Input tensor
        targets: Target tensor
    """
    # Zero gradients
    amp_context.optimizer.zero_grad()
    
    # Forward pass with autocast
    with amp_context.autocast_context():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    
    # Scaled backward pass
    amp_context.backward(loss)
    
    # Optimizer step (includes compression and scaling)
    amp_context.optimizer.step()
    
    yield loss, outputs


class AMPCompatibilityChecker:
    """
    Checks model and optimizer compatibility with AMP.
    """
    
    @staticmethod
    def check_model_compatibility(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Check if model is compatible with AMP.
        
        Args:
            model: PyTorch model
            
        Returns:
            Compatibility report
        """
        issues = []
        recommendations = []
        
        # Check for problematic layer types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                issues.append(f"BatchNorm1d in {name} may cause issues with AMP")
                recommendations.append(f"Consider using LayerNorm instead of BatchNorm1d in {name}")
            
            if hasattr(module, 'dtype') and module.dtype not in [torch.float32, torch.float16]:
                issues.append(f"Module {name} uses unsupported dtype {module.dtype}")
                recommendations.append(f"Ensure {name} uses float32 or float16")
        
        # Check parameter dtypes
        param_dtypes = set()
        for param in model.parameters():
            param_dtypes.add(param.dtype)
        
        if len(param_dtypes) > 2:  # More than float32 and float16
            issues.append(f"Model has parameters with multiple dtypes: {param_dtypes}")
            recommendations.append("Ensure all parameters are float32 or float16")
        
        compatibility_score = max(0, 100 - len(issues) * 20)
        
        return {
            'compatible': len(issues) == 0,
            'compatibility_score': compatibility_score,
            'issues': issues,
            'recommendations': recommendations,
            'parameter_dtypes': list(param_dtypes)
        }
    
    @staticmethod
    def check_optimizer_compatibility(optimizer) -> Dict[str, Any]:
        """
        Check if optimizer is compatible with AMP.
        
        Args:
            optimizer: Optimizer instance
            
        Returns:
            Compatibility report
        """
        issues = []
        recommendations = []
        
        # Check if it's a MANGO optimizer
        if not hasattr(optimizer, 'compress'):
            issues.append("Optimizer doesn't appear to be a MANGO optimizer")
            recommendations.append("Use MANGO optimizer for best AMP compatibility")
        
        # Check parameter groups
        for i, group in enumerate(optimizer.param_groups):
            for param in group['params']:
                if param.dtype not in [torch.float32, torch.float16]:
                    issues.append(f"Parameter group {i} contains unsupported dtype {param.dtype}")
                    recommendations.append(f"Convert parameter group {i} to float32 or float16")
        
        compatibility_score = max(0, 100 - len(issues) * 25)
        
        return {
            'compatible': len(issues) == 0,
            'compatibility_score': compatibility_score,
            'issues': issues,
            'recommendations': recommendations
        }
    
    @classmethod
    def full_compatibility_check(cls, model: torch.nn.Module, optimizer) -> Dict[str, Any]:
        """
        Perform full AMP compatibility check.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            
        Returns:
            Complete compatibility report
        """
        model_report = cls.check_model_compatibility(model)
        optimizer_report = cls.check_optimizer_compatibility(optimizer)
        
        overall_compatible = model_report['compatible'] and optimizer_report['compatible']
        overall_score = (model_report['compatibility_score'] + optimizer_report['compatibility_score']) / 2
        
        return {
            'overall_compatible': overall_compatible,
            'overall_score': overall_score,
            'model_compatibility': model_report,
            'optimizer_compatibility': optimizer_report,
            'recommendations': model_report['recommendations'] + optimizer_report['recommendations']
        }