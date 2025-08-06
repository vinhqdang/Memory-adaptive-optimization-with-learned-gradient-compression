"""
Utility functions and classes for MANGO optimizer.

Includes memory profiling, device management, and other helper functions.
"""

import torch
import numpy as np
import gc
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import defaultdict


class MemoryProfiler:
    """
    Tracks GPU and CPU memory usage during training.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory profiler.
        
        Args:
            device: Target device (auto-detect if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        
        # Memory tracking
        self.memory_history = []
        self.peak_memory = 0
        self.baseline_memory = self._get_current_memory()
        
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        if self.is_cuda and torch.cuda.is_available():
            # GPU memory
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            memory_info['gpu_total'] = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            memory_info['gpu_free'] = memory_info['gpu_total'] - memory_info['gpu_reserved']
        else:
            memory_info.update({
                'gpu_allocated': 0.0,
                'gpu_reserved': 0.0,
                'gpu_total': 0.0,
                'gpu_free': 0.0
            })
        
        # CPU memory
        process = psutil.Process(os.getpid())
        memory_info['cpu_used'] = process.memory_info().rss / 1024**3  # GB
        memory_info['cpu_percent'] = process.memory_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_total'] = system_memory.total / 1024**3  # GB
        memory_info['system_available'] = system_memory.available / 1024**3  # GB
        memory_info['system_used'] = (system_memory.total - system_memory.available) / 1024**3  # GB
        
        return memory_info
    
    def record_memory(self, step: int, tag: str = ""):
        """Record current memory usage."""
        current_memory = self._get_current_memory()
        current_memory.update({
            'step': step,
            'timestamp': time.time(),
            'tag': tag
        })
        
        self.memory_history.append(current_memory)
        
        # Update peak memory
        if self.is_cuda:
            current_peak = current_memory['gpu_allocated']
        else:
            current_peak = current_memory['cpu_used']
        
        if current_peak > self.peak_memory:
            self.peak_memory = current_peak
    
    def get_usage_ratio(self) -> float:
        """Get current memory usage ratio [0, 1]."""
        current_memory = self._get_current_memory()
        
        if self.is_cuda:
            return current_memory['gpu_allocated'] / max(current_memory['gpu_total'], 1.0)
        else:
            return current_memory['cpu_used'] / max(current_memory['system_total'], 1.0)
    
    def get_detailed_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        return self._get_current_memory()
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB."""
        return self.peak_memory
    
    def get_memory_savings(self) -> float:
        """Estimate memory savings compared to baseline."""
        if not self.memory_history:
            return 0.0
        
        current_memory = self.memory_history[-1]
        if self.is_cuda:
            current_usage = current_memory['gpu_allocated']
            baseline_usage = self.baseline_memory.get('gpu_allocated', current_usage)
        else:
            current_usage = current_memory['cpu_used']
            baseline_usage = self.baseline_memory.get('cpu_used', current_usage)
        
        if baseline_usage > 0:
            return max(0.0, 1.0 - current_usage / baseline_usage)
        return 0.0
    
    def clear_cache(self):
        """Clear GPU/CPU cache to free memory."""
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        if not self.memory_history:
            return {}
        
        recent_history = self.memory_history[-10:]  # Last 10 records
        
        summary = {
            'current': self._get_current_memory(),
            'peak_memory_gb': self.peak_memory,
            'memory_savings': self.get_memory_savings(),
            'usage_ratio': self.get_usage_ratio(),
            'history_length': len(self.memory_history)
        }
        
        # Compute trends
        if len(recent_history) >= 2:
            if self.is_cuda:
                usage_values = [h['gpu_allocated'] for h in recent_history]
            else:
                usage_values = [h['cpu_used'] for h in recent_history]
            
            summary['usage_trend'] = usage_values[-1] - usage_values[0]
            summary['avg_usage'] = float(np.mean(usage_values))
            summary['usage_variance'] = float(np.var(usage_values))
        
        return summary


class DeviceManager:
    """
    Manages device placement and memory optimization for CUDA environments.
    """
    
    def __init__(self):
        """Initialize device manager."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        
        if self.is_cuda:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_properties = [
                torch.cuda.get_device_properties(i) for i in range(self.gpu_count)
            ]
        else:
            self.gpu_count = 0
            self.gpu_properties = []
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'device': str(self.device),
            'is_cuda': self.is_cuda,
            'gpu_count': self.gpu_count
        }
        
        if self.is_cuda:
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_names': [prop.name for prop in self.gpu_properties],
                'gpu_memory_gb': [prop.total_memory / 1024**3 for prop in self.gpu_properties],
                'gpu_compute_capability': [f"{prop.major}.{prop.minor}" for prop in self.gpu_properties]
            })
        
        return info
    
    def optimize_memory_settings(self):
        """Apply memory optimization settings."""
        if self.is_cuda:
            # Enable memory fraction control if available
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            except:
                pass
            
            # Enable memory pooling for better allocation
            try:
                torch.cuda.empty_cache()
            except:
                pass
    
    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the managed device."""
        return tensor.to(self.device)
    
    def get_recommended_batch_size(self, model_size_gb: float) -> int:
        """Estimate recommended batch size based on available memory."""
        if not self.is_cuda:
            return 32  # Conservative default for CPU
        
        # Get available GPU memory
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory_gb = available_memory_gb - torch.cuda.memory_allocated(0) / 1024**3
        
        # Conservative estimate: use 70% of free memory, assume 4x memory overhead
        usable_memory_gb = free_memory_gb * 0.7
        estimated_batch_size = max(1, int(usable_memory_gb / (model_size_gb * 4)))
        
        # Clamp to reasonable range
        return min(max(estimated_batch_size, 1), 128)


class CompressionAnalyzer:
    """
    Analyzes compression effectiveness and provides insights.
    """
    
    def __init__(self):
        """Initialize compression analyzer."""
        self.compression_history = []
        self.performance_metrics = defaultdict(list)
    
    def record_compression_step(
        self,
        step: int,
        compression_config: Dict,
        loss: float,
        memory_usage: float,
        compression_ratio: float,
        compression_error: float
    ):
        """Record compression step for analysis."""
        record = {
            'step': step,
            'gradient_bits': compression_config.get('gradient_bits', 32),
            'momentum_bits': compression_config.get('momentum_bits', 32),
            'sparsity_ratio': compression_config.get('sparsity_ratio', 0.0),
            'loss': loss,
            'memory_usage_gb': memory_usage,
            'compression_ratio': compression_ratio,
            'compression_error': compression_error,
            'timestamp': time.time()
        }
        
        self.compression_history.append(record)
    
    def analyze_compression_effectiveness(self) -> Dict[str, Any]:
        """Analyze overall compression effectiveness."""
        if len(self.compression_history) < 10:
            return {}
        
        recent_records = self.compression_history[-100:]  # Last 100 steps
        
        # Compute averages
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in recent_records])
        avg_memory_usage = np.mean([r['memory_usage_gb'] for r in recent_records])
        avg_compression_error = np.mean([r['compression_error'] for r in recent_records])
        
        # Analyze compression-loss correlation
        losses = [r['loss'] for r in recent_records]
        compression_ratios = [r['compression_ratio'] for r in recent_records]
        
        if len(losses) > 1 and np.var(compression_ratios) > 0:
            correlation = np.corrcoef(losses, compression_ratios)[0, 1]
        else:
            correlation = 0.0
        
        # Analyze compression stability
        gradient_bits = [r['gradient_bits'] for r in recent_records]
        momentum_bits = [r['momentum_bits'] for r in recent_records]
        sparsity_ratios = [r['sparsity_ratio'] for r in recent_records]
        
        analysis = {
            'avg_compression_ratio': float(avg_compression_ratio),
            'avg_memory_usage_gb': float(avg_memory_usage),
            'avg_compression_error': float(avg_compression_error),
            'loss_compression_correlation': float(correlation),
            'gradient_bits_stability': float(np.std(gradient_bits)),
            'momentum_bits_stability': float(np.std(momentum_bits)),
            'sparsity_stability': float(np.std(sparsity_ratios)),
            'total_steps_analyzed': len(recent_records)
        }
        
        # Phase analysis
        if len(self.compression_history) >= 300:
            analysis.update(self._analyze_compression_phases())
        
        return analysis
    
    def _analyze_compression_phases(self) -> Dict[str, Any]:
        """Analyze compression behavior across training phases."""
        history = self.compression_history
        n = len(history)
        
        # Divide into phases
        phase1 = history[:n//3]          # Early training
        phase2 = history[n//3:2*n//3]    # Mid training  
        phase3 = history[2*n//3:]        # Late training
        
        phases = {'early': phase1, 'mid': phase2, 'late': phase3}
        phase_analysis = {}
        
        for phase_name, phase_data in phases.items():
            if not phase_data:
                continue
                
            phase_analysis[f'{phase_name}_avg_compression_ratio'] = float(
                np.mean([r['compression_ratio'] for r in phase_data])
            )
            phase_analysis[f'{phase_name}_avg_gradient_bits'] = float(
                np.mean([r['gradient_bits'] for r in phase_data])
            )
            phase_analysis[f'{phase_name}_avg_sparsity'] = float(
                np.mean([r['sparsity_ratio'] for r in phase_data])
            )
        
        return phase_analysis
    
    def get_compression_recommendations(self) -> Dict[str, Any]:
        """Provide compression optimization recommendations."""
        analysis = self.analyze_compression_effectiveness()
        if not analysis:
            return {}
        
        recommendations = {}
        
        # Memory usage recommendations
        if analysis['avg_memory_usage_gb'] > 8.0:  # High memory usage
            recommendations['memory'] = "Consider more aggressive compression (lower bits, higher sparsity)"
        elif analysis['avg_memory_usage_gb'] < 2.0:  # Low memory usage
            recommendations['memory'] = "Can afford higher precision for better accuracy"
        
        # Compression stability recommendations
        if analysis['gradient_bits_stability'] > 5.0:  # High instability
            recommendations['stability'] = "Compression policy is unstable, consider smoothing"
        
        # Error recommendations
        if analysis['avg_compression_error'] > 1.0:  # High compression error
            recommendations['error'] = "High compression error detected, consider reducing compression ratio"
        
        # Correlation recommendations
        if abs(analysis['loss_compression_correlation']) > 0.5:
            if analysis['loss_compression_correlation'] > 0:
                recommendations['correlation'] = "Positive loss-compression correlation suggests over-compression"
            else:
                recommendations['correlation'] = "Negative correlation suggests compression is helping convergence"
        
        return recommendations


def setup_cuda_environment():
    """
    Set up optimal CUDA environment for MANGO training.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Set optimal CUDA settings
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    # Enable Tensor Core usage if available
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device


def estimate_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """
    Estimate memory usage of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with memory estimates in GB
    """
    param_memory = 0
    buffer_memory = 0
    
    # Calculate parameter memory
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    # Calculate buffer memory
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    # Convert to GB
    param_memory_gb = param_memory / 1024**3
    buffer_memory_gb = buffer_memory / 1024**3
    total_memory_gb = param_memory_gb + buffer_memory_gb
    
    # Estimate gradient memory (same as parameters)
    gradient_memory_gb = param_memory_gb
    
    # Estimate optimizer state memory (approximation for Adam)
    optimizer_memory_gb = param_memory_gb * 2  # exp_avg + exp_avg_sq
    
    return {
        'parameters_gb': param_memory_gb,
        'buffers_gb': buffer_memory_gb,
        'gradients_gb': gradient_memory_gb,
        'optimizer_states_gb': optimizer_memory_gb,
        'total_model_gb': total_memory_gb,
        'estimated_training_gb': total_memory_gb + gradient_memory_gb + optimizer_memory_gb
    }