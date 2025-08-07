"""
Memory Profiler for MANGO

PyTorch 2.5 memory profiler integration for tracking peak/median GPU memory usage
during training with detailed breakdown by operations.
"""

import torch
import torch.profiler
import time
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager
import json
import os


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    allocated: int
    reserved: int
    peak_allocated: int
    peak_reserved: int
    active: int
    inactive: int


@dataclass
class MemoryStats:
    """Aggregated memory statistics."""
    peak_allocated_gb: float
    peak_reserved_gb: float
    median_allocated_gb: float
    median_reserved_gb: float
    mean_allocated_gb: float
    mean_reserved_gb: float
    min_allocated_gb: float
    max_allocated_gb: float
    total_snapshots: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'peak_allocated_gb': self.peak_allocated_gb,
            'peak_reserved_gb': self.peak_reserved_gb,
            'median_allocated_gb': self.median_allocated_gb,
            'median_reserved_gb': self.median_reserved_gb,
            'mean_allocated_gb': self.mean_allocated_gb,
            'mean_reserved_gb': self.mean_reserved_gb,
            'min_allocated_gb': self.min_allocated_gb,
            'max_allocated_gb': self.max_allocated_gb,
            'total_snapshots': self.total_snapshots
        }


class GPUMemoryTracker:
    """
    Tracks GPU memory usage with high temporal resolution.
    """
    
    def __init__(self, device: Optional[torch.device] = None, sampling_interval: float = 0.1):
        """
        Initialize memory tracker.
        
        Args:
            device: CUDA device to track (auto-detect if None)
            sampling_interval: Sampling interval in seconds
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sampling_interval = sampling_interval
        
        if not torch.cuda.is_available() or self.device.type != 'cuda':
            print("Warning: CUDA not available, memory tracking disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.snapshots: List[MemorySnapshot] = []
        self.tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def start_tracking(self):
        """Start continuous memory tracking."""
        if not self.enabled or self.tracking:
            return
        
        self.tracking = True
        self.snapshots.clear()
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
    
    def stop_tracking(self):
        """Stop memory tracking."""
        if not self.enabled or not self.tracking:
            return
        
        self.tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        
    def _tracking_loop(self):
        """Main tracking loop running in separate thread."""
        while self.tracking:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Memory tracking error: {e}")
                break
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a single memory snapshot."""
        memory_stats = torch.cuda.memory_stats(self.device)
        
        return MemorySnapshot(
            timestamp=time.time(),
            allocated=memory_stats.get('allocated_bytes.all.current', 0),
            reserved=memory_stats.get('reserved_bytes.all.current', 0),
            peak_allocated=memory_stats.get('allocated_bytes.all.peak', 0),
            peak_reserved=memory_stats.get('reserved_bytes.all.peak', 0),
            active=memory_stats.get('active_bytes.all.current', 0),
            inactive=memory_stats.get('inactive_bytes.all.current', 0)
        )
    
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not self.enabled or not self.snapshots:
            return MemoryStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        allocated_values = [s.allocated for s in self.snapshots]
        reserved_values = [s.reserved for s in self.snapshots]
        
        # Convert bytes to GB
        allocated_gb = np.array(allocated_values) / (1024**3)
        reserved_gb = np.array(reserved_values) / (1024**3)
        
        return MemoryStats(
            peak_allocated_gb=float(np.max(allocated_gb)),
            peak_reserved_gb=float(np.max(reserved_gb)),
            median_allocated_gb=float(np.median(allocated_gb)),
            median_reserved_gb=float(np.median(reserved_gb)),
            mean_allocated_gb=float(np.mean(allocated_gb)),
            mean_reserved_gb=float(np.mean(reserved_gb)),
            min_allocated_gb=float(np.min(allocated_gb)),
            max_allocated_gb=float(np.max(allocated_gb)),
            total_snapshots=len(self.snapshots)
        )
    
    def reset(self):
        """Reset tracking data and CUDA memory stats."""
        self.stop_tracking()
        self.snapshots.clear()
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_memory_timeline(self) -> Tuple[List[float], List[float], List[float]]:
        """Get memory usage timeline for plotting."""
        if not self.snapshots:
            return [], [], []
        
        timestamps = [s.timestamp - self.snapshots[0].timestamp for s in self.snapshots]
        allocated_gb = [s.allocated / (1024**3) for s in self.snapshots]
        reserved_gb = [s.reserved / (1024**3) for s in self.snapshots]
        
        return timestamps, allocated_gb, reserved_gb


class ProfilerConfig:
    """Configuration for PyTorch profiler."""
    
    def __init__(
        self,
        activities: List[torch.profiler.ProfilerActivity] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = True
    ):
        self.activities = activities or [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ]
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules


class MANGOMemoryProfiler:
    """
    Comprehensive memory profiler for MANGO optimizer.
    
    Combines continuous GPU memory tracking with PyTorch's detailed profiler
    to provide insights into memory usage patterns during compression.
    """
    
    def __init__(
        self, 
        device: Optional[torch.device] = None,
        config: Optional[ProfilerConfig] = None,
        output_dir: str = "./profiler_logs"
    ):
        """
        Initialize MANGO memory profiler.
        
        Args:
            device: CUDA device to profile
            config: Profiler configuration
            output_dir: Directory to save profiler outputs
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or ProfilerConfig()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.gpu_tracker = GPUMemoryTracker(device)
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiling_active = False
        
        # Statistics storage
        self.compression_memory_stats = []
        self.operation_memory_breakdown = {}
        
    @contextmanager
    def profile_step(self, step_name: str = "training_step"):
        """
        Context manager for profiling a single training step.
        
        Args:
            step_name: Name identifier for this step
        """
        if not self.gpu_tracker.enabled:
            yield
            return
        
        # Start GPU tracking
        self.gpu_tracker.start_tracking()
        
        # Create PyTorch profiler
        with torch.profiler.profile(
            activities=self.config.activities,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
            on_trace_ready=self._save_trace
        ) as prof:
            self.profiler = prof
            self.profiling_active = True
            
            try:
                yield self
            finally:
                self.profiling_active = False
                self.gpu_tracker.stop_tracking()
                
                # Store step statistics
                stats = self.gpu_tracker.get_current_stats()
                self.compression_memory_stats.append({
                    'step_name': step_name,
                    'timestamp': time.time(),
                    'stats': stats.to_dict()
                })
    
    def _save_trace(self, prof: torch.profiler.profile):
        """Save profiler trace to file."""
        timestamp = int(time.time())
        trace_path = os.path.join(self.output_dir, f"trace_{timestamp}.json")
        
        try:
            prof.export_chrome_trace(trace_path)
            print(f"Profiler trace saved to {trace_path}")
        except Exception as e:
            print(f"Failed to save profiler trace: {e}")
    
    @contextmanager
    def profile_compression_operation(self, operation_name: str):
        """
        Profile specific compression operations.
        
        Args:
            operation_name: Name of the compression operation
        """
        if not self.profiling_active:
            yield
            return
        
        # Record memory before operation
        pre_stats = self.gpu_tracker._take_snapshot()
        
        with torch.profiler.record_function(operation_name):
            yield
        
        # Record memory after operation
        post_stats = self.gpu_tracker._take_snapshot()
        
        # Calculate memory delta
        memory_delta = post_stats.allocated - pre_stats.allocated
        
        # Store operation breakdown
        if operation_name not in self.operation_memory_breakdown:
            self.operation_memory_breakdown[operation_name] = []
        
        self.operation_memory_breakdown[operation_name].append({
            'memory_delta_bytes': memory_delta,
            'memory_delta_mb': memory_delta / (1024**2),
            'pre_allocated_gb': pre_stats.allocated / (1024**3),
            'post_allocated_gb': post_stats.allocated / (1024**3)
        })
    
    def get_compression_memory_report(self) -> Dict:
        """Generate comprehensive memory usage report."""
        if not self.compression_memory_stats:
            return {"error": "No profiling data available"}
        
        # Overall statistics
        all_stats = [entry['stats'] for entry in self.compression_memory_stats]
        
        overall_stats = {
            'peak_memory_gb': max(s['peak_allocated_gb'] for s in all_stats),
            'avg_peak_memory_gb': np.mean([s['peak_allocated_gb'] for s in all_stats]),
            'median_memory_gb': np.median([s['median_allocated_gb'] for s in all_stats]),
            'total_profiled_steps': len(all_stats)
        }
        
        # Operation breakdown
        operation_breakdown = {}
        for op_name, measurements in self.operation_memory_breakdown.items():
            deltas = [m['memory_delta_mb'] for m in measurements]
            operation_breakdown[op_name] = {
                'avg_memory_delta_mb': float(np.mean(deltas)),
                'max_memory_delta_mb': float(np.max(deltas)),
                'min_memory_delta_mb': float(np.min(deltas)),
                'total_calls': len(measurements),
                'total_memory_impact_mb': float(np.sum(deltas))
            }
        
        # Memory efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics()
        
        return {
            'overall_stats': overall_stats,
            'operation_breakdown': operation_breakdown,
            'efficiency_metrics': efficiency_metrics,
            'profiling_summary': {
                'total_steps_profiled': len(self.compression_memory_stats),
                'total_operations_tracked': len(self.operation_memory_breakdown),
                'device': str(self.device),
                'profiling_enabled': self.gpu_tracker.enabled
            }
        }
    
    def _calculate_efficiency_metrics(self) -> Dict:
        """Calculate memory efficiency metrics."""
        if not self.operation_memory_breakdown:
            return {}
        
        # Compression-specific efficiency
        compression_ops = ['low_rank_projection', 'quantization', 'error_feedback']
        total_compression_memory = 0
        compression_count = 0
        
        for op_name, measurements in self.operation_memory_breakdown.items():
            if any(comp_op in op_name.lower() for comp_op in compression_ops):
                total_compression_memory += sum(m['memory_delta_mb'] for m in measurements)
                compression_count += len(measurements)
        
        efficiency_metrics = {}
        if compression_count > 0:
            efficiency_metrics['avg_compression_overhead_mb'] = total_compression_memory / compression_count
            efficiency_metrics['total_compression_operations'] = compression_count
        
        return efficiency_metrics
    
    def save_report(self, filename: str = "memory_report.json"):
        """Save memory profiling report to file."""
        report = self.get_compression_memory_report()
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Memory profiling report saved to {report_path}")
        return report_path
    
    def reset(self):
        """Reset all profiling data."""
        self.gpu_tracker.reset()
        self.compression_memory_stats.clear()
        self.operation_memory_breakdown.clear()
    
    def get_peak_memory_gb(self) -> float:
        """Get peak memory usage in GB."""
        if not self.compression_memory_stats:
            return 0.0
        
        return max(
            entry['stats']['peak_allocated_gb'] 
            for entry in self.compression_memory_stats
        )
    
    def get_median_memory_gb(self) -> float:
        """Get median memory usage in GB."""
        if not self.compression_memory_stats:
            return 0.0
        
        medians = [
            entry['stats']['median_allocated_gb'] 
            for entry in self.compression_memory_stats
        ]
        return float(np.median(medians))


# Convenience functions for easy integration

def create_memory_profiler(output_dir: str = "./profiler_logs") -> MANGOMemoryProfiler:
    """Create a memory profiler with default settings."""
    return MANGOMemoryProfiler(output_dir=output_dir)


@contextmanager
def profile_memory_usage(profiler: MANGOMemoryProfiler, step_name: str = "step"):
    """Simple context manager for memory profiling."""
    with profiler.profile_step(step_name):
        yield profiler


def get_current_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    
    return torch.cuda.memory_allocated() / (1024**3)


def get_peak_gpu_memory_gb() -> float:
    """Get peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_peak_memory_stats():
    """Reset CUDA peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()