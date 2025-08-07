"""
Power Sampler for MANGO Energy-Aware Optimization

1Hz nvidia-smi polling with Queue-based data collection for
multi-objective reward computation in TinyFormer policy network.
"""

import subprocess
import threading
import time
import queue
from typing import Dict, Optional, List, Any
import logging
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class PowerSample:
    """Single power measurement sample."""
    timestamp: float
    gpu_power_watts: float
    gpu_temperature_c: float
    gpu_utilization_percent: float
    memory_usage_mb: float
    memory_total_mb: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'gpu_power_watts': self.gpu_power_watts,
            'gpu_temperature_c': self.gpu_temperature_c,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_total_mb': self.memory_total_mb,
            'memory_utilization_percent': (self.memory_usage_mb / max(self.memory_total_mb, 1.0)) * 100.0
        }


class PowerSampler:
    """
    High-frequency power sampling using nvidia-smi with Queue-based data flow.
    
    Designed for energy-aware multi-objective optimization in MANGO-LRQ.
    Samples at 1Hz by default and maintains a queue for real-time access.
    """
    
    def __init__(
        self,
        sampling_interval: float = 1.0,  # 1Hz sampling as specified in planv4.md
        queue_size: int = 1000,
        gpu_device_id: int = 0,
        enable_logging: bool = True
    ):
        """
        Initialize power sampler.
        
        Args:
            sampling_interval: Time between power measurements in seconds
            queue_size: Maximum size of power sample queue
            gpu_device_id: GPU device ID to monitor
            enable_logging: Enable logging for debugging
        """
        self.sampling_interval = sampling_interval
        self.gpu_device_id = gpu_device_id
        self.enable_logging = enable_logging
        
        # Data structures
        self.power_queue: queue.Queue[PowerSample] = queue.Queue(maxsize=queue_size)
        self.sample_history: deque[PowerSample] = deque(maxlen=queue_size)
        
        # Threading control
        self.sampling_thread: Optional[threading.Thread] = None
        self.is_sampling: bool = False
        self.stop_event = threading.Event()
        
        # Statistics tracking
        self.total_samples: int = 0
        self.failed_samples: int = 0
        self.start_time: Optional[float] = None
        
        # EWMA state for smoothed power metrics
        self.ewma_power: Optional[float] = None
        self.ewma_alpha: float = 0.1  # Exponential smoothing factor
        
        # Logging setup
        if enable_logging:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None
        
        # Check nvidia-smi availability
        self._check_nvidia_smi()
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available and working."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--help'],
                capture_output=True,
                check=True,
                timeout=5
            )
            if self.logger:
                self.logger.info("nvidia-smi is available for power monitoring")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            if self.logger:
                self.logger.error("nvidia-smi not available - power monitoring disabled")
            return False
    
    def start_sampling(self) -> bool:
        """
        Start power sampling in background thread.
        
        Returns:
            bool: True if sampling started successfully
        """
        if self.is_sampling:
            if self.logger:
                self.logger.warning("Power sampling already running")
            return True
        
        if not self._check_nvidia_smi():
            return False
        
        self.is_sampling = True
        self.stop_event.clear()
        self.start_time = time.time()
        
        self.sampling_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="PowerSampler"
        )
        self.sampling_thread.start()
        
        if self.logger:
            self.logger.info(f"Power sampling started at {self.sampling_interval}Hz")
        return True
    
    def stop_sampling(self) -> None:
        """Stop power sampling and cleanup resources."""
        if not self.is_sampling:
            return
        
        self.is_sampling = False
        self.stop_event.set()
        
        if self.sampling_thread and self.sampling_thread.is_alive():
            self.sampling_thread.join(timeout=3.0)
        
        if self.logger:
            self.logger.info(f"Power sampling stopped - collected {self.total_samples} samples")
    
    def _sampling_loop(self) -> None:
        """Main sampling loop running in background thread."""
        while self.is_sampling and not self.stop_event.is_set():
            try:
                # Sample power using nvidia-smi
                sample = self._sample_power()
                
                if sample is not None:
                    # Update EWMA
                    self._update_ewma(sample.gpu_power_watts)
                    
                    # Add to queue (non-blocking)
                    try:
                        self.power_queue.put_nowait(sample)
                    except queue.Full:
                        # Remove oldest sample and add new one
                        try:
                            self.power_queue.get_nowait()
                            self.power_queue.put_nowait(sample)
                        except queue.Empty:
                            pass
                    
                    # Add to history
                    self.sample_history.append(sample)
                    self.total_samples += 1
                else:
                    self.failed_samples += 1
                
                # Wait for next sampling interval
                self.stop_event.wait(self.sampling_interval)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in power sampling loop: {e}")
                self.failed_samples += 1
                self.stop_event.wait(self.sampling_interval)
    
    def _sample_power(self) -> Optional[PowerSample]:
        """
        Sample GPU power using nvidia-smi.
        
        Returns:
            PowerSample or None if sampling failed
        """
        try:
            # nvidia-smi query for comprehensive metrics
            cmd = [
                'nvidia-smi',
                f'--id={self.gpu_device_id}',
                '--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2.0
            )
            
            if result.returncode != 0:
                return None
            
            # Parse the output
            values = result.stdout.strip().split(', ')
            if len(values) != 5:
                return None
            
            # Convert to appropriate types
            power_watts = float(values[0])
            temperature_c = float(values[1])
            utilization_percent = float(values[2])
            memory_used_mb = float(values[3])
            memory_total_mb = float(values[4])
            
            return PowerSample(
                timestamp=time.time(),
                gpu_power_watts=power_watts,
                gpu_temperature_c=temperature_c,
                gpu_utilization_percent=utilization_percent,
                memory_usage_mb=memory_used_mb,
                memory_total_mb=memory_total_mb
            )
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError):
            return None
    
    def _update_ewma(self, power_watts: float) -> None:
        """Update exponentially weighted moving average of power."""
        if self.ewma_power is None:
            self.ewma_power = power_watts
        else:
            self.ewma_power = (1 - self.ewma_alpha) * self.ewma_power + self.ewma_alpha * power_watts
    
    def get_latest_sample(self) -> Optional[PowerSample]:
        """
        Get the most recent power sample from queue.
        
        Returns:
            PowerSample or None if queue is empty
        """
        try:
            return self.power_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_ewma_power(self) -> float:
        """
        Get exponentially weighted moving average power consumption.
        
        Returns:
            EWMA power in watts, or 0.0 if no samples available
        """
        return self.ewma_power or 0.0
    
    def get_power_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive power consumption statistics.
        
        Returns:
            Dictionary with power statistics
        """
        if not self.sample_history:
            return {
                'samples': 0,
                'mean_power_watts': 0.0,
                'max_power_watts': 0.0,
                'min_power_watts': 0.0,
                'std_power_watts': 0.0,
                'ewma_power_watts': 0.0,
                'mean_temperature_c': 0.0,
                'mean_utilization_percent': 0.0,
                'mean_memory_utilization_percent': 0.0
            }
        
        # Extract power values from recent samples
        powers = [s.gpu_power_watts for s in self.sample_history]
        temperatures = [s.gpu_temperature_c for s in self.sample_history]
        utilizations = [s.gpu_utilization_percent for s in self.sample_history]
        memory_utils = [(s.memory_usage_mb / max(s.memory_total_mb, 1.0)) * 100.0 
                       for s in self.sample_history]
        
        return {
            'samples': len(powers),
            'mean_power_watts': float(np.mean(powers)),
            'max_power_watts': float(np.max(powers)),
            'min_power_watts': float(np.min(powers)),
            'std_power_watts': float(np.std(powers)),
            'ewma_power_watts': self.get_ewma_power(),
            'mean_temperature_c': float(np.mean(temperatures)),
            'mean_utilization_percent': float(np.mean(utilizations)),
            'mean_memory_utilization_percent': float(np.mean(memory_utils)),
            'sampling_rate_hz': len(powers) / max(time.time() - self.start_time, 1.0) if self.start_time else 0.0,
            'failed_samples': self.failed_samples,
            'total_samples': self.total_samples
        }
    
    def get_energy_efficiency_metric(self, training_loss: float) -> float:
        """
        Calculate energy efficiency metric for multi-objective optimization.
        
        Args:
            training_loss: Current training loss value
            
        Returns:
            Energy efficiency score (higher is better)
        """
        if not self.sample_history or training_loss <= 0:
            return 0.0
        
        # Calculate average power consumption
        avg_power = np.mean([s.gpu_power_watts for s in self.sample_history])
        
        # Energy efficiency: loss improvement per watt
        # Higher efficiency = better loss reduction per unit power
        efficiency = 1.0 / (avg_power + 1.0)  # Add 1 to avoid division by zero
        
        return min(1.0, efficiency)
    
    def export_samples(self, filename: str, max_samples: Optional[int] = None) -> bool:
        """
        Export power samples to CSV file.
        
        Args:
            filename: Output CSV filename
            max_samples: Maximum number of samples to export (most recent)
            
        Returns:
            bool: True if export successful
        """
        try:
            import csv
            
            samples_to_export = list(self.sample_history)
            if max_samples and len(samples_to_export) > max_samples:
                samples_to_export = samples_to_export[-max_samples:]
            
            if not samples_to_export:
                return False
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'gpu_power_watts', 'gpu_temperature_c',
                    'gpu_utilization_percent', 'memory_usage_mb', 'memory_total_mb',
                    'memory_utilization_percent'
                ])
                
                for sample in samples_to_export:
                    sample_dict = sample.to_dict()
                    writer.writerow([
                        sample_dict['timestamp'],
                        sample_dict['gpu_power_watts'],
                        sample_dict['gpu_temperature_c'],
                        sample_dict['gpu_utilization_percent'],
                        sample_dict['memory_usage_mb'],
                        sample_dict['memory_total_mb'],
                        sample_dict['memory_utilization_percent']
                    ])
            
            if self.logger:
                self.logger.info(f"Exported {len(samples_to_export)} power samples to {filename}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to export power samples: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start_sampling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_sampling()


def create_power_sampler(
    sampling_interval: float = 1.0,
    queue_size: int = 1000,
    gpu_device_id: int = 0,
    enable_logging: bool = True
) -> PowerSampler:
    """
    Factory function to create a PowerSampler instance.
    
    Args:
        sampling_interval: Sampling interval in seconds
        queue_size: Queue size for power samples
        gpu_device_id: GPU device ID to monitor
        enable_logging: Enable logging
        
    Returns:
        PowerSampler instance
    """
    return PowerSampler(
        sampling_interval=sampling_interval,
        queue_size=queue_size,
        gpu_device_id=gpu_device_id,
        enable_logging=enable_logging
    )