"""
Power and Energy Monitoring for MANGO Optimizer

Multi-objective reward function including power consumption tracking
through nvidia-smi integration for comprehensive energy-aware optimization.
"""

import subprocess
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from collections import deque
import logging
import psutil
import os


class PowerMonitor:
    """
    Real-time power monitoring using nvidia-smi for GPU power tracking.
    
    Provides energy-aware metrics for multi-objective optimization.
    """
    
    def __init__(
        self,
        sampling_interval: float = 0.1,  # 100ms sampling
        buffer_size: int = 1000,
        enable_cpu_monitoring: bool = True,
        enable_memory_monitoring: bool = True
    ):
        """
        Initialize power monitor.
        
        Args:
            sampling_interval: Time between power measurements (seconds)
            buffer_size: Number of samples to keep in buffer
            enable_cpu_monitoring: Monitor CPU power (approximate)
            enable_memory_monitoring: Monitor memory power consumption
        """
        self.sampling_interval = sampling_interval
        self.buffer_size = buffer_size
        self.enable_cpu_monitoring = enable_cpu_monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        
        # Power measurement buffers
        self.gpu_power_buffer = deque(maxlen=buffer_size)
        self.cpu_power_buffer = deque(maxlen=buffer_size)
        self.memory_power_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # Threading for async monitoring
        self.monitoring_thread = None
        self.power_queue = queue.Queue()
        self.is_monitoring = False
        self.lock = threading.Lock()
        
        # Power statistics
        self.total_energy_joules = 0.0
        self.start_time = None
        self.baseline_power = None
        
        # GPU device info
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self) -> bool:
        """Start asynchronous power monitoring."""
        if self.is_monitoring:
            self.logger.warning("Power monitoring already running")
            return True
        
        if self.gpu_count == 0:
            self.logger.warning("No CUDA GPUs detected, GPU power monitoring disabled")
            return False
        
        # Check if nvidia-smi is available
        try:
            subprocess.run(['nvidia-smi', '--help'], 
                          capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.error("nvidia-smi not available, cannot monitor GPU power")
            return False
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Get baseline power consumption
        time.sleep(0.5)  # Wait for initial measurements
        self._establish_baseline()
        
        self.logger.info(f"Power monitoring started with {self.gpu_count} GPUs")
        return True
    
    def stop_monitoring(self):
        """Stop power monitoring and cleanup."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        self.logger.info("Power monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                # Measure GPU power
                gpu_power = self._measure_gpu_power()
                
                # Measure CPU power (approximate)
                cpu_power = self._estimate_cpu_power() if self.enable_cpu_monitoring else 0.0
                
                # Measure memory power (approximate)
                memory_power = self._estimate_memory_power() if self.enable_memory_monitoring else 0.0
                
                # Store measurements
                timestamp = time.time()
                with self.lock:
                    self.gpu_power_buffer.append(gpu_power)
                    self.cpu_power_buffer.append(cpu_power)
                    self.memory_power_buffer.append(memory_power)
                    self.timestamps.append(timestamp)
                    
                    # Update total energy consumption
                    total_power = gpu_power + cpu_power + memory_power
                    if len(self.timestamps) > 1:
                        dt = timestamp - self.timestamps[-2]
                        self.total_energy_joules += total_power * dt
                
                # Put sample in queue for real-time access
                try:
                    power_sample = {
                        'timestamp': timestamp,
                        'gpu_power_w': gpu_power,
                        'cpu_power_w': cpu_power,
                        'memory_power_w': memory_power,
                        'total_power_w': total_power
                    }
                    self.power_queue.put_nowait(power_sample)
                except queue.Full:
                    # Drop oldest sample if queue is full
                    try:
                        self.power_queue.get_nowait()
                        self.power_queue.put_nowait(power_sample)
                    except queue.Empty:
                        pass
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in power monitoring: {e}")
                time.sleep(self.sampling_interval)
    
    def _measure_gpu_power(self) -> float:
        """Measure current GPU power consumption using nvidia-smi."""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=2.0
            )
            
            if result.returncode != 0:
                return 0.0
            
            # Parse power values for all GPUs
            power_values = []
            for line in result.stdout.strip().split('\\n'):
                if line.strip():
                    try:
                        power = float(line.strip())
                        power_values.append(power)
                    except ValueError:
                        continue
            
            return sum(power_values) if power_values else 0.0
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception):
            return 0.0
    
    def _estimate_cpu_power(self) -> float:
        """Estimate CPU power consumption based on utilization."""
        try:
            # Get CPU utilization
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Rough CPU power estimation (varies by CPU model)
            # Typical desktop CPU: 65-125W TDP, server CPU: 150-250W TDP
            estimated_tdp = 100.0  # Conservative estimate in watts
            idle_power = estimated_tdp * 0.1  # ~10% at idle
            
            # Linear relationship between utilization and power
            estimated_power = idle_power + (estimated_tdp - idle_power) * (cpu_percent / 100.0)
            
            return estimated_power
            
        except Exception:
            return 0.0
    
    def _estimate_memory_power(self) -> float:
        """Estimate memory power consumption."""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_usage_percent = memory.percent
            
            # Rough memory power estimation
            # DDR4: ~3-5W per 8GB module
            power_per_gb = 0.5  # Watts per GB
            base_power = memory_gb * power_per_gb * 0.2  # Base power
            usage_power = memory_gb * power_per_gb * 0.8 * (memory_usage_percent / 100.0)
            
            return base_power + usage_power
            
        except Exception:
            return 0.0
    
    def _establish_baseline(self):
        """Establish baseline power consumption."""
        if len(self.gpu_power_buffer) >= 5:
            with self.lock:
                recent_samples = list(self.gpu_power_buffer)[-5:]
                self.baseline_power = np.mean(recent_samples)
    
    def get_current_power(self) -> Dict[str, float]:
        """Get current power consumption."""
        try:
            power_sample = self.power_queue.get_nowait()
            return power_sample
        except queue.Empty:
            with self.lock:
                if self.gpu_power_buffer and self.cpu_power_buffer and self.memory_power_buffer:
                    return {
                        'timestamp': time.time(),
                        'gpu_power_w': self.gpu_power_buffer[-1],
                        'cpu_power_w': self.cpu_power_buffer[-1],
                        'memory_power_w': self.memory_power_buffer[-1],
                        'total_power_w': (self.gpu_power_buffer[-1] + 
                                        self.cpu_power_buffer[-1] + 
                                        self.memory_power_buffer[-1])
                    }
                else:
                    return {
                        'timestamp': time.time(),
                        'gpu_power_w': 0.0,
                        'cpu_power_w': 0.0,
                        'memory_power_w': 0.0,
                        'total_power_w': 0.0
                    }
    
    def get_power_statistics(self) -> Dict[str, Any]:
        """Get comprehensive power statistics."""
        with self.lock:
            if not self.gpu_power_buffer:
                return {'error': 'No power data available'}
            
            gpu_powers = np.array(list(self.gpu_power_buffer))
            cpu_powers = np.array(list(self.cpu_power_buffer)) if self.cpu_power_buffer else np.array([0])
            memory_powers = np.array(list(self.memory_power_buffer)) if self.memory_power_buffer else np.array([0])
            total_powers = gpu_powers + cpu_powers + memory_powers
            
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            stats = {
                'gpu_power': {
                    'current_w': float(gpu_powers[-1]) if len(gpu_powers) > 0 else 0.0,
                    'mean_w': float(np.mean(gpu_powers)),
                    'max_w': float(np.max(gpu_powers)),
                    'min_w': float(np.min(gpu_powers)),
                    'std_w': float(np.std(gpu_powers))
                },
                'total_power': {
                    'current_w': float(total_powers[-1]) if len(total_powers) > 0 else 0.0,
                    'mean_w': float(np.mean(total_powers)),
                    'max_w': float(np.max(total_powers)),
                    'min_w': float(np.min(total_powers))
                },
                'energy': {
                    'total_joules': self.total_energy_joules,
                    'total_kwh': self.total_energy_joules / 3.6e6,  # Convert to kWh
                    'average_power_w': self.total_energy_joules / max(elapsed_time, 1.0)
                },
                'monitoring': {
                    'elapsed_time_s': elapsed_time,
                    'samples_collected': len(gpu_powers),
                    'sampling_rate_hz': len(gpu_powers) / max(elapsed_time, 1.0),
                    'baseline_power_w': self.baseline_power if self.baseline_power else 0.0
                }
            }
            
            return stats
    
    def get_energy_efficiency_metric(self, training_loss: float) -> float:
        """Calculate energy efficiency metric for multi-objective optimization."""
        stats = self.get_power_statistics()
        
        if 'energy' not in stats:
            return 0.0
        
        total_energy = stats['energy']['total_joules']
        
        if total_energy == 0 or training_loss == 0:
            return 0.0
        
        # Energy efficiency: lower is better (energy per unit loss reduction)
        # Normalize by baseline energy consumption
        baseline_energy = (self.baseline_power or 100.0) * stats['monitoring']['elapsed_time_s']
        excess_energy = max(0, total_energy - baseline_energy)
        
        # Efficiency metric: penalize high energy consumption
        efficiency = 1.0 / (1.0 + excess_energy / max(training_loss, 1e-6))
        
        return efficiency
    
    def export_power_data(self, filename: str) -> bool:
        """Export power monitoring data to CSV file."""
        try:
            with self.lock:
                if not self.gpu_power_buffer:
                    return False
                
                import csv
                
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'timestamp', 'gpu_power_w', 'cpu_power_w', 
                        'memory_power_w', 'total_power_w'
                    ])
                    
                    for i in range(len(self.timestamps)):
                        writer.writerow([
                            self.timestamps[i],
                            self.gpu_power_buffer[i],
                            self.cpu_power_buffer[i] if i < len(self.cpu_power_buffer) else 0,
                            self.memory_power_buffer[i] if i < len(self.memory_power_buffer) else 0,
                            (self.gpu_power_buffer[i] + 
                             (self.cpu_power_buffer[i] if i < len(self.cpu_power_buffer) else 0) +
                             (self.memory_power_buffer[i] if i < len(self.memory_power_buffer) else 0))
                        ])
                
                self.logger.info(f"Power data exported to {filename}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export power data: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


class MultiObjectiveReward:
    """Multi-objective reward function for energy-aware optimization."""
    
    def __init__(
        self,
        power_monitor: Optional[PowerMonitor] = None,
        loss_weight: float = 1.0,
        memory_weight: float = 0.1,
        energy_weight: float = 0.05,
        time_weight: float = 0.02
    ):
        """Initialize multi-objective reward."""
        self.power_monitor = power_monitor
        self.loss_weight = loss_weight
        self.memory_weight = memory_weight
        self.energy_weight = energy_weight
        self.time_weight = time_weight
        
        # Baseline values for normalization
        self.baseline_loss = None
        self.baseline_memory = None
        self.baseline_energy = None
        self.baseline_time = None
    
    def compute_reward(
        self,
        current_loss: float,
        memory_usage_gb: float,
        training_time_s: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-objective reward.
        
        Args:
            current_loss: Current training loss
            memory_usage_gb: Current memory usage in GB
            training_time_s: Training time in seconds
            
        Returns:
            Tuple of (total_reward, component_rewards)
        """
        rewards = {}
        
        # Loss reward (negative loss, higher is better)
        if self.baseline_loss is None:
            self.baseline_loss = current_loss
        
        loss_improvement = max(0, self.baseline_loss - current_loss)
        rewards['loss'] = self.loss_weight * loss_improvement
        
        # Memory efficiency reward (lower memory usage is better)
        if self.baseline_memory is None:
            self.baseline_memory = memory_usage_gb
        
        memory_efficiency = max(0, self.baseline_memory - memory_usage_gb) / max(self.baseline_memory, 1.0)
        rewards['memory'] = self.memory_weight * memory_efficiency
        
        # Energy efficiency reward
        if self.power_monitor:
            energy_efficiency = self.power_monitor.get_energy_efficiency_metric(current_loss)
            rewards['energy'] = self.energy_weight * energy_efficiency
        else:
            rewards['energy'] = 0.0
        
        # Time efficiency reward (faster training is better)
        if self.baseline_time is None:
            self.baseline_time = training_time_s
        
        if training_time_s > 0:
            time_efficiency = self.baseline_time / max(training_time_s, 1e-6)
            rewards['time'] = self.time_weight * min(1.0, time_efficiency)
        else:
            rewards['time'] = 0.0
        
        # Total reward
        total_reward = sum(rewards.values())
        
        return total_reward, rewards
    
    def update_baselines(
        self,
        loss: float,
        memory_gb: float,
        time_s: float
    ):
        """Update baseline values for normalization."""
        alpha = 0.1  # Exponential moving average factor
        
        if self.baseline_loss is None:
            self.baseline_loss = loss
        else:
            self.baseline_loss = (1 - alpha) * self.baseline_loss + alpha * loss
        
        if self.baseline_memory is None:
            self.baseline_memory = memory_gb
        else:
            self.baseline_memory = (1 - alpha) * self.baseline_memory + alpha * memory_gb
        
        if self.baseline_time is None:
            self.baseline_time = time_s
        else:
            self.baseline_time = (1 - alpha) * self.baseline_time + alpha * time_s