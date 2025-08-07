"""
Gradient Statistics Collection

Tracks and computes gradient statistics needed for compression policy decisions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import math


class GradientStatistics:
    """
    Collects and maintains statistics about gradients during training.
    
    Tracks gradient norms, variances, direction changes, and other metrics
    needed by the compression policy network.
    """
    
    def __init__(self, history_length: int = 100, ema_decay: float = 0.99):
        """
        Initialize gradient statistics collector.
        
        Args:
            history_length: Number of historical values to maintain
            ema_decay: Exponential moving average decay factor
        """
        self.history_length = history_length
        self.ema_decay = ema_decay
        
        # Per-parameter statistics
        self.gradient_norms = defaultdict(lambda: deque(maxlen=history_length))
        self.gradient_directions = defaultdict(lambda: deque(maxlen=history_length))
        self.gradient_ema_norms = defaultdict(float)
        self.gradient_ema_vars = defaultdict(float)
        
        # Global statistics
        self.global_gradient_norm_history = deque(maxlen=history_length)
        self.loss_history = deque(maxlen=history_length)
        self.step_count = 0
        
        # Compression impact tracking
        self.compression_errors = defaultdict(lambda: deque(maxlen=history_length))
        self.compression_ratios = deque(maxlen=history_length)
    
    def update(self, step: int, compression_config: Dict, param_groups: List) -> None:
        """
        Update statistics with current step information.
        
        Args:
            step: Current training step
            compression_config: Current compression configuration
            param_groups: Optimizer parameter groups
        """
        self.step_count = step
        
        total_grad_norm = 0.0
        param_count = 0
        
        for group in param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                grad = p.grad.data
                
                # Gradient norm
                grad_norm = grad.norm().item()
                self.gradient_norms[param_id].append(grad_norm)
                total_grad_norm += grad_norm ** 2
                param_count += 1
                
                # EMA gradient norm
                if param_id not in self.gradient_ema_norms:
                    self.gradient_ema_norms[param_id] = grad_norm
                else:
                    self.gradient_ema_norms[param_id] = (
                        self.ema_decay * self.gradient_ema_norms[param_id] +
                        (1 - self.ema_decay) * grad_norm
                    )
                
                # Gradient direction (normalized gradient)
                if grad_norm > 1e-8:
                    grad_direction = grad / grad_norm
                    self.gradient_directions[param_id].append(grad_direction.cpu())
                
                # EMA gradient variance
                if len(self.gradient_norms[param_id]) > 1:
                    recent_norms = list(self.gradient_norms[param_id])[-10:]  # Last 10 steps
                    grad_var = np.var(recent_norms)
                    
                    if param_id not in self.gradient_ema_vars:
                        self.gradient_ema_vars[param_id] = grad_var
                    else:
                        self.gradient_ema_vars[param_id] = (
                            self.ema_decay * self.gradient_ema_vars[param_id] +
                            (1 - self.ema_decay) * grad_var
                        )
        
        # Global gradient norm
        if param_count > 0:
            global_norm = math.sqrt(total_grad_norm)
            self.global_gradient_norm_history.append(global_norm)
        
        # Track compression ratio
        if compression_config:
            grad_bits = compression_config.get('gradient_bits', 32)
            momentum_bits = compression_config.get('momentum_bits', 32) 
            sparsity = compression_config.get('sparsity_ratio', 0.0)
            
            # Estimate compression ratio
            base_bits = 32
            compressed_bits = grad_bits * (1 - sparsity) + momentum_bits * 0.5  # Approximate
            compression_ratio = base_bits / max(compressed_bits, 1)
            self.compression_ratios.append(compression_ratio)
    
    def get_gradient_variance(self, param_id: int, current_grad: torch.Tensor) -> float:
        """Get gradient variance for a specific parameter."""
        if param_id in self.gradient_norms and len(self.gradient_norms[param_id]) > 1:
            recent_norms = list(self.gradient_norms[param_id])[-10:]
            return float(np.var(recent_norms))
        return 0.0
    
    def get_hessian_estimate(self, param_id: int, current_grad: torch.Tensor) -> float:
        """
        Estimate Hessian diagonal using gradient differences.
        
        Uses finite differences of gradients to approximate diagonal Hessian elements.
        """
        if param_id not in self.gradient_directions or len(self.gradient_directions[param_id]) < 2:
            return 0.0
        
        # Get recent gradient directions
        directions = self.gradient_directions[param_id]
        if len(directions) < 2:
            return 0.0
        
        current_dir = directions[-1]
        prev_dir = directions[-2]
        
        # Estimate Hessian diagonal as gradient difference magnitude
        grad_diff = current_dir - prev_dir
        hessian_estimate = grad_diff.norm().item()
        
        return hessian_estimate
    
    def get_gradient_direction_change(self, param_id: int) -> float:
        """
        Get the angle between current and previous gradient direction.
        
        Returns cosine similarity between consecutive gradient directions.
        """
        if param_id not in self.gradient_directions or len(self.gradient_directions[param_id]) < 2:
            return 1.0  # No change if insufficient history
        
        directions = self.gradient_directions[param_id]
        current_dir = directions[-1]
        prev_dir = directions[-2]
        
        # Compute cosine similarity
        similarity = torch.cosine_similarity(
            current_dir.flatten(), prev_dir.flatten(), dim=0
        ).item()
        
        return similarity
    
    def get_curvature_estimate(self) -> float:
        """
        Estimate local curvature from loss history.
        
        Uses second-order difference of loss as curvature proxy.
        """
        if len(self.loss_history) < 3:
            return 0.0
        
        losses = list(self.loss_history)[-3:]
        
        # Second derivative approximation: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        second_derivative = losses[-1] - 2 * losses[-2] + losses[-3]
        return abs(second_derivative)
    
    def get_training_phase(self) -> str:
        """
        Classify current training phase based on gradient statistics.
        
        Returns one of: 'early', 'middle', 'late', 'converged'
        """
        if len(self.global_gradient_norm_history) < 10:
            return 'early'
        
        recent_norms = list(self.global_gradient_norm_history)[-10:]
        norm_trend = np.mean(np.diff(recent_norms))
        norm_variance = np.var(recent_norms)
        
        # High variance and norm suggests early training
        if norm_variance > 1.0 and np.mean(recent_norms) > 1.0:
            return 'early'
        
        # Decreasing norms with moderate variance suggests middle training
        elif norm_trend < -0.01 and norm_variance > 0.1:
            return 'middle'
        
        # Low variance and norms suggests late training or convergence
        elif norm_variance < 0.1 and np.mean(recent_norms) < 0.1:
            return 'converged'
        else:
            return 'late'
    
    def get_layer_importance_scores(self, param_groups: List) -> Dict[int, float]:
        """
        Compute importance scores for each layer based on gradient statistics.
        
        Higher scores indicate layers that should use higher precision.
        """
        importance_scores = {}
        
        for group in param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                
                # Factors contributing to importance:
                # 1. Gradient norm (higher norm = more important)
                # 2. Gradient variance (higher variance = more important) 
                # 3. Direction stability (less stable = more important)
                
                grad_norm = self.gradient_ema_norms.get(param_id, 0.0)
                grad_var = self.gradient_ema_vars.get(param_id, 0.0)
                direction_change = 1.0 - self.get_gradient_direction_change(param_id)
                
                # Combine factors (weighted sum)
                importance = (
                    0.4 * grad_norm +
                    0.3 * grad_var + 
                    0.3 * direction_change
                )
                
                importance_scores[param_id] = importance
        
        # Normalize scores to [0, 1] range
        if importance_scores:
            max_importance = max(importance_scores.values())
            if max_importance > 0:
                importance_scores = {
                    k: v / max_importance for k, v in importance_scores.items()
                }
        
        return importance_scores
    
    def get_compression_tolerance(self, param_id: int) -> float:
        """
        Estimate how much compression a parameter can tolerate.
        
        Returns tolerance score in [0, 1] where higher means more tolerant.
        """
        # Parameters with stable gradients can tolerate more compression
        direction_stability = self.get_gradient_direction_change(param_id)
        norm_stability = 1.0 / (1.0 + self.gradient_ema_vars.get(param_id, 1.0))
        
        # Combine stability measures
        tolerance = 0.6 * direction_stability + 0.4 * norm_stability
        return min(1.0, max(0.0, tolerance))
    
    def get_memory_efficiency_score(self) -> float:
        """
        Get current memory efficiency score based on compression ratios.
        
        Returns average compression ratio achieved recently.
        """
        if not self.compression_ratios:
            return 1.0
        
        recent_ratios = list(self.compression_ratios)[-10:]
        return float(np.mean(recent_ratios))
    
    def add_loss(self, loss: float) -> None:
        """Add loss value to history for curvature estimation."""
        self.loss_history.append(loss)
    
    def add_compression_error(self, param_id: int, error_norm: float) -> None:
        """Track compression error for a specific parameter."""
        self.compression_errors[param_id].append(error_norm)
    
    def get_compression_error_stats(self) -> Dict[str, float]:
        """Get statistics about compression errors."""
        if not self.compression_errors:
            return {'mean_error': 0.0, 'max_error': 0.0, 'error_trend': 0.0}
        
        all_errors = []
        for param_errors in self.compression_errors.values():
            all_errors.extend(list(param_errors))
        
        if not all_errors:
            return {'mean_error': 0.0, 'max_error': 0.0, 'error_trend': 0.0}
        
        mean_error = float(np.mean(all_errors))
        max_error = float(np.max(all_errors))
        
        # Error trend (increasing or decreasing)
        if len(all_errors) >= 10:
            recent_errors = all_errors[-10:]
            early_errors = all_errors[-20:-10] if len(all_errors) >= 20 else recent_errors
            error_trend = np.mean(recent_errors) - np.mean(early_errors)
        else:
            error_trend = 0.0
        
        return {
            'mean_error': mean_error,
            'max_error': max_error, 
            'error_trend': float(error_trend)
        }
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get comprehensive summary of all statistics."""
        stats = {
            'step_count': self.step_count,
            'training_phase': self.get_training_phase(),
            'curvature_estimate': self.get_curvature_estimate(),
            'memory_efficiency': self.get_memory_efficiency_score(),
        }
        
        # Global gradient statistics
        if self.global_gradient_norm_history:
            recent_norms = list(self.global_gradient_norm_history)[-10:]
            stats.update({
                'global_grad_norm_mean': float(np.mean(recent_norms)),
                'global_grad_norm_var': float(np.var(recent_norms)),
                'global_grad_norm_trend': float(np.mean(np.diff(recent_norms))) if len(recent_norms) > 1 else 0.0
            })
        
        # Compression error statistics
        stats.update(self.get_compression_error_stats())
        
        return stats