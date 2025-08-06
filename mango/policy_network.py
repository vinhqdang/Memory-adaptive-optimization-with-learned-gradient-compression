"""
Compression Policy Network (CPN)

Neural network that learns optimal compression policies based on training dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class CompressionPolicyNet(nn.Module):
    """
    Lightweight LSTM-based network that observes training dynamics and outputs
    compression decisions for gradients and momentum states.
    
    The network takes gradient statistics as input and outputs compression 
    configurations that adapt to the current training phase.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_parameter_groups: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize the Compression Policy Network.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_parameter_groups: Number of parameter groups to handle
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_parameter_groups = num_parameter_groups
        
        # Feature preprocessing
        self.feature_projector = nn.Sequential(
            nn.Linear(32, feature_dim),  # Adjust based on input features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Core LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output heads for different compression decisions
        self.gradient_bits_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)  # {32, 16, 8, 4, 2} bits
        )
        
        self.momentum_bits_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # {32, 16, 8} bits
        )
        
        self.sparsity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Sparsity ratio [0, 1]
        )
        
        # Value head for policy gradient training
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # LSTM state
        self.hidden_state = None
        self.cell_state = None
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state (call between training episodes)."""
        self.hidden_state = None
        self.cell_state = None
    
    def process_features(self, raw_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process raw gradient statistics into network input.
        
        Args:
            raw_features: Dictionary of gradient statistics
            
        Returns:
            Processed feature tensor [batch_size, feature_dim]
        """
        # Extract and normalize key features
        features = []
        
        # Global features
        step_ratio = raw_features.get('step_ratio', torch.tensor([0.0]))
        memory_usage = raw_features.get('memory_usage', torch.tensor([0.0]))
        features.extend([step_ratio, memory_usage])
        
        # Parameter group features (aggregate across groups)
        for i in range(self.num_parameter_groups):
            # Gradient norms
            grad_norms = raw_features.get(f'group_{i}_grad_norms', torch.tensor([0.0]))
            grad_norm_mean = grad_norms.mean().unsqueeze(0)
            grad_norm_std = grad_norms.std().unsqueeze(0) if grad_norms.numel() > 1 else torch.tensor([0.0])
            
            # Gradient variances  
            grad_vars = raw_features.get(f'group_{i}_grad_variances', torch.tensor([0.0]))
            grad_var_mean = grad_vars.mean().unsqueeze(0)
            
            # Momentum alignments
            momentum_aligns = raw_features.get(f'group_{i}_momentum_alignments', torch.tensor([0.0]))
            momentum_align_mean = momentum_aligns.mean().unsqueeze(0)
            
            # Parameter counts (log scale for better normalization)
            param_counts = raw_features.get(f'group_{i}_param_counts', torch.tensor([1.0]))
            log_param_count = torch.log(param_counts.float()).mean().unsqueeze(0)
            
            # Layer depths
            layer_depths = raw_features.get(f'group_{i}_layer_depths', torch.tensor([0.0]))
            max_depth = layer_depths.max().unsqueeze(0) if layer_depths.numel() > 0 else torch.tensor([0.0])
            
            features.extend([
                grad_norm_mean, grad_norm_std, grad_var_mean,
                momentum_align_mean, log_param_count, max_depth
            ])
        
        # Pad or truncate to expected size
        while len(features) < 32:
            features.append(torch.tensor([0.0]))
        features = features[:32]
        
        # Stack into tensor
        feature_tensor = torch.stack(features).unsqueeze(0)  # [1, 32]
        
        # Project to feature dimension
        processed_features = self.feature_projector(feature_tensor)  # [1, feature_dim]
        
        return processed_features
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            features: Dictionary of gradient statistics
            
        Returns:
            Dictionary containing compression configuration
        """
        # Process input features
        x = self.process_features(features)  # [1, feature_dim]
        x = x.unsqueeze(1)  # Add sequence dimension: [1, 1, feature_dim]
        
        # LSTM forward pass
        if self.hidden_state is not None and self.cell_state is not None:
            lstm_out, (h_new, c_new) = self.lstm(x, (self.hidden_state, self.cell_state))
        else:
            lstm_out, (h_new, c_new) = self.lstm(x)
        
        # Update hidden state
        self.hidden_state = h_new.detach()
        self.cell_state = c_new.detach()
        
        # Extract features from LSTM output
        lstm_features = lstm_out.squeeze(1)  # [1, hidden_dim]
        
        # Generate compression decisions
        
        # Gradient bits (softmax over discrete choices)
        gradient_bits_logits = self.gradient_bits_head(lstm_features)  # [1, 5]
        gradient_bits_probs = F.softmax(gradient_bits_logits, dim=-1)
        
        # Momentum bits (softmax over discrete choices)  
        momentum_bits_logits = self.momentum_bits_head(lstm_features)  # [1, 3]
        momentum_bits_probs = F.softmax(momentum_bits_logits, dim=-1)
        
        # Sparsity ratio (sigmoid for [0, 1] range)
        sparsity_logits = self.sparsity_head(lstm_features)  # [1, 1]
        sparsity_ratio = torch.sigmoid(sparsity_logits)
        
        # Value estimate for policy gradient
        value = self.value_head(lstm_features)  # [1, 1]
        
        # Convert probabilities to actual bit values
        gradient_bits_options = torch.tensor([32, 16, 8, 4, 2], dtype=torch.float32, device=x.device)
        momentum_bits_options = torch.tensor([32, 16, 8], dtype=torch.float32, device=x.device)
        
        # Expected values (for deterministic inference)
        gradient_bits_expected = (gradient_bits_probs * gradient_bits_options).sum(dim=-1)
        momentum_bits_expected = (momentum_bits_probs * momentum_bits_options).sum(dim=-1)
        
        return {
            'gradient_bits': gradient_bits_expected.item(),
            'momentum_bits': momentum_bits_expected.item(), 
            'sparsity_ratio': sparsity_ratio.item(),
            'value': value.item(),
            'gradient_bits_logits': gradient_bits_logits,
            'momentum_bits_logits': momentum_bits_logits,
            'sparsity_logits': sparsity_logits,
            'gradient_bits_probs': gradient_bits_probs,
            'momentum_bits_probs': momentum_bits_probs
        }
    
    def sample_action(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        """
        Sample compression action for policy gradient training.
        
        Args:
            features: Dictionary of gradient statistics
            
        Returns:
            Tuple of (sampled_action, action_log_probs)
        """
        output = self.forward(features)
        
        # Sample from categorical distributions
        gradient_bits_dist = torch.distributions.Categorical(output['gradient_bits_probs'])
        momentum_bits_dist = torch.distributions.Categorical(output['momentum_bits_probs'])
        
        gradient_bits_idx = gradient_bits_dist.sample()
        momentum_bits_idx = momentum_bits_dist.sample()
        
        # Convert indices to bit values
        gradient_bits_options = [32, 16, 8, 4, 2]
        momentum_bits_options = [32, 16, 8]
        
        sampled_action = {
            'gradient_bits': gradient_bits_options[gradient_bits_idx.item()],
            'momentum_bits': momentum_bits_options[momentum_bits_idx.item()],
            'sparsity_ratio': output['sparsity_ratio']
        }
        
        # Compute log probabilities for policy gradient
        action_log_probs = {
            'gradient_bits_log_prob': gradient_bits_dist.log_prob(gradient_bits_idx),
            'momentum_bits_log_prob': momentum_bits_dist.log_prob(momentum_bits_idx),
            'sparsity_log_prob': output['sparsity_logits']  # For continuous action
        }
        
        return sampled_action, action_log_probs
    
    def compute_loss(
        self, 
        features: Dict[str, torch.Tensor],
        actions: Dict[str, int],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss for policy updates.
        
        Args:
            features: Input features
            actions: Taken actions
            rewards: Received rewards
            advantages: Computed advantages
            old_log_probs: Log probabilities from old policy
            
        Returns:
            Dictionary of loss components
        """
        output = self.forward(features)
        
        # Current policy log probabilities
        gradient_bits_options = [32, 16, 8, 4, 2]
        momentum_bits_options = [32, 16, 8]
        
        gradient_bits_idx = gradient_bits_options.index(actions['gradient_bits'])
        momentum_bits_idx = momentum_bits_options.index(actions['momentum_bits'])
        
        gradient_bits_log_prob = torch.log(output['gradient_bits_probs'][0, gradient_bits_idx] + 1e-8)
        momentum_bits_log_prob = torch.log(output['momentum_bits_probs'][0, momentum_bits_idx] + 1e-8)
        
        # PPO clipping
        epsilon = 0.2
        
        # Gradient bits ratio
        ratio_grad = torch.exp(gradient_bits_log_prob - old_log_probs['gradient_bits_log_prob'])
        clipped_ratio_grad = torch.clamp(ratio_grad, 1 - epsilon, 1 + epsilon)
        loss_grad = -torch.min(ratio_grad * advantages, clipped_ratio_grad * advantages).mean()
        
        # Momentum bits ratio
        ratio_momentum = torch.exp(momentum_bits_log_prob - old_log_probs['momentum_bits_log_prob'])
        clipped_ratio_momentum = torch.clamp(ratio_momentum, 1 - epsilon, 1 + epsilon)
        loss_momentum = -torch.min(ratio_momentum * advantages, clipped_ratio_momentum * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(output['value'], rewards)
        
        # Entropy bonus for exploration
        gradient_entropy = -(output['gradient_bits_probs'] * torch.log(output['gradient_bits_probs'] + 1e-8)).sum()
        momentum_entropy = -(output['momentum_bits_probs'] * torch.log(output['momentum_bits_probs'] + 1e-8)).sum()
        entropy_bonus = gradient_entropy + momentum_entropy
        
        # Total loss
        total_loss = loss_grad + loss_momentum + 0.5 * value_loss - 0.01 * entropy_bonus
        
        return {
            'total_loss': total_loss,
            'policy_loss': loss_grad + loss_momentum,
            'value_loss': value_loss,
            'entropy': entropy_bonus
        }


class FixedCompressionPolicy:
    """
    Simple fixed compression policy for baseline comparisons.
    """
    
    def __init__(self, gradient_bits: int = 16, momentum_bits: int = 16, sparsity_ratio: float = 0.1):
        """
        Initialize fixed policy.
        
        Args:
            gradient_bits: Fixed gradient quantization bits
            momentum_bits: Fixed momentum quantization bits  
            sparsity_ratio: Fixed sparsity ratio
        """
        self.gradient_bits = gradient_bits
        self.momentum_bits = momentum_bits
        self.sparsity_ratio = sparsity_ratio
    
    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Return fixed compression configuration."""
        return {
            'gradient_bits': self.gradient_bits,
            'momentum_bits': self.momentum_bits,
            'sparsity_ratio': self.sparsity_ratio
        }
    
    def reset_hidden_state(self):
        """No-op for compatibility."""
        pass


class AdaptiveCompressionPolicy:
    """
    Hand-crafted adaptive compression policy based on training phase.
    """
    
    def __init__(self):
        """Initialize adaptive policy."""
        self.step_count = 0
    
    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Return compression configuration based on training dynamics.
        """
        self.step_count += 1
        
        # Extract key features
        step_ratio = features.get('step_ratio', torch.tensor([0.0])).item()
        memory_usage = features.get('memory_usage', torch.tensor([0.5])).item()
        
        # Compute average gradient norm across groups
        avg_grad_norm = 0.0
        group_count = 0
        for i in range(10):  # Check up to 10 groups
            grad_norms = features.get(f'group_{i}_grad_norms')
            if grad_norms is not None and grad_norms.numel() > 0:
                avg_grad_norm += grad_norms.mean().item()
                group_count += 1
        
        if group_count > 0:
            avg_grad_norm /= group_count
        
        # Adaptive policy based on training phase
        if step_ratio < 0.1 or avg_grad_norm > 1.0:
            # Early training: high precision
            gradient_bits = 32
            momentum_bits = 32
            sparsity_ratio = 0.0
        elif step_ratio < 0.5 and avg_grad_norm > 0.1:
            # Mid training: moderate compression
            gradient_bits = 16
            momentum_bits = 16
            sparsity_ratio = min(0.2, memory_usage * 0.4)
        elif step_ratio < 0.8:
            # Late training: more compression
            gradient_bits = 8
            momentum_bits = 16
            sparsity_ratio = min(0.5, memory_usage * 0.6)
        else:
            # Final phase: aggressive compression
            gradient_bits = 8
            momentum_bits = 8
            sparsity_ratio = min(0.7, memory_usage * 0.8)
        
        return {
            'gradient_bits': gradient_bits,
            'momentum_bits': momentum_bits,
            'sparsity_ratio': sparsity_ratio
        }
    
    def reset_hidden_state(self):
        """No-op for compatibility."""
        pass