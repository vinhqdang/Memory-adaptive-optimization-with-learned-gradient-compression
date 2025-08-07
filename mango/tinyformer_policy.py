"""
TinyFormer Policy Network

Replaces LSTM with a lightweight Transformer for better long-range trend capture
in compression policy learning. Maintains compatibility with existing PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feedforward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TinyFormerPolicyNet(nn.Module):
    """
    Lightweight Transformer-based policy network for compression decisions.
    
    Features:
    - 6 transformer layers with 128-d hidden size
    - Multi-head attention for long-range dependencies
    - Maintains compatibility with existing PPO training
    - <100k parameters as specified in the plan
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 256,
        max_seq_len: int = 200,
        num_parameter_groups: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize TinyFormer Policy Network.
        
        Args:
            feature_dim: Input feature dimension
            d_model: Transformer hidden dimension (128 as per plan)
            num_layers: Number of transformer layers (6 as per plan)
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            max_seq_len: Maximum sequence length for positional encoding
            num_parameter_groups: Number of parameter groups to handle
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_parameter_groups = num_parameter_groups
        self.max_seq_len = max_seq_len
        
        # Input projection and feature preprocessing
        self.feature_projector = nn.Sequential(
            nn.Linear(35, feature_dim),  # Raw features to intermediate (increased for power features)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, d_model),  # Project to model dimension
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for MANGO-LRQ decisions
        head_dim = 64
        
        # Low-rank dimension head (supports variable ranks)
        self.rank_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 8)  # Ranks: {1, 2, 4, 8, 16, 32, 64, 128}
        )
        
        # P matrix quantization bits
        self.bits_p_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 4)  # {32, 16, 8, 4} bits
        )
        
        # Q matrix quantization bits
        self.bits_q_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 4)  # {32, 16, 8, 4} bits
        )
        
        # Momentum precision selection
        self.momentum_precision_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 3)  # {"fp32", "fp16", "nf4"}
        )
        
        # Use NF4 quantization switch
        self.use_nf4_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)  # Binary decision
        )
        
        # Multi-objective value heads for policy gradient training
        self.loss_value_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)
        )
        
        self.memory_value_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)
        )
        
        self.energy_value_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)
        )
        
        self.time_value_head = nn.Sequential(
            nn.Linear(d_model, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)
        )
        
        # Sequence memory for maintaining context
        self.register_buffer('sequence_memory', torch.zeros(1, max_seq_len, d_model))
        self.register_buffer('memory_position', torch.zeros(1, dtype=torch.long))
        
        # Initialize weights
        self._init_weights()
        
        # Parameter count verification
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TinyFormer parameters: {total_params:,} (<100k target)")
        
    def _init_weights(self):
        """Initialize network weights with appropriate scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def reset_hidden_state(self):
        """Reset sequence memory (call between training episodes)."""
        self.sequence_memory.zero_()
        self.memory_position.zero_()
    
    def process_features(self, raw_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process raw gradient statistics into network input.
        
        Args:
            raw_features: Dictionary of gradient statistics including power data
            
        Returns:
            Processed feature tensor [batch_size, feature_dim]
        """
        # Extract and normalize key features (same as LSTM version for compatibility)
        features = []
        
        # Global features
        step_ratio = raw_features.get('step_ratio', torch.tensor([0.0]))
        memory_usage = raw_features.get('memory_usage', torch.tensor([0.0]))
        
        # Power-related features from PowerSampler EWMA
        ewma_power = raw_features.get('ewma_power', torch.tensor([0.0]))
        power_efficiency = raw_features.get('power_efficiency', torch.tensor([0.0]))
        gpu_utilization = raw_features.get('gpu_utilization', torch.tensor([0.0]))
        
        features.extend([step_ratio, memory_usage, ewma_power, power_efficiency, gpu_utilization])
        
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
        
        # Pad or truncate to expected size (increased to 35 to accommodate power features)
        while len(features) < 35:
            features.append(torch.tensor([0.0]))
        features = features[:35]
        
        # Stack into tensor and flatten
        feature_tensor = torch.stack(features).flatten().unsqueeze(0)  # [1, 35]
        
        return feature_tensor
    
    def update_sequence_memory(self, new_features: torch.Tensor):
        """Update sequence memory with new features."""
        batch_size = new_features.size(0)
        
        # Get current position
        pos = self.memory_position.item()
        
        # Update memory
        self.sequence_memory[0, pos] = new_features.squeeze(0)
        
        # Update position (circular buffer)
        self.memory_position[0] = (pos + 1) % self.max_seq_len
    
    def get_sequence_context(self, seq_len: int = 50) -> torch.Tensor:
        """Get recent sequence context for attention."""
        pos = self.memory_position.item()
        
        if pos >= seq_len:
            # Take recent seq_len entries
            context = self.sequence_memory[0, pos-seq_len:pos]
        else:
            # Handle wraparound
            if pos == 0:
                # No entries yet, use zeros
                context = torch.zeros(seq_len, self.d_model, device=self.sequence_memory.device)
            else:
                # Combine end and beginning
                end_part = self.sequence_memory[0, self.max_seq_len-seq_len+pos:]
                begin_part = self.sequence_memory[0, :pos]
                context = torch.cat([end_part, begin_part], dim=0)
        
        return context.unsqueeze(0)  # Add batch dimension
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TinyFormer.
        
        Args:
            features: Dictionary of gradient statistics
            
        Returns:
            Dictionary containing MANGO-LRQ compression configuration
        """
        # Process input features
        raw_input = self.process_features(features)  # [1, 32]
        
        # Project to model dimension
        x = self.feature_projector(raw_input)  # [1, d_model]
        
        # Update sequence memory
        self.update_sequence_memory(x)
        
        # Get sequence context for attention
        seq_context = self.get_sequence_context()  # [1, seq_len, d_model]
        
        # Add positional encoding
        seq_with_pos = self.pos_encoding(seq_context)  # [1, seq_len, d_model]
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            seq_with_pos = layer(seq_with_pos)
        
        # Use the last position for decision making
        final_repr = seq_with_pos[:, -1, :]  # [1, d_model]
        
        # Generate MANGO-LRQ decisions
        
        # Low-rank dimension (softmax over options)
        rank_logits = self.rank_head(final_repr)  # [1, 8]
        rank_probs = F.softmax(rank_logits, dim=-1)
        
        # P matrix quantization bits
        bits_p_logits = self.bits_p_head(final_repr)  # [1, 4]
        bits_p_probs = F.softmax(bits_p_logits, dim=-1)
        
        # Q matrix quantization bits
        bits_q_logits = self.bits_q_head(final_repr)  # [1, 4]
        bits_q_probs = F.softmax(bits_q_logits, dim=-1)
        
        # Momentum precision
        momentum_logits = self.momentum_precision_head(final_repr)  # [1, 3]
        momentum_probs = F.softmax(momentum_logits, dim=-1)
        
        # Use NF4 quantization (sigmoid for binary decision)
        use_nf4_logits = self.use_nf4_head(final_repr)  # [1, 1]
        use_nf4_prob = torch.sigmoid(use_nf4_logits)
        
        # Multi-objective value estimates
        loss_value = self.loss_value_head(final_repr)  # [1, 1]
        memory_value = self.memory_value_head(final_repr)  # [1, 1]
        energy_value = self.energy_value_head(final_repr)  # [1, 1]
        time_value = self.time_value_head(final_repr)  # [1, 1]
        
        # Combined value (weighted sum)
        value = loss_value + 0.1 * memory_value + 0.05 * energy_value + 0.02 * time_value
        
        # Convert probabilities to expected values
        rank_options = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32, device=x.device)
        bits_options = torch.tensor([32, 16, 8, 4], dtype=torch.float32, device=x.device)
        momentum_options = torch.tensor([0, 1, 2], dtype=torch.float32, device=x.device)  # indices for ["fp32", "fp16", "nf4"]
        
        rank_expected = (rank_probs * rank_options).sum(dim=-1)
        bits_p_expected = (bits_p_probs * bits_options).sum(dim=-1)
        bits_q_expected = (bits_q_probs * bits_options).sum(dim=-1)
        momentum_expected = (momentum_probs * momentum_options).sum(dim=-1)
        
        return {
            # MANGO-LRQ specific parameters
            'rank': int(rank_expected.item()),
            'bits_p': int(bits_p_expected.item()),
            'bits_q': int(bits_q_expected.item()),
            'momentum_precision': ["fp32", "fp16", "nf4"][int(momentum_expected.item())],
            'use_nf4': use_nf4_prob.item() > 0.5,
            
            # Legacy parameters for compatibility
            'gradient_bits': int(bits_p_expected.item()),  # Use P bits as gradient bits
            'momentum_bits': int(bits_p_expected.item()),  # For compatibility
            'sparsity_ratio': 0.0,  # MANGO-LRQ uses low-rank instead of sparsity
            
            # Multi-objective policy training components
            'value': value.item(),
            'loss_value': loss_value.item(),
            'memory_value': memory_value.item(),
            'energy_value': energy_value.item(),
            'time_value': time_value.item(),
            'rank_logits': rank_logits,
            'bits_p_logits': bits_p_logits,
            'bits_q_logits': bits_q_logits,
            'momentum_logits': momentum_logits,
            'use_nf4_logits': use_nf4_logits,
            'rank_probs': rank_probs,
            'bits_p_probs': bits_p_probs,
            'bits_q_probs': bits_q_probs,
            'momentum_probs': momentum_probs
        }
    
    def sample_action(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, any], Dict[str, torch.Tensor]]:
        """
        Sample compression action for policy gradient training.
        
        Args:
            features: Dictionary of gradient statistics
            
        Returns:
            Tuple of (sampled_action, action_log_probs)
        """
        output = self.forward(features)
        
        # Sample from categorical distributions
        rank_dist = torch.distributions.Categorical(output['rank_probs'])
        bits_p_dist = torch.distributions.Categorical(output['bits_p_probs'])
        bits_q_dist = torch.distributions.Categorical(output['bits_q_probs'])
        momentum_dist = torch.distributions.Categorical(output['momentum_probs'])
        use_nf4_dist = torch.distributions.Bernoulli(output['use_nf4_logits'].sigmoid())
        
        rank_idx = rank_dist.sample()
        bits_p_idx = bits_p_dist.sample()
        bits_q_idx = bits_q_dist.sample()
        momentum_idx = momentum_dist.sample()
        use_nf4_sample = use_nf4_dist.sample()
        
        # Convert indices to actual values
        rank_options = [1, 2, 4, 8, 16, 32, 64, 128]
        bits_options = [32, 16, 8, 4]
        momentum_options = ["fp32", "fp16", "nf4"]
        
        sampled_action = {
            'rank': rank_options[rank_idx.item()],
            'bits_p': bits_options[bits_p_idx.item()],
            'bits_q': bits_options[bits_q_idx.item()],
            'momentum_precision': momentum_options[momentum_idx.item()],
            'use_nf4': bool(use_nf4_sample.item()),
            
            # Legacy compatibility
            'gradient_bits': bits_options[bits_p_idx.item()],
            'momentum_bits': bits_options[bits_p_idx.item()],
            'sparsity_ratio': 0.0
        }
        
        # Compute log probabilities for policy gradient
        action_log_probs = {
            'rank_log_prob': rank_dist.log_prob(rank_idx),
            'bits_p_log_prob': bits_p_dist.log_prob(bits_p_idx),
            'bits_q_log_prob': bits_q_dist.log_prob(bits_q_idx),
            'momentum_log_prob': momentum_dist.log_prob(momentum_idx),
            'use_nf4_log_prob': use_nf4_dist.log_prob(use_nf4_sample),
            
            # Legacy compatibility
            'gradient_bits_log_prob': bits_p_dist.log_prob(bits_p_idx),
            'momentum_bits_log_prob': bits_p_dist.log_prob(bits_p_idx)
        }
        
        return sampled_action, action_log_probs
    
    def compute_loss(
        self, 
        features: Dict[str, torch.Tensor],
        actions: Dict[str, any],
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
        rank_options = [1, 2, 4, 8, 16, 32, 64, 128]
        bits_options = [32, 16, 8, 4]
        momentum_options = ["fp32", "fp16", "nf4"]
        
        rank_idx = rank_options.index(actions['rank'])
        bits_p_idx = bits_options.index(actions['bits_p'])
        bits_q_idx = bits_options.index(actions['bits_q'])
        momentum_idx = momentum_options.index(actions['momentum_precision'])
        
        rank_log_prob = torch.log(output['rank_probs'][0, rank_idx] + 1e-8)
        bits_p_log_prob = torch.log(output['bits_p_probs'][0, bits_p_idx] + 1e-8)
        bits_q_log_prob = torch.log(output['bits_q_probs'][0, bits_q_idx] + 1e-8)
        momentum_log_prob = torch.log(output['momentum_probs'][0, momentum_idx] + 1e-8)
        
        # PPO clipping
        epsilon = 0.2
        
        def compute_ppo_loss(current_log_prob, old_log_prob, advantages):
            ratio = torch.exp(current_log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Individual losses
        loss_rank = compute_ppo_loss(rank_log_prob, old_log_probs['rank_log_prob'], advantages)
        loss_bits_p = compute_ppo_loss(bits_p_log_prob, old_log_probs['bits_p_log_prob'], advantages)
        loss_bits_q = compute_ppo_loss(bits_q_log_prob, old_log_probs['bits_q_log_prob'], advantages)
        loss_momentum = compute_ppo_loss(momentum_log_prob, old_log_probs['momentum_log_prob'], advantages)
        
        # Value loss
        value_loss = F.mse_loss(output['value'], rewards)
        
        # Entropy bonus for exploration
        entropy_bonus = (
            -(output['rank_probs'] * torch.log(output['rank_probs'] + 1e-8)).sum() +
            -(output['bits_p_probs'] * torch.log(output['bits_p_probs'] + 1e-8)).sum() +
            -(output['bits_q_probs'] * torch.log(output['bits_q_probs'] + 1e-8)).sum() +
            -(output['momentum_probs'] * torch.log(output['momentum_probs'] + 1e-8)).sum()
        )
        
        # Total loss
        policy_loss = loss_rank + loss_bits_p + loss_bits_q + loss_momentum
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy_bonus,
            'rank_loss': loss_rank,
            'bits_p_loss': loss_bits_p,
            'bits_q_loss': loss_bits_q,
            'momentum_loss': loss_momentum
        }


class TinyFormerConfig:
    """Configuration for TinyFormer policy network."""
    
    def __init__(
        self,
        feature_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 256,
        max_seq_len: int = 200,
        num_parameter_groups: int = 1,
        dropout: float = 0.1
    ):
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_parameter_groups = num_parameter_groups
        self.dropout = dropout