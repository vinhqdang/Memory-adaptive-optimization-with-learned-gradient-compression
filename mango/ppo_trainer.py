"""
PPO (Proximal Policy Optimization) Trainer for Compression Policy Network

Trains the compression policy network using reinforcement learning to learn
optimal compression strategies based on training dynamics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
import random
import time

from .policy_network import CompressionPolicyNet
from .utils import setup_cuda_environment


# Named tuple for storing trajectory data
Experience = namedtuple('Experience', [
    'features', 'action', 'reward', 'value', 'log_prob', 'advantage', 'return_'
])


class PPOTrainer:
    """
    PPO trainer for the compression policy network.
    
    Collects trajectories from training runs and uses PPO to update the policy
    to maximize cumulative reward (minimize loss while saving memory).
    """
    
    def __init__(
        self,
        policy_net: CompressionPolicyNet,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        buffer_size: int = 2048
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy_net: Compression policy network
            learning_rate: Learning rate for policy updates
            gamma: Discount factor for rewards
            gae_lambda: Lambda for Generalized Advantage Estimation
            eps_clip: PPO clipping parameter
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for PPO updates
            buffer_size: Maximum buffer size for experiences
        """
        self.device = setup_cuda_environment()
        
        self.policy_net = policy_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Experience buffer
        self.buffer = deque(maxlen=buffer_size)
        self.episode_buffer = []
        
        # Training statistics
        self.training_stats = {
            'total_updates': 0,
            'policy_losses': [],
            'value_losses': [],
            'entropy_values': [],
            'learning_rates': [],
            'explained_variance': []
        }
        
        # Reward computation parameters
        self.reward_weights = {
            'loss_weight': -1.0,        # Negative because we want to minimize loss
            'memory_weight': 0.5,       # Positive because we want to save memory
            'compression_error_weight': -0.1  # Negative because we want to minimize error
        }
        
        # Baseline tracking for reward normalization
        self.reward_baseline = None
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        self.baseline_momentum = 0.99
    
    def collect_experience(
        self,
        features: Dict[str, torch.Tensor],
        action: Dict[str, Any],
        reward: float,
        done: bool = False
    ) -> None:
        """
        Collect experience tuple for training.
        
        Args:
            features: Input features for policy
            action: Action taken by policy
            reward: Reward received for the action
            done: Whether episode is done
        """
        # Get policy output for the given features
        with torch.no_grad():
            policy_output = self.policy_net(features)
            value = policy_output['value']
            
            # Compute log probability for the taken action
            log_prob = self._compute_log_prob(policy_output, action)
        
        experience = {
            'features': {k: v.clone() for k, v in features.items()},
            'action': action.copy(),
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done
        }
        
        self.episode_buffer.append(experience)
        
        # If episode is done, compute advantages and add to main buffer
        if done:
            self._process_episode()
    
    def _compute_log_prob(self, policy_output: Dict, action: Dict[str, Any]) -> float:
        """Compute log probability of taken action."""
        # Gradient bits log prob
        gradient_bits_options = [32, 16, 8, 4, 2]
        gradient_bits_idx = gradient_bits_options.index(action['gradient_bits'])
        grad_log_prob = torch.log(policy_output['gradient_bits_probs'][0, gradient_bits_idx] + 1e-8)
        
        # Momentum bits log prob  
        momentum_bits_options = [32, 16, 8]
        momentum_bits_idx = momentum_bits_options.index(action['momentum_bits'])
        momentum_log_prob = torch.log(policy_output['momentum_bits_probs'][0, momentum_bits_idx] + 1e-8)
        
        # Total log prob (sum for independent actions)
        total_log_prob = grad_log_prob + momentum_log_prob
        
        return total_log_prob.item()
    
    def _process_episode(self) -> None:
        """Process completed episode and compute advantages."""
        if not self.episode_buffer:
            return
        
        # Extract values and rewards
        values = [exp['value'] for exp in self.episode_buffer]
        rewards = [exp['reward'] for exp in self.episode_buffer]
        
        # Compute returns and advantages using GAE
        advantages, returns = self._compute_gae(rewards, values)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Create experience tuples and add to buffer
        for i, exp in enumerate(self.episode_buffer):
            experience = Experience(
                features=exp['features'],
                action=exp['action'], 
                reward=rewards[i],
                value=values[i],
                log_prob=exp['log_prob'],
                advantage=advantages[i],
                return_=returns[i]
            )
            self.buffer.append(experience)
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def _compute_gae(self, rewards: List[float], values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        # Compute returns (discounted cumulative rewards)
        returns[-1] = rewards[-1] + self.gamma * values[-1]  # Bootstrap with value
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t]  # Bootstrap
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        return advantages, returns
    
    def compute_reward(
        self,
        loss: float,
        memory_saved: float,
        compression_error: float,
        step: int = 0
    ) -> float:
        """
        Compute reward signal for RL training.
        
        Args:
            loss: Current training loss
            memory_saved: Memory saved by compression (0-1 ratio)
            compression_error: Compression error magnitude
            step: Current training step
            
        Returns:
            Reward value
        """
        # Normalize loss (use running baseline)
        if self.reward_baseline is None:
            self.reward_baseline = loss
        else:
            self.reward_baseline = self.baseline_momentum * self.reward_baseline + (1 - self.baseline_momentum) * loss
        
        normalized_loss = (loss - self.reward_baseline) / max(abs(self.reward_baseline), 1e-6)
        
        # Compute reward components
        loss_reward = self.reward_weights['loss_weight'] * normalized_loss
        memory_reward = self.reward_weights['memory_weight'] * memory_saved
        error_penalty = self.reward_weights['compression_error_weight'] * compression_error
        
        # Base reward
        reward = loss_reward + memory_reward + error_penalty
        
        # Add bonus for maintaining stable performance
        stability_bonus = 0.1 if abs(normalized_loss) < 0.1 else 0.0
        reward += stability_bonus
        
        # Update running statistics
        self.reward_running_mean = 0.99 * self.reward_running_mean + 0.01 * reward
        self.reward_running_std = 0.99 * self.reward_running_std + 0.01 * (reward - self.reward_running_mean) ** 2
        
        return reward
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Convert buffer to training data
        experiences = list(self.buffer)
        
        # Extract components
        features_batch = self._collate_features([exp.features for exp in experiences])
        actions_batch = [exp.action for exp in experiences]
        advantages = torch.tensor([exp.advantage for exp in experiences], dtype=torch.float32, device=self.device)
        returns = torch.tensor([exp.return_ for exp in experiences], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([exp.log_prob for exp in experiences], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32, device=self.device)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        
        # PPO updates
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(experiences))
            
            for start_idx in range(0, len(experiences), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(experiences))
                batch_indices = indices[start_idx:end_idx]
                
                # Batch data
                batch_features = {k: v[batch_indices] for k, v in features_batch.items()}
                batch_actions = [actions_batch[i] for i in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                policy_outputs = []
                for i, action in enumerate(batch_actions):
                    # Get features for single sample
                    sample_features = {k: v[i:i+1] for k, v in batch_features.items()}
                    output = self.policy_net(sample_features)
                    policy_outputs.append(output)
                
                # Compute losses
                policy_loss, value_loss, entropy = self._compute_ppo_loss(
                    policy_outputs, batch_actions, batch_advantages, 
                    batch_returns, batch_old_log_probs
                )
                
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Record statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        # Update training statistics
        self.training_stats['total_updates'] += 1
        self.training_stats['policy_losses'].extend(policy_losses)
        self.training_stats['value_losses'].extend(value_losses)
        self.training_stats['entropy_values'].extend(entropies)
        
        # Clear buffer after training
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'total_updates': self.training_stats['total_updates']
        }
    
    def _collate_features(self, features_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate list of feature dictionaries into batched tensors."""
        if not features_list:
            return {}
        
        batched_features = {}
        keys = features_list[0].keys()
        
        for key in keys:
            values = [features[key] for features in features_list]
            
            # Stack tensors
            try:
                batched_features[key] = torch.stack(values).to(self.device)
            except:
                # Handle case where tensors have different sizes
                max_size = max(v.numel() for v in values)
                padded_values = []
                for v in values:
                    if v.numel() < max_size:
                        padded = torch.zeros(max_size)
                        padded[:v.numel()] = v.flatten()
                        padded_values.append(padded)
                    else:
                        padded_values.append(v.flatten()[:max_size])
                batched_features[key] = torch.stack(padded_values).to(self.device)
        
        return batched_features
    
    def _compute_ppo_loss(
        self,
        policy_outputs: List[Dict],
        actions: List[Dict],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO loss components."""
        batch_size = len(policy_outputs)
        
        # Compute current log probabilities and values
        current_log_probs = []
        current_values = []
        entropies = []
        
        for i, (output, action) in enumerate(zip(policy_outputs, actions)):
            # Current log probability
            log_prob = self._compute_log_prob_tensor(output, action)
            current_log_probs.append(log_prob)
            
            # Current value
            current_values.append(output['value'])
            
            # Entropy
            grad_entropy = -(output['gradient_bits_probs'] * torch.log(output['gradient_bits_probs'] + 1e-8)).sum()
            momentum_entropy = -(output['momentum_bits_probs'] * torch.log(output['momentum_bits_probs'] + 1e-8)).sum()
            entropies.append(grad_entropy + momentum_entropy)
        
        current_log_probs = torch.stack(current_log_probs)
        current_values = torch.stack(current_values).squeeze()
        entropy = torch.stack(entropies).mean()
        
        # PPO policy loss
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = torch.nn.functional.mse_loss(current_values, returns)
        
        return policy_loss, value_loss, entropy
    
    def _compute_log_prob_tensor(self, policy_output: Dict, action: Dict) -> torch.Tensor:
        """Compute log probability as tensor for gradient computation."""
        # Gradient bits log prob
        gradient_bits_options = [32, 16, 8, 4, 2]
        gradient_bits_idx = gradient_bits_options.index(action['gradient_bits'])
        grad_log_prob = torch.log(policy_output['gradient_bits_probs'][0, gradient_bits_idx] + 1e-8)
        
        # Momentum bits log prob
        momentum_bits_options = [32, 16, 8]
        momentum_bits_idx = momentum_bits_options.index(action['momentum_bits'])
        momentum_log_prob = torch.log(policy_output['momentum_bits_probs'][0, momentum_bits_idx] + 1e-8)
        
        return grad_log_prob + momentum_log_prob
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'reward_baseline': self.reward_baseline,
            'reward_running_mean': self.reward_running_mean,
            'reward_running_std': self.reward_running_std
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.reward_baseline = checkpoint.get('reward_baseline', None)
        self.reward_running_mean = checkpoint.get('reward_running_mean', 0.0)
        self.reward_running_std = checkpoint.get('reward_running_std', 1.0)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_stats['policy_losses']:
            return {}
        
        recent_window = 100
        recent_policy_losses = self.training_stats['policy_losses'][-recent_window:]
        recent_value_losses = self.training_stats['value_losses'][-recent_window:]
        recent_entropies = self.training_stats['entropy_values'][-recent_window:]
        
        summary = {
            'total_updates': self.training_stats['total_updates'],
            'recent_policy_loss': float(np.mean(recent_policy_losses)),
            'recent_value_loss': float(np.mean(recent_value_losses)),
            'recent_entropy': float(np.mean(recent_entropies)),
            'policy_loss_trend': float(np.mean(np.diff(recent_policy_losses[-20:]))) if len(recent_policy_losses) >= 20 else 0.0,
            'buffer_size': len(self.buffer),
            'reward_baseline': self.reward_baseline or 0.0,
            'reward_running_mean': self.reward_running_mean,
            'reward_running_std': self.reward_running_std
        }
        
        return summary