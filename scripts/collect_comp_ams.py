"""
Imitation Learning Data Collection for MANGO Compression Policies

Collects compression decisions and multi-objective rewards from expert policies
to train TinyFormer policy network through behavioral cloning and inverse
reinforcement learning approaches.

This script implements the COMP-AMS (Compression-Aware Memory Sampling) 
algorithm for collecting high-quality training data for policy learning.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
import os
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import MANGO components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mango.enhanced_optimizer import EnhancedMANGO
from mango.mango_lrq import CompressionConfig
from mango.tinyformer_policy import TinyFormerPolicyNet, extract_gradient_features
from mango.statistics import GradientStatistics
from mango.power_monitor import PowerMonitor, MultiObjectiveReward
from mango.memory_profiler import create_memory_profiler
from experiments.baseline_optimizers import get_baseline_optimizer


@dataclass
class CompressionSample:
    """
    Single data sample for imitation learning.
    
    Contains gradient features, compression decisions, and multi-objective rewards
    from expert policy execution.
    """
    # Input features (35 dimensions)
    gradient_features: np.ndarray
    
    # Expert compression decisions
    compression_rank: int
    compression_bits_p: int  
    compression_bits_q: int
    use_nf4: bool
    enable_error_feedback: bool
    enable_variance_reduction: bool
    
    # Multi-objective rewards  
    loss_reward: float
    memory_reward: float
    energy_reward: float
    time_reward: float
    combined_reward: float
    
    # Context information
    step: int
    epoch: int
    model_name: str
    layer_name: str
    param_shape: Tuple[int, ...]
    param_count: int
    
    # Performance metrics
    compression_ratio: float
    compression_error: float
    memory_usage_mb: float
    processing_time_ms: float


class CompressionExpert:
    """
    Expert compression policy for generating training data.
    
    Implements heuristic-based and oracle-based compression decisions
    to provide high-quality demonstration data for imitation learning.
    """
    
    def __init__(
        self, 
        expert_type: str = "heuristic",
        memory_budget_mb: float = 2048,
        energy_budget_watts: float = 250,
        compression_target: float = 8.0
    ):
        """
        Initialize compression expert.
        
        Args:
            expert_type: Type of expert ("heuristic", "oracle", "hybrid")
            memory_budget_mb: Memory budget in MB
            energy_budget_watts: Energy budget in watts
            compression_target: Target compression ratio
        """
        self.expert_type = expert_type
        self.memory_budget_mb = memory_budget_mb
        self.energy_budget_watts = energy_budget_watts
        self.compression_target = compression_target
        
        # Adaptive thresholds based on training progress
        self.adaptive_thresholds = {
            'memory_pressure': 0.8,  # High memory usage threshold
            'energy_efficiency': 0.6,  # Energy efficiency target
            'accuracy_tolerance': 0.01,  # Max acceptable accuracy drop
            'convergence_speed': 0.9  # Minimum convergence rate
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_compression_decision(
        self,
        gradient_features: np.ndarray,
        context: Dict[str, Any],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate expert compression decision.
        
        Args:
            gradient_features: 35-dimensional feature vector
            context: Training context (step, epoch, layer info, etc.)
            current_metrics: Current performance metrics
            
        Returns:
            Compression decision dictionary
        """
        if self.expert_type == "heuristic":
            return self._heuristic_decision(gradient_features, context, current_metrics)
        elif self.expert_type == "oracle":
            return self._oracle_decision(gradient_features, context, current_metrics)
        elif self.expert_type == "hybrid":
            return self._hybrid_decision(gradient_features, context, current_metrics)
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")
    
    def _heuristic_decision(
        self,
        features: np.ndarray,
        context: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Heuristic-based compression decisions."""
        # Extract key features
        grad_norm = features[0]
        grad_sparsity = features[1]
        effective_rank = features[14] if len(features) > 14 else 4
        memory_usage = metrics.get('memory_usage_mb', 0)
        
        # Memory-driven decisions
        memory_pressure = memory_usage / self.memory_budget_mb
        
        if memory_pressure > 0.9:
            # Aggressive compression under high memory pressure
            rank = max(2, int(effective_rank * 0.25))
            bits_p, bits_q = 2, 4
            use_nf4 = True
            error_feedback = True
            variance_reduction = True
        elif memory_pressure > 0.7:
            # Moderate compression
            rank = max(4, int(effective_rank * 0.5))
            bits_p, bits_q = 4, 8
            use_nf4 = True
            error_feedback = True
            variance_reduction = grad_sparsity > 0.5
        else:
            # Conservative compression
            rank = max(8, int(effective_rank * 0.75))
            bits_p, bits_q = 8, 16
            use_nf4 = False
            error_feedback = grad_norm > 0.1
            variance_reduction = False
        
        # Layer-specific adjustments
        layer_name = context.get('layer_name', '')
        if 'attention' in layer_name.lower():
            # Attention layers compress well
            rank = max(2, rank // 2)
            use_nf4 = True
        elif 'embedding' in layer_name.lower():
            # Embedding layers are sensitive
            rank = min(32, rank * 2)
            bits_p = max(8, bits_p * 2)
        
        # Training phase adjustments
        epoch = context.get('epoch', 1)
        if epoch <= 2:
            # Early training: more conservative
            rank = min(16, rank * 2)
            bits_p = max(4, bits_p)
        elif epoch >= 8:
            # Late training: more aggressive
            rank = max(2, rank // 2)
            
        return {
            'compression_rank': rank,
            'compression_bits_p': bits_p,
            'compression_bits_q': bits_q,
            'use_nf4': use_nf4,
            'enable_error_feedback': error_feedback,
            'enable_variance_reduction': variance_reduction
        }
    
    def _oracle_decision(
        self,
        features: np.ndarray,
        context: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Oracle-based compression decisions with perfect information."""
        # Simulate oracle knowledge of optimal compression
        param_count = context.get('param_count', 1000000)
        
        # Oracle selects compression based on theoretical optimal ratios
        if param_count > 1e7:  # Large layers
            optimal_config = {
                'compression_rank': 2,
                'compression_bits_p': 2,
                'compression_bits_q': 4,
                'use_nf4': True,
                'enable_error_feedback': True,
                'enable_variance_reduction': True
            }
        elif param_count > 1e6:  # Medium layers
            optimal_config = {
                'compression_rank': 4,
                'compression_bits_p': 4,
                'compression_bits_q': 8,
                'use_nf4': True,
                'enable_error_feedback': True,
                'enable_variance_reduction': True
            }
        else:  # Small layers
            optimal_config = {
                'compression_rank': 8,
                'compression_bits_p': 8,
                'compression_bits_q': 16,
                'use_nf4': False,
                'enable_error_feedback': False,
                'enable_variance_reduction': False
            }
        
        return optimal_config
    
    def _hybrid_decision(
        self,
        features: np.ndarray,
        context: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Hybrid approach combining heuristic and oracle knowledge."""
        heuristic_decision = self._heuristic_decision(features, context, metrics)
        oracle_decision = self._oracle_decision(features, context, metrics)
        
        # Weighted combination (70% heuristic, 30% oracle)
        alpha = 0.7
        
        # Blend numerical parameters
        rank_blend = int(alpha * heuristic_decision['compression_rank'] + 
                        (1-alpha) * oracle_decision['compression_rank'])
        
        bits_p_blend = int(alpha * heuristic_decision['compression_bits_p'] + 
                          (1-alpha) * oracle_decision['compression_bits_p'])
        
        bits_q_blend = int(alpha * heuristic_decision['compression_bits_q'] + 
                          (1-alpha) * oracle_decision['compression_bits_q'])
        
        # Boolean parameters: use heuristic if high confidence, oracle otherwise
        confidence = metrics.get('compression_confidence', 0.5)
        use_heuristic_bools = confidence > 0.6
        
        return {
            'compression_rank': max(1, rank_blend),
            'compression_bits_p': max(1, bits_p_blend),
            'compression_bits_q': max(4, bits_q_blend),
            'use_nf4': (heuristic_decision['use_nf4'] if use_heuristic_bools 
                       else oracle_decision['use_nf4']),
            'enable_error_feedback': (heuristic_decision['enable_error_feedback'] 
                                    if use_heuristic_bools 
                                    else oracle_decision['enable_error_feedback']),
            'enable_variance_reduction': (heuristic_decision['enable_variance_reduction']
                                        if use_heuristic_bools
                                        else oracle_decision['enable_variance_reduction'])
        }


class CompressionDataCollector:
    """
    Main data collector for MANGO imitation learning.
    
    Orchestrates data collection across multiple models, datasets, and training
    scenarios to create a comprehensive dataset for policy learning.
    """
    
    def __init__(
        self,
        output_dir: str = "./comp_ams_data",
        expert_type: str = "heuristic",
        models: List[str] = None,
        num_samples_per_model: int = 10000,
        enable_visualization: bool = True
    ):
        """
        Initialize compression data collector.
        
        Args:
            output_dir: Directory to save collected data
            expert_type: Type of expert policy to use
            models: List of models to collect data from
            num_samples_per_model: Number of samples to collect per model
            enable_visualization: Enable data visualization
        """
        self.output_dir = output_dir
        self.expert_type = expert_type
        self.num_samples_per_model = num_samples_per_model
        self.enable_visualization = enable_visualization
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Default models if none specified
        self.models = models or [
            'resnet18', 'resnet34', 'resnet50',
            'vit_small', 'vit_base', 
            'bert_base', 'gpt2_medium'
        ]
        
        # Initialize components
        self.expert = CompressionExpert(expert_type=expert_type)
        self.power_monitor = PowerMonitor(sampling_interval=0.1)
        self.multi_objective_reward = MultiObjectiveReward(self.power_monitor)
        
        # Data storage
        self.collected_samples: List[CompressionSample] = []
        self.collection_stats = defaultdict(int)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized COMP-AMS collector for {len(self.models)} models")
        self.logger.info(f"Target: {num_samples_per_model} samples per model")
        self.logger.info(f"Expert type: {expert_type}")
    
    def collect_from_model(
        self,
        model_name: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 3
    ) -> List[CompressionSample]:
        """
        Collect compression samples from a specific model.
        
        Args:
            model_name: Name of the model
            model: PyTorch model instance
            train_loader: Training data loader
            num_epochs: Number of epochs to collect data
            
        Returns:
            List of collected compression samples
        """
        self.logger.info(f"Starting data collection for {model_name}")
        
        # Initialize MANGO optimizer for data collection
        compression_config = CompressionConfig(
            rank=4, bits_P=8, bits_Q=8,
            momentum_precision='fp16',
            use_nf4=True,
            error_feedback=True,
            variance_reduction=True
        )
        
        optimizer = EnhancedMANGO(
            model.parameters(),
            lr=1e-3,
            compression_config=compression_config,
            use_mango_lrq=True,
            use_tinyformer=False,  # Don't use policy network for data collection
            enable_profiling=True
        )
        
        # Initialize statistics collector
        grad_stats = GradientStatistics()
        
        # Start monitoring
        self.power_monitor.start_monitoring()
        
        model_samples = []
        sample_count = 0
        target_samples = self.num_samples_per_model
        
        model.train()
        
        for epoch in range(num_epochs):
            if sample_count >= target_samples:
                break
                
            epoch_start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                           desc=f"{model_name} Epoch {epoch+1}")):
                if sample_count >= target_samples:
                    break
                
                # Move to GPU if available
                device = next(model.parameters()).device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                # Backward pass to compute gradients
                loss.backward()
                
                # Collect samples from each parameter
                for param_idx, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is None or not param.requires_grad:
                        continue
                    
                    # Extract gradient features
                    grad_tensor = param.grad.data
                    gradient_features = extract_gradient_features(
                        grad_tensor, param_idx, grad_stats
                    )
                    
                    # Add power features
                    power_sample = self.power_monitor.get_latest_sample()
                    if power_sample:
                        power_features = np.array([
                            power_sample.ewma_power / 300.0,  # Normalize by max power
                            power_sample.gpu_utilization / 100.0,
                            power_sample.power_efficiency
                        ])
                        gradient_features = np.concatenate([gradient_features, power_features])
                    
                    # Pad to 35 dimensions if needed
                    if len(gradient_features) < 35:
                        padding = np.zeros(35 - len(gradient_features))
                        gradient_features = np.concatenate([gradient_features, padding])
                    elif len(gradient_features) > 35:
                        gradient_features = gradient_features[:35]
                    
                    # Context information
                    context = {
                        'step': batch_idx,
                        'epoch': epoch,
                        'layer_name': name,
                        'param_shape': param.shape,
                        'param_count': param.numel(),
                        'model_name': model_name
                    }
                    
                    # Current metrics
                    memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    current_metrics = {
                        'loss': loss.item(),
                        'memory_usage_mb': memory_usage,
                        'batch_size': data.size(0)
                    }
                    
                    # Get expert compression decision
                    expert_decision = self.expert.get_compression_decision(
                        gradient_features, context, current_metrics
                    )
                    
                    # Create compression config and apply compression
                    expert_config = CompressionConfig(
                        rank=expert_decision['compression_rank'],
                        bits_P=expert_decision['compression_bits_p'],
                        bits_Q=expert_decision['compression_bits_q'],
                        use_nf4=expert_decision['use_nf4'],
                        error_feedback=expert_decision['enable_error_feedback'],
                        variance_reduction=expert_decision['enable_variance_reduction']
                    )
                    
                    # Measure compression performance
                    start_time = time.time()
                    try:
                        compressed_grad, metadata = optimizer.mango_lrq_compressor.compress(
                            grad_tensor, param_idx, expert_config
                        )
                        compression_ratio = metadata.get('compression_ratio', 1.0)
                        compression_error = metadata.get('compression_error', 0.0)
                        processing_time_ms = (time.time() - start_time) * 1000
                    except Exception as e:
                        self.logger.warning(f"Compression failed for {name}: {e}")
                        continue
                    
                    # Compute multi-objective rewards
                    memory_usage_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    processing_time = time.time() - epoch_start_time
                    
                    reward, reward_components = self.multi_objective_reward.compute_reward(
                        loss.item(),
                        memory_usage_after, 
                        processing_time
                    )
                    
                    # Create compression sample
                    sample = CompressionSample(
                        gradient_features=gradient_features,
                        compression_rank=expert_decision['compression_rank'],
                        compression_bits_p=expert_decision['compression_bits_p'],
                        compression_bits_q=expert_decision['compression_bits_q'],
                        use_nf4=expert_decision['use_nf4'],
                        enable_error_feedback=expert_decision['enable_error_feedback'],
                        enable_variance_reduction=expert_decision['enable_variance_reduction'],
                        loss_reward=reward_components['loss_reward'],
                        memory_reward=reward_components['memory_reward'],
                        energy_reward=reward_components.get('energy_reward', 0.0),
                        time_reward=reward_components.get('time_reward', 0.0),
                        combined_reward=reward,
                        step=batch_idx,
                        epoch=epoch,
                        model_name=model_name,
                        layer_name=name,
                        param_shape=param.shape,
                        param_count=param.numel(),
                        compression_ratio=compression_ratio,
                        compression_error=compression_error,
                        memory_usage_mb=memory_usage_after,
                        processing_time_ms=processing_time_ms
                    )
                    
                    model_samples.append(sample)
                    sample_count += 1
                    
                    # Update collection statistics
                    self.collection_stats[f'{model_name}_samples'] += 1
                    self.collection_stats[f'{model_name}_avg_compression'] = (
                        (self.collection_stats[f'{model_name}_avg_compression'] * (sample_count - 1) + 
                         compression_ratio) / sample_count
                    )
                    
                    if sample_count % 1000 == 0:
                        self.logger.info(f"{model_name}: {sample_count}/{target_samples} samples collected")
                        
                # Optimizer step (needed for proper gradient flow)
                optimizer.step()
        
        # Stop monitoring
        self.power_monitor.stop_monitoring()
        
        self.logger.info(f"Completed data collection for {model_name}: {len(model_samples)} samples")
        return model_samples
    
    def collect_full_dataset(self) -> str:
        """
        Collect the complete COMP-AMS dataset across all models.
        
        Returns:
            Path to the saved dataset file
        """
        self.logger.info(f"Starting full dataset collection across {len(self.models)} models")
        
        total_samples_collected = 0
        
        for model_name in self.models:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing model: {model_name}")
            self.logger.info(f"{'='*50}")
            
            # Create model and dataloader (dummy for demonstration)
            model, train_loader = self._create_model_and_dataloader(model_name)
            
            # Collect samples from this model
            model_samples = self.collect_from_model(
                model_name, model, train_loader, num_epochs=2
            )
            
            self.collected_samples.extend(model_samples)
            total_samples_collected += len(model_samples)
            
            self.logger.info(f"Model {model_name} completed: {len(model_samples)} samples")
        
        # Save collected dataset
        dataset_path = self._save_dataset()
        
        # Generate collection report
        self._generate_collection_report()
        
        # Create visualizations
        if self.enable_visualization:
            self._create_visualizations()
        
        self.logger.info(f"\nDataset collection completed!")
        self.logger.info(f"Total samples: {total_samples_collected:,}")
        self.logger.info(f"Dataset saved to: {dataset_path}")
        
        return dataset_path
    
    def _create_model_and_dataloader(self, model_name: str) -> Tuple[nn.Module, torch.utils.data.DataLoader]:
        """Create model and dataloader for data collection."""
        # Create dummy model and dataloader for demonstration
        # In practice, you would load actual models and datasets
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if 'resnet' in model_name:
            if model_name == 'resnet18':
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            elif model_name == 'resnet34':
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
            elif model_name == 'resnet50':
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
            
            # CIFAR-10 style dataloader
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Create dummy dataset for demonstration
            dummy_data = torch.randn(1000, 3, 224, 224)
            dummy_targets = torch.randint(0, 1000, (1000,))
            dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
            
        else:
            # Create simple MLP for other model types
            model = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(), 
                nn.Linear(128, 10)
            )
            
            # Create dummy MNIST-style data
            dummy_data = torch.randn(1000, 784)
            dummy_targets = torch.randint(0, 10, (1000,))
            dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
        
        model = model.to(device)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=0
        )
        
        return model, train_loader
    
    def _save_dataset(self) -> str:
        """Save collected samples to disk."""
        # Convert samples to DataFrame
        sample_dicts = [asdict(sample) for sample in self.collected_samples]
        
        # Handle numpy arrays in gradient_features
        for i, sample_dict in enumerate(sample_dicts):
            sample_dict['gradient_features'] = sample_dict['gradient_features'].tolist()
            sample_dict['param_shape'] = str(sample_dict['param_shape'])
        
        df = pd.DataFrame(sample_dicts)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'comp_ams_dataset.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for complex structures
        json_path = os.path.join(self.output_dir, 'comp_ams_dataset.json')
        with open(json_path, 'w') as f:
            json.dump(sample_dicts, f, indent=2)
        
        # Save as PyTorch tensors for ML training
        torch_path = os.path.join(self.output_dir, 'comp_ams_dataset.pt')
        
        # Prepare tensors
        features = torch.FloatTensor([sample.gradient_features for sample in self.collected_samples])
        compression_ranks = torch.LongTensor([sample.compression_rank for sample in self.collected_samples])
        compression_bits_p = torch.LongTensor([sample.compression_bits_p for sample in self.collected_samples])
        compression_bits_q = torch.LongTensor([sample.compression_bits_q for sample in self.collected_samples])
        combined_rewards = torch.FloatTensor([sample.combined_reward for sample in self.collected_samples])
        
        torch.save({
            'features': features,
            'compression_ranks': compression_ranks,
            'compression_bits_p': compression_bits_p,
            'compression_bits_q': compression_bits_q,
            'combined_rewards': combined_rewards,
            'metadata': {
                'num_samples': len(self.collected_samples),
                'feature_dim': 35,
                'expert_type': self.expert_type,
                'collection_stats': dict(self.collection_stats)
            }
        }, torch_path)
        
        self.logger.info(f"Dataset saved in multiple formats:")
        self.logger.info(f"  CSV: {csv_path}")
        self.logger.info(f"  JSON: {json_path}")
        self.logger.info(f"  PyTorch: {torch_path}")
        
        return torch_path
    
    def _generate_collection_report(self):
        """Generate comprehensive collection report."""
        report_path = os.path.join(self.output_dir, 'collection_report.json')
        
        # Calculate statistics
        total_samples = len(self.collected_samples)
        
        compression_ratios = [s.compression_ratio for s in self.collected_samples]
        compression_errors = [s.compression_error for s in self.collected_samples]
        combined_rewards = [s.combined_reward for s in self.collected_samples]
        
        model_distribution = defaultdict(int)
        layer_type_distribution = defaultdict(int)
        
        for sample in self.collected_samples:
            model_distribution[sample.model_name] += 1
            
            # Extract layer type
            layer_name = sample.layer_name.lower()
            if 'conv' in layer_name:
                layer_type_distribution['convolutional'] += 1
            elif 'linear' in layer_name or 'fc' in layer_name:
                layer_type_distribution['linear'] += 1
            elif 'attention' in layer_name or 'attn' in layer_name:
                layer_type_distribution['attention'] += 1
            elif 'embedding' in layer_name or 'embed' in layer_name:
                layer_type_distribution['embedding'] += 1
            else:
                layer_type_distribution['other'] += 1
        
        report = {
            'collection_summary': {
                'total_samples': total_samples,
                'expert_type': self.expert_type,
                'num_models': len(self.models),
                'collection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'compression_statistics': {
                'avg_compression_ratio': float(np.mean(compression_ratios)),
                'std_compression_ratio': float(np.std(compression_ratios)),
                'avg_compression_error': float(np.mean(compression_errors)),
                'std_compression_error': float(np.std(compression_errors))
            },
            'reward_statistics': {
                'avg_combined_reward': float(np.mean(combined_rewards)),
                'std_combined_reward': float(np.std(combined_rewards)),
                'min_reward': float(np.min(combined_rewards)),
                'max_reward': float(np.max(combined_rewards))
            },
            'distribution_analysis': {
                'model_distribution': dict(model_distribution),
                'layer_type_distribution': dict(layer_type_distribution)
            },
            'expert_decision_analysis': self._analyze_expert_decisions(),
            'collection_quality_metrics': self._calculate_quality_metrics()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Collection report saved to: {report_path}")
    
    def _analyze_expert_decisions(self) -> Dict[str, Any]:
        """Analyze expert decision patterns."""
        ranks = [s.compression_rank for s in self.collected_samples]
        bits_p = [s.compression_bits_p for s in self.collected_samples]
        bits_q = [s.compression_bits_q for s in self.collected_samples]
        use_nf4 = [s.use_nf4 for s in self.collected_samples]
        
        return {
            'rank_distribution': {
                'mean': float(np.mean(ranks)),
                'std': float(np.std(ranks)),
                'min': int(np.min(ranks)),
                'max': int(np.max(ranks)),
                'histogram': np.histogram(ranks, bins=10)[0].tolist()
            },
            'bits_p_distribution': {
                'mean': float(np.mean(bits_p)),
                'unique_values': list(set(bits_p))
            },
            'bits_q_distribution': {
                'mean': float(np.mean(bits_q)),
                'unique_values': list(set(bits_q))
            },
            'nf4_usage_rate': float(np.mean(use_nf4))
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate dataset quality metrics."""
        # Feature diversity
        features_array = np.array([s.gradient_features for s in self.collected_samples])
        feature_stds = np.std(features_array, axis=0)
        feature_diversity = float(np.mean(feature_stds))
        
        # Reward coverage
        rewards = [s.combined_reward for s in self.collected_samples]
        reward_range = max(rewards) - min(rewards)
        
        # Decision diversity
        unique_configs = set()
        for sample in self.collected_samples:
            config = (sample.compression_rank, sample.compression_bits_p, 
                     sample.compression_bits_q, sample.use_nf4)
            unique_configs.add(config)
        
        config_diversity = len(unique_configs) / len(self.collected_samples)
        
        return {
            'feature_diversity': feature_diversity,
            'reward_range': float(reward_range),
            'config_diversity': float(config_diversity),
            'overall_quality_score': float((feature_diversity + config_diversity) / 2)
        }
    
    def _create_visualizations(self):
        """Create visualizations of the collected dataset."""
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Compression ratio distribution
        compression_ratios = [s.compression_ratio for s in self.collected_samples]
        
        plt.figure(figsize=(10, 6))
        plt.hist(compression_ratios, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Compression Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Compression Ratios')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'compression_ratio_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reward distribution
        rewards = [s.combined_reward for s in self.collected_samples]
        
        plt.figure(figsize=(10, 6))
        plt.hist(rewards, bins=50, alpha=0.7, color='green')
        plt.xlabel('Combined Reward')
        plt.ylabel('Frequency') 
        plt.title('Distribution of Multi-Objective Rewards')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'reward_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model-wise statistics
        model_stats = defaultdict(list)
        for sample in self.collected_samples:
            model_stats[sample.model_name].append(sample.compression_ratio)
        
        plt.figure(figsize=(12, 8))
        for i, (model_name, ratios) in enumerate(model_stats.items()):
            plt.boxplot(ratios, positions=[i], labels=[model_name])
        
        plt.xlabel('Model')
        plt.ylabel('Compression Ratio')
        plt.title('Compression Ratio by Model')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'compression_by_model.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to: {viz_dir}")


def main():
    """Main function for COMP-AMS data collection."""
    parser = argparse.ArgumentParser(description='COMP-AMS: Compression-Aware Memory Sampling')
    
    parser.add_argument('--output-dir', type=str, default='./comp_ams_data',
                       help='Output directory for collected data')
    parser.add_argument('--expert-type', type=str, default='heuristic',
                       choices=['heuristic', 'oracle', 'hybrid'],
                       help='Type of expert policy to use')
    parser.add_argument('--models', nargs='+', 
                       default=['resnet18', 'resnet34', 'vit_small'],
                       help='Models to collect data from')
    parser.add_argument('--samples-per-model', type=int, default=5000,
                       help='Number of samples to collect per model')
    parser.add_argument('--disable-visualization', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--memory-budget-mb', type=float, default=2048,
                       help='Memory budget in MB')
    parser.add_argument('--energy-budget-watts', type=float, default=250,
                       help='Energy budget in watts')
    parser.add_argument('--compression-target', type=float, default=8.0,
                       help='Target compression ratio')
    
    args = parser.parse_args()
    
    print("🔬 COMP-AMS: Compression-Aware Memory Sampling")
    print("=" * 50)
    print(f"Expert type: {args.expert_type}")
    print(f"Models: {args.models}")
    print(f"Samples per model: {args.samples_per_model}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Initialize collector
    collector = CompressionDataCollector(
        output_dir=args.output_dir,
        expert_type=args.expert_type,
        models=args.models,
        num_samples_per_model=args.samples_per_model,
        enable_visualization=not args.disable_visualization
    )
    
    # Collect dataset
    dataset_path = collector.collect_full_dataset()
    
    print(f"\n✅ Data collection completed!")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Total samples: {len(collector.collected_samples):,}")
    
    # Print summary statistics
    compression_ratios = [s.compression_ratio for s in collector.collected_samples]
    rewards = [s.combined_reward for s in collector.collected_samples]
    
    print(f"\nDataset Summary:")
    print(f"  Average compression ratio: {np.mean(compression_ratios):.2f}x")
    print(f"  Average reward: {np.mean(rewards):.4f}")
    print(f"  Feature dimensionality: 35")
    print(f"  Models covered: {len(set(s.model_name for s in collector.collected_samples))}")
    
    return dataset_path


if __name__ == "__main__":
    main()