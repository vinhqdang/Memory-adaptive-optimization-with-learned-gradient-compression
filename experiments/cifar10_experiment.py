"""
CIFAR-10 Experimental Pipeline

Implements the proof-of-concept experiment on CIFAR-10 with ResNet-50
as described in the research plan.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from mango import MANGO, CompressionPolicyNet, AdaptiveCompressionPolicy
from mango.ppo_trainer import PPOTrainer
from mango.utils import MemoryProfiler, DeviceManager, setup_cuda_environment, estimate_model_memory


class ResNet50CIFAR10(nn.Module):
    """
    ResNet-50 adapted for CIFAR-10 (32x32 input images).
    """
    
    def __init__(self, num_classes=10):
        """Initialize ResNet-50 for CIFAR-10."""
        super().__init__()
        
        # Use torchvision ResNet-50 as base but adapt for CIFAR-10
        from torchvision.models import resnet50
        self.resnet = resnet50(pretrained=False, num_classes=num_classes)
        
        # Adapt first convolution for 32x32 input (instead of 224x224)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for small input size
        
    def forward(self, x):
        return self.resnet(x)


class CIFAR10Experiment:
    """
    Main experimental class for CIFAR-10 experiments.
    
    Handles data loading, model training, and comparison between
    MANGO and baseline optimizers.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_epochs: int = 100,
        learning_rate: float = 0.1,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        """
        Initialize CIFAR-10 experiment.
        
        Args:
            data_dir: Directory to store CIFAR-10 dataset
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Base learning rate
            device: Device for training (auto-detect if None)
            seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Setup device
        self.device = device or setup_cuda_environment()
        
        # Experiment configuration
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        
        # Initialize components
        self.device_manager = DeviceManager()
        self.memory_profiler = MemoryProfiler(self.device)
        
        # Models and optimizers will be created for each experiment
        self.model = None
        self.optimizer = None
        
        # Results storage
        self.results = {}
        self.training_logs = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Data loaders will be initialized in setup_data
        self.train_loader = None
        self.test_loader = None
        
        # Initialize data loaders
        self.setup_data()
    
    def setup_data(self):
        """Setup CIFAR-10 data loaders with appropriate transforms."""
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"CIFAR-10 dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    def create_model(self) -> nn.Module:
        """Create and initialize ResNet-50 model for CIFAR-10."""
        model = ResNet50CIFAR10(num_classes=10)
        model = model.to(self.device)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        return model
    
    def run_baseline_experiment(self, optimizer_name: str) -> Dict[str, Any]:
        """
        Run experiment with baseline optimizer.
        
        Args:
            optimizer_name: Name of baseline optimizer ('adam', 'sgd', 'adamw')
            
        Returns:
            Experiment results dictionary
        """
        self.logger.info(f"Starting baseline experiment: {optimizer_name}")
        
        # Create fresh model
        self.model = self.create_model()
        
        # Create baseline optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Run training
        results = self._train_model(f"baseline_{optimizer_name}")
        
        return results
    
    def run_mango_experiment(
        self,
        use_learned_policy: bool = False,
        policy_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run experiment with MANGO optimizer.
        
        Args:
            use_learned_policy: Whether to use learned compression policy
            policy_config: Configuration for policy network
            
        Returns:
            Experiment results dictionary
        """
        experiment_name = "mango_learned" if use_learned_policy else "mango_adaptive"
        self.logger.info(f"Starting MANGO experiment: {experiment_name}")
        
        # Create fresh model
        self.model = self.create_model()
        
        # Setup compression policy
        if use_learned_policy:
            # Create and initialize policy network
            policy_config = policy_config or {}
            policy_net = CompressionPolicyNet(
                feature_dim=policy_config.get('feature_dim', 64),
                hidden_dim=policy_config.get('hidden_dim', 128),
                num_layers=policy_config.get('num_layers', 2)
            )
            
            # Setup PPO trainer for online learning
            ppo_trainer = PPOTrainer(policy_net)
            
            compression_policy = policy_net
        else:
            # Use adaptive heuristic policy
            compression_policy = AdaptiveCompressionPolicy()
            ppo_trainer = None
        
        # Create MANGO optimizer
        self.optimizer = MANGO(
            self.model.parameters(),
            lr=self.learning_rate,
            policy_net=compression_policy,
            error_feedback=True,
            policy_update_freq=100
        )
        
        # Run training with policy learning
        results = self._train_model(experiment_name, ppo_trainer=ppo_trainer)
        
        return results
    
    def _train_model(
        self,
        experiment_name: str,
        ppo_trainer: Optional[PPOTrainer] = None
    ) -> Dict[str, Any]:
        """
        Core training loop.
        
        Args:
            experiment_name: Name for logging and results
            ppo_trainer: Optional PPO trainer for policy learning
            
        Returns:
            Training results dictionary
        """
        # Initialize results tracking
        results = {
            'experiment_name': experiment_name,
            'start_time': time.time(),
            'train_losses': [],
            'train_accuracies': [],
            'test_losses': [],
            'test_accuracies': [],
            'memory_usage': [],
            'compression_stats': [],
            'learning_rates': []
        }
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[50, 75],
            gamma=0.1
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0.0
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_accuracy = self._train_epoch(
                epoch, criterion, ppo_trainer
            )
            
            # Testing phase
            test_loss, test_accuracy = self._test_epoch(epoch, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Record results
            results['train_losses'].append(train_loss)
            results['train_accuracies'].append(train_accuracy)
            results['test_losses'].append(test_loss)
            results['test_accuracies'].append(test_accuracy)
            results['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Memory profiling
            self.memory_profiler.record_memory(epoch, f"epoch_{epoch}")
            memory_stats = self.memory_profiler.get_detailed_usage()
            results['memory_usage'].append(memory_stats)
            
            # Compression statistics (for MANGO)
            if hasattr(self.optimizer, 'get_compression_stats'):
                comp_stats = self.optimizer.get_compression_stats()
                results['compression_stats'].append(comp_stats)
            
            # Update best accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Memory usage logging
            if self.device.type == 'cuda':
                gpu_memory = memory_stats.get('gpu_allocated', 0)
                self.logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Finalize results
        results.update({
            'end_time': time.time(),
            'total_training_time': time.time() - results['start_time'],
            'best_test_accuracy': best_accuracy,
            'final_test_accuracy': test_accuracy,
            'peak_memory_gb': self.memory_profiler.get_peak_memory(),
            'memory_savings': self.memory_profiler.get_memory_savings()
        })
        
        # Add model and optimizer specific stats
        results['model_parameters'] = sum(p.numel() for p in self.model.parameters())
        results['model_memory_estimate'] = estimate_model_memory(self.model)
        
        if hasattr(self.optimizer, 'get_compression_ratio'):
            results['final_compression_ratio'] = self.optimizer.compressor.get_compression_ratio()
        
        # PPO training statistics
        if ppo_trainer:
            results['ppo_stats'] = ppo_trainer.get_training_summary()
        
        return results
    
    def _train_epoch(
        self,
        epoch: int,
        criterion: nn.Module,
        ppo_trainer: Optional[PPOTrainer] = None
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Collect features for policy learning (if using MANGO with learned policy)
            if ppo_trainer and hasattr(self.optimizer, 'collect_gradient_features'):
                features = self.optimizer.collect_gradient_features()
                
                # Get current compression configuration
                compression_config = self.optimizer.get_compression_config()
                
                # Compute reward for RL training
                memory_usage = self.memory_profiler.get_usage_ratio()
                compression_error = 0.0  # TODO: get actual compression error
                reward = ppo_trainer.compute_reward(
                    loss.item(),
                    1.0 - memory_usage,  # Memory saved
                    compression_error,
                    epoch * len(self.train_loader) + batch_idx
                )
                
                # Collect experience for PPO
                ppo_trainer.collect_experience(features, compression_config, reward)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update policy (if using PPO)
        if ppo_trainer:
            # Signal episode end for PPO
            if hasattr(self.optimizer, 'collect_gradient_features'):
                features = self.optimizer.collect_gradient_features()
                compression_config = self.optimizer.get_compression_config()
                final_reward = ppo_trainer.compute_reward(total_loss / len(self.train_loader), 0.0, 0.0)
                ppo_trainer.collect_experience(features, compression_config, final_reward, done=True)
            
            # Update policy
            ppo_stats = ppo_trainer.update_policy()
            if ppo_stats:
                self.logger.info(f"PPO Update - Policy Loss: {ppo_stats['policy_loss']:.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _test_epoch(self, epoch: int, criterion: nn.Module) -> Tuple[float, float]:
        """Test for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def run_comparative_study(self) -> Dict[str, Any]:
        """
        Run comprehensive comparison between all methods.
        
        Returns:
            Complete comparison results
        """
        self.logger.info("Starting comprehensive comparative study")
        
        comparison_results = {
            'experiment_config': {
                'dataset': 'CIFAR-10',
                'model': 'ResNet-50',
                'epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'device': str(self.device),
                'seed': self.seed
            },
            'results': {}
        }
        
        # Baseline experiments
        baselines = ['adam', 'sgd', 'adamw']
        for baseline in baselines:
            try:
                results = self.run_baseline_experiment(baseline)
                comparison_results['results'][f'baseline_{baseline}'] = results
                
                # Clear GPU memory between experiments
                del self.model, self.optimizer
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Failed to run baseline {baseline}: {e}")
                comparison_results['results'][f'baseline_{baseline}'] = {'error': str(e)}
        
        # MANGO experiments
        mango_configs = [
            {'use_learned_policy': False, 'name': 'adaptive'},
            {'use_learned_policy': True, 'name': 'learned'}
        ]
        
        for config in mango_configs:
            try:
                results = self.run_mango_experiment(
                    use_learned_policy=config['use_learned_policy']
                )
                comparison_results['results'][f"mango_{config['name']}"] = results
                
                # Clear GPU memory between experiments
                del self.model, self.optimizer
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Failed to run MANGO {config['name']}: {e}")
                comparison_results['results'][f"mango_{config['name']}"] = {'error': str(e)}
        
        # Add comparison summary
        comparison_results['summary'] = self._generate_comparison_summary(
            comparison_results['results']
        )
        
        return comparison_results
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparison statistics."""
        summary = {
            'best_accuracy': {'method': '', 'value': 0.0},
            'lowest_memory': {'method': '', 'value': float('inf')},
            'fastest_training': {'method': '', 'value': float('inf')},
            'best_compression': {'method': '', 'value': 1.0}
        }
        
        for method, result in results.items():
            if 'error' in result:
                continue
            
            # Best accuracy
            final_acc = result.get('final_test_accuracy', 0.0)
            if final_acc > summary['best_accuracy']['value']:
                summary['best_accuracy'] = {'method': method, 'value': final_acc}
            
            # Lowest memory
            peak_memory = result.get('peak_memory_gb', float('inf'))
            if peak_memory < summary['lowest_memory']['value']:
                summary['lowest_memory'] = {'method': method, 'value': peak_memory}
            
            # Fastest training
            training_time = result.get('total_training_time', float('inf'))
            if training_time < summary['fastest_training']['value']:
                summary['fastest_training'] = {'method': method, 'value': training_time}
            
            # Best compression
            compression_ratio = result.get('final_compression_ratio', 1.0)
            if compression_ratio > summary['best_compression']['value']:
                summary['best_compression'] = {'method': method, 'value': compression_ratio}
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save experimental results to JSON file."""
        # Convert any tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_tensors(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")