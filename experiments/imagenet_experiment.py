"""
ImageNet Experiment with ResNet-50 and ViT-B

Implements comprehensive ImageNet experiments as specified in planv2.md:
- Full training with ResNet-50 and ViT-B/16 models
- MANGO-LRQ vs baseline optimizer comparisons  
- Memory profiling and performance analysis
- 4-bit gradient + rank-4 low-rank compression tests
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import timm
import wandb
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import argparse

# Import MANGO components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mango.enhanced_optimizer import EnhancedMANGO
from mango.mango_lrq import CompressionConfig
from mango.tinyformer_policy import TinyFormerPolicyNet
from mango.memory_profiler import create_memory_profiler
from mango.amp_support import create_amp_context
from experiments.baseline_optimizers import get_baseline_optimizer


class ImageNetDataModule:
    """DataModule for ImageNet dataset."""
    
    def __init__(
        self, 
        data_dir: str = "./data/imagenet",
        batch_size: int = 256,
        num_workers: int = 8,
        image_size: int = 224
    ):
        """
        Initialize ImageNet data module.
        
        Args:
            data_dir: Path to ImageNet dataset
            batch_size: Training batch size
            num_workers: Number of data loading workers
            image_size: Input image size
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            self.normalize
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256/224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders."""
        # Check if ImageNet dataset exists
        if not os.path.exists(os.path.join(self.data_dir, 'train')):
            print(f"Warning: ImageNet dataset not found at {self.data_dir}")
            print("Using CIFAR-100 as a substitute for demonstration")
            return self._get_cifar100_loaders()
        
        train_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir,
            split='train',
            transform=self.train_transform
        )
        
        val_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir,
            split='val',
            transform=self.val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, val_loader
    
    def _get_cifar100_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Fallback to CIFAR-100 if ImageNet not available."""
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            self.normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=val_transform
        )
        
        # Adjust number of classes for CIFAR-100
        self.num_classes = 100
        
        train_loader = DataLoader(
            train_dataset, batch_size=min(self.batch_size, 128),
            shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=min(self.batch_size, 128),
            shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        
        return train_loader, val_loader


def create_model(model_name: str, num_classes: int = 1000) -> nn.Module:
    """
    Create model (ResNet-50 or ViT-B).
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        return model
    
    elif model_name == "vit_small_patch16_224":
        # Smaller ViT for memory-constrained experiments
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ImageNetTrainer:
    """Trainer for ImageNet experiments with comprehensive evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str,
        config: Dict[str, Any],
        device: torch.device,
        use_wandb: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer_name: Name of optimizer to use
            config: Experiment configuration
            device: Training device
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.optimizer_name = optimizer_name
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize memory profiler
        self.memory_profiler = create_memory_profiler("./imagenet_profiler_logs")
        
        # Initialize AMP context
        if config.get('enable_amp', True):
            self.amp_context = create_amp_context(self.optimizer, enabled=True)
            if hasattr(self.optimizer, 'set_amp_context'):
                self.optimizer.set_amp_context(self.amp_context)
        else:
            self.amp_context = None
        
        # Training metrics
        self.training_metrics = defaultdict(list)
        self.best_val_acc = 0.0
        
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        lr = self.config.get('learning_rate', 0.1)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if self.optimizer_name == "enhanced_mango":
            compression_config = CompressionConfig(
                rank=self.config.get('rank', 4),
                bits_P=self.config.get('bits_p', 8),
                bits_Q=self.config.get('bits_q', 8),
                momentum_precision=self.config.get('momentum_precision', 'fp16'),
                use_nf4=self.config.get('use_nf4', True),
                error_feedback=True,
                variance_reduction=True,
                reference_steps=10
            )
            
            return EnhancedMANGO(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                compression_config=compression_config,
                use_mango_lrq=True,
                use_tinyformer=True,
                enable_amp=self.config.get('enable_amp', True),
                enable_profiling=True
            )
        
        else:
            return get_baseline_optimizer(
                self.optimizer_name,
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Profile memory usage for this step
            with self.memory_profiler.profile_step(f"epoch_{epoch}_batch_{batch_idx}"):
                # Forward pass with AMP if enabled
                if self.amp_context:
                    with self.amp_context.autocast_context():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    # Scaled backward pass
                    self.amp_context.backward(loss)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Calculate metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)
            
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['accuracy'] += accuracy
            epoch_metrics['batches'] += 1
            
            # Log intermediate results
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\tAccuracy: {accuracy:.4f}')
                
                # Log memory usage
                memory_stats = self.optimizer.get_memory_usage() if hasattr(self.optimizer, 'get_memory_usage') else {}
                if memory_stats:
                    print(f'Memory: {memory_stats.get("current_memory_gb", 0):.2f}GB current, '
                          f'{memory_stats.get("peak_memory_gb", 0):.2f}GB peak')
                
                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_accuracy': accuracy,
                        'epoch': epoch,
                        'batch': batch_idx,
                        **{f'memory_{k}': v for k, v in memory_stats.items()}
                    })
        
        # Calculate epoch averages
        epoch_time = time.time() - start_time
        epoch_metrics = {k: v / epoch_metrics['batches'] for k, v in epoch_metrics.items() if k != 'batches'}
        epoch_metrics['epoch_time'] = epoch_time
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.amp_context:
                    with self.amp_context.autocast_context():
                        output = self.model(data)
                        val_loss += self.criterion(output, target).item()
                else:
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        print(f'\nValidation set: Average loss: {val_loss:.4f}, '
              f'Accuracy: {correct}/{total} ({100. * val_accuracy:.2f}%)\n')
        
        if self.use_wandb:
            wandb.log({
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch': epoch
            })
        
        return {'val_loss': val_loss, 'val_accuracy': val_accuracy}
    
    def run_experiment(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int
    ) -> Dict[str, Any]:
        """Run complete training experiment."""
        print(f"Starting {self.optimizer_name} experiment on {self.config['model_name']}")
        
        results = {
            'optimizer': self.optimizer_name,
            'model': self.config['model_name'],
            'config': self.config,
            'epoch_results': [],
            'final_metrics': {},
            'memory_report': {}
        }
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update best validation accuracy
            if val_metrics['val_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_accuracy']
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.config
                }, f'best_model_{self.optimizer_name}_{self.config["model_name"]}.pth')
            
            # Store epoch results
            epoch_result = {**train_metrics, **val_metrics, 'epoch': epoch}
            results['epoch_results'].append(epoch_result)
            
            # Get compression statistics if available
            if hasattr(self.optimizer, 'get_compression_stats'):
                compression_stats = self.optimizer.get_compression_stats()
                epoch_result['compression_stats'] = compression_stats
                
                if self.use_wandb:
                    wandb.log({
                        'compression_ratio': compression_stats.get('avg_compression_ratio', 1.0),
                        'compression_error': compression_stats.get('avg_compression_error', 0.0),
                        'epoch': epoch
                    })
        
        # Final results
        results['final_metrics'] = {
            'best_val_accuracy': self.best_val_acc,
            'final_val_accuracy': results['epoch_results'][-1]['val_accuracy'],
            'final_train_accuracy': results['epoch_results'][-1]['accuracy'],
            'total_epochs': num_epochs
        }
        
        # Memory profiling report
        if self.memory_profiler:
            memory_report_path = self.memory_profiler.save_report(
                f'memory_report_{self.optimizer_name}_{self.config["model_name"]}.json'
            )
            results['memory_report'] = self.memory_profiler.get_compression_memory_report()
            results['memory_report_path'] = memory_report_path
        
        # Optimizer-specific statistics
        if hasattr(self.optimizer, 'get_compression_stats'):
            results['final_compression_stats'] = self.optimizer.get_compression_stats()
        
        return results


def run_imagenet_experiments(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive ImageNet experiments.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Complete experiment results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if config.get('use_wandb', True):
        wandb.init(
            project="mango-imagenet-experiments",
            config=config,
            name=f"{config['optimizer']}_{config['model_name']}"
        )
    
    # Initialize data module
    data_module = ImageNetDataModule(
        data_dir=config.get('data_dir', './data/imagenet'),
        batch_size=config.get('batch_size', 256),
        num_workers=config.get('num_workers', 8),
        image_size=config.get('image_size', 224)
    )
    
    train_loader, val_loader = data_module.get_dataloaders()
    
    # Adjust num_classes if using CIFAR-100 fallback
    num_classes = getattr(data_module, 'num_classes', 1000)
    
    # Create model
    model = create_model(config['model_name'], num_classes)
    print(f"Created {config['model_name']} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = ImageNetTrainer(
        model=model,
        optimizer_name=config['optimizer'],
        config=config,
        device=device,
        use_wandb=config.get('use_wandb', True)
    )
    
    # Run training
    results = trainer.run_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 90)
    )
    
    # Save results
    output_path = f'imagenet_results_{config["optimizer"]}_{config["model_name"]}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")
    
    if config.get('use_wandb', True):
        wandb.finish()
    
    return results


def main():
    """Main function for running ImageNet experiments."""
    parser = argparse.ArgumentParser(description='ImageNet Experiments with MANGO-LRQ')
    
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'vit_base_patch16_224', 'vit_small_patch16_224'],
                       help='Model architecture')
    parser.add_argument('--optimizer', type=str, default='enhanced_mango',
                       choices=['enhanced_mango', 'adam', 'adamw', 'sgd', 'adafactor'],
                       help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--rank', type=int, default=4,
                       help='Low-rank approximation rank for MANGO-LRQ')
    parser.add_argument('--bits-p', type=int, default=4,
                       help='Quantization bits for P matrix')
    parser.add_argument('--bits-q', type=int, default=4,
                       help='Quantization bits for Q matrix')
    parser.add_argument('--use-nf4', action='store_true',
                       help='Use NF4 quantization')
    parser.add_argument('--disable-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = {
        'model_name': args.model,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'data_dir': args.data_dir,
        'rank': args.rank,
        'bits_p': args.bits_p,
        'bits_q': args.bits_q,
        'use_nf4': args.use_nf4,
        'momentum_precision': 'fp16',
        'weight_decay': 1e-4,
        'enable_amp': not args.disable_amp,
        'use_wandb': not args.disable_wandb,
        'num_workers': 8,
        'image_size': 224
    }
    
    print(f"Running ImageNet experiment with config: {config}")
    
    # Run experiment
    results = run_imagenet_experiments(config)
    
    print("Experiment completed!")
    print(f"Best validation accuracy: {results['final_metrics']['best_val_accuracy']:.4f}")
    
    if 'memory_report' in results:
        memory_report = results['memory_report']
        if 'overall_stats' in memory_report:
            peak_memory = memory_report['overall_stats'].get('peak_memory_gb', 0)
            print(f"Peak memory usage: {peak_memory:.2f} GB")
    
    return results


if __name__ == "__main__":
    main()