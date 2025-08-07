"""
MANGO-LRQ Demo: ResNet-18 on CIFAR-10 (2GB VRAM)

Demonstrates the enhanced MANGO-LRQ optimizer with:
- Hybrid low-rank + quantized compression
- TinyFormer policy network
- Memory profiling and optimization
- Comparison with baseline optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

# Import MANGO components
from mango.enhanced_optimizer import EnhancedMANGO
from mango.mango_lrq import CompressionConfig
from mango.memory_profiler import create_memory_profiler, get_current_gpu_memory_gb, reset_peak_memory_stats
from mango.amp_support import create_amp_context
from mango.power_monitor import PowerMonitor


class ResNet18CIFAR10(nn.Module):
    """Lightweight ResNet-18 adapted for CIFAR-10."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_planes, planes, blocks, stride):
        """Create a residual layer."""
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_cifar10_dataloaders(batch_size=64, num_workers=2):
    """Get CIFAR-10 train and test dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader


def create_optimizer(name, model_params, lr=0.01, **kwargs):
    """Create optimizer by name."""
    if name == 'mango_lrq':
        compression_config = CompressionConfig(
            rank=kwargs.get('rank', 4),
            bits_P=kwargs.get('bits_p', 8),
            bits_Q=kwargs.get('bits_q', 8),
            momentum_precision='fp16',
            use_nf4=kwargs.get('use_nf4', True),
            error_feedback=True,
            variance_reduction=True,
            reference_steps=10
        )
        
        return EnhancedMANGO(
            model_params,
            lr=lr,
            weight_decay=kwargs.get('weight_decay', 5e-4),
            compression_config=compression_config,
            use_mango_lrq=True,
            use_tinyformer=True,
            enable_amp=kwargs.get('enable_amp', True),
            enable_profiling=True,
            profiler_output_dir="./demo_profiler_logs"
        )
    
    elif name == 'adam':
        return torch.optim.Adam(
            model_params, lr=lr, weight_decay=kwargs.get('weight_decay', 5e-4)
        )
    
    elif name == 'adamw':
        return torch.optim.AdamW(
            model_params, lr=lr, weight_decay=kwargs.get('weight_decay', 0.01)
        )
    
    elif name == 'sgd':
        return torch.optim.SGD(
            model_params, lr=lr, momentum=0.9, weight_decay=kwargs.get('weight_decay', 5e-4)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


class CIFAR10Trainer:
    """Trainer for CIFAR-10 experiments."""
    
    def __init__(self, model, optimizer, device, memory_profiler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.memory_profiler = memory_profiler
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize AMP if optimizer supports it
        if hasattr(optimizer, 'set_amp_context'):
            self.amp_context = create_amp_context(optimizer, enabled=True)
            optimizer.set_amp_context(self.amp_context)
        else:
            self.amp_context = None
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.memory_usage = []
        self.compression_stats = []
    
    def train_epoch(self, trainloader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Profile memory usage
            if self.memory_profiler:
                with self.memory_profiler.profile_step(f"epoch_{epoch}_batch_{batch_idx}"):
                    loss, predictions = self._training_step(inputs, targets)
            else:
                loss, predictions = self._training_step(inputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log progress
            if batch_idx % 100 == 0:
                current_memory = get_current_gpu_memory_gb()
                print(f'Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, Memory: {current_memory:.2f}GB')
        
        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch} completed in {epoch_time:.1f}s - '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def _training_step(self, inputs, targets):
        """Single training step."""
        self.optimizer.zero_grad()
        
        if self.amp_context:
            with self.amp_context.autocast_context():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            self.amp_context.backward(loss)
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
        
        self.optimizer.step()
        
        return loss, outputs
    
    def test(self, testloader):
        """Test model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.amp_context:
                    with self.amp_context.autocast_context():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        self.test_accuracies.append(accuracy)
        
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy
    
    def get_optimizer_stats(self):
        """Get optimizer-specific statistics."""
        stats = {}
        
        if hasattr(self.optimizer, 'get_compression_stats'):
            stats['compression'] = self.optimizer.get_compression_stats()
        
        if hasattr(self.optimizer, 'get_memory_usage'):
            stats['memory'] = self.optimizer.get_memory_usage()
        
        return stats


def run_demo_experiment(config):
    """Run a single demo experiment."""
    print(f"\n{'='*60}")
    print(f"Running {config['optimizer']} experiment")
    print(f"Configuration: {config}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Reset memory stats
    if torch.cuda.is_available():
        reset_peak_memory_stats()
    
    # Create data loaders
    trainloader, testloader = get_cifar10_dataloaders(
        batch_size=config['batch_size'], 
        num_workers=2
    )
    
    # Create model
    model = ResNet18CIFAR10(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = create_optimizer(
        config['optimizer'], 
        model.parameters(), 
        lr=config['learning_rate'],
        **config.get('optimizer_kwargs', {})
    )
    
    # Create memory profiler
    memory_profiler = create_memory_profiler("./demo_profiler_logs") if config.get('enable_profiling', True) else None
    
    # Create trainer
    trainer = CIFAR10Trainer(model, optimizer, device, memory_profiler)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_acc = trainer.train_epoch(trainloader, epoch)
        
        # Test
        test_acc = trainer.test(testloader)
        
        # Record memory usage
        current_memory = get_current_gpu_memory_gb()
        trainer.memory_usage.append(current_memory)
        
        # Get compression statistics
        if hasattr(optimizer, 'get_compression_stats'):
            compression_stats = optimizer.get_compression_stats()
            trainer.compression_stats.append(compression_stats)
            
            if 'avg_compression_ratio' in compression_stats:
                print(f'Compression ratio: {compression_stats["avg_compression_ratio"]:.2f}x')
    
    total_time = time.time() - start_time
    
    # Final results
    results = {
        'optimizer': config['optimizer'],
        'config': config,
        'training_time': total_time,
        'final_train_accuracy': trainer.train_accuracies[-1] if trainer.train_accuracies else 0,
        'final_test_accuracy': trainer.test_accuracies[-1] if trainer.test_accuracies else 0,
        'best_test_accuracy': max(trainer.test_accuracies) if trainer.test_accuracies else 0,
        'peak_memory_gb': torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
        'avg_memory_gb': np.mean(trainer.memory_usage) if trainer.memory_usage else 0,
        'train_losses': trainer.train_losses,
        'train_accuracies': trainer.train_accuracies,
        'test_accuracies': trainer.test_accuracies,
        'memory_usage': trainer.memory_usage
    }
    
    # Add optimizer-specific statistics
    optimizer_stats = trainer.get_optimizer_stats()
    if optimizer_stats:
        results['optimizer_stats'] = optimizer_stats
    
    # Add compression statistics
    if trainer.compression_stats:
        results['compression_evolution'] = trainer.compression_stats
    
    # Memory profiling report
    if memory_profiler:
        memory_report_path = memory_profiler.save_report(f'demo_memory_{config["optimizer"]}.json')
        results['memory_report_path'] = memory_report_path
        results['detailed_memory_stats'] = memory_profiler.get_compression_memory_report()
    
    print(f"\n{config['optimizer']} Results:")
    print(f"Training time: {total_time:.1f}s")
    print(f"Final test accuracy: {results['final_test_accuracy']:.2f}%")
    print(f"Best test accuracy: {results['best_test_accuracy']:.2f}%")
    print(f"Peak memory usage: {results['peak_memory_gb']:.2f}GB")
    
    return results


def plot_comparison_results(all_results, save_path="demo_comparison.png"):
    """Plot comparison of different optimizers."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MANGO-LRQ Demo: ResNet-18 on CIFAR-10', fontsize=16)
    
    optimizers = list(all_results.keys())
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Test accuracy over epochs
    axes[0, 0].set_title('Test Accuracy')
    for i, (opt_name, results) in enumerate(all_results.items()):
        epochs = range(1, len(results['test_accuracies']) + 1)
        axes[0, 0].plot(epochs, results['test_accuracies'], 
                       label=opt_name, color=colors[i % len(colors)])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training loss over epochs
    axes[0, 1].set_title('Training Loss')
    for i, (opt_name, results) in enumerate(all_results.items()):
        epochs = range(1, len(results['train_losses']) + 1)
        axes[0, 1].plot(epochs, results['train_losses'], 
                       label=opt_name, color=colors[i % len(colors)])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory usage comparison
    axes[1, 0].set_title('Memory Usage Comparison')
    memory_data = []
    labels = []
    for opt_name, results in all_results.items():
        memory_data.append(results['peak_memory_gb'])
        labels.append(opt_name)
    
    bars = axes[1, 0].bar(labels, memory_data, color=colors[:len(labels)])
    axes[1, 0].set_ylabel('Peak Memory (GB)')
    axes[1, 0].set_xticklabels(labels, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, memory_data):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    # Final accuracy vs memory trade-off
    axes[1, 1].set_title('Accuracy vs Memory Trade-off')
    for i, (opt_name, results) in enumerate(all_results.items()):
        x = results['peak_memory_gb']
        y = results['best_test_accuracy']
        axes[1, 1].scatter(x, y, s=100, color=colors[i % len(colors)], 
                          label=opt_name, alpha=0.7)
        axes[1, 1].annotate(opt_name, (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    axes[1, 1].set_xlabel('Peak Memory (GB)')
    axes[1, 1].set_ylabel('Best Test Accuracy (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    
    return fig


def main():
    """Main function for demo."""
    parser = argparse.ArgumentParser(description='MANGO-LRQ Demo on CIFAR-10')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--compare-all', action='store_true',
                       help='Run comparison with all optimizers')
    parser.add_argument('--optimizer', type=str, default='mango_lrq',
                       choices=['mango_lrq', 'adam', 'adamw', 'sgd'],
                       help='Single optimizer to test')
    parser.add_argument('--rank', type=int, default=4,
                       help='Low-rank approximation rank')
    parser.add_argument('--bits-p', type=int, default=8,
                       help='Quantization bits for P matrix')
    parser.add_argument('--bits-q', type=int, default=8,
                       help='Quantization bits for Q matrix')
    parser.add_argument('--use-nf4', action='store_true',
                       help='Use NF4 quantization')
    
    args = parser.parse_args()
    
    print("MANGO-LRQ Demo: ResNet-18 on CIFAR-10")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    
    if args.compare_all:
        # Run comparison with multiple optimizers
        optimizer_configs = [
            {
                'optimizer': 'mango_lrq',
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'enable_profiling': True,
                'optimizer_kwargs': {
                    'rank': args.rank,
                    'bits_p': args.bits_p,
                    'bits_q': args.bits_q,
                    'use_nf4': args.use_nf4,
                    'enable_amp': True
                }
            },
            {
                'optimizer': 'adam',
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'enable_profiling': False,
                'optimizer_kwargs': {}
            },
            {
                'optimizer': 'adamw',
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'enable_profiling': False,
                'optimizer_kwargs': {}
            },
            {
                'optimizer': 'sgd',
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'enable_profiling': False,
                'optimizer_kwargs': {}
            }
        ]
        
        # Run all experiments
        all_results = {}
        for config in optimizer_configs:
            try:
                results = run_demo_experiment(config)
                all_results[config['optimizer']] = results
                
                # Save individual results
                with open(f'demo_results_{config["optimizer"]}.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Clear GPU cache between experiments
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error running {config['optimizer']}: {e}")
                continue
        
        # Create comparison plots
        if len(all_results) > 1:
            plot_comparison_results(all_results, "mango_lrq_demo_comparison.png")
        
        # Save combined results
        with open('demo_comparison_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*60}")
        print("DEMO SUMMARY")
        print(f"{'='*60}")
        
        for opt_name, results in all_results.items():
            print(f"{opt_name:12}: Accuracy={results['best_test_accuracy']:5.2f}%, "
                  f"Memory={results['peak_memory_gb']:4.2f}GB, "
                  f"Time={results['training_time']:5.1f}s")
        
        # Highlight MANGO-LRQ advantages
        if 'mango_lrq' in all_results:
            mango_results = all_results['mango_lrq']
            print(f"\nMANGO-LRQ Compression Stats:")
            if 'compression_evolution' in mango_results and mango_results['compression_evolution']:
                final_stats = mango_results['compression_evolution'][-1]
                if 'avg_compression_ratio' in final_stats:
                    print(f"  Final compression ratio: {final_stats['avg_compression_ratio']:.2f}x")
                if 'avg_compression_error' in final_stats:
                    print(f"  Average compression error: {final_stats['avg_compression_error']:.6f}")
    
    else:
        # Run single optimizer experiment
        config = {
            'optimizer': args.optimizer,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'enable_profiling': args.optimizer == 'mango_lrq',
            'optimizer_kwargs': {
                'rank': args.rank,
                'bits_p': args.bits_p,
                'bits_q': args.bits_q,
                'use_nf4': args.use_nf4,
                'enable_amp': True
            } if args.optimizer == 'mango_lrq' else {}
        }
        
        results = run_demo_experiment(config)
        
        # Save results
        with open(f'demo_results_{args.optimizer}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to demo_results_{args.optimizer}.json")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()