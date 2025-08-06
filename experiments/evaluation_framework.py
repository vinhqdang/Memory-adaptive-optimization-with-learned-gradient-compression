"""
Evaluation Framework

Comprehensive evaluation and metrics collection framework for comparing
MANGO optimizer against baselines across multiple criteria.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
import time
from collections import defaultdict
import logging
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from mango.utils import MemoryProfiler, CompressionAnalyzer


class EvaluationFramework:
    """
    Comprehensive evaluation framework for optimizer comparison.
    
    Handles metrics collection, statistical analysis, visualization,
    and report generation.
    """
    
    def __init__(self, output_dir: str = "./results", experiment_name: str = "mango_evaluation"):
        """
        Initialize evaluation framework.
        
        Args:
            output_dir: Directory to save results and visualizations
            experiment_name: Name of the experiment for organization
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create output directories
        self.results_dir = self.output_dir / experiment_name
        self.plots_dir = self.results_dir / "plots"
        self.data_dir = self.results_dir / "data"
        
        for dir_path in [self.results_dir, self.plots_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        self.metrics = {}
        self.comparisons = {}
        
        # Analysis components
        self.compression_analyzer = CompressionAnalyzer()
        
        # Setup logging
        log_file = self.results_dir / f"{experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Plotting configuration
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def add_experiment_results(self, optimizer_name: str, results: Dict[str, Any]):
        """
        Add experiment results for an optimizer.
        
        Args:
            optimizer_name: Name of the optimizer
            results: Dictionary containing experiment results
        """
        self.results[optimizer_name] = results
        self.logger.info(f"Added results for optimizer: {optimizer_name}")
        
        # Extract and store key metrics
        self._extract_metrics(optimizer_name, results)
    
    def _extract_metrics(self, optimizer_name: str, results: Dict[str, Any]):
        """Extract key metrics from experiment results."""
        metrics = {}
        
        # Performance metrics
        metrics['final_accuracy'] = results.get('final_test_accuracy', 0.0)
        metrics['best_accuracy'] = results.get('best_test_accuracy', 0.0)
        metrics['final_loss'] = results.get('test_losses', [float('inf')])[-1] if results.get('test_losses') else float('inf')
        
        # Training efficiency metrics
        metrics['total_training_time'] = results.get('total_training_time', 0.0)
        metrics['convergence_epoch'] = self._find_convergence_epoch(results)
        metrics['training_stability'] = self._compute_training_stability(results)
        
        # Memory metrics
        metrics['peak_memory_gb'] = results.get('peak_memory_gb', 0.0)
        metrics['memory_savings'] = results.get('memory_savings', 0.0)
        memory_estimate = results.get('model_memory_estimate', {})
        metrics['estimated_training_memory_gb'] = memory_estimate.get('estimated_training_gb', 0.0)
        
        # Compression metrics (for MANGO and compression baselines)
        if 'compression_stats' in results and results['compression_stats']:
            final_stats = results['compression_stats'][-1] if results['compression_stats'] else {}
            metrics['compression_ratio'] = results.get('final_compression_ratio', 1.0)
            metrics['avg_gradient_bits'] = final_stats.get('avg_gradient_bits', 32)
            metrics['avg_sparsity'] = final_stats.get('avg_sparsity_ratio', 0.0)
        else:
            metrics['compression_ratio'] = 1.0
            metrics['avg_gradient_bits'] = 32
            metrics['avg_sparsity'] = 0.0
        
        # PPO training metrics (for learned MANGO)
        if 'ppo_stats' in results:
            ppo_stats = results['ppo_stats']
            metrics['ppo_updates'] = ppo_stats.get('total_updates', 0)
            metrics['final_policy_loss'] = ppo_stats.get('recent_policy_loss', 0.0)
            metrics['final_entropy'] = ppo_stats.get('recent_entropy', 0.0)
        
        self.metrics[optimizer_name] = metrics
    
    def _find_convergence_epoch(self, results: Dict[str, Any]) -> int:
        """Find epoch where training converged (accuracy stabilized)."""
        test_accuracies = results.get('test_accuracies', [])
        if len(test_accuracies) < 10:
            return len(test_accuracies)
        
        # Look for stabilization (variance in last 10 epochs < threshold)
        for i in range(10, len(test_accuracies)):
            recent_acc = test_accuracies[i-10:i]
            if np.var(recent_acc) < 0.1:  # Low variance indicates convergence
                return i
        
        return len(test_accuracies)
    
    def _compute_training_stability(self, results: Dict[str, Any]) -> float:
        """Compute training stability score based on loss variance."""
        train_losses = results.get('train_losses', [])
        if len(train_losses) < 5:
            return 1.0
        
        # Compute variance in loss changes (lower is more stable)
        loss_changes = np.diff(train_losses)
        stability = 1.0 / (1.0 + np.var(loss_changes))
        
        return stability
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Returns:
            Dictionary containing detailed comparison analysis
        """
        if len(self.results) < 2:
            self.logger.warning("Need at least 2 optimizers for comparison")
            return {}
        
        self.logger.info("Generating comprehensive comparison report")
        
        report = {
            'timestamp': time.time(),
            'experiment_name': self.experiment_name,
            'optimizers_compared': list(self.results.keys()),
            'summary': {},
            'detailed_comparison': {},
            'statistical_analysis': {},
            'recommendations': {}
        }
        
        # Summary statistics
        report['summary'] = self._generate_summary_statistics()
        
        # Detailed comparison
        report['detailed_comparison'] = self._generate_detailed_comparison()
        
        # Statistical significance tests
        report['statistical_analysis'] = self._perform_statistical_analysis()
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        # Save report
        report_file = self.results_dir / "comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comparison report saved to {report_file}")
        
        return report
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics across all optimizers."""
        summary = {}
        
        # Find best performers for each metric
        metrics_to_track = [
            'final_accuracy', 'best_accuracy', 'total_training_time',
            'peak_memory_gb', 'compression_ratio', 'training_stability'
        ]
        
        for metric in metrics_to_track:
            if metric in ['total_training_time', 'peak_memory_gb']:
                # Lower is better
                best_optimizer = min(
                    self.metrics.keys(),
                    key=lambda k: self.metrics[k].get(metric, float('inf'))
                )
                best_value = self.metrics[best_optimizer].get(metric, float('inf'))
            else:
                # Higher is better
                best_optimizer = max(
                    self.metrics.keys(),
                    key=lambda k: self.metrics[k].get(metric, 0)
                )
                best_value = self.metrics[best_optimizer].get(metric, 0)
            
            summary[f'best_{metric}'] = {
                'optimizer': best_optimizer,
                'value': best_value
            }
        
        # Compute averages and ranges
        for metric in metrics_to_track:
            values = [self.metrics[opt].get(metric, 0) for opt in self.metrics.keys()]
            summary[f'{metric}_stats'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return summary
    
    def _generate_detailed_comparison(self) -> Dict[str, Any]:
        """Generate detailed optimizer comparison."""
        comparison = {}
        
        # Performance comparison
        comparison['accuracy_comparison'] = {}
        for opt in self.metrics.keys():
            comparison['accuracy_comparison'][opt] = {
                'final_accuracy': self.metrics[opt].get('final_accuracy', 0.0),
                'best_accuracy': self.metrics[opt].get('best_accuracy', 0.0),
                'accuracy_improvement': self.metrics[opt].get('best_accuracy', 0.0) - 
                                      self.metrics[opt].get('final_accuracy', 0.0)
            }
        
        # Memory efficiency comparison
        comparison['memory_comparison'] = {}
        for opt in self.metrics.keys():
            comparison['memory_comparison'][opt] = {
                'peak_memory_gb': self.metrics[opt].get('peak_memory_gb', 0.0),
                'memory_savings': self.metrics[opt].get('memory_savings', 0.0),
                'compression_ratio': self.metrics[opt].get('compression_ratio', 1.0),
                'memory_efficiency_score': self._compute_memory_efficiency_score(opt)
            }
        
        # Training efficiency comparison
        comparison['training_efficiency'] = {}
        for opt in self.metrics.keys():
            comparison['training_efficiency'][opt] = {
                'training_time': self.metrics[opt].get('total_training_time', 0.0),
                'convergence_epoch': self.metrics[opt].get('convergence_epoch', 100),
                'stability': self.metrics[opt].get('training_stability', 0.0),
                'time_to_accuracy': self._compute_time_to_accuracy(opt)
            }
        
        return comparison
    
    def _compute_memory_efficiency_score(self, optimizer_name: str) -> float:
        """Compute composite memory efficiency score."""
        metrics = self.metrics[optimizer_name]
        
        # Combine memory usage, compression ratio, and savings
        memory_gb = metrics.get('peak_memory_gb', 10.0)
        compression_ratio = metrics.get('compression_ratio', 1.0)
        memory_savings = metrics.get('memory_savings', 0.0)
        
        # Normalize and combine (lower memory usage is better)
        memory_score = 1.0 / (1.0 + memory_gb)  # 0 to 1, higher is better
        compression_score = min(compression_ratio / 10.0, 1.0)  # 0 to 1
        savings_score = memory_savings  # Already 0 to 1
        
        # Weighted combination
        efficiency_score = 0.4 * memory_score + 0.3 * compression_score + 0.3 * savings_score
        
        return efficiency_score
    
    def _compute_time_to_accuracy(self, optimizer_name: str, target_accuracy: float = 70.0) -> int:
        """Compute epochs to reach target accuracy."""
        if optimizer_name not in self.results:
            return 100
        
        test_accuracies = self.results[optimizer_name].get('test_accuracies', [])
        
        for epoch, accuracy in enumerate(test_accuracies):
            if accuracy >= target_accuracy:
                return epoch + 1
        
        return len(test_accuracies)
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical significance analysis."""
        # For single runs, we can't do statistical tests
        # This would be expanded for multiple independent runs
        analysis = {
            'note': 'Statistical analysis requires multiple independent runs',
            'single_run_analysis': {}
        }
        
        # Compute relative improvements
        if 'mango_learned' in self.metrics or 'mango_adaptive' in self.metrics:
            baseline_opt = 'adam' if 'adam' in self.metrics else list(self.metrics.keys())[0]
            
            for mango_variant in ['mango_learned', 'mango_adaptive']:
                if mango_variant in self.metrics:
                    analysis['single_run_analysis'][mango_variant] = {
                        'accuracy_improvement_vs_baseline': (
                            self.metrics[mango_variant].get('final_accuracy', 0) -
                            self.metrics[baseline_opt].get('final_accuracy', 0)
                        ),
                        'memory_reduction_vs_baseline': (
                            self.metrics[baseline_opt].get('peak_memory_gb', 0) -
                            self.metrics[mango_variant].get('peak_memory_gb', 0)
                        ),
                        'compression_ratio': self.metrics[mango_variant].get('compression_ratio', 1.0)
                    }
        
        return analysis
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        recommendations = {
            'best_overall': '',
            'best_for_memory_constrained': '',
            'best_for_accuracy': '',
            'best_for_speed': '',
            'insights': []
        }
        
        # Find best performers
        recommendations['best_for_accuracy'] = max(
            self.metrics.keys(),
            key=lambda k: self.metrics[k].get('final_accuracy', 0)
        )
        
        recommendations['best_for_memory_constrained'] = max(
            self.metrics.keys(),
            key=lambda k: self._compute_memory_efficiency_score(k)
        )
        
        recommendations['best_for_speed'] = min(
            self.metrics.keys(),
            key=lambda k: self.metrics[k].get('total_training_time', float('inf'))
        )
        
        # Generate insights
        insights = []
        
        # Memory efficiency insights
        mango_variants = [k for k in self.metrics.keys() if 'mango' in k.lower()]
        if mango_variants:
            baseline_memory = min(
                self.metrics[k].get('peak_memory_gb', float('inf'))
                for k in self.metrics.keys() if 'mango' not in k.lower()
            )
            mango_memory = min(
                self.metrics[k].get('peak_memory_gb', float('inf'))
                for k in mango_variants
            )
            
            if mango_memory < baseline_memory:
                memory_reduction = ((baseline_memory - mango_memory) / baseline_memory) * 100
                insights.append(f"MANGO achieved {memory_reduction:.1f}% memory reduction vs baselines")
        
        # Accuracy insights
        best_accuracy = max(self.metrics[k].get('final_accuracy', 0) for k in self.metrics.keys())
        worst_accuracy = min(self.metrics[k].get('final_accuracy', 0) for k in self.metrics.keys())
        
        if (best_accuracy - worst_accuracy) > 1.0:
            insights.append(f"Significant accuracy difference: {best_accuracy - worst_accuracy:.2f}% spread")
        
        # Compression insights
        compression_ratios = [self.metrics[k].get('compression_ratio', 1.0) for k in self.metrics.keys()]
        max_compression = max(compression_ratios)
        
        if max_compression > 2.0:
            insights.append(f"Maximum compression ratio achieved: {max_compression:.1f}x")
        
        recommendations['insights'] = insights
        
        return recommendations
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the results."""
        self.logger.info("Creating visualizations")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Training curves comparison
        self._plot_training_curves()
        
        # 2. Memory usage comparison
        self._plot_memory_comparison()
        
        # 3. Performance vs Memory trade-off
        self._plot_performance_memory_tradeoff()
        
        # 4. Compression analysis (for MANGO variants)
        self._plot_compression_analysis()
        
        # 5. Summary radar chart
        self._plot_summary_radar()
        
        self.logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def _plot_training_curves(self):
        """Plot training and validation curves for all optimizers."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        for optimizer_name, results in self.results.items():
            epochs = range(1, len(results.get('train_losses', [])) + 1)
            
            # Training loss
            if 'train_losses' in results:
                ax1.plot(epochs, results['train_losses'], label=optimizer_name, alpha=0.8)
            
            # Training accuracy
            if 'train_accuracies' in results:
                ax2.plot(epochs, results['train_accuracies'], label=optimizer_name, alpha=0.8)
            
            # Test loss
            if 'test_losses' in results:
                ax3.plot(epochs, results['test_losses'], label=optimizer_name, alpha=0.8)
            
            # Test accuracy
            if 'test_accuracies' in results:
                ax4.plot(epochs, results['test_accuracies'], label=optimizer_name, alpha=0.8)
        
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        ax3.set_title('Test Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title('Test Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_comparison(self):
        """Plot memory usage comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        optimizers = list(self.metrics.keys())
        peak_memory = [self.metrics[opt].get('peak_memory_gb', 0) for opt in optimizers]
        compression_ratios = [self.metrics[opt].get('compression_ratio', 1.0) for opt in optimizers]
        
        # Peak memory usage
        bars1 = ax1.bar(optimizers, peak_memory, alpha=0.7)
        ax1.set_title('Peak Memory Usage')
        ax1.set_ylabel('Memory (GB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, peak_memory):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Compression ratios
        bars2 = ax2.bar(optimizers, compression_ratios, alpha=0.7, color='orange')
        ax2.set_title('Compression Ratio')
        ax2.set_ylabel('Ratio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Compression')
        
        # Add value labels on bars
        for bar, value in zip(bars2, compression_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}x', ha='center', va='bottom')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_memory_tradeoff(self):
        """Plot performance vs memory trade-off scatter plot."""
        plt.figure(figsize=(10, 8))
        
        for optimizer_name in self.metrics.keys():
            accuracy = self.metrics[optimizer_name].get('final_accuracy', 0)
            memory = self.metrics[optimizer_name].get('peak_memory_gb', 0)
            
            plt.scatter(memory, accuracy, s=100, alpha=0.7, label=optimizer_name)
            plt.annotate(optimizer_name, (memory, accuracy), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Peak Memory Usage (GB)')
        plt.ylabel('Final Test Accuracy (%)')
        plt.title('Performance vs Memory Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add Pareto frontier
        data_points = [(self.metrics[opt].get('peak_memory_gb', 0), 
                       self.metrics[opt].get('final_accuracy', 0)) for opt in self.metrics.keys()]
        data_points.sort()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_memory_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_compression_analysis(self):
        """Plot compression analysis for MANGO variants."""
        mango_results = {k: v for k, v in self.results.items() if 'mango' in k.lower()}
        
        if not mango_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (optimizer_name, results) in enumerate(mango_results.items()):
            if idx >= 4:  # Limit to 4 subplots
                break
            
            compression_stats = results.get('compression_stats', [])
            if not compression_stats:
                continue
            
            steps = range(len(compression_stats))
            gradient_bits = [stat.get('avg_gradient_bits', 32) for stat in compression_stats]
            sparsity_ratios = [stat.get('avg_sparsity_ratio', 0) for stat in compression_stats]
            
            ax = axes[idx]
            ax2 = ax.twinx()
            
            line1 = ax.plot(steps, gradient_bits, 'b-', label='Gradient Bits', alpha=0.8)
            line2 = ax2.plot(steps, sparsity_ratios, 'r-', label='Sparsity Ratio', alpha=0.8)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Bits', color='b')
            ax2.set_ylabel('Sparsity Ratio', color='r')
            ax.set_title(f'Compression Policy: {optimizer_name}')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        
        # Hide unused subplots
        for idx in range(len(mango_results), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_radar(self):
        """Create radar chart summary of all optimizers."""
        metrics_for_radar = [
            'final_accuracy', 'training_stability', 'memory_efficiency',
            'compression_ratio', 'convergence_speed'
        ]
        
        # Prepare data
        optimizers = list(self.metrics.keys())
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        for metric in metrics_for_radar:
            if metric == 'memory_efficiency':
                values = [self._compute_memory_efficiency_score(opt) for opt in optimizers]
            elif metric == 'convergence_speed':
                # Inverse of convergence epoch (faster is better)
                values = [1.0 / max(self.metrics[opt].get('convergence_epoch', 100), 1) for opt in optimizers]
            elif metric == 'final_accuracy':
                values = [self.metrics[opt].get(metric, 0) / 100.0 for opt in optimizers]  # Convert to 0-1
            elif metric == 'compression_ratio':
                values = [min(self.metrics[opt].get(metric, 1.0) / 10.0, 1.0) for opt in optimizers]  # Cap at 10x
            else:
                values = [self.metrics[opt].get(metric, 0) for opt in optimizers]
            
            normalized_data[metric] = values
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, optimizer in enumerate(optimizers):
            values = [normalized_data[metric][i] for metric in metrics_for_radar]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=optimizer, alpha=0.8)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_for_radar])
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{val:.1f}' for val in np.arange(0, 1.1, 0.2)])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Optimizer Comparison Radar Chart', size=16, y=1.08)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'summary_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results_csv(self):
        """Export results to CSV format for further analysis."""
        # Create comprehensive DataFrame
        data_for_export = []
        
        for optimizer_name in self.metrics.keys():
            row = {'optimizer': optimizer_name}
            row.update(self.metrics[optimizer_name])
            data_for_export.append(row)
        
        df = pd.DataFrame(data_for_export)
        csv_file = self.data_dir / "optimizer_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results exported to CSV: {csv_file}")
        
        return df
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper inclusion."""
        latex_metrics = [
            ('final_accuracy', 'Final Acc. (\\%)', '.2f'),
            ('peak_memory_gb', 'Peak Memory (GB)', '.2f'),
            ('total_training_time', 'Training Time (s)', '.1f'),
            ('compression_ratio', 'Compression Ratio', '.1f')
        ]
        
        # Start LaTeX table
        latex = "\\begin{table}[ht]\n"
        latex += "\\centering\n"
        latex += "\\begin{tabular}{|l|" + "c|" * len(latex_metrics) + "}\n"
        latex += "\\hline\n"
        
        # Header
        header = "Optimizer & " + " & ".join([metric[1] for metric in latex_metrics]) + " \\\\\n"
        latex += header
        latex += "\\hline\n"
        
        # Data rows
        for optimizer_name in sorted(self.metrics.keys()):
            row_data = [optimizer_name.replace('_', '\\_')]
            
            for metric_key, _, format_str in latex_metrics:
                value = self.metrics[optimizer_name].get(metric_key, 0)
                if format_str == '.2f':
                    row_data.append(f"{value:.2f}")
                elif format_str == '.1f':
                    row_data.append(f"{value:.1f}")
                else:
                    row_data.append(str(value))
            
            latex += " & ".join(row_data) + " \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Comparison of MANGO with baseline optimizers on CIFAR-10}\n"
        latex += "\\label{tab:optimizer_comparison}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        latex_file = self.data_dir / "comparison_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"LaTeX table saved to {latex_file}")
        
        return latex
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Complete evaluation results
        """
        self.logger.info("Starting complete evaluation pipeline")
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report()
        
        # Create visualizations
        self.create_visualizations()
        
        # Export to CSV
        df = self.export_results_csv()
        
        # Generate LaTeX table
        latex_table = self.generate_latex_table()
        
        # Compile final results
        final_results = {
            'comparison_report': comparison_report,
            'metrics_dataframe': df.to_dict(),
            'latex_table': latex_table,
            'output_directories': {
                'results': str(self.results_dir),
                'plots': str(self.plots_dir),
                'data': str(self.data_dir)
            }
        }
        
        self.logger.info("Complete evaluation finished")
        self.logger.info(f"Results available in: {self.results_dir}")
        
        return final_results


def create_evaluation_summary(results_dir: str) -> str:
    """
    Create a markdown summary of evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Markdown summary string
    """
    results_path = Path(results_dir)
    
    # Load comparison report
    report_file = results_path / "comparison_report.json"
    if not report_file.exists():
        return "# Evaluation Summary\n\nNo comparison report found."
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    # Generate markdown summary
    summary = "# MANGO Optimizer Evaluation Summary\n\n"
    
    if 'summary' in report:
        summary += "## Key Findings\n\n"
        
        best_accuracy = report['summary'].get('best_final_accuracy', {})
        best_memory = report['summary'].get('best_peak_memory_gb', {})
        
        summary += f"- **Best Accuracy**: {best_accuracy.get('optimizer', 'N/A')} "
        summary += f"({best_accuracy.get('value', 0):.2f}%)\n"
        summary += f"- **Lowest Memory**: {best_memory.get('optimizer', 'N/A')} "
        summary += f"({best_memory.get('value', 0):.2f} GB)\n\n"
    
    if 'recommendations' in report:
        recommendations = report['recommendations']
        summary += "## Recommendations\n\n"
        
        for key, value in recommendations.items():
            if key != 'insights' and value:
                summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        if 'insights' in recommendations:
            summary += "\n### Key Insights\n\n"
            for insight in recommendations['insights']:
                summary += f"- {insight}\n"
    
    summary += f"\n## Files Generated\n\n"
    summary += f"- Comparison report: `comparison_report.json`\n"
    summary += f"- Training curves: `plots/training_curves.png`\n"
    summary += f"- Memory comparison: `plots/memory_comparison.png`\n"
    summary += f"- Performance trade-off: `plots/performance_memory_tradeoff.png`\n"
    summary += f"- Summary radar chart: `plots/summary_radar.png`\n"
    summary += f"- CSV data: `data/optimizer_comparison.csv`\n"
    summary += f"- LaTeX table: `data/comparison_table.tex`\n"
    
    # Save summary
    summary_file = results_path / "evaluation_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    return summary