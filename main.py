#!/usr/bin/env python3
"""
Main script for running MANGO experiments

This script provides the entry point for running CIFAR-10 experiments
comparing MANGO optimizer with baseline methods.
"""

import argparse
import sys
import os
import logging
import traceback
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from experiments.cifar10_experiment import CIFAR10Experiment
from experiments.evaluation_framework import EvaluationFramework, create_evaluation_summary


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mango_experiment.log')
        ]
    )


def run_single_experiment(args):
    """Run a single experiment with specified optimizer."""
    print(f"Running single experiment with {args.optimizer}")
    
    # Initialize experiment
    experiment = CIFAR10Experiment(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    # Run experiment based on optimizer type
    if args.optimizer in ['adam', 'sgd', 'adamw']:
        results = experiment.run_baseline_experiment(args.optimizer)
    elif args.optimizer == 'mango_adaptive':
        results = experiment.run_mango_experiment(use_learned_policy=False)
    elif args.optimizer == 'mango_learned':
        results = experiment.run_mango_experiment(use_learned_policy=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Save results
    output_file = f"results_{args.optimizer}_seed{args.seed}.json"
    experiment.save_results(results, output_file)
    
    print(f"Results saved to {output_file}")
    print(f"Final accuracy: {results['final_test_accuracy']:.2f}%")
    print(f"Peak memory: {results['peak_memory_gb']:.2f} GB")


def run_comparative_study(args):
    """Run comprehensive comparative study."""
    print("Running comprehensive comparative study...")
    
    # Initialize experiment
    experiment = CIFAR10Experiment(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    # Run comparative study
    comparison_results = experiment.run_comparative_study()
    
    # Save raw results
    experiment.save_results(comparison_results, "comparative_study_results.json")
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework(
        output_dir=args.output_dir,
        experiment_name=f"mango_comparison_seed{args.seed}"
    )
    
    # Add results to evaluator
    for method_name, method_results in comparison_results['results'].items():
        if 'error' not in method_results:
            evaluator.add_experiment_results(method_name, method_results)
    
    # Run complete evaluation
    evaluation_results = evaluator.run_complete_evaluation()
    
    # Create summary
    summary = create_evaluation_summary(evaluator.results_dir)
    print("\n" + "="*60)
    print(summary)
    print("="*60)
    
    return evaluation_results


def run_ablation_study(args):
    """Run ablation study on MANGO components."""
    print("Running MANGO ablation study...")
    
    experiment = CIFAR10Experiment(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    evaluator = EvaluationFramework(
        output_dir=args.output_dir,
        experiment_name=f"mango_ablation_seed{args.seed}"
    )
    
    # Test different MANGO configurations
    configs = [
        {'use_learned_policy': False, 'name': 'mango_adaptive'},
        {'use_learned_policy': True, 'name': 'mango_learned'},
    ]
    
    for config in configs:
        print(f"Testing configuration: {config['name']}")
        try:
            results = experiment.run_mango_experiment(
                use_learned_policy=config['use_learned_policy']
            )
            evaluator.add_experiment_results(config['name'], results)
            
            # Clear GPU memory
            if hasattr(experiment, 'model'):
                del experiment.model, experiment.optimizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Failed to run {config['name']}: {e}")
            traceback.print_exc()
    
    # Also run baseline for comparison
    try:
        baseline_results = experiment.run_baseline_experiment('adam')
        evaluator.add_experiment_results('baseline_adam', baseline_results)
    except Exception as e:
        print(f"Failed to run baseline: {e}")
    
    # Run evaluation
    evaluation_results = evaluator.run_complete_evaluation()
    
    print(f"\nAblation study complete. Results saved to: {evaluator.results_dir}")
    
    return evaluation_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MANGO Optimizer Experiments")
    
    # Experiment type
    parser.add_argument(
        '--mode', 
        choices=['single', 'comparative', 'ablation'],
        default='comparative',
        help='Experiment mode to run'
    )
    
    # Single experiment options
    parser.add_argument(
        '--optimizer',
        choices=['adam', 'sgd', 'adamw', 'mango_adaptive', 'mango_learned'],
        default='mango_learned',
        help='Optimizer to use for single experiment'
    )
    
    # Experiment parameters
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    # GPU memory limit (optional)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.9,
                       help='Fraction of GPU memory to use')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set GPU memory limit if using CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
            logger.info(f"Set GPU memory fraction to {args.gpu_memory_fraction}")
    except Exception as e:
        logger.warning(f"Could not set GPU memory fraction: {e}")
    
    try:
        # Run experiment based on mode
        if args.mode == 'single':
            run_single_experiment(args)
            
        elif args.mode == 'comparative':
            evaluation_results = run_comparative_study(args)
            logger.info("Comparative study completed successfully")
            
        elif args.mode == 'ablation':
            evaluation_results = run_ablation_study(args)
            logger.info("Ablation study completed successfully")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()