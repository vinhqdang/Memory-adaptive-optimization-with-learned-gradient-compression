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
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from experiments.cifar10_experiment import CIFAR10Experiment
from experiments.evaluation_framework import EvaluationFramework, create_evaluation_summary


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge YAML configuration with command line arguments.
    Command line arguments take precedence over config file values.
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed command line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Start with config file values
    merged_config = config.copy()
    
    # Override with command line arguments (if provided)
    if hasattr(args, 'data_dir') and args.data_dir != './data':
        merged_config.setdefault('dataset', {})['data_dir'] = args.data_dir
    
    if hasattr(args, 'batch_size') and args.batch_size != 128:
        merged_config.setdefault('dataset', {})['batch_size'] = args.batch_size
    
    if hasattr(args, 'epochs') and args.epochs != 100:
        merged_config.setdefault('training', {})['epochs'] = args.epochs
    
    if hasattr(args, 'learning_rate') and args.learning_rate != 0.1:
        merged_config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'optimizer'):
        merged_config.setdefault('optimizer', {})['type'] = args.optimizer
    
    if hasattr(args, 'seed') and args.seed != 42:
        merged_config.setdefault('experiment', {})['seed'] = args.seed
    
    if hasattr(args, 'output_dir') and args.output_dir != './results':
        merged_config.setdefault('experiment', {})['output_dir'] = args.output_dir
    
    # Add mode and other experiment settings
    if hasattr(args, 'mode'):
        merged_config.setdefault('experiment', {})['mode'] = args.mode
    
    return merged_config


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


def run_single_experiment_with_config(args: argparse.Namespace, config: Dict[str, Any]):
    """Run a single experiment with YAML configuration."""
    optimizer_type = config.get('optimizer', {}).get('type', args.optimizer)
    print(f"Running single experiment with {optimizer_type}")
    
    # Extract configuration values
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    experiment_config = config.get('experiment', {})
    
    # Initialize experiment with config values
    experiment = CIFAR10Experiment(
        data_dir=dataset_config.get('data_dir', args.data_dir),
        batch_size=dataset_config.get('batch_size', args.batch_size),
        num_epochs=training_config.get('epochs', args.epochs),
        learning_rate=training_config.get('learning_rate', args.learning_rate),
        seed=experiment_config.get('seed', args.seed)
    )
    
    # Run experiment based on optimizer type
    if optimizer_type in ['adam', 'sgd', 'adamw']:
        results = experiment.run_baseline_experiment(optimizer_type)
    elif optimizer_type == 'mango_adaptive':
        results = experiment.run_mango_experiment(use_learned_policy=False)
    elif optimizer_type == 'mango_learned':
        results = experiment.run_mango_experiment(use_learned_policy=True)
    elif optimizer_type == 'enhanced_mango':
        # Use enhanced MANGO with full configuration
        results = experiment.run_enhanced_mango_experiment(config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Save results
    output_dir = experiment_config.get('output_dir', args.output_dir)
    seed = experiment_config.get('seed', args.seed)
    output_file = os.path.join(output_dir, f"results_{optimizer_type}_seed{seed}.json")
    experiment.save_results(results, output_file)
    
    print(f"Results saved to {output_file}")
    print(f"Final accuracy: {results['final_test_accuracy']:.2f}%")
    print(f"Peak memory: {results['peak_memory_gb']:.2f} GB")


def run_comparative_study_with_config(args: argparse.Namespace, config: Dict[str, Any]):
    """Run comprehensive comparative study with YAML configuration."""
    print("Running comprehensive comparative study...")
    
    # Extract configuration values
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    experiment_config = config.get('experiment', {})
    
    # Initialize experiment with config values
    experiment = CIFAR10Experiment(
        data_dir=dataset_config.get('data_dir', args.data_dir),
        batch_size=dataset_config.get('batch_size', args.batch_size),
        num_epochs=training_config.get('epochs', args.epochs),
        learning_rate=training_config.get('learning_rate', args.learning_rate),
        seed=experiment_config.get('seed', args.seed)
    )
    
    # Run comparative study
    comparison_results = experiment.run_comparative_study()
    
    # Save raw results
    output_dir = experiment_config.get('output_dir', args.output_dir)
    results_file = os.path.join(output_dir, "comparative_study_results.json")
    experiment.save_results(comparison_results, results_file)
    
    # Initialize evaluation framework
    seed = experiment_config.get('seed', args.seed)
    evaluator = EvaluationFramework(
        output_dir=output_dir,
        experiment_name=f"mango_comparison_seed{seed}"
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


def run_ablation_study_with_config(args: argparse.Namespace, config: Dict[str, Any]):
    """Run ablation study on MANGO components with YAML configuration."""
    print("Running MANGO ablation study...")
    
    # Extract configuration values
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    experiment_config = config.get('experiment', {})
    
    # Initialize experiment with config values
    experiment = CIFAR10Experiment(
        data_dir=dataset_config.get('data_dir', args.data_dir),
        batch_size=dataset_config.get('batch_size', args.batch_size),
        num_epochs=training_config.get('epochs', args.epochs),
        learning_rate=training_config.get('learning_rate', args.learning_rate),
        seed=experiment_config.get('seed', args.seed)
    )
    
    output_dir = experiment_config.get('output_dir', args.output_dir)
    seed = experiment_config.get('seed', args.seed)
    
    evaluator = EvaluationFramework(
        output_dir=output_dir,
        experiment_name=f"mango_ablation_seed{seed}"
    )
    
    # Test different MANGO configurations based on config
    ablation_config = config.get('experiments', {}).get('ablation', {})
    
    if ablation_config.get('enabled', False):
        # Use components specified in config
        components = ablation_config.get('components', [])
        configs = []
        
        for component in components:
            if component == 'tinyformer_policy':
                configs.append({'use_learned_policy': True, 'name': 'mango_learned'})
            elif component == 'nf4_quantization':
                configs.append({'use_nf4': True, 'name': 'mango_nf4'})
            # Add more component-specific configs as needed
    else:
        # Default ablation configurations
        configs = [
            {'use_learned_policy': False, 'name': 'mango_adaptive'},
            {'use_learned_policy': True, 'name': 'mango_learned'},
        ]
    
    for config_item in configs:
        print(f"Testing configuration: {config_item['name']}")
        try:
            if 'use_learned_policy' in config_item:
                results = experiment.run_mango_experiment(
                    use_learned_policy=config_item['use_learned_policy']
                )
            else:
                # Run enhanced MANGO with specific component config
                results = experiment.run_enhanced_mango_experiment(config, config_item)
            
            evaluator.add_experiment_results(config_item['name'], results)
            
            # Clear GPU memory
            if hasattr(experiment, 'model'):
                del experiment.model, experiment.optimizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Failed to run {config_item['name']}: {e}")
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
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.yaml',
        help='Path to YAML configuration file'
    )
    
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
        choices=['adam', 'sgd', 'adamw', 'mango_adaptive', 'mango_learned', 'enhanced_mango'],
        default='enhanced_mango',
        help='Optimizer to use for single experiment'
    )
    
    # Experiment parameters (these override config file values)
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
    
    # Additional options for YAML config integration
    parser.add_argument('--print-config', action='store_true',
                       help='Print loaded configuration and exit')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configuration file and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Merge with command line arguments
    final_config = merge_config_with_args(config, args)
    
    # Print configuration if requested
    if args.print_config:
        print("Final Configuration:")
        print(yaml.dump(final_config, indent=2))
        return
    
    # Validate configuration if requested
    if args.validate_config:
        print(f"Configuration file {args.config} loaded successfully")
        print(f"Configuration contains {len(final_config)} sections")
        
        # Basic validation
        required_sections = ['experiment', 'training', 'dataset', 'optimizer']
        missing_sections = [s for s in required_sections if s not in final_config]
        if missing_sections:
            print(f"Warning: Missing sections in config: {missing_sections}")
        else:
            print("✅ All required sections present")
        return
    
    # Setup logging
    log_level = final_config.get('debug', {}).get('log_level', args.log_level)
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Log configuration info
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Running in {final_config.get('experiment', {}).get('mode', args.mode)} mode")
    
    # Create output directory
    output_dir = final_config.get('experiment', {}).get('output_dir', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set GPU memory limit if using CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_fraction = final_config.get('hardware', {}).get('gpu', {}).get('memory_fraction', args.gpu_memory_fraction)
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            logger.info(f"Set GPU memory fraction to {gpu_memory_fraction}")
    except Exception as e:
        logger.warning(f"Could not set GPU memory fraction: {e}")
    
    # Save final configuration for reproducibility
    config_save_path = os.path.join(output_dir, 'experiment_config.yaml')
    try:
        with open(config_save_path, 'w') as f:
            yaml.dump(final_config, f, indent=2)
        logger.info(f"Experiment configuration saved to: {config_save_path}")
    except Exception as e:
        logger.warning(f"Could not save experiment configuration: {e}")
    
    try:
        # Run experiment based on mode
        mode = final_config.get('experiment', {}).get('mode', args.mode)
        
        if mode == 'single':
            run_single_experiment_with_config(args, final_config)
            
        elif mode == 'comparative':
            evaluation_results = run_comparative_study_with_config(args, final_config)
            logger.info("Comparative study completed successfully")
            
        elif mode == 'ablation':
            evaluation_results = run_ablation_study_with_config(args, final_config)
            logger.info("Ablation study completed successfully")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()