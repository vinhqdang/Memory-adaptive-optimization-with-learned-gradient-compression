"""
LLaMA LoRA Fine-tuning Experiment with MANGO-LRQ

Implements comprehensive LLaMA fine-tuning experiments using:
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- MANGO-LRQ for gradient compression 
- 4-bit quantized base weights with bitsandbytes
- Distributed training support with FSDP/DeepSpeed
- Multi-objective energy-aware optimization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import argparse
import logging

# Import MANGO components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mango.enhanced_optimizer import EnhancedMANGO
from mango.mango_lrq import CompressionConfig
from mango.memory_profiler import create_memory_profiler
from mango.power_monitor import PowerMonitor, MultiObjectiveReward
from mango.zero_utils import create_mango_distributed_trainer
from experiments.baseline_optimizers import get_baseline_optimizer


class LLaMALoRATrainer:
    """
    LLaMA LoRA trainer with MANGO-LRQ optimization.
    
    Supports efficient fine-tuning of large language models with:
    - Parameter-efficient LoRA adaptation
    - MANGO-LRQ gradient compression
    - Multi-objective optimization (loss, memory, energy, time)
    - Distributed training capabilities
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        optimizer_name: str = "enhanced_mango",
        config: Dict[str, Any] = None,
        use_wandb: bool = True,
        enable_distributed: bool = False
    ):
        """
        Initialize LLaMA LoRA trainer.
        
        Args:
            model_name: HuggingFace model name
            optimizer_name: Optimizer to use
            config: Training configuration
            use_wandb: Enable Weights & Biases logging
            enable_distributed: Enable distributed training
        """
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.config = config or self._get_default_config()
        self.use_wandb = use_wandb
        self.enable_distributed = enable_distributed
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.distributed_trainer = None
        
        # Monitoring components
        self.power_monitor = None
        self.memory_profiler = None
        self.multi_objective_reward = None
        
        # Training metrics
        self.training_metrics = defaultdict(list)
        self.best_eval_loss = float('inf')
        
        # Initialize model and tokenizer
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_monitoring()
        
        if enable_distributed:
            self._initialize_distributed_training()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            # Model configuration
            'max_seq_length': 512,
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': 'float16',
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_quant_type': 'nf4',
            
            # LoRA configuration
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            
            # Training configuration
            'learning_rate': 2e-4,
            'num_epochs': 3,
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            
            # MANGO-LRQ configuration
            'compression_rank': 4,
            'compression_bits_p': 4,
            'compression_bits_q': 8,
            'use_nf4_compression': True,
            'enable_amp': True,
            'enable_energy_monitoring': True,
            
            # Dataset configuration
            'dataset_name': 'alpaca',
            'dataset_config': None,
            'max_train_samples': 10000,
            'max_eval_samples': 1000,
            
            # Distributed training
            'distributed_backend': 'fsdp',  # or 'deepspeed'
            'enable_gradient_checkpointing': True,
            
            # Monitoring
            'eval_steps': 500,
            'save_steps': 1000,
            'logging_steps': 50,
            'output_dir': './llama_lora_outputs'\n        }\n    \n    def _initialize_model(self):\n        \"\"\"Initialize LLaMA model with LoRA and quantization.\"\"\"\n        self.logger.info(f\"Loading model: {self.model_name}\")\n        \n        # Setup tokenizer\n        self.tokenizer = AutoTokenizer.from_pretrained(\n            self.model_name,\n            padding_side=\"right\",\n            use_fast=False\n        )\n        \n        # Add pad token if missing\n        if self.tokenizer.pad_token is None:\n            self.tokenizer.pad_token = self.tokenizer.eos_token\n        \n        # Setup quantization config\n        if self.config['load_in_4bit']:\n            bnb_config = BitsAndBytesConfig(\n                load_in_4bit=True,\n                bnb_4bit_compute_dtype=getattr(torch, self.config['bnb_4bit_compute_dtype']),\n                bnb_4bit_use_double_quant=self.config['bnb_4bit_use_double_quant'],\n                bnb_4bit_quant_type=self.config['bnb_4bit_quant_type']\n            )\n        else:\n            bnb_config = None\n        \n        # Load base model\n        self.model = AutoModelForCausalLM.from_pretrained(\n            self.model_name,\n            quantization_config=bnb_config,\n            device_map=\"auto\" if not self.enable_distributed else None,\n            trust_remote_code=True,\n            torch_dtype=torch.float16 if self.config.get('enable_amp', True) else torch.float32\n        )\n        \n        # Enable gradient checkpointing\n        if self.config.get('enable_gradient_checkpointing', True):\n            self.model.gradient_checkpointing_enable()\n        \n        # Setup LoRA configuration\n        lora_config = LoraConfig(\n            task_type=TaskType.CAUSAL_LM,\n            inference_mode=False,\n            r=self.config['lora_r'],\n            lora_alpha=self.config['lora_alpha'],\n            lora_dropout=self.config['lora_dropout'],\n            target_modules=self.config['lora_target_modules']\n        )\n        \n        # Apply LoRA to model\n        self.model = get_peft_model(self.model, lora_config)\n        \n        # Print model info\n        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)\n        total_params = sum(p.numel() for p in self.model.parameters())\n        \n        self.logger.info(f\"Trainable parameters: {trainable_params:,}\")\n        self.logger.info(f\"Total parameters: {total_params:,}\")\n        self.logger.info(f\"Trainable ratio: {trainable_params/total_params:.4f}\")\n    \n    def _initialize_optimizer(self):\n        \"\"\"Initialize MANGO or baseline optimizer.\"\"\"\n        trainable_params = [p for p in self.model.parameters() if p.requires_grad]\n        \n        if self.optimizer_name == \"enhanced_mango\":\n            compression_config = CompressionConfig(\n                rank=self.config['compression_rank'],\n                bits_P=self.config['compression_bits_p'],\n                bits_Q=self.config['compression_bits_q'],\n                momentum_precision='fp16',\n                use_nf4=self.config['use_nf4_compression'],\n                error_feedback=True,\n                variance_reduction=True,\n                reference_steps=10\n            )\n            \n            self.optimizer = EnhancedMANGO(\n                trainable_params,\n                lr=self.config['learning_rate'],\n                weight_decay=self.config['weight_decay'],\n                compression_config=compression_config,\n                use_mango_lrq=True,\n                use_tinyformer=True,\n                enable_amp=self.config['enable_amp'],\n                enable_profiling=True,\n                profiler_output_dir=os.path.join(self.config['output_dir'], 'profiler')\n            )\n        else:\n            self.optimizer = get_baseline_optimizer(\n                self.optimizer_name,\n                trainable_params,\n                lr=self.config['learning_rate'],\n                weight_decay=self.config['weight_decay']\n            )\n        \n        self.logger.info(f\"Initialized optimizer: {self.optimizer_name}\")\n    \n    def _initialize_monitoring(self):\n        \"\"\"Initialize monitoring components.\"\"\"\n        # Memory profiler\n        profiler_dir = os.path.join(self.config['output_dir'], 'profiler')\n        os.makedirs(profiler_dir, exist_ok=True)\n        self.memory_profiler = create_memory_profiler(profiler_dir)\n        \n        # Power monitor\n        if self.config.get('enable_energy_monitoring', True):\n            self.power_monitor = PowerMonitor(\n                sampling_interval=0.1,\n                enable_cpu_monitoring=True,\n                enable_memory_monitoring=True\n            )\n            \n            # Multi-objective reward\n            self.multi_objective_reward = MultiObjectiveReward(\n                power_monitor=self.power_monitor,\n                loss_weight=1.0,\n                memory_weight=0.1,\n                energy_weight=0.05,\n                time_weight=0.02\n            )\n        \n        self.logger.info(\"Monitoring components initialized\")\n    \n    def _initialize_distributed_training(self):\n        \"\"\"Initialize distributed training wrapper.\"\"\"\n        if not torch.distributed.is_initialized():\n            torch.distributed.init_process_group(backend='nccl')\n        \n        training_config = {\n            'distributed_backend': self.config['distributed_backend'],\n            'fsdp_config': {\n                'mixed_precision': True,\n                'sharding_strategy': 'full_shard'\n            },\n            'deepspeed_config': {\n                'train_batch_size': self.config['batch_size'],\n                'gradient_accumulation_steps': self.config['gradient_accumulation_steps'],\n                'fp16': {'enabled': True},\n                'zero_optimization': {'stage': 2}  # Stage 2 for LoRA\n            }\n        }\n        \n        self.distributed_trainer = create_mango_distributed_trainer(\n            self.model, self.optimizer, training_config\n        )\n        \n        self.logger.info(f\"Distributed training initialized: {self.config['distributed_backend']}\")\n    \n    def prepare_datasets(self) -> Tuple[DataLoader, DataLoader]:\n        \"\"\"Prepare training and evaluation datasets.\"\"\"\n        self.logger.info(f\"Loading dataset: {self.config['dataset_name']}\")\n        \n        if self.config['dataset_name'] == 'alpaca':\n            # Load Alpaca dataset\n            dataset = load_dataset(\"tatsu-lab/alpaca\", split=\"train\")\n            \n            def format_alpaca(example):\n                if example['input']:\n                    text = f\"### Instruction:\\n{example['instruction']}\\n\\n### Input:\\n{example['input']}\\n\\n### Response:\\n{example['output']}\"\n                else:\n                    text = f\"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}\"\n                return {'text': text}\n            \n            dataset = dataset.map(format_alpaca)\n            \n        elif self.config['dataset_name'] == 'dolly':\n            # Load Dolly dataset\n            dataset = load_dataset(\"databricks/databricks-dolly-15k\", split=\"train\")\n            \n            def format_dolly(example):\n                context = f\"\\n\\nContext: {example['context']}\" if example['context'] else \"\"\n                text = f\"### Instruction:\\n{example['instruction']}{context}\\n\\n### Response:\\n{example['response']}\"\n                return {'text': text}\n            \n            dataset = dataset.map(format_dolly)\n            \n        else:\n            raise ValueError(f\"Unsupported dataset: {self.config['dataset_name']}\")\n        \n        # Split dataset\n        if len(dataset) > self.config['max_train_samples'] + self.config['max_eval_samples']:\n            dataset = dataset.train_test_split(\n                train_size=self.config['max_train_samples'],\n                test_size=self.config['max_eval_samples'],\n                seed=42\n            )\n            train_dataset = dataset['train']\n            eval_dataset = dataset['test']\n        else:\n            # Use 90/10 split for smaller datasets\n            dataset = dataset.train_test_split(test_size=0.1, seed=42)\n            train_dataset = dataset['train']\n            eval_dataset = dataset['test']\n        \n        # Tokenize datasets\n        def tokenize_function(examples):\n            outputs = self.tokenizer(\n                examples['text'],\n                truncation=True,\n                padding=False,\n                max_length=self.config['max_seq_length'],\n                return_overflowing_tokens=False,\n                return_length=False\n            )\n            return outputs\n        \n        train_dataset = train_dataset.map(\n            tokenize_function,\n            batched=True,\n            remove_columns=train_dataset.column_names,\n            desc=\"Tokenizing train dataset\"\n        )\n        \n        eval_dataset = eval_dataset.map(\n            tokenize_function,\n            batched=True,\n            remove_columns=eval_dataset.column_names,\n            desc=\"Tokenizing eval dataset\"\n        )\n        \n        # Create data collator\n        data_collator = DataCollatorForLanguageModeling(\n            tokenizer=self.tokenizer,\n            mlm=False\n        )\n        \n        # Create dataloaders\n        train_dataloader = DataLoader(\n            train_dataset,\n            batch_size=self.config['batch_size'],\n            shuffle=True,\n            collate_fn=data_collator,\n            num_workers=4,\n            pin_memory=True\n        )\n        \n        eval_dataloader = DataLoader(\n            eval_dataset,\n            batch_size=self.config['batch_size'],\n            shuffle=False,\n            collate_fn=data_collator,\n            num_workers=4,\n            pin_memory=True\n        )\n        \n        self.logger.info(f\"Train samples: {len(train_dataset):,}\")\n        self.logger.info(f\"Eval samples: {len(eval_dataset):,}\")\n        \n        return train_dataloader, eval_dataloader\n    \n    def train_epoch(\n        self,\n        train_dataloader: DataLoader,\n        epoch: int\n    ) -> Dict[str, float]:\n        \"\"\"Train for one epoch.\"\"\"\n        self.model.train()\n        epoch_metrics = defaultdict(float)\n        \n        total_batches = len(train_dataloader)\n        start_time = time.time()\n        \n        for batch_idx, batch in enumerate(train_dataloader):\n            # Move batch to device\n            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v \n                    for k, v in batch.items()}\n            \n            # Training step\n            if self.distributed_trainer:\n                # Use distributed trainer\n                step_metrics = self.distributed_trainer.train_step(\n                    batch['input_ids'],\n                    batch['labels'],\n                    nn.CrossEntropyLoss()\n                )\n            else:\n                # Standard training step\n                step_metrics = self._training_step(batch)\n            \n            # Accumulate metrics\n            for k, v in step_metrics.items():\n                epoch_metrics[k] += v\n            \n            # Multi-objective reward computation\n            if self.multi_objective_reward and batch_idx % 10 == 0:\n                memory_usage = self._get_current_memory_usage()\n                training_time = time.time() - start_time\n                \n                reward, reward_components = self.multi_objective_reward.compute_reward(\n                    step_metrics['loss'],\n                    memory_usage,\n                    training_time\n                )\n                \n                # Update baselines\n                self.multi_objective_reward.update_baselines(\n                    step_metrics['loss'],\n                    memory_usage,\n                    training_time\n                )\n            \n            # Logging\n            if batch_idx % self.config['logging_steps'] == 0:\n                avg_loss = epoch_metrics['loss'] / (batch_idx + 1)\n                memory_gb = self._get_current_memory_usage()\n                \n                self.logger.info(\n                    f\"Epoch {epoch}, Step {batch_idx}/{total_batches}: \"\n                    f\"Loss={avg_loss:.4f}, Memory={memory_gb:.2f}GB\"\n                )\n                \n                # Log compression stats\n                if hasattr(self.optimizer, 'get_compression_stats'):\n                    comp_stats = self.optimizer.get_compression_stats()\n                    if isinstance(comp_stats, dict) and 'error' not in comp_stats:\n                        self.logger.info(\n                            f\"Compression: {comp_stats.get('avg_compression_ratio', 1.0):.2f}x ratio, \"\n                            f\"{comp_stats.get('avg_compression_error', 0.0):.6f} error\"\n                        )\n                \n                # Wandb logging\n                if self.use_wandb:\n                    log_dict = {\n                        'train_loss': step_metrics['loss'],\n                        'learning_rate': self._get_current_lr(),\n                        'memory_gb': memory_gb,\n                        'epoch': epoch,\n                        'step': batch_idx\n                    }\n                    \n                    # Add compression stats\n                    if hasattr(self.optimizer, 'get_compression_stats'):\n                        comp_stats = self.optimizer.get_compression_stats()\n                        if isinstance(comp_stats, dict) and 'error' not in comp_stats:\n                            log_dict.update({\n                                'compression_ratio': comp_stats.get('avg_compression_ratio', 1.0),\n                                'compression_error': comp_stats.get('avg_compression_error', 0.0)\n                            })\n                    \n                    # Add reward components\n                    if self.multi_objective_reward and 'reward_components' in locals():\n                        log_dict.update({f'reward_{k}': v for k, v in reward_components.items()})\n                    \n                    wandb.log(log_dict)\n        \n        # Calculate epoch averages\n        epoch_time = time.time() - start_time\n        for k in epoch_metrics:\n            epoch_metrics[k] /= total_batches\n        epoch_metrics['epoch_time'] = epoch_time\n        \n        return dict(epoch_metrics)\n    \n    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:\n        \"\"\"Perform single training step.\"\"\"\n        self.optimizer.zero_grad()\n        \n        # Forward pass\n        outputs = self.model(**batch)\n        loss = outputs.loss\n        \n        # Backward pass\n        loss.backward()\n        \n        # Gradient clipping\n        if self.config['max_grad_norm'] > 0:\n            torch.nn.utils.clip_grad_norm_(\n                self.model.parameters(),\n                self.config['max_grad_norm']\n            )\n        \n        # Optimizer step\n        self.optimizer.step()\n        \n        return {'loss': loss.item()}\n    \n    def evaluate(\n        self,\n        eval_dataloader: DataLoader,\n        epoch: int\n    ) -> Dict[str, float]:\n        \"\"\"Evaluate model.\"\"\"\n        self.model.eval()\n        eval_loss = 0.0\n        eval_steps = 0\n        \n        with torch.no_grad():\n            for batch in eval_dataloader:\n                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v \n                        for k, v in batch.items()}\n                \n                outputs = self.model(**batch)\n                eval_loss += outputs.loss.item()\n                eval_steps += 1\n        \n        avg_eval_loss = eval_loss / eval_steps\n        \n        self.logger.info(f\"Evaluation: Loss={avg_eval_loss:.4f}\")\n        \n        # Wandb logging\n        if self.use_wandb:\n            wandb.log({\n                'eval_loss': avg_eval_loss,\n                'epoch': epoch\n            })\n        \n        return {'eval_loss': avg_eval_loss}\n    \n    def run_experiment(self) -> Dict[str, Any]:\n        \"\"\"Run complete LLaMA LoRA experiment.\"\"\"\n        self.logger.info(\"Starting LLaMA LoRA experiment\")\n        \n        # Initialize wandb\n        if self.use_wandb:\n            wandb.init(\n                project=\"mango-llama-lora\",\n                config=self.config,\n                name=f\"{self.optimizer_name}_r{self.config['lora_r']}\"\n            )\n        \n        # Start monitoring\n        if self.power_monitor:\n            self.power_monitor.start_monitoring()\n        \n        # Prepare datasets\n        train_dataloader, eval_dataloader = self.prepare_datasets()\n        \n        # Training loop\n        results = {\n            'model_name': self.model_name,\n            'optimizer': self.optimizer_name,\n            'config': self.config,\n            'epoch_results': [],\n            'final_metrics': {}\n        }\n        \n        for epoch in range(1, self.config['num_epochs'] + 1):\n            # Train\n            train_metrics = self.train_epoch(train_dataloader, epoch)\n            \n            # Evaluate\n            eval_metrics = self.evaluate(eval_dataloader, epoch)\n            \n            # Combine metrics\n            epoch_result = {**train_metrics, **eval_metrics, 'epoch': epoch}\n            results['epoch_results'].append(epoch_result)\n            \n            # Save best model\n            if eval_metrics['eval_loss'] < self.best_eval_loss:\n                self.best_eval_loss = eval_metrics['eval_loss']\n                self._save_model(f\"best_model_epoch_{epoch}\")\n            \n            # Save periodic checkpoint\n            if epoch % (self.config['num_epochs'] // 2) == 0:\n                self._save_model(f\"checkpoint_epoch_{epoch}\")\n        \n        # Final results\n        results['final_metrics'] = {\n            'best_eval_loss': self.best_eval_loss,\n            'final_train_loss': results['epoch_results'][-1]['loss'],\n            'final_eval_loss': results['epoch_results'][-1]['eval_loss'],\n            'total_epochs': self.config['num_epochs']\n        }\n        \n        # Memory and power statistics\n        if self.memory_profiler:\n            memory_report_path = self.memory_profiler.save_report('llama_lora_memory_report.json')\n            results['memory_report_path'] = memory_report_path\n        \n        if self.power_monitor:\n            power_stats = self.power_monitor.get_power_statistics()\n            results['power_statistics'] = power_stats\n            \n            # Export power data\n            power_csv_path = os.path.join(self.config['output_dir'], 'power_data.csv')\n            self.power_monitor.export_power_data(power_csv_path)\n            results['power_data_path'] = power_csv_path\n            \n            # Stop monitoring\n            self.power_monitor.stop_monitoring()\n        \n        # Compression statistics\n        if hasattr(self.optimizer, 'get_compression_stats'):\n            final_compression_stats = self.optimizer.get_compression_stats()\n            if isinstance(final_compression_stats, dict) and 'error' not in final_compression_stats:\n                results['final_compression_stats'] = final_compression_stats\n        \n        # Save results\n        results_path = os.path.join(self.config['output_dir'], 'experiment_results.json')\n        os.makedirs(os.path.dirname(results_path), exist_ok=True)\n        with open(results_path, 'w') as f:\n            json.dump(results, f, indent=2, default=str)\n        \n        # Finish wandb\n        if self.use_wandb:\n            wandb.finish()\n        \n        self.logger.info(f\"Experiment completed. Results saved to {results_path}\")\n        return results\n    \n    def _save_model(self, checkpoint_name: str):\n        \"\"\"Save model checkpoint.\"\"\"\n        save_path = os.path.join(self.config['output_dir'], checkpoint_name)\n        os.makedirs(save_path, exist_ok=True)\n        \n        # Save LoRA weights\n        self.model.save_pretrained(save_path)\n        \n        # Save tokenizer\n        self.tokenizer.save_pretrained(save_path)\n        \n        # Save optimizer state\n        optimizer_path = os.path.join(save_path, 'optimizer_state.pt')\n        torch.save(self.optimizer.state_dict(), optimizer_path)\n        \n        self.logger.info(f\"Model saved to {save_path}\")\n    \n    def _get_current_memory_usage(self) -> float:\n        \"\"\"Get current GPU memory usage in GB.\"\"\"\n        if torch.cuda.is_available():\n            return torch.cuda.memory_allocated() / (1024**3)\n        return 0.0\n    \n    def _get_current_lr(self) -> float:\n        \"\"\"Get current learning rate.\"\"\"\n        return self.optimizer.param_groups[0]['lr']\n\n\ndef main():\n    \"\"\"Main function for LLaMA LoRA experiments.\"\"\"\n    parser = argparse.ArgumentParser(description='LLaMA LoRA Fine-tuning with MANGO-LRQ')\n    \n    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',\n                       help='LLaMA model name')\n    parser.add_argument('--optimizer', type=str, default='enhanced_mango',\n                       choices=['enhanced_mango', 'adamw', 'adafactor'],\n                       help='Optimizer to use')\n    parser.add_argument('--dataset', type=str, default='alpaca',\n                       choices=['alpaca', 'dolly'],\n                       help='Dataset to use')\n    parser.add_argument('--lora-r', type=int, default=16,\n                       help='LoRA rank')\n    parser.add_argument('--lora-alpha', type=int, default=32,\n                       help='LoRA alpha')\n    parser.add_argument('--batch-size', type=int, default=4,\n                       help='Training batch size')\n    parser.add_argument('--learning-rate', type=float, default=2e-4,\n                       help='Learning rate')\n    parser.add_argument('--num-epochs', type=int, default=3,\n                       help='Number of training epochs')\n    parser.add_argument('--compression-rank', type=int, default=4,\n                       help='MANGO-LRQ compression rank')\n    parser.add_argument('--compression-bits', type=int, default=4,\n                       help='MANGO-LRQ compression bits')\n    parser.add_argument('--output-dir', type=str, default='./llama_lora_outputs',\n                       help='Output directory')\n    parser.add_argument('--disable-wandb', action='store_true',\n                       help='Disable Weights & Biases logging')\n    parser.add_argument('--enable-distributed', action='store_true',\n                       help='Enable distributed training')\n    parser.add_argument('--max-train-samples', type=int, default=10000,\n                       help='Maximum training samples')\n    \n    args = parser.parse_args()\n    \n    # Create configuration\n    config = {\n        # Model configuration\n        'max_seq_length': 512,\n        'load_in_4bit': True,\n        'bnb_4bit_compute_dtype': 'float16',\n        'bnb_4bit_use_double_quant': True,\n        'bnb_4bit_quant_type': 'nf4',\n        \n        # LoRA configuration\n        'lora_r': args.lora_r,\n        'lora_alpha': args.lora_alpha,\n        'lora_dropout': 0.1,\n        'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n        \n        # Training configuration\n        'learning_rate': args.learning_rate,\n        'num_epochs': args.num_epochs,\n        'batch_size': args.batch_size,\n        'gradient_accumulation_steps': 4,\n        'warmup_steps': 100,\n        'weight_decay': 0.01,\n        'max_grad_norm': 1.0,\n        \n        # MANGO-LRQ configuration\n        'compression_rank': args.compression_rank,\n        'compression_bits_p': args.compression_bits,\n        'compression_bits_q': max(args.compression_bits, 8),  # Q matrix needs more precision\n        'use_nf4_compression': True,\n        'enable_amp': True,\n        'enable_energy_monitoring': True,\n        \n        # Dataset configuration\n        'dataset_name': args.dataset,\n        'max_train_samples': args.max_train_samples,\n        'max_eval_samples': min(1000, args.max_train_samples // 10),\n        \n        # Output\n        'output_dir': args.output_dir,\n        'logging_steps': 50,\n        'eval_steps': 500,\n        'save_steps': 1000\n    }\n    \n    print(f\"Running LLaMA LoRA experiment with config: {config}\")\n    \n    # Initialize trainer\n    trainer = LLaMALoRATrainer(\n        model_name=args.model,\n        optimizer_name=args.optimizer,\n        config=config,\n        use_wandb=not args.disable_wandb,\n        enable_distributed=args.enable_distributed\n    )\n    \n    # Run experiment\n    results = trainer.run_experiment()\n    \n    print(\"\\nExperiment completed!\")\n    print(f\"Best eval loss: {results['final_metrics']['best_eval_loss']:.4f}\")\n    \n    if 'final_compression_stats' in results:\n        comp_stats = results['final_compression_stats']\n        print(f\"Final compression ratio: {comp_stats.get('avg_compression_ratio', 1.0):.2f}x\")\n        print(f\"Final compression error: {comp_stats.get('avg_compression_error', 0.0):.6f}\")\n    \n    if 'power_statistics' in results:\n        power_stats = results['power_statistics']\n        total_energy = power_stats['energy']['total_kwh']\n        print(f\"Total energy consumption: {total_energy:.4f} kWh\")\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    main()