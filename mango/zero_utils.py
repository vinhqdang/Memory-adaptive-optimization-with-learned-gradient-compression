"""
ZeRO-3 and FSDP Integration for MANGO Optimizer

Provides distributed training capabilities with intelligent parameter sharding
based on MANGO compression policies. Integrates with PyTorch FSDP and
DeepSpeed ZeRO for large-scale model training.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from collections import defaultdict
import functools

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("Warning: DeepSpeed not available, ZeRO-3 features disabled")


class MANGOShardingPolicy:
    """
    Intelligent sharding policy based on MANGO compression decisions.
    
    Determines which parameters should be sharded based on:
    - Compression ratios predicted by the policy network
    - Parameter size and communication costs
    - Memory constraints and bandwidth limitations
    """
    
    def __init__(
        self,
        compression_config: Dict[str, Any],
        min_shard_size: int = 1e6,  # Minimum parameters to shard
        communication_budget: float = 0.1,  # Fraction of time for communication
        memory_efficiency_threshold: float = 0.3  # Minimum memory savings to shard
    ):
        """
        Initialize sharding policy.
        
        Args:
            compression_config: MANGO compression configuration
            min_shard_size: Minimum parameter count to consider sharding
            communication_budget: Maximum communication overhead fraction
            memory_efficiency_threshold: Minimum memory efficiency to enable sharding
        """
        self.compression_config = compression_config
        self.min_shard_size = min_shard_size
        self.communication_budget = communication_budget
        self.memory_efficiency_threshold = memory_efficiency_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Sharding statistics
        self.sharding_stats = {
            'total_parameters': 0,
            'sharded_parameters': 0,
            'compression_enabled_parameters': 0,
            'communication_savings': 0.0
        }
    
    def should_shard_module(
        self, 
        module: nn.Module, 
        module_name: str = ""
    ) -> bool:
        """
        Determine if a module should be sharded.
        
        Args:
            module: PyTorch module to evaluate
            module_name: Name/path of the module
            
        Returns:
            True if module should be sharded
        """
        # Count parameters in module
        param_count = sum(p.numel() for p in module.parameters())
        
        # Don't shard small modules
        if param_count < self.min_shard_size:
            return False
        
        # Estimate compression ratio for this module type
        compression_ratio = self._estimate_compression_ratio(module, module_name)
        
        # Calculate memory efficiency
        memory_efficiency = 1.0 - (1.0 / compression_ratio)
        
        # Shard if compression provides sufficient memory savings
        should_shard = memory_efficiency > self.memory_efficiency_threshold
        
        # Update statistics
        self.sharding_stats['total_parameters'] += param_count
        if should_shard:
            self.sharding_stats['sharded_parameters'] += param_count
            if compression_ratio > 1.1:  # Significant compression
                self.sharding_stats['compression_enabled_parameters'] += param_count
        
        self.logger.debug(f"Module {module_name}: {param_count:,} params, "
                         f"compression={compression_ratio:.2f}x, shard={should_shard}")
        
        return should_shard
    
    def _estimate_compression_ratio(
        self, 
        module: nn.Module, 
        module_name: str
    ) -> float:
        """
        Estimate compression ratio for a specific module.
        
        Args:
            module: PyTorch module
            module_name: Module name/path
            
        Returns:
            Estimated compression ratio
        """
        # Base compression from MANGO configuration
        base_ratio = 1.0
        
        if 'rank' in self.compression_config:
            # Low-rank compression ratio estimation
            rank = self.compression_config['rank']
            
            # Estimate based on module type and typical shapes
            if isinstance(module, nn.Linear):
                # For linear layers, compression depends on input/output dims
                in_features = getattr(module, 'in_features', 512)
                out_features = getattr(module, 'out_features', 512)
                
                # Compression ratio for low-rank approximation
                original_params = in_features * out_features
                compressed_params = rank * (in_features + out_features)
                base_ratio = original_params / max(compressed_params, 1)
                
            elif isinstance(module, nn.Conv2d):
                # For conv layers, compression is more limited
                base_ratio = 1.5  # Conservative estimate
                
            elif 'attention' in module_name.lower() or 'transformer' in module_name.lower():\n                # Attention modules compress well with low-rank\n                base_ratio = 3.0  # Aggressive compression for attention\n        \n        # Additional compression from quantization\n        if 'bits_p' in self.compression_config:\n            bits_p = self.compression_config['bits_p']\n            quant_ratio = 32.0 / max(bits_p, 4)  # Quantization compression\n            base_ratio *= quant_ratio\n        \n        return max(1.0, base_ratio)\n    \n    def get_sharding_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive sharding statistics.\"\"\"\n        total_params = self.sharding_stats['total_parameters']\n        sharded_params = self.sharding_stats['sharded_parameters']\n        \n        return {\n            'total_parameters': total_params,\n            'sharded_parameters': sharded_params,\n            'sharding_ratio': sharded_params / max(total_params, 1),\n            'compression_enabled_parameters': self.sharding_stats['compression_enabled_parameters'],\n            'estimated_memory_savings': (sharded_params / max(total_params, 1)) * 0.7,  # Estimate\n            'communication_overhead': self.sharding_stats.get('communication_overhead', 0.0)\n        }\n\n\nclass MANGOFSDPWrapper:\n    \"\"\"\n    FSDP wrapper with MANGO-aware sharding and compression.\n    \n    Integrates MANGO optimizer with PyTorch Fully Sharded Data Parallel\n    for efficient distributed training.\n    \"\"\"\n    \n    def __init__(\n        self,\n        model: nn.Module,\n        mango_optimizer,\n        auto_wrap_policy: Optional[Callable] = None,\n        sharding_strategy: str = \"full_shard\",\n        backward_prefetch: str = \"backward_pre\",\n        mixed_precision: bool = True\n    ):\n        \"\"\"\n        Initialize MANGO FSDP wrapper.\n        \n        Args:\n            model: PyTorch model to wrap\n            mango_optimizer: MANGO optimizer instance\n            auto_wrap_policy: FSDP auto-wrapping policy\n            sharding_strategy: FSDP sharding strategy\n            backward_prefetch: Backward prefetch strategy\n            mixed_precision: Enable mixed precision training\n        \"\"\"\n        self.base_model = model\n        self.mango_optimizer = mango_optimizer\n        self.mixed_precision = mixed_precision\n        \n        # Create MANGO-aware sharding policy\n        compression_config = getattr(mango_optimizer, 'compression_config', {})\n        if hasattr(compression_config, '__dict__'):\n            compression_config = compression_config.__dict__\n        \n        self.sharding_policy = MANGOShardingPolicy(compression_config)\n        \n        # Setup FSDP configuration\n        if auto_wrap_policy is None:\n            auto_wrap_policy = functools.partial(\n                transformer_auto_wrap_policy,\n                transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}\n            )\n        \n        # Mixed precision setup\n        if mixed_precision:\n            from torch.distributed.fsdp.mixed_precision import MixedPrecision\n            mp_policy = MixedPrecision(\n                param_dtype=torch.float16,\n                reduce_dtype=torch.float16,\n                buffer_dtype=torch.float16\n            )\n        else:\n            mp_policy = None\n        \n        # Wrap model with FSDP\n        self.model = FSDP(\n            model,\n            auto_wrap_policy=auto_wrap_policy,\n            mixed_precision=mp_policy,\n            backward_prefetch=backward_prefetch,\n            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None\n        )\n        \n        # Setup gradient scaler for mixed precision\n        if mixed_precision:\n            self.grad_scaler = ShardedGradScaler()\n        else:\n            self.grad_scaler = None\n        \n        self.logger = logging.getLogger(__name__)\n        self.logger.info(\"MANGO FSDP wrapper initialized\")\n    \n    def setup_compression_hooks(self):\n        \"\"\"\n        Setup compression hooks that work with FSDP parameter lifecycle.\n        \n        FSDP manages parameter sharding, so we need to hook into the\n        appropriate points for compression.\n        \"\"\"\n        def compression_hook(module, grad_input, grad_output):\n            \"\"\"Hook to apply MANGO compression after gradient computation.\"\"\"\n            if hasattr(module, '_fsdp_wrapped_module'):\n                # Apply compression to FSDP-wrapped parameters\n                wrapped_module = module._fsdp_wrapped_module\n                \n                for param in wrapped_module.parameters():\n                    if param.grad is not None:\n                        param_id = id(param)\n                        \n                        # Get compression config from MANGO optimizer\n                        compression_config = self.mango_optimizer.get_compression_config()\n                        \n                        # Apply compression\n                        compressed_grad, metadata = self.mango_optimizer._compress_gradient(\n                            param, param_id, compression_config\n                        )\n                        \n                        # Replace gradient with compressed version\n                        param.grad.data = compressed_grad\n        \n        # Register hooks on FSDP modules\n        for module in self.model.modules():\n            if hasattr(module, '_fsdp_wrapped_module'):\n                module.register_backward_hook(compression_hook)\n    \n    def train_step(\n        self,\n        inputs: torch.Tensor,\n        targets: torch.Tensor,\n        criterion: nn.Module\n    ) -> Dict[str, float]:\n        \"\"\"\n        Perform one training step with FSDP + MANGO compression.\n        \n        Args:\n            inputs: Input batch\n            targets: Target batch\n            criterion: Loss function\n            \n        Returns:\n            Dictionary of training metrics\n        \"\"\"\n        self.model.train()\n        \n        # Forward pass\n        if self.grad_scaler:\n            with torch.cuda.amp.autocast():\n                outputs = self.model(inputs)\n                loss = criterion(outputs, targets)\n            \n            # Scaled backward pass\n            self.grad_scaler.scale(loss).backward()\n            \n            # Optimizer step with gradient scaling\n            self.grad_scaler.step(self.mango_optimizer)\n            self.grad_scaler.update()\n        else:\n            outputs = self.model(inputs)\n            loss = criterion(outputs, targets)\n            \n            loss.backward()\n            self.mango_optimizer.step()\n        \n        self.mango_optimizer.zero_grad()\n        \n        # Collect metrics\n        metrics = {\n            'loss': loss.item(),\n            'batch_size': inputs.size(0)\n        }\n        \n        # Add MANGO compression stats\n        if hasattr(self.mango_optimizer, 'get_compression_stats'):\n            compression_stats = self.mango_optimizer.get_compression_stats()\n            if isinstance(compression_stats, dict) and 'error' not in compression_stats:\n                metrics.update({\n                    'compression_ratio': compression_stats.get('avg_compression_ratio', 1.0),\n                    'compression_error': compression_stats.get('avg_compression_error', 0.0)\n                })\n        \n        return metrics\n    \n    def get_model_state_dict(self) -> Dict[str, torch.Tensor]:\n        \"\"\"Get consolidated model state dict from FSDP shards.\"\"\"\n        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):\n            return self.model.state_dict()\n    \n    def load_model_state_dict(self, state_dict: Dict[str, torch.Tensor]):\n        \"\"\"Load model state dict into FSDP shards.\"\"\"\n        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):\n            self.model.load_state_dict(state_dict)\n    \n    def get_memory_stats(self) -> Dict[str, float]:\n        \"\"\"Get memory usage statistics.\"\"\"\n        stats = {}\n        \n        if torch.cuda.is_available():\n            stats.update({\n                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),\n                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),\n                'gpu_max_memory_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)\n            })\n        \n        # Add MANGO memory stats\n        if hasattr(self.mango_optimizer, 'get_memory_usage'):\n            mango_stats = self.mango_optimizer.get_memory_usage()\n            stats.update({f'mango_{k}': v for k, v in mango_stats.items()})\n        \n        # Add sharding stats\n        sharding_stats = self.sharding_policy.get_sharding_statistics()\n        stats.update({f'sharding_{k}': v for k, v in sharding_stats.items()})\n        \n        return stats\n\n\nclass MANGODeepSpeedEngine:\n    \"\"\"\n    DeepSpeed integration with MANGO optimizer for ZeRO-3 support.\n    \n    Provides large-scale distributed training capabilities with\n    MANGO compression and DeepSpeed ZeRO optimizations.\n    \"\"\"\n    \n    def __init__(\n        self,\n        model: nn.Module,\n        mango_optimizer,\n        deepspeed_config: Dict[str, Any]\n    ):\n        \"\"\"\n        Initialize MANGO DeepSpeed engine.\n        \n        Args:\n            model: PyTorch model\n            mango_optimizer: MANGO optimizer instance  \n            deepspeed_config: DeepSpeed configuration dictionary\n        \"\"\"\n        if not HAS_DEEPSPEED:\n            raise ImportError(\"DeepSpeed not available. Install with: pip install deepspeed\")\n        \n        self.base_model = model\n        self.mango_optimizer = mango_optimizer\n        self.deepspeed_config = deepspeed_config\n        \n        # Initialize DeepSpeed engine\n        self.engine, self.optimizer, self.training_dataloader, self.lr_scheduler = \\\n            deepspeed.initialize(\n                model=model,\n                config=deepspeed_config,\n                optimizer=mango_optimizer\n            )\n        \n        self.logger = logging.getLogger(__name__)\n        self.logger.info(\"MANGO DeepSpeed engine initialized\")\n    \n    def train_step(\n        self,\n        inputs: torch.Tensor,\n        targets: torch.Tensor,\n        criterion: nn.Module\n    ) -> Dict[str, float]:\n        \"\"\"\n        Perform training step with DeepSpeed + MANGO.\n        \n        Args:\n            inputs: Input batch\n            targets: Target batch \n            criterion: Loss function\n            \n        Returns:\n            Training metrics dictionary\n        \"\"\"\n        # Forward pass\n        outputs = self.engine(inputs)\n        loss = criterion(outputs, targets)\n        \n        # Backward pass with DeepSpeed\n        self.engine.backward(loss)\n        \n        # Optimizer step\n        self.engine.step()\n        \n        # Collect metrics\n        metrics = {\n            'loss': loss.item(),\n            'batch_size': inputs.size(0)\n        }\n        \n        # Add MANGO compression statistics\n        if hasattr(self.mango_optimizer, 'get_compression_stats'):\n            compression_stats = self.mango_optimizer.get_compression_stats()\n            if isinstance(compression_stats, dict) and 'error' not in compression_stats:\n                metrics.update({\n                    'compression_ratio': compression_stats.get('avg_compression_ratio', 1.0),\n                    'compression_error': compression_stats.get('avg_compression_error', 0.0)\n                })\n        \n        return metrics\n    \n    def save_checkpoint(self, save_dir: str, tag: Optional[str] = None):\n        \"\"\"Save DeepSpeed checkpoint with MANGO state.\"\"\"\n        # Save DeepSpeed checkpoint\n        self.engine.save_checkpoint(save_dir, tag)\n        \n        # Save MANGO optimizer state separately\n        if hasattr(self.mango_optimizer, 'state_dict'):\n            mango_state_path = f\"{save_dir}/mango_optimizer_state_{tag or 'latest'}.pt\"\n            torch.save(self.mango_optimizer.state_dict(), mango_state_path)\n            self.logger.info(f\"MANGO state saved to {mango_state_path}\")\n    \n    def load_checkpoint(self, load_dir: str, tag: Optional[str] = None):\n        \"\"\"Load DeepSpeed checkpoint with MANGO state.\"\"\"\n        # Load DeepSpeed checkpoint\n        _, client_state = self.engine.load_checkpoint(load_dir, tag)\n        \n        # Load MANGO optimizer state\n        try:\n            mango_state_path = f\"{load_dir}/mango_optimizer_state_{tag or 'latest'}.pt\"\n            mango_state = torch.load(mango_state_path, map_location='cpu')\n            self.mango_optimizer.load_state_dict(mango_state)\n            self.logger.info(f\"MANGO state loaded from {mango_state_path}\")\n        except FileNotFoundError:\n            self.logger.warning(\"MANGO optimizer state not found, using default initialization\")\n        \n        return client_state\n    \n    def get_memory_stats(self) -> Dict[str, float]:\n        \"\"\"Get comprehensive memory statistics.\"\"\"\n        stats = {}\n        \n        # DeepSpeed memory stats\n        if hasattr(self.engine, 'memory_breakdown'):\n            memory_breakdown = self.engine.memory_breakdown()\n            stats.update(memory_breakdown)\n        \n        # MANGO memory stats\n        if hasattr(self.mango_optimizer, 'get_memory_usage'):\n            mango_stats = self.mango_optimizer.get_memory_usage()\n            stats.update({f'mango_{k}': v for k, v in mango_stats.items()})\n        \n        return stats\n\n\ndef create_mango_distributed_trainer(\n    model: nn.Module,\n    mango_optimizer,\n    training_config: Dict[str, Any]\n) -> Union[MANGOFSDPWrapper, MANGODeepSpeedEngine]:\n    \"\"\"\n    Factory function to create distributed trainer with MANGO optimization.\n    \n    Args:\n        model: PyTorch model\n        mango_optimizer: MANGO optimizer instance\n        training_config: Training configuration\n        \n    Returns:\n        Configured distributed trainer\n    \"\"\"\n    distributed_backend = training_config.get('distributed_backend', 'fsdp')\n    \n    if distributed_backend.lower() == 'deepspeed':\n        if not HAS_DEEPSPEED:\n            raise ImportError(\"DeepSpeed not available. Install with: pip install deepspeed\")\n        \n        deepspeed_config = training_config.get('deepspeed_config', {})\n        return MANGODeepSpeedEngine(model, mango_optimizer, deepspeed_config)\n    \n    elif distributed_backend.lower() == 'fsdp':\n        fsdp_config = training_config.get('fsdp_config', {})\n        return MANGOFSDPWrapper(\n            model, \n            mango_optimizer,\n            **fsdp_config\n        )\n    \n    else:\n        raise ValueError(f\"Unknown distributed backend: {distributed_backend}\")\n\n\ndef get_default_fsdp_config() -> Dict[str, Any]:\n    \"\"\"Get default FSDP configuration optimized for MANGO.\"\"\"\n    return {\n        'sharding_strategy': 'full_shard',\n        'backward_prefetch': 'backward_pre',\n        'mixed_precision': True,\n        'auto_wrap_policy': None  # Will use transformer auto-wrap\n    }\n\n\ndef get_default_deepspeed_config() -> Dict[str, Any]:\n    \"\"\"Get default DeepSpeed configuration optimized for MANGO.\"\"\"\n    return {\n        \"train_batch_size\": 32,\n        \"gradient_accumulation_steps\": 1,\n        \"fp16\": {\n            \"enabled\": True,\n            \"loss_scale\": 0,\n            \"loss_scale_window\": 1000,\n            \"hysteresis\": 2,\n            \"min_loss_scale\": 1\n        },\n        \"zero_optimization\": {\n            \"stage\": 3,\n            \"offload_optimizer\": {\n                \"device\": \"cpu\",\n                \"pin_memory\": True\n            },\n            \"offload_param\": {\n                \"device\": \"cpu\",\n                \"pin_memory\": True\n            },\n            \"overlap_comm\": True,\n            \"contiguous_gradients\": True,\n            \"sub_group_size\": 1e9,\n            \"reduce_bucket_size\": 5e8,\n            \"stage3_prefetch_bucket_size\": 5e8,\n            \"stage3_param_persistence_threshold\": 1e6,\n            \"stage3_max_live_parameters\": 1e9,\n            \"stage3_max_reuse_distance\": 1e9,\n            \"gather_16bit_weights_on_model_save\": True\n        },\n        \"gradient_clipping\": 1.0,\n        \"wall_clock_breakdown\": False\n    }