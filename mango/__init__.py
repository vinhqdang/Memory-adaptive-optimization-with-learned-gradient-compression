"""
MANGO: Memory-Adaptive Neural Gradient Optimizer with Low-Rank Quantization

A novel optimization algorithm that dynamically adjusts memory allocation 
between gradient precision and optimizer states based on training dynamics.

Features MANGO-LRQ v4.0 with:
- Hybrid low-rank + quantized compression  
- TinyFormer policy network
- 8-bit optimizer states
- Multi-objective energy-aware optimization
- Distributed training support
"""

# Core optimizer components
try:
    from .optimizer import MANGO
except ImportError:
    pass

try:
    from .enhanced_optimizer import EnhancedMANGO
except ImportError:
    pass

# Compression and quantization
try:
    from .compression import GradientCompressor
except ImportError:
    pass

try:
    from .mango_lrq import MangoLRQCompressor, CompressionConfig
except ImportError:
    pass

# Policy networks
try:
    from .policy_network import CompressionPolicyNet, AdaptiveCompressionPolicy
except ImportError:
    pass

try:
    from .tinyformer_policy import TinyFormerPolicyNet
except ImportError:
    pass

# Statistics and monitoring
try:
    from .statistics import GradientStatistics
except ImportError:
    pass

try:
    from .memory_profiler import create_memory_profiler
except ImportError:
    pass

try:
    from .power_monitor import PowerMonitor, MultiObjectiveReward
except ImportError:
    pass

# Utilities
try:
    from .utils import MemoryProfiler
except ImportError:
    pass

try:
    from .ef21_buffer import EF21Buffer
except ImportError:
    pass

__version__ = "4.0.0"
__all__ = [
    "MANGO",
    "EnhancedMANGO",
    "GradientCompressor", 
    "MangoLRQCompressor",
    "CompressionConfig",
    "CompressionPolicyNet",
    "AdaptiveCompressionPolicy",
    "TinyFormerPolicyNet", 
    "GradientStatistics",
    "MemoryProfiler",
    "create_memory_profiler",
    "PowerMonitor",
    "MultiObjectiveReward",
    "EF21Buffer"
]