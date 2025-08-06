"""
MANGO: Memory-Adaptive Neural Gradient Optimizer

A novel optimization algorithm that dynamically adjusts memory allocation 
between gradient precision and optimizer states based on training dynamics.
"""

from .optimizer import MANGO
from .compression import GradientCompressor
from .policy_network import CompressionPolicyNet, AdaptiveCompressionPolicy
from .statistics import GradientStatistics
from .utils import MemoryProfiler

__version__ = "0.1.0"
__all__ = [
    "MANGO",
    "GradientCompressor", 
    "CompressionPolicyNet",
    "AdaptiveCompressionPolicy",
    "GradientStatistics",
    "MemoryProfiler"
]