"""
Experimental framework for MANGO optimizer evaluation.

Includes CIFAR-10 experiments, baseline comparisons, and evaluation metrics.
"""

from .cifar10_experiment import CIFAR10Experiment
from .baseline_optimizers import *
from .evaluation_framework import EvaluationFramework

__all__ = [
    'CIFAR10Experiment',
    'EvaluationFramework'
]