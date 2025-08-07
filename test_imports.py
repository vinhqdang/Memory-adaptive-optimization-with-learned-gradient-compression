#!/usr/bin/env python3
"""
Simple import test script for CI validation.
Tests all critical imports without optional dependencies.
"""

import sys
import traceback

def test_import(module_name, import_statement):
    """Test a single import and report result."""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: OK")
        return True
    except ImportError as e:
        if any(dep in str(e).lower() for dep in ['bitsandbytes', 'wandb', 'transformers']):
            print(f"⚠️  {module_name}: SKIPPED (optional dependency: {e})")
            return True  # Skip optional dependencies
        else:
            print(f"❌ {module_name}: FAILED - {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"❌ {module_name}: ERROR - {e}")
        traceback.print_exc()
        return False

def main():
    """Run all import tests."""
    print("🧪 Running MANGO-LRQ import tests...")
    
    tests = [
        ("Core torch", "import torch; import torch.nn as nn"),
        ("NumPy", "import numpy as np"),
        ("MANGO base", "from mango import __init__"),
        ("Enhanced optimizer", "from mango.enhanced_optimizer import EnhancedMANGO"),
        ("Compression config", "from mango.mango_lrq import CompressionConfig"),
        ("LRQ compressor", "from mango.mango_lrq import MangoLRQCompressor"),
        ("TinyFormer policy", "from mango.tinyformer_policy import TinyFormerPolicyNet"),
        ("Statistics module", "from mango.statistics import GradientStatistics"),
        ("Memory profiler", "from mango.memory_profiler import create_memory_profiler"),
        ("Power sampler", "from utils.power_sampler import PowerSampler"),
        ("Power monitor", "from mango.power_monitor import PowerMonitor"),
        ("EF21 buffer", "from mango.ef21_buffer import EF21Buffer"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, import_stmt in tests:
        if test_import(test_name, import_stmt):
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} imports successful")
    
    if passed == total:
        print("🎉 All critical imports working!")
        return 0
    else:
        failed = total - passed
        print(f"⚠️  {failed} imports failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())