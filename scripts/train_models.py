#!/usr/bin/env python3
"""
Convenient script to run GPU-accelerated model training.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.train_all import main

if __name__ == "__main__":
    # Run the main training function
    exit(main()) 