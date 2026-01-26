#!/usr/bin/env python3
"""
Setup script to create all required directories for the fake news detector
Run this before training to ensure all directories exist
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config_english import (
        DATASET_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODEL_SAVE_DIR, LOGS_DIR, CACHE_DIR,
        BERT_CONFIG, ROBERTA_CONFIG
    )
except ImportError:
    print("Error: config_english.py not found!")
    sys.exit(1)

def create_directories():
    """Create all required directories."""
    
    directories = [
        DATASET_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_SAVE_DIR,
        LOGS_DIR,
        CACHE_DIR,
        BERT_CONFIG['save_dir'],
        BERT_CONFIG['log_dir'],
        ROBERTA_CONFIG['save_dir'],
        ROBERTA_CONFIG['log_dir'],
    ]
    
    print("Creating required directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")
    
    print("\n✅ All directories created successfully!")
    print("\nYou can now run:")
    print("  - python train_bert.py (or train_bert_fixed.py)")
    print("  - python train_roberta.py")

if __name__ == "__main__":
    create_directories()
