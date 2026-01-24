"""
Optimized Configuration for BERT Training on Google Colab
Designed to work within Colab's free tier GPU limits (T4 with ~15GB RAM)
MINIMAL RESOURCE USAGE - Sequential Parameter Training
"""

import os
from pathlib import Path

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / 'datasets'
PROCESSED_DIR = DATASETS_DIR / 'processed'
RAW_DIR = DATASETS_DIR / 'raw'
MODELS_DIR = BASE_DIR / 'trained_models'
LOGS_DIR = BASE_DIR / 'logs'

# ============================================================================
# DATA FILES - Using your unified dataset
# ============================================================================
DATA_FILES = {
    'train': PROCESSED_DIR / 'unified_all.tsv',
    'val': PROCESSED_DIR / 'unified_all.tsv',  # We'll split this
    'test': PROCESSED_DIR / 'unified_all.tsv',  # We'll split this
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,
}

# ============================================================================
# TRAINING CONFIGURATION - OPTIMIZED FOR COLAB FREE TIER
# ============================================================================
TRAIN_CONFIG = {
    'early_stopping_patience': 3,  # Reduced from 5 to save time
    'random_seed': 42,
    'train_split': 0.7,  # 70% train
    'val_split': 0.15,   # 15% validation
    'test_split': 0.15,  # 15% test
}

# ============================================================================
# BERT CONFIGURATION - MINIMAL RESOURCE USAGE
# ============================================================================
BERT_CONFIG = {
    # Model settings
    'model_name': 'bert-base-uncased',  # Using base model (smaller than large)
    'num_labels': 2,
    
    # Training parameters - OPTIMIZED FOR LOW MEMORY
    'epochs': 3,  # Reduced from typical 5-10 epochs
    'batch_size': 8,  # VERY SMALL - Colab can handle this
    'max_length': 128,  # Reduced from 512 - HUGE memory savings
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,  # Reduced
    'gradient_accumulation_steps': 4,  # Simulate batch_size of 32
    
    # Paths
    'save_dir': MODELS_DIR / 'bert',
    'log_dir': LOGS_DIR / 'bert',
}

# ============================================================================
# SEQUENTIAL TRAINING CONFIGURATION
# This allows training model in segments to avoid memory issues
# ============================================================================
SEQUENTIAL_CONFIG = {
    'enabled': True,  # Set to False for normal training
    'freeze_layers': True,  # Freeze earlier layers when training later ones
    
    # Define training segments (which layers to train in each phase)
    'segments': [
        {
            'name': 'classifier_only',
            'epochs': 2,
            'layers_to_train': ['classifier'],  # Just the classification head
            'freeze_bert': True,
            'learning_rate': 3e-5,
        },
        {
            'name': 'top_layers',
            'epochs': 2,
            'layers_to_train': ['encoder.layer.10', 'encoder.layer.11', 'classifier'],
            'freeze_bert': False,
            'learning_rate': 2e-5,
        },
        {
            'name': 'all_layers',
            'epochs': 1,
            'layers_to_train': None,  # Train all
            'freeze_bert': False,
            'learning_rate': 1e-5,
        },
    ]
}

# ============================================================================
# MEMORY OPTIMIZATION SETTINGS
# ============================================================================
MEMORY_CONFIG = {
    'use_gradient_checkpointing': True,  # Trade compute for memory
    'use_mixed_precision': True,  # fp16 training
    'empty_cache_frequency': 50,  # Clear cache every N batches
    'max_grad_norm': 1.0,  # Gradient clipping
}

# ============================================================================
# DATA LOADING CONFIGURATION
# ============================================================================
DATA_LOADER_CONFIG = {
    'num_workers': 2,  # Reduced for Colab
    'pin_memory': True,
    'prefetch_factor': 2,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'log_every_n_steps': 50,  # Log less frequently
    'save_checkpoints': True,
    'checkpoint_frequency': 1,  # Save every epoch
}

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR, BERT_CONFIG['save_dir'], BERT_CONFIG['log_dir']]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Configuration loaded!")
print(f"BERT batch size: {BERT_CONFIG['batch_size']} (effective: {BERT_CONFIG['batch_size'] * BERT_CONFIG['gradient_accumulation_steps']})")
print(f"Max sequence length: {BERT_CONFIG['max_length']}")
print(f"Total epochs: {BERT_CONFIG['epochs']}")
print(f"Sequential training: {'Enabled' if SEQUENTIAL_CONFIG['enabled'] else 'Disabled'}")