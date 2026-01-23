"""
Configuration file for English Fake News Detection
Supports separate BERT and RoBERTa training
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
MODEL_SAVE_DIR = BASE_DIR / "trained_models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"

# Create directories if they don't exist
for directory in [DATASET_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODEL_SAVE_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Language settings
LANGUAGE = "english"
MAX_LENGTH = 512  # Maximum sequence length for BERT/RoBERTa

# Model configurations
BERT_CONFIG = {
    "model_name": "bert-base-uncased",
    "model_type": "bert",
    "max_length": MAX_LENGTH,
    "batch_size": 16,  # Adjust based on your GPU memory
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,  # Set to 2-4 if GPU memory is limited
    "save_dir": MODEL_SAVE_DIR / "bert",
    "log_dir": LOGS_DIR / "bert",
    "num_labels": 2,  # Binary classification: fake (0) or real (1)
    "dropout": 0.1,
}

ROBERTA_CONFIG = {
    "model_name": "roberta-base",
    "model_type": "roberta",
    "max_length": MAX_LENGTH,
    "batch_size": 16,
    "learning_rate": 1e-5,  # RoBERTa often works better with slightly lower LR
    "epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "save_dir": MODEL_SAVE_DIR / "roberta",
    "log_dir": LOGS_DIR / "roberta",
    "num_labels": 2,
    "dropout": 0.1,
}

# Training settings
TRAIN_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
    "early_stopping_patience": 3,  # Stop if no improvement for 3 epochs
    "save_best_only": True,
    "mixed_precision": True,  # Use automatic mixed precision (AMP) if GPU supports it
    "num_workers": 4,  # For data loading
}

# Data preprocessing
PREPROCESSING_CONFIG = {
    "lowercase": True,  # BERT-uncased requires lowercase
    "remove_urls": True,
    "remove_special_chars": False,  # Keep for BERT tokenizer
    "remove_numbers": False,
    "min_length": 10,  # Minimum article length in words
    "max_samples_per_class": None,  # None for no limit, int for balanced sampling
}

# Dataset file names
DATA_FILES = {
    "raw_fake": RAW_DATA_DIR / "fake",
    "raw_real": RAW_DATA_DIR / "real",
    "train": PROCESSED_DATA_DIR / "train.csv",
    "val": PROCESSED_DATA_DIR / "val.csv",
    "test": PROCESSED_DATA_DIR / "test.csv",
    "full": PROCESSED_DATA_DIR / "full_dataset.csv",
}

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Class labels
LABEL_MAP = {
    0: "fake",
    1: "real"
}

REVERSE_LABEL_MAP = {
    "fake": 0,
    "real": 1
}

# Device configuration (will be determined at runtime)
DEVICE_CONFIG = {
    "use_cuda": True,  # Set to False to force CPU
    "cuda_device": 0,  # GPU device ID
}

# Checkpoint configuration
CHECKPOINT_CONFIG = {
    "save_every_n_epochs": 1,
    "keep_last_n": 3,  # Keep only last 3 checkpoints
    "save_optimizer_state": True,
}

# Ensemble configuration (for future use)
ENSEMBLE_CONFIG = {
    "models": ["bert", "roberta"],
    "weights": [0.5, 0.5],  # Equal weights initially
    "voting": "soft",  # 'soft' for probability averaging, 'hard' for majority vote
    "save_dir": MODEL_SAVE_DIR / "ensemble",
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "tensorboard": True,
    "wandb": False,  # Set to True if using Weights & Biases
}

# Memory optimization settings (useful for limited GPU memory)
MEMORY_CONFIG = {
    "gradient_checkpointing": False,  # Set to True to save memory at cost of speed
    "max_batch_size_finder": False,  # Auto-find max batch size
    "clear_cache_every_n_steps": 100,
}

def get_model_config(model_type: str) -> dict:
    """
    Get configuration for a specific model type.
    
    Args:
        model_type: Either 'bert' or 'roberta'
        
    Returns:
        Configuration dictionary for the specified model
    """
    if model_type.lower() == "bert":
        return BERT_CONFIG
    elif model_type.lower() == "roberta":
        return ROBERTA_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'bert' or 'roberta'")

def print_config(config_dict: dict, name: str = "Configuration"):
    """Pretty print configuration."""
    print(f"\n{'='*50}")
    print(f"{name:^50}")
    print(f"{'='*50}")
    for key, value in config_dict.items():
        print(f"{key:30} : {value}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Print all configurations when run directly
    print_config(BERT_CONFIG, "BERT Configuration")
    print_config(ROBERTA_CONFIG, "RoBERTa Configuration")
    print_config(TRAIN_CONFIG, "Training Configuration")
    print_config(PREPROCESSING_CONFIG, "Preprocessing Configuration")
