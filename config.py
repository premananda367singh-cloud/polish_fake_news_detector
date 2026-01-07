import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import AutoTokenizer

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    # Paths
    models_dir: str = "models/saved"
    data_dir: str = "data"
    
    # Model names
    bert_model_name: str = "dkleczek/bert-base-polish-uncased-v1"
    roberta_model_name: str = "sdadas/polish-roberta-base-v2"
    
    # Training parameters
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 5
    
    # Inference parameters
    threshold: float = 0.5
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'bert': 0.4,
                'roberta': 0.4,
                'tfidf': 0.2
            }

@dataclass
class DataConfig:
    """Configuration for data processing"""
    label_map: Dict[str, int] = None
    
    def __post_init__(self):
        if self.label_map is None:
            self.label_map = {
                'fake': 0,
                'real': 1
            }

# Initialize configurations
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()