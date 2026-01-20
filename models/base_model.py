from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import pickle
import torch
import torch.nn as nn

class BaseFakeNewsDetector(ABC):
    """Abstract base class for all fake news detectors"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, texts: list, labels: list, **kwargs):
        """Train the model on given texts and labels"""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict if text contains fake news"""
        pass
    
    @abstractmethod
    def predict_proba(self, text: str) -> float:
        """Return probability of being fake news"""
        pass
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'is_trained': self.is_trained
            }, path)
    
    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        return self
    
    def predict_batch(self, texts: list) -> list:
        """Predict for multiple texts"""
        return [self.predict(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'is_trained': self.is_trained,
            'parameters': sum(p.numel() for p in self.model.parameters()) 
            if self.model else 0
        }