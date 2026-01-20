from typing import Dict, Any, List
import numpy as np
from .base_model import BaseFakeNewsDetector
from .bert_detector import BertDetector
from .roberta_detector import RobertaDetector
from .tfidf_detector import TfidfDetector
from config import MODEL_CONFIG

class EnsembleDetector(BaseFakeNewsDetector):
    """Ensemble model combining multiple detectors"""
    
    def __init__(self, models: List[BaseFakeNewsDetector] = None, weights: Dict[str, float] = None):
        super().__init__("ensemble_detector")
        
        if models is None:
            # Initialize default models
            self.models = {
                'bert': BertDetector(),
                'roberta': RobertaDetector(),
                'tfidf': TfidfDetector()
            }
        else:
            self.models = {model.model_name: model for model in models}
        
        # Set weights
        if weights is None:
            self.weights = MODEL_CONFIG.ensemble_weights
        else:
            self.weights = weights
            
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make ensemble prediction"""
        predictions = []
        probabilities = []
        confidences = []
        model_results = {}
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            if model.is_trained:
                result = model.predict(text)
                model_results[model_name] = result
                
                predictions.append(result['prediction'])
                probabilities.append(result['probabilities'])
                confidences.append(result['confidence'])
            else:
                print(f"Warning: Model {model_name} is not trained")
        
        if not model_results:
            raise ValueError("No trained models available in ensemble")
        
        # Weighted voting for final prediction
        weighted_votes = {'fake': 0.0, 'real': 0.0}
        
        for model_name, result in model_results.items():
            weight = self.weights.get(model_name, 1.0/len(model_results))
            weighted_votes['fake'] += result['probabilities']['fake'] * weight
            weighted_votes['real'] += result['probabilities']['real'] * weight
        
        # Final decision
        fake_prob = weighted_votes['fake']
        real_prob = weighted_votes['real']
        
        prediction = 1 if real_prob > MODEL_CONFIG.threshold else 0
        confidence = max(fake_prob, real_prob)
        
        # Get disagreement measure
        disagreement = self._calculate_disagreement(predictions)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'fake': float(fake_prob),
                'real': float(real_prob)
            },
            'model': 'ensemble',
            'model_results': model_results,
            'disagreement': disagreement,
            'weights': self.weights
        }
    
    def _calculate_disagreement(self, predictions: List[int]) -> float:
        """Calculate disagreement between models"""
        if not predictions:
            return 0.0
        
        # Variance of predictions
        return np.var(predictions) if len(predictions) > 1 else 0.0
    
    def predict_proba(self, text: str) -> float:
        """Return probability of being fake news"""
        result = self.predict(text)
        return result['probabilities']['fake']
    
    def train(self, texts: list, labels: list, **kwargs):
        """Train all models in the ensemble"""
        print("Training ensemble models...")
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.train(texts, labels, **kwargs)
        
        self.is_trained = all(model.is_trained for model in self.models.values())
    
    def load_all_models(self, model_paths: Dict[str, str]):
        """Load pre-trained models from paths"""
        for model_name, path in model_paths.items():
            if model_name in self.models:
                self.models[model_name].load(path)
        
        self.is_trained = all(model.is_trained for model in self.models.values())
    
    def get_model_performance(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Evaluate performance of each model in ensemble"""
        from sklearn.metrics import accuracy_score, f1_score
        
        results = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                preds = []
                for text in texts:
                    result = model.predict(text)
                    preds.append(result['prediction'])
                
                results[model_name] = {
                    'accuracy': accuracy_score(labels, preds),
                    'f1_score': f1_score(labels, preds, average='weighted')
                }
        
        return results