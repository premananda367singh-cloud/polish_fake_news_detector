from typing import Dict, Any, List
import numpy as np
from models.ensemble_detector import EnsembleDetector
from utils.credibility_scorer import CredibilityScorer
import time
from datetime import datetime

class InferenceService:
    """Main service for fake news detection inference"""
    
    def __init__(self, ensemble_model: EnsembleDetector = None):
        if ensemble_model is None:
            self.ensemble = EnsembleDetector()
        else:
            self.ensemble = ensemble_model
        
        self.credibility_scorer = CredibilityScorer()
        self.inference_history = []
    
    def analyze_news(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive news analysis
        
        Args:
            text: News text to analyze
            metadata: Additional metadata (source, author, date, etc.)
        
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        # Get ensemble prediction
        prediction_result = self.ensemble.predict(text)
        
        # Calculate credibility score if metadata provided
        credibility_score = None
        if metadata:
            credibility_score = self.credibility_scorer.calculate_score(metadata)
        
        # Prepare response
        result = {
            'text': text[:500] + '...' if len(text) > 500 else text,  # Truncate for response
            'prediction': 'real' if prediction_result['prediction'] == 1 else 'fake',
            'confidence': prediction_result['confidence'],
            'probability_fake': prediction_result['probabilities']['fake'],
            'probability_real': prediction_result['probabilities']['real'],
            'timestamp': datetime.now().isoformat(),
            'inference_time': time.time() - start_time,
            'model_details': {
                'type': 'ensemble',
                'disagreement': prediction_result.get('disagreement', 0.0)
            }
        }
        
        # Add credibility information
        if credibility_score:
            result['credibility'] = {
                'score': credibility_score['score'],
                'factors': credibility_score['factors'],
                'level': credibility_score['level']
            }
            
            # Adjust confidence based on credibility
            if credibility_score['score'] < 0.3:  # Low credibility source
                result['adjusted_confidence'] = prediction_result['confidence'] * 0.8
            elif credibility_score['score'] > 0.7:  # High credibility source
                result['adjusted_confidence'] = min(prediction_result['confidence'] * 1.2, 1.0)
        
        # Add individual model predictions if available
        if 'model_results' in prediction_result:
            result['model_predictions'] = {
                model_name: {
                    'prediction': 'real' if res['prediction'] == 1 else 'fake',
                    'confidence': res['confidence']
                }
                for model_name, res in prediction_result['model_results'].items()
            }
        
        # Store in history
        self.inference_history.append({
            'timestamp': result['timestamp'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'text_length': len(text)
        })
        
        # Keep only last 1000 inferences in memory
        if len(self.inference_history) > 1000:
            self.inference_history = self.inference_history[-1000:]
        
        return result
    
    def analyze_batch(self, texts: List[str], metadata_list: List[Dict] = None) -> List[Dict[str, Any]]:
        """Analyze multiple news items"""
        results = []
        
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            result = self.analyze_news(text, metadata)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        if not self.inference_history:
            return {}
        
        predictions = [item['prediction'] for item in self.inference_history]
        
        return {
            'total_inferences': len(self.inference_history),
            'fake_count': predictions.count('fake'),
            'real_count': predictions.count('real'),
            'avg_confidence': np.mean([item['confidence'] for item in self.inference_history]),
            'avg_text_length': np.mean([item['text_length'] for item in self.inference_history])
        }
    
    def load_models(self, model_paths: Dict[str, str]):
        """Load pre-trained models"""
        self.ensemble.load_all_models(model_paths)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models in ensemble"""
        status = {}
        
        for model_name, model in self.ensemble.models.items():
            status[model_name] = {
                'is_trained': model.is_trained,
                'model_info': model.get_model_info()
            }
        
        return status 