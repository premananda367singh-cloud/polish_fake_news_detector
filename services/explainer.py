import shap
import lime
import lime.lime_text
from typing import Dict, Any, List
import numpy as np
from models.tfidf_detector import TfidfDetector
import torch
from transformers import AutoTokenizer

class ExplanationService:
    """Service for generating explanations for predictions"""
    
    def __init__(self, models: Dict[str, Any]):
        """
        Args:
            models: Dictionary of trained models
        """
        self.models = models
        self.explainer_lime = lime.lime_text.LimeTextExplainer(
            class_names=['fake', 'real']
        )
    
    def explain_with_shap(self, text: str, model_name: str = 'tfidf', top_k: int = 20) -> Dict[str, Any]:
        """
        Generate SHAP explanation for prediction
        
        Args:
            text: Input text
            model_name: Name of model to explain
            top_k: Number of top features to return
        
        Returns:
            Explanation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'tfidf':
            return self._explain_tfidf_shap(model, text, top_k)
        elif model_name in ['bert', 'roberta']:
            return self._explain_transformer_shap(model, text, top_k)
        else:
            raise ValueError(f"SHAP explanation not implemented for {model_name}")
    
    def _explain_tfidf_shap(self, model: TfidfDetector, text: str, top_k: int) -> Dict[str, Any]:
        """Generate SHAP explanation for TF-IDF model"""
        # Create SHAP explainer
        vectorizer = model.model.named_steps['tfidf']
        classifier = model.model.named_steps['clf']
        
        def predict_proba_wrapper(texts):
            return model.model.predict_proba(texts)
        
        # Create SHAP explainer
        explainer = shap.Explainer(
            predict_proba_wrapper,
            masker=vectorizer,
            output_names=['fake', 'real']
        )
        
        # Generate explanation
        shap_values = explainer([text])
        
        # Extract feature importance
        feature_importance = []
        for i in range(len(shap_values[0].values)):
            feature_importance.append({
                'feature': shap_values[0].feature_names[i],
                'value': float(shap_values[0].values[i][0]),  # Fake class
                'impact': float(shap_values[0].values[i][0])  # Impact on fake probability
            })
        
        # Sort by absolute impact
        feature_importance.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return {
            'method': 'shap',
            'model': 'tfidf',
            'base_value': float(shap_values[0].base_values[0]),
            'features': feature_importance[:top_k],
            'prediction': model.predict(text)['prediction']
        }
    
    def _explain_transformer_shap(self, model, text: str, top_k: int) -> Dict[str, Any]:
        """Generate SHAP explanation for transformer model"""
        # Tokenize text
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Get model prediction
        model.model.eval()
        with torch.no_grad():
            outputs = model.model(tokens)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Create SHAP explainer for transformers
        explainer = shap.Explainer(
            lambda x: self._transformer_predict(model, x),
            tokenizer
        )
        
        # Generate SHAP values
        shap_values = explainer([text])
        
        # Process token-level importance
        token_importance = []
        tokens_decoded = tokenizer.convert_ids_to_tokens(tokens[0])
        
        for i, token in enumerate(tokens_decoded):
            if i < len(shap_values[0].values):
                token_importance.append({
                    'token': token,
                    'importance': float(shap_values[0].values[i][0]),  # Fake class importance
                    'normalized_importance': abs(float(shap_values[0].values[i][0]))
                })
        
        # Sort by importance
        token_importance.sort(key=lambda x: x['normalized_importance'], reverse=True)
        
        return {
            'method': 'shap',
            'model': model.model_name,
            'tokens': token_importance[:top_k],
            'prediction': 'fake' if probabilities[0][0] > 0.5 else 'real'
        }
    
    def _transformer_predict(self, model, texts):
        """Helper function for transformer predictions"""
        predictions = []
        for text in texts:
            encoding = model.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            model.model.eval()
            with torch.no_grad():
                outputs = model.model(**encoding)
                prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions.append(prob.numpy())
        
        return np.vstack(predictions)
    
    def explain_with_lime(self, text: str, model_name: str = 'tfidf', num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for prediction
        
        Args:
            text: Input text
            model_name: Name of model to explain
            num_features: Number of features to show
        
        Returns:
            Explanation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        def predictor_wrapper(texts):
            if model_name == 'tfidf':
                return model.model.predict_proba(texts)
            else:
                # For transformer models
                probs = []
                for t in texts:
                    result = model.predict(t)
                    probs.append([result['probabilities']['fake'], result['probabilities']['real']])
                return np.array(probs)
        
        # Generate LIME explanation
        exp = self.explainer_lime.explain_instance(
            text,
            predictor_wrapper,
            num_features=num_features,
            num_samples=100
        )
        
        # Extract explanation
        explanation = {
            'method': 'lime',
            'model': model_name,
            'prediction': 'fake' if exp.predict_proba[0] > 0.5 else 'real',
            'confidence': max(exp.predict_proba),
            'features': []
        }
        
        # Add feature contributions
        for feature, weight in exp.as_list():
            explanation['features'].append({
                'feature': feature,
                'weight': weight
            })
        
        return explanation
    
    def explain_ensemble(self, text: str, method: str = 'lime') -> Dict[str, Any]:
        """
        Generate explanation for ensemble prediction
        
        Args:
            text: Input text
            method: Explanation method (lime or shap)
        
        Returns:
            Ensemble explanation
        """
        explanations = {}
        
        for model_name in self.models:
            try:
                if method == 'lime':
                    explanations[model_name] = self.explain_with_lime(text, model_name)
                elif method == 'shap':
                    explanations[model_name] = self.explain_with_shap(text, model_name)
            except Exception as e:
                explanations[model_name] = {
                    'error': str(e),
                    'method': method
                }
        
        # Get ensemble prediction
        ensemble_prediction = None
        if 'ensemble' in self.models:
            ensemble_prediction = self.models['ensemble'].predict(text)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'model_explanations': explanations,
            'method': method,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        }