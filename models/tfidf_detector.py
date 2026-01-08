import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List
import pickle
import re
from .base_model import BaseFakeNewsDetector

class TfidfDetector(BaseFakeNewsDetector):
    """TF-IDF + classifier for fake news detection"""
    
    def __init__(self, model_name: str = "tfidf_logreg"):
        super().__init__(model_name)
        
        # Text preprocessing for Polish
        self.stopwords = self._load_polish_stopwords()
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=self.stopwords,
                min_df=2,
                max_df=0.9
            )),
            ('clf', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
    
    def _load_polish_stopwords(self) -> list:
        """Load Polish stopwords"""
        # Basic Polish stopwords - can be extended
        polish_stopwords = [
            'i', 'oraz', 'z', 'ze', 'w', 'we', 'o', 'na', 'do', 'od', 'po', 
            'za', 'przez', 'pod', 'dla', 'a', 'ale', 'lub', 'czy', 'co',
            'to', 'ten', 'ta', 'to', 'te', 'się', 'nie', 'jest', 'jak'
        ]
        return polish_stopwords
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess Polish text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, texts: List[str], labels: List[int], **kwargs):
        """Train TF-IDF + classifier"""
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Train the model
        self.model.fit(processed_texts, labels)
        self.is_trained = True
        
        # Store feature names for explanation
        if hasattr(self.model.named_steps['tfidf'], 'get_feature_names_out'):
            self.feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
        else:
            self.feature_names = None
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction for single text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Get probabilities
        try:
            probabilities = self.model.predict_proba([processed_text])[0]
        except:
            # If model hasn't been trained with enough classes
            probabilities = [0.5, 0.5]
        
        fake_prob = probabilities[0]
        real_prob = probabilities[1] if len(probabilities) > 1 else 1 - fake_prob
        
        prediction = 1 if real_prob > 0.5 else 0
        
        return {
            'prediction': prediction,
            'confidence': max(fake_prob, real_prob),
            'probabilities': {
                'fake': float(fake_prob),
                'real': float(real_prob)
            },
            'model': 'tfidf'
        }
    
    def predict_proba(self, text: str) -> float:
        """Return probability of being fake news (class 0)"""
        result = self.predict(text)
        return result['probabilities']['fake']
    
    def get_top_features(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top contributing features for prediction"""
        if not self.is_trained or self.feature_names is None:
            return []
        
        processed_text = self._preprocess_text(text)
        
        # Get TF-IDF vector
        tfidf_vectorizer = self.model.named_steps['tfidf']
        vector = tfidf_vectorizer.transform([processed_text])
        
        # Get feature coefficients
        classifier = self.model.named_steps['clf']
        coefficients = classifier.coef_[0]
        
        # Get non-zero features
        feature_indices = vector.nonzero()[1]
        feature_values = vector.data
        
        # Calculate contributions
        contributions = []
        for idx, value in zip(feature_indices, feature_values):
            feature_name = self.feature_names[idx]
            contribution = value * coefficients[idx]
            contributions.append({
                'feature': feature_name,
                'value': float(value),
                'coefficient': float(coefficients[idx]),
                'contribution': float(contribution)
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return contributions[:top_n]