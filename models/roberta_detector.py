import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List
from .base_model import BaseFakeNewsDetector
from config import MODEL_CONFIG

class RobertaDetector(BaseFakeNewsDetector):
    """RoBERTa-based fake news detector for Polish"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = MODEL_CONFIG.roberta_model_name
        super().__init__(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and preprocess text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MODEL_CONFIG.max_length,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def train(self, texts: List[str], labels: List[int], **kwargs):
        """Fine-tune RoBERTa on fake news detection task"""
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        # Prepare dataset
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MODEL_CONFIG.max_length,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=MODEL_CONFIG.batch_size,
            shuffle=True
        )
        
        # Training setup
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=MODEL_CONFIG.learning_rate
        )
        
        # Training loop
        self.model.train()
        for epoch in range(MODEL_CONFIG.epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids, attention_mask, batch_labels = batch
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_trained = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction for single text"""
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess(text)
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
            
            prediction = 1 if real_prob > MODEL_CONFIG.threshold else 0
            
            return {
                'prediction': prediction,
                'confidence': max(fake_prob, real_prob),
                'probabilities': {
                    'fake': float(fake_prob),
                    'real': float(real_prob)
                },
                'model': 'roberta'
            }
    
    def predict_proba(self, text: str) -> float:
        """Return probability of being fake news (class 0)"""
        result = self.predict(text)
        return result['probabilities']['fake']