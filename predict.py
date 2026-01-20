"""
Prediction script for fake news detection
Use trained BERT or RoBERTa models to classify new articles
"""

import torch
import argparse
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification
)
from pathlib import Path

def load_model(model_path, model_type='bert'):
    """Load a trained model and tokenizer."""
    
    print(f"Loading {model_type.upper()} model from {model_path}")
    
    if model_type.lower() == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    elif model_type.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer, device

def predict_text(text, model, tokenizer, device, max_length=512):
    """Predict whether a text is fake or real."""
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    # Get results
    pred_label = prediction.item()
    confidence = probabilities[0][pred_label].item()
    
    label_map = {0: "FAKE", 1: "REAL"}
    
    return {
        'prediction': label_map[pred_label],
        'confidence': confidence,
        'fake_probability': probabilities[0][0].item(),
        'real_probability': probabilities[0][1].item()
    }

def print_result(result, text_preview=None):
    """Pretty print prediction result."""
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    
    if text_preview:
        preview = text_preview[:200] + "..." if len(text_preview) > 200 else text_preview
        print(f"\nText preview: {preview}\n")
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nDetailed probabilities:")
    print(f"  FAKE: {result['fake_probability']:.2%}")
    print(f"  REAL: {result['real_probability']:.2%}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Predict fake news using trained models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['bert', 'roberta'], default='bert',
                       help='Type of model to use')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to classify')
    parser.add_argument('--file', type=str, default=None,
                       help='Path to text file to classify')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path, args.model_type)
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("INTERACTIVE FAKE NEWS DETECTION")
        print("="*60)
        print("Enter text to analyze (or 'quit' to exit)")
        print("="*60 + "\n")
        
        while True:
            try:
                text = input("\nEnter news text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not text:
                    print("Please enter some text.")
                    continue
                
                result = predict_text(text, model, tokenizer, device)
                print_result(result, text)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # File mode
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = predict_text(text, model, tokenizer, device)
            print_result(result, text)
            
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Direct text mode
    elif args.text:
        result = predict_text(args.text, model, tokenizer, device)
        print_result(result, args.text)
    
    else:
        print("Please provide --text, --file, or use --interactive mode")
        parser.print_help()

if __name__ == "__main__":
    main()
