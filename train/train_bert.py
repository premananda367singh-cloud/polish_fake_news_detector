"""
Train BERT model for fake news detection
This script trains BERT independently without ensemble
"""

import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

# Import configuration
try:
    from config_english import BERT_CONFIG, TRAIN_CONFIG, DATA_FILES, DEVICE_CONFIG
except ImportError:
    print("Error: config_english.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BERT_CONFIG['log_dir'] / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """Dataset class for fake news articles."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load preprocessed data from CSV files."""
    logger.info("Loading data...")
    
    train_df = pd.read_csv(DATA_FILES['train'])
    val_df = pd.read_csv(DATA_FILES['val'])
    test_df = pd.read_csv(DATA_FILES['test'])
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_data_loaders(train_df, val_df, test_df, tokenizer, config):
    """Create PyTorch data loaders."""
    
    train_dataset = FakeNewsDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    val_dataset = FakeNewsDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    test_dataset = FakeNewsDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def calculate_metrics(predictions, labels, probabilities=None):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if probabilities is not None:
        try:
            auc = roc_auc_score(labels, probabilities[:, 1])
            metrics['auc'] = auc
        except:
            logger.warning("Could not calculate AUC score")
    
    return metrics

def train_epoch(model, data_loader, optimizer, scheduler, device, scaler, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        # Gradient accumulation
        loss = loss / config['gradient_accumulation_steps']
        scaler.scale(loss).backward()
        
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        
        # Get predictions
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item() * config['gradient_accumulation_steps']})
    
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(predictions, true_labels)
    metrics['loss'] = avg_loss
    
    return metrics

def evaluate(model, data_loader, device):
    """Evaluate the model."""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    probabilities = np.array(probabilities)
    metrics = calculate_metrics(predictions, true_labels, probabilities)
    metrics['loss'] = avg_loss
    
    return metrics

def save_checkpoint(model, tokenizer, optimizer, epoch, metrics, config):
    """Save model checkpoint."""
    checkpoint_dir = config['save_dir'] / f"checkpoint_epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_dir / 'training_state.pt')
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    """Main training function."""
    logger.info("Starting BERT training for fake news detection")
    logger.info(f"Configuration: {BERT_CONFIG}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE_CONFIG['use_cuda'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Initialize tokenizer and model
    logger.info(f"Loading BERT model: {BERT_CONFIG['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(BERT_CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(
        BERT_CONFIG['model_name'],
        num_labels=BERT_CONFIG['num_labels']
    )
    model.to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, BERT_CONFIG
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=BERT_CONFIG['learning_rate'],
        weight_decay=BERT_CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * BERT_CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=BERT_CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(BERT_CONFIG['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{BERT_CONFIG['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, BERT_CONFIG
        )
        logger.info(f"Train metrics: {train_metrics}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(model, tokenizer, optimizer, epoch, val_metrics, BERT_CONFIG)
            logger.info(f"New best model! F1: {best_val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final results
    results = {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'config': BERT_CONFIG
    }
    
    results_file = BERT_CONFIG['save_dir'] / 'results.txt'
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"\nTraining complete! Results saved to {results_file}")

if __name__ == "__main__":
    main()
