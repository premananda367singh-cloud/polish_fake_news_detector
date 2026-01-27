"""
SPST Training Script for RoBERTa (Minimal Resource Configuration)
Adapted from BERT SPST methodology for low-resource environments
"""

import torch
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('roberta_spst_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== MINIMAL CONFIGURATION ==========
CONFIG = {
    # Model settings - using smallest RoBERTa
    "model_name": "distilroberta-base",  # Smaller than roberta-base
    "max_length": 128,  # Reduced from 512
    "num_labels": 2,
    
    # Training settings - minimal
    "batch_size": 4,  # Smallest practical batch size
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "learning_rate": 2e-5,
    "num_epochs": 2,  # Reduced from 3
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    
    # SPST settings
    "segment_size": 500,  # Small segments for memory efficiency
    "save_steps": 100,
    
    # Paths
    "data_path": "datasets/processed/unified_all.tsv",
    "output_dir": "models/roberta-spst-minimal",
    "log_dir": "logs/roberta",
    
    # Memory optimization
    "fp16": True,  # Mixed precision training
    "dataloader_num_workers": 0,  # Set to 0 for minimal RAM
    "pin_memory": False,
    
    # Seed
    "seed": 42
}

# ========== DATASET CLASS ==========
class FakeNewsDataset(Dataset):
    """Lightweight dataset for fake news detection"""
    
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
        
        # Tokenize on-the-fly to save memory
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ========== UTILITY FUNCTIONS ==========
def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_data(data_path):
    """Load and prepare dataset"""
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path, sep='\t')
        logger.info(f"Loaded {len(df)} samples")
        
        # Ensure we have required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns")
        
        # Convert labels to binary (0=real, 1=fake)
        df['label'] = df['label'].apply(lambda x: 1 if x in ['fake', 'false', 1] else 0)
        
        return df['text'].tolist(), df['label'].tolist()
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_segments(texts, labels, segment_size):
    """Split data into segments for SPST"""
    num_segments = len(texts) // segment_size + (1 if len(texts) % segment_size != 0 else 0)
    segments = []
    
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(texts))
        segments.append({
            'texts': texts[start_idx:end_idx],
            'labels': labels[start_idx:end_idx]
        })
    
    logger.info(f"Created {len(segments)} segments of size {segment_size}")
    return segments

# ========== TRAINING FUNCTIONS ==========
def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / config['gradient_accumulation_steps']
        
        # Backward pass
        if config['fp16']:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item()
        
        # Gradient accumulation
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            if config['fp16']:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            if config['fp16']:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item() * config['gradient_accumulation_steps']})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ========== MAIN TRAINING LOOP ==========
def train_spst(config):
    """Main SPST training function"""
    
    # Setup
    set_seed(config['seed'])
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading {config['model_name']}")
    tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])
    model = RobertaForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_labels']
    )
    model.to(device)
    
    # Load and segment data
    texts, labels = load_data(config['data_path'])
    segments = create_segments(texts, labels, config['segment_size'])
    
    # Split last segment for validation
    train_segments = segments[:-1]
    val_segment = segments[-1]
    
    logger.info(f"Training on {len(train_segments)} segments, validating on 1 segment")
    
    # Prepare validation set
    val_dataset = FakeNewsDataset(
        val_segment['texts'], 
        val_segment['labels'], 
        tokenizer, 
        config['max_length']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        num_workers=config['dataloader_num_workers']
    )
    
    # SPST Sequential Training
    for segment_idx, segment in enumerate(train_segments):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Segment {segment_idx + 1}/{len(train_segments)}")
        logger.info(f"{'='*50}")
        
        # Create dataset for this segment
        train_dataset = FakeNewsDataset(
            segment['texts'],
            segment['labels'],
            tokenizer,
            config['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['dataloader_num_workers'],
            pin_memory=config['pin_memory']
        )
        
        # Setup optimizer and scheduler for this segment
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
        total_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Train on this segment
        for epoch in range(config['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
            
            avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
            logger.info(f"Average Loss: {avg_loss:.4f}")
            
            # Evaluate after each epoch
            metrics = evaluate(model, val_loader, device)
            logger.info(f"Validation - Acc: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}")
        
        # Save checkpoint after segment
        checkpoint_path = os.path.join(config['output_dir'], f'segment_{segment_idx}')
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clear memory
        del train_dataset, train_loader
        clear_memory()
    
    # Final save
    final_path = os.path.join(config['output_dir'], 'final_model')
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete! Final model saved to {final_path}")
    
    # Final evaluation
    logger.info("\nFinal Evaluation:")
    final_metrics = evaluate(model, val_loader, device)
    for metric, value in final_metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    logger.info("Starting RoBERTa SPST Training (Minimal Configuration)")
    logger.info(f"Configuration: {CONFIG}")
    
    try:
        train_spst(CONFIG)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
