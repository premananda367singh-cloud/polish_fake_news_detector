# English Fake News Detector - Migration Guide

## Overview
This guide explains how to adapt the Polish fake news detector for English with segmented BERT and RoBERTa training.

## Key Changes Required

### 1. Language Model Changes
**Polish → English**
- Replace Polish BERT models with English ones
- Update tokenizers for English
- Change preprocessing for English stop words

### 2. Dataset Organization
**Recommended Structure:**
```
datasets/
├── raw/
│   ├── fake/
│   │   ├── article_001.txt
│   │   ├── article_002.txt
│   │   └── ...
│   └── real/
│       ├── article_001.txt
│       ├── article_002.txt
│       └── ...
├── processed/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── metadata/
    └── sources.json
```

**Why this structure?**
- ✅ Easy to add new data incrementally
- ✅ Clear separation of fake/real labels
- ✅ Version control friendly
- ✅ Supports multiple data formats

### 3. English Dataset Sources

**Popular English Fake News Datasets:**

1. **LIAR Dataset** (12.8K statements)
   - Multi-class labels (pants-fire, false, barely-true, half-true, mostly-true, true)
   - Short statements with metadata
   - Good for initial testing

2. **ISOT Fake News Dataset** (44K articles)
   - Binary classification (fake/real)
   - Full articles
   - Recommended for your use case

3. **Kaggle Fake News Dataset** (20K+ articles)
   - Binary classification
   - News articles with titles and text
   - Widely used benchmark

4. **FakeNewsNet** (Multi-modal)
   - Includes social context
   - PolitiFact and GossipCop sources

**Recommended:** Start with ISOT dataset (most similar to Polish project structure)

### 4. Model Configuration

**English Pre-trained Models:**

**BERT Options:**
- `bert-base-uncased` (110M params) - Recommended for limited resources
- `bert-large-uncased` (340M params) - Better performance, needs more GPU
- `bert-base-cased` - If you want case sensitivity

**RoBERTa Options:**
- `roberta-base` (125M params) - Recommended
- `roberta-large` (355M params) - Better but resource intensive
- `distilroberta-base` (82M params) - Faster, slightly lower accuracy

### 5. Segmented Training Strategy

**Why Segmented Training?**
- Limited GPU memory
- Experiment with different models separately
- Easier debugging
- Flexibility in architecture decisions

**Training Order:**
1. Train BERT first (faster convergence)
2. Train RoBERTa (usually better performance)
3. Compare results
4. Later: Combine in ensemble

## Detailed Implementation Changes

### File-by-File Modifications

#### 1. `config.py` Changes

**Before (Polish):**
```python
MODEL_NAME = "allegro/herbert-base-cased"
LANGUAGE = "polish"
```

**After (English):**
```python
# Model configurations
BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
LANGUAGE = "english"

# Dataset paths
DATASET_DIR = "datasets"
RAW_DATA_DIR = f"{DATASET_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATASET_DIR}/processed"
MODEL_SAVE_DIR = "trained_models"

# Training configurations
BERT_CONFIG = {
    "model_name": BERT_MODEL_NAME,
    "max_length": 512,
    "batch_size": 16,  # Adjust based on GPU
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01
}

ROBERTA_CONFIG = {
    "model_name": ROBERTA_MODEL_NAME,
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01
}
```

#### 2. Text Preprocessing Changes

**Polish preprocessing includes:**
- Polish-specific stop words
- Polish character handling
- Polish stemming/lemmatization

**English preprocessing should include:**
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
```

### 6. Memory-Efficient Training Tips

**Gradient Accumulation:**
```python
# Instead of batch_size=32
# Use batch_size=8 with gradient_accumulation_steps=4
GRADIENT_ACCUMULATION_STEPS = 4
```

**Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

**Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
```

### 7. Dataset Organization Best Practices

**Pros of datasets/ folder:**
- ✅ Centralized data management
- ✅ Easy to gitignore large files
- ✅ Clear data versioning
- ✅ Supports multiple dataset versions
- ✅ Easy backup and transfer

**Structure:**
```
datasets/
├── .gitignore          # Ignore large data files
├── README.md           # Dataset documentation
├── download_data.py    # Script to fetch datasets
├── raw/
│   ├── ISOT/
│   ├── LIAR/
│   └── custom/
├── processed/
│   ├── v1/
│   └── v2/
└── statistics/
    └── data_analysis.json
```

### 8. Creating Separate Training Scripts

**Recommended Approach:**

```
training_scripts/
├── train_bert.py           # Train BERT only
├── train_roberta.py        # Train RoBERTa only
├── train_ensemble.py       # Train ensemble (later)
└── utils/
    ├── data_loader.py
    ├── metrics.py
    └── checkpoint.py
```

## Quick Start Implementation

### Step 1: Update requirements.txt
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
nltk>=3.8.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tensorboard>=2.13.0
accelerate>=0.20.0  # For distributed training
```

### Step 2: Organize Your Data
1. Download ISOT dataset
2. Place in `datasets/raw/ISOT/`
3. Run preprocessing script
4. Generate train/val/test splits

### Step 3: Start with BERT
```bash
python training_scripts/train_bert.py --epochs 3 --batch_size 16
```

### Step 4: Train RoBERTa
```bash
python training_scripts/train_roberta.py --epochs 3 --batch_size 16
```

### Step 5: Compare Results
```bash
python evaluate_models.py --models bert roberta
```

## Next Steps

After successful individual training:
1. Analyze model predictions
2. Identify strengths/weaknesses
3. Design ensemble strategy
4. Implement ensemble training
5. Fine-tune hyperparameters

## Common Pitfalls to Avoid

1. **Don't mix languages in training data**
2. **Don't skip validation set**
3. **Don't use Polish-specific preprocessing for English**
4. **Don't train ensemble before individual models work**
5. **Don't ignore class imbalance**

## Resource Requirements

**Minimum for BERT-base:**
- GPU: 8GB VRAM (RTX 2070 or better)
- RAM: 16GB
- Storage: 20GB

**Minimum for RoBERTa-base:**
- GPU: 8GB VRAM
- RAM: 16GB
- Storage: 20GB

**For both + ensemble:**
- GPU: 12GB VRAM (RTX 3060 or better)
- RAM: 32GB
- Storage: 50GB

## Monitoring Training

Use TensorBoard:
```bash
tensorboard --logdir=runs/
```

Track:
- Training loss
- Validation accuracy
- F1 score
- Confusion matrix
- Learning rate

## Questions to Consider

1. **What's your GPU memory?** → Determines batch size
2. **How much data do you have?** → Determines epochs needed
3. **What's your accuracy target?** → Determines when to stop
4. **Will you deploy this?** → Consider model size

## Summary

✅ Use `datasets/` folder for data organization
✅ Train BERT and RoBERTa separately first
✅ Use English pre-trained models
✅ Start with ISOT or similar dataset
✅ Monitor training with TensorBoard
✅ Keep ensemble for later

The segmented approach gives you flexibility and better understanding of each model's performance before combining them.
