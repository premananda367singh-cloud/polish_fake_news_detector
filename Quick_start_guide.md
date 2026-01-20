# Quick Start Guide: English Fake News Detection

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Download NLTK data:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### 2. Prepare Your Dataset

You have two options:

#### Option A: CSV File (Recommended)
Place a CSV file at `datasets/raw/dataset.csv` with columns:
- `text`: The article content
- `label`: 0 for fake, 1 for real

Example CSV format:
```csv
text,label
"Breaking: Scientists discover new planet...","1"
"You won't believe what happened next...","0"
```

#### Option B: Text Files
Create this directory structure:
```
datasets/
└── raw/
    ├── fake/
    │   ├── article_001.txt
    │   ├── article_002.txt
    │   └── ...
    └── real/
        ├── article_001.txt
        ├── article_002.txt
        └── ...
```

### 3. Download a Dataset (Examples)

**ISOT Fake News Dataset (Recommended):**
```bash
# Download from: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
# Or Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# After downloading, place files in datasets/raw/
```

**Kaggle Dataset:**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset

# Unzip and organize
unzip fake-and-real-news-dataset.zip -d datasets/raw/
```

### 4. Preprocess Data

```bash
python preprocess_data.py
```

This will:
- Load your raw data
- Clean and preprocess text
- Create train/val/test splits (80/10/10)
- Save processed data to `datasets/processed/`

### 5. Train BERT

```bash
python train_bert.py
```

**Expected output:**
- Training progress with metrics
- Best model saved to `trained_models/bert/`
- Training logs in `logs/bert/`

**Training time:** ~1-3 hours on GPU (depends on dataset size)

### 6. Train RoBERTa

```bash
python train_roberta.py
```

**Expected output:**
- Training progress with metrics
- Best model saved to `trained_models/roberta/`
- Training logs in `logs/roberta/`

**Training time:** ~1-3 hours on GPU

### 7. Monitor Training (Optional)

In a separate terminal:
```bash
tensorboard --logdir=logs/
```

Then open: http://localhost:6006

## Adjusting for Limited GPU Memory

If you get CUDA out of memory errors:

### Option 1: Reduce Batch Size
Edit `config_english.py`:
```python
BERT_CONFIG = {
    "batch_size": 8,  # Reduced from 16
    # ... rest of config
}
```

### Option 2: Use Gradient Accumulation
Edit `config_english.py`:
```python
BERT_CONFIG = {
    "batch_size": 8,
    "gradient_accumulation_steps": 2,  # Effective batch size = 8 * 2 = 16
    # ... rest of config
}
```

### Option 3: Enable Gradient Checkpointing
Add to training script after model initialization:
```python
model.gradient_checkpointing_enable()
```

### Option 4: Use Smaller Models
Edit `config_english.py`:
```python
BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",  # Smaller, faster
    # ... rest of config
}

ROBERTA_CONFIG = {
    "model_name": "distilroberta-base",  # Smaller, faster
    # ... rest of config
}
```

## Common Issues and Solutions

### Issue 1: "No data loaded"
**Solution:** Check that your data files are in the correct location and format.

```bash
# Check directory structure
ls -R datasets/raw/
```

### Issue 2: "CUDA out of memory"
**Solutions:**
1. Reduce batch size to 8 or 4
2. Enable gradient accumulation
3. Use gradient checkpointing
4. Use a smaller model (distilbert/distilroberta)
5. Reduce max_length to 256

### Issue 3: "Low accuracy"
**Possible causes:**
1. Dataset too small → Collect more data
2. Class imbalance → Balance your dataset
3. Model underfitting → Train more epochs
4. Model overfitting → Add regularization/dropout

### Issue 4: "Training too slow"
**Solutions:**
1. Use mixed precision training (already enabled)
2. Increase num_workers in data loader
3. Use a smaller max_length (256 instead of 512)
4. Use gradient accumulation with smaller batch size

## Expected Results

**Good performance indicators:**
- Validation accuracy: >85%
- F1 score: >0.85
- Training loss decreasing consistently
- Val loss not increasing (no overfitting)

**Typical results with ISOT dataset:**
- BERT: 95-98% accuracy
- RoBERTa: 96-99% accuracy

## Next Steps After Training

1. **Compare Models:**
   - Check `trained_models/bert/results.txt`
   - Check `trained_models/roberta/results.txt`
   - Compare F1 scores, accuracy, etc.

2. **Test on New Data:**
   Create a simple prediction script

3. **Build Ensemble (Later):**
   Once both models work well individually

4. **Deploy:**
   - Create a simple API with Flask/FastAPI
   - Build a web interface with Streamlit

## File Structure After Setup

```
your_project/
├── config_english.py          # Configuration
├── preprocess_data.py         # Data preprocessing
├── train_bert.py              # BERT training
├── train_roberta.py           # RoBERTa training
├── requirements.txt           # Dependencies
├── MIGRATION_GUIDE.md         # Detailed guide
├── QUICK_START.md            # This file
│
├── datasets/
│   ├── raw/                   # Your raw data
│   │   ├── fake/
│   │   └── real/
│   └── processed/             # Preprocessed splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── trained_models/
│   ├── bert/                  # BERT checkpoints
│   │   └── checkpoint_epoch_X/
│   └── roberta/               # RoBERTa checkpoints
│       └── checkpoint_epoch_X/
│
└── logs/
    ├── bert/                  # BERT training logs
    └── roberta/               # RoBERTa training logs
```

## Useful Commands

**Check GPU:**
```bash
nvidia-smi
```

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check Python environment:**
```bash
python --version
pip list | grep torch
```

**Clear GPU cache (if needed):**
```python
import torch
torch.cuda.empty_cache()
```

## Tips for Success

1. **Start small:** Test with a subset of data first
2. **Monitor closely:** Watch training metrics in real-time
3. **Save checkpoints:** Don't lose progress if training stops
4. **Compare results:** Train both models before choosing
5. **Document everything:** Keep notes on what works
6. **Be patient:** Training takes time, especially on large datasets

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review the MIGRATION_GUIDE.md for detailed explanations
3. Check your GPU memory with `nvidia-smi`
4. Verify data format and paths
5. Try with a smaller subset of data first

## Summary

```bash
# Complete workflow
pip install -r requirements.txt
python preprocess_data.py
python train_bert.py
python train_roberta.py
```

That's it! You now have two trained models for fake news detection.
