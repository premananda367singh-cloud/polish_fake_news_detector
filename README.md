# English Fake News Detector

A transformer-based fake news detection system using BERT and RoBERTa models, adapted from a Polish fake news detector for English language datasets.

## Features

✅ **Segmented Training**: Train BERT and RoBERTa independently  
✅ **Memory Efficient**: Gradient accumulation and mixed precision support  
✅ **Easy Data Management**: Organized dataset folder structure  
✅ **Production Ready**: Clean service-oriented architecture  
✅ **Comprehensive Logging**: TensorBoard integration  
✅ **Flexible Configuration**: Easy-to-modify config file  

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place your data in `datasets/raw/` as either:
- CSV file with `text` and `label` columns
- Text files in `fake/` and `real/` subdirectories

### 3. Preprocess Data
```bash
python preprocess_data.py
```

### 4. Train Models
```bash
# Train BERT
python train_bert.py

# Train RoBERTa
python train_roberta.py
```

### 5. Make Predictions
```bash
# Interactive mode
python predict.py --model_path trained_models/bert/checkpoint_epoch_0 --model_type bert --interactive

# Single text
python predict.py --model_path trained_models/bert/checkpoint_epoch_0 --model_type bert --text "Your news article here"

# From file
python predict.py --model_path trained_models/bert/checkpoint_epoch_0 --model_type bert --file article.txt
```

## Project Structure

```
english-fake-news-detector/
├── config_english.py          # Configuration file
├── preprocess_data.py         # Data preprocessing
├── train_bert.py              # BERT training script
├── train_roberta.py           # RoBERTa training script
├── predict.py                 # Prediction script
├── requirements.txt           # Dependencies
├── MIGRATION_GUIDE.md         # Detailed migration guide
├── QUICK_START.md            # Quick start guide
│
├── datasets/
│   ├── raw/                   # Raw data
│   │   ├── fake/             # Fake news articles
│   │   └── real/             # Real news articles
│   └── processed/            # Processed data splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── trained_models/
│   ├── bert/                 # BERT model checkpoints
│   └── roberta/              # RoBERTa model checkpoints
│
└── logs/
    ├── bert/                 # BERT training logs
    └── roberta/              # RoBERTa training logs
```

## Configuration

All settings are in `config_english.py`:

### Model Settings
```python
BERT_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
}

ROBERTA_CONFIG = {
    "model_name": "roberta-base",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "epochs": 3,
}
```

### Training Settings
```python
TRAIN_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "early_stopping_patience": 3,
}
```

## Dataset Recommendations

### Recommended Datasets for English:

1. **ISOT Fake News Dataset** (Recommended)
   - 44,000 articles
   - Binary classification
   - Download: https://www.uvic.ca/engineering/ece/isot/datasets/

2. **Kaggle Fake News Dataset**
   - 20,000+ articles
   - Binary classification
   - Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

3. **LIAR Dataset**
   - 12,800 statements
   - Multi-class labels
   - Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

## Memory Optimization

If you encounter GPU memory issues:

### 1. Reduce Batch Size
```python
# In config_english.py
BERT_CONFIG["batch_size"] = 8  # or 4
```

### 2. Enable Gradient Accumulation
```python
BERT_CONFIG["gradient_accumulation_steps"] = 2
# Effective batch size = batch_size * gradient_accumulation_steps
```

### 3. Use Smaller Models
```python
BERT_CONFIG["model_name"] = "distilbert-base-uncased"
ROBERTA_CONFIG["model_name"] = "distilroberta-base"
```

### 4. Reduce Sequence Length
```python
BERT_CONFIG["max_length"] = 256  # Instead of 512
```

## Training Tips

1. **Monitor Training**: Use TensorBoard
   ```bash
   tensorboard --logdir=logs/
   ```

2. **Start Small**: Test with a subset first
   ```python
   PREPROCESSING_CONFIG["max_samples_per_class"] = 1000
   ```

3. **Check GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Save Checkpoints**: Already configured in training scripts

## Expected Performance

With ISOT dataset:
- **BERT**: 95-98% accuracy, F1: 0.95-0.98
- **RoBERTa**: 96-99% accuracy, F1: 0.96-0.99

Training time (on RTX 3060):
- **BERT**: ~1-2 hours
- **RoBERTa**: ~1-2 hours

## Key Differences from Polish Version

### Changed:
- ❌ Polish BERT → ✅ English BERT
- ❌ Polish preprocessing → ✅ English preprocessing
- ❌ Polish stop words → ✅ English stop words
- ❌ Ensemble training → ✅ Separate training scripts

### Added:
- ✅ Segmented training support
- ✅ Memory optimization options
- ✅ Interactive prediction mode
- ✅ Comprehensive documentation

### Kept:
- ✅ Service-oriented architecture
- ✅ Clean separation of concerns
- ✅ Professional code structure

## Usage Examples

### Training
```bash
# Basic training
python train_bert.py

# With custom config (modify config_english.py first)
python train_bert.py
```

### Prediction
```bash
# Interactive mode (recommended for testing)
python predict.py \
  --model_path trained_models/bert/checkpoint_epoch_0 \
  --model_type bert \
  --interactive

# Batch prediction from file
python predict.py \
  --model_path trained_models/roberta/checkpoint_epoch_0 \
  --model_type roberta \
  --file news_article.txt

# Single text prediction
python predict.py \
  --model_path trained_models/bert/checkpoint_epoch_0 \
  --model_type bert \
  --text "Scientists discover new planet in habitable zone"
```

## Future Enhancements

- [ ] Ensemble model combining BERT and RoBERTa
- [ ] Multi-class classification (beyond binary)
- [ ] Multi-modal analysis (text + images)
- [ ] Real-time API deployment
- [ ] Streamlit web interface
- [ ] Explainability features (LIME, SHAP)

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size to 8 or 4
- Enable gradient accumulation
- Use smaller models (distilbert/distilroberta)

### "No data loaded"
- Check dataset path in config_english.py
- Verify CSV format or text file structure
- Ensure files are UTF-8 encoded

### "Low accuracy"
- Check data quality and balance
- Increase training epochs
- Try different learning rates
- Ensure proper train/val/test split

### "Training too slow"
- Use smaller max_length (256)
- Enable mixed precision (already enabled)
- Use gradient accumulation
- Reduce num_workers if CPU-bound

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ RAM

## License

This project is adapted from the Polish fake news detector. Please check the original repository for license information.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Better preprocessing techniques
- Ensemble methods
- Deployment solutions
- Documentation improvements

## Acknowledgments

- Original Polish fake news detector by premananda-cloud
- Hugging Face for transformer models
- ISOT for the fake news dataset

## Contact

For issues or questions, please create an issue in the repository.

---

**Note**: This is an educational project. Always verify news from multiple reliable sources!
