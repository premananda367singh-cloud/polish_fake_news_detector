# SPST-BERT: Segmented Parameter Sequential Training for Fake News Detection

A practical implementation of Segmented Parameter Sequential Training (SPST) for fine-tuning BERT on low-resource fake news detection tasks. This approach enables efficient training under computational and memory constraints while maintaining model performance.

> **Note:** This repository focuses on model training and fine-tuning. For inference and production deployment, see the companion [Text Detection System](https://github.com/your-username/text-detection-system) repository.

---

## Overview

**Core Concept:** Train a single BERT model sequentially across dataset segments rather than jointly or with full retraining.

### Key Advantages

- **Lower Resource Requirements** – Reduced memory and compute compared to joint training
- **Improved Stability** – Better convergence across heterogeneous fake news datasets
- **Reproducible Training** – Achievable results without large-scale infrastructure
- **Practical Focus** – Emphasis on reproducibility over aesthetic polish

---

## Problem Statement

Training robust fake news classifiers typically requires:
- Large computational resources for joint multi-dataset training
- Significant memory for handling diverse datasets simultaneously
- Complex infrastructure for managing training at scale

**SPST addresses these challenges** by enabling sequential training that:
- Adapts gradually across domains
- Retains previously learned representations
- Avoids catastrophic forgetting and parameter reinitialization

---

## Technical Details

### Model & Task

- **Base Model:** BERT (base, uncased)
- **Training Strategy:** Segmented Parameter Sequential Training (SPST)
- **Task:** Binary classification (real vs. fake news)
- **Output:** Single robust classifier trained under resource constraints

### Datasets

Three widely-used fake news detection datasets:

1. **LIAR** – Political statements dataset
2. **GossipCop** – Entertainment and celebrity news
3. **PolitiFact** – Fact-checked political claims

All datasets are unified into a common schema before training. Data is intentionally minimally cleaned to reflect real-world noise and variability.

---

## Repository Structure

```
spst-bert/
├── config.py                      # Training and model hyperparameters
├── train_bert.py                  # Main SPST training entry point
├── setup_directories.py           # Directory structure initialization
├── requirements.txt               # Python dependencies
│
├── datasets/
│   ├── raw/                       # Original datasets (LIAR, GossipCop, PolitiFact)
│   └── processed/                 # Unified datasets for training
│       ├── unified_all.tsv
│       └── unified_all_with_source.tsv
│
├── processor/
│   ├── format_preprocessor.py     # Dataset formatting utilities
│   ├── unify.py                   # Cross-dataset unification logic
│   └── schemaa.txt                # Target unified schema definition
│
├── services/
│   ├── inference.py               # Model inference interface
│   └── explainer.py               # Prediction explanation utilities
│
├── logs/
│   └── bert/                      # Training logs and metrics
│
└── test_file/
    └── test.tsv                   # Sample test input
```

---

## SPST Methodology

### Sequential Training Process

Instead of training BERT on all datasets simultaneously, SPST performs:

1. **Dataset Unification** – Normalize labels and align column structures
2. **Sequential Training** – Train over dataset segments in sequence
3. **Parameter Reuse** – Continue from previous weights without reinitialization

### Benefits of Sequential Approach

- **Gradual Domain Adaptation** – Model learns cross-dataset patterns incrementally
- **Knowledge Retention** – Previously learned representations are preserved
- **Catastrophic Forgetting Mitigation** – Avoids parameter resets between segments
- **Resource Efficiency** – Lower peak memory usage compared to joint training

Full methodological details, theoretical motivation, and empirical validation are provided in the accompanying research paper.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/premananda-cloud/Bert_training_via_SPST
cd Bert_training_via_SPST

# Install dependencies
pip install -r requirements.txt

# Initialize directory structure
python setup_directories.py
```

---

## Data Preparation

### Unified Dataset Format

All datasets are processed into TSV format under `datasets/processed/`:

- `unified_all.tsv` – Combined dataset without source labels
- `unified_all_with_source.tsv` – Combined dataset with dataset origin preserved

### Unification Process

The preprocessing pipeline:

1. **Label Normalization** – Convert multi-class labels to binary (real/fake)
2. **Column Alignment** – Standardize feature names and data types
3. **Source Preservation** – Optionally retain dataset origin for analysis
4. **Schema Validation** – Ensure consistency with `processor/schemaa.txt`

### Running Unification

```bash
python processor/unify.py --input datasets/raw/ --output datasets/processed/
```

---

## Training

### Basic Training

```bash
python train_bert.py --config config.py
```

### Configuration

Edit `config.py` to customize training parameters:

```python
# Model settings
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SEGMENT_SIZE = 1000  # Records per training segment

# Paths
DATA_PATH = "datasets/processed/unified_all.tsv"
OUTPUT_DIR = "models/spst-bert"
LOG_DIR = "logs/bert"
```

### Training Process

The training script:

1. Loads unified dataset
2. Segments data according to `SEGMENT_SIZE`
3. Trains sequentially across segments
4. Saves checkpoints after each segment
5. Logs metrics and training progress

### Monitoring

Training logs are saved to `logs/bert/`:

```bash
# View training progress
tail -f logs/bert/training.log

# Tensorboard (if enabled)
tensorboard --logdir logs/bert/
```

---

## Inference

### Basic Inference

```python
from services.inference import InferencePipeline

pipeline = InferencePipeline(model_path="models/spst-bert")
result = pipeline.predict("Breaking: New study shows surprising results...")
print(result)  # {'label': 'fake', 'confidence': 0.87}
```

### Batch Inference

```python
import pandas as pd

test_data = pd.read_csv("test_file/test.tsv", sep="\t")
predictions = pipeline.predict_batch(test_data["text"].tolist())
```

### Explanation

```python
from services.explainer import Explainer

explainer = Explainer(pipeline)
explanation = explainer.interpret("Sample news article text...")
# Returns token-level importance scores
```

---

## Model Outputs

After training, the following artifacts are saved:

```
models/spst-bert/
├── pytorch_model.bin          # Model weights
├── config.json                # Model configuration
├── tokenizer_config.json      # Tokenizer settings
├── vocab.txt                  # Vocabulary
└── training_args.bin          # Training hyperparameters
```

These can be loaded directly into the inference pipeline or exported to other frameworks.

---

## Integration with Deployment System

This training repository produces model artifacts that integrate with the [Text Detection System](https://github.com/your-username/text-detection-system) for production deployment:

1. Train model using SPST methodology (this repository)
2. Export trained model to `models/spst-bert/`
3. Copy model artifacts to deployment system's model directory
4. Configure deployment system to load SPST-trained model
5. Run inference through production-grade orchestration layer

---

## Evaluation

### Running Evaluation

```bash
python train_bert.py --mode evaluate --checkpoint models/spst-bert
```

### Metrics

The evaluation script reports:

- **Accuracy** – Overall classification accuracy
- **Precision/Recall/F1** – Per-class and macro-averaged
- **Confusion Matrix** – Error analysis
- **Dataset-Specific Performance** – Breakdown by source dataset (if using `unified_all_with_source.tsv`)

---

## Extending the System

### Adding New Datasets

1. Place raw data in `datasets/raw/`
2. Update `processor/unify.py` to handle new schema
3. Run unification to include in training set
4. Retrain using SPST on expanded dataset

### Modifying Training Strategy

- Edit `train_bert.py` to change segment ordering
- Adjust `SEGMENT_SIZE` in `config.py` for different granularity
- Implement custom loss functions or regularization

### Custom Model Architectures

- Replace `MODEL_NAME` in config with other transformer models (RoBERTa, DeBERTa, etc.)
- Adjust `MAX_LENGTH` and `BATCH_SIZE` accordingly
- Maintain SPST training loop structure

---

## Reproducibility

### Environment

- Python 3.8.10
- PyTorch 2.0.1
- Transformers 4.30.2
- CUDA 11.8 (for GPU training)

See `requirements.txt` for full dependency versions.

### Random Seeds

All random seeds are fixed in `config.py`:

```python
SEED = 42  # Set for reproducibility
```

### Hardware

Experiments were conducted on:
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- CPU: Intel i7-12700K
- RAM: 32GB DDR4

Training times:
- Full SPST training: ~4-6 hours (depending on segment size)
- Single epoch joint training: ~8-10 hours

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{your-paper-2025,
  title={SPST-BERT: Segmented Parameter Sequential Training for Low-Resource Fake News Detection},
  author={Your Name and Collaborators},
  journal={Journal/Conference Name},
  year={2025}
}
```

---

## Known Limitations

- **Sequential Order Dependency** – Performance may vary with different segment orderings
- **Domain Shift Sensitivity** – Large distribution gaps between segments may require tuning
- **Memory Accumulation** – Very long training runs may require periodic cache clearing
- **Dataset Balance** – No automatic rebalancing across segments

These are areas for future research and improvement.

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

See `requirements.txt` for complete list.

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows existing style
5. Submit a pull request with clear description

---

## Contact

For questions, issues, or collaboration:

- **GitHub Issues:** [Open an issue](https://github.com/premananda-cloud/Bert_training_via_SPST/issues)
- **Email:** (Add your email here)

---

## Acknowledgments

- BERT implementation based on HuggingFace Transformers
- Datasets: LIAR, GossipCop, PolitiFact (see respective papers)
- Inspired by research on continual learning and low-resource NLP

---

**Note:** This is a research implementation. Validate thoroughly before production use.
