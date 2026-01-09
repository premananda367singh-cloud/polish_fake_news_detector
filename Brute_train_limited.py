# ============================================
# MINIMAL BERT TRAINER - Everything in Project
# ============================================

#@title Step 1: Mount Drive & Setup
from google.colab import drive
drive.mount('/content/drive')

#@title Step 2: Clone Your Git Project to Colab
import os

# Your Git repository URL
GIT_REPO = "https://github.com/premananda367singh-cloud/fake_news_detector"  # ← CHANGE THIS

# Clone to Colab
!git clone {GIT_REPO} /content/fake-news-project

# Navigate to project
os.chdir("/content/fake-news-project")
print("📁 Project location:", os.getcwd())
print("📂 Files:", os.listdir())

#@title Step 3: Install Requirements
!pip install transformers torch

#@title Step 4: Minimal Training Script
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd
import json

def train_minimal_bert():
    """Train a tiny BERT and save IN the project folder"""

    print("🚀 Starting minimal BERT training...")

    # ====================================
    # 1. USE TINY MODEL (for 8GB RAM inference)
    # ====================================
    MODEL_NAME = "prajjwal1/bert-tiny"  # 4.4M params (4MB!) - PERFECT for your needs
    # Other options:
    # - "prajjwal1/bert-mini" (11M params)
    # - "prajjwal1/bert-small" (29M params)
    # - "distilbert-base-uncased" (66M params)

    # ====================================
    # 2. LOAD YOUR DATA from project folder
    # ====================================
    print("📊 Loading data...")

    # Assuming your data is in: fake_news_data/liar_dataset/
    data_path = "./fake_news_data/liar_dataset"

    try:
        # Try to load your actual data
        # Adjust this based on your data format
        train_data = pd.read_csv(f"{data_path}/train.tsv", sep='\t')
        test_data = pd.read_csv(f"{data_path}/test.tsv", sep='\t')

        # Your data should have 'statement' (text) and 'label' columns
        train_texts = train_data['statement'].tolist()
        train_labels = train_data['label'].apply(lambda x: 1 if x == 'fake' else 0).tolist()

        test_texts = test_data['statement'].tolist()
        test_labels = test_data['label'].apply(lambda x: 1 if x == 'fake' else 0).tolist()

    except Exception as e:
        print(f"⚠️ Couldn't load data: {e}")
        print("Creating dummy data for testing...")

        # Dummy data (100 samples)
        train_texts = ["This is real news."] * 50 + ["This is fake news!"] * 50
        train_labels = [0] * 50 + [1] * 50
        test_texts = ["Test real news."] * 10 + ["Test fake news!"] * 10
        test_labels = [0] * 10 + [1] * 10

    print(f"✅ Loaded {len(train_texts)} train, {len(test_texts)} test samples")

    # ====================================
    # 3. PREPARE DATASETS
    # ====================================
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })

    test_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": test_labels
    })

    # ====================================
    # 4. LOAD TINY MODEL & TOKENIZER
    # ====================================
    print(f"🤖 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # fake vs real
        ignore_mismatched_sizes=True
    )

    # Resize embeddings if we added new tokens
    if tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))

    # ====================================
    # 5. TOKENIZE DATA
    # ====================================
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128  # Short for tiny model
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Remove text column (keep only tokenized)
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    # Set format for PyTorch
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

       # ====================================
    # 6. TRAINING ARGUMENTS (Minimal) - FIXED
    # ====================================
    training_args = TrainingArguments(
        output_dir="./training_output",  # Temporary folder
        num_train_epochs=3,  # Just 3 epochs for quick training
        per_device_train_batch_size=16,  # Can be larger with tiny model
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        save_strategy="epoch",
        eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
        logging_steps=10,
        report_to="none",  # No wandb/tensorboard
        save_total_limit=1,  # Keep only best model
        load_best_model_at_end=True,
    )

    # ====================================
    # 7. TRAIN
    # ====================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    print("🎯 Training tiny BERT...")
    trainer.train()

       # ====================================
    # 8. SAVE MODEL IN PROJECT FOLDER - FIXED
    # ====================================
    print("\n💾 Saving trained model to project folder...")

    # Create models directory
    models_dir = "./trained_models"
    os.makedirs(models_dir, exist_ok=True)

    # Save path
    save_path = f"{models_dir}/bert_tiny_fake_news"

    # FIX 1: Save model explicitly (not just trainer.save_model)
    print(f"Saving model to: {save_path}")

    # Save using model's save_pretrained (more reliable)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Also save using trainer
    trainer.save_model(save_path)

    # FIX 2: Check ALL possible save locations
    print("\n🔍 Checking save locations...")

    possible_locations = [
        save_path,
        "./training_output",  # Original output_dir
        "./training_output/checkpoint-latest",  # Latest checkpoint
        f"./training_output/checkpoint-{len(train_dataset)}",  # Final checkpoint
    ]

    for location in possible_locations:
        weight_file = f"{location}/pytorch_model.bin"
        if os.path.exists(weight_file):
            size = os.path.getsize(weight_file) / 1024 / 1024
            print(f"✅ Found weights at: {location} ({size:.1f}MB)")

            # Copy to our desired location if not already there
            if location != save_path:
                print(f"📋 Copying from {location} to {save_path}")
                import shutil
                shutil.copytree(location, save_path, dirs_exist_ok=True)
            break
    else:
        print("❌ No weights found anywhere!")
        print("Creating a simple test file to debug...")

        # Create a dummy weight file for debugging
        os.makedirs(save_path, exist_ok=True)
        with open(f"{save_path}/debug.txt", "w") as f:
            f.write("Training completed but weights not saved properly\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Epochs: 3\n")
            f.write(f"Samples: {len(train_dataset)}\n")
      # ====================================
    # 9. CREATE SIMPLE INFERENCE SCRIPT (ADD THIS)
    # ====================================
    inference_code = '''
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class TinyFakeNewsDetector:
    def __init__(self, model_path="./trained_models/bert_tiny_fake_news"):
        """Load the trained tiny BERT model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode

    def predict(self, text):
        """Predict if text is fake news"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)

        # Return probabilities
        return {
            "fake": float(probabilities[0][1]),
            "real": float(probabilities[0][0]),
            "prediction": "FAKE" if probabilities[0][1] > 0.5 else "REAL"
        }

# Quick test
if __name__ == "__main__":
    detector = TinyFakeNewsDetector()
    test_text = "BREAKING: Aliens land in Washington!"
    result = detector.predict(test_text)
    print(f"Text: {test_text}")
    print(f"Prediction: {result['prediction']}")
    print(f"Fake probability: {result['fake']:.2%}")
'''

    # Save inference script
    with open(f"{models_dir}/inference.py", "w") as f:
        f.write(inference_code)

    # ====================================
    # 10. VERIFY & DOWNLOAD (ADD THIS)
    # ====================================
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)

    # Check file sizes
    weight_file = f"{save_path}/pytorch_model.bin"
    if os.path.exists(weight_file):
        size_mb = os.path.getsize(weight_file) / 1024 / 1024
        print(f"\n📦 Model size: {size_mb:.1f}MB (Tiny!)")
        print(f"📍 Saved to: {save_path}")
    else:
        print("❌ Warning: Weight file not found!")

    return save_path

#@title Step 5: RUN TRAINING
save_path = train_minimal_bert()

#@title Step 6: Download Entire Project
print("\n" + "="*50)
print("📦 DOWNLOAD YOUR PROJECT")
print("="*50)

# Create zip of entire project
import shutil
from google.colab import files

zip_path = "/content/fake_news_project_complete.zip"
shutil.make_archive(zip_path.replace('.zip', ''), 'zip', '/content/fake-news-project')

print(f"✅ Project zipped: {zip_path}")
print(f"📁 Size: {os.path.getsize(zip_path) / 1024 / 1024:.1f}MB")

# Download
print("\nClick below to download or run:")
print("files.download('/content/fake_news_project_complete.zip')")

# Auto-download (uncomment if you want)
# files.download(zip_path)
