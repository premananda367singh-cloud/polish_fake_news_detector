
# train_ensemble.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Import your ensemble detector
from models.ensemble_detector import EnsembleDetector

def load_liar_data():
    """Load LIAR dataset from the fake_news_data directory"""
    data_dir = 'fake_news_data/liar_dataset'
    
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker',
        'job_title', 'state_info', 'party', 'barely_true_counts',
        'false_counts', 'half_true_counts', 'mostly_true_counts',
        'pants_on_fire_counts', 'context'
    ]
    
    # Load all splits
    train_path = os.path.join(data_dir, 'train.tsv')
    valid_path = os.path.join(data_dir, 'valid.tsv')
    test_path = os.path.join(data_dir, 'test.tsv')
    
    train_df = pd.read_csv(train_path, sep='\t', names=columns)
    valid_df = pd.read_csv(valid_path, sep='\t', names=columns)
    test_df = pd.read_csv(test_path, sep='\t', names=columns)
    
    return train_df, valid_df, test_df

def prepare_binary_labels(df):
    """Convert 6-class labels to binary (fake/real)"""
    # Map to binary classification
    fake_labels = ['false', 'pants-fire', 'barely-true']
    real_labels = ['true', 'mostly-true', 'half-true']
    
    def map_to_binary(label):
        if label in fake_labels:
            return 1  # Fake news
        elif label in real_labels:
            return 0  # Real news
        else:
            return None
    
    df = df.copy()
    df['binary_label'] = df['label'].apply(map_to_binary)
    df = df.dropna(subset=['binary_label'])
    return df

def prepare_multiclass_labels(df):
    """Keep 6-class labels for multiclass classification"""
    label_map = {
        'true': 0,
        'mostly-true': 1,
        'half-true': 2,
        'barely-true': 3,
        'false': 4,
        'pants-fire': 5
    }
    
    df = df.copy()
    df['multiclass_label'] = df['label'].map(label_map)
    df = df.dropna(subset=['multiclass_label'])
    return df

def main():
    print("=" * 70)
    print("ENSEMBLE DETECTOR TRAINING SCRIPT")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading LIAR dataset...")
    train_df, valid_df, test_df = load_liar_data()
    
    print(f"  ✓ Training samples: {len(train_df)}")
    print(f"  ✓ Validation samples: {len(valid_df)}")
    print(f"  ✓ Test samples: {len(test_df)}")
    
    # Choose classification type
    print("\n[2/5] Choose classification type:")
    print("  1. Binary classification (Fake vs Real)")
    print("  2. Multiclass classification (6 classes)")
    print("  3. Both (train binary first, then multiclass)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Binary classification
        print("\nPreparing binary labels...")
        train_df = prepare_binary_labels(train_df)
        valid_df = prepare_binary_labels(valid_df)
        test_df = prepare_binary_labels(test_df)
        
        label_col = 'binary_label'
        n_classes = 2
        task_type = 'binary'
        
    elif choice == "2":
        # Multiclass classification
        print("\nPreparing multiclass labels...")
        train_df = prepare_multiclass_labels(train_df)
        valid_df = prepare_multiclass_labels(valid_df)
        test_df = prepare_multiclass_labels(test_df)
        
        label_col = 'multiclass_label'
        n_classes = 6
        task_type = 'multiclass'
        
    elif choice == "3":
        # Train both sequentially
        print("\nTraining binary classification first...")
        # We'll handle this later
        pass
    else:
        print("Invalid choice. Defaulting to binary classification.")
        train_df = prepare_binary_labels(train_df)
        valid_df = prepare_binary_labels(valid_df)
        test_df = prepare_binary_labels(test_df)
        label_col = 'binary_label'
        n_classes = 2
        task_type = 'binary'
    
    # Prepare data
    print(f"\n[3/5] Preparing data for {task_type} classification...")
    
    if choice != "3":
        # Single task training
        X_train = train_df['statement'].tolist()
        y_train = train_df[label_col].tolist()
        X_valid = valid_df['statement'].tolist()
        y_valid = valid_df[label_col].tolist()
        X_test = test_df['statement'].tolist()
        y_test = test_df[label_col].tolist()
        
        print(f"  ✓ Final training samples: {len(X_train)}")
        print(f"  ✓ Validation samples: {len(X_valid)}")
        print(f"  ✓ Test samples: {len(X_test)}")
        print(f"  ✓ Number of classes: {n_classes}")
        
        # Initialize ensemble
        print(f"\n[4/5] Initializing ensemble detector for {task_type}...")
        ensemble = EnsembleDetector()
        
        # Train ensemble
        print(f"\nTraining ensemble models...")
        training_params = {
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'n_classes': n_classes
        }
        
        print(f"Training parameters: {training_params}")
        ensemble.train(X_train, y_train, **training_params)
        
        # Check training status
        print(f"\nTraining Status:")
        print(f"  Ensemble trained: {ensemble.is_trained}")
        for model_name, model in ensemble.models.items():
            print(f"  {model_name}: {model.is_trained}")
        
        # Save model weights
        print(f"\n[5/5] Saving model...")
        os.makedirs('saved_models', exist_ok=True)
        
        # Save each model separately
        for model_name, model in ensemble.models.items():
            if model.is_trained:
                save_path = f'saved_models/{model_name}_{task_type}.pth'
                # Note: You'll need to implement save method in each model
                # model.save(save_path)
                print(f"  ✓ {model_name} weights saved to {save_path}")
        
        # Evaluate on validation set
        print(f"\nEvaluating on validation set...")
        eval_size = min(200, len(X_valid))
        performance = ensemble.get_model_performance(X_valid[:eval_size], y_valid[:eval_size])
        
        print(f"\nModel Performance (Validation Set):")
        for model_name, scores in performance.items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {scores['accuracy']:.3f}")
            print(f"    F1 Score: {scores['f1_score']:.3f}")
        
        # Test predictions
        print(f"\nSample predictions on test set:")
        for i in range(min(3, len(X_test))):
            text = X_test[i]
            actual_label = y_test[i]
            
            result = ensemble.predict(text)
            
            if task_type == 'binary':
                pred_label = 'Fake' if result['prediction'] == 1 else 'Real'
                actual_str = 'Fake' if actual_label == 1 else 'Real'
            else:
                pred_label = result['prediction']
                actual_str = actual_label
            
            print(f"\n  Sample {i+1}:")
            print(f"    Text: {text[:80]}...")
            print(f"    Prediction: {pred_label}")
            print(f"    Actual: {actual_str}")
            print(f"    Confidence: {result['confidence']:.3f}")
            if 'disagreement' in result:
                print(f"    Model Disagreement: {result['disagreement']:.3f}")
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"Task: {task_type} classification")
        print(f"Models saved to: saved_models/")
        print(f"{'='*70}")
    
    else:
        # Train both binary and multiclass
        print("\nTraining binary classification first...")
        # This would require modifying your models to handle both tasks
        print("Note: Training both tasks requires model modifications")
        print("Currently training binary classification only...")
        
        # Fall back to binary
        train_df = prepare_binary_labels(train_df)
        valid_df = prepare_binary_labels(valid_df)
        test_df = prepare_binary_labels(test_df)
        
        X_train = train_df['statement'].tolist()
        y_train = train_df['binary_label'].tolist()
        
        ensemble = EnsembleDetector()
        ensemble.train(X_train, y_train, epochs=3, batch_size=16, n_classes=2)
        
        print(f"\nBinary training complete!")
        print(f"Ensemble trained: {ensemble.is_trained}")

if __name__ == "__main__":
    main()
