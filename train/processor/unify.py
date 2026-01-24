#!/usr/bin/env python3
"""
Comprehensive Fake News Dataset Unifier
Unifies LIAR, PolitiFact, and GossipCop datasets into a single TSV file
with text and label columns only
"""

import pandas as pd
from pathlib import Path
import sys

def process_liar_dataset(filepath, dataset_name):
    """
    Process LIAR dataset (train.tsv, valid.tsv)
    Format: id, label, statement, subject, speaker, job, state, party, counts..., context
    Column 2 is the statement (text)
    """
    print(f"\n📂 Processing {dataset_name}: {filepath}")
    
    # LIAR has no header, columns are:
    # 0: id, 1: label, 2: statement, 3: subject, 4: speaker, etc.
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    print(f"   Loaded {len(df)} records")
    
    # Extract text (column 2) and label (column 1)
    unified = pd.DataFrame({
        'text': df[2].astype(str),
        'label': df[1].astype(str).str.lower().str.strip(),
        'source': dataset_name
    })
    
    # Standardize labels
    label_map = {
        'pants-fire': 'fake',
        'false': 'fake',
        'barely-true': 'fake',
        'half-true': 'fake',  # You can change this to 'neutral' if you want 3 classes
        'mostly-true': 'true',
        'true': 'true'
    }
    
    unified['label'] = unified['label'].map(label_map)
    
    print(f"   Label distribution:")
    print(unified['label'].value_counts().to_string().replace('\n', '\n   '))
    
    return unified

def process_fakereal_dataset(filepath, label, dataset_name):
    """
    Process PolitiFact/GossipCop datasets
    Format: id, news_url, title, tweet_ids
    Column 'title' is the text
    """
    print(f"\n📂 Processing {dataset_name}: {filepath}")
    
    # These files have headers
    df = pd.read_csv(filepath, sep='\t')
    
    print(f"   Loaded {len(df)} records")
    
    # Check if 'title' column exists
    if 'title' not in df.columns:
        print(f"   ⚠️  Warning: 'title' column not found. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Extract text (title) and assign label
    unified = pd.DataFrame({
        'text': df['title'].astype(str),
        'label': label,
        'source': dataset_name
    })
    
    # Remove empty or NaN texts
    unified = unified[unified['text'].str.strip() != '']
    unified = unified[unified['text'] != 'nan']
    
    print(f"   Kept {len(unified)} non-empty records")
    
    return unified

def unify_all_datasets(input_dir='raw', output_dir='processed'):
    """
    Main function to unify all datasets
    """
    print("=" * 70)
    print("🔄 COMPREHENSIVE FAKE NEWS DATASET UNIFIER")
    print("=" * 70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_dataframes = []
    
    # Define all datasets to process
    datasets_config = [
        # LIAR datasets
        {'file': 'train.tsv', 'type': 'liar', 'name': 'liar_train'},
        {'file': 'valid.tsv', 'type': 'liar', 'name': 'liar_valid'},
        
        # PolitiFact datasets
        {'file': 'politifact_fake.tsv', 'type': 'labeled', 'label': 'fake', 'name': 'politifact_fake'},
        {'file': 'politifact_real.tsv', 'type': 'labeled', 'label': 'true', 'name': 'politifact_real'},
        
        # GossipCop datasets
        {'file': 'gossipcop_fake.tsv', 'type': 'labeled', 'label': 'fake', 'name': 'gossipcop_fake'},
        {'file': 'gossipcop_real.tsv', 'type': 'labeled', 'label': 'true', 'name': 'gossipcop_real'},
    ]
    
    # Process each dataset
    for config in datasets_config:
        filepath = input_path / config['file']
        
        if not filepath.exists():
            print(f"\n⚠️  Skipping {config['file']} (not found)")
            continue
        
        try:
            if config['type'] == 'liar':
                df = process_liar_dataset(filepath, config['name'])
            elif config['type'] == 'labeled':
                df = process_fakereal_dataset(filepath, config['label'], config['name'])
            
            if len(df) > 0:
                all_dataframes.append(df)
        
        except Exception as e:
            print(f"   ❌ Error processing {config['file']}: {e}")
            continue
    
    # Combine all datasets
    if not all_dataframes:
        print("\n❌ No datasets were successfully processed!")
        return None
    
    print(f"\n" + "=" * 70)
    print("🔧 COMBINING ALL DATASETS")
    print("=" * 70)
    
    unified_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates based on text
    print(f"\n📊 Before deduplication: {len(unified_df):,} records")
    unified_df = unified_df.drop_duplicates(subset=['text'], keep='first')
    print(f"📊 After deduplication:  {len(unified_df):,} records")
    
    # Shuffle the data
    unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save with source column
    output_file_with_source = output_path / 'unified_all_with_source.tsv'
    unified_df.to_csv(output_file_with_source, sep='\t', index=False)
    
    # Save without source column (just text and label)
    unified_simple = unified_df[['text', 'label']].copy()
    output_file_simple = output_path / 'unified_all.tsv'
    unified_simple.to_csv(output_file_simple, sep='\t', index=False)
    
    # Print final statistics
    print(f"\n" + "=" * 70)
    print("✅ UNIFICATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Output files:")
    print(f"   1. {output_file_simple}")
    print(f"      → Simple format (text, label)")
    print(f"   2. {output_file_with_source}")
    print(f"      → With source tracking (text, label, source)")
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total records: {len(unified_df):,}")
    
    print(f"\n📈 Label Distribution:")
    label_counts = unified_df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(unified_df)) * 100
        print(f"   {label:10s}: {count:6,} ({percentage:5.2f}%)")
    
    print(f"\n📚 Source Distribution:")
    source_counts = unified_df['source'].value_counts()
    for source, count in source_counts.items():
        percentage = (count / len(unified_df)) * 100
        print(f"   {source:20s}: {count:6,} ({percentage:5.2f}%)")
    
    print(f"\n📝 Sample Records:")
    print("-" * 70)
    for idx, row in unified_df.head(5).iterrows():
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        print(f"[{row['label'].upper():4s}] [{row['source']:20s}] {text_preview}")
    
    print("\n" + "=" * 70)
    
    return unified_df

def main():
    """
    Main entry point - can be called with command line args or directly
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Unify all fake news datasets')
    parser.add_argument('-i', '--input', default='raw', 
                       help='Input directory containing TSV files (default: raw)')
    parser.add_argument('-o', '--output', default='processed',
                       help='Output directory (default: processed)')
    
    args = parser.parse_args()
    
    try:
        df = unify_all_datasets(args.input, args.output)
        if df is not None:
            print("\n✅ Success! Your unified datasets are ready.")
            sys.exit(0)
        else:
            print("\n❌ Failed to create unified dataset.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()