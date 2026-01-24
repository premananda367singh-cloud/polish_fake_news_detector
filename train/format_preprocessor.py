import os
from pathlib import Path
import pandas as pd
path = "/home/premananda/Documents/fake_news_detector/train/datasets/raw"
tsv_path = "/home/premananda/Documents/fake_news_detector/train/datasets/processed"
os.makedirs(tsv_path, exist_ok=True)

def getfile_name(path):
    files = os.listdir(path)
    return files
print(f"files found at {path}")
data_files = getfile_name(path)
print(data_files)

for filename in data_files:
    if filename.endswith('.csv'):
        full_path = os.path.join(path, filename)
        df = pd.read_csv(full_path)
        tsv_name = filename.replace('.csv', '.tsv')
        tsv_full_path = os.path.join(tsv_path, tsv_name)
        df.to_csv(tsv_full_path, sep = '\t', index=False)
        
    elif filename.endswith('.tsv'):
        full_path = os.path.join(path, filename)
        df = pd.read_csv(full_path, sep = '\t')
        tsv_full_path = os.path.join(tsv_path, filename)
        df.to_csv(tsv_full_path, sep = '\t', index=False)
    else:
        continue
   