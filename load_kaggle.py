# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
import os
import glob

print("Downloading dataset... (this may take a while for large datasets)")
# Download the dataset
# This returns the local path where the dataset is cached
dataset_path = kagglehub.dataset_download("deathtrooper/multichannel-glaucoma-benchmark-dataset")

print(f"Dataset downloaded to: {dataset_path}")

# Search for CSV files recursively in the downloaded dataset
csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)

if csv_files:
    # Use the first CSV found, or modify this logic to select a specific one
    target_file = csv_files[0] 
    print(f"Found {len(csv_files)} CSV file(s). Loading the first one: {target_file}")
    
    df = pd.read_csv(target_file)
    print("First 5 records:")
    print(df.head())
else:
    print("No CSV files found in the downloaded dataset.")
    print("All files in root folder:", os.listdir(dataset_path))