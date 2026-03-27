import os
import shutil
import glob
from tqdm import tqdm
import pandas as pd

# Define source directory
SOURCE_ROOT = '/root/ZYZ/GRINLAB/SMDG_test/test'
DEST_ROOT = '/root/ZYZ/GRINLAB/SMDG_per_dataset'

# Define dataset prefixes mapping
def get_dataset_name(filename):
    # Mapping logic same as calc_metrics_per_dataset.py
    if filename.startswith('REFUGE'): return 'REFUGE'
    if filename.startswith('G1020'): return 'G1020'
    if filename.startswith('PAPILA'): return 'PAPILA'
    if filename.startswith('DRISHTI'): return 'DRISHTI'
    if filename.startswith('ORIGA'): return 'ORIGA'
    if filename.startswith('BEH'): return 'BEH'
    if filename.startswith('EyePACS'): return 'EyePACS'
    if filename.startswith('OIA'): return 'OIA-ODIR'
    if filename.startswith('FIVES'): return 'FIVES'
    if filename.startswith('sjchoi86'): return 'sjchoi-HRF'
    if filename.startswith('HRF'): return 'HRF'
    if filename.startswith('CRFO'): return 'CRFO'
    if filename.startswith('JSIEC'): return 'JSIEC'
    if filename.startswith('LES'): return 'LES'
    if filename.startswith('DR'): return 'DR'
    
    # Fallback
    if '-' in filename: return filename.split('-')[0]
    if '_' in filename: return filename.split('_')[0]
    return 'Unknown'

def main():
    if not os.path.exists(DEST_ROOT):
        os.makedirs(DEST_ROOT)
        print(f"Created destination root: {DEST_ROOT}")
    
    # Process Relabel CSV if exists
    relabel_csv_path = os.path.join(SOURCE_ROOT, 'smdg_relabel.csv')
    df_relabel = None
    if os.path.exists(relabel_csv_path):
        print(f"Reading Relabel CSV: {relabel_csv_path}")
        df_relabel = pd.read_csv(relabel_csv_path)
    
    # Folders to process
    subfolders = ['0_neg', '1_pos', '0_roi_800_clahe', '1_roi_800_clahe']
    
    # Helper to ensure destination dataset structure
    dataset_paths = {}

    for folder in subfolders:
        src_folder_path = os.path.join(SOURCE_ROOT, folder)
        if not os.path.exists(src_folder_path):
            print(f"Skipping missing folder: {src_folder_path}")
            continue
            
        files = [f for f in os.listdir(src_folder_path) if not f.startswith('.')]
        print(f"Processing {folder}: {len(files)} files...")
        
        for filename in tqdm(files):
            dataset_name = get_dataset_name(filename)
            
            # Destination path: SMDG_per_dataset/{DATASET_NAME}/test/{folder}
            dest_dataset_root = os.path.join(DEST_ROOT, dataset_name, 'test')
            dest_folder_path = os.path.join(dest_dataset_root, folder)
            
            if dest_folder_path not in dataset_paths:
                os.makedirs(dest_folder_path, exist_ok=True)
                dataset_paths[dest_folder_path] = True
            
            src_file = os.path.join(src_folder_path, filename)
            dest_file = os.path.join(dest_folder_path, filename)
            
            # Copy (or link to save space)
            if not os.path.exists(dest_file):
                try:
                    # Try hardlink first
                    os.link(src_file, dest_file)
                except OSError:
                    shutil.copy2(src_file, dest_file)

    # Split Relabel CSV per dataset
    if df_relabel is not None:
        print("Splitting relabel CSV...")
        df_relabel['dataset'] = df_relabel['x'].apply(get_dataset_name)
        datasets = df_relabel['dataset'].unique()
        
        for ds in datasets:
            subset = df_relabel[df_relabel['dataset'] == ds]
            # Save to: SMDG_per_dataset/{DATASET_NAME}/test/smdg_relabel.csv
            dest_csv_path = os.path.join(DEST_ROOT, ds, 'test', 'smdg_relabel.csv')
            subset[['x', 'y']].to_csv(dest_csv_path, index=False)
            print(f"Saved relabel CSV for {ds}: {dest_csv_path}")

    print("Done splitting datasets.")

if __name__ == '__main__':
    main()
