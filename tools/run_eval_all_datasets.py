import os
import subprocess
import glob
from tqdm import tqdm
import pandas as pd

# Configurations
EVAL_SCRIPT = 'eval_v2.py' # Assuming eval_v2.py is in the root
MODEL_NAME = '3branch-rcbam' # Adjust if needed (e.g., 3branch-cbam)
MODEL_PATH = 'checkpoints/1-resnet152rcbam-3b-f12/model_epoch_best.pth' # Validate this path
BASE_NAME_PREFIX = '3b-rcbam-smdg-per'
DATASETS_ROOT = '/root/ZYZ/GRINLAB/SMDG_per_dataset'

def main():
    # Find all dataset folders
    dataset_paths = glob.glob(os.path.join(DATASETS_ROOT, '*'))
    datasets = [os.path.basename(p) for p in dataset_paths if os.path.isdir(p)]
    datasets = sorted(datasets)
    
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    results_summary = []

    for dataset in tqdm(datasets, desc="Evaluating Datasets"):
        print(f"\n[{dataset}] Starting evaluation...")
        
        dataroot = os.path.join(DATASETS_ROOT, dataset)
        experiment_name = f"{BASE_NAME_PREFIX}-{dataset}"
        
        # Command construction
        # Note: eval_v2.py appends the testset name (e.g., 'test') to dataroot.
        # So if our data is at .../OIA-ODIR/test, and we pass .../OIA-ODIR,
        # eval_v2.py will look for .../OIA-ODIR/test. Correct.
        
        cmd = [
            'python', EVAL_SCRIPT,
            '--model_name', MODEL_NAME,
            '--model_path', MODEL_PATH,
            '--name', experiment_name,
            '--dataroot', dataroot,
            '--test_threshold', '0.5',
            '--testsets', 'test',
            '--isRecord'
            # Add --mode 3cls if needed (default seems to check n_output based on mode?)
            # The default mode in TestOptions usually is 'binary' or needs check.
            # Assuming default or 'binary' unless specified. previous runs used 3cls for some?
            # Let's check TestOptions default. But usually binary is default.
        ]
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
            
            # Read result
            result_csv = os.path.join('results', experiment_name, 'result.csv')
            if os.path.exists(result_csv):
                df = pd.read_csv(result_csv)
                # Assuming simple structure, maybe 'Accuracy' column or similar
                # eval_v2 saves transposed results? No, rows are metrics? Or columns?
                # Based on eval_v2: result_df = pd.DataFrame(results). 
                # results[val] = result (dict)
                # So indices are metrics, columns are testsets (e.g., 'test')
                
                acc = df.loc['Accuracy', 'test'] if 'Accuracy' in df.index and 'test' in df.columns else 'N/A'
                auc = df.loc['AUC', 'test'] if 'AUC' in df.index and 'test' in df.columns else 'N/A'
                
                results_summary.append({
                    'Dataset': dataset,
                    'Accuracy': acc,
                    'AUC': auc,
                    'Path': result_csv
                })
                print(f"[{dataset}] Finished. Acc: {acc}, AUC: {auc}")
            else:
                 print(f"[{dataset}] Warning: result.csv not found.")
                 results_summary.append({'Dataset': dataset, 'Status': 'Missing Results'})
                 
        except subprocess.CalledProcessError as e:
            print(f"[{dataset}] Error during evaluation: {e}")
            results_summary.append({'Dataset': dataset, 'Status': 'Failed'})

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_csv_path = 'results/smdg_per_dataset_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nAll evaluations completed. Summary saved to {summary_csv_path}")
    print(summary_df)

if __name__ == '__main__':
    main()
