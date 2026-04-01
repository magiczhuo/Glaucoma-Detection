import pandas as pd
import os
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

def get_dataset_name(filepath):
    filename = os.path.basename(filepath)
    # Mapping based on prefixes
    if filename.startswith('REFUGE'):
        return 'REFUGE'
    elif filename.startswith('G1020'):
        return 'G1020'
    elif filename.startswith('PAPILA'):
        return 'PAPILA'
    elif filename.startswith('DRISHTI'):
        return 'DRISHTI'
    elif filename.startswith('ORIGA'):
        return 'ORIGA'
    elif filename.startswith('BEH'):
        return 'BEH'
    elif filename.startswith('EyePACS'):
        return 'EyePACS'
    elif filename.startswith('OIA'):
        return 'OIA-ODIR'
    elif filename.startswith('FIVES'):
        return 'FIVES'
    elif filename.startswith('sjchoi86'):
        return 'sjchoi-HRF'
    elif filename.startswith('HRF'):
        return 'HRF' # Or merge with sjchoi-HRF if needed, kept separate for now based on stats
    elif filename.startswith('CRFO'):
        return 'CRFO'
    elif filename.startswith('JSIEC'):
        return 'JSIEC'
    elif filename.startswith('LES'):
        return 'LES'
    elif filename.startswith('DR'):
        # DR might be DRISHTI or separate, but based on grep stats 'DR: 10', likely distinct or typo
        return 'DR'
    else:
        # Fallback: extract first part before '-' or '_'
        if '-' in filename:
            return filename.split('-')[0]
        elif '_' in filename:
            return filename.split('_')[0]
        return 'Unknown'

def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy per dataset from prediction CSV.')
    parser.add_argument('--csv_path', type=str, default='results/3b-rcbam-smdg/prediction_test.csv', help='Path to the prediction CSV file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the result CSV')
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Reading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    required_cols = ['filename', 'gt', 'pred_label']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV.")
            return

    # Add Dataset column
    df['dataset'] = df['filename'].apply(get_dataset_name)
    
    # Calculate metrics per dataset
    results = []
    
    # Overall metrics
    overall_acc = accuracy_score(df['gt'], df['pred_label'])
    results.append({
        'Dataset': 'Overall',
        'Count': len(df),
        'Accuracy': overall_acc
    })
    
    datasets = sorted(df['dataset'].unique())
    print("-" * 60)
    print(f"{'Dataset':<20} | {'Count':<10} | {'Accuracy':<10}")
    print("-" * 60)
    
    print(f"{'Overall':<20} | {len(df):<10} | {overall_acc:.4f}")
    
    for ds in datasets:
        subset = df[df['dataset'] == ds]
        acc = accuracy_score(subset['gt'], subset['pred_label'])
        
        # Optional: Calculate additional metrics if probability columns exist
        auc = 'N/A'
        f1 = f1_score(subset['gt'], subset['pred_label'], average='macro')
        
        # Check for probability columns (prob_1 or prob) for AUC
        if 'prob_1' in subset.columns:
            try:
                # Need at least one sample of each class for robust ROC AUC, or handle exception
                if len(subset['gt'].unique()) > 1:
                    auc = roc_auc_score(subset['gt'], subset['prob_1'])
                else:
                    auc = 0.5 # Default or avoid calc
            except:
                pass
        
        results.append({
            'Dataset': ds,
            'Count': len(subset),
            'Accuracy': acc,
            'F1': f1,
            'AUC': auc
        })
        
        print(f"{ds:<20} | {len(subset):<10} | {acc:.4f}")

    print("-" * 60)
    
    # Save results
    if args.output_path:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_path, index=False)
        print(f"Results saved to {args.output_path}")
    else:
        # Save to the same directory as input csv with suffix
        base_dir = os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path)
        out_name = os.path.splitext(base_name)[0] + '_per_dataset.csv'
        out_path = os.path.join(base_dir, out_name)
        out_df = pd.DataFrame(results)
        out_df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

if __name__ == '__main__':
    main()
