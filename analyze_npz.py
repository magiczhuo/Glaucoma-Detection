import numpy as np
import os
import glob
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def analyze_npz(folder_path):
    # Find all val_predictions npz files
    files = glob.glob(os.path.join(folder_path, 'val_predictions_epoch_*.npz'))
    if not files:
        print(f"No npz files found in {folder_path}")
        return

    # Sort by epoch number
    files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.npz')[0]))
    
    # Pick the last one (latest epoch)
    latest_file = files[-1]
    print(f"Analyzing most recent training validation file: {latest_file}")
    
    try:
        data = np.load(latest_file)
        if 'y_pred_logit' in data:
            logits = data['y_pred_logit']
            probs = sigmoid(logits)
            labels = data['y_true']
            
            print(f"\nStats for {len(probs)} samples:")
            print(f"Mean Probability: {np.mean(probs):.4f}")
            print(f"Min Probability:  {np.min(probs):.4f}")
            print(f"Max Probability:  {np.max(probs):.4f}")
            
            # Show a few examples
            print("\nSample Predictions (first 10):")
            for i in range(min(10, len(probs))):
                print(f"Sample {i}: Label={labels[i]}, Logit={logits[i]:.4f}, Prob={probs[i]:.4f}")
                
            # Distribution check
            high_conf = np.sum((probs > 0.9) | (probs < 0.1))
            print(f"\nNumber of high confidence predictions (>0.9 or <0.1): {high_conf} / {len(probs)}")
            
        else:
            print("Could not find 'y_pred_logit' in npz file keys:", list(data.keys()))
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    folder = 'checkpoints/1-resnet152rcbam-3b-f12'
    analyze_npz(folder)
