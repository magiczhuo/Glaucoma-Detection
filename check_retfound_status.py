import os
import torch
import glob
import numpy as np
from pathlib import Path
import sys

# Append path to import loader
sys.path.append(os.getcwd())
try:
    from RETFound_Feature_Loader import RETFoundFeatureLoader
except ImportError:
    # Mock if not found in path, though it is in root
    print("Warning: Could not import RETFound_Feature_Loader, defining mock")
    class RETFoundFeatureLoader:
        def __init__(self):
            self.feature_root = os.path.abspath("retfound_features")
        def get_feature(self, img_path, dataset_root):
             # Simplified logic from file
            base_root = os.path.dirname(dataset_root.rstrip(os.sep))
            rel_path = os.path.relpath(img_path, base_root)
            feat_rel_path = os.path.splitext(rel_path)[0] + '.pt'
            feat_path = os.path.join(self.feature_root, feat_rel_path)
            if os.path.exists(feat_path):
                return feat_path, torch.load(feat_path, map_location='cpu')
            return feat_path, None

def check_status():
    print("=== 1. Checking Directory Structure ===")
    dataset_test_root = os.path.abspath("dataset/val")
    feature_root = os.path.abspath("retfound_features")
    
    if not os.path.exists(dataset_test_root):
        print(f"Dataset root not found: {dataset_test_root}")
        return
    if not os.path.exists(feature_root):
        print(f"Feature root not found: {feature_root}")
        return

    print("Directories exist.")
    
    print("\n=== 2. Checking Feature Consistency ===")
    loader = RETFoundFeatureLoader()
    
    # Check 0_neg and 1_pos
    for cls_folder in ['0_neg', '1_pos']:
        img_dir = os.path.join(dataset_test_root, cls_folder)
        if not os.path.exists(img_dir):
            print(f"Warning: Image dir {img_dir} does not exist")
            continue
            
        images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        print(f"\nChecking class {cls_folder}: found {len(images)} images")
        
        if len(images) == 0:
            continue
            
        # Check first 5 images
        for img_path in images[:5]:
            try:
                # Real loader returns only the tensor
                feat = loader.get_feature(img_path, dataset_test_root)
                # Reconstruct path for display manually for debugging info
                # Assuming loader.feature_root exists
                if hasattr(loader, 'feature_root'):
                    base_root = os.path.dirname(dataset_test_root.rstrip(os.sep))
                    rel_path = os.path.relpath(img_path, base_root)
                    feat_rel_path = os.path.splitext(rel_path)[0] + '.pt'
                    expected_feat_path = os.path.join(loader.feature_root, feat_rel_path)
                else:
                    expected_feat_path = "Unknown (loader.feature_root missing)"
                    
            except Exception as e:
                print(f"Error calling loader: {e}")
                feat = None
                expected_feat_path = "Error"
            
            print(f"Image: {os.path.basename(img_path)}")
            print(f"  -> Expect Feature: {expected_feat_path}")
            
            if feat is None:
                print("  -> ERROR: Feature file NOT FOUND")
            else:
                is_zero = torch.all(feat == 0)
                mean_val = feat.mean().item()
                std_val = feat.std().item()
                shape = feat.shape
                
                status = "INVALID (All Zeros)" if is_zero else "VALID"
                print(f"  -> Status: {status} | Shape: {shape} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")

if __name__ == "__main__":
    check_status()
