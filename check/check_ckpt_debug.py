import torch
import sys
import os
from networks.trainer import Patch5Model, Branch2CBAM, Branch3CBAM

def check_keys():
    # Path from test.sh
    model_path = 'checkpoints/3_resnet/model_epoch_best.pth' 
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    print(f"Loading checkpoint from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    if 'model' not in state_dict:
        print("Key 'model' not found in state_dict. Available keys:", state_dict.keys())
        ckpt_keys = list(state_dict.keys())
    else:
        print("Found 'model' key in state_dict.")
        ckpt_keys = list(state_dict['model'].keys())

    print(f"\nExample checkpoint keys (first 5): {ckpt_keys[:5]}")
    
    # Initialize model - 3branch uses Patch5Model(1) according to eval.py
    model = Patch5Model(1)
    model_keys = list(model.state_dict().keys())
    print(f"\nExample model keys (first 5): {model_keys[:5]}")

    # Check for mismatch
    has_module = any(k.startswith('module.') for k in ckpt_keys)
    model_has_module = any(k.startswith('module.') for k in model_keys)

    print(f"\nCheckpoint has 'module.' prefix: {has_module}")
    print(f"Model has 'module.' prefix: {model_has_module}")
    
    common_keys = set(ckpt_keys).intersection(set(model_keys))
    print(f"Common keys count (direct match): {len(common_keys)}")
    
    # Check match without module prefix
    ckpt_keys_clean = [k[7:] if k.startswith('module.') else k for k in ckpt_keys]
    common_clean = set(ckpt_keys_clean).intersection(set(model_keys))
    print(f"Common keys count if 'module.' removed from checkpoint: {len(common_clean)}")
    

    print(f"Total model keys: {len(model_keys)}")
    print(f"Total checkpoint keys: {len(ckpt_keys)}")
    
    # Check for ResNet depth
    ckpt_layer3_keys = [k for k in ckpt_keys if 'layer3' in k]
    model_layer3_keys = [k for k in model_keys if 'layer3' in k]
    print(f"\nCheckpoint layer3 keys count: {len(ckpt_layer3_keys)}")
    print(f"Model layer3 keys count: {len(model_layer3_keys)}")
    
    # Check for specific blocks (Resnet50 has 6 blocks in layer3: 0..5. Resnet152 has 36: 0..35)
    ckpt_has_block6 = any('layer3.6' in k for k in ckpt_layer3_keys)
    model_has_block6 = any('layer3.6' in k for k in model_layer3_keys)
    print(f"Checkpoint has layer3.6: {ckpt_has_block6}")
    print(f"Model has layer3.6: {model_has_block6}")

    if ckpt_has_block6 != model_has_block6:
        print("CRITICAL MISMATCH: Architecture depth difference (ResNet50 vs ResNet152 likely).")

if __name__ == "__main__":
    check_keys()
