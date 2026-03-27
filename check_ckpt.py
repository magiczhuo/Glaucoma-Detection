import torch
import os
import sys

# Mocking the path
sys.path.append('./data')

checkpoint_path = './checkpoints/1-resnet152cbam-3b-3cls/model_epoch_best.pth'

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found: {checkpoint_path}")
    # Try looking for other checkpoints
    checkpoints_dir = './checkpoints/1-resnet152cbam-3b-3cls'
    if os.path.exists(checkpoints_dir):
        print(f"Contents of {checkpoints_dir}: {os.listdir(checkpoints_dir)}")
    sys.exit(1)

try:
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print("Keys in state_dict:")
    if 'model' in state_dict:
        model_state = state_dict['model']
        keys = list(model_state.keys())
        print(f"Total keys: {len(keys)}")
        print("First 5 keys:", keys[:5])
        
        has_module = any(k.startswith('module.') for k in keys)
        print(f"Has 'module.' prefix: {has_module}")
    else:
        print("state_dict does not contain 'model' key")
        print("Top level keys:", state_dict.keys())

except Exception as e:
    print(f"Error loading checkpoint: {e}")
