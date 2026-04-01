import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from openTSNE import TSNE

# Add necessary paths
sys.path.append('./data')

from options.train_options import TrainOptions
from data import create_dataloader
from RETFound_Feature_Loader import RETFoundFeatureLoader
from networks.trainer import Trainer

# Global list to store features
features_list = []

def hook_fn(module, input, output):
    # input[0] of the fc layer is the feature vector (128-dim)
    feature = input[0].detach().cpu().numpy()
    features_list.append(feature)

def process_balanced_sampling(X, y, max_samples_per_class=None):
    """
    Balance the dataset for visualization.
    If max_samples_per_class is None, uses the size of the smallest class (undersampling).
    Alternatively, set a fixed number e.g. 500.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Original Class Distribution: {dict(zip(unique_classes, counts))}")
    
    if max_samples_per_class is None:
        max_samples_per_class = np.min(counts)
        print(f"Auto-balancing to {max_samples_per_class} samples per class.")
    
    indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > max_samples_per_class:
            # Randomly select
            selected = np.random.choice(cls_indices, max_samples_per_class, replace=False)
        else:
            selected = cls_indices
        indices.append(selected)
    
    all_indices = np.concatenate(indices)
    return X[all_indices], y[all_indices]

def get_options():
    # Simulate arguments setup
    opt = TrainOptions()
    
    # Parse generic options
    # We override sys.argv to avoid conflict if run with other args, 
    # but here we simply rely on default parsing or manual setting.
    # To be safe, we can manually construct the opt object or pass default args.
    
    # Let's try to parse with empty list to get defaults
    # And then overwrite with specific project settings
    import sys
    sys.argv = ['visualize_tsne.py'] # Reset args
    
    opt = opt.parse(print_options=False, mode='val')
    
    # --- Project Specific Configuration ---
    opt.name = '1-resnet152-3b-3cls'
    opt.model_name = '3branch'
    opt.dataroot = '/root/ZYZ/GRINLAB/dataset'
    
    # Set to validation mode properties
    opt.val_split = 'val' # 'val' or 'test'
    opt.dataroot = os.path.join(opt.dataroot, opt.val_split)
    opt.isTrain = False
    opt.serial_batches = True
    opt.data_aug = False
    opt.batch_size = 32
    opt.gpu_ids = [0] 
    opt.checkpoints_dir = './checkpoints'
    opt.load_thread = 4
    
    # Ensure correct number of classes
    # 3cls means 3 classes
    opt.mode = '3cls' 
    
    # Specific epoch to load? 'latest' or specific number.
    # Re-calc metrics used epoch 20 which was good.
    opt.epoch = 'best' # or 'latest'
    
    return opt

def main():
    print(">>> 1. Configuring Options...")
    opt = get_options()
    
    # Force batch size to 1 to ensure hook captures features correctly aligned one-by-one? 
    # Actually hook captures batch info, but features_list append needs care.
    # If hook gets a batch (32, 128), and we append it, features_list will be list of arrays.
    # Concatenate later handles this.
    
    print(f"Data Root: {opt.dataroot}")
    print(f"Model: {opt.name}")
    
    print("\n>>> 2. Loading Model...")
    trainer = Trainer(opt)
    trainer.model.eval()
    
    if isinstance(trainer.model, torch.nn.DataParallel):
        real_model = trainer.model.module
    else:
        real_model = trainer.model
        
    # Hook implementation
    # Branch3RCBAM definition: 
    # self.fc = nn.Linear(128, n_output) 
    # This is indeed the final classifier.
    handle = real_model.fc.register_forward_hook(hook_fn)
    print("Hook registered on 'real_model.fc'")

    print("\n>>> 3. Loading Dataset...")
    # 只有当模型名称包含 'rcbam' 时，才加载 RETFound 特征加载器
    if 'rcbam' in opt.model_name.lower():
        print(f"检测到模型 {opt.model_name} 需要 RETFound 特征，正在初始化加载器...")
        retfound_loader = RETFoundFeatureLoader()
        dataloader = create_dataloader(opt, retfound_loader)
    else:
        print(f"模型 {opt.model_name} 不需要 RETFound 特征，跳过加载器初始化。")
        retfound_loader = None
        dataloader = create_dataloader(opt)
        
    print(f"Dataset size: {len(dataloader)} batches")

    print("\n>>> 4. Extracting Features...")
    labels_list = []
    
    # Clear global list just in case
    features_list.clear()
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            trainer.set_input(data)
            trainer.forward()
            
            # Label is data[2]
            labels = data[2].cpu().numpy()
            labels_list.append(labels.flatten())
            
    handle.remove()
    
    # Concatenate results
    # features_list is list of (B, 128) arrays
    X = np.concatenate(features_list, axis=0) 
    y = np.concatenate(labels_list, axis=0)  
    
    print(f"\nExtracted Feature Shape: {X.shape}")
    print(f"Extracted Labels Shape: {y.shape}")
    
    # --- Balance Data for Visualization ---
    print("\n>>> 5. Balancing Data for Visualization...")
    # Use a reasonable limit, e.g., 500 or 1000 per class to avoid clutter
    # If dataset is small, use min(counts)
    X_vis, y_vis = process_balanced_sampling(X, y, max_samples_per_class=2500)
    
    print(f"Visualization Feature Shape: {X_vis.shape}")
    print(f"Visualization Labels Shape: {y_vis.shape}")
    print(f"Final Class Distribution: {dict(zip(*np.unique(y_vis, return_counts=True)))}")

    # Save features
    np.savez(f'features_tsne_{opt.val_split}.npz', X=X, y=y)
    
    print("\n>>> 6. Running OpenTSNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30, # Default 30 is good for balanced sets ~500pts
        metric="cosine", # Cosine is often better for high-dim embeddings than euclidean
        n_jobs=8,
        random_state=42,
        initialization="pca",
        verbose=True
    )
    
    embedding = tsne.fit(X_vis)
    
    print("\n>>> 7. Plotting...")
    plot_save_path = f'tSNE_{opt.name}_{opt.val_split}_balanced.png'
    
    # --- Advanced Academic Styling ---
    import seaborn as sns
    import pandas as pd
    
    # Set overall aesthetic style
    # "whitegrid" provides a clean academic look with gridlines for readability
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})
    sns.set_context("talk", font_scale=1.0) # 'talk' context makes fonts slightly larger/readable

    # Create figure with high DPI (Publication Quality)
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Classes & Colors
    # Palette logic:
    # 0: Normal -> Blue (Calm, Standard) - Tableau Blue or similar
    # 1: Glaucoma -> Red (Danger, Attention) - Tableau Red
    # 2: Suspect -> Green (Intermediate) - Tableau Green or Orange
    unique_classes = np.unique(y_vis)
    
    # Using hex codes for precise control
    # Normal: #4E79A7 (Blue)
    # Glaucoma: #E15759 (Red)
    # Suspect: #59A14F (Green)
    palette_raw = {0: '#4E79A7', 1: '#E15759', 2: '#59A14F'}
    
    class_map = {0: 'Normal', 1: 'Glaucoma', 2: 'Suspect'}
    
    # Calculate counts for Legend
    # Create DataFrame for easier Seaborn mapping
    df_plot = pd.DataFrame({
        'Dim 1': embedding[:, 0],
        'Dim 2': embedding[:, 1],
        'Class_ID': y_vis,
        'Label_Base': [class_map.get(c, f"Class {c}") for c in y_vis]
    })
    
    counts = df_plot['Label_Base'].value_counts()
    
    # Create new label column with counts: "Normal (n=500)"
    df_plot['Label_Full'] = df_plot['Label_Base'].apply(lambda x: f"{x} (n={counts[x]})")
    
    # Map raw palette to new full labels
    palette_final = {}
    for c_id in unique_classes:
        base_name = class_map.get(c_id)
        full_name = f"{base_name} (n={counts[base_name]})"
        palette_final[full_name] = palette_raw.get(c_id, '#333333')

    # Scatter Plot using Seaborn
    # s=50: clear point size
    # alpha=0.8: slight transparency to show overlap density
    # edgecolor='w', linewidth=0.5: clear separation between points
    ax = sns.scatterplot(
        data=df_plot,
        x='Dim 1',
        y='Dim 2',
        hue='Label_Full',
        palette=palette_final,
        style='Label_Full', # Also use different markers if desired, or keep same
        alpha=0.8,
        s=50,
        edgecolor='white',
        linewidth=0.5,
    )
    
    # --- Aesthetic Refinements ---
    
    # Remove the top and right spines (border box) for a cleaner look
    sns.despine(trim=False)
    
    # Title & Axis Labels
    plt.title(f't-SNE Feature Projection\n({opt.model_name})', fontsize=20, fontweight='bold', pad=20, fontname='DejaVu Sans')
    plt.xlabel('Dimension 1', fontsize=16, labelpad=15, fontweight='bold')
    plt.ylabel('Dimension 2', fontsize=16, labelpad=15, fontweight='bold')
    
    # Customize Ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Legend Improvements
    legend = plt.legend(
        title='Class Distribution', 
        title_fontsize='14',
        fontsize='12', 
        loc='best', # Automatically find best spot
        frameon=True, 
        framealpha=0.95, # Almost opaque background
        facecolor='white',
        edgecolor='#cccccc', # Light grey border
        shadow=True,
        borderpad=1
    )
    legend.get_title().set_fontweight('bold')
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save as PNG and PDF (Vector format for papers)
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Done! High-quality plot saved to {plot_save_path} (and .pdf)")

if __name__ == '__main__':
    main()
