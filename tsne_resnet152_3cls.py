import os
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from openTSNE import TSNE
import pandas as pd
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet152

# Global list to store features
features_list = []


@dataclass
class TSNEConfig:
    model_name: str = 'ResNet152'
    dataset_root: str = '/root/ZYZ/GRINLAB/dataset'
    split: str = 'val'  # train / val / test
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 299
    num_classes: int = 3
    checkpoint_path: str = '/root/ZYZ/GRINLAB/checkpoints/resnet152_3cls/1_resnet152_3cls/model_epoch_001.pth'
    max_samples_per_class: int = 2500
    tsne_perplexity: int = 30
    tsne_metric: str = 'cosine'
    tsne_n_jobs: int = 8
    random_state: int = 42

    @property
    def data_root(self):
        return os.path.join(self.dataset_root, self.split)

    @property
    def feature_save_path(self):
        return f'features_tsne_{self.split}.npz'

    @property
    def plot_save_path(self):
        return f'tSNE_{self.model_name}_{self.split}_balanced.png'


class ThreeClassFundusDataset(Dataset):
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def __init__(self, data_root: str, split: str, image_size: int):
        self.data_root = data_root
        self.split = split
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._build_samples()

    def _get_valid_files(self, img_dir: str, roi_dir: str):
        if not os.path.exists(img_dir) or not os.path.exists(roi_dir):
            return []

        files = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(self.IMAGE_EXTS):
                continue
            if os.path.exists(os.path.join(roi_dir, fname)):
                files.append(fname)
        return files

    def _get_csv_path(self):
        base_root = '/root/ZYZ/GRINLAB'
        if self.split == 'train':
            return os.path.join(base_root, 'dataset/train-glaucoma-uod-relabel.csv')
        if self.split == 'val':
            return os.path.join(base_root, 'dataset/valid-glaucoma-uod-relabel.csv')
        if self.split == 'test':
            return os.path.join(base_root, 'dataset/test-glaucoma-uod-relabel.csv')
        return None

    def _load_label_map(self):
        csv_path = self._get_csv_path()
        if csv_path is None or not os.path.exists(csv_path):
            return {}

        df = pd.read_csv(csv_path)
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        y_col = 'y' if 'y' in df.columns else df.columns[1]
        return dict(zip(df[x_col], df[y_col]))

    def _build_samples(self):
        neg_dir = os.path.join(self.data_root, '0_neg')
        neg_roi_dir = os.path.join(self.data_root, '0_roi_800_clahe')
        pos_dir = os.path.join(self.data_root, '1_pos')
        pos_roi_dir = os.path.join(self.data_root, '1_roi_800_clahe')

        neg_files = self._get_valid_files(neg_dir, neg_roi_dir)
        for fname in neg_files:
            self.image_paths.append(os.path.join(neg_dir, fname))
            self.labels.append(0)

        label_map = self._load_label_map()
        pos_files = self._get_valid_files(pos_dir, pos_roi_dir)
        for fname in pos_files:
            label = int(label_map.get(fname, 1))
            self.image_paths.append(os.path.join(pos_dir, fname))
            self.labels.append(label)

        print(f'all images/labels: {len(self.image_paths)} / {len(self.labels)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def hook_fn(module, input, output):
    # input[0] of the fc layer is the feature vector (2048-dim for resnet152)
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

def create_manual_dataloader(cfg: TSNEConfig):
    dataset = ThreeClassFundusDataset(
        data_root=cfg.data_root,
        split=cfg.split,
        image_size=cfg.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model_from_checkpoint(cfg: TSNEConfig, device: torch.device):
    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    # Build the same backbone/classifier shape as training, then load fine-tuned weights.
    try:
        model = resnet152(weights=None)
    except Exception:
        model = resnet152(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg.num_classes)

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Handle DataParallel checkpoints.
    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model

def main():
    print(">>> 1. Configuring Manual Settings...")
    cfg = TSNEConfig()
    np.random.seed(cfg.random_state)

    print(f"Data Root: {cfg.data_root}")
    print(f"Model: {cfg.model_name}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    
    print("\n>>> 2. Loading ResNet152 From Checkpoint...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    real_model = build_model_from_checkpoint(cfg, device)
    
    # Hook implementation
    # input[0] of the fc layer is the feature vector (2048-dim for resnet152)
    handle = real_model.fc.register_forward_hook(hook_fn)
    print("Hook registered on 'real_model.fc'")

    print("\n>>> 3. Loading Dataset...")
    print(f"直接使用手动定义的数据集与DataLoader。")
    dataloader = create_manual_dataloader(cfg)
        
    print(f"Dataset size: {len(dataloader)} batches")

    print("\n>>> 4. Extracting Features...")
    labels_list = []
    
    # Clear global list just in case
    features_list.clear()
    
    with torch.no_grad():
        for img, labels in tqdm(dataloader):
            img = img.to(device, non_blocking=True)
            _ = real_model(img)
            
            labels = labels.cpu().numpy()
            labels_list.append(labels.flatten())
            
    handle.remove()
    
    # Concatenate results
    # features_list is list of (B, 2048) arrays
    X = np.concatenate(features_list, axis=0) 
    y = np.concatenate(labels_list, axis=0)  
    
    print(f"\nExtracted Feature Shape: {X.shape}")
    print(f"Extracted Labels Shape: {y.shape}")
    
    # --- Balance Data for Visualization ---
    print("\n>>> 5. Balancing Data for Visualization...")
    # Use a reasonable limit, e.g., 500 or 1000 per class to avoid clutter
    # If dataset is small, use min(counts)
    X_vis, y_vis = process_balanced_sampling(X, y, max_samples_per_class=cfg.max_samples_per_class)
    
    print(f"Visualization Feature Shape: {X_vis.shape}")
    print(f"Visualization Labels Shape: {y_vis.shape}")
    print(f"Final Class Distribution: {dict(zip(*np.unique(y_vis, return_counts=True)))}")

    # Save features
    np.savez(cfg.feature_save_path, X=X, y=y)
    
    print("\n>>> 6. Running OpenTSNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        metric=cfg.tsne_metric,
        n_jobs=cfg.tsne_n_jobs,
        random_state=cfg.random_state,
        initialization="pca",
        verbose=True
    )
    
    embedding = tsne.fit(X_vis)
    
    print("\n>>> 7. Plotting...")
    plot_save_path = cfg.plot_save_path
    
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
    plt.title(f't-SNE Feature Projection\n({cfg.model_name})', fontsize=20, fontweight='bold', pad=20, fontname='DejaVu Sans')
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
