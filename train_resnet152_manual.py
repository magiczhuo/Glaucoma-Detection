import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm


@dataclass
class TrainConfig:
    # Manual settings (edit these values directly)
    model_name: str = "1_resnet152_3cls"
    num_classes: int = 3
    train_dir: str = "/root/ZYZ/GRINLAB/dataset/train"
    val_dir: str = "/root/ZYZ/GRINLAB/dataset/val"
    train_csv: str = "/root/ZYZ/GRINLAB/dataset/train-glaucoma-uod-relabel.csv"
    val_csv: str = "/root/ZYZ/GRINLAB/dataset/valid-glaucoma-uod-relabel.csv"

    # Data folders under each split dir
    neg_dir_name: str = "0_neg"
    pos_dir_name: str = "1_pos"

    # Optimization settings
    image_size: int = 299
    batch_size: int = 24
    num_workers: int = 4
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    use_class_weights: bool = True
    seed: int = 42

    # Output settings
    checkpoint_root: str = "/root/ZYZ/GRINLAB/checkpoints/resnet152_3cls"

    @property
    def run_dir(self) -> Path:
        return Path(self.checkpoint_root) / self.model_name


class ManualFundusDataset(Dataset):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        data_dir: str,
        num_classes: int,
        csv_path: str,
        transform: transforms.Compose,
        neg_dir_name: str,
        pos_dir_name: str,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.csv_path = csv_path
        self.transform = transform
        self.neg_dir_name = neg_dir_name
        self.pos_dir_name = pos_dir_name

        self.samples: List[Tuple[str, int]] = []
        self._build_samples()

    def _load_label_map(self) -> Dict[str, int]:
        label_map: Dict[str, int] = {}
        if self.num_classes <= 2:
            return label_map

        if not self.csv_path or not os.path.exists(self.csv_path):
            print(f"[WARN] CSV not found: {self.csv_path}. Positive samples default to label 1.")
            return label_map

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return label_map

            key_col = "x" if "x" in reader.fieldnames else reader.fieldnames[0]
            label_col = "y" if "y" in reader.fieldnames else reader.fieldnames[1]

            for row in reader:
                if key_col not in row or label_col not in row:
                    continue
                name = os.path.basename(str(row[key_col]).strip())
                try:
                    label = int(float(row[label_col]))
                except (TypeError, ValueError):
                    continue
                label_map[name] = label

        return label_map

    def _list_images(self, folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        images = []
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                images.append(p)
        return images

    def _build_samples(self) -> None:
        neg_dir = self.data_dir / self.neg_dir_name
        pos_dir = self.data_dir / self.pos_dir_name
        label_map = self._load_label_map()

        neg_images = self._list_images(neg_dir)
        for p in neg_images:
            self.samples.append((str(p), 0))

        pos_images = self._list_images(pos_dir)
        for p in pos_images:
            if self.num_classes <= 2:
                label = 1
            else:
                label = int(label_map.get(p.name, 1))

            if not (0 <= label < self.num_classes):
                print(f"[WARN] Skip {p.name}: invalid label={label} for num_classes={self.num_classes}")
                continue

            self.samples.append((str(p), label))

        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes) if labels else np.array([])
        print(f"[DATA] {self.data_dir} -> total={len(self.samples)}, class_counts={counts.tolist() if len(counts) else []}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    try:
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    except Exception:
        # Fallback for older torchvision versions.
        model = resnet152(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def prepare_dirs(cfg: TrainConfig) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)


def main():
    cfg = TrainConfig()
    seed_everything(cfg.seed)
    prepare_dirs(cfg)

    print("=" * 80)
    print("Manual ResNet152 Training (no TrainOptions)")
    print(f"Model Name: {cfg.model_name}")
    print(f"Train Dir : {cfg.train_dir}")
    print(f"Val Dir   : {cfg.val_dir}")
    print(f"Save Dir  : {cfg.run_dir}")
    print("=" * 80)

    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ManualFundusDataset(
        data_dir=cfg.train_dir,
        num_classes=cfg.num_classes,
        csv_path=cfg.train_csv,
        transform=train_transform,
        neg_dir_name=cfg.neg_dir_name,
        pos_dir_name=cfg.pos_dir_name,
    )
    val_dataset = ManualFundusDataset(
        data_dir=cfg.val_dir,
        num_classes=cfg.num_classes,
        csv_path=cfg.val_csv,
        transform=val_transform,
        neg_dir_name=cfg.neg_dir_name,
        pos_dir_name=cfg.pos_dir_name,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Dataset is empty. Please check your train_dir/val_dir and folder names.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.num_classes).to(device)

    if cfg.use_class_weights:
        train_labels = [label for _, label in train_dataset.samples]
        class_weights = compute_class_weights(train_labels, cfg.num_classes).to(device)
        print(f"[INFO] Using class weights: {class_weights.detach().cpu().numpy().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    log_csv = cfg.run_dir / "training_log.csv"
    with open(log_csv, "w") as f:
        f.write("epoch,lr,train_loss,train_acc,val_loss,val_acc,epoch_weight,best_weight\n")

    best_acc = -1.0
    best_epoch = -1
    best_weight_path = cfg.run_dir / "model_epoch_best.pth"

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Save every epoch weights.
        epoch_weight_path = cfg.run_dir / f"model_epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), epoch_weight_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_weight_path)

        elapsed = time.time() - start_time
        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
            f"lr={current_lr:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        with open(log_csv, "a") as f:
            f.write(
                f"{epoch},{current_lr:.8f},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},{epoch_weight_path},{best_weight_path}\n"
            )

    with open(cfg.run_dir / "best_info.txt", "w") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_acc={best_acc:.6f}\n")
        f.write(f"best_weight={best_weight_path}\n")

    print("\nTraining finished.")
    print(f"All epoch weights: {cfg.run_dir}/model_epoch_*.pth")
    print(f"Best weight      : {best_weight_path}")
    print(f"Training log CSV : {log_csv}")


if __name__ == "__main__":
    main()