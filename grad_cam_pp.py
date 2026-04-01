import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from RETFound_Feature_Loader import RETFoundFeatureLoader
from networks.trainer import Branch3CBAM, Branch3RCBAM


class FullImageWrapper(nn.Module):
	def __init__(self, model, model_type, roi, scale, features=None):
		super().__init__()
		self.model = model
		self.model_type = model_type
		self.roi = roi
		self.scale = scale
		self.features = features

	def forward(self, x):
		# Model input follows trainer signatures:
		# Branch3RCBAM: (img, full_img, roi, scale, retfound_features)
		# Branch3CBAM:  (img, full_img, roi, scale)
		if self.model_type == "3branch-rcbam":
			output = self.model(x, x, self.roi, self.scale, self.features)
		else:
			output = self.model(x, x, self.roi, self.scale)

		if isinstance(output, tuple):
			return output[0]
		return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Manual settings
DATASET_ROOT = "./dataset/test"
NUM_CLASSES = 3
TARGET_CLASS = 1  # 1: Glaucoma, 2: Suspect
SAMPLES_PER_CLASS = 15
RANDOM_SEED = 42
OUTPUT_ROOT = "./Grad-cam"

# Run both models on the same samples.
MODEL_CONFIGS = [
	{
		"name": "1-resnet152rcbam-3b-3cls-f12",
		"model_type": "3branch-rcbam",
		"checkpoint": "./checkpoints/1-resnet152rcbam-3b-3cls-f12/model_epoch_best.pth",
	},
	{
		"name": "1-resnetcbam-3b-3cls",
		"model_type": "3branch-cbam",
		"checkpoint": "./checkpoints/1-resnetcbam-3b-3cls/model_epoch_best.pth",
	},
]


def resolve_checkpoint_path(path):
	if os.path.exists(path):
		return path

	# Common folder naming fallback in this repo.
	alt = path.replace("1-resnetcbam-3b-3cls", "1-resnet152cbam-3b-3cls")
	if alt != path and os.path.exists(alt):
		print(f"[Info] checkpoint not found: {path}")
		print(f"[Info] use fallback checkpoint: {alt}")
		return alt

	raise FileNotFoundError(f"Checkpoint not found: {path}")


def find_roi_path(img_path):
	img_dir = os.path.dirname(img_path)
	img_name = os.path.basename(img_path)

	if "1_pos" in img_dir:
		roi_dir = img_dir.replace("1_pos", "1_roi_800_clahe")
	elif "0_neg" in img_dir:
		roi_dir = img_dir.replace("0_neg", "0_roi_800_clahe")
	else:
		roi_dir = img_dir

	roi_path = os.path.join(roi_dir, img_name)
	if not os.path.exists(roi_path):
		print(f"[Warn] ROI not found, fallback to full image: {roi_path}")
		return img_path
	return roi_path


def prepare_input(img_path, feature_loader):
	target_size = (299, 299)

	full_pil = Image.open(img_path).convert("RGB")
	roi_pil = Image.open(find_roi_path(img_path)).convert("RGB")

	transform = transforms.Compose([
		transforms.Resize(target_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])

	full_tensor = transform(full_pil).unsqueeze(0).to(device)
	roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
	scale = torch.tensor([[target_size[0], target_size[1]]], device=device)

	# Only used by RCBAM models.
	try:
		features = feature_loader.get_feature(os.path.abspath(img_path), os.path.abspath(DATASET_ROOT))
	except Exception as e:
		print(f"[Warn] feature loading failed: {e}. Use zeros.")
		features = torch.zeros(1024)

	features = features.unsqueeze(0).to(device)

	full_rgb = np.array(full_pil)
	full_rgb = cv2.resize(full_rgb, target_size)
	return full_tensor, roi_tensor, scale, features, full_rgb


def build_model(model_type):
	if model_type == "3branch-rcbam":
		return Branch3RCBAM(n_output=NUM_CLASSES, use_offline_features=True)
	if model_type == "3branch-cbam":
		return Branch3CBAM(n_output=NUM_CLASSES)
	raise ValueError(f"Unsupported model_type: {model_type}")


def load_checkpoint(model, ckpt_path):
	checkpoint = torch.load(ckpt_path, map_location=device)
	if isinstance(checkpoint, dict) and "model" in checkpoint:
		state_dict = checkpoint["model"]
	elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint

	cleaned = {}
	for k, v in state_dict.items():
		if k.startswith("module."):
			cleaned[k.replace("module.", "", 1)] = v
		else:
			cleaned[k] = v

	model.load_state_dict(cleaned, strict=False)
	model.to(device)
	model.eval()
	return model


def select_samples():
	random.seed(RANDOM_SEED)
	pos_dir = os.path.join(DATASET_ROOT, "1_pos")
	neg_dir = os.path.join(DATASET_ROOT, "0_neg")

	pos_files = [
		os.path.join(pos_dir, f)
		for f in os.listdir(pos_dir)
		if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
	]
	neg_files = [
		os.path.join(neg_dir, f)
		for f in os.listdir(neg_dir)
		if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
	]

	selected_pos = random.sample(pos_files, min(SAMPLES_PER_CLASS, len(pos_files)))
	selected_neg = random.sample(neg_files, min(SAMPLES_PER_CLASS, len(neg_files)))
	return selected_pos, selected_neg


def process_single_image(model, model_type, img_path, subset_save_dir, feature_loader):
	try:
		full_tensor, roi_tensor, scale, features, full_rgb = prepare_input(img_path, feature_loader)
	except Exception as e:
		print(f"[Error] prepare failed for {img_path}: {e}")
		return

	wrapper_full = FullImageWrapper(
		model=model,
		model_type=model_type,
		roi=roi_tensor,
		scale=scale,
		features=features,
	)

	cam_full = GradCAMPlusPlus(model=wrapper_full, target_layers=[model.resnet.layer4[-1]])
	targets = [ClassifierOutputTarget(TARGET_CLASS)]
	grayscale_cam = cam_full(input_tensor=full_tensor, targets=targets)[0, :]

	rgb_img_float = full_rgb.astype(np.float32) / 255.0
	cam_full_img = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

	basename = os.path.basename(img_path)
	out_orig = os.path.join(subset_save_dir, f"orig_{basename}")
	out_cam = os.path.join(subset_save_dir, f"cam_{basename}")

	cv2.imwrite(out_orig, full_rgb[:, :, ::-1])
	cv2.imwrite(out_cam, cam_full_img[:, :, ::-1])

	# Required output: print processed image paths.
	print(f"[Saved] {out_orig}")
	print(f"[Saved] {out_cam}")


def ensure_output_dirs(checkpoint_name):
	model_root = os.path.join(OUTPUT_ROOT, checkpoint_name)
	pos_root = os.path.join(model_root, "1-pos")
	neg_root = os.path.join(model_root, "0-neg")
	os.makedirs(pos_root, exist_ok=True)
	os.makedirs(neg_root, exist_ok=True)
	return pos_root, neg_root


def main():
	selected_pos, selected_neg = select_samples()
	print(f"Selected {len(selected_pos)} positive and {len(selected_neg)} negative samples.")
	print("The same samples will be used for all models.")

	feature_loader = RETFoundFeatureLoader()

	for cfg in MODEL_CONFIGS:
		model_type = cfg["model_type"]
		ckpt_path = resolve_checkpoint_path(cfg["checkpoint"])
		checkpoint_name = os.path.basename(os.path.dirname(ckpt_path))

		print("\n" + "=" * 80)
		print(f"Model: {checkpoint_name}")
		print(f"Type : {model_type}")
		print(f"Ckpt : {ckpt_path}")
		print("=" * 80)

		model = build_model(model_type)
		model = load_checkpoint(model, ckpt_path)

		pos_root, neg_root = ensure_output_dirs(checkpoint_name)

		print("\n--- Processing 1-pos ---")
		for img_path in selected_pos:
			process_single_image(model, model_type, img_path, pos_root, feature_loader)

		print("\n--- Processing 0-neg ---")
		for img_path in selected_neg:
			process_single_image(model, model_type, img_path, neg_root, feature_loader)

	print("\nAll done.")
	print(f"Outputs are saved under: {os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
	main()
