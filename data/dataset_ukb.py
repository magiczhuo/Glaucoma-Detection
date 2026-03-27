import copy
import os
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas

# 导入RETFound特征加载器
try:
    from RETFound_Feature_Loader import RETFoundFeatureLoader
    RETFOUND_AVAILABLE = True
except ImportError:
    RETFOUND_AVAILABLE = False
    print("RETFound特征加载器不可用，将使用在线特征提取")

class read_data_ukb():
    """
    UKB 二分类数据读取器 (0_neg vs 1_pos, 不细分 suspect/certain)
    """
    def __init__(self, opt, retfound_feature_loader):
        self.opt = opt
        self.root = opt.dataroot
        self.retfound_feature_loader = retfound_feature_loader

        def get_valid_pairs(image_dir, roi_dir, label_value):
            valid_img_paths = []
            valid_roi_paths = []
            valid_labels = []
            
            if not os.path.exists(image_dir):
                return [], [], []
                
            img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'jpeg', 'tif', 'bmp'))])
            
            for f in img_files:
                img_path = os.path.join(image_dir, f)
                roi_path = os.path.join(roi_dir, f) # Assuming ROI has same filename
                
                if os.path.exists(roi_path):
                    valid_img_paths.append(img_path)
                    valid_roi_paths.append(roi_path)
                    valid_labels.append(label_value)
            
            return valid_img_paths, valid_roi_paths, valid_labels

        # Define directories
        neg_dir = os.path.join(self.root, '0_neg')
        neg_roi_dir = os.path.join(self.root, '0_roi_300_clahe')
        
        pos_dir = os.path.join(self.root, '1_pos')
        pos_roi_dir = os.path.join(self.root, '1_roi_300_clahe')

        # 1. Negative
        neg_img, neg_roi, neg_lbl = get_valid_pairs(neg_dir, neg_roi_dir, 0)

        # 2. Positive
        pos_img, pos_roi, pos_lbl = get_valid_pairs(pos_dir, pos_roi_dir, 1)

        self.img = neg_img + pos_img
        self.roi = neg_roi + pos_roi
        self.label = neg_lbl + pos_lbl

        print(f'[UKB Binary Dataset] Root: {self.root}')
        print(f'[UKB Binary Dataset] Total: {len(self.img)} | Neg: {len(neg_img)} | Pos: {len(pos_img)}')

    def __getitem__(self, index):
        imgname = self.img[index]
        try:
            # 读取图片
            img = imageio.imread(self.img[index])
            roi = imageio.imread(self.roi[index])
            target = self.label[index]

            # 转换维度
            if len(img.shape) < 3:
                img = np.asarray(img)[..., np.newaxis]
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = np.tile(np.asarray(img), (1, 1, 3))
            img = Image.fromarray(img, mode='RGB')

            if len(roi.shape) < 3:
                roi = np.asarray(roi)[..., np.newaxis]
            if len(roi.shape) == 3 and roi.shape[-1] == 1:
                roi = np.tile(np.asarray(roi), (1, 1, 3))
            roi = Image.fromarray(roi, mode='RGB')

            # 预处理 Full Image
            img = transforms.Resize(self.opt.loadSize)(img)
            height, width = img.height, img.width
            
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            scale = torch.tensor([height, width])

            # 预处理 ROI
            roi = transforms.Resize(self.opt.roiSize)(roi)
            roi = transforms.ToTensor()(roi)
            roi = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(roi)

            # 获取 RETFound 特征 (使用离线 Feature Loader)
            retfound_feature = None
            if self.retfound_feature_loader is not None:
                try:
                    # 传入的是绝对路径文件名 "imgname" 和 dataset root "self.root"
                    retfound_feature = self.retfound_feature_loader.get_feature(imgname, self.root)
                except FileNotFoundError:
                    feature_dim = 1024
                    retfound_feature = torch.zeros(feature_dim)
                except Exception as e:
                    print(f" 获取RETFound特征失败: {e}")
                    retfound_feature = torch.zeros(1024)
            
            return img, roi, target, scale, imgname, retfound_feature

        except Exception as e:
            print(f"Error loading {imgname}: {e}")
            return None

    def __len__(self):
        return len(self.label)


class read_data_ukb_3cls():

    def __init__(self, opt, retfound_feature_loader=None):
        self.opt = opt
        self.root = opt.dataroot
        self.retfound_feature_loader = retfound_feature_loader
        
        # Helper function to pair images and ROIs
        def get_valid_pairs(image_dir, roi_dir, label_value, name_to_label=None):
            valid_img_paths = []
            valid_roi_paths = []
            valid_labels = []
            
            if not os.path.exists(image_dir):
                print(f"Warning: Directory not found: {image_dir}")
                return [], [], []
                
            img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'jpeg', 'tif', 'bmp'))])
            
            for f in img_files:
                img_path = os.path.join(image_dir, f)
                roi_path = os.path.join(roi_dir, f) # Assuming ROI has same filename
                
                if os.path.exists(roi_path):
                    valid_img_paths.append(img_path)
                    valid_roi_paths.append(roi_path)
                    
                    if name_to_label is not None:
                        # Dynamic label lookup
                         valid_labels.append(int(name_to_label.get(f, 1)))
                    else:
                        # Static label
                        valid_labels.append(label_value)
                else:
                    # Optional: Print warning for missing ROI
                    # print(f"Warning: ROI Missing for {f}")
                    pass
            
            return valid_img_paths, valid_roi_paths, valid_labels

        # Define directories
        neg_dir = os.path.join(self.root, '0_neg')
        neg_roi_dir = os.path.join(self.root, '0_roi_300_clahe')
        
        pos_dir = os.path.join(self.root, '1_pos')
        pos_roi_dir = os.path.join(self.root, '1_roi_300_clahe')

        # CSV Handling for Labels
        csv_path = os.path.join(self.root, 'ukb_relabel.csv')
        name_to_label = {}
        if os.path.exists(csv_path):
             try:
                df = pandas.read_csv(csv_path)
                if 'x' in df.columns and 'y' in df.columns:
                    name_to_label = dict(zip(df['x'], df['y']))
             except Exception:
                pass
        else:
             parent_csv_path = os.path.join(os.path.dirname(self.root.rstrip('/')), 'ukb_relabel.csv')
             if os.path.exists(parent_csv_path):
                 try:
                    df = pandas.read_csv(parent_csv_path)
                    if 'x' in df.columns and 'y' in df.columns:
                        name_to_label = dict(zip(df['x'], df['y']))
                 except:
                    pass

        # 1. Process Negative
        neg_img, neg_roi, neg_lbl = get_valid_pairs(neg_dir, neg_roi_dir, 0, None) # Label 0

        # 2. Process Positive (with CSV lookup)
        pos_img, pos_roi, pos_lbl = get_valid_pairs(pos_dir, pos_roi_dir, 1, name_to_label)

        self.img = neg_img + pos_img
        self.roi = neg_roi + pos_roi
        self.label = neg_lbl + pos_lbl
        
        print(f'[UKB 3-Class Dataset] Root: {self.root}')
        print(f'[UKB 3-Class Dataset] Total: {len(self.img)} | Neg: {len(neg_img)} | Pos(Mixed): {len(pos_img)}')



    def __getitem__(self, index):
        imgname = self.img[index]
        try:
            # 读取图片
            img = imageio.imread(self.img[index])
            roi = imageio.imread(self.roi[index])
            target = self.label[index]

            # 转换维度
            if len(img.shape) < 3:
                img = np.asarray(img)[..., np.newaxis]
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = np.tile(np.asarray(img), (1, 1, 3))
            img = Image.fromarray(img, mode='RGB')

            if len(roi.shape) < 3:
                roi = np.asarray(roi)[..., np.newaxis]
            if len(roi.shape) == 3 and roi.shape[-1] == 1:
                roi = np.tile(np.asarray(roi), (1, 1, 3))
            roi = Image.fromarray(roi, mode='RGB')

            # 预处理 Full Image
            img = transforms.Resize(self.opt.loadSize)(img)
            height, width = img.height, img.width
            
            # 定义数据增强操作
            data_augmentation = transforms.Compose([
                transforms.RandomApply(
                    [transforms.RandomHorizontalFlip(p=0.5)],
                    p=self.opt.aug_prob),
                transforms.RandomApply([transforms.RandomVerticalFlip(p=0.5)],
                                       p=self.opt.aug_prob),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ],
                                       p=self.opt.aug_prob),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(1, 5),
                                            sigma=(0.1, 2.))
                ],
                                       p=self.opt.aug_prob)
            ])

            if self.opt.isTrain and self.opt.data_aug:
                img = data_augmentation(img)
                roi = data_augmentation(roi)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            scale = torch.tensor([height, width])

            # 预处理 ROI
            roi = transforms.Resize(self.opt.roiSize)(roi)
            roi = transforms.ToTensor()(roi)
            roi = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(roi)

            # 获取 RETFound 特征
            retfound_feature = None
            if self.retfound_feature_loader is not None:
                try:
                    # 关键修改：RETFoundFeatureLoader 在 dataset_enhanced 中调用方式是 get_feature(imgname, root)
                    retfound_feature = self.retfound_feature_loader.get_feature(imgname, self.root)
                except FileNotFoundError:
                    feature_dim = 1024
                    retfound_feature = torch.zeros(feature_dim)
                except Exception as e:
                    print(f" 获取RETFound特征失败: {e}")
                    retfound_feature = torch.zeros(1024)
            
            return img, roi, target, scale, imgname, retfound_feature

        except Exception as e:
            print(f"Error loading {imgname}: {e}")
            return None

    def __len__(self):
        return len(self.label)
