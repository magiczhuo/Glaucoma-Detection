import copy
import os
from io import BytesIO
from random import choice, random, sample

import cv2
import imageio
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage.filters import gaussian_filter
import pandas

# 导入RETFound特征加载器
try:
    from RETFound_Feature_Loader import RETFoundFeatureLoader
    RETFOUND_AVAILABLE = True
except ImportError:
    RETFOUND_AVAILABLE = False
    print("RETFound特征加载器不可用，将使用在线特征提取")

class read_data():

    def __init__(self, opt, retfound_feature_loader):
        self.opt = opt
        self.root = opt.dataroot
        self.retfound_feature_loader = retfound_feature_loader

        def get_valid_pairs(image_dir, roi_dir):
            if not os.path.exists(image_dir) or not os.path.exists(roi_dir):
                # print(f"Warning: Directory missing {image_dir} or {roi_dir}")
                return [], []
            img_paths = []
            roi_paths = []
            # 获取所有图片文件名
            fnames = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'jpeg', 'tif', 'bmp'))])
            for f in fnames:
                r_path = os.path.join(roi_dir, f)
                if os.path.exists(r_path):
                    img_paths.append(os.path.join(image_dir, f))
                    roi_paths.append(r_path)
                else:
                    pass
                    # print(f"Skipping {f}, ROI not found.")
            return img_paths, roi_paths

        neg_img_list, neg_roi_list = get_valid_pairs(
            os.path.join(self.root, '0_neg'),
            os.path.join(self.root, '0_roi_800_clahe')
        )
        neg_label_list = [0] * len(neg_img_list)

        pos_img_list, pos_roi_list = get_valid_pairs(
            os.path.join(self.root, '1_pos'),
            os.path.join(self.root, '1_roi_800_clahe')
        )
        pos_label_list = [1] * len(pos_img_list)

        # --- 限制测试集样本数量以加快评测速度 ---
        if not self.opt.isTrain:
            max_test_per_class = 10000 # 每一类最多选择500张图片，你可以根据需要修改这个值
            if len(neg_img_list) > max_test_per_class:
                indices = sample(range(len(neg_img_list)), max_test_per_class)
                neg_img_list = [neg_img_list[i] for i in indices]
                neg_roi_list = [neg_roi_list[i] for i in indices]
                neg_label_list = [neg_label_list[i] for i in indices]
            if len(pos_img_list) > max_test_per_class:
                indices = sample(range(len(pos_img_list)), max_test_per_class)
                pos_img_list = [pos_img_list[i] for i in indices]
                pos_roi_list = [pos_roi_list[i] for i in indices]
                pos_label_list = [pos_label_list[i] for i in indices]
            print(f"Testing mode: Sampling {len(neg_img_list)} neg and {len(pos_img_list)} pos images.")

        self.img = neg_img_list + pos_img_list
        self.roi = neg_roi_list + pos_roi_list
        self.label = neg_label_list + pos_label_list

        print('directory:', self.root)
        print(f'negative images: {len(neg_img_list)}, positive images: {len(pos_img_list)}')
        print(f'all images/rois/labels: {len(self.img)} {len(self.roi)} {len(self.label)}')

    def __getitem__(self, index):
        imgname = self.img[index]
        try:
            img, roi, target = imageio.imread(self.img[index]), imageio.imread(
                self.roi[index]), self.label[index]
            # print("img file: ", self.img[index])

            # print(imgname)
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

            # compute scaling
            # preprocess full images. Images should have been cropped to square already.
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

            # input_img = copy.deepcopy(img)
            # input_img = transforms.ToTensor()(input_img)
            # input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

            # img = transforms.CenterCrop(self.opt.cropSize)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])(img)

            scale = torch.tensor([height, width])

            # preprocess ROI images. Images should have been center cropped to square already.
            roi = transforms.Resize(self.opt.roiSize)(roi)
            # img = transforms.CenterCrop(self.opt.cropSize)(img)
            roi = transforms.ToTensor()(roi)
            roi = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])(roi)

            # 获取RETFound特征（如果可用）
            retfound_feature = None
            if self.retfound_feature_loader is not None:
                try:
                    retfound_feature = self.retfound_feature_loader.get_feature(imgname, self.root)
                except FileNotFoundError:
                    # 如果特征不存在，使用零向量
                    feature_dim = 1024
                    retfound_feature = torch.zeros(feature_dim)
                except Exception as e:
                    print(f" 获取RETFound特征失败: {e}")
                    retfound_feature = torch.zeros(1024)

            return img, roi, target, scale, imgname, retfound_feature

        except ValueError:
            print(imgname)
            return None

    def __len__(self):
        return len(self.label)

class read_data_3cls():

    def __init__(self, opt, retfound_feature_loader=None):
        self.opt = opt
        self.root = opt.dataroot
        self.retfound_feature_loader = retfound_feature_loader

        # 定义一个辅助函数来配对图片和ROI，并过滤掉没有对应ROI的图片
        def get_valid_files(img_dir, roi_dir):
            if not os.path.exists(img_dir) or not os.path.exists(roi_dir):
                return []
                
            valid_list = []
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('jpg', 'png'))])
            
            for f in img_files:
                img_path = os.path.join(img_dir, f)
                roi_path = os.path.join(roi_dir, f) 
                
                # 检查ROI是否存在
                if os.path.exists(roi_path):
                    valid_list.append(f) # 存储文件名
                else:
                    # Optional: print(f"Warning: Missing ROI for {f} at {roi_path}")
                    pass
            return valid_list

        neg_dir = os.path.join(self.root, '0_neg')
        neg_roi_dir = os.path.join(self.root, '0_roi_800_clahe')
        
        pos_dir = os.path.join(self.root, '1_pos')
        pos_roi_dir = os.path.join(self.root, '1_roi_800_clahe')
        
        # 获取有效的文件列表（文件名）
        # 0_neg
        neg_files = get_valid_files(neg_dir, neg_roi_dir)
        neg_img_list = [os.path.join(neg_dir, f) for f in neg_files]
        neg_roi_list = [os.path.join(neg_roi_dir, f) for f in neg_files]
        neg_label_list = [0] * len(neg_img_list)

        # 1_pos
        pos_files = get_valid_files(pos_dir, pos_roi_dir)
        pos_img_list = [os.path.join(pos_dir, f) for f in pos_files]
        pos_roi_list = [os.path.join(pos_roi_dir, f) for f in pos_files]

        if 'test' in self.root:
            df = pandas.read_csv('/root/ZYZ/GRINLAB/SMDG_test/test/smdg_relabel.csv')
            print("Found relabel smdg_test.csv file!\n")
        pos_sus_label_list = []
        for filepath in pos_img_list:
            filename = os.path.split(filepath)[-1]
            pos_sus_label_list.append(int(df[df['x'] == filename]['y'].values))
        
        self.img = neg_img_list + pos_img_list
        self.roi = neg_roi_list + pos_roi_list
        self.label = neg_label_list + pos_sus_label_list
        # print('directory, negative images, suspect images, positive images:', self.root,
        #       len(neg_img_list), len(sus)len(pos_img_list))
        print('all images/rois/labels:', len(self.img), len(self.roi),
              len(self.label))
        # print('label overview:', sample(self.label, 100))

    def __getitem__(self, index):
        imgname = self.img[index]
        try:
            img, roi, target = imageio.imread(self.img[index]), imageio.imread(
                self.roi[index]), self.label[index]
            # print("img file: ", self.img[index])

            # print(imgname)
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

            # compute scaling
            # preprocess full images. Images should have been cropped to square already.
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

            # input_img = copy.deepcopy(img)
            # input_img = transforms.ToTensor()(input_img)
            # input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

            # img = transforms.CenterCrop(self.opt.cropSize)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])(img)

            scale = torch.tensor([height, width])

            # preprocess ROI images. Images should have been center cropped to square already.
            roi = transforms.Resize(self.opt.roiSize)(roi)
            # img = transforms.CenterCrop(self.opt.cropSize)(img)
            roi = transforms.ToTensor()(roi)
            roi = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])(roi)

            # 获取RETFound特征（如果可用）
            retfound_feature = None
            if self.retfound_feature_loader is not None:
                try:
                    retfound_feature = self.retfound_feature_loader.get_feature(imgname, self.root)
                except FileNotFoundError:
                    # 如果特征不存在，使用零向量
                    feature_dim = 1024
                    retfound_feature = torch.zeros(feature_dim)
                except Exception as e:
                    print(f" 获取RETFound特征失败: {e}")
                    retfound_feature = torch.zeros(1024)

            return img, roi, target, scale, imgname, retfound_feature

        except ValueError:
            print(imgname)
            return None

    def __len__(self):
        return len(self.label)
