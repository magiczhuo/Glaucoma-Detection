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


# def data_augment(img, opt):
#     img = np.array(img)

#     if random() < opt.blur_prob:
#         sig = sample_continuous(opt.blur_sig)
#         gaussian_blur(img, sig)

#     if random() < opt.jpg_prob:
#         method = sample_discrete(opt.jpg_method)
#         qual = sample_discrete(opt.jpg_qual)
#         img = jpeg_from_key(img, qual, method)

#     return Image.fromarray(img)

# def sample_continuous(s):
#     if len(s) == 1:
#         return s[0]
#     if len(s) == 2:
#         rg = s[1] - s[0]
#         return random() * rg + s[0]
#     raise ValueError("Length of iterable s should be 1 or 2.")

# def sample_discrete(s):
#     if len(s) == 1:
#         return s[0]
#     return choice(s)

# def gaussian_blur(img, sigma):
#     gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
#     gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
#     gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

# def cv2_jpg(img, compress_val):
#     img_cv2 = img[:, :, ::-1]
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
#     result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
#     decimg = cv2.imdecode(encimg, 1)
#     return decimg[:, :, ::-1]

# def pil_jpg(img, compress_val):
#     out = BytesIO()
#     img = Image.fromarray(img)
#     img.save(out, format='jpeg', quality=compress_val)
#     img = Image.open(out)
#     # load from memory before ByteIO closes
#     img = np.array(img)
#     out.close()
#     return img

# jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

# def jpeg_from_key(img, compress_val, key):
#     method = jpeg_dict[key]
#     return method(img, compress_val)


class read_data():

    def __init__(self, opt, retfound_feature_loader=None):
        self.opt = opt
        self.root = opt.dataroot
        self.retfound_feature_loader = retfound_feature_loader

        neg_img_list = [
            os.path.join(self.root, '0_neg', train_file)
            for train_file in os.listdir(os.path.join(self.root, '0_neg'))
            if train_file.endswith(('jpg', 'png'))
        ]
        neg_roi_list = [
            os.path.join(self.root, '0_roi_800_clahe', train_file)
            for train_file in os.listdir(
                os.path.join(self.root, '0_roi_800_clahe'))
            if train_file.endswith(('jpg', 'png'))
        ]
        neg_label_list = [0 for _ in range(len(neg_img_list))]

        pos_img_list = [
            os.path.join(self.root, '1_pos', train_file)
            for train_file in os.listdir(os.path.join(self.root, '1_pos'))
            if train_file.endswith(('jpg', 'png'))
        ]
        pos_roi_list = [
            os.path.join(self.root, '1_roi_800_clahe', train_file)
            for train_file in os.listdir(
                os.path.join(self.root, '1_roi_800_clahe'))
            if train_file.endswith(('jpg', 'png'))
        ]
        pos_label_list = [1 for _ in range(len(pos_img_list))]

        self.img = neg_img_list + pos_img_list
        self.roi = neg_roi_list + pos_roi_list
        self.label = neg_label_list + pos_label_list

        print('directory, negative images, positive images:', self.root,
              len(neg_img_list), len(pos_img_list))
        print('all images/rois/labels:', len(self.img), len(self.roi),
              len(self.label))

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

            return img, roi, target, scale, imgname

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

        # 标签处理
        csv_path = None
        if 'train' in self.root:
            csv_path = '/root/ZYZ/GRINLAB/dataset/train-glaucoma-uod-relabel.csv'
        elif 'val' in self.root:
            csv_path = '/root/ZYZ/GRINLAB/dataset/valid-glaucoma-uod-relabel.csv'
        elif 'test' in self.root:
            # 优先检查本地 ukb_relabel.csv，或者使用特定路径
            if os.path.exists('/root/ZYZ/GRINLAB/UKB-test/test/ukb_relabel.csv'):
                csv_path = '/root/ZYZ/GRINLAB/UKB-test/test/ukb_relabel.csv'
            else:
                 csv_path = '/root/ZYZ/GRINLAB/dataset/test-glaucoma-uod-relabel.csv'
        
        name_to_label = {}
        if csv_path and os.path.exists(csv_path):
            try:
                df = pandas.read_csv(csv_path)
                # Ensure columns exist
                col_x = 'x' if 'x' in df.columns else df.columns[0]
                col_y = 'y' if 'y' in df.columns else df.columns[1]
                name_to_label = dict(zip(df[col_x], df[col_y]))
            except Exception as e:
                print(f"Error reading CSV {csv_path}: {e}")

        pos_sus_label_list = []
        for filepath in pos_img_list:
            filename = os.path.split(filepath)[-1]
            lbl = name_to_label.get(filename)
            if lbl is not None:
                pos_sus_label_list.append(int(lbl))
            else:
                # 默认值处理，防止崩溃
                # 如果是 UKB 数据集，可能默认为 1
                pos_sus_label_list.append(1)

        
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

            return img, roi, target, scale, imgname

        except ValueError:
            print(imgname)
            return None

    def __len__(self):
        return len(self.label)
