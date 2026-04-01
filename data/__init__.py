import collections
import os

from . import dataset
from . import dataset_enhanced
from . import dataset_ukb
from . import dataset_enhanced_smdg

import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

# 导入RETFound特征加载器
from RETFound_Feature_Loader import RETFoundFeatureLoader
RETFOUND_AVAILABLE = True


def get_dataset(opt, retfound_feature_loader=None):
    if "SMDG" in opt.dataroot:
        print(">>> Using SMDG dataset loader")
        if opt.mode == 'binary':
            loader = dataset_enhanced_smdg.read_data
        else:
            loader = dataset_enhanced_smdg.read_data_3cls
        if retfound_feature_loader is not None and RETFOUND_AVAILABLE:
            return loader(opt, retfound_feature_loader)     
        else:
            return loader(opt)
    
    # 优先检测 UKB 数据集 (只要路径包含 UKB-test)
    if 'UKB-test' in opt.dataroot:
        if opt.mode == 'binary':
            # binary模式下使用read_data_ukb
            loader_fn = dataset_ukb.read_data_ukb
        else:
            # 3cls模式下使用read_data_ukb_3cls
            loader_fn = dataset_ukb.read_data_ukb_3cls

        if retfound_feature_loader is not None and RETFOUND_AVAILABLE:
            return loader_fn(opt, retfound_feature_loader)
        else:
            return loader_fn(opt)

    """if opt.mode == 'binary':
        if retfound_feature_loader is not None and RETFOUND_AVAILABLE:
            return dataset_ukb.read_data_ukb(opt, retfound_feature_loader)
        else:
            return dataset_ukb.read_data_ukb(opt)
    elif opt.mode == '3cls':
        if retfound_feature_loader is not None and RETFOUND_AVAILABLE:
            return dataset_ukb.read_data_ukb_3cls(opt, retfound_feature_loader)
        else:
            return dataset_ukb.read_data_ukb_3cls(opt)"""
            
    if opt.mode == 'binary':
        if retfound_feature_loader is not None and RETFOUND_AVAILABLE:
            return dataset_enhanced.read_data(opt, retfound_feature_loader)
        else:
            return dataset.read_data(opt)
    if opt.mode == '3cls':
        if retfound_feature_loader is not None and RETFOUND_AVAILABLE:
            return dataset_enhanced.read_data_3cls(opt, retfound_feature_loader)
        else:
            return dataset.read_data_3cls(opt)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def print_label_distribution(dataset, num):
    length = len(dataset)
    # print('>>> Size of dataset:', length)
    # print(len(dataset[0]))
    # print(dataset[0][0].shape, dataset[0][1].shape, dataset[100][2])
    l = [dataset[np.random.randint(0, length)][2] for _ in range(num)]
    d = collections.Counter(l)
    k = sorted(d.keys())
    print(
        '>>> Label distribution for %d random samples in original dataset:    '
        % (num))
    for i in range(len(k)):
        print('>>>', k[i], ':', d[k[i]])


def patch_collate_train(batch):
    # 过滤掉None项
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    img = torch.stack([item[0] for item in batch], dim=0)
    roi = torch.stack([item[1] for item in batch], dim=0)
    target = torch.tensor([item[2] for item in batch])
    scale = torch.stack([item[3] for item in batch], dim=0)
    filename = [item[4] for item in batch]
    
    # 检查是否有RETFound特征
    if len(batch[0]) > 5 and batch[0][5] is not None:  # 有RETFound特征且不为None
        retfound_features = torch.stack([item[5] for item in batch], dim=0)
        return [img, roi, target, scale, filename, retfound_features]
    else:
        return [img, roi, target, scale,filename]


def create_dataloader(opt, retfound_feature_loader=None):
    shuffle = not opt.serial_batches if (opt.isTrain
                                         and not opt.class_bal) else False
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    dataset = get_dataset(opt, retfound_feature_loader)
    print_label_distribution(dataset, 100)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              collate_fn=patch_collate_train,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return data_loader


def patch_collate_test(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    img = torch.stack([item[0] for item in batch], dim=0)
    roi = torch.stack([item[1] for item in batch], dim=0)
    target = torch.tensor([item[2] for item in batch])
    scale = torch.stack([item[3] for item in batch], dim=0)
    filename = [item[4] for item in batch]
    
    # Check for RETFound features
    if len(batch[0]) > 5 and batch[0][5] is not None:  # Feature index 5 exists and is not None
        retfound_features = torch.stack([item[5] for item in batch], dim=0)
        return [img, roi, target, scale, filename, retfound_features]
    else:
        return [img, roi, target, scale,filename]


def create_dataloader_test(opt, retfound_feature_loader=None):
    shuffle = not opt.serial_batches if (opt.isTrain
                                         and not opt.class_bal) else False
    dataset = get_dataset(opt, retfound_feature_loader)
    print_label_distribution(dataset, len(dataset))
    # print('>>> Size of dataset: ', len(dataset))
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        collate_fn=patch_collate_test,
        sampler=sampler,
        num_workers=int(opt.num_threads))
    return data_loader_test
