import os
from pathlib import Path
import torch

class RETFoundFeatureLoader:
    def __init__(self):
        """
        :param feature_root: 离线特征的根目录，
                             例如: "/path/to/project/retfound_features"
                             "/path/to/project/retfound_features_ukb"
        """
        # 1. 获取当前文件 (loader.py) 的绝对路径
        current_file = Path(__file__).resolve()
        
        # 2. 获取项目根目录 
        # .parent 是 RETFound 文件夹
        project_root = current_file.parent
        
        # 3. 拼接得到 retfound_features 的绝对路径
        self.feature_root = os.path.join(project_root, "retfound_features_smdg")
        # self.feature_root = os.path.join(project_root, "retfound_features_smdg")

        # 打印一下，方便调试时确认路径是否正确
        #print(f"RETFoundFeatureLoader 自动定位特征目录: {self.feature_root}")

    def get_feature(self, img_path, dataset_root):
        """
        根据图像路径找到对应的特征文件
        :param img_path: 图像的绝对路径，例如 ".../dataset/train/0_neg/123.jpg"
        :param dataset_root: 数据集的根路径，例如 ".../dataset/train"
        :return: torch.Tensor [1024]
        """
        # dataset_root 是 ".../dataset/train/"，则 base_root 会变成 ".../dataset"
        base_root = os.path.dirname(dataset_root.rstrip(os.sep))
        # print(dataset_root)
        # print(base_root)
        # 1. 计算相对路径 (例如: "train/0_neg/123.jpg")
        rel_path = os.path.relpath(img_path, base_root)
        # print(rel_path)

        # 2. 修改后缀为 .pt (例如: "train/0_neg/123.pt")
        feat_rel_path = os.path.splitext(rel_path)[0] + '.pt'
        
        # 3. 拼接特征根目录
        feat_path = os.path.join(self.feature_root, feat_rel_path)
        
        # 4. 加载特征
        if os.path.exists(feat_path):
            return torch.load(feat_path, map_location='cpu')
        else:
            # 如果特征缺失，返回零向量并报警
            print(f"Warning: Feature file missing at {feat_path}")
            return torch.zeros(1024)