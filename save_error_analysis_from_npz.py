import os
import sys
import numpy as np
import pandas as pd
import glob
import torch

# 添加项目根目录以便导入模块
sys.path.append('/root/ZYZ/GRINLAB')
from data.dataset_enhanced import read_data_3cls

def main():
    checkpoint_dir = '/root/ZYZ/GRINLAB/checkpoints/1-resnet15rcbam-3b-3cls-nofreeze-weighted-loss'
    
    # 1. 寻找最佳的 .npz 文件
    print("正在搜索 .npz 记录...")
    npz_pattern = os.path.join(checkpoint_dir, 'val_predictions_epoch_*.npz')
    files = glob.glob(npz_pattern)
    if not files:
        print("未找到 .npz 文件。")
        return

    best_file = None
    best_acc = -1
    best_epoch = -1
    
    # 遍历所有记录，找出准确率最高的 Epoch
    for f in files:
        try:
            d = np.load(f)
            y_true = d['y_true']
            y_pred_logit = d['y_pred_logit']
            
            # 计算准确率
            if np.min(y_pred_logit) < 0 or np.max(y_pred_logit) > 1:
                preds = np.argmax(y_pred_logit, axis=1)
            else:
                preds = np.argmax(y_pred_logit, axis=1)
                
            acc = np.mean(preds == y_true)
            
            # 从文件名提取 Epoch
            base = os.path.basename(f)
            epoch_str = base.replace('val_predictions_epoch_', '').replace('.npz', '')
            
            if acc > best_acc:
                best_acc = acc
                best_file = f
                best_epoch = epoch_str
        except Exception as e:
            pass
            
    print(f"选中最佳记录: {best_file}")
    print(f"Epoch: {best_epoch}, Accuracy: {best_acc:.4f}")
    
    # 2. 加载数据
    data = np.load(best_file)
    y_true_npz = data['y_true']
    y_pred_logit = data['y_pred_logit']
    
    # logit 转概率 (Softmax)
    exp_x = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    
    # 3. 重建数据集以获取文件名
    class MockOpt:
        def __init__(self):
            # 指向验证集路径
            self.dataroot = '/root/ZYZ/GRINLAB/dataset/val' 
            self.loadSize = 256
            self.roiSize = 256 
            self.aug_prob = 0.0
            self.isTrain = False
            self.data_aug = False
            self.mode = '3cls'
            
    opt = MockOpt()
    print("正在加载数据集文件列表以匹配图片名称...")
    # 初始化数据集 (只读取列表，不读取图片内容，速度很快)
    dataset = read_data_3cls(opt, retfound_feature_loader=None)
    
    ds_labels = np.array(dataset.label)
    ds_imgs = dataset.img
    
    # 4. 校验对齐 (Validation Alignment)
    # 确保保存的 log 和现在读取的文件顺序一致
    min_len = min(len(ds_labels), len(y_true_npz))
    mismatches = np.sum(ds_labels[:min_len] != y_true_npz[:min_len])
    
    if mismatches > 0:
        print(f"\n[严重警告] 发现 {mismatches} 个标签不匹配！")
        print("这说明文件夹内的文件顺序与训练时不同，生成的文件名可能不对应。")
    else:
        print("\n[成功] 验证校验通过，数据集顺序完全匹配，文件名映射可靠。")

    # 5. 生成 CSV
    results = []
    for i in range(min_len):
        path = ds_imgs[i]
        fname = os.path.basename(path)
        gt = int(y_true_npz[i])
        pred = int(preds[i])
        
        row = {
            'Image_Name': fname,
            'GT_Label': gt,
            'Pred_Label': pred,
            'Is_Correct': 1 if gt == pred else 0,
            'Prob_Class0': probs[i][0],
            'Prob_Class1': probs[i][1],
            'Prob_Class2': probs[i][2],
            'Confidence': np.max(probs[i])
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # 保存完整分析
    save_csv_path = os.path.join(checkpoint_dir, f'Analysis_Epoch_{best_epoch}.csv')
    df.to_csv(save_csv_path, index=False)
    print(f"\n完整结果已保存至: {save_csv_path}")
    
    # 专门提取 Class 2 的错误
    cls2_err = df[(df['GT_Label'] == 2) & (df['Is_Correct'] == 0)]
    save_cls2_path = os.path.join(checkpoint_dir, f'Class2_Errors_Epoch_{best_epoch}.csv')
    cls2_err.to_csv(save_cls2_path, index=False)
    print(f"Class 2 错误样本已保存至: {save_cls2_path}")
    print(f"共发现 {len(cls2_err)} 张 Class 2 误判图片。")

if __name__ == "__main__":
    main()