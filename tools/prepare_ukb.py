import os
import shutil
import csv
from tqdm import tqdm

# 配置路径
SOURCE_ROOT = "/root/ZYZ/GRINLAB/UKB-test"
DEST_ROOT = "/root/ZYZ/GRINLAB/UKB-test/test"

# 定义映射关系
# 源文件夹名 -> (目标子文件夹名, 标签值)
# 标签值: 0=unlikely, 1=certain, 2=suspect
MAPPING = {
    "unlikely": ("0_neg", 0),
    "certain": ("1_pos", 1),
    "suspect": ("1_pos", 2)
}

def setup_dirs():
    """创建必要的目录结构"""
    for sub in ["0_neg", "1_pos", "0_roi", "1_roi", "0_roi_800_clahe", "1_roi_800_clahe"]:
        path = os.path.join(DEST_ROOT, sub)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created: {path}")

def process_data():
    setup_dirs()
    
    csv_rows = []
    
    # 遍历源文件夹
    for source_sub, (dest_folder, label_val) in MAPPING.items():
        source_path = os.path.join(SOURCE_ROOT, source_sub)
        if not os.path.isdir(source_path):
            print(f"Warning: Source folder not found: {source_path}")
            continue
            
        print(f"Processing {source_sub} -> {dest_folder} (Label: {label_val})...")
        
        files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.bmp'))]
        
        for fname in tqdm(files):
            # 复制文件
            src_file = os.path.join(source_path, fname)
            dst_file = os.path.join(DEST_ROOT, dest_folder, fname)
            
            # 使用 shutil.copy2 保留元数据
            shutil.copy2(src_file, dst_file)
            
            # 记录到 CSV
            csv_rows.append({"x": fname, "y": label_val})

    # 生成 CSV 文件
    csv_path = os.path.join(DEST_ROOT, "ukb_relabel.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\nCompleted! Data organized in {DEST_ROOT}")
    print(f"Label mapping CSV saved to {csv_path}")
    print(f"Total images processed: {len(csv_rows)}")

if __name__ == "__main__":
    process_data()
