import os
import shutil
import pandas as pd
from tqdm import tqdm

# 配置路径
DATA_ROOT = "/root/ZYZ/GRINLAB/SMDG/full-fundus/full-fundus"
CSV_PATH = "/root/ZYZ/GRINLAB/SMDG/metadata - standardized.csv"
DEST_ROOT = "/root/ZYZ/GRINLAB/SMDG_test/test"

# 定义映射关系
# 标签值: 0=neg, 1=pos, -1=suspect
# 目标子文件夹名, 最终标签值
MAPPING = {
    0: ("0_neg", 0),
    1: ("1_pos", 1),
    -1: ("1_pos", 2)
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
    
    # 读取CSV文件
    df = pd.read_csv(CSV_PATH)
    
    csv_rows = []
    
    print(f"Total images in CSV: {len(df)}")
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        label_val = row['types']
        
        if label_val not in MAPPING:
            # print(f"Unknown label {label_val} for {row['names']}")
            continue
            
        dest_folder, final_label = MAPPING[label_val]
        
        img_name = str(row['names']) + '.png'
        img_path = os.path.join(DATA_ROOT, img_name)
        
        if not os.path.exists(img_path):
            img_name = str(row['names']) + '.jpg'
            img_path = os.path.join(DATA_ROOT, img_name)
            if not os.path.exists(img_path):
                if pd.notna(row['fundus']):
                    img_name = os.path.basename(str(row['fundus']))
                    img_path = os.path.join(DATA_ROOT, img_name)
                
                if not os.path.exists(img_path):
                    # print(f"Image not found for {row['names']}")
                    continue
        
        # 复制文件
        dst_file = os.path.join(DEST_ROOT, dest_folder, img_name)
        
        # 使用 shutil.copy2 保留元数据
        if not os.path.exists(dst_file):
            shutil.copy2(img_path, dst_file)
        
        # 记录到 CSV
        csv_rows.append({"x": img_name, "y": final_label})

    # 生成 CSV 文件
    csv_path = os.path.join(DEST_ROOT, "smdg_relabel.csv")
    
    # 使用 pandas 保存 CSV
    out_df = pd.DataFrame(csv_rows)
    out_df.to_csv(csv_path, index=False)
    
    print(f"\nCompleted! Data organized in {DEST_ROOT}")
    print(f"Label mapping CSV saved to {csv_path}")
    print(f"Total images processed: {len(csv_rows)}")

if __name__ == "__main__":
    process_data()
