import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def draw_academic_loss_curve(csv_path, save_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    epochs = df['Epoch']
    train_loss = df['Train_Loss']
    val_loss = df['Val_Loss']
    
    # 设置绘图风格
    # 使用 classic 风格通常比较接近这种简单风格，也可以手动调整
    # plt.style.use('seaborn-white') 
    
    plt.figure(figsize=(8, 6))
    
    # 绘制曲线
    # blue for train, orange for val (matching the request image style)
    plt.plot(epochs, train_loss, label='train', color='#1f77b4', linewidth=1.5)
    plt.plot(epochs, val_loss, label='val', color='#ff7f0e', linewidth=1.5)
    
    # 设置标题和标签
    plt.title('model loss', fontsize=14)
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    
    # 图例
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    
    # 调整坐标轴范围 (可选，让图更好看)
    # plt.ylim(0, max(max(train_loss), max(val_loss)) * 1.1)
    
    # 去除网格 (图片中看起来没有明显的网格)
    plt.grid(False)
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Academic style loss curve saved to: {save_path}")

if __name__ == "__main__":
    checkpoint_dir = '/root/ZYZ/GRINLAB/checkpoints/1-resnet152rcbam-3b-f12'
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    save_path = os.path.join(checkpoint_dir, 'Academic_Loss_Curve.png')
    
    if os.path.exists(csv_path):
        draw_academic_loss_curve(csv_path, save_path)
    else:
        print(f"Error: {csv_path} not found.")
