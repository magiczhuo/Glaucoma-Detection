import os
import glob
import numpy as np
import matplotlib
# Set backend to 'Agg' to write to file without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve, auc, average_precision_score, log_loss)

def compute_metric_binary(y_true, y_prob, threshold=0.5):
    # Determine predicted labels based on threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5
        
    # AP
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = 0
        
    # Log Loss
    try:
        # Avoid log(0)
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        loss = log_loss(y_true, y_prob_clipped)
    except Exception:
        loss = 0

    return {
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1': f1,
        'AUC': roc_auc,
        'AP': ap,
        'Loss': loss
    }

def draw_roc_curve_binary(y_true, y_prob, epoch, save_dir):
    """
    Draw ROC Curve for Binary class and save to file
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Binary ROC (Epoch {epoch})')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(save_dir, f'ROC_Epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path

def main():
    # Configure the checkpoint directory here
    checkpoint_dir = '/root/ZYZ/GRINLAB/checkpoints/1-resnet152rcbam-3b-f12'
    plot_dir = os.path.join(checkpoint_dir, 'ROC_Plots_Binary')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Read Training Log for comparison
    train_log_path = os.path.join(checkpoint_dir, 'training_log.csv')
    train_loss_map = {}
    if os.path.exists(train_log_path):
        try:
            with open(train_log_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 0:
                header = lines[0].strip().split(',')
                # Check for headers
                e_idx, l_idx = -1, -1
                if 'Epoch' in header: e_idx = header.index('Epoch')
                if 'Train_Loss' in header: l_idx = header.index('Train_Loss')
                
                if e_idx != -1 and l_idx != -1:
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) > max(e_idx, l_idx):
                            try:
                                train_loss_map[int(parts[e_idx])] = float(parts[l_idx])
                            except ValueError:
                                pass
        except Exception as e:
            print(f"Error reading training log: {e}")

    print(f"Searching for .npz files in {checkpoint_dir}...")
    pattern = os.path.join(checkpoint_dir, 'val_predictions_epoch_*.npz')
    files = glob.glob(pattern)
    
    if not files:
        print("No .npz files found.")
        return

    def get_epoch(fname):
        try:
            base = os.path.basename(fname)
            return int(base.replace('val_predictions_epoch_', '').replace('.npz', ''))
        except:
            return -1
    
    files.sort(key=get_epoch)
    
    print(f"\nProcessing {len(files)} files. Plots will be saved to: {plot_dir}")
    print(f"{'Epoch':<6} {'Acc':<10} {'Sen':<10} {'Spe':<10} {'AUC':<10} {'Loss(Val)':<10}")
    print("-" * 80)
    
    epoch_list = []
    val_loss_list = []
    train_loss_list = []
    
    for fname in files:
        epoch = get_epoch(fname)
        try:
            data = np.load(fname)
            # Validate contents
            # train.py for binary saves: y_true, y_pred_logit (which might be 0/1 ints)
            if 'y_true' not in data:
                print(f"Skipping {fname}: missing y_true")
                continue
            
            y_true = data['y_true']
            
            if 'y_pred_logit' in data:
                raw_pred = data['y_pred_logit']
            elif 'y_prob' in data:
                raw_pred = data['y_prob']
            else:
                print(f"Skipping {fname}: missing prediction data")
                continue
            
            # Helper to get probs for plotting
            # If values are binary (0 or 1), treated as probs 0.0 or 1.0
            # If values are logits (e.g. -5 to 5), apply sigmoid
            if np.min(raw_pred) < 0 or np.max(raw_pred) > 1:
                # Apply Sigmoid for binary logits
                probs = 1 / (1 + np.exp(-raw_pred))
            else:
                probs = raw_pred.astype(float)

            # Metrics
            res = compute_metric_binary(y_true, probs)
            
            # Plot ROC
            plot_path = draw_roc_curve_binary(y_true, probs, epoch, plot_dir)
            
            # Store for Loss Plot
            epoch_list.append(epoch)
            val_loss_list.append(res['Loss'])
            if epoch in train_loss_map:
                train_loss_list.append(train_loss_map[epoch])
            else:
                train_loss_list.append(None)

            print(f"{epoch:<6} {res['Accuracy']:.4f}     {res['Sensitivity']:.4f}     {res['Specificity']:.4f}     {res['AUC']:.4f}     {res['Loss']:.4f}")
            
        except Exception as e:
            print(f"Error processing epoch {epoch}: {e}")

    # Draw Loss Curve
    if len(epoch_list) > 1:
        plt.figure(figsize=(10, 6))
        
        # Filter None from train_loss
        valid_train = [(e, l) for e, l in zip(epoch_list, train_loss_list) if l is not None]
        if valid_train:
            te, tl = zip(*valid_train)
            plt.plot(te, tl, label='Train Loss', color='orange', linewidth=2, marker='o')
        
        plt.plot(epoch_list, val_loss_list, label='Val Loss', color='blue', linewidth=2, marker='s')
        
        plt.title('Learning Curve (Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        loss_path = os.path.join(plot_dir, 'Loss_Curve.png')
        plt.savefig(loss_path)
        plt.close()
        print(f"\nLoss curve saved to: {loss_path}")

if __name__ == "__main__":
    main()
