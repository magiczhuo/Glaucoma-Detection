import os
import glob
import numpy as np
import pandas as pd
import matplotlib
# Set backend to 'Agg' to write to file without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve, auc, average_precision_score)
from sklearn.preprocessing import label_binarize

def compute_metric_macro(datanpGT, datanpPRED, target_names, threshold):
    # Ensure inputs are probabilities
    if np.min(datanpPRED) < 0 or np.max(datanpPRED) > 1:
        # Assuming logits, apply softmax
        exp_x = np.exp(datanpPRED - np.max(datanpPRED, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        probs = datanpPRED

    n_class = len(target_names)
    if n_class == 2:
        argmaxPRED = np.array([1 if i[1] >= threshold else 0 for i in probs])
    else:
        argmaxPRED = np.argmax(probs, axis=1)

    Accuracy_score = accuracy_score(datanpGT, argmaxPRED)
    
    F1_metric = np.zeros((n_class, ))
    tn = np.zeros((n_class, ))
    fp = np.zeros((n_class, ))
    fn = np.zeros((n_class, ))
    tp = np.zeros((n_class, ))
    
    mAUC = 0
    
    # Calculate per-class metrics
    for i in range(n_class):
        tmp_label = datanpGT == i
        tmp_pred = argmaxPRED == i
        F1_metric[i] = f1_score(tmp_label, tmp_pred)

        if n_class == 2:
             c_tn, c_fp, c_fn, c_tp = confusion_matrix(tmp_label, tmp_pred).ravel()
             tn[i], fp[i], fn[i], tp[i] = c_tn, c_fp, c_fn, c_tp
        else:
            c_tn, c_fp, c_fn, c_tp = confusion_matrix(tmp_label, tmp_pred).ravel()
            tn[i], fp[i], fn[i], tp[i] = c_tn, c_fp, c_fn, c_tp

        try:
            outAUROC = roc_auc_score(tmp_label, probs[:, i])
        except ValueError:
            outAUROC = 0.5
        mAUC += outAUROC

    # --- MACRO AVERAGE CALCULATION ---
    denom_sens = tp + fn
    sens_list = np.divide(tp, denom_sens, out=np.zeros_like(tp), where=(denom_sens!=0))
    mSensitivity = np.mean(sens_list)
    
    denom_spec = tn + fp
    spec_list = np.divide(tn, denom_spec, out=np.zeros_like(tn), where=(denom_spec!=0))
    mSpecificity = np.mean(spec_list)
    
    mF1 = np.mean(F1_metric)
    
    try:
        if n_class == 2:
            y_score = probs[:, 1]
            ap = average_precision_score(datanpGT, y_score)
        else:
            y_true_onehot = np.eye(n_class)[datanpGT]
            ap = average_precision_score(y_true_onehot, probs, average='macro')
    except Exception:
        ap = 0

    return {
        'Accuracy': Accuracy_score,
        'Sensitivity': mSensitivity,
        'Specificity': mSpecificity,
        'F1': mF1,
        'AUC': mAUC / n_class,
        'AP': ap
    }

def draw_roc_curve(y_true, y_probs, n_classes, epoch, save_dir):
    """
    Draw ROC Curve for Multi-class and save to file
    """
    # 1. Binarize the output for roc_curve (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    # Fix for n_classes=2 where label_binarize returns column vector
    # If 2 classes, roc_curve usually expects y_true to be 0 or 1.
    # But to unify multi-class logic, we can make it 2 columns (one-hot).
    if n_classes == 2:
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # 2. Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 3. Compute Macro-average ROC curve
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 4. Plot all ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC (area = {0:0.4f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'magenta'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC (Epoch {epoch})')
    plt.legend(loc="lower right")
    
    # Save fig
    save_path = os.path.join(save_dir, f'ROC_Epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path

def main():
    checkpoint_dir = '/root/ZYZ/GRINLAB/checkpoints/1-resnet152rcbam-3b-3cls-f12'
    plot_dir = os.path.join(checkpoint_dir, 'ROC_Plots')
    os.makedirs(plot_dir, exist_ok=True)
    
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
    print(f"{'Epoch':<6} {'Acc':<10} {'Sen(Mac)':<10} {'Spe(Mac)':<10} {'AUC(Mac)':<10} {'Plot Saved'}")
    print("-" * 90)
    
    all_benchmark_results = []

    for fname in files:
        epoch = get_epoch(fname)
        try:
            data = np.load(fname)
            if 'y_true' not in data or 'y_pred_logit' not in data:
                continue
                
            y_true = data['y_true']
            y_pred_logit = data['y_pred_logit']
            
            # Helper to get probs for plotting
            if np.min(y_pred_logit) < 0 or np.max(y_pred_logit) > 1:
                exp_x = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))
                probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            else:
                probs = y_pred_logit

            n_classes = probs.shape[1]
            target_names = [str(i) for i in range(n_classes)]
            
            # Metrics
            res = compute_metric_macro(y_true, probs, target_names, 1.0/n_classes)
            
            # Plot
            plot_path = draw_roc_curve(y_true, probs, n_classes, epoch, plot_dir)
            
            print(f"{epoch:<6} {res['Accuracy']:.4f}     {res['Sensitivity']:.4f}     {res['Specificity']:.4f}     {res['AUC']:.4f}         Yes")
            
            res['Epoch'] = epoch
            all_benchmark_results.append(res)
            
        except Exception as e:
            print(f"Error processing epoch {epoch}: {e}")

    if all_benchmark_results:
        df = pd.DataFrame(all_benchmark_results)
        cols = ['Epoch', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AP', 'Loss']
        # Filter only existing columns
        existing_cols = [c for c in cols if c in df.columns]
        # Add any other columns that might be in res but not in our list, if any
        remaining_cols = [c for c in df.columns if c not in existing_cols]
        df = df[existing_cols + remaining_cols]
        
        save_csv_path = os.path.join(checkpoint_dir, 'metrics_summary.csv')
        df.to_csv(save_csv_path, index=False)
        print(f"\nResults saved to {save_csv_path}")

if __name__ == "__main__":
    main()
