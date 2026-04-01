import pandas as pd
import numpy as np
import os
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, average_precision_score, log_loss)

def compute_metric_binary_standard(y_true, y_prob, threshold=0.5):
    """
    Computes Standard Binary Classification metrics (Class 1 vs Class 0).
    Focuses on the Positive Class (1).
    """
    # Predictions based on threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # 2. Confusion Matrix to derive Sensitivity & Specificity
    # TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Sensitivity (Recall of Class 1) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity (Recall of Class 0) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1 Score (Standard Binary F1 for Class 1)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # AUC (Standard ROC AUC)
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except:
        auc_val = 0.5
        
    # Standard Metrics Dictionary
    metrics = {
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1': f1,
        'AUC': auc_val,
    }
    
    # AP
    try:
        metrics['AP'] = average_precision_score(y_true, y_prob)
    except:
        metrics['AP'] = 0
        
    # Log Loss
    try:
        # Prepare prob matrix for log_loss
        probs = np.column_stack((1-y_prob, y_prob))
        metrics['Loss'] = log_loss(y_true, probs)
    except:
        metrics['Loss'] = 0
        
    return metrics

def main():
    # Folder containing the prediction results
    folder_path = 'results/3b-rcbam-f12-ukb' 
    
    file_path = os.path.join(folder_path, 'prediction.xlsx')
    output_path = os.path.join(folder_path, 'result2.csv')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        # Try checking current directory just in case
        if os.path.exists('prediction.xlsx'):
            file_path = 'prediction.xlsx'
            output_path = 'result2.csv'
            print(f"Found prediction.xlsx in current directory, using that.")
        else:
            return

    print(f"Reading {file_path}...")
    try:
        # Read Excel file
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        print(f"Found sheets: {sheet_names}")
        
        all_results = {}
        
        for sheet in sheet_names:
            print(f"Processing sheet: {sheet}")
            df = pd.read_excel(xls, sheet_name=sheet)
            
            # Check for required columns
            # prediction.xlsx usually has: filename, gt, pred, pred_label
            if 'gt' not in df.columns or 'pred_prob' not in df.columns:
                print(f"Skipping sheet {sheet}: 'gt' or 'pred' columns missing.")
                print(f"Columns found: {df.columns}")
                continue
                
            y_true = df['gt'].values
            y_prob = df['pred_prob'].values 
            
            # Changed to use standard binary metrics
            metrics = compute_metric_binary_standard(y_true, y_prob)
            all_results[sheet] = metrics
            
        if all_results:
            result_df = pd.DataFrame(all_results)
            print("\nCalculated Standard Binary Metrics:")
            print(result_df)
            result_df.to_csv(output_path)
            print(f"\nSaved results to {output_path}")
        else:
            print("No valid data processed.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
