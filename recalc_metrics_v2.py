import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score, average_precision_score, log_loss
import os

def calculate_macro_metrics(file_path, output_path):
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    # Read the Excel file
    try:
        # Load the first sheet
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading excel file: {e}")
        return

    print("Columns found:", df.columns.tolist())
    
    # Identify columns
    col_gt = 'gt'
    col_pred_prob = 'pred'
    col_pred_label = 'pred_label' # Default
    
    # Handle possible column name variations based on user feedback/screenshot
    if col_pred_label not in df.columns:
        for c in df.columns:
             if 'label' in str(c) and 'pred' in str(c):
                 col_pred_label = c
                 break
             # Handle the specific case seen in screenshot if it's 'red_label' (substring of pred_label cut off?)
             if 'red_label' in str(c): 
                 col_pred_label = c
                 break
    
    if col_gt not in df.columns or col_pred_prob not in df.columns or col_pred_label not in df.columns:
        print(f"Could not find necessary columns. Looking for: {col_gt}, {col_pred_prob}, {col_pred_label}")
        print("Columns available:", df.columns.tolist())
        return

    print(f"Using columns: GT='{col_gt}', PredProb='{col_pred_prob}', PredLabel='{col_pred_label}'")

    y_true = df[col_gt].values
    y_pred_label = df[col_pred_label].values
    y_pred_prob = df[col_pred_prob].values 

    # --- Macro Metrics Calculation ---
    
    # Accuracy (Global)
    accuracy = accuracy_score(y_true, y_pred_label)
    
    # Macro F1
    f1_macro = f1_score(y_true, y_pred_label, average='macro')
    
    # Macro Sensitivity (Recall)
    sensitivity_macro = recall_score(y_true, y_pred_label, average='macro')
    
    # Macro Precision
    precision_macro = precision_score(y_true, y_pred_label, average='macro')
    
    # Macro Specificity
    # Specificity is Recall of the negative class.
    # For binary 0/1:
    # Spec_class0 = Recall_class1
    # Spec_class1 = Recall_class0
    # Macro Spec = (Spec_0 + Spec_1)/2 = (Recall_1 + Recall_0)/2 = Macro Recall
    # So Sensitivity and Specificity in Macro Average for Binary Classification yield the same number.
    
    # Let's calculate manually to confirm and follow typical "Macro Average" conventions
    cm = confusion_matrix(y_true, y_pred_label)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # For Class 0 (Negative=1): TP_0 = TN, FN_0 = FP. Spec_0 = True Negative Rate for Class 0 = "Correctly identified as NOT 0" / "Total NOT 0"
        # NOT 0 is 1. Identified as NOT 0 means Identified as 1.
        # "Correctly identified as 1" is TP. "Total 1" is TP+FN.
        # So Spec_0 = TP / (TP+FN) = Sensitivity (Recall for class 1)
        
        # For Class 1 (Negative=0): TP_1 = TP, FN_1 = FN. Spec_1 = TN / (TN+FP)
        
        # Macro Spec = (Spec_0 + Spec_1) / 2 = (Sensitivity + Specificity) / 2
        spec_1 = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_0 = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        specificity_macro = (spec_1 + spec_0) / 2
    else:
        specificity_macro = 0 
        
    # Validation Loss estimate
    try:
        # Construct probability matrix for log_loss
        # y_pred_prob is prob of class 1.
        y_prob_mat = np.column_stack([1-y_pred_prob, y_pred_prob])
        loss = log_loss(y_true, y_prob_mat)
    except:
        loss = 0

    # AUC
    try:
         auc_score = roc_auc_score(y_true, y_pred_prob)
    except:
        auc_score = 0
        
    # AP
    ap_score = average_precision_score(y_true, y_pred_prob)

    # --- Save Results ---
    
    results_dict = {
        'class_name': "[0, 1]",
        'F1': f1_macro,
        'AUC': auc_score,
        'AP': ap_score,
        'loss': loss,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity_macro,
        'Specificity': specificity_macro
    }
    
    # Reconstruct DataFrame matching original format
    # Index: Metric names. Column: 'test' (dataset name)
    
    index_order = ['class_name', 'F1', 'AUC', 'AP', 'loss', 'Accuracy', 'Sensitivity', 'Specificity']
    data = [results_dict.get(k, 0) for k in index_order]
    
    result_df = pd.DataFrame(data, index=index_order, columns=['test'])
    
    print("\nCalculated Macro Metrics:")
    print(result_df)
    
    result_df.to_csv(output_path)
    print(f"\nSaved result to {output_path}")

if __name__ == "__main__":
    # Adjust paths as per workspace structure
    pred_file = '/root/ZYZ/GRINLAB/results/3b-test-best/prediction.xlsx'
    result_file = '/root/ZYZ/GRINLAB/results/3b-test-best/result.csv'
    
    calculate_macro_metrics(pred_file, result_file)
