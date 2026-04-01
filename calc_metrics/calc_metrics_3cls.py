import pandas as pd
import numpy as np
import os
import ast
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    log_loss,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

def compute_metric_3cls_from_labels(y_true, y_pred):
    """
    Computes 3-Class metrics when only discrete predictions are available.
    Skips probability-dependent metrics (AUC, Loss).
    """
    acc = accuracy_score(y_true, y_pred)

    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens_per_class = np.diag(cm) / row_sums
        sens_per_class = np.nan_to_num(sens_per_class)
    macro_sens = np.mean(sens_per_class)

    specs = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - tp)
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    macro_spec = np.mean(specs)

    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'Accuracy': acc,
        'Sensitivity_Macro': macro_sens,
        'Specificity_Macro': macro_spec,
        'F1_Macro': f1,
        'AP_Macro': np.nan,
        'AUC_Macro': np.nan,
        'Loss': np.nan,
    }

def compute_metric_3cls_standard(y_true, y_prob_raw):
    """
    Computes Standard 3-Class Metrics (Macro-averaged).
    Assumes y_prob_raw contains probability lists e.g. [p0, p1, p2].
    """
    # Parse the probability strings/lists
    try:
        # Check if first element is string
        if len(y_prob_raw) > 0 and isinstance(y_prob_raw[0], str):
            y_prob = np.array([ast.literal_eval(x) for x in y_prob_raw])
        else:
            y_prob = np.array(list(y_prob_raw))
    except Exception as e:
        print(f"Error parsing probabilities: {e}")
        return {}

    # Get predictions (class index with max probability)
    y_pred = np.argmax(y_prob, axis=1)
    
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # 2. Macro Sensitivity/Recall & Specificity
    # Note: confusion_matrix might be smaller than 3x3 if some classes are missing in y_true/y_pred,
    # but usually for 3-class problem we expect 0,1,2.
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Sensitivity per class = TP / (TP + FN)
    # TP is diag, FN is sum(axis=1) - TP
    # So TP / sum(axis=1)
    row_sums = cm.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens_per_class = np.diag(cm) / row_sums
        sens_per_class = np.nan_to_num(sens_per_class) # Replace nan with 0 if row sum is 0
    macro_sens = np.mean(sens_per_class)
    
    # Specificity per class = TN / (TN + FP)
    specs = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        tp = cm[i, i]
        # fn = row_sums[i] - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - tp)
        
        denom = tn + fp
        if denom > 0:
            specs.append(tn / denom)
        else:
            specs.append(0.0)
    macro_spec = np.mean(specs)

    # 3. Macro F1
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # 4. Macro AUC (One-vs-Rest)
    try:
        if y_prob.shape[1] > 1:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        else:
            auc = 0.5
    except:
        auc = 0.5
        
    # 5. Macro Average Precision (One-vs-Rest)
    try:
        classes = list(range(y_prob.shape[1]))
        y_true_bin = label_binarize(y_true, classes=classes)
        ap_macro = average_precision_score(y_true_bin, y_prob, average='macro')
    except Exception:
        ap_macro = np.nan

    # 6. Log Loss
    try:
        loss = log_loss(y_true, y_prob)
    except:
        loss = 0
        
    metrics = {
        'Accuracy': acc,
        'Sensitivity_Macro': macro_sens,
        'Specificity_Macro': macro_spec,
        'F1_Macro': f1,
        'AP_Macro': ap_macro,
        'AUC_Macro': auc,
        'Loss': loss
    }
    return metrics

def main():
    # Folder containing the prediction results
    # Updated to the folder used in test.sh
    folder_path = 'results/3b-rcbam-f12-ukb-3cls'
    
    file_path = os.path.join(folder_path, 'prediction.xlsx')
    output_path = os.path.join(folder_path, 'result_3cls.csv')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        # Try checking current directory just in case
        if os.path.exists('prediction.xlsx'):
            file_path = 'prediction.xlsx'
            output_path = 'result_3cls.csv'
            print(f"Found prediction.xlsx in current directory, using that.")
        # Try checking for CSVs if XLSX format is not found (as eval output might be CSV)
        elif os.path.exists(os.path.join(folder_path, 'prediction_test.csv')):
             file_path = os.path.join(folder_path, 'prediction_test.csv')
             print(f"Found {file_path}, using that (assuming it's formatted similarly).")
        else:
            print(f"Could not find prediction.xlsx in {folder_path} or current directory.")
            return

    print(f"Reading {file_path}...")
    try:
        # Determine loader
        if file_path.endswith('.xlsx'):
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            data_map = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names}
        else:
            # Assume CSV
            df = pd.read_csv(file_path)
            # Use filename or 'test' as key
            data_map = {'test': df}
            
        print(f"Found datasets: {list(data_map.keys())}")
        
        all_results = {}
        
        for sheet, df in data_map.items():
            print(f"Processing: {sheet}")
            
            # Drop obvious index columns
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

            if 'gt' not in df.columns:
                print(f"Skipping {sheet}: 'gt' column missing.")
                print(f"Columns found: {df.columns}")
                continue

            y_true = df['gt'].values

            y_prob_raw = None
            if 'pred' in df.columns:
                y_prob_raw = df['pred'].values
            elif set(['prob_0', 'prob_1', 'prob_2']).issubset(df.columns):
                y_prob_raw = df[['prob_0', 'prob_1', 'prob_2']].values

            y_pred_label = df['pred_label'].values if 'pred_label' in df.columns else None

            # Prefer probability-based metrics; fall back to label-only if needed
            if y_prob_raw is not None:
                print(f"Calculating 3-class metrics (probabilities) for {sheet}...")
                metrics = compute_metric_3cls_standard(y_true, y_prob_raw)
            elif y_pred_label is not None:
                print(f"Calculating 3-class metrics (labels only) for {sheet}...")
                metrics = compute_metric_3cls_from_labels(y_true, y_pred_label)
            else:
                print(f"Skipping {sheet}: no probability columns ('pred' or prob_0/1/2) and no 'pred_label'.")
                print(f"Columns found: {df.columns}")
                continue

            if metrics:
                all_results[sheet] = metrics
            
        if all_results:
            result_df = pd.DataFrame(all_results)
            print("\nCalculated 3-Class Metrics:")
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
