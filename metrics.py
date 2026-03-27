import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve, average_precision_score, log_loss)


def compute_metric(datanpGT, datanpPRED, target_names, threshold):
    
    # Check reshape behavior for (N, 1) input
    if datanpPRED.ndim == 2 and datanpPRED.shape[1] == 1:
        datanpPRED = datanpPRED.flatten()
        
    # Handle 1D input (binary classification probabilities/logits for class 1)
    if datanpPRED.ndim == 1:
        if np.min(datanpPRED) < 0 or np.max(datanpPRED) > 1:
             # Assume 1D logits, convert to probabilities via sigmoid
             datanpPRED = 1 / (1 + np.exp(-datanpPRED))
        # Now datanpPRED assumes probabilities of class 1
        datanpPRED = np.stack([1 - datanpPRED, datanpPRED], axis=1)

    # Pre-process: Convert logits to probabilities if necessary
    # Check if values are likely logits (not in [0, 1])
    # Note: This is a heuristic. Ideally, the caller should specify.
    # But safe softmax usually helps for AUC/LogLoss if inputs are logits.
    if np.min(datanpPRED) < 0 or np.max(datanpPRED) > 1:
        # Assuming logits, apply stable softmax
        exp_x = np.exp(datanpPRED - np.max(datanpPRED, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        probs = datanpPRED

    n_class = len(target_names)
    
    # Adjust n_class if input was binary but target_names implied 1 class
    if probs.shape[1] == 2 and n_class == 1:
        n_class = 2

    if n_class == 2:
        argmaxPRED = np.array([1 if i[1] >= threshold else 0 for i in probs])
    elif n_class == 3:
        argmaxPRED = np.argmax(probs, axis=1)

    # print(datanpGT)
    # print(datanpPRED)
    # print(argmaxPRED)
    # F1_metric = np.zeros((n_class, ))
    # tn = np.zeros((n_class, ))
    # fp = np.zeros((n_class, ))
    # fn = np.zeros((n_class, ))
    # tp = np.zeros((n_class, ))

    Accuracy_score = accuracy_score(datanpGT, argmaxPRED)
    
    F1_metric = np.zeros((n_class, ))
    tn = np.zeros((n_class, ))
    fp = np.zeros((n_class, ))
    fn = np.zeros((n_class, ))
    tp = np.zeros((n_class, ))
    
    ROC_curve = {}
    mAUC = 0
    # tn, fp, fn, tp = confusion_matrix(datanpGT, argmaxPRED).ravel()
    # F1_metric = f1_score(datanpGT, argmaxPRED)
    for i in range(n_class):
        tmp_label = datanpGT == i
        tmp_pred = argmaxPRED == i
        F1_metric[i] = f1_score(tmp_label, tmp_pred)

        if n_class == 2:
            # For binary, usually we care about class 1. The loop runs for 0 then 1.
            # If we want per-class metrics, we should store them.
            # However, original code structure suggests simple storage.
            # Let's use the array storage to be safe for n_class=2 as well.
            c_tn, c_fp, c_fn, c_tp = confusion_matrix(tmp_label, tmp_pred).ravel()
            tn[i], fp[i], fn[i], tp[i] = c_tn, c_fp, c_fn, c_tp
        else:
            tn[i], fp[i], fn[i], tp[i] = confusion_matrix(tmp_label, tmp_pred).ravel()

        # Moved INSIDE the loop
        try:
            # Use probabilities for AUC
            outAUROC = roc_auc_score(tmp_label, probs[:, i])
        except ValueError:
            outAUROC = 0.5 # Handle case where only one class is present in batch
            
        mAUC = mAUC + outAUROC
        [roc_fpr, roc_tpr, roc_thresholds] = roc_curve(tmp_label, probs[:, i])

        ROC_curve.update({'ROC_fpr_'+str(i): roc_fpr,
                          'ROC_tpr_' + str(i): roc_tpr,
                          'ROC_T_' + str(i): roc_thresholds,
                          'AUC_' + str(i): outAUROC})

    # mPrecision = sum(tp) / sum(tp + fp)
    # mRecall = sum(tp) / sum(tp + fn)
    
    # Avoid division by zero
    # with np.errstate(divide='ignore', invalid='ignore'):
    #    Sensitivity = tp / (tp + fn)
    #    Specificity = tn / (fp + tn)
        # Precision = tp / (tp + fp)
    
    # Replace NaNs with 0
    # Sensitivity = np.nan_to_num(Sensitivity)
    # Specificity = np.nan_to_num(Specificity)

    # Calculate Micro-averages (Summing TP, TN, FP, FN across all classes)
    sum_tp = np.sum(tp)
    sum_tn = np.sum(tn)
    sum_fp = np.sum(fp)
    sum_fn = np.sum(fn)

    mPrecision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    mSensitivity = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    mRecall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    mSpecificity = sum_tn / (sum_fp + sum_tn) if (sum_fp + sum_tn) > 0 else 0
    mF1 = 2 * mPrecision * mRecall / (mPrecision + mRecall) if (mPrecision + mRecall) > 0 else 0
    
    """
    # Calculate Macro-averages (Average of per-class metrics)
    # This decouples Sensitivity from Accuracy and reflects performance on all classes equally.
    
    # Sensitivity (Recall) per class
    denom_sens = tp + fn
    sens_list = np.divide(tp, denom_sens, out=np.zeros_like(tp), where=(denom_sens!=0))
    mSensitivity = np.mean(sens_list)
    mRecall = mSensitivity

    # Specificity per class
    denom_spec = tn + fp
    spec_list = np.divide(tn, denom_spec, out=np.zeros_like(tn), where=(denom_spec!=0))
    mSpecificity = np.mean(spec_list)

    # Precision per class (Optional, for F1 check)
    denom_prec = tp + fp
    prec_list = np.divide(tp, denom_prec, out=np.zeros_like(tp), where=(denom_prec!=0))
    mPrecision = np.mean(prec_list)

    # Macro F1 (Average of per-class F1s calculated in the loop)
    mF1 = np.mean(F1_metric)
    """
    # Calculate AUC and Average Precision
    try:
        if n_class == 2:
            # For binary classification, use the probability of the positive class (class 1)
            # Assuming datanpPRED has shape [n_samples, 2]
            y_score = probs[:, 1]
            ap = average_precision_score(datanpGT, y_score)
        else:
            # For AP in multi-class, need one-hot encoded labels
            y_true_onehot = np.eye(n_class)[datanpGT]
            ap = average_precision_score(y_true_onehot, probs, average='macro')
    except Exception as e:
        print(f"Metric calculation error: {e}")
        ap = 0
        
    # Calculate Log Loss (Validation Loss)
    try:
        # log_loss computes the cross-entropy loss
        # datanpGT: [n_samples] (true labels)
        # datanpPRED: [n_samples, n_classes] (predicted probabilities)
        v_loss = log_loss(datanpGT, probs, labels=list(range(n_class)))
    except Exception as e:
        print(f"Loss calculation error: {e}")
        v_loss = 0

    output = {
        'class_name': target_names,
        'F1': mF1,
        'AUC': mAUC / n_class,
        'AP': ap,
        'loss': v_loss,
        'Accuracy': Accuracy_score,
        'Sensitivity': mSensitivity,
        # 'Precision': tp / (tp + fp),
        'Specificity': mSpecificity,
        'ROC_curve': ROC_curve,
        # 'ROC_curve': ROC_curve,
        # 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,

        # 'micro-Precision': mPrecision,
        # 'micro-Sensitivity': mRecall,
        # 'micro-Specificity': sum(tn) / sum(fp + tn),
        # 'micro-F1': 2*mPrecision * mRecall / (mPrecision + mRecall),
    }
    # print(output['Accuracy'].shape, output['Sensitivity'].shape,
    #       output['Specificity'].shape, output['F1'].shape)
    return output
