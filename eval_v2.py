import os
import csv
import torch
from networks.trainer import Patch5Model, Branch2CBAM, Branch3CBAM, Branch3RCBAM
from networks.resnet import resnet50
from options.test_options import TestOptions
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, log_loss
import sys

sys.path.append('./data')
from data import create_dataloader_test
import numpy as np
from PIL import ImageFile
import copy
# from metrics import compute_metric # We will use internal function

ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import pandas as pd

from RETFound_Feature_Loader import RETFoundFeatureLoader

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
            try:
                c_tn, c_fp, c_fn, c_tp = confusion_matrix(tmp_label, tmp_pred).ravel()
                tn[i], fp[i], fn[i], tp[i] = c_tn, c_fp, c_fn, c_tp
            except ValueError:
                # Handle cases where confusion matrix is smaller (e.g., missing classes in batch)
                tn[i], fp[i], fn[i], tp[i] = 0, 0, 0, 0 

        try:
            if n_class == 2:
                 outAUROC = roc_auc_score(tmp_label, probs[:, 1])
            else:
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
    
    # AP Calculation
    ap = 0
    try:
        if n_class == 2:
            y_score = probs[:, 1]
            ap = average_precision_score(datanpGT, y_score)
        else:
            y_true_onehot = np.eye(n_class)[datanpGT]
            ap = average_precision_score(y_true_onehot, probs, average='macro')
    except Exception:
        ap = 0

    # Loss Calculation
    v_loss = 0
    try:
        if n_class == 2:
             v_loss = log_loss(datanpGT, probs, labels=[0, 1])
        else:
            # Need to ensure all labels are present or passed explicitly
             v_loss = log_loss(datanpGT, probs, labels=list(range(n_class)))
    except Exception:
        v_loss = 0

    return {
        'Accuracy': Accuracy_score,
        'Sensitivity': mSensitivity,
        'Specificity': mSpecificity,
        'F1': mF1,
        'AUC': mAUC / n_class,
        'AP': ap,
        'Loss': v_loss
    }



def validate(model, opt):
    if opt.model_name == '3branch-rcbam':
        loader = RETFoundFeatureLoader()
        data_loader = create_dataloader_test(opt, loader)
    else:
        data_loader = create_dataloader_test(opt)
    with torch.no_grad():
        y_true, y_pred, filenames = [], [], []

        for data in tqdm(data_loader):
            img = data[0]  # [batch_size, 3, height, width]
            full_img = copy.deepcopy(img)
            roi = data[1].cuda()  # [batch_size, 3, 224, 224]
            full_img = full_img.cuda()

            roi = data[1].cuda()  # [batch_size, 3, 224, 224]
            label = data[2].cuda()  # [batch_size, 1]
            scale = data[3].cuda()  # [batch_size, 1, 2]
            filename = data[4]

            retfounf_features = None
            if len(data) > 5:  # retfound_feature
                raw_feature = data[5]
                if raw_feature is not None:
                    retfounf_features = raw_feature.to(
                        next(model.parameters()).device)
            if len(data) > 5:
                logits = model(img, full_img, roi, scale, retfounf_features)
            else:
                logits = model(img, full_img, roi, scale)
            # Store (B, C) logits/probs instead of flattening
            # Use softmax for multi-class to ensure sum to 1
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                probs = torch.cat([1-probs, probs], dim=1)
                y_pred.extend(probs.cpu().tolist())
            else:
                y_pred.extend(logits.softmax(dim=1).cpu().tolist())
            y_true.extend(label.flatten().cpu().tolist())
            filenames.extend(filename)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Handle Metrics
    n_classes = y_pred.shape[1]
    if n_classes == 1: # Binary logical but output might be single probability
         n_classes = 2 # Assume binary if 1 output? Usually networks output 2 for binary or 1 sigmoid.
         # This part depends on model. Logic below assumes y_pred has appropriate shape.
    
    # Assuming y_pred is logits or probabilities with shape (N, n_classes)
    target_names = [str(i) for i in range(n_classes)]
    results = compute_metric_macro(y_true, y_pred, target_names, opt.test_threshold)
    
    # Calculate predicted class indices (argmax) for saving
    if np.min(y_pred) < 0 or np.max(y_pred) > 1:
        # Softmax if logits
        exp_x = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        probs = y_pred

    if n_classes == 2:
         predict_logits = np.array([1 if p[1] >= opt.test_threshold else 0 for p in probs])
    else:
         predict_logits = np.argmax(probs, axis=1)

    predictions = {
        'filename': filenames,
        'gt': y_true,
        'pred_label': predict_logits
    }
    
    # Split probability columns for DataFrame compatibility
    if probs.ndim > 1:
        for i in range(probs.shape[1]):
            predictions[f'prob_{i}'] = probs[:, i]
    else:
        predictions['prob'] = probs

    return results, predictions


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=True, mode='eval')

    n_output = 1 if opt.mode == 'binary' else 3
    
    if opt.model_name == '3branch':
        model = Patch5Model(n_output)
    elif opt.model_name == '2branch-cbam':
        model = Branch2CBAM(n_output)
    elif opt.model_name == '3branch-cbam':
        model = Branch3CBAM(n_output)
    elif opt.model_name == '3branch-rcbam':
        model = Branch3RCBAM(n_output)
    print(f'Load {opt.model_name} successfully with output dim: {n_output}')

    state_dict = torch.load(opt.model_path, map_location='cpu')
    # model.load_state_dict(state_dict['model'],strict = False)
    
    # use this if testing model is trained on single GPU
    ## uncomment following lines if testing model is trained on multiple GPUs
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()
    #for k, v in state_dict['model'].items():
    #    name = k[7:] # remove `module.`
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['model'].items():
       name = k[7:] if k.startswith('module.') else k # remove `module.`
       new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    results, predictions = {}, {}
    for v_id, val in enumerate(opt.testsets):
        print("testing {}-generated images".format(val))
        opt.dataroot = '{}/{}'.format(opt.dataroot, val)
        # opt.no_resize = True    # testing without resizing by default

        result, prediction = validate(model, opt)
        results[val] = result
        predictions[val] = prediction
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(opt.result_path, opt.name, 'result.csv'))

    if opt.isRecord:
        for v_id, val in enumerate(opt.testsets):
            """with pd.ExcelWriter(
                    os.path.join(opt.result_path, opt.name,
                                 'prediction.xlsx')) as writer:
                df = pd.DataFrame(predictions[val],
                                  columns=predictions[val].keys())
                df.to_excel(writer, sheet_name=val)"""
            # 改为保存 CSV，避免 Excel/openpyxl 的复杂依赖和 Index 错误
            csv_path = os.path.join(opt.result_path, opt.name, f'prediction_{val}.csv')
            
            # 构建 DataFrame
            # 注意：'pred' 列包含数组/列表，CSV 会将其保存为字符串形式 "[0.1, 0.9...]"，这通常是可以接受的
            df = pd.DataFrame(predictions[val])
            
            # 保存
            df.to_csv(csv_path, index=False)
            print(f"[{val}] Detailed predictions saved to: {csv_path}")


            
