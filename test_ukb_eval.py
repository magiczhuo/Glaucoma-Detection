import os
import csv
import torch
import sys
import numpy as np
import copy
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_curve, auc

# Append path to import internal modules
sys.path.append('./data')

# Import project modules
from networks.trainer import Patch5Model, Branch2CBAM, Branch3CBAM, Branch3RCBAM
from options.test_options import TestOptions
from metrics import compute_metric
from RETFound_Feature_Loader import RETFoundFeatureLoader
from data import create_dataloader_test

def validate(model, opt):
    """
    Validate/Test the model on the given data_loader.
    Re-implements logic from eval.py but specialized for single dataset pass logic if needed.
    """
    if opt.model_name == '3branch-rcbam':
        loader = RETFoundFeatureLoader()
        data_loader = create_dataloader_test(opt, loader)
    else:
        data_loader = create_dataloader_test(opt)
    
    with torch.no_grad():
        y_true, y_pred, filenames = [], [], []

        for data in tqdm(data_loader):
            # Unpack data based on dataset_ukb / create_dataloader_test structure
            # dataset_ukb returns: img, roi, target, scale, imgname, [retfound_feature]
            
            img = data[0]  # [batch_size, 3, height, width]
            roi = data[1].cuda()  # [batch_size, 3, 224, 224]
            full_img = copy.deepcopy(img).cuda() # Some models need full_img separately on GPU
            img = img.cuda()

            label = data[2].cuda()  # [batch_size]
            scale = data[3].cuda()  # [batch_size, 2]
            filename = data[4] # list of strings

            retfound_features = None
            if len(data) > 5:  # Check for retfound_feature
                raw_feature = data[5]
                if raw_feature is not None:
                    # Move to same device as model
                    retfound_features = raw_feature.to(next(model.parameters()).device)
            
            # Forward pass matching trainer.py / eval.py logic
            # Models typically: model(img, full_img, roi, scale, ret_feat=None)
            
            # Note: eval.py passes (img, full_img, roi, scale, retfound_features)
            # regardless of model type if features exist.
            if retfound_features is not None:
                 logits = model(img, full_img, roi, scale, retfound_features)
            else:
                 logits = model(img, full_img, roi, scale)
            
            # Post-processing logits
            if opt.mode == 'binary':
                 # Binary: flatten and sigmoid
                 preds = logits.sigmoid().flatten().tolist()
                 y_pred.extend(preds)
            else:
                 # Multiclass: softmax
                 preds = logits.softmax(dim=1).tolist()
                 y_pred.extend(preds)

            y_true.extend(label.flatten().tolist())
            filenames.extend(filename)
            
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Compute Metrics
    results = {}
    predict_logits = []
    
    if opt.mode == 'binary':
        # Binary Metrics: ROC, AUC, AP
        # Check if we have positive samples to calculate ROC
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            ap = average_precision_score(y_true, y_pred)
            results = compute_metric(y_true, y_pred, [0, 1], opt.test_threshold)
            results['AUC'] = roc_auc
            results['AP'] = ap
        else:
            print("Warning: Only one class present in y_true. ROC AUC not defined.")
            results = compute_metric(y_true, y_pred, [0, 1], opt.test_threshold)
            results['AUC'] = 0.0
            results['AP'] = 0.0
            
        predict_logits = (y_pred >= opt.test_threshold).astype(int)
    else:
        # Multiclass Metrics
        target_names = [str(i) for i in range(y_pred.shape[1])]
        results = compute_metric(y_true, y_pred, target_names, opt.test_threshold)
        predict_logits = np.argmax(y_pred, axis=1)

    # Detailed Predictions Dictionary
    predictions = {
        'filename': filenames,
        'gt': y_true,
        'pred_label': predict_logits
    }

    if opt.mode == 'binary':
        predictions['pred_prob'] = y_pred
    else:
        # Save prob for each class
        for i in range(y_pred.shape[1]):
            predictions[f'prob_{i}'] = y_pred[:, i]

    return results, predictions

if __name__ == '__main__':
    # Parse options
    # Important: User must pass --dataset_mode ukb
    opt = TestOptions().parse(print_options=True)
    
    # Overwrite specific options required for this test script logic
    opt.serial_batches = True  # no shuffle for testing
    opt.no_flip = True         # no data augmentation
    
    n_output = 1 if opt.mode == 'binary' else 3

    # Load Model Architecture based on path string (Heuristic matching eval.py logic)
    # eval.py uses opt.model_name to decide class.
    # Here we infer from path or rely on opt.model_name if passed.
    if opt.model_name == '3branch-rcbam':
        model = Branch3RCBAM(n_output)
    elif opt.model_name == '3branch-cbam':
        model = Branch3CBAM(n_output)
    elif opt.model_name == '2branch-cbam':
        model = Branch2CBAM(n_output)
    elif opt.model_name == '3branch':
        model = Patch5Model(n_output)
            
    print(f'Loading model: {model.__class__.__name__} with output dim {n_output}')

    # Load Weights
    print(f"Loading weights from {opt.model_path}")
    state_dict = torch.load(opt.model_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False) 
    model.cuda()
    model.eval()
    
    # Prepare RETFound Feature Loader (Required for RCBAM models)
    # Always initialize if possible, create_dataloader_test will pass it to dataset
    feature_loader = None
    # Assuming RETFound path is fixed or passed. Using hardcoded path from context.
    retfound_path = '/root/ZYZ/GRINLAB/RETFound/RETFound_cfp_weights.pth'
    if os.path.exists(retfound_path):
         print("Initializing RETFound Feature Loader...")
         feature_loader = RETFoundFeatureLoader()
    else:
         print(f"Warning: RETFound weights not found at {retfound_path}. Feature loader disabled.")

    
    # Run Validation
    print("Starting validation...")

    # Save original dataroot to restore or use as base
    base_dataroot = opt.dataroot
    
    if len(opt.testsets) == 0:
        print("Warning: No testsets specified. Using dataroot as is.")
        opt.testsets = ['']

    # Initialize collections
    all_results, all_predictions = {}, {}

    # Ensure output directory exists: result_path/name/
    save_dir = os.path.join(opt.result_path, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for v_id, val in enumerate(opt.testsets):
        print("testing {}-generated images".format(val))
        
        if val:
            opt.dataroot = os.path.join(base_dataroot, val)
        else:
            opt.dataroot = base_dataroot
            
        print(f"Data root set to: {opt.dataroot}")

        result, prediction = validate(model, opt)
        
        all_results[val] = result
        all_predictions[val] = prediction
        
        # Print Summary
        print("\n" + "="*30)
        print(f"      UKB TEST RESULTS [{val}]      ")
        print("="*30)
        for metric, v in result.items():
            if isinstance(v, (int, float)):
                print(f"{metric}: {v:.4f}")
            else:
                print(f"{metric}:\n{v}")
        print("="*30 + "\n")
        
        # Save .npz (Matching eval.py logic)
        y_prob = None
        if opt.mode == 'binary':
             # Handle both keys just in case
             y_prob = prediction.get('pred_prob', prediction.get('pred'))
        else:
             # Stack prob_i columns
             prob_keys = sorted([k for k in prediction.keys() if k.startswith('prob_')])
             if prob_keys:
                 y_prob = np.stack([prediction[k] for k in prob_keys], axis=1)

        npz_name = f'test_predictions_{val}.npz' if val else 'test_predictions.npz'
        np.savez(os.path.join(save_dir, npz_name),
                 filenames=prediction['filename'],
                 y_true=prediction['gt'],
                 y_prob=y_prob,
                 y_pred=prediction['pred_label']
                 )
        
        # Restore dataroot
        opt.dataroot = base_dataroot

    # Save summary result.csv
    result_df = pd.DataFrame(all_results)
    result_csv_path = os.path.join(save_dir, 'result.csv')
    result_df.to_csv(result_csv_path)
    print(f"Results saved to: {result_csv_path}")

    # Save prediction.xlsx if requested or default
    if opt.isRecord:
        xlsx_path = os.path.join(save_dir, 'prediction.xlsx')
        try:
            with pd.ExcelWriter(xlsx_path) as writer:
                for val in opt.testsets:
                    if val in all_predictions:
                        # Convert dict to DF
                        sheet_name = val if val else 'default'
                        df = pd.DataFrame(all_predictions[val])
                        df.to_excel(writer, sheet_name=sheet_name)
            print(f"Detailed predictions saved to: {xlsx_path}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")
