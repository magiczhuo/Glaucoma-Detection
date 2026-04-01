import os
import csv
import torch
from networks.trainer import Patch5Model, Branch2CBAM, Branch3CBAM, Branch3RCBAM
from networks.resnet import resnet50
from options.test_options import TestOptions
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_curve, auc
import sys

sys.path.append('./data')
from data import create_dataloader_test
import numpy as np
from PIL import ImageFile
import copy
from metrics import compute_metric

ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import pandas as pd

from RETFound_Feature_Loader import RETFoundFeatureLoader



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
            
            if opt.mode == 'binary':
                y_pred.extend(logits.sigmoid().flatten().tolist())
            else:
                y_pred.extend(logits.softmax(dim=1).tolist())

            y_true.extend(label.flatten().tolist())
            filenames.extend(filename)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if opt.mode == 'binary':
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y_true, y_pred)
        results = compute_metric(y_true, y_pred, [0, 1], opt.test_threshold)
        predict_logits = (y_pred >= opt.test_threshold).astype(int)
        results['AUC'] = roc_auc
        results['AP'] = ap
    else:
        target_names = [i for i in range(y_pred.shape[1])]
        results = compute_metric(y_true, y_pred, target_names, opt.test_threshold)
        predict_logits = np.argmax(y_pred, axis=1)

    predictions = {
        'filename': filenames,
        'gt': y_true,
        'pred_label': predict_logits
    }

    if opt.mode == 'binary':
        predictions['pred'] = y_pred
    else:
        for i in range(y_pred.shape[1]):
            predictions[f'prob_{i}'] = y_pred[:, i]

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
    print(f'Load {opt.model_name} successfully with output dim {n_output}')

    state_dict = torch.load(opt.model_path, map_location='cpu')
    # model.load_state_dict(
    #     state_dict['model'],strict = False
    # )  # use this if testing model is trained on single GPU
    ## uncomment following lines if testing model is trained on multiple GPUs
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

        if 'pred' in prediction:
             y_prob = prediction['pred']
        else:
             # Stack prob_i columns
             prob_keys = sorted([k for k in prediction.keys() if k.startswith('prob_')])
             y_prob = np.stack([prediction[k] for k in prob_keys], axis=1)

        # Save probabilities to npz for analysis
        np.savez(os.path.join(opt.result_path, opt.name, f'test_predictions_{val}.npz'),
                 filenames=prediction['filename'],
                 y_true=prediction['gt'],
                 y_prob=y_prob,
                 y_pred=prediction['pred_label']
                 )

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(opt.result_path, opt.name, 'result.csv'))

    if opt.isRecord:
        for v_id, val in enumerate(opt.testsets):
            with pd.ExcelWriter(
                    os.path.join(opt.result_path, opt.name,
                                 'prediction.xlsx')) as writer:
                df = pd.DataFrame(predictions[val],
                                  columns=predictions[val].keys())
                df.to_excel(writer, sheet_name=val)
