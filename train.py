import argparse
import copy
import os
import sys
import time

# import comet_ml
import numpy as np
import torch
import torch.nn as nn
from earlystop import EarlyStopping
from metrics import compute_metric
from networks.trainer import Trainer
from options.train_options import TrainOptions
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_curve)
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append('./data')

comet_conf = {
    "api_key": "1JGHAJOFsdqx5MDgirg3HPvzW",
    "project_name": "GON",
    "disabled": False
}

from data import create_dataloader  # NOQA
from RETFound_Feature_Loader import RETFoundFeatureLoader

def validate(model, opt):
    device = next(model.parameters()).device
    loader = RETFoundFeatureLoader()
    data_loader = create_dataloader(opt, loader)
    print("number of validation dataset: ", len(data_loader))
    with torch.no_grad():
        y_true, y_pred_list, y_logits_list = [], [], []
        for data in tqdm(data_loader):
            img = data[0]  # [batch_size, 3, height, width]
            full_img = copy.deepcopy(img)
            full_img = full_img.cuda()
            roi = data[1].cuda()  # [batch_size, 3, 224, 224]
            label = data[2].cuda()  # [batch_size, 1]
            scale = data[3].cuda()  # [batch_size, 1, 2]
            retfound_features = None
            if len(data) > 5: # retfound_feature
                raw_feature = data[5]
                if raw_feature is not None:
                    retfound_features = raw_feature.to(device)
            
            # Forward pass
            if opt.model_name == '3branch-rcbam':
                pred = model(img, full_img, roi, scale, retfound_features)
            else:
                pred = model(img, full_img, roi, scale)
            
            if opt.mode == 'binary':
                # Binary classification: output is [Batch, 1], use sigmoid
                batch_prob = pred.sigmoid()
                y_pred_list.extend(batch_prob.flatten().tolist())
                y_logits_list.extend(pred.flatten().tolist())
            else:
                # Multi-class: output is [Batch, N], use softmax
                batch_prob = pred.softmax(dim=1)
                y_pred_list.extend(batch_prob.tolist())
                y_logits_list.extend(pred.tolist())
            
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred_list)
    y_logits = np.array(y_logits_list)
    
    if opt.mode == 'binary':
        # For binary, follow eval.py logic
        # target_names=[0, 1] (length 2) ensures metrics.py works correctly
        threshold = 0.5 # binary threshold for class 1
        results = compute_metric(y_true, y_pred, [0, 1], threshold)
        results['predict_logits'] = y_logits
    else:
        # For multi-class (e.g., 3cls)
        n_classes = len(y_pred[0])
        results = compute_metric(y_true, y_pred, range(n_classes), 1/n_classes)
        results['predict_logits'] = y_logits
    
    results['y_true'] = y_true
    return results


def get_val_opt():
    opt = TrainOptions()
    val_opt = opt.parse(print_options=False, mode='val')
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    # val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, 'test')
    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.data_aug = False
    val_opt.batch_size = 64
    opt.print_options(val_opt, mode='val')
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse(print_options=True, mode='train')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()

    loader = RETFoundFeatureLoader()

    data_loader = create_dataloader(opt,loader)
    dataset_size = len(data_loader)
    print('#training batch counts = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name,
                                              "train"),
                                 comet_config=comet_conf)
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name,
                                            "val"),
                               comet_config=comet_conf)

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch,
                                   delta=-0.0001,
                                   verbose=True)
    for epoch in range(opt.epoch_count, opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0
        train_loss_accumulator = 0.0
        train_batches = 0

        for i, data in enumerate(tqdm(data_loader)):
            iter_start_time = time.time()
            model.total_steps += 1
            epoch_iter += opt.batch_size

            # print("data loader中数据长度：",len(data))
            model.set_input(data)
            model.optimize_parameters()
            
            # Accumulate Loss
            train_batches += 1
            if isinstance(model.loss, torch.Tensor):
                train_loss_accumulator += model.loss.item()
            else:
                train_loss_accumulator += model.loss

            # exit()

            if model.total_steps % opt.loss_freq == 0:
                iter_end_time = time.time()
                tqdm.write("Train loss: {} at step: {}. Time: {}".format(
                    model.loss, model.total_steps,
                    iter_end_time - iter_start_time))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                tqdm.write(
                    'saving the latest model %s (epoch %d, model.total_steps %d)'
                    % (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            tqdm.write('saving the model at the end of epoch %d, iters %d' %
                       (epoch, model.total_steps))
            # model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        results = validate(model.model, val_opt)
        
        # Tensorboard Logging
        val_writer.add_scalar('Accuracy', results['Accuracy'], model.total_steps)
        val_writer.add_scalar('Sensitivity', results['Sensitivity'], model.total_steps)
        val_writer.add_scalar('Specificity', results['Specificity'], model.total_steps)
        val_writer.add_scalar('F1', results['F1'], model.total_steps)
        val_writer.add_scalar('Val_Loss', results['loss'], model.total_steps)
        val_writer.add_scalar('AUC', results['AUC'], model.total_steps)
        val_writer.add_scalar('AP', results['AP'], model.total_steps)
        epoch_end_time = time.time()
        print(f"(Val @ epoch {epoch}) Acc: {results['Accuracy']:.6f}, Sen: {results['Sensitivity']:.6f}, "
              f"Spe: {results['Specificity']:.6f}, F1: {results['F1']:.6f}, "
              f"AUC: {results['AUC']:.6f}, AP: {results['AP']:.6f}, Val_Loss: {results['loss']:.6f}, Time: {epoch_end_time - epoch_start_time:.2f}s")
        
        # Save Raw Predictions for Plotting (Saved as .npz)
        np.savez(os.path.join(opt.checkpoints_dir, opt.name, f'val_predictions_epoch_{epoch}.npz'), 
                 y_true=results['y_true'], 
                 y_pred_logit=results['predict_logits'])

        # CSV Logging for easier checking
        csv_path = os.path.join(opt.checkpoints_dir, opt.name, 'training_log.csv')
        # Training Loss (Average over epoch)
        if train_batches > 0:
            train_loss_val = train_loss_accumulator / train_batches
        else:
            if hasattr(model, 'loss'):
                train_loss_val = model.loss.item() if isinstance(model.loss, torch.Tensor) else model.loss
            else:
                train_loss_val = 0
             
        csv_header = "Epoch,Step,Accuracy,Sensitivity,Specificity,F1,AUC,AP,Val_Loss,Train_Loss\n"
        csv_line = f"{epoch},{model.total_steps},{results['Accuracy']:.4f},{results['Sensitivity']:.4f},{results['Specificity']:.4f},{results['F1']:.4f},{results['AUC']:.4f},{results['AP']:.4f},{results['loss']},{train_loss_val}\n"
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(csv_header)
        with open(csv_path, 'a') as f:
            f.write(csv_line)

        # Legacy Text Logging (Validation Log)
        info = [
            str(epoch),
            ', Acc',
            str(results['Accuracy']),
            ', Sen',
            str(results['Sensitivity']),
            ', Spe',
            str(results['Specificity']),
            ', F1',
            str(results['F1']),
            ', AUC',
            str(results['AUC']),
            ', AP',
            str(results['AP']),
        ]
        with open(os.path.join(opt.checkpoints_dir, opt.name, 'validation_log.txt'), 'a') as f:
            f.writelines(info)
            f.writelines('\n')

        early_stopping(results['Accuracy'], model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 2, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch,
                                               delta=-0.00005,
                                               verbose=True)
            else:
                print(
                    "Learning rate dropped to minimum, still training with minimum learning rate..."
                )
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch,
                                               delta=-0.00005,
                                               verbose=True)
                break

        model.train()
