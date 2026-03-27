import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel, init_weights
from networks.resnet import resnet50, resnet152
from networks.resnet_cbam import resnet152_cbam
from networks.resnet_rcbam import resnet152_rcbam
from torchvision.ops import sigmoid_focal_loss


class SA_layer(nn.Module):

    def __init__(self, dim=128, head_size=4):
        super(SA_layer, self).__init__()
        self.mha = nn.MultiheadAttention(dim, head_size)
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim = x.shape
        x = torch.transpose(x, 1, 0)
        y, _ = self.mha(x, x, x)
        x = self.ln1(x + y)
        x = torch.transpose(x, 1, 0)
        x = x.reshape(batch_size * len_size, fea_dim)
        x = x + self.fc2(self.ac(self.fc1(x)))
        x = x.reshape(batch_size, len_size, fea_dim)
        x = self.ln2(x)
        return x

class PromptEmbedding(nn.Module):
    def __init__(self,prompt_length,embed_dim):
        super(PromptEmbedding,self).__init__()


class COOI():  # Coordinates On Original Image

    def __init__(self):
        self.stride = 32
        self.cropped_size = 224
        self.score_filter_size_list = [[3, 3], [2, 2]]
        self.score_filter_num_list = [3, 3]
        self.score_nms_size_list = [[3, 3], [3, 3]]
        self.score_nms_padding_list = [[1, 1], [1, 1]]
        self.score_corresponding_patch_size_list = [[224, 224], [112, 112]]
        self.score_filter_type_size = len(self.score_filter_size_list)

    def get_coordinates(self, fm, scale):
        with torch.no_grad():
            batch_size, _, fm_height, fm_width = fm.size()
            scale_min = torch.min(scale, axis=1, keepdim=True)[0].long()
            # scale_base = (scale-scale_min).long()//2  # torch.div(scale-scale_min,2,rounding_mode='floor')
            scale_base = torch.div(scale - scale_min, 2, rounding_mode='trunc')
            input_loc_list = []
            for type_no in range(self.score_filter_type_size):
                score_avg = nn.functional.avg_pool2d(
                    fm, self.score_filter_size_list[type_no],
                    stride=1)  # (7,2048,5,5), (7,2048,6,6)
                # (7,1,5,5), (7,1,6,6) #since the last operation in layer 4 of the resnet50 is relu, thus the score_sum are greater than zero
                score_sum = torch.sum(score_avg, dim=1, keepdim=True)
                _, _, score_height, score_width = score_sum.size()
                patch_height, patch_width = self.score_corresponding_patch_size_list[
                    type_no]

                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat = score_sum.view(batch_size, -1)
                    value_max, loc_max_flat = torch.max(score_sum_flat, dim=1)
                    loc_max = torch.stack((torch.div(
                        loc_max_flat, score_width,
                        rounding_mode='trunc'), loc_max_flat % score_width),
                                          dim=1)
                    # loc_max = torch.stack((loc_max_flat//score_width, loc_max_flat % score_width), dim=1)
                    top_patch = nn.functional.max_pool2d(
                        score_sum,
                        self.score_nms_size_list[type_no],
                        stride=1,
                        padding=self.score_nms_padding_list[type_no])
                    value_max = value_max.view(-1, 1, 1, 1)
                    # due to relu operation, the value are greater than 0, thus can be erase by multiply by 1.0/0.0
                    erase = (top_patch != value_max).float()
                    score_sum = score_sum * erase

                    # location in the original images
                    loc_rate_h = (2 * loc_max[:, 0] + fm_height -
                                  score_height + 1) / (2 * fm_height)
                    loc_rate_w = (2 * loc_max[:, 1] + fm_width - score_width +
                                  1) / (2 * fm_width)
                    loc_rate = torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center = (scale_base + scale_min * loc_rate).long()
                    loc_top = loc_center[:, 0] - patch_height // 2
                    loc_bot = loc_center[:,
                                         0] + patch_height // 2 + patch_height % 2
                    loc_lef = loc_center[:, 1] - patch_width // 2
                    loc_rig = loc_center[:,
                                         1] + patch_width // 2 + patch_width % 2
                    loc_tl = torch.stack((loc_top, loc_lef), dim=1)
                    loc_br = torch.stack((loc_bot, loc_rig), dim=1)

                    # For boundary conditions
                    loc_below = loc_tl.detach().clone()  # too low
                    loc_below[loc_below > 0] = 0
                    loc_br -= loc_below
                    loc_tl -= loc_below
                    loc_over = loc_br - scale.long()  # too high
                    loc_over[loc_over < 0] = 0
                    loc_tl -= loc_over
                    loc_br -= loc_over
                    loc_tl[loc_tl < 0] = 0  # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor = torch.stack(input_loc_list, dim=1)  # (7,6,4)
            # print(input_loc_tensor)
            return input_loc_tensor


class Branch2CBAM(nn.Module):

    def __init__(self, n_output):
        super(Branch2CBAM, self).__init__()
        # self.resnet1 = resnet152_cbam(pretrained=True)
        # self.resnet2 = resnet152_cbam(pretrained=True)

        self.resnet = resnet152_cbam(pretrained=True)
        # self.resnet_roi = resnet50(pretrained=False)
        # self.resnet_window = resnet50(pretrained=False)

        self.mha_list = nn.Sequential(SA_layer(128, 4), SA_layer(128, 4),
                                      SA_layer(128, 4))
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, n_output)
       
            
    def forward(self, img, full_img, roi, scale):

        x1 = full_img
        x2 = roi
        _, whole_embedding = self.resnet(
            x1
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        _, roi_embedding = self.resnet(
            x2
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        # print(whole_embedding.shape)
        # print(fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))  # 128
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)
        # print(s_whole_embedding.shape)
        s_roi_embedding = self.ac(self.fc1(roi_embedding))  # 128
        s_roi_embedding = s_roi_embedding.view(-1, 1, 128)

        all_embeddings = torch.cat((s_whole_embedding, s_roi_embedding),
                                   1)  # [1, 1+self.proposalN, 128]
        # all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        # print(all_embeddings.shape)
        all_embeddings = self.mha_list(all_embeddings)
        # print(all_embeddings.shape)
        all_logits = self.fc(all_embeddings[:, -1])
        # exit()

        return all_logits


class Branch3CBAM(nn.Module):

    def __init__(self,  n_output):
        super(Branch3CBAM, self).__init__()
        # self.resnet_full = resnet50(pretrained=True)  # debug
        # self.resnet_roi = resnet50(pretrained=True)

        self.resnet = resnet152_cbam(pretrained=True)
        # self.resnet_roi = resnet50(pretrained=False)
        # self.resnet_window = resnet50(pretrained=False)

        self.COOI = COOI()
        self.mha_list = nn.Sequential(SA_layer(128, 4), SA_layer(128, 4),
                                      SA_layer(128, 4))
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, n_output)

    def forward(self, img, full_img, roi, scale):

        x1 = full_img
        x2 = roi
        batch_size, p, _, _ = x1.shape  # [batch_size, 3, 299, 299]

        fm, whole_embedding = self.resnet(
            x1
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        _, roi_embedding = self.resnet(
            x2
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        # print(whole_embedding.shape)
        # print(fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))  # 128
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)
        # print(s_whole_embedding.shape)
        s_roi_embedding = self.ac(self.fc1(roi_embedding))  # 128
        s_roi_embedding = s_roi_embedding.view(-1, 1, 128)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale)

        _, proposal_size, _ = input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 299,
                                   299]).to(fm.device)  # [N, 4, 3, 299, 299]

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                # print('************************')
                img_patch = img[batch_no][:, t:b, l:r]
                # print(img_patch.size())
                _, patch_height, patch_width = img_patch.size()
                if patch_height == 299 and patch_width == 299:
                    window_imgs[batch_no, proposal_no] = img_patch
                else:
                    window_imgs[batch_no,
                                proposal_no:proposal_no + 1] = F.interpolate(
                                    img_patch[None, ...],
                                    size=(299, 299),
                                    mode='bilinear',
                                    align_corners=True)  # [N, 4, 3, 299, 299]
        # print(window_imgs.shape)
        # exit()

        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 299,
                                          299)  # [N*4, 3, 299, 299]
        _, window_embeddings = self.resnet(
            window_imgs.detach())  # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(
            self.fc1(window_embeddings))  # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)
        # print(s_window_embedding.shape)
        # exit()

        all_embeddings = torch.cat(
            (s_window_embedding, s_whole_embedding, s_roi_embedding),
            # all_embeddings = torch.cat((s_whole_embedding, s_roi_embedding),
            1)  # [1, 1+self.proposalN, 128]
        # all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        # print(all_embeddings.shape)
        all_embeddings = self.mha_list(all_embeddings)
        # print(all_embeddings.shape)
        all_logits = self.fc(all_embeddings[:, -1])
        # exit()

        return all_logits


class Patch5Model(nn.Module):

    def __init__(self, n_output):
        super(Patch5Model, self).__init__()
        # self.resnet_full = resnet50(pretrained=True)  # debug
        # self.resnet_roi = resnet50(pretrained=True)

        self.resnet = resnet152(pretrained=True)
        # self.resnet_roi = resnet50(pretrained=False)
        # self.resnet_window = resnet50(pretrained=False)

        self.COOI = COOI()
        self.mha_list = nn.Sequential(SA_layer(128, 4), SA_layer(128, 4),
                                      SA_layer(128, 4))
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, n_output)

    def forward(self, img, full_img, roi, scale):

        x1 = full_img
        x2 = roi
        batch_size, p, _, _ = x1.shape  # [batch_size, 3, 299, 299]

        fm, whole_embedding = self.resnet(
            x1
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        _, roi_embedding = self.resnet(
            x2
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        # print(whole_embedding.shape)
        # print(fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))  # 128
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)
        # print(s_whole_embedding.shape)
        s_roi_embedding = self.ac(self.fc1(roi_embedding))  # 128
        s_roi_embedding = s_roi_embedding.view(-1, 1, 128)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale)

        _, proposal_size, _ = input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 299,
                                   299]).to(fm.device)  # [N, 4, 3, 299, 299]

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                # print('************************')
                img_patch = img[batch_no][:, t:b, l:r]
                # print(img_patch.size())
                _, patch_height, patch_width = img_patch.size()
                if patch_height == 299 and patch_width == 299:
                    window_imgs[batch_no, proposal_no] = img_patch
                else:
                    window_imgs[batch_no,
                                proposal_no:proposal_no + 1] = F.interpolate(
                                    img_patch[None, ...],
                                    size=(299, 299),
                                    mode='bilinear',
                                    align_corners=True)  # [N, 4, 3, 299, 299]
        # print(window_imgs.shape)
        # exit()

        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 299,
                                          299)  # [N*4, 3, 299, 299]
        _, window_embeddings = self.resnet(
            window_imgs.detach())  # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(
            self.fc1(window_embeddings))  # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)
        # print(s_window_embedding.shape)
        # exit()

        all_embeddings = torch.cat(
            (s_window_embedding, s_whole_embedding, s_roi_embedding),
            # all_embeddings = torch.cat((s_whole_embedding, s_roi_embedding),
            1)  # [1, 1+self.proposalN, 128]
        # all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        # print(all_embeddings.shape)
        all_embeddings = self.mha_list(all_embeddings)
        # print(all_embeddings.shape)
        all_logits = self.fc(all_embeddings[:, -1])
        # exit()

        return all_logits


class Branch3RCBAM(nn.Module):

    def __init__(self,  n_output, use_offline_features=True):
        super(Branch3RCBAM, self).__init__()
        # self.resnet_full = resnet50(pretrained=True)  # debug
        # self.resnet_roi = resnet50(pretrained=True)

        self.use_offline_features = use_offline_features
        self.resnet = resnet152_rcbam(pretrained=True, use_offline_features=use_offline_features)
        # self.resnet_roi = resnet50(pretrained=False)
        # self.resnet_window = resnet50(pretrained=False)

        self.COOI = COOI()
        self.mha_list = nn.Sequential(SA_layer(128, 4), SA_layer(128, 4),
                                      SA_layer(128, 4))
        # self.resnet.fc = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(2048, 128)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(128, n_output)

    def forward(self, img, full_img, roi, scale, retfound_features = None):

        x1 = full_img
        x2 = roi
        batch_size, p, _, _ = x1.shape  # [batch_size, 3, 299, 299]

        fm, whole_embedding = self.resnet(
            x1, retfound_features
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        _, roi_embedding = self.resnet(
            x2, retfound_features
        )  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        # print(whole_embedding.shape)
        # print(fm.shape)
        s_whole_embedding = self.ac(self.fc1(whole_embedding))  # 128
        s_whole_embedding = s_whole_embedding.view(-1, 1, 128)
        # print(s_whole_embedding.shape)
        s_roi_embedding = self.ac(self.fc1(roi_embedding))  # 128
        s_roi_embedding = s_roi_embedding.view(-1, 1, 128)

        input_loc = self.COOI.get_coordinates(fm.detach(), scale)

        _, proposal_size, _ = input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 299,
                                   299]).to(fm.device)  # [N, 4, 3, 299, 299]

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                # print('************************')
                img_patch = img[batch_no][:, t:b, l:r]
                # print(img_patch.size())
                _, patch_height, patch_width = img_patch.size()
                if patch_height == 299 and patch_width == 299:
                    window_imgs[batch_no, proposal_no] = img_patch
                else:
                    window_imgs[batch_no,
                                proposal_no:proposal_no + 1] = F.interpolate(
                                    img_patch[None, ...],
                                    size=(299, 299),
                                    mode='bilinear',
                                    align_corners=True)  # [N, 4, 3, 299, 299]
        # print(window_imgs.shape)
        # exit()

        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 299,
                                          299)  # [N*4, 3, 299, 299]
        # 使用 repeat_interleave 确保特征与 window_imgs 的 Batch 顺序完全对应
        expanded_retfound_features = torch.repeat_interleave(retfound_features, 
                                            proposal_size, dim=0) # [N * 4, 1024]
        _, window_embeddings = self.resnet(
            window_imgs.detach(), expanded_retfound_features)  # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(
            self.fc1(window_embeddings))  # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 128)
        # print(s_window_embedding.shape)
        # exit()

        all_embeddings = torch.cat(
            (s_window_embedding, s_whole_embedding, s_roi_embedding),
            # all_embeddings = torch.cat((s_whole_embedding, s_roi_embedding),
            1)  # [1, 1+self.proposalN, 128]
        # all_embeddings=all_embeddings.view(-1, (1+proposal_size), 128)
        # print(all_embeddings.shape)
        all_embeddings = self.mha_list(all_embeddings)
        # print(all_embeddings.shape)
        all_logits = self.fc(all_embeddings[:, -1])
        # exit()

        return all_logits





class Trainer(BaseModel):

    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        # if self.isTrain and not opt.continue_train:
        if opt.mode == 'binary':
            if opt.model_name == '3branch':
                self.model = Patch5Model(1)
            if opt.model_name == '2branch-cbam':
                self.model = Branch2CBAM(1)
            if opt.model_name == '3branch-cbam':
                self.model = Branch3CBAM(1)
            if opt.model_name == '3branch-rcbam':
                self.model = Branch3RCBAM(1)

            
        if opt.mode == '3cls':
            if opt.model_name == '3branch':
                self.model = Patch5Model(3)
            if opt.model_name == '2branch-cbam':
                self.model = Branch2CBAM(3)
            if opt.model_name == '3branch-cbam':
                self.model = Branch3CBAM(3)
            if opt.model_name == '3branch-rcbam':
                self.model = Branch3RCBAM(3)
        
        print(f'Load {opt.model_name} successfully')

        if len(opt.gpu_ids) >= 2:
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
            self.model = self.model.cuda()
        else:
            self.model.to(opt.gpu_ids[0])

        # if not self.isTrain or opt.continue_train:
        #     # self.model = resnet50(num_classes=1)
        #     self.model = Patch5Model()
        #     if len(opt.gpu_ids) >= 2:
        #         self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
        #     else:
        #         self.model.to(opt.gpu_ids[0])

        if self.isTrain:
            if opt.mode == 'binary':
                self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.mode == '3cls':
                """
                - 1st : 1, 1.2, 1.5
                - 2nd : 1, 1.05, 1.15

                class_weights = torch.tensor([1.0, 1.05, 1.15]).float()
                if len(opt.gpu_ids) > 0:
                        class_weights = class_weights.cuda()
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)"""
                
                self.loss_fn = nn.CrossEntropyLoss()
            # self.loss_fn = sigmoid_focal_loss
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr,
                                                  betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr,
                                                 momentum=0.0,
                                                 weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

            if len(opt.gpu_ids) == 0:
                self.model.to('cpu')
            else:
                self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.img = data[0]  # (batch_size, 6, 3, 224, 224)
        self.full_img = copy.deepcopy(self.img)
        self.full_img = self.full_img.to(self.device)
        self.roi = data[1].to(self.device)
        self.label = data[2].to(self.device).long()  # (batch_size)
        self.scale = data[3].to(self.device).float()
        
        # 检查是否有RETFound特征
        if len(data) > 5:
            raw_feature = data[5]
            if raw_feature is not None:
                self.retfound_features = raw_feature.to(self.device)
                # print("特征加载成功，已送入设备")
            else:
                self.retfound_features = None
                # print("特征数据为 None")
        # self.imgname = data[4]
        
    def forward(self):
        # 判断是否使用了 DataParallel 包装
        """if isinstance(self.model, torch.nn.DataParallel):
            use_offline = self.model.module.use_offline_features
        else:
            use_offline = self.model.use_offline_features"""
        # 默认使用离线模式

        if isinstance(self.model, (Branch3RCBAM)):
             # print("使用离线特征模式")
             self.output = self.model(self.img, self.full_img, self.roi, self.scale, self.retfound_features)
        elif hasattr(self.model, 'module') and isinstance(self.model.module, (Branch3RCBAM)): # Handle DataParallel
              # print("使用离线特征模式 (DataParallel)")
             self.output = self.model(self.img, self.full_img, self.roi, self.scale, self.retfound_features)
        else:
            # print("使用在线特征模式")
            self.output = self.model(self.img, self.full_img, self.roi, self.scale)

    def get_loss(self):
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            return self.loss_fn(self.output, self.label.float().view(-1, 1))
        return self.loss_fn(self.output, self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
