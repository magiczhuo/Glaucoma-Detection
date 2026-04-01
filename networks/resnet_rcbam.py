import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        fm = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        embedding = x

        return fm, embedding



def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_rcbam(pretrained=False, **kwargs):
    model = RCBAMResnet(RCBAMBottleneck, [3,8,36,3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
        # print("成功加载 ResNet-152 骨架预训练权重（已跳过新增注意力层）")
    
    return model




class RFuseAttention(nn.Module):
    def __init__(self, channel, retfound_dim=1024, reduction=16):
        super(RFuseAttention, self).__init__()
        self.channel = channel

        # 将RETFound全局特征投影到与channel维度匹配
        self.global_proj = nn.Sequential(
            nn.Linear(retfound_dim, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

        # 跨模态注意力融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channel * 2, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, global_features):
        """
        x: SA输出的特征图 [B, C, H, W]
        global_features: RETFound提取的全局特征 [B, D]
        """
        B, C, H, W = x.size()
        if global_features is None:
            print("global_features is None")
            
        
        # 方式1：全局特征调制 必须是 [B, 1024]
        global_weights = self.global_proj(global_features)  # [B, C]
        global_weights = global_weights.view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # 特征拼接后融合
        global_expanded = global_weights.expand(B, C, H, W)  # [B, C, H, W]
        concat_features = torch.cat([x, global_expanded], dim=1)  # [B, 2C, H, W]
        fusion_weights = self.fusion(concat_features)  # [B, C, H, W]
        
        return fusion_weights



class RCBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, retfound_dim=1024):
        super(RCBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.rfuse_attention = RFuseAttention(planes * 4, retfound_dim)
        self.rfuse_scale = nn.Parameter(torch.zeros(1))

        self.downsample = downsample
        self.stride = stride



    def forward(self, x, retfound_features):
        residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        out = out * self.ca(out)
        out = out * self.sa(out)

        out = out + out * self.rfuse_attention(out, retfound_features) * self.rfuse_scale
            
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  
        out = self.relu(out)

        return out
    

class RCBAMResnet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 retfound_model_path="./RETFound/RETFound_mae_natureCFP.pth", retfound_dim=1024,
                 use_offline_features=True):
        super(RCBAMResnet, self).__init__()

        self.inplanes = 64
        self.num_classes = num_classes
        self.has_retfound = retfound_model_path
        self.retfound_dim = retfound_dim
        self.use_offline_features = use_offline_features
        
        # === ResNet主干结构 ===
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # RCBAM增强层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # === RETFound特征提取器 ===
        if self.has_retfound and not self.use_offline_features:
            self.retfound_extractor = RETFoundFeatureExtractor(retfound_model_path)
            # 冻结RETFound
            for param in self.retfound_extractor.parameters():
                param.requires_grad = False
            print(" RETFound模型已加载并冻结（在线模式）")
        elif self.use_offline_features:
            print(" 使用离线RETFound特征模式")
        
        # === 权重初始化 ===
        self._initialize_weights()
        
        # === 冻结预训练部分 ===
        self._freeze_pretrained_parts()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """构建RCBAM层"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.retfound_dim))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, retfound_dim=self.retfound_dim))

        return nn.ModuleList(layers)  # 使用ModuleList以支持RETFound特征传递
    
    def _initialize_weights(self):
        """权重初始化 - 与原ResNet完全一致"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def _freeze_pretrained_parts(self):
        """
        Modified Freezing Strategy (Progressive Unfreezing + High-Level Semantics Adaptation):
        
        Scientific Rationale:
        1. Low-level features (Lines/Shapes) from ImageNet are universal and should be FROZEN to prevent catastrophic forgetting.
           -> Layer 1 & 2 are frozen.
        2. High-level features (Abstract concepts) need to adapt from "Natural Objects" (ImageNet) to "Medical Features" (Fundus).
           -> Layer 3 & 4 are UNFROZEN to allow domain adaptation.
        3. The Fusion Module (RFuseAttention) interacts strongly with high-level features, so high-level CNN layers must be pliable 
           to align with the global prompt (RETFound).
        """
        # 1. Freeze Initial Conventional Layers (Universal Features)
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
            
        # 2. Freeze Layer 1 & Layer 2 (Textures / Low-level patterns)
        # These features (edges, curves) are shared between ImageNet & Retina.
        for layer in [self.layer1, self.layer2]:
            for bottleneck in layer:
                for name, module in bottleneck.named_children():
                     # Only train the new attention modules, freeze everything else
                    if name not in ['rfuse_attention', 'ca', 'sa']:
                        for param in module.parameters():
                            param.requires_grad = False
                    else:
                        for param in module.parameters():
                            param.requires_grad = True
                if bottleneck.downsample is not None:
                     for param in bottleneck.downsample.parameters():
                        param.requires_grad = False

        # 3. Unfreeze Layer 3 & Layer 4 (Semantic / Domain-Specific Features)
        # CRITICAL CHANGE: We allow these layers to update.
        # Rationale: "Dog vs Cat" features (ImageNet) are different from "Cup-to-Disc Ratio" (Medical).
        # We need the network to re-learn high-level semantics while being guided by RFuseAttention.
        for layer in [self.layer3, self.layer4]:
            for bottleneck in layer:
                for param in bottleneck.parameters():
                    param.requires_grad = True

        # 4. FC Head is always trainable
        for param in self.fc.parameters():
            param.requires_grad = True
            
        print("Freezing Strategy Applied: Layers 1-2 Frozen (Low-level), Layers 3-4 Unfrozen (High-level Domain Adaptation).")



    def forward(self, x, retfound_features):
        """前向传播"""
        if self.use_offline_features:
            # 使用离线预提取的特征
            if retfound_features is None:
                print("离线模式需要提供RETFound特征")
                retfound_features = torch.zeros(x.size(0), self.retfound_dim, device=x.device)
        elif self.has_retfound:
            with torch.no_grad():
                retfound_features = self.retfound_extractor(x)
        else:
            retfound_features = None

        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        
        # RCBAM增强层（CA/SA/融合注意力可训练）
        x = self._forward_layer(self.layer1, x, retfound_features)
        x = self._forward_layer(self.layer2, x, retfound_features)
        x = self._forward_layer(self.layer3, x, retfound_features)
        x = self._forward_layer(self.layer4, x, retfound_features)
        
        fm = x  # 特征图

        # 分类头
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = x
        x = self.fc(x)

        return fm, embedding
    
    def _forward_layer(self, layer_modules, x, retfound_features):
        for module in layer_modules:
            if retfound_features is not None:
                # print("retfound_feature is not None")
                x = module(x, retfound_features)
            else:
                x = module(x)
        return x



import torchvision.transforms as transforms
import random
# PIL/Pillow
from PIL import Image, ImageEnhance
# 或者
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance
import PIL.ImageFilter as ImageFilter
import numpy as np

# OpenCV
import cv2

import subprocess
import os



class RETFoundFeatureExtractor(nn.Module):
    def __init__(self, model_weights, enable_enhancement=True, enhancement_prob=0.8):
        super().__init__()
        
        self.model_weights = model_weights  # 保存权重路径
        self.enable_enhancement = enable_enhancement
        self.enhancement_prob = enhancement_prob  # 应用增强的概率
        
        # 基础预处理（总是应用）
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        ])
        
        # 最终标准化（总是应用）
        self.final_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, C, H, W] 图像张量
        Returns:
            features: [B, D] 特征向量
        """
        B, C, H, W = x.size()
        
        # 对batch中的每张图像进行预处理
        processed_batch = []
        for i in range(B):
            img_tensor = x[i]  # [C, H, W]
            
            # 预处理单张图像
            enhanced_img = self._data_preprocess(img_tensor)
            processed_batch.append(enhanced_img)
        
        # 合并batch
        x_processed = torch.stack(processed_batch)
        
        tmp_input = "temp_input.pt"
        tmp_output = "temp_output.pt"
        torch.save(x_processed.cpu(), tmp_input)

        subprocess.run([
                "conda", "run", "--no-capture-output", "--name",
                "retfound", "python", "/home/grinlab/mnt/GrinLab/RETFound/retfound_extract.py",
                self.model_weights,
                tmp_input,
                tmp_output
            ], check=True
        )
        features = torch.load(tmp_output, weights_only=True)  # [B,D]
        os.remove(tmp_input)
        os.remove(tmp_output)
        features = features.to(x.device)

        return features
    
    def _data_preprocess(self, x):
        """
        图像质量增强预处理 - 集成7-8种方法
        Args:
            x: [C, H, W] 单张图像张量
        Returns:
            enhanced: [C, H, W] 增强后的图像张量
        """
        # 转换为PIL图像进行增强
        if x.max() <= 1.0:
            pil_img = self.base_transform(x)
        else:
            pil_img = self.base_transform(x / 255.0)
        
        if self.enable_enhancement and random.random() < self.enhancement_prob:
            # === 方法1: 自适应直方图均衡化（CLAHE）===
            pil_img = self._apply_clahe(pil_img)
            
            # === 方法2: 对比度和亮度增强 ===
            pil_img = self._enhance_contrast_brightness(pil_img)
            
            # === 方法3: 锐化滤波 ===
            if random.random() < 0.6:
                pil_img = self._apply_sharpening(pil_img)
            
            # === 方法4: 去噪处理 ===
            if random.random() < 0.4:
                pil_img = self._apply_denoising(pil_img)
            
            # === 方法5: 伽马校正 ===
            if random.random() < 0.5:
                pil_img = self._apply_gamma_correction(pil_img)
            
            # === 方法6: 色彩增强 ===
            pil_img = self._enhance_color(pil_img)
            
            # === 方法7: 边缘增强 ===
            if random.random() < 0.3:
                pil_img = self._apply_edge_enhancement(pil_img)
            
            # === 方法8: 多尺度融合 ===
            if random.random() < 0.2:
                pil_img = self._apply_multiscale_fusion(pil_img)
        
        # 转换回张量并标准化
        final_tensor = self.final_normalize(pil_img)
        return final_tensor
    
    def _apply_clahe(self, pil_img):
        """自适应直方图均衡化"""
        try:
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cv_img[:, :, 0] = clahe.apply(cv_img[:, :, 0])  # 只对L通道应用
            enhanced = cv2.cvtColor(cv_img, cv2.COLOR_LAB2RGB)
            return Image.fromarray(enhanced)
        except:
            return pil_img
    
    def _enhance_contrast_brightness(self, pil_img):
        """对比度和亮度增强"""
        try:
            # 对比度增强
            contrast_factor = random.uniform(0.9, 1.3)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast_factor)
            
            # 亮度增强
            brightness_factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(brightness_factor)
            
            return pil_img
        except:
            return pil_img
    
    def _apply_sharpening(self, pil_img):
        """锐化滤波"""
        try:
            sharpness_factor = random.uniform(1.0, 1.5)
            enhancer = ImageEnhance.Sharpness(pil_img)
            return enhancer.enhance(sharpness_factor)
        except:
            return pil_img
    
    def _apply_denoising(self, pil_img):
        """去噪处理"""
        try:
            # 使用高斯模糊进行轻微去噪
            radius = random.uniform(0.5, 1.0)
            return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        except:
            return pil_img
    
    def _apply_gamma_correction(self, pil_img):
        """伽马校正"""
        try:
            gamma = random.uniform(0.8, 1.2)
            np_img = np.array(pil_img) / 255.0
            corrected = np.power(np_img, gamma)
            corrected = (corrected * 255).astype(np.uint8)
            return Image.fromarray(corrected)
        except:
            return pil_img
    
    def _enhance_color(self, pil_img):
        """色彩增强"""
        try:
            saturation_factor = random.uniform(0.9, 1.2)
            enhancer = ImageEnhance.Color(pil_img)
            return enhancer.enhance(saturation_factor)
        except:
            return pil_img
    
    def _apply_edge_enhancement(self, pil_img):
        """边缘增强"""
        try:
            return pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        except:
            return pil_img
    
    def _apply_multiscale_fusion(self, pil_img):
        """多尺度融合增强"""
        try:
            # 创建不同尺度的图像
            w, h = pil_img.size
            small = pil_img.resize((w//2, h//2), Image.BICUBIC)
            small = small.resize((w, h), Image.BICUBIC)
            
            # 简单的加权融合
            np_orig = np.array(pil_img).astype(np.float32)
            np_small = np.array(small).astype(np.float32)
            
            alpha = 0.7
            fused = alpha * np_orig + (1 - alpha) * np_small
            fused = np.clip(fused, 0, 255).astype(np.uint8)
            
            return Image.fromarray(fused)
        except:
            return pil_img
