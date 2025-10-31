"""
SINet检测器封装
用于对抗性攻击的显著性目标检测器

参考论文:
- SINet: Camouflaged Object Detection via Semantic Information Network
- 基于语义信息网络的伪装目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple, Dict, List, Optional
import numpy as np
import cv2
from PIL import Image


class SINetBackbone(nn.Module):
    """
    SINet的ResNet50骨干网络
    提取多尺度特征用于显著性检测
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 使用ResNet50作为backbone
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=pretrained)
        
        # 提取各层特征
        self.conv1 = resnet.conv1          # 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1        # 256 channels
        self.layer2 = resnet.layer2        # 512 channels  
        self.layer3 = resnet.layer3        # 1024 channels
        self.layer4 = resnet.layer4        # 2048 channels
    
    def forward(self, x):
        """
        前向传播，返回多尺度特征
        
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            features: List of [B, C, H', W'] 多尺度特征
        """
        features = []
        
        # Stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # [B, 64, H/2, W/2]
        
        x = self.maxpool(x)
        
        # Stage 1-4
        x = self.layer1(x)
        features.append(x)  # [B, 256, H/4, W/4]
        
        x = self.layer2(x)
        features.append(x)  # [B, 512, H/8, W/8]
        
        x = self.layer3(x)
        features.append(x)  # [B, 1024, H/16, W/16]
        
        x = self.layer4(x)
        features.append(x)  # [B, 2048, H/32, W/32]
        
        return features


class SearchAttentionModule(nn.Module):
    """
    搜索注意力模块 (Search Attention)
    SINet的核心创新: 模拟人类搜索伪装目标的注意力机制
    """
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
        
        # 查询(Query)生成
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        
        # 键(Key)生成  
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        
        # 值(Value)生成
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # 输出投影
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Gamma参数(可学习的注意力强度)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            out: [B, C, H, W] 注意力增强特征
        """
        B, C, H, W = x.size()
        
        # 生成Query, Key, Value
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        key = self.key_conv(x).view(B, -1, H * W)                       # [B, C//8, HW]
        value = self.value_conv(x).view(B, -1, H * W)                   # [B, C, HW]
        
        # 计算注意力权重
        attention = torch.bmm(query, key)                               # [B, HW, HW]
        attention = self.softmax(attention)
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))              # [B, C, HW]
        out = out.view(B, C, H, W)
        
        # 残差连接 + 可学习权重
        out = self.gamma * self.out_conv(out) + x
        
        return out


class IdentificationModule(nn.Module):
    """
    识别模块 (Identification Module)
    结合搜索注意力和语义信息进行最终的显著性预测
    """
    
    def __init__(self, in_channels):
        super().__init__()
        
        # 搜索注意力
        self.search_attention = SearchAttentionModule(in_channels)
        
        # 语义增强
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 显著性预测头
        self.saliency_head = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            saliency_map: [B, 1, H, W] 显著性图
        """
        # 搜索注意力增强
        x = self.search_attention(x)
        
        # 语义特征提取
        x = self.semantic_conv(x)
        
        # 显著性预测
        saliency_map = self.saliency_head(x)
        
        return saliency_map


class SINetDetector(nn.Module):
    """
    完整的SINet检测器
    专门用于检测伪装/显著目标
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # 配置参数
        self.input_size = (352, 352) if config is None else config.input_size
        self.threshold = 0.5 if config is None else config.threshold
        
        # Backbone网络
        self.backbone = SINetBackbone(pretrained=True)
        
        # 特征金字塔网络 (FPN-like structure)
        self.fpn = self._build_fpn()
        
        # 识别模块
        self.identification = IdentificationModule(256)
        
        # 最终上采样到原图尺寸
        self.final_upsampling = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 预处理
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _build_fpn(self):
        """构建特征金字塔网络"""
        fpn_layers = nn.ModuleDict()
        
        # 侧向连接
        fpn_layers['lateral_conv1'] = nn.Conv2d(256, 256, 1)    # layer1
        fpn_layers['lateral_conv2'] = nn.Conv2d(512, 256, 1)    # layer2  
        fpn_layers['lateral_conv3'] = nn.Conv2d(1024, 256, 1)   # layer3
        fpn_layers['lateral_conv4'] = nn.Conv2d(2048, 256, 1)   # layer4
        
        # 融合卷积
        fpn_layers['fusion_conv'] = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        return fpn_layers
    
    def _preprocess(self, x):
        """图像预处理: 归一化"""
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像 (0-1范围)
            return_features: 是否返回中间特征
            
        Returns:
            saliency_map: [B, 1, H, W] 显著性图 (0-1范围)
            features: 如果return_features=True，返回中间特征
        """
        # 预处理
        x = self._preprocess(x)
        
        # Backbone特征提取
        features = self.backbone(x)  # [f0, f1, f2, f3, f4]
        
        # FPN特征融合
        # 使用layer1-4的特征 (跳过conv1的特征)
        f1 = self.fpn['lateral_conv1'](features[1])  # [B, 256, H/4, W/4]
        f2 = self.fpn['lateral_conv2'](features[2])  # [B, 256, H/8, W/8]
        f3 = self.fpn['lateral_conv3'](features[3])  # [B, 256, H/16, W/16]
        f4 = self.fpn['lateral_conv4'](features[4])  # [B, 256, H/32, W/32]
        
        # 上采样到统一尺寸 (H/4, W/4)
        target_size = f1.shape[2:]
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        f4 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        
        # 特征拼接
        fused_features = torch.cat([f1, f2, f3, f4], dim=1)  # [B, 1024, H/4, W/4]
        fused_features = self.fpn['fusion_conv'](fused_features)  # [B, 256, H/4, W/4]
        
        # 识别模块
        saliency_map = self.identification(fused_features)  # [B, 1, H/4, W/4]
        
        # 上采样到原图尺寸
        saliency_map = self.final_upsampling(saliency_map)  # [B, 1, H, W]
        
        if return_features:
            return saliency_map, {
                'backbone_features': features,
                'fpn_features': fused_features,
                'raw_saliency': saliency_map
            }
        
        return saliency_map
    
    def compute_adversarial_loss(self, x, target_confidence=0.1):
        """
        计算对抗损失
        目标: 让SINet检测不到目标(降低显著性)
        
        Args:
            x: [B, 3, H, W] 输入图像
            target_confidence: 目标置信度(越低越好)
            
        Returns:
            loss: 对抗损失值
        """
        saliency_map = self.forward(x)
        
        # 损失: 最大化显著性图与目标值的差距
        # 目标是让整张图的显著性都很低
        target = torch.full_like(saliency_map, target_confidence)
        loss = F.mse_loss(saliency_map, target)
        
        return loss
    
    def detect_objects(self, x, threshold=None):
        """
        目标检测接口
        
        Args:
            x: [B, 3, H, W] 输入图像
            threshold: 检测阈值
            
        Returns:
            detections: List of dict, 每个dict包含检测结果
        """
        if threshold is None:
            threshold = self.threshold
        
        with torch.no_grad():
            saliency_map = self.forward(x)
            
            detections = []
            for i in range(x.shape[0]):
                # 二值化显著性图
                binary_mask = (saliency_map[i, 0] > threshold).cpu().numpy().astype(np.uint8)
                
                # 查找连通区域
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                batch_detections = []
                for contour in contours:
                    # 计算边界框
                    x_min, y_min, w, h = cv2.boundingRect(contour)
                    x_max, y_max = x_min + w, y_min + h
                    
                    # 计算置信度(区域内平均显著性)
                    confidence = float(saliency_map[i, 0, y_min:y_max, x_min:x_max].mean())
                    
                    batch_detections.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'confidence': confidence,
                        'area': w * h
                    })
                
                detections.append(batch_detections)
        
        return detections


def load_pretrained_sinet(checkpoint_path: str, device: str = 'cuda') -> SINetDetector:
    """
    加载预训练的SINet模型
    
    Args:
        checkpoint_path: 预训练权重路径
        device: 设备
        
    Returns:
        model: 加载好的SINet模型
    """
    model = SINetDetector()
    
    if checkpoint_path and checkpoint_path != "":
        try:
            # 加载预训练权重
            state_dict = torch.load(checkpoint_path, map_location=device)
            
            # 处理可能的键名不匹配
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ 成功加载SINet预训练权重: {checkpoint_path}")
            
        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
            print("使用随机初始化权重")
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================
# 工具函数
# ============================================================

def visualize_saliency(image: torch.Tensor, saliency_map: torch.Tensor) -> np.ndarray:
    """
    可视化显著性图
    
    Args:
        image: [3, H, W] 原图
        saliency_map: [1, H, W] 显著性图
        
    Returns:
        vis_image: [H, W, 3] 可视化结果
    """
    # 转换为numpy
    image = image.cpu().permute(1, 2, 0).numpy()
    saliency = saliency_map.cpu().squeeze().numpy()
    
    # 归一化到0-1
    image = (image - image.min()) / (image.max() - image.min())
    
    # 热力图颜色映射
    import matplotlib.cm as cm
    saliency_colored = cm.jet(saliency)[:, :, :3]  # 去掉alpha通道
    
    # 叠加显示
    alpha = 0.6
    vis_image = alpha * saliency_colored + (1 - alpha) * image
    
    return (vis_image * 255).astype(np.uint8)


if __name__ == "__main__":
    # 测试SINet检测器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SINetDetector().to(device)
    
    # 测试输入
    x = torch.randn(2, 3, 352, 352).to(device)
    
    # 前向传播
    saliency_map = model(x)
    print(f"✅ SINet测试成功")
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {saliency_map.shape}")
    print(f"显著性范围: [{saliency_map.min():.3f}, {saliency_map.max():.3f}]")
    
    # 测试对抗损失
    adv_loss = model.compute_adversarial_loss(x)
    print(f"对抗损失: {adv_loss.item():.4f}")
