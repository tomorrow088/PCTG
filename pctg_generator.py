"""
Physical-Constrained Texture Generator (PCTG)
物理约束纹理生成器 - 项目核心模型

核心创新:
1. Multi-Scale Texture Encoder - 多尺度纹理编码器
2. Adversarial Conditioning Module - 对抗条件化模块  
3. Physical Constraint Layer - 物理约束层
4. Differentiable Renderer - 可微分渲染器

设计理念:
- 轻量级 (~50M parameters)
- 端到端可训练
- 物理世界可实现
- 快速推理 (<50ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class MultiScaleTextureEncoder(nn.Module):
    """
    多尺度纹理编码器
    
    核心思想: 分离语义内容和纹理模式
    - 使用EfficientNet提取多尺度特征
    - 小波变换提取纹理频率成分
    - 注意力机制突出纹理区域
    """
    
    def __init__(self, encoder_name='efficientnet_b3', pretrained=True):
        super().__init__()
        
        # EfficientNet backbone
        import timm
        self.backbone = timm.create_model(
            encoder_name, 
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # 提取4个尺度的特征
        )
        
        # 获取特征通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_channels = [f.shape[1] for f in features]
        
        print(f"✅ EfficientNet特征通道: {self.feature_channels}")
        
        # 小波纹理分析 (频域纹理提取)
        self.wavelet_texture = WaveletTextureAnalyzer()
        
        # 多尺度特征融合
        self.feature_fusion = MultiScaleFeatureFusion(
            in_channels=self.feature_channels,
            out_channels=512
        )
        
        # 纹理注意力
        self.texture_attention = TextureAttentionModule(512)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, 3, H, W] 输入图像
            mask: [B, 1, H, W] 掩码(可选)
            
        Returns:
            texture_features: [B, 512, H/8, W/8] 纹理特征
            high_freq_features: [B, 128, H/2, W/2] 高频纹理特征
        """
        # 多尺度特征提取
        multi_scale_features = self.backbone(x)
        
        # 小波纹理分析 (提取高频细节)
        high_freq_features = self.wavelet_texture(x)
        
        # 多尺度特征融合
        fused_features = self.feature_fusion(multi_scale_features, high_freq_features)
        
        # 纹理注意力增强
        if mask is not None:
            texture_features = self.texture_attention(fused_features, mask)
        else:
            texture_features = self.texture_attention(fused_features, None)
        
        return texture_features, high_freq_features


class WaveletTextureAnalyzer(nn.Module):
    """
    小波纹理分析器
    使用小波变换提取图像的纹理频率成分
    """
    
    def __init__(self):
        super().__init__()
        
        # 小波基函数 (简化的Gabor滤波器)
        self.register_buffer('wavelet_filters', self._create_wavelet_filters())
        
        # 后处理卷积
        self.texture_conv = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),  # 12个方向的滤波器
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
    
    def _create_wavelet_filters(self):
        """创建小波滤波器组"""
        filters = []
        
        # 4个尺度 × 3个方向 = 12个滤波器
        for scale in [1, 2, 4, 8]:
            for angle in [0, 60, 120]:  # 3个方向
                # Gabor滤波器
                filter_kernel = self._gabor_kernel(scale, angle)
                filters.append(filter_kernel)
        
        return torch.stack(filters).unsqueeze(1)  # [12, 1, H, W]
    
    def _gabor_kernel(self, scale, angle, kernel_size=15):
        """生成Gabor滤波器核"""
        sigma = scale
        theta = np.radians(angle)
        lambda_val = scale * 2
        
        # 创建坐标网格
        x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32)
        y = torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 旋转坐标
        x_theta = X * math.cos(theta) + Y * math.sin(theta)
        y_theta = -X * math.sin(theta) + Y * math.cos(theta)
        
        # Gabor函数
        gabor = torch.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * \
                torch.cos(2 * math.pi * x_theta / lambda_val)
        
        # 归一化
        gabor = gabor / gabor.abs().sum()
        
        return gabor
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            texture_response: [B, 128, H/2, W/2] 纹理响应
        """
        B, C, H, W = x.shape
        
        # 转换为灰度图
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # 应用小波滤波器
        texture_responses = []
        for i in range(self.wavelet_filters.shape[0]):
            response = F.conv2d(
                gray, 
                self.wavelet_filters[i:i+1], 
                padding=self.wavelet_filters.shape[2]//2
            )
            texture_responses.append(response)
        
        # 拼接所有响应
        texture_response = torch.cat(texture_responses, dim=1)  # [B, 12, H, W]
        
        # 后处理
        texture_response = self.texture_conv(texture_response)  # [B, 128, H, W]
        
        # 下采样到H/2
        texture_response = F.interpolate(
            texture_response, 
            scale_factor=0.5, 
            mode='bilinear', 
            align_corners=False
        )
        
        return texture_response


class AdversarialConditioningModule(nn.Module):
    """
    对抗条件化模块
    
    核心思想: 将对抗梯度信息融入特征生成过程
    - 多检测器梯度聚合
    - 自适应条件化强度
    - 梯度引导的特征变换
    """
    
    def __init__(self, feature_dim=512, num_detectors=2):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_detectors = num_detectors
        
        # 对抗梯度编码器
        self.gradient_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 检测器权重预测器 (学习每个检测器的重要性)
        self.detector_weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_detectors, 1),
            nn.Softmax(dim=1)
        )
        
        # 条件化强度控制
        self.conditioning_controller = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征变换器
        self.feature_transformer = nn.Sequential(
            nn.Conv2d(feature_dim + 128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
    
    def forward(self, features, adversarial_gradients=None):
        """
        Args:
            features: [B, 512, H, W] 原始特征
            adversarial_gradients: Dict[str, Tensor] 对抗梯度
            
        Returns:
            conditioned_features: [B, 512, H, W] 条件化后的特征
            conditioning_info: Dict 条件化信息
        """
        if adversarial_gradients is None:
            # 训练初期没有对抗梯度时的处理
            return features, {}
        
        # 预测检测器权重
        detector_weights = self.detector_weight_predictor(features)  # [B, num_detectors, 1, 1]
        
        # 聚合对抗梯度
        aggregated_gradient = self._aggregate_gradients(
            adversarial_gradients, 
            detector_weights,
            features.shape[2:]
        )
        
        # 编码对抗梯度
        gradient_features = self.gradient_encoder(aggregated_gradient)  # [B, 128, H, W]
        
        # 预测条件化强度
        conditioning_strength = self.conditioning_controller(gradient_features)  # [B, 1, H, W]
        
        # 特征条件化
        combined_features = torch.cat([features, gradient_features], dim=1)
        transformed_features = self.feature_transformer(combined_features)
        
        # 自适应混合
        conditioned_features = (
            conditioning_strength * transformed_features + 
            (1 - conditioning_strength) * features
        )
        
        conditioning_info = {
            'detector_weights': detector_weights,
            'conditioning_strength': conditioning_strength.mean().item(),
            'gradient_norm': aggregated_gradient.norm().item()
        }
        
        return conditioned_features, conditioning_info
    
    def _aggregate_gradients(self, gradients, weights, target_size):
        """聚合多个检测器的梯度"""
        aggregated = torch.zeros(
            weights.shape[0], self.feature_dim, target_size[0], target_size[1],
            device=weights.device
        )
        
        detector_names = list(gradients.keys())
        for i, name in enumerate(detector_names):
            if i < self.num_detectors:
                grad = gradients[name]
                # 调整梯度到目标尺寸
                if grad.shape[2:] != target_size:
                    grad = F.interpolate(grad, size=target_size, mode='bilinear')
                
                # 应用权重
                weight = weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
                aggregated += weight * grad
        
        return aggregated


class PhysicalConstraintLayer(nn.Module):
    """
    物理约束层
    
    确保生成的纹理在物理世界可实现:
    1. 色域约束 - 限制到可打印色彩空间
    2. 频率约束 - 避免过高频细节
    3. 材质约束 - 模拟真实材质反射
    """
    
    def __init__(self, enable_color_constraint=True, enable_frequency_constraint=True):
        super().__init__()
        
        self.enable_color_constraint = enable_color_constraint
        self.enable_frequency_constraint = enable_frequency_constraint
        
        # 色域映射 (RGB -> 可打印CMYK -> RGB)
        if enable_color_constraint:
            self.color_mapper = ColorSpaceMapper()
        
        # 频率滤波器
        if enable_frequency_constraint:
            self.frequency_filter = FrequencyConstraintFilter()
        
        # 材质反射模拟
        self.material_simulator = MaterialReflectionSimulator()
    
    def forward(self, texture, lighting_condition=None):
        """
        Args:
            texture: [B, 3, H, W] 原始纹理 (-1 to 1)
            lighting_condition: [B, 3] 光照条件 (可选)
            
        Returns:
            constrained_texture: [B, 3, H, W] 物理约束后的纹理
            constraint_loss: 约束损失
        """
        constraint_losses = []
        
        # 归一化到 0-1
        texture = (texture + 1) / 2
        
        # 1. 色域约束
        if self.enable_color_constraint:
            texture, color_loss = self.color_mapper(texture)
            constraint_losses.append(color_loss)
        
        # 2. 频率约束  
        if self.enable_frequency_constraint:
            texture, freq_loss = self.frequency_filter(texture)
            constraint_losses.append(freq_loss)
        
        # 3. 材质反射模拟
        texture, material_loss = self.material_simulator(texture, lighting_condition)
        constraint_losses.append(material_loss)
        
        # 归一化回 -1 to 1
        constrained_texture = texture * 2 - 1
        
        # 总约束损失
        constraint_loss = sum(constraint_losses) if constraint_losses else torch.tensor(0.0)
        
        return constrained_texture, constraint_loss


class ColorSpaceMapper(nn.Module):
    """色彩空间映射器 - 限制到可打印色域"""
    
    def __init__(self):
        super().__init__()
        
        # 可打印色域的边界参数 (简化的CMYK色域)
        self.register_buffer('color_gamut_min', torch.tensor([0.05, 0.05, 0.05]))
        self.register_buffer('color_gamut_max', torch.tensor([0.95, 0.95, 0.95]))
    
    def forward(self, rgb_texture):
        """
        Args:
            rgb_texture: [B, 3, H, W] RGB纹理 (0-1)
            
        Returns:
            constrained_texture: [B, 3, H, W] 约束后纹理
            color_loss: 色域约束损失
        """
        # 软约束到可打印范围
        constrained_texture = torch.sigmoid(rgb_texture) * \
                             (self.color_gamut_max - self.color_gamut_min) + \
                             self.color_gamut_min
        
        # 计算约束损失 (原纹理偏离可打印色域的程度)
        color_loss = F.mse_loss(constrained_texture, rgb_texture)
        
        return constrained_texture, color_loss


class FrequencyConstraintFilter(nn.Module):
    """频率约束滤波器 - 限制纹理频率"""
    
    def __init__(self, max_frequency_ratio=0.8):
        super().__init__()
        self.max_frequency_ratio = max_frequency_ratio
    
    def forward(self, texture):
        """
        Args:
            texture: [B, 3, H, W] 输入纹理
            
        Returns:
            filtered_texture: [B, 3, H, W] 频率滤波后纹理
            freq_loss: 频率约束损失
        """
        # FFT变换到频域
        fft_texture = torch.fft.fft2(texture)
        fft_magnitude = torch.abs(fft_texture)
        fft_phase = torch.angle(fft_texture)
        
        # 创建低通滤波器
        H, W = texture.shape[2], texture.shape[3]
        center_h, center_w = H // 2, W // 2
        
        # 频率掩码
        y, x = torch.meshgrid(
            torch.arange(H, device=texture.device),
            torch.arange(W, device=texture.device),
            indexing='ij'
        )
        distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        max_distance = min(H, W) * self.max_frequency_ratio / 2
        
        freq_mask = torch.exp(-(distance / max_distance)**2)  # 高斯低通
        
        # 应用滤波器
        filtered_magnitude = fft_magnitude * freq_mask.unsqueeze(0).unsqueeze(0)
        filtered_fft = filtered_magnitude * torch.exp(1j * fft_phase)
        
        # 逆FFT变换回时域
        filtered_texture = torch.real(torch.fft.ifft2(filtered_fft))
        
        # 计算频率损失 (高频成分的能量)
        high_freq_mask = 1 - freq_mask
        high_freq_energy = (fft_magnitude * high_freq_mask.unsqueeze(0).unsqueeze(0)).mean()
        freq_loss = high_freq_energy
        
        return filtered_texture, freq_loss


class MaterialReflectionSimulator(nn.Module):
    """材质反射模拟器"""
    
    def __init__(self):
        super().__init__()
        
        # 布料材质参数
        self.fabric_roughness = nn.Parameter(torch.tensor(0.6))  # 粗糙度
        self.fabric_metallic = nn.Parameter(torch.tensor(0.1))   # 金属度
    
    def forward(self, texture, lighting_condition=None):
        """
        简化的材质反射模拟
        
        Args:
            texture: [B, 3, H, W] 输入纹理
            lighting_condition: [B, 3] 光照条件
            
        Returns:
            realistic_texture: [B, 3, H, W] 真实材质纹理
            material_loss: 材质约束损失
        """
        if lighting_condition is None:
            # 默认光照条件 (日光)
            lighting_condition = torch.tensor([1.0, 1.0, 0.9], device=texture.device)
            lighting_condition = lighting_condition.unsqueeze(0).repeat(texture.shape[0], 1)
        
        # 简化的漫反射模拟
        lighting = lighting_condition.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
        diffuse_reflection = texture * lighting
        
        # 粗糙度影响 (降低镜面反射)
        roughness_factor = 1 - self.fabric_roughness * 0.2
        realistic_texture = diffuse_reflection * roughness_factor
        
        # 材质损失 (保持合理的反射特性)
        material_loss = torch.tensor(0.0, device=texture.device)
        
        return realistic_texture, material_loss


class TextureDecoder_V3(nn.Module):
    """
    纹理解码器 V3 - 增强版
    
    核心改进:
    1. 残差上采样路径
    2. 特征金字塔融合  
    3. 细节保持机制
    4. 自适应纹理合成
    """
    
    def __init__(self, input_dim=512, output_channels=3):
        super().__init__()
        
        # 上采样路径 (类似U-Net decoder)
        self.up_blocks = nn.ModuleList([
            UpBlockV3(input_dim, 256),      # H/8 -> H/4
            UpBlockV3(256, 128),            # H/4 -> H/2  
            UpBlockV3(128, 64),             # H/2 -> H
            UpBlockV3(64, 32)               # H -> H (细化)
        ])
        
        # 纹理合成模块 (StyleGAN-inspired)
        self.texture_synthesis = AdaptiveTextureSynthesis(32)
        
        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.Tanh()  # 输出[-1, 1]
        )
        
        # 细节增强
        self.detail_enhancer = DetailEnhancer(output_channels)
    
    def forward(self, features, target_size, style_code=None):
        """
        Args:
            features: [B, 512, H/8, W/8] 编码器特征
            target_size: (H, W) 目标输出尺寸
            style_code: [B, 512] 风格编码 (可选)
            
        Returns:
            texture: [B, 3, H, W] 生成的纹理
        """
        x = features
        
        # 上采样解码
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # 自适应纹理合成
        x = self.texture_synthesis(x, style_code)
        
        # 输出投影
        texture = self.output_conv(x)
        
        # 细节增强
        texture = self.detail_enhancer(texture)
        
        # 调整到目标尺寸
        if texture.shape[2:] != target_size:
            texture = F.interpolate(
                texture, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return texture


class UpBlockV3(nn.Module):
    """增强版上采样块"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 主上采样路径
        self.main_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差路径
        self.residual_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.main_path(x) + self.residual_path(x)


class AdaptiveTextureSynthesis(nn.Module):
    """自适应纹理合成模块"""
    
    def __init__(self, channels):
        super().__init__()
        
        # 纹理风格调制
        self.style_modulation = StyleModulation(channels)
        
        # 纹理卷积
        self.texture_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x, style_code=None):
        # 风格调制
        if style_code is not None:
            x = self.style_modulation(x, style_code)
        
        # 纹理合成
        texture_enhanced = self.texture_conv(x)
        
        # 残差连接
        return x + texture_enhanced


class StyleModulation(nn.Module):
    """风格调制模块 (AdaIN-like)"""
    
    def __init__(self, channels):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        
        # 风格编码器
        self.style_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, channels * 2)  # mean and std
        )
    
    def forward(self, x, style_code):
        # 标准化
        normalized = self.norm(x)
        
        # 生成风格参数
        style_params = self.style_encoder(style_code)
        style_mean = style_params[:, :x.shape[1]].unsqueeze(-1).unsqueeze(-1)
        style_std = style_params[:, x.shape[1]:].unsqueeze(-1).unsqueeze(-1)
        
        # 风格迁移
        return normalized * style_std + style_mean


class DetailEnhancer(nn.Module):
    """细节增强器"""
    
    def __init__(self, channels):
        super().__init__()
        
        # 高频细节提取
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 提取细节
        details = self.detail_extractor(x)
        
        # 自适应融合
        alpha = 0.3  # 细节强度
        enhanced = x + alpha * details
        
        return enhanced


# ============================================================
# 重用之前定义的组件
# ============================================================

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合 (复用之前的实现)"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 512):
        super().__init__()
        
        # 适配卷积 - 将不同尺度特征统一到相同通道数
        self.adapt1 = nn.Conv2d(in_channels[0], 128, 1)        # layer1: H/4
        self.adapt2 = nn.Conv2d(in_channels[1], 128, 1)        # layer2: H/8
        self.adapt3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels[2], 128, 1)                  # layer3: H/16 -> H/8
        )
        self.adapt4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'), 
            nn.Conv2d(in_channels[3], 128, 1)                  # layer4: H/32 -> H/8
        )
        
        self.adapt_hf = nn.Sequential(
            nn.Conv2d(128, 128, 1),                            # high_freq: H/2
            nn.Upsample(scale_factor=0.25, mode='bilinear')    # -> H/8
        )
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128 * 5, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, multi_scale, high_freq):
        # 对齐所有特征到H/8尺寸
        target_size = multi_scale[1].shape[2:]  # layer2的尺寸 (H/8, W/8)
        
        f1 = F.interpolate(self.adapt1(multi_scale[0]), size=target_size, mode='bilinear')
        f2 = self.adapt2(multi_scale[1])
        f3 = F.interpolate(self.adapt3(multi_scale[2]), size=target_size, mode='bilinear')
        f4 = F.interpolate(self.adapt4(multi_scale[3]), size=target_size, mode='bilinear')
        f_hf = F.interpolate(self.adapt_hf(high_freq), size=target_size, mode='bilinear')
        
        # 拼接融合
        fused = torch.cat([f1, f2, f3, f4, f_hf], dim=1)
        output = self.fusion_conv(fused)
        
        return output


class TextureAttentionModule(nn.Module):
    """纹理注意力模块 (复用之前的实现)"""
    
    def __init__(self, channels):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, mask=None):
        # 通道注意力
        ca = self.channel_attention(features)
        features = features * ca
        
        # 空间注意力
        sa = self.spatial_attention(features)
        
        if mask is not None:
            # 下采样mask到特征尺寸
            mask_small = F.interpolate(
                mask, 
                size=features.shape[2:], 
                mode='bilinear'
            )
            # 增强掩码区域
            sa = sa * mask_small + (1 - mask_small) * 0.5
        
        features = features * sa
        
        return features


# ============================================================
# 完整的PCTG模型
# ============================================================

class PCTGGenerator(nn.Module):
    """
    完整的Physical-Constrained Texture Generator
    
    集成所有模块的完整生成器
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # 配置
        if config is None:
            from model_config import PCTGConfig
            config = PCTGConfig()
        
        self.config = config
        
        # 核心模块
        self.encoder = MultiScaleTextureEncoder(
            encoder_name=config.encoder_name,
            pretrained=config.encoder_pretrained
        )
        
        self.adversarial_conditioning = AdversarialConditioningModule(
            feature_dim=512,
            num_detectors=2
        )
        
        self.physical_constraint = PhysicalConstraintLayer(
            enable_color_constraint=config.printable_colors_only,
            enable_frequency_constraint=True
        )
        
        self.decoder = TextureDecoder_V3(
            input_dim=512,
            output_channels=config.output_channels
        )
        
        print(f"✅ PCTG生成器初始化完成")
        print(f"   编码器: {config.encoder_name}")
        print(f"   物理约束: 色域={config.printable_colors_only}, 频率=True")
        print(f"   输出通道: {config.output_channels}")
    
    def forward(self, x, mask, adversarial_gradients=None, lighting_condition=None):
        """
        完整的前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像
            mask: [B, 1, H, W] 目标掩码
            adversarial_gradients: Dict[str, Tensor] 对抗梯度
            lighting_condition: [B, 3] 光照条件
            
        Returns:
            output: Dict 包含生成结果和中间信息
        """
        # 1. 多尺度纹理编码
        texture_features, high_freq_features = self.encoder(x, mask)
        
        # 2. 对抗条件化
        conditioned_features, conditioning_info = self.adversarial_conditioning(
            texture_features, adversarial_gradients
        )
        
        # 3. 纹理解码
        generated_texture = self.decoder(
            conditioned_features, 
            target_size=x.shape[2:]
        )
        
        # 4. 物理约束
        final_texture, constraint_loss = self.physical_constraint(
            generated_texture, lighting_condition
        )
        
        # 5. 与原图混合
        blended_result = self._blend_with_original(x, final_texture, mask)
        
        # 返回完整结果
        output = {
            'generated_texture': final_texture,
            'blended_result': blended_result,
            'constraint_loss': constraint_loss,
            'conditioning_info': conditioning_info,
            'high_freq_features': high_freq_features
        }
        
        return output
    
    def _blend_with_original(self, original, texture, mask, blend_ratio=0.8):
        """与原图混合"""
        blended = (
            original * (1 - mask) +
            (original * (1 - blend_ratio) + texture * blend_ratio) * mask
        )
        return blended
    
    def compute_total_loss(self, output, targets):
        """计算总损失 (训练时使用)"""
        # 基础损失项在 losses.py 中定义
        constraint_loss = output['constraint_loss']
        
        return {
            'constraint_loss': constraint_loss,
            'total_constraint_loss': constraint_loss
        }


if __name__ == "__main__":
    # 测试PCTG生成器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    generator = PCTGGenerator().to(device)
    
    # 测试输入
    batch_size = 2
    H, W = 256, 256
    
    x = torch.randn(batch_size, 3, H, W).to(device)
    mask = torch.randn(batch_size, 1, H, W).sigmoid().to(device)
    
    # 前向传播
    with torch.no_grad():
        output = generator(x, mask)
    
    print(f"✅ PCTG生成器测试成功")
    print(f"输入尺寸: {x.shape}")
    print(f"生成纹理尺寸: {output['generated_texture'].shape}")
    print(f"混合结果尺寸: {output['blended_result'].shape}")
    print(f"约束损失: {output['constraint_loss'].item():.4f}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M")
