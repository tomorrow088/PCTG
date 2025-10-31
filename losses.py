"""
损失函数模块
Multi-Modal Adversarial Loss for Camouflage Generation

损失函数组成:
1. AdversarialLoss - 对抗损失 (针对SINet + CLIP)
2. ContentLoss - 内容保持损失  
3. PerceptualLoss - 感知损失 (VGG + LPIPS)
4. TextureLoss - 纹理损失 (Gram矩阵)
5. PhysicalConstraintLoss - 物理约束损失
6. RegularizationLoss - 正则化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class MultiModalAdversarialLoss(nn.Module):
    """
    多模态对抗损失
    
    目标: 让生成的图像在多个检测器上都难以被检测到
    - SINet: 降低显著性检测分数
    - CLIP: 降低与目标类别的相似度
    - 其他检测器: 可扩展架构
    """
    
    def __init__(self, detector_weights=None):
        super().__init__()
        
        # 检测器权重
        if detector_weights is None:
            detector_weights = {
                'sinet': 1.0,
                'clip': 0.8
            }
        self.detector_weights = detector_weights
        
        # 目标检测置信度
        self.target_confidence = {
            'sinet': 0.1,      # SINet显著性目标值
            'clip': 0.3        # CLIP相似度目标值
        }
    
    def forward(self, detectors_output: Dict[str, torch.Tensor], 
                detectors: Dict[str, nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        计算多模态对抗损失
        
        Args:
            detectors_output: 各检测器的输出
            detectors: 检测器模型字典 (可选，用于梯度回传)
            
        Returns:
            loss_dict: 各检测器的对抗损失
        """
        losses = {}
        total_loss = 0
        
        # SINet对抗损失
        if 'sinet' in detectors_output:
            sinet_loss = self._compute_sinet_adversarial_loss(
                detectors_output['sinet']
            )
            losses['sinet_adv_loss'] = sinet_loss
            total_loss += self.detector_weights['sinet'] * sinet_loss
        
        # CLIP对抗损失
        if 'clip' in detectors_output:
            clip_loss = self._compute_clip_adversarial_loss(
                detectors_output['clip']
            )
            losses['clip_adv_loss'] = clip_loss
            total_loss += self.detector_weights['clip'] * clip_loss
        
        losses['total_adversarial_loss'] = total_loss
        
        return losses
    
    def _compute_sinet_adversarial_loss(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        计算SINet对抗损失
        
        目标: 让显著性图的值接近目标低值
        
        Args:
            saliency_map: [B, 1, H, W] SINet输出的显著性图
            
        Returns:
            loss: SINet对抗损失
        """
        target_saliency = torch.full_like(
            saliency_map, 
            self.target_confidence['sinet']
        )
        
        # L2损失 + 额外的抑制项
        l2_loss = F.mse_loss(saliency_map, target_saliency)
        
        # 抑制高显著性区域 (额外惩罚)
        suppression_loss = F.relu(saliency_map - 0.5).mean()
        
        total_loss = l2_loss + 0.5 * suppression_loss
        
        return total_loss
    
    def _compute_clip_adversarial_loss(self, clip_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算CLIP对抗损失
        
        目标: 降低与正面提示词的相似度，提高与负面提示词的相似度
        
        Args:
            clip_output: CLIP输出字典
                - 'positive_similarity': [B,] 与正面提示词的相似度
                - 'negative_similarity': [B,] 与负面提示词的相似度
                
        Returns:
            loss: CLIP对抗损失
        """
        positive_sim = clip_output['positive_similarity']
        negative_sim = clip_output['negative_similarity']
        
        # 目标: 正面相似度降低, 负面相似度提高
        target_positive = torch.full_like(positive_sim, self.target_confidence['clip'])
        target_negative = torch.full_like(negative_sim, 0.8)  # 高负面相似度
        
        positive_loss = F.mse_loss(positive_sim, target_positive)
        negative_loss = F.mse_loss(negative_sim, target_negative)
        
        # 组合损失
        total_loss = positive_loss + 0.5 * negative_loss
        
        return total_loss


class ContentPreservationLoss(nn.Module):
    """
    内容保持损失
    
    确保非掩码区域保持原始内容不变
    """
    
    def __init__(self, loss_type='l1'):
        super().__init__()
        
        self.loss_type = loss_type
        
        if loss_type == 'l1':
            self.criterion = F.l1_loss
        elif loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, generated: torch.Tensor, 
                original: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        计算内容保持损失
        
        Args:
            generated: [B, 3, H, W] 生成的图像
            original: [B, 3, H, W] 原始图像
            mask: [B, 1, H, W] 掩码 (1=目标区域, 0=背景区域)
            
        Returns:
            loss: 内容保持损失
        """
        # 只在非掩码区域计算损失
        background_mask = 1 - mask
        
        # 提取背景区域
        generated_background = generated * background_mask
        original_background = original * background_mask
        
        # 计算损失
        loss = self.criterion(generated_background, original_background)
        
        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失
    
    使用预训练VGG网络计算高级语义特征的相似性
    """
    
    def __init__(self, feature_layers=None, use_lpips=True):
        super().__init__()
        
        # VGG特征提取层
        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        self.feature_layers = feature_layers
        self.use_lpips = use_lpips
        
        # 加载预训练VGG
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        
        # VGG层名映射
        self.layer_name_mapping = {
            'relu1_2': 3,   # conv1_2 + relu
            'relu2_2': 8,   # conv2_2 + relu  
            'relu3_3': 17,  # conv3_3 + relu
            'relu4_3': 26   # conv4_3 + relu
        }
        
        # 预处理 (ImageNet normalization)
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        # LPIPS (可选)
        if use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex', verbose=False)
                for param in self.lpips_model.parameters():
                    param.requires_grad = False
            except ImportError:
                print("⚠️ LPIPS未安装，跳过LPIPS损失")
                self.lpips_model = None
        else:
            self.lpips_model = None
    
    def _preprocess(self, x):
        """VGG预处理"""
        # 假设输入是[-1, 1]，转换到[0, 1]
        if x.min() < 0:
            x = (x + 1) / 2
        
        # ImageNet标准化
        x = (x - self.mean) / self.std
        return x
    
    def _extract_vgg_features(self, x):
        """提取VGG特征"""
        x = self._preprocess(x)
        
        features = {}
        for name, layer_idx in self.layer_name_mapping.items():
            if name in self.feature_layers:
                for i in range(layer_idx + 1):
                    x = self.vgg[i](x)
                features[name] = x.clone()
        
        return features
    
    def forward(self, generated: torch.Tensor, 
                target: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算感知损失
        
        Args:
            generated: [B, 3, H, W] 生成图像
            target: [B, 3, H, W] 目标图像
            mask: [B, 1, H, W] 掩码 (可选)
            
        Returns:
            loss_dict: 感知损失字典
        """
        losses = {}
        
        # VGG感知损失
        gen_features = self._extract_vgg_features(generated)
        target_features = self._extract_vgg_features(target)
        
        vgg_loss = 0
        for layer in self.feature_layers:
            layer_loss = F.mse_loss(gen_features[layer], target_features[layer])
            losses[f'vgg_{layer}_loss'] = layer_loss
            vgg_loss += layer_loss
        
        losses['vgg_perceptual_loss'] = vgg_loss
        
        # LPIPS损失
        if self.lpips_model is not None:
            # LPIPS期望输入范围[-1, 1]
            gen_norm = generated if generated.min() >= -1 else generated * 2 - 1
            target_norm = target if target.min() >= -1 else target * 2 - 1
            
            lpips_loss = self.lpips_model(gen_norm, target_norm).mean()
            losses['lpips_loss'] = lpips_loss
        
        return losses


class TextureLoss(nn.Module):
    """
    纹理损失
    
    使用Gram矩阵保持纹理统计特性
    参考: "Neural Style Transfer" (Gatys et al.)
    """
    
    def __init__(self, feature_layers=None):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = ['relu2_2', 'relu3_3', 'relu4_3']
        
        self.feature_layers = feature_layers
        
        # 复用PerceptualLoss的VGG
        self.perceptual_extractor = PerceptualLoss(feature_layers, use_lpips=False)
    
    def _compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算Gram矩阵
        
        Args:
            features: [B, C, H, W] 特征图
            
        Returns:
            gram: [B, C, C] Gram矩阵
        """
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)
        
        # Gram矩阵 = F × F^T
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # 归一化
        gram = gram / (C * H * W)
        
        return gram
    
    def forward(self, generated: torch.Tensor, 
                reference: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算纹理损失
        
        Args:
            generated: [B, 3, H, W] 生成图像
            reference: [B, 3, H, W] 参考图像 (纹理源)
            mask: [B, 1, H, W] 掩码 (可选)
            
        Returns:
            texture_loss: 纹理损失
        """
        # 提取特征
        gen_features = self.perceptual_extractor._extract_vgg_features(generated)
        ref_features = self.perceptual_extractor._extract_vgg_features(reference)
        
        total_loss = 0
        
        for layer in self.feature_layers:
            # 计算Gram矩阵
            gen_gram = self._compute_gram_matrix(gen_features[layer])
            ref_gram = self._compute_gram_matrix(ref_features[layer])
            
            # Gram矩阵损失
            layer_loss = F.mse_loss(gen_gram, ref_gram)
            total_loss += layer_loss
        
        return total_loss


class PhysicalConstraintLoss(nn.Module):
    """
    物理约束损失
    
    确保生成的迷彩在物理世界可实现
    """
    
    def __init__(self):
        super().__init__()
        
        # 色域边界
        self.register_buffer('color_min', torch.tensor([0.05, 0.05, 0.05]))
        self.register_buffer('color_max', torch.tensor([0.95, 0.95, 0.95]))
    
    def forward(self, generated: torch.Tensor, 
                constraint_outputs: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算物理约束损失
        
        Args:
            generated: [B, 3, H, W] 生成图像
            constraint_outputs: 物理约束层的输出
            
        Returns:
            loss_dict: 物理约束损失字典
        """
        losses = {}
        
        # 1. 色域约束损失
        color_loss = self._compute_color_gamut_loss(generated)
        losses['color_gamut_loss'] = color_loss
        
        # 2. 平滑度约束损失 (避免高频噪声)
        smoothness_loss = self._compute_smoothness_loss(generated)
        losses['smoothness_loss'] = smoothness_loss
        
        # 3. 对比度约束损失 (保持合理对比度)
        contrast_loss = self._compute_contrast_loss(generated)
        losses['contrast_loss'] = contrast_loss
        
        # 4. 来自PhysicalConstraintLayer的损失
        if constraint_outputs is not None and 'constraint_loss' in constraint_outputs:
            losses['physical_constraint_loss'] = constraint_outputs['constraint_loss']
        
        # 总物理约束损失
        total_loss = sum(losses.values())
        losses['total_physical_loss'] = total_loss
        
        return losses
    
    def _compute_color_gamut_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算色域约束损失"""
        # 转换到[0, 1]范围
        if x.min() < 0:
            x_norm = (x + 1) / 2
        else:
            x_norm = x
        
        # 计算超出色域的程度
        under_min = F.relu(self.color_min.view(1, 3, 1, 1) - x_norm)
        over_max = F.relu(x_norm - self.color_max.view(1, 3, 1, 1))
        
        gamut_loss = (under_min + over_max).mean()
        
        return gamut_loss
    
    def _compute_smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算平滑度约束损失 (总变分损失)"""
        # 水平方向梯度
        h_grad = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        
        # 垂直方向梯度  
        v_grad = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        smoothness_loss = h_grad.mean() + v_grad.mean()
        
        return smoothness_loss
    
    def _compute_contrast_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算对比度约束损失"""
        # 计算局部标准差 (对比度指标)
        kernel_size = 5
        padding = kernel_size // 2
        
        # 使用均值滤波器
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
        mean_kernel = mean_kernel / (kernel_size * kernel_size)
        
        # 计算局部均值和方差
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        local_mean = F.conv2d(x_gray, mean_kernel, padding=padding)
        local_var = F.conv2d(x_gray**2, mean_kernel, padding=padding) - local_mean**2
        local_std = torch.sqrt(local_var + 1e-8)
        
        # 对比度损失: 惩罚过低或过高的对比度
        target_contrast = 0.2  # 目标对比度
        contrast_loss = F.mse_loss(local_std, torch.full_like(local_std, target_contrast))
        
        return contrast_loss


class RegularizationLoss(nn.Module):
    """
    正则化损失
    
    防止过拟合并稳定训练
    """
    
    def __init__(self, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay
    
    def forward(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        计算正则化损失
        
        Args:
            model: 要正则化的模型
            
        Returns:
            loss_dict: 正则化损失字典
        """
        losses = {}
        
        # L2权重衰减
        l2_loss = 0
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, 2)**2
        
        losses['l2_regularization'] = self.weight_decay * l2_loss
        
        return losses


class CompositeLoss(nn.Module):
    """
    复合损失函数
    
    整合所有损失组件，根据配置加权组合
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # 默认配置
        if config is None:
            from model_config import TrainingConfig
            config = TrainingConfig()
        
        self.config = config
        
        # 损失组件
        self.adversarial_loss = MultiModalAdversarialLoss()
        self.content_loss = ContentPreservationLoss()
        self.perceptual_loss = PerceptualLoss()
        self.texture_loss = TextureLoss()
        self.physical_loss = PhysicalConstraintLoss()
        self.regularization_loss = RegularizationLoss()
        
        # 损失权重
        self.weights = {
            'adversarial': config.adversarial_weight,
            'content': config.content_weight,
            'perceptual': config.perceptual_weight,
            'lpips': config.lpips_weight,
            'texture': config.texture_weight,
            'physical': config.physical_constraint_weight,
            'color': config.color_constraint_weight,
            'frequency': config.frequency_constraint_weight
        }
        
        print(f"✅ 复合损失函数初始化完成")
        print(f"   损失权重: {self.weights}")
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                model: nn.Module = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            predictions: 模型预测结果
            targets: 目标数据
            model: 模型 (用于正则化)
            
        Returns:
            loss_dict: 完整的损失字典
        """
        all_losses = {}
        total_loss = 0
        
        # 1. 对抗损失
        if 'detectors_output' in predictions:
            adv_losses = self.adversarial_loss(predictions['detectors_output'])
            all_losses.update(adv_losses)
            total_loss += self.weights['adversarial'] * adv_losses['total_adversarial_loss']
        
        # 2. 内容保持损失
        if 'generated_image' in predictions:
            content_loss = self.content_loss(
                predictions['generated_image'],
                targets['original_image'],
                targets['mask']
            )
            all_losses['content_loss'] = content_loss
            total_loss += self.weights['content'] * content_loss
        
        # 3. 感知损失
        if 'generated_image' in predictions:
            perc_losses = self.perceptual_loss(
                predictions['generated_image'],
                targets['original_image'],
                targets.get('mask')
            )
            all_losses.update(perc_losses)
            total_loss += self.weights['perceptual'] * perc_losses['vgg_perceptual_loss']
            
            if 'lpips_loss' in perc_losses:
                total_loss += self.weights['lpips'] * perc_losses['lpips_loss']
        
        # 4. 纹理损失
        if 'generated_texture' in predictions:
            texture_loss = self.texture_loss(
                predictions['generated_texture'],
                targets.get('reference_texture', targets['original_image'])
            )
            all_losses['texture_loss'] = texture_loss
            total_loss += self.weights['texture'] * texture_loss
        
        # 5. 物理约束损失
        if 'generated_image' in predictions:
            phys_losses = self.physical_loss(
                predictions['generated_image'],
                predictions.get('constraint_outputs')
            )
            all_losses.update(phys_losses)
            total_loss += self.weights['physical'] * phys_losses['total_physical_loss']
        
        # 6. 正则化损失
        if model is not None:
            reg_losses = self.regularization_loss(model)
            all_losses.update(reg_losses)
            total_loss += reg_losses['l2_regularization']
        
        # 总损失
        all_losses['total_loss'] = total_loss
        
        return all_losses
    
    def compute_metrics(self, predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 模型预测
            targets: 目标数据
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {}
        
        if 'detectors_output' in predictions:
            # SINet攻击成功率
            if 'sinet' in predictions['detectors_output']:
                sinet_output = predictions['detectors_output']['sinet']
                sinet_success_rate = (sinet_output < 0.5).float().mean().item()
                metrics['sinet_attack_success_rate'] = sinet_success_rate
            
            # CLIP攻击成功率
            if 'clip' in predictions['detectors_output']:
                clip_output = predictions['detectors_output']['clip']
                if 'positive_similarity' in clip_output:
                    clip_success_rate = (clip_output['positive_similarity'] < 0.5).float().mean().item()
                    metrics['clip_attack_success_rate'] = clip_success_rate
        
        # 图像质量指标
        if 'generated_image' in predictions:
            # PSNR
            psnr = self._compute_psnr(
                predictions['generated_image'],
                targets['original_image']
            )
            metrics['psnr'] = psnr
            
            # SSIM (简化版)
            ssim = self._compute_ssim(
                predictions['generated_image'],
                targets['original_image']
            )
            metrics['ssim'] = ssim
        
        return metrics
    
    def _compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(img1, img2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        return psnr.item()
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """计算SSIM (简化版)"""
        # 简化的SSIM实现
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()


def test_losses():
    """测试损失函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建复合损失
    composite_loss = CompositeLoss().to(device)
    
    # 测试数据
    batch_size = 2
    H, W = 256, 256
    
    predictions = {
        'generated_image': torch.randn(batch_size, 3, H, W).to(device),
        'generated_texture': torch.randn(batch_size, 3, H, W).to(device),
        'detectors_output': {
            'sinet': torch.sigmoid(torch.randn(batch_size, 1, H, W)).to(device),
            'clip': {
                'positive_similarity': torch.sigmoid(torch.randn(batch_size)).to(device),
                'negative_similarity': torch.sigmoid(torch.randn(batch_size)).to(device)
            }
        }
    }
    
    targets = {
        'original_image': torch.randn(batch_size, 3, H, W).to(device),
        'mask': torch.randint(0, 2, (batch_size, 1, H, W)).float().to(device)
    }
    
    # 计算损失
    losses = composite_loss(predictions, targets)
    
    print("✅ 损失函数测试成功")
    print("损失组件:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    # 计算指标
    metrics = composite_loss.compute_metrics(predictions, targets)
    print("\n评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_losses()
