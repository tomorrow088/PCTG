"""
双重对抗损失函数
Dual Adversarial Loss - 同时对抗人眼和AI检测

核心功能:
1. AI检测器对抗 (SINet + CLIP)
2. 人眼视觉对抗 (轮廓 + 显著性 + 颜色)
3. 自适应权重平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ContourSuppressionLoss(nn.Module):
    """
    轮廓抑制损失
    目标: 破坏目标区域的边缘清晰度，使人眼难以识别轮廓
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel算子 (边缘检测)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] 生成的图像
            mask: [B, 1, H, W] 目标掩码
            
        Returns:
            loss: 轮廓抑制损失
        """
        # 转换为灰度图
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # Sobel边缘检测
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # 边缘强度
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        
        # 只在掩码区域内计算
        masked_edges = edge_magnitude * mask
        
        # 损失: 最小化掩码区域的边缘强度
        loss = masked_edges.mean()
        
        return loss


class HumanSaliencyLoss(nn.Module):
    """
    人眼显著性损失
    基于简化的显著性检测算法
    """
    
    def __init__(self):
        super().__init__()
        
        # 高斯滤波器（用于多尺度显著性）
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size=5, sigma=1.0)
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """创建高斯核"""
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] 生成的图像
            mask: [B, 1, H, W] 目标掩码
            
        Returns:
            loss: 人眼显著性损失
        """
        # 计算局部对比度（简化的显著性）
        saliency_map = self._compute_local_contrast(image)
        
        # 只在掩码区域内计算
        masked_saliency = saliency_map * mask
        
        # 损失: 降低掩码区域的显著性
        loss = masked_saliency.mean()
        
        return loss
    
    def _compute_local_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """计算局部对比度作为显著性指标"""
        # 转换为Lab色彩空间的L通道（简化为灰度）
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # 高斯滤波获得局部平均
        if not hasattr(self, 'gaussian_kernel') or self.gaussian_kernel.device != image.device:
            self.gaussian_kernel = self._create_gaussian_kernel(5, 1.0).to(image.device)
        
        local_mean = F.conv2d(gray, self.gaussian_kernel.expand(1, 1, -1, -1), padding=2)
        
        # 局部对比度 = |像素值 - 局部平均|
        local_contrast = torch.abs(gray - local_mean)
        
        return local_contrast


class ColorCamouflageLoss(nn.Module):
    """
    颜色伪装损失
    使目标区域颜色与背景融合
    """
    
    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
    
    def forward(self, generated: torch.Tensor, 
                original: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: [B, 3, H, W] 生成的图像
            original: [B, 3, H, W] 原始图像
            mask: [B, 1, H, W] 目标掩码
            
        Returns:
            loss: 颜色伪装损失
        """
        # 提取背景和目标区域
        background_mask = 1 - mask
        background_colors = original * background_mask
        target_colors = generated * mask
        
        # 计算颜色分布统计
        bg_mean, bg_std = self._compute_color_statistics(background_colors, background_mask)
        target_mean, target_std = self._compute_color_statistics(target_colors, mask)
        
        # 损失: 目标区域颜色分布接近背景
        mean_loss = F.mse_loss(target_mean, bg_mean)
        std_loss = F.mse_loss(target_std, bg_std)
        
        total_loss = mean_loss + 0.5 * std_loss
        
        return total_loss
    
    def _compute_color_statistics(self, image: torch.Tensor, 
                                  mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算区域内的颜色统计"""
        # 展平空间维度
        B, C, H, W = image.shape
        image_flat = image.view(B, C, -1)  # [B, 3, H*W]
        mask_flat = mask.view(B, 1, -1)    # [B, 1, H*W]
        
        # 只统计掩码区域
        masked_pixels = image_flat * mask_flat
        num_pixels = mask_flat.sum(dim=2, keepdim=True).clamp(min=1)
        
        # 均值
        mean = masked_pixels.sum(dim=2) / num_pixels.squeeze(-1)  # [B, 3]
        
        # 标准差
        variance = ((masked_pixels - mean.unsqueeze(-1))**2 * mask_flat).sum(dim=2) / num_pixels.squeeze(-1)
        std = torch.sqrt(variance + 1e-8)  # [B, 3]
        
        return mean, std


class TextureContinuityLoss(nn.Module):
    """
    纹理连续性损失
    确保目标区域边界与背景纹理平滑过渡
    """
    
    def __init__(self, boundary_width: int = 5):
        super().__init__()
        self.boundary_width = boundary_width
    
    def forward(self, generated: torch.Tensor, 
                original: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: [B, 3, H, W] 生成的图像
            original: [B, 3, H, W] 原始图像
            mask: [B, 1, H, W] 目标掩码
            
        Returns:
            loss: 边界平滑损失
        """
        # 提取边界区域
        boundary_mask = self._extract_boundary(mask)
        
        # 在边界区域计算生成图像和原图的梯度
        gen_grad = self._compute_gradient(generated)
        orig_grad = self._compute_gradient(original)
        
        # 边界区域的梯度应该相似
        boundary_grad_loss = F.mse_loss(
            gen_grad * boundary_mask,
            orig_grad * boundary_mask
        )
        
        return boundary_grad_loss
    
    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """提取掩码边界"""
        # 膨胀
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        # 腐蚀
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        # 边界 = 膨胀 - 腐蚀
        boundary = dilated - eroded
        
        return boundary
    
    def _compute_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """计算图像梯度"""
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        # Padding to match original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return gradient


class DualAdversarialLoss(nn.Module):
    """
    双重对抗损失 - 主类
    
    整合AI对抗和人眼对抗
    """
    
    def __init__(self, 
                 ai_weight: float = 1.0,
                 human_weight: float = 0.7,
                 contour_weight: float = 0.5,
                 saliency_weight: float = 0.5,
                 color_weight: float = 0.3,
                 texture_weight: float = 0.2):
        """
        Args:
            ai_weight: AI对抗总权重
            human_weight: 人眼对抗总权重
            contour_weight: 轮廓抑制权重
            saliency_weight: 显著性降低权重
            color_weight: 颜色伪装权重
            texture_weight: 纹理连续性权重
        """
        super().__init__()
        
        # 权重
        self.ai_weight = ai_weight
        self.human_weight = human_weight
        
        # AI对抗损失（在主损失函数中计算）
        # 这里只定义人眼对抗损失
        
        # 人眼对抗损失组件
        self.contour_loss = ContourSuppressionLoss()
        self.saliency_loss = HumanSaliencyLoss()
        self.color_loss = ColorCamouflageLoss()
        self.texture_loss = TextureContinuityLoss()
        
        # 组件权重
        self.contour_weight = contour_weight
        self.saliency_weight = saliency_weight
        self.color_weight = color_weight
        self.texture_weight = texture_weight
        
        print(f"✅ 双重对抗损失初始化")
        print(f"   AI权重: {ai_weight}, 人眼权重: {human_weight}")
        print(f"   轮廓: {contour_weight}, 显著性: {saliency_weight}")
        print(f"   颜色: {color_weight}, 纹理: {texture_weight}")
    
    def compute_human_adversarial_loss(self, 
                                       generated: torch.Tensor,
                                       original: torch.Tensor,
                                       mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算人眼对抗损失
        
        Args:
            generated: [B, 3, H, W] 生成的图像
            original: [B, 3, H, W] 原始图像
            mask: [B, 1, H, W] 目标掩码
            
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 1. 轮廓抑制
        contour_loss = self.contour_loss(generated, mask)
        losses['contour_suppression'] = contour_loss
        
        # 2. 人眼显著性降低
        saliency_loss = self.saliency_loss(generated, mask)
        losses['human_saliency'] = saliency_loss
        
        # 3. 颜色伪装
        color_loss = self.color_loss(generated, original, mask)
        losses['color_camouflage'] = color_loss
        
        # 4. 纹理连续性
        texture_loss = self.texture_loss(generated, original, mask)
        losses['texture_continuity'] = texture_loss
        
        # 总人眼对抗损失
        total_human_loss = (
            self.contour_weight * contour_loss +
            self.saliency_weight * saliency_loss +
            self.color_weight * color_loss +
            self.texture_weight * texture_loss
        )
        
        losses['total_human_adversarial'] = total_human_loss
        
        return losses
    
    def forward(self,
                generated: torch.Tensor,
                original: torch.Tensor,
                mask: torch.Tensor,
                ai_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算完整的双重对抗损失
        
        Args:
            generated: [B, 3, H, W] 生成的图像
            original: [B, 3, H, W] 原始图像
            mask: [B, 1, H, W] 目标掩码
            ai_losses: AI检测器的对抗损失
            
        Returns:
            all_losses: 完整损失字典
        """
        all_losses = {}
        
        # 1. AI对抗损失（已在外部计算）
        ai_adv_loss = ai_losses.get('total_adversarial_loss', torch.tensor(0.0))
        all_losses['ai_adversarial'] = ai_adv_loss
        all_losses.update({f"ai_{k}": v for k, v in ai_losses.items()})
        
        # 2. 人眼对抗损失
        human_losses = self.compute_human_adversarial_loss(
            generated, original, mask
        )
        all_losses.update({f"human_{k}": v for k, v in human_losses.items()})
        
        # 3. 总损失
        total_dual_loss = (
            self.ai_weight * ai_adv_loss +
            self.human_weight * human_losses['total_human_adversarial']
        )
        
        all_losses['total_dual_adversarial_loss'] = total_dual_loss
        
        return all_losses
    
    def get_metrics(self, 
                   generated: torch.Tensor,
                   original: torch.Tensor,
                   mask: torch.Tensor) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            generated: [B, 3, H, W] 生成的图像
            original: [B, 3, H, W] 原始图像
            mask: [B, 1, H, W] 目标掩码
            
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        with torch.no_grad():
            # 边缘强度
            gray = 0.299 * generated[:, 0:1] + 0.587 * generated[:, 1:2] + 0.114 * generated[:, 2:3]
            sobel_x = F.conv2d(gray, self.contour_loss.sobel_x, padding=1)
            sobel_y = F.conv2d(gray, self.contour_loss.sobel_y, padding=1)
            edge_strength = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
            masked_edge_strength = (edge_strength * mask).sum() / mask.sum()
            metrics['edge_strength'] = masked_edge_strength.item()
            
            # 显著性
            saliency = self.saliency_loss._compute_local_contrast(generated)
            masked_saliency = (saliency * mask).sum() / mask.sum()
            metrics['saliency_level'] = masked_saliency.item()
            
            # 颜色差异
            bg_mask = 1 - mask
            bg_mean = (original * bg_mask).sum(dim=[2, 3]) / bg_mask.sum(dim=[2, 3]).clamp(min=1)
            target_mean = (generated * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3]).clamp(min=1)
            color_distance = torch.norm(bg_mean - target_mean, dim=1).mean()
            metrics['color_distance'] = color_distance.item()
        
        return metrics


# ============================================================
# 工具函数：集成到现有训练流程
# ============================================================

def integrate_dual_adversarial_loss(composite_loss, config):
    """
    将双重对抗损失集成到现有的CompositeLoss中
    
    Args:
        composite_loss: 现有的CompositeLoss实例
        config: 配置字典
        
    Returns:
        enhanced_loss: 增强后的损失函数
    """
    # 创建双重对抗损失
    dual_adv_loss = DualAdversarialLoss(
        ai_weight=config.get('ai_weight', 1.0),
        human_weight=config.get('human_weight', 0.7),
        contour_weight=config.get('contour_weight', 0.5),
        saliency_weight=config.get('saliency_weight', 0.5),
        color_weight=config.get('color_weight', 0.3),
        texture_weight=config.get('texture_weight', 0.2)
    )
    
    # 将其添加到composite_loss
    composite_loss.dual_adversarial_loss = dual_adv_loss
    
    return composite_loss


def compute_dual_adversarial_loss_wrapper(predictions, targets, composite_loss):
    """
    计算包含双重对抗的完整损失
    
    在trainer.py的_train_step中使用
    """
    # 1. 计算原有损失（包含AI对抗）
    original_losses = composite_loss(predictions, targets)
    
    # 2. 提取AI对抗损失
    ai_losses = {
        'total_adversarial_loss': original_losses.get('total_adversarial_loss', torch.tensor(0.0)),
        'sinet_adv_loss': original_losses.get('sinet_adv_loss', torch.tensor(0.0)),
        'clip_adv_loss': original_losses.get('clip_adv_loss', torch.tensor(0.0))
    }
    
    # 3. 计算双重对抗损失
    if hasattr(composite_loss, 'dual_adversarial_loss'):
        dual_losses = composite_loss.dual_adversarial_loss(
            generated=predictions['generated_image'],
            original=targets['original_image'],
            mask=targets['mask'],
            ai_losses=ai_losses
        )
        
        # 4. 合并损失
        original_losses.update(dual_losses)
        
        # 5. 更新总损失
        original_losses['total_loss'] = (
            original_losses['total_loss'] - ai_losses['total_adversarial_loss'] +
            dual_losses['total_dual_adversarial_loss']
        )
    
    return original_losses


if __name__ == "__main__":
    # 测试双重对抗损失
    print("🧪 测试双重对抗损失...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试数据
    B, C, H, W = 2, 3, 256, 256
    generated = torch.randn(B, C, H, W).to(device)
    original = torch.randn(B, C, H, W).to(device)
    mask = torch.randint(0, 2, (B, 1, H, W)).float().to(device)
    
    # 创建双重对抗损失
    dual_loss = DualAdversarialLoss(
        ai_weight=1.0,
        human_weight=0.7
    ).to(device)
    
    # 模拟AI损失
    ai_losses = {
        'total_adversarial_loss': torch.tensor(0.5).to(device),
        'sinet_adv_loss': torch.tensor(0.3).to(device),
        'clip_adv_loss': torch.tensor(0.2).to(device)
    }
    
    # 计算损失
    all_losses = dual_loss(generated, original, mask, ai_losses)
    
    print("✅ 双重对抗损失测试成功")
    print("损失组件:")
    for key, value in all_losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    # 计算指标
    metrics = dual_loss.get_metrics(generated, original, mask)
    print("\n评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
