"""
评估指标模块
用于计算模型性能的各种指标
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import average_precision_score, roc_auc_score


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有累积的指标"""
        self.metrics_history = {
            'attack_success_rate': [],
            'sinet_score': [],
            'clip_similarity': [],
            'perceptual_distance': [],
            'physical_feasibility': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        results = {}
        for key, values in self.metrics_history.items():
            if values:
                results[key] = np.mean(values)
            else:
                results[key] = 0.0
        return results
    
    @staticmethod
    def compute_attack_success_rate(
        sinet_scores: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        计算SINet攻击成功率
        
        Args:
            sinet_scores: SINet显著性分数 [B, 1, H, W]
            threshold: 成功阈值
        
        Returns:
            攻击成功率（0-1）
        """
        # 计算平均显著性分数
        mean_scores = sinet_scores.mean(dim=[1, 2, 3])
        
        # 分数低于阈值视为攻击成功
        success = (mean_scores < threshold).float()
        success_rate = success.mean().item()
        
        return success_rate
    
    @staticmethod
    def compute_clip_attack_success_rate(
        clip_similarities: torch.Tensor,
        threshold: float = 0.3
    ) -> float:
        """
        计算CLIP攻击成功率
        
        Args:
            clip_similarities: CLIP相似度 [B, N]
            threshold: 成功阈值
        
        Returns:
            攻击成功率（0-1）
        """
        # 取最大相似度
        max_similarity = clip_similarities.max(dim=1)[0]
        
        # 相似度低于阈值视为攻击成功
        success = (max_similarity < threshold).float()
        success_rate = success.mean().item()
        
        return success_rate
    
    @staticmethod
    def compute_perceptual_distance(
        original: torch.Tensor,
        adversarial: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        计算感知距离（L2距离）
        
        Args:
            original: 原始图像 [B, 3, H, W]
            adversarial: 对抗图像 [B, 3, H, W]
            mask: 掩码 [B, 1, H, W]
        
        Returns:
            平均L2距离
        """
        if mask is not None:
            diff = (original - adversarial) * mask
        else:
            diff = original - adversarial
        
        l2_dist = torch.norm(diff.reshape(diff.size(0), -1), p=2, dim=1)
        mean_dist = l2_dist.mean().item()
        
        return mean_dist
    
    @staticmethod
    def compute_physical_feasibility(
        adversarial: torch.Tensor,
        color_range: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        计算物理可行性（颜色是否在可打印范围内）
        
        Args:
            adversarial: 对抗图像 [B, 3, H, W]
            color_range: 可打印颜色范围
        
        Returns:
            可行性分数（0-1）
        """
        min_val, max_val = color_range
        
        # 检查是否在范围内
        in_range = ((adversarial >= min_val) & (adversarial <= max_val)).float()
        feasibility = in_range.mean().item()
        
        return feasibility
    
    @staticmethod
    def compute_ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11
    ) -> float:
        """
        计算结构相似性指标（SSIM）
        
        Args:
            img1: 图像1 [B, C, H, W]
            img2: 图像2 [B, C, H, W]
            window_size: 窗口大小
        
        Returns:
            SSIM分数（0-1）
        """
        # 简化版SSIM实现
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    @staticmethod
    def compute_psnr(
        img1: torch.Tensor,
        img2: torch.Tensor,
        max_val: float = 1.0
    ) -> float:
        """
        计算峰值信噪比（PSNR）
        
        Args:
            img1: 图像1 [B, C, H, W]
            img2: 图像2 [B, C, H, W]
            max_val: 最大像素值
        
        Returns:
            PSNR值（dB）
        """
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def compute_iou(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        计算IoU（交并比）
        
        Args:
            pred_mask: 预测掩码 [B, 1, H, W]
            gt_mask: 真实掩码 [B, 1, H, W]
            threshold: 二值化阈值
        
        Returns:
            IoU分数（0-1）
        """
        pred_binary = (pred_mask > threshold).float()
        gt_binary = (gt_mask > threshold).float()
        
        intersection = (pred_binary * gt_binary).sum(dim=[1, 2, 3])
        union = (pred_binary + gt_binary).clamp(0, 1).sum(dim=[1, 2, 3])
        
        iou = (intersection / (union + 1e-6)).mean().item()
        return iou
    
    @staticmethod
    def compute_average_precision(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        计算平均精度（AP）
        
        Args:
            predictions: 预测分数
            targets: 真实标签
        
        Returns:
            AP分数（0-1）
        """
        try:
            ap = average_precision_score(targets, predictions)
        except:
            ap = 0.0
        return ap
    
    def compute_comprehensive_metrics(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        masks: torch.Tensor,
        sinet_scores: torch.Tensor,
        clip_similarities: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算综合评估指标
        
        Args:
            original_images: 原始图像 [B, 3, H, W]
            adversarial_images: 对抗图像 [B, 3, H, W]
            masks: 掩码 [B, 1, H, W]
            sinet_scores: SINet分数 [B, 1, H, W]
            clip_similarities: CLIP相似度 [B, N]
        
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 攻击成功率
        metrics['sinet_asr'] = self.compute_attack_success_rate(sinet_scores)
        metrics['clip_asr'] = self.compute_clip_attack_success_rate(clip_similarities)
        metrics['combined_asr'] = (metrics['sinet_asr'] + metrics['clip_asr']) / 2
        
        # 感知质量
        metrics['l2_distance'] = self.compute_perceptual_distance(
            original_images, adversarial_images, masks
        )
        metrics['ssim'] = self.compute_ssim(original_images, adversarial_images)
        metrics['psnr'] = self.compute_psnr(original_images, adversarial_images)
        
        # 物理可行性
        metrics['physical_feasibility'] = self.compute_physical_feasibility(adversarial_images)
        
        # 显著性统计
        metrics['mean_sinet_score'] = sinet_scores.mean().item()
        metrics['max_clip_similarity'] = clip_similarities.max(dim=1)[0].mean().item()
        
        return metrics


class AttackEvaluator:
    """攻击效果评估器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_single_image(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        mask: torch.Tensor,
        sinet_model,
        clip_model,
        positive_prompts: List[str]
    ) -> Dict[str, float]:
        """
        评估单张图像的攻击效果
        
        Args:
            original: 原始图像
            adversarial: 对抗图像
            mask: 掩码
            sinet_model: SINet模型
            clip_model: CLIP模型
            positive_prompts: 正面提示词
        
        Returns:
            评估指标字典
        """
        with torch.no_grad():
            # 获取SINet分数
            sinet_scores = sinet_model(adversarial)
            
            # 获取CLIP相似度
            clip_similarities = clip_model.compute_similarity(adversarial, positive_prompts)
            
            # 计算综合指标
            metrics = self.metrics_calculator.compute_comprehensive_metrics(
                original.unsqueeze(0),
                adversarial.unsqueeze(0),
                mask.unsqueeze(0),
                sinet_scores,
                clip_similarities
            )
        
        return metrics
    
    def evaluate_batch(
        self,
        dataloader,
        sinet_model,
        clip_model,
        generator,
        positive_prompts: List[str]
    ) -> Dict[str, float]:
        """
        批量评估攻击效果
        
        Args:
            dataloader: 数据加载器
            sinet_model: SINet模型
            clip_model: CLIP模型
            generator: 生成器模型
            positive_prompts: 正面提示词
        
        Returns:
            平均评估指标
        """
        self.metrics_calculator.reset()
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            backgrounds = batch.get('background', torch.zeros_like(images)).to(self.device)
            
            with torch.no_grad():
                # 生成对抗纹理
                adversarial_images = generator(images, masks, backgrounds)
                
                # 获取检测结果
                sinet_scores = sinet_model(adversarial_images)
                clip_similarities = clip_model.compute_similarity(adversarial_images, positive_prompts)
                
                # 计算指标
                batch_metrics = self.metrics_calculator.compute_comprehensive_metrics(
                    images, adversarial_images, masks, sinet_scores, clip_similarities
                )
                
                self.metrics_calculator.update(batch_metrics)
        
        return self.metrics_calculator.compute()


if __name__ == "__main__":
    # 测试指标计算
    print("=== 测试评估指标 ===")
    
    calculator = MetricsCalculator()
    
    # 测试攻击成功率
    dummy_scores = torch.rand(4, 1, 64, 64)
    asr = calculator.compute_attack_success_rate(dummy_scores, threshold=0.5)
    print(f"✓ 攻击成功率: {asr:.4f}")
    
    # 测试感知距离
    img1 = torch.rand(4, 3, 256, 256)
    img2 = torch.rand(4, 3, 256, 256)
    l2_dist = calculator.compute_perceptual_distance(img1, img2)
    print(f"✓ L2距离: {l2_dist:.4f}")
    
    # 测试SSIM
    ssim = calculator.compute_ssim(img1, img2)
    print(f"✓ SSIM: {ssim:.4f}")
    
    # 测试PSNR
    psnr = calculator.compute_psnr(img1, img2)
    print(f"✓ PSNR: {psnr:.2f} dB")
    
    # 测试IoU
    mask1 = torch.rand(4, 1, 64, 64)
    mask2 = torch.rand(4, 1, 64, 64)
    iou = calculator.compute_iou(mask1, mask2)
    print(f"✓ IoU: {iou:.4f}")
    
    print("\n所有测试通过! ✓")
