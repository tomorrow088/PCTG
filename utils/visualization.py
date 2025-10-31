"""
可视化工具模块
用于训练过程和结果的可视化
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from torchvision.utils import make_grid


class Visualizer:
    """训练和评估可视化器"""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib风格
        plt.style.use('seaborn-v0_8-darkgrid')
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """
        将Tensor转换为可显示的numpy图像
        
        Args:
            tensor: 图像tensor [C, H, W] 或 [B, C, H, W]
        
        Returns:
            numpy图像 [H, W, C]
        """
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        image = tensor.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)
        return image
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线
        
        Args:
            history: 包含训练历史的字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # 损失曲线
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 攻击成功率
        if 'attack_success_rate' in history:
            axes[0, 1].plot(history['attack_success_rate'], 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Attack Success Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # SINet分数
        if 'sinet_score' in history:
            axes[1, 0].plot(history['sinet_score'], 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('SINet Saliency Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # CLIP相似度
        if 'clip_similarity' in history:
            axes[1, 1].plot(history['clip_similarity'], 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Similarity')
            axes[1, 1].set_title('CLIP Similarity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_adversarial_examples(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        mask: torch.Tensor,
        sinet_scores: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        num_samples: int = 4
    ):
        """
        可视化对抗样本
        
        Args:
            original: 原始图像 [B, 3, H, W]
            adversarial: 对抗图像 [B, 3, H, W]
            mask: 掩码 [B, 1, H, W]
            sinet_scores: SINet显著性图 [B, 1, H, W]
            save_path: 保存路径
            num_samples: 显示样本数量
        """
        batch_size = min(num_samples, original.size(0))
        
        if sinet_scores is not None:
            fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4*batch_size))
            cols = ['Original', 'Mask', 'Adversarial', 'Difference', 'SINet Score']
        else:
            fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4*batch_size))
            cols = ['Original', 'Mask', 'Adversarial', 'Difference']
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # 原始图像
            orig_img = self.tensor_to_image(original[i])
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(cols[0] if i == 0 else '')
            axes[i, 0].axis('off')
            
            # 掩码
            mask_img = mask[i, 0].detach().cpu().numpy()
            axes[i, 1].imshow(mask_img, cmap='gray')
            axes[i, 1].set_title(cols[1] if i == 0 else '')
            axes[i, 1].axis('off')
            
            # 对抗图像
            adv_img = self.tensor_to_image(adversarial[i])
            axes[i, 2].imshow(adv_img)
            axes[i, 2].set_title(cols[2] if i == 0 else '')
            axes[i, 2].axis('off')
            
            # 差异
            diff = np.abs(orig_img - adv_img)
            axes[i, 3].imshow(diff)
            axes[i, 3].set_title(cols[3] if i == 0 else '')
            axes[i, 3].axis('off')
            
            # SINet显著性图
            if sinet_scores is not None:
                sinet_map = sinet_scores[i, 0].detach().cpu().numpy()
                axes[i, 4].imshow(sinet_map, cmap='jet')
                axes[i, 4].set_title(cols[4] if i == 0 else '')
                axes[i, 4].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'adversarial_examples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_attention_maps(
        self,
        images: torch.Tensor,
        attention_maps: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        可视化注意力图
        
        Args:
            images: 输入图像 [B, 3, H, W]
            attention_maps: 注意力图 [B, 1, H, W]
            save_path: 保存路径
        """
        batch_size = min(4, images.size(0))
        fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4*batch_size))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # 原始图像
            img = self.tensor_to_image(images[i])
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Image' if i == 0 else '')
            axes[i, 0].axis('off')
            
            # 注意力图
            attn = attention_maps[i, 0].detach().cpu().numpy()
            axes[i, 1].imshow(attn, cmap='jet')
            axes[i, 1].set_title('Attention' if i == 0 else '')
            axes[i, 1].axis('off')
            
            # 叠加
            overlay = img.copy()
            attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]))
            attn_colored = plt.cm.jet(attn_resized)[:, :, :3]
            overlay = 0.6 * overlay + 0.4 * attn_colored
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay' if i == 0 else '')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'attention_maps.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_comparison_grid(
        self,
        images_dict: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
        nrow: int = 4
    ):
        """
        创建图像比较网格
        
        Args:
            images_dict: 图像字典 {名称: tensor}
            save_path: 保存路径
            nrow: 每行图像数量
        """
        fig, axes = plt.subplots(len(images_dict), 1, figsize=(16, 4*len(images_dict)))
        
        if len(images_dict) == 1:
            axes = [axes]
        
        for idx, (name, images) in enumerate(images_dict.items()):
            grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
            grid_img = self.tensor_to_image(grid)
            
            axes[idx].imshow(grid_img)
            axes[idx].set_title(name, fontsize=14, fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'comparison_grid.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics_list: List[Dict[str, float]],
        labels: List[str],
        save_path: Optional[str] = None
    ):
        """
        绘制多个模型的指标对比
        
        Args:
            metrics_list: 指标列表
            labels: 标签列表
            save_path: 保存路径
        """
        metric_names = list(metrics_list[0].keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(labels)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
            values = [metrics[name] for name in metric_names]
            offset = (i - len(labels)/2) * width + width/2
            ax.bar(x + offset, values, width, label=label)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_image_grid(
        self,
        images: torch.Tensor,
        save_path: str,
        nrow: int = 8,
        normalize: bool = True
    ):
        """
        保存图像网格
        
        Args:
            images: 图像tensor [B, C, H, W]
            save_path: 保存路径
            nrow: 每行图像数量
            normalize: 是否归一化
        """
        grid = make_grid(images, nrow=nrow, normalize=normalize, padding=2)
        grid_img = self.tensor_to_image(grid)
        
        plt.figure(figsize=(16, 16))
        plt.imshow(grid_img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class TensorboardLogger:
    """Tensorboard日志记录器"""
    
    def __init__(self, log_dir: str = "outputs/logs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except:
            print("警告: Tensorboard不可用，日志将不会记录")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量"""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """记录图像"""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int):
        """记录多张图像"""
        if self.enabled:
            grid = make_grid(images, normalize=True)
            self.writer.add_image(tag, grid, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """记录直方图"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """关闭writer"""
        if self.enabled:
            self.writer.close()


if __name__ == "__main__":
    # 测试可视化
    print("=== 测试可视化工具 ===")
    
    visualizer = Visualizer()
    
    # 测试训练曲线
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
        'attack_success_rate': [0.3, 0.5, 0.7, 0.8, 0.9]
    }
    visualizer.plot_training_curves(history, 'test_curves.png')
    print("✓ 训练曲线已保存")
    
    # 测试对抗样本可视化
    original = torch.rand(4, 3, 256, 256)
    adversarial = torch.rand(4, 3, 256, 256)
    mask = torch.rand(4, 1, 256, 256)
    visualizer.visualize_adversarial_examples(
        original, adversarial, mask,
        save_path='test_adversarial.png'
    )
    print("✓ 对抗样本可视化已保存")
    
    print("\n所有测试通过! ✓")
