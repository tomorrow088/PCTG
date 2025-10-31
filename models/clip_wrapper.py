"""
CLIP模型封装器
用于文本-图像匹配和语义对抗攻击
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import clip
from PIL import Image


class CLIPWrapper(nn.Module):
    """
    CLIP模型封装器
    支持文本提示词匹配和对抗损失计算
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None
    ):
        """
        Args:
            model_name: CLIP模型名称
            device: 设备
            checkpoint_path: 本地权重路径（可选）
        """
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        
        # 加载CLIP模型
        if checkpoint_path:
            self.model, self.preprocess = clip.load(
                checkpoint_path,
                device=device,
                download_root=None
            )
        else:
            self.model, self.preprocess = clip.load(
                model_name,
                device=device
            )
        
        # 冻结CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # CLIP输入标准化参数
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    
    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        标准化图像到CLIP输入格式
        
        Args:
            image: 输入图像 [B, 3, H, W]，范围[0,1]
        
        Returns:
            标准化后的图像
        """
        return (image - self.clip_mean) / self.clip_std
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        编码文本提示词
        
        Args:
            text_prompts: 文本提示词列表
        
        Returns:
            文本特征 [N, D]
        """
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 输入图像 [B, 3, H, W]
        
        Returns:
            图像特征 [B, D]
        """
        # 调整尺寸到CLIP输入大小
        if images.shape[-1] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 标准化
        images = self.normalize_image(images)
        
        # 编码
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def compute_similarity(
        self,
        images: torch.Tensor,
        text_prompts: List[str]
    ) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        Args:
            images: 输入图像 [B, 3, H, W]
            text_prompts: 文本提示词列表
        
        Returns:
            相似度矩阵 [B, N]
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_prompts)
        
        # 计算余弦相似度
        similarity = 100.0 * image_features @ text_features.T
        
        return similarity
    
    def adversarial_loss(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        positive_prompts: List[str],
        negative_prompts: List[str],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算CLIP对抗损失
        目标：让对抗样本与positive_prompts相似度降低，与negative_prompts相似度提高
        
        Args:
            original_images: 原始图像 [B, 3, H, W]
            adversarial_images: 对抗图像 [B, 3, H, W]
            positive_prompts: 正面提示词（要降低相似度）
            negative_prompts: 负面提示词（要提高相似度）
            mask: 掩码 [B, 1, H, W]（可选）
        
        Returns:
            对抗损失标量
        """
        # 编码文本
        positive_features = self.encode_text(positive_prompts)  # [N_pos, D]
        negative_features = self.encode_text(negative_prompts)  # [N_neg, D]
        
        # 编码图像
        if mask is not None:
            # 仅对掩码区域应用对抗扰动
            adversarial_images = original_images * (1 - mask) + adversarial_images * mask
        
        adv_image_features = self.encode_image(adversarial_images)  # [B, D]
        
        # 计算与正面提示词的相似度（要降低）
        positive_similarity = 100.0 * adv_image_features @ positive_features.T  # [B, N_pos]
        positive_loss = positive_similarity.mean()
        
        # 计算与负面提示词的相似度（要提高）
        negative_similarity = 100.0 * adv_image_features @ negative_features.T  # [B, N_neg]
        negative_loss = -negative_similarity.mean()  # 负号表示要最大化
        
        # 总损失
        total_loss = positive_loss + negative_loss
        
        return total_loss
    
    def evaluate_attack_success(
        self,
        adversarial_images: torch.Tensor,
        positive_prompts: List[str],
        threshold: float = 0.5
    ) -> float:
        """
        评估攻击成功率
        
        Args:
            adversarial_images: 对抗图像 [B, 3, H, W]
            positive_prompts: 正面提示词
            threshold: 成功阈值
        
        Returns:
            攻击成功率（0-1之间）
        """
        with torch.no_grad():
            similarity = self.compute_similarity(adversarial_images, positive_prompts)
            max_similarity = similarity.max(dim=1)[0]  # 每张图与最相似提示词的相似度
            
            # 相似度低于阈值视为攻击成功
            success = (max_similarity < threshold).float()
            success_rate = success.mean().item()
        
        return success_rate
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        text_prompts: List[str],
        return_probs: bool = True
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        预测图像对应的文本类别
        
        Args:
            images: 输入图像 [B, 3, H, W]
            text_prompts: 候选文本提示词
            return_probs: 是否返回概率
        
        Returns:
            如果return_probs=True: 概率分布 [B, N]
            否则: 预测类别索引 [B]
        """
        similarity = self.compute_similarity(images, text_prompts)
        
        if return_probs:
            probs = F.softmax(similarity, dim=-1)
            return probs
        else:
            predictions = similarity.argmax(dim=-1)
            return predictions
    
    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像
            text_prompts: 文本提示词（可选）
        
        Returns:
            如果提供text_prompts: 相似度矩阵
            否则: 图像特征
        """
        if text_prompts is not None:
            return self.compute_similarity(images, text_prompts)
        else:
            return self.encode_image(images)


# 辅助函数
def load_clip_model(
    model_name: str = "ViT-L/14",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> CLIPWrapper:
    """
    加载CLIP模型的便捷函数
    
    Args:
        model_name: 模型名称
        checkpoint_path: 本地权重路径
        device: 设备
    
    Returns:
        CLIPWrapper实例
    """
    return CLIPWrapper(model_name, device, checkpoint_path)


if __name__ == "__main__":
    # 测试CLIP模型
    print("=== 测试CLIP模型 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    clip_model = CLIPWrapper(model_name="ViT-B/32", device=device)
    print(f"✓ 加载CLIP模型: ViT-B/32")
    
    # 测试图像编码
    dummy_images = torch.randn(2, 3, 224, 224).to(device)
    image_features = clip_model.encode_image(dummy_images)
    print(f"✓ 图像特征形状: {image_features.shape}")
    
    # 测试文本编码
    text_prompts = ["a cat", "a dog", "a bird"]
    text_features = clip_model.encode_text(text_prompts)
    print(f"✓ 文本特征形状: {text_features.shape}")
    
    # 测试相似度计算
    similarity = clip_model.compute_similarity(dummy_images, text_prompts)
    print(f"✓ 相似度矩阵形状: {similarity.shape}")
    print(f"  相似度值: {similarity[0].tolist()}")
    
    # 测试预测
    predictions = clip_model.predict(dummy_images, text_prompts, return_probs=False)
    print(f"✓ 预测类别: {predictions.tolist()}")
    
    print("\n所有测试通过! ✓")
