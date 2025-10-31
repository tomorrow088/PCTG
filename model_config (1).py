"""
模型配置文件
Physical-Constrained Texture Generator + SINet 对抗攻击配置
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch.nn as nn


@dataclass
class PCTGConfig:
    """PCTG生成器配置"""
    
    # ============================================================
    # 网络架构参数
    # ============================================================
    
    # 编码器配置
    encoder_name: str = "efficientnet_b3"          # backbone模型
    encoder_pretrained: bool = True                # 是否使用预训练权重
    encoder_frozen_stages: int = 2                 # 冻结前N层
    
    # 解码器配置
    decoder_channels: List[int] = [512, 256, 128, 64]  # 解码器通道数
    decoder_upsampling: str = "bilinear"           # 上采样方式
    
    # 纹理合成模块
    texture_synthesis_layers: int = 3              # 纹理合成层数
    texture_kernel_size: int = 3                   # 纹理卷积核大小
    use_texture_attention: bool = True             # 是否使用纹理注意力
    
    # 输出配置
    output_channels: int = 3                       # 输出通道数(RGB)
    output_activation: str = "tanh"                # 输出激活函数
    
    # ============================================================
    # 物理约束参数
    # ============================================================
    
    # 色域约束
    color_space: str = "rgb"                       # 颜色空间
    printable_colors_only: bool = True             # 是否限制为可打印颜色
    cmyk_gamma: float = 2.2                        # CMYK gamma校正
    
    # 频率约束  
    max_frequency_ratio: float = 0.8               # 最大频率比例
    texture_smoothness: float = 0.1                # 纹理平滑度约束
    
    # 光照不变性
    lighting_augmentation: bool = True             # 光照增强
    shadow_simulation: bool = True                 # 阴影模拟


@dataclass 
class SINetConfig:
    """SINet检测器配置"""
    
    # 模型参数
    model_name: str = "sinet_resnet50"             # SINet变种
    pretrained_path: str = "checkpoints/sinet_pretrained.pth"
    input_size: Tuple[int, int] = (352, 352)       # 输入尺寸
    
    # 推理参数
    threshold: float = 0.5                         # 检测阈值
    nms_threshold: float = 0.4                     # NMS阈值
    
    # 对抗参数
    target_confidence: float = 0.1                 # 目标置信度(攻击目标)
    gradient_clip: float = 1.0                     # 梯度裁剪


@dataclass
class CLIPConfig:
    """CLIP检测器配置"""
    
    # 模型参数
    model_name: str = "ViT-B/32"                   # CLIP模型变种
    device: str = "cuda"                           # 设备
    
    # 文本提示
    positive_prompts: List[str] = None             # 正面提示词（描述目标）
    negative_prompts: List[str] = None             # 负面提示词（描述背景）
    
    # 对抗参数
    target_similarity: float = 0.3                # 目标相似度（越低越好）
    
    def __post_init__(self):
        if self.positive_prompts is None:
            self.positive_prompts = [
                "a person",
                "a human", 
                "a soldier",
                "military personnel",
                "someone wearing camouflage"
            ]
        
        if self.negative_prompts is None:
            self.negative_prompts = [
                "background",
                "trees",
                "grass", 
                "nature",
                "empty scene"
            ]


@dataclass
class MultiDetectorConfig:
    """多检测器配置"""
    
    # 检测器列表
    detector_types: List[str] = None               # 检测器类型
    detector_weights: Dict[str, float] = None      # 检测器权重
    
    # 联合优化
    joint_optimization: bool = True                # 联合优化
    alternating_training: bool = False             # 交替训练
    
    def __post_init__(self):
        if self.detector_types is None:
            self.detector_types = ["sinet", "clip"]
            
        if self.detector_weights is None:
            self.detector_weights = {
                "sinet": 1.0,
                "clip": 0.8
            }


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # ============================================================
    # 基础训练参数
    # ============================================================
    
    # 优化器配置
    optimizer: str = "adamw"                       # 优化器类型
    learning_rate: float = 5e-6                   # 学习率
    weight_decay: float = 1e-4                    # 权重衰减
    beta1: float = 0.9                            # Adam beta1
    beta2: float = 0.999                          # Adam beta2
    
    # 学习率调度
    lr_scheduler: str = "cosine"                   # 学习率调度器
    warmup_epochs: int = 5                        # 预热轮数
    min_lr_ratio: float = 0.1                     # 最小学习率比例
    
    # 训练设置
    epochs: int = 100                              # 训练轮数
    batch_size: int = 4                           # 批次大小
    accumulation_steps: int = 4                   # 梯度累积步数
    mixed_precision: bool = True                  # 混合精度训练
    
    # ============================================================
    # 损失函数权重
    # ============================================================
    
    # 对抗损失
    adversarial_weight: float = 1.0               # 对抗损失权重
    
    # 内容保持损失
    content_weight: float = 10.0                  # 内容损失权重
    
    # 感知损失
    perceptual_weight: float = 1.0                # 感知损失权重
    lpips_weight: float = 0.5                     # LPIPS损失权重
    
    # 纹理损失
    texture_weight: float = 0.5                   # 纹理损失权重(Gram矩阵)
    
    # 物理约束损失
    physical_constraint_weight: float = 0.2       # 物理约束损失权重
    color_constraint_weight: float = 0.1          # 色域约束权重
    frequency_constraint_weight: float = 0.1      # 频率约束权重
    
    # ============================================================
    # 验证与保存
    # ============================================================
    
    # 验证设置
    val_interval: int = 5                         # 验证间隔(epochs)
    val_batch_size: int = 8                       # 验证批次大小
    
    # 模型保存
    save_interval: int = 10                       # 保存间隔(epochs)
    save_best_only: bool = True                   # 仅保存最佳模型
    monitor_metric: str = "attack_success_rate"    # 监控指标
    
    # 早停
    early_stopping: bool = True                   # 早停
    patience: int = 20                            # 早停耐心
    
    # ============================================================
    # 数据增强
    # ============================================================
    
    # 几何变换
    random_crop: bool = True                      # 随机裁剪
    random_flip: bool = True                      # 随机翻转
    random_rotation: float = 10.0                 # 随机旋转角度
    
    # 颜色变换
    color_jitter: bool = True                     # 颜色抖动
    brightness_range: Tuple[float, float] = (0.8, 1.2)   # 亮度范围
    contrast_range: Tuple[float, float] = (0.8, 1.2)     # 对比度范围
    
    # 物理增强
    lighting_augmentation: bool = True            # 光照增强
    shadow_augmentation: bool = True              # 阴影增强
    noise_augmentation: bool = True               # 噪声增强


@dataclass
class InferenceConfig:
    """推理配置"""
    
    # 模型加载
    checkpoint_path: str = "checkpoints/best_model.pth"
    device: str = "cuda"                          # 推理设备
    
    # 推理参数
    batch_size: int = 1                           # 推理批次大小
    output_size: Tuple[int, int] = None           # 输出尺寸(None=保持原尺寸)
    
    # 后处理
    apply_postprocessing: bool = True             # 后处理
    gamma_correction: float = 1.0                 # Gamma校正
    
    # 可视化
    save_intermediate: bool = False               # 保存中间结果
    visualization: bool = True                    # 可视化


# ============================================================
# 配置工厂函数
# ============================================================

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'pctg': PCTGConfig(),
        'sinet': SINetConfig(), 
        'clip': CLIPConfig(),
        'multi_detector': MultiDetectorConfig(),
        'training': TrainingConfig(),
        'inference': InferenceConfig()
    }


def get_paper_config() -> Dict[str, Any]:
    """获取论文实验配置"""
    config = get_default_config()
    
    # 论文实验的特殊设置
    config['training'].epochs = 200
    config['training'].learning_rate = 3e-6
    config['training'].batch_size = 8
    
    # 更严格的物理约束
    config['training'].physical_constraint_weight = 0.5
    config['pctg'].printable_colors_only = True
    
    return config


def get_debug_config() -> Dict[str, Any]:
    """获取调试配置"""
    config = get_default_config()
    
    # 调试设置 - 小数据量快速验证
    config['training'].epochs = 10
    config['training'].batch_size = 2
    config['training'].val_interval = 2
    config['training'].save_interval = 5
    
    return config


if __name__ == "__main__":
    # 配置测试
    config = get_default_config()
    print("✅ 配置加载成功")
    print(f"PCTG编码器: {config['pctg'].encoder_name}")
    print(f"训练轮数: {config['training'].epochs}")
    print(f"学习率: {config['training'].learning_rate}")
