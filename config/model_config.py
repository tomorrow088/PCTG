"""
模型配置文件
定义PCTG、SINet、CLIP等所有模型的配置参数
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PCTGConfig:
    """PCTG生成器配置"""
    
    # 编码器配置
    encoder_name: str = "efficientnet-b3"  # 编码器骨干网络
    pretrained: bool = True
    
    # 网络结构
    hidden_dim: int = 256
    num_residual_blocks: int = 6
    num_scale_levels: int = 3
    
    # 对抗条件化
    adversarial_dim: int = 128
    use_attention: bool = True
    
    # 物理约束
    enable_physical_constraints: bool = True
    color_constraint_weight: float = 0.4
    smoothness_weight: float = 0.2
    printability_weight: float = 0.3
    
    # 输入输出
    input_channels: int = 3
    output_channels: int = 3
    image_size: Tuple[int, int] = (512, 512)


@dataclass
class SINetConfig:
    """SINet检测器配置"""
    
    # 模型路径
    checkpoint: str = "checkpoints/sinet/SINet_COD10K.pth"
    
    # 骨干网络
    backbone: str = "resnet50"
    
    # 输入配置
    input_size: Tuple[int, int] = (352, 352)
    input_channels: int = 3
    
    # 输出配置
    output_stride: int = 4
    return_intermediate: bool = False
    
    # 检测阈值
    detection_threshold: float = 0.5
    
    # 是否使用原始SINet代码
    use_original_code: bool = False
    original_code_path: Optional[str] = "third_party/SINet"


@dataclass
class CLIPConfig:
    """CLIP模型配置"""
    
    # 模型选择
    model_name: str = "ViT-L/14"  # ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14
    checkpoint: Optional[str] = "checkpoints/clip/vit-l-14.pt"
    
    # 输入配置
    input_size: Tuple[int, int] = (224, 224)
    
    # 文本提示词（需要根据数据集目标修改）
    positive_prompts: Optional[List[str]] = None  # 会在__post_init__中初始化
    negative_prompts: Optional[List[str]] = None
    
    # CLIP参数
    embed_dim: int = 768  # ViT-L/14的嵌入维度
    context_length: int = 77
    
    # 对抗攻击阈值
    attack_threshold: float = 0.3
    
    def __post_init__(self):
        """初始化默认提示词"""
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
class ModelConfig:
    """
    完整模型配置
    包含PCTG、SINet、CLIP的所有配置
    """
    
    # 子模型配置
    pctg: PCTGConfig = field(default_factory=PCTGConfig)
    sinet: SINetConfig = field(default_factory=SINetConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    
    # 设备配置
    device: str = "cuda"
    mixed_precision: bool = True
    
    # 损失权重
    sinet_weight: float = 1.0
    clip_weight: float = 0.8
    perceptual_weight: float = 0.5
    physical_weight: float = 0.3
    
    # 双重对抗配置
    enable_dual_adversarial: bool = False
    human_adversarial_weight: float = 0.7
    ai_adversarial_weight: float = 1.0
    
    # 人眼对抗损失权重
    contour_suppression_weight: float = 0.5
    human_saliency_weight: float = 0.5
    color_camouflage_weight: float = 0.3
    texture_continuity_weight: float = 0.2
    
    # 性能配置
    use_compile: bool = False  # PyTorch 2.0 compile
    inference_batch_size: int = 8
    
    def __post_init__(self):
        """验证配置"""
        # 确保权重为正数
        assert self.sinet_weight >= 0, "sinet_weight必须>=0"
        assert self.clip_weight >= 0, "clip_weight必须>=0"
        
        # 如果启用双重对抗，权重之和应该合理
        if self.enable_dual_adversarial:
            total_weight = self.human_adversarial_weight + self.ai_adversarial_weight
            assert total_weight > 0, "双重对抗权重之和必须>0"
    
    def get_loss_weights(self) -> dict:
        """获取所有损失权重"""
        weights = {
            'sinet': self.sinet_weight,
            'clip': self.clip_weight,
            'perceptual': self.perceptual_weight,
            'physical': self.physical_weight
        }
        
        if self.enable_dual_adversarial:
            weights.update({
                'contour': self.contour_suppression_weight,
                'human_saliency': self.human_saliency_weight,
                'color_camouflage': self.color_camouflage_weight,
                'texture_continuity': self.texture_continuity_weight
            })
        
        return weights
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'pctg': self.pctg.__dict__,
            'sinet': self.sinet.__dict__,
            'clip': self.clip.__dict__,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'loss_weights': self.get_loss_weights()
        }


# 预定义配置
def get_default_config() -> ModelConfig:
    """获取默认配置"""
    return ModelConfig()


def get_cat_config() -> ModelConfig:
    """获取猫数据集配置"""
    config = ModelConfig()
    config.clip.positive_prompts = [
        "a cat",
        "a kitten",
        "a feline",
        "a domestic cat",
        "a pet cat",
        "a cat sitting",
        "a cat lying down",
        "a furry animal",
        "an animal with whiskers"
    ]
    config.clip.negative_prompts = [
        "background",
        "floor",
        "carpet",
        "furniture",
        "sofa",
        "empty room",
        "no animals"
    ]
    return config


def get_person_config() -> ModelConfig:
    """获取人物数据集配置"""
    config = ModelConfig()
    config.clip.positive_prompts = [
        "a person",
        "a human",
        "someone",
        "a man",
        "a woman",
        "a soldier",
        "military personnel",
        "a person standing",
        "a person walking"
    ]
    config.clip.negative_prompts = [
        "background",
        "trees",
        "grass",
        "nature",
        "empty scene",
        "landscape",
        "forest"
    ]
    return config


def get_dual_adversarial_config() -> ModelConfig:
    """获取双重对抗配置"""
    config = ModelConfig()
    config.enable_dual_adversarial = True
    config.human_adversarial_weight = 0.7
    config.ai_adversarial_weight = 1.0
    return config


def get_paper_config() -> ModelConfig:
    """获取论文实验配置"""
    config = ModelConfig()
    config.mixed_precision = True
    config.sinet_weight = 1.0
    config.clip_weight = 0.8
    config.perceptual_weight = 0.5
    config.physical_weight = 0.3
    return config


# 配置工厂函数
def create_config(
    dataset_type: str = "default",
    enable_dual_adversarial: bool = False,
    **kwargs
) -> ModelConfig:
    """
    创建配置的便捷函数
    
    Args:
        dataset_type: 数据集类型 (default, cat, person)
        enable_dual_adversarial: 是否启用双重对抗
        **kwargs: 其他配置参数
    
    Returns:
        ModelConfig实例
    """
    # 根据数据集类型选择基础配置
    config_map = {
        'default': get_default_config,
        'cat': get_cat_config,
        'person': get_person_config,
        'paper': get_paper_config
    }
    
    if dataset_type in config_map:
        config = config_map[dataset_type]()
    else:
        config = get_default_config()
    
    # 启用双重对抗
    if enable_dual_adversarial:
        config.enable_dual_adversarial = True
        config.human_adversarial_weight = kwargs.get('human_weight', 0.7)
        config.ai_adversarial_weight = kwargs.get('ai_weight', 1.0)
    
    # 应用其他参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # 测试配置
    print("=== 测试模型配置 ===\n")
    
    # 默认配置
    print("1. 默认配置:")
    config = get_default_config()
    print(f"  PCTG编码器: {config.pctg.encoder_name}")
    print(f"  SINet权重: {config.sinet.checkpoint}")
    print(f"  CLIP模型: {config.clip.model_name}")
    print(f"  正面提示词: {config.clip.positive_prompts[:2]}...")
    
    # 猫配置
    print("\n2. 猫数据集配置:")
    cat_config = get_cat_config()
    print(f"  正面提示词: {cat_config.clip.positive_prompts[:3]}")
    print(f"  负面提示词: {cat_config.clip.negative_prompts[:3]}")
    
    # 双重对抗配置
    print("\n3. 双重对抗配置:")
    dual_config = get_dual_adversarial_config()
    print(f"  启用双重对抗: {dual_config.enable_dual_adversarial}")
    print(f"  人眼权重: {dual_config.human_adversarial_weight}")
    print(f"  AI权重: {dual_config.ai_adversarial_weight}")
    
    # 论文配置
    print("\n4. 论文实验配置:")
    paper_config = get_paper_config()
    print(f"  损失权重: {paper_config.get_loss_weights()}")
    
    # 自定义配置
    print("\n5. 自定义配置:")
    custom_config = create_config(
        dataset_type='cat',
        enable_dual_adversarial=True,
        human_weight=0.8
    )
    print(f"  数据集类型: cat")
    print(f"  双重对抗: {custom_config.enable_dual_adversarial}")
    print(f"  CLIP提示词数量: {len(custom_config.clip.positive_prompts)}")
    
    print("\n✓ 所有配置测试通过!")
