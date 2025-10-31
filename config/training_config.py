"""
训练配置文件
定义所有训练相关的超参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """训练超参数配置"""
    
    # 基础训练参数
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 学习率调度
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 优化器配置
    optimizer: str = "adamw"  # adamw, adam, sgd
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)
    
    # 梯度相关
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    
    # 混合精度训练
    use_amp: bool = True
    amp_opt_level: str = "O1"
    
    # 分布式训练
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # 验证和保存
    val_interval: int = 1  # 每多少个epoch验证一次
    save_interval: int = 5  # 每多少个epoch保存一次
    keep_checkpoint_max: int = 5  # 最多保留几个checkpoint
    
    # 早停
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # 日志和可视化
    log_interval: int = 10  # 每多少个step记录一次
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "PCTG"
    
    # 实验管理
    experiment_name: str = "default"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None  # 恢复训练的checkpoint路径
    
    # 数据增强
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # 调试模式
    debug: bool = False
    debug_samples: int = 100
    
    # 损失权重
    adversarial_weight: float = 1.0
    perceptual_weight: float = 0.5
    physical_weight: float = 0.3
    dual_adversarial_weight: float = 0.7
    
    # GPU设置
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # 随机种子
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """验证配置的有效性"""
        assert self.epochs > 0, "epochs必须大于0"
        assert self.batch_size > 0, "batch_size必须大于0"
        assert self.learning_rate > 0, "learning_rate必须大于0"
        assert 0 <= self.augmentation_prob <= 1, "augmentation_prob必须在[0,1]之间"


@dataclass
class LossWeights:
    """损失函数权重配置"""
    
    # AI对抗损失
    sinet_loss: float = 1.0
    clip_loss: float = 0.8
    
    # 人眼对抗损失
    contour_loss: float = 0.5
    saliency_loss: float = 0.5
    color_camouflage_loss: float = 0.3
    texture_continuity_loss: float = 0.2
    
    # 感知损失
    perceptual_loss: float = 0.5
    style_loss: float = 0.3
    
    # 物理约束损失
    color_constraint_loss: float = 0.4
    smoothness_loss: float = 0.2
    printability_loss: float = 0.3
    
    # 正则化损失
    tv_loss: float = 0.1
    l1_loss: float = 0.1
    
    def get_ai_weights(self) -> dict:
        """获取AI对抗权重"""
        return {
            'sinet': self.sinet_loss,
            'clip': self.clip_loss
        }
    
    def get_human_weights(self) -> dict:
        """获取人眼对抗权重"""
        return {
            'contour': self.contour_loss,
            'saliency': self.saliency_loss,
            'color': self.color_camouflage_loss,
            'texture': self.texture_continuity_loss
        }
    
    def get_physical_weights(self) -> dict:
        """获取物理约束权重"""
        return {
            'color': self.color_constraint_loss,
            'smoothness': self.smoothness_loss,
            'printability': self.printability_loss
        }


# 预定义训练配置
def get_debug_config() -> TrainingConfig:
    """调试模式配置"""
    config = TrainingConfig()
    config.debug = True
    config.epochs = 2
    config.batch_size = 2
    config.debug_samples = 20
    config.log_interval = 1
    config.val_interval = 1
    return config


def get_paper_config() -> TrainingConfig:
    """论文实验配置"""
    config = TrainingConfig()
    config.epochs = 100
    config.batch_size = 16
    config.learning_rate = 1e-4
    config.experiment_name = "paper_experiment"
    config.early_stopping = True
    config.patience = 20
    return config


def get_dual_adversarial_config() -> TrainingConfig:
    """双重对抗模式配置"""
    config = TrainingConfig()
    config.epochs = 150
    config.dual_adversarial_weight = 1.0
    config.experiment_name = "dual_adversarial"
    return config


if __name__ == "__main__":
    # 测试配置
    config = TrainingConfig()
    print("默认训练配置:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    
    debug_config = get_debug_config()
    print("\n调试模式配置:")
    print(f"  Debug: {debug_config.debug}")
    print(f"  Epochs: {debug_config.epochs}")
    print(f"  Samples: {debug_config.debug_samples}")
