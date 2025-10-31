"""
数据集配置文件
定义数据集路径、预处理和加载相关配置
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DatasetConfig:
    """数据集配置"""
    
    # 数据集路径
    root_dir: str = "data"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    
    # 图像路径
    image_dir: str = "images"
    mask_dir: str = "masks"
    background_dir: str = "backgrounds"
    
    # 数据集类型
    dataset_type: str = "camouflage"  # camouflage, coco, custom
    
    # 图像尺寸
    image_size: Tuple[int, int] = (512, 512)
    input_channels: int = 3
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # 数据增强
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: bool = True
    rotation_degrees: Tuple[int, int] = (-30, 30)
    color_jitter: bool = True
    color_jitter_params: dict = field(default_factory=lambda: {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    })
    random_crop: bool = True
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    
    # 归一化参数
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 缓存设置
    cache_images: bool = False
    cache_mode: str = "memory"  # memory, disk, none
    
    # 数据分割
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 文件格式
    image_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png'])
    mask_extensions: List[str] = field(default_factory=lambda: ['.png', '.jpg'])
    
    # 数据过滤
    min_object_size: int = 32  # 最小目标尺寸（像素）
    max_aspect_ratio: float = 10.0  # 最大宽高比
    
    def __post_init__(self):
        """验证配置的有效性"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "数据集分割比例之和必须为1.0"
        assert self.image_size[0] > 0 and self.image_size[1] > 0, \
            "图像尺寸必须大于0"
        assert self.num_workers >= 0, "num_workers必须>=0"
    
    def get_train_path(self) -> Path:
        """获取训练集路径"""
        return Path(self.root_dir) / self.train_dir
    
    def get_val_path(self) -> Path:
        """获取验证集路径"""
        return Path(self.root_dir) / self.val_dir
    
    def get_test_path(self) -> Path:
        """获取测试集路径"""
        return Path(self.root_dir) / self.test_dir
    
    def get_image_path(self, split: str = "train") -> Path:
        """获取图像路径"""
        return Path(self.root_dir) / split / self.image_dir
    
    def get_mask_path(self, split: str = "train") -> Path:
        """获取掩码路径"""
        return Path(self.root_dir) / split / self.mask_dir


@dataclass
class COCODatasetConfig(DatasetConfig):
    """COCO数据集特定配置"""
    
    dataset_type: str = "coco"
    annotation_file: str = "annotations/instances_train2017.json"
    
    # COCO类别
    target_categories: List[str] = field(default_factory=lambda: [
        "person", "cat", "dog", "bird", "car"
    ])
    
    # 过滤设置
    min_bbox_area: int = 1000
    min_mask_area: int = 500


@dataclass
class CamouflageDatasetConfig(DatasetConfig):
    """伪装数据集特定配置"""
    
    dataset_type: str = "camouflage"
    
    # 伪装特定参数
    include_hard_samples: bool = True
    hard_sample_ratio: float = 0.3
    
    # 背景复杂度
    min_background_complexity: float = 0.5  # 0-1之间


@dataclass
class CustomDatasetConfig(DatasetConfig):
    """自定义数据集配置"""
    
    dataset_type: str = "custom"
    
    # 自定义类别
    class_names: List[str] = field(default_factory=lambda: ["object"])
    num_classes: int = 1
    
    # 自定义标注格式
    annotation_format: str = "mask"  # mask, bbox, polygon
    annotation_file: Optional[str] = None


# 预定义数据集配置
def get_debug_dataset_config() -> DatasetConfig:
    """调试模式数据集配置"""
    config = DatasetConfig()
    config.cache_images = False
    config.num_workers = 0
    config.persistent_workers = False
    return config


def get_cat_dataset_config() -> CamouflageDatasetConfig:
    """猫数据集配置"""
    config = CamouflageDatasetConfig()
    config.root_dir = "data/cats"
    config.image_size = (512, 512)
    return config


def get_coco_person_config() -> COCODatasetConfig:
    """COCO人物数据集配置"""
    config = COCODatasetConfig()
    config.root_dir = "data/coco"
    config.target_categories = ["person"]
    return config


# 数据集工厂函数
def get_dataset_config(dataset_type: str, **kwargs) -> DatasetConfig:
    """
    根据数据集类型获取配置
    
    Args:
        dataset_type: 数据集类型 (camouflage, coco, custom)
        **kwargs: 额外配置参数
    
    Returns:
        DatasetConfig: 数据集配置对象
    """
    config_map = {
        'camouflage': CamouflageDatasetConfig,
        'coco': COCODatasetConfig,
        'custom': CustomDatasetConfig,
    }
    
    if dataset_type not in config_map:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    config_class = config_map[dataset_type]
    return config_class(**kwargs)


if __name__ == "__main__":
    # 测试配置
    print("=== 基础数据集配置 ===")
    config = DatasetConfig()
    print(f"训练集路径: {config.get_train_path()}")
    print(f"图像尺寸: {config.image_size}")
    print(f"数据增强: {config.color_jitter}")
    
    print("\n=== 猫数据集配置 ===")
    cat_config = get_cat_dataset_config()
    print(f"根目录: {cat_config.root_dir}")
    print(f"硬样本比例: {cat_config.hard_sample_ratio}")
    
    print("\n=== COCO数据集配置 ===")
    coco_config = get_coco_person_config()
    print(f"目标类别: {coco_config.target_categories}")
    print(f"标注文件: {coco_config.annotation_file}")
