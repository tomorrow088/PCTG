"""
数据集模块
Adversarial Camouflage Dataset for SINet Attack

数据集结构:
data/
├── images/          # 原始图像
├── masks/           # 目标掩码 (可选)
├── backgrounds/     # 背景图像 (可选)
└── annotations/     # 标注文件 (可选)

支持的数据格式:
- 图像: .jpg, .png, .jpeg
- 掩码: .png (二值图像)
- 标注: .json, .xml
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AdversarialCamouflageDataset(Dataset):
    """
    对抗性迷彩数据集
    
    支持多种数据源:
    1. 人物图像 + 自动生成掩码
    2. 人物图像 + 手动标注掩码  
    3. 通用目标检测数据集
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transforms: Optional[A.Compose] = None,
                 mask_mode: str = 'auto',  # 'auto', 'provided', 'bbox'
                 target_size: Tuple[int, int] = (256, 256),
                 debug: bool = False):
        """
        Args:
            data_dir: 数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            transforms: 数据变换
            mask_mode: 掩码模式
            target_size: 目标图像尺寸
            debug: 调试模式 (使用少量数据)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms
        self.mask_mode = mask_mode
        self.target_size = target_size
        self.debug = debug
        
        # 数据路径
        self.images_dir = self.data_dir / 'images' / split
        self.masks_dir = self.data_dir / 'masks' / split
        self.annotations_dir = self.data_dir / 'annotations' / split
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        # 调试模式 - 限制数据量
        if debug:
            self.data_list = self.data_list[:min(100, len(self.data_list))]
        
        print(f"✅ {split}数据集加载完成: {len(self.data_list)}张图像")
        print(f"   数据目录: {self.data_dir}")
        print(f"   掩码模式: {mask_mode}")
        print(f"   图像尺寸: {target_size}")
    
    def _load_data_list(self) -> List[Dict]:
        """加载数据列表"""
        data_list = []
        
        if not self.images_dir.exists():
            # 如果没有分离的train/val目录，使用根目录
            self.images_dir = self.data_dir / 'images'
            self.masks_dir = self.data_dir / 'masks'
            self.annotations_dir = self.data_dir / 'annotations'
        
        if not self.images_dir.exists():
            print(f"⚠️ 图像目录不存在: {self.images_dir}")
            print("📝 创建示例数据结构...")
            self._create_dummy_data()
            return self._load_data_list()
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # 遍历图像文件
        for image_path in self.images_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                
                # 构建对应的掩码和标注路径
                mask_path = self.masks_dir / f"{image_path.stem}.png"
                annotation_path = self.annotations_dir / f"{image_path.stem}.json"
                
                data_item = {
                    'image_path': image_path,
                    'mask_path': mask_path if mask_path.exists() else None,
                    'annotation_path': annotation_path if annotation_path.exists() else None,
                    'image_id': image_path.stem
                }
                
                data_list.append(data_item)
        
        return data_list
    
    def _create_dummy_data(self):
        """创建示例数据 (调试用)"""
        print("🎭 创建示例数据...")
        
        # 创建目录
        for dir_path in [self.images_dir, self.masks_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 生成示例图像
        for i in range(10):
            # 创建随机图像
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            image_pil = Image.fromarray(image)
            image_path = self.images_dir / f"sample_{i:03d}.jpg"
            image_pil.save(image_path)
            
            # 创建随机掩码
            mask = np.zeros((256, 256), dtype=np.uint8)
            # 在中心区域创建一个椭圆掩码
            cv2.ellipse(mask, (128, 128), (60, 40), 0, 0, 360, 255, -1)
            mask_pil = Image.fromarray(mask)
            mask_path = self.masks_dir / f"sample_{i:03d}.png"
            mask_pil.save(mask_path)
            
            # 创建示例标注
            annotation = {
                'image_id': f"sample_{i:03d}",
                'objects': [{
                    'class': 'person',
                    'bbox': [68, 88, 128, 128],  # [x, y, w, h]
                    'segmentation': [[68, 88, 196, 88, 196, 216, 68, 216]]
                }]
            }
            
            annotation_path = self.annotations_dir / f"sample_{i:03d}.json"
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f)
        
        print(f"   创建了{10}个示例数据文件")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        data_item = self.data_list[idx]
        
        # 加载图像
        image = self._load_image(data_item['image_path'])
        
        # 加载或生成掩码
        mask = self._load_or_generate_mask(data_item, image.shape[:2])
        
        # 应用变换
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 确保掩码是正确的形状
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif len(mask.shape) == 3 and mask.shape[0] == 3:
            mask = mask[0:1]  # 取第一个通道
        
        # 掩码二值化
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'image_id': data_item['image_id'],
            'image_path': str(data_item['image_path'])
        }
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """加载图像"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def _load_or_generate_mask(self, data_item: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """加载或生成掩码"""
        
        if self.mask_mode == 'provided' and data_item['mask_path'] is not None:
            # 加载提供的掩码
            mask = cv2.imread(str(data_item['mask_path']), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                return mask
        
        elif self.mask_mode == 'bbox' and data_item['annotation_path'] is not None:
            # 从边界框生成掩码
            mask = self._generate_mask_from_bbox(data_item['annotation_path'])
            if mask is not None:
                return mask
        
        # 自动生成掩码 (默认)
        return self._generate_automatic_mask(image_shape)
    
    def _generate_mask_from_bbox(self, annotation_path: Path) -> Optional[np.ndarray]:
        """从边界框标注生成掩码"""
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            mask = np.zeros(self.target_size[::-1], dtype=np.uint8)  # (H, W)
            
            for obj in annotation.get('objects', []):
                if obj.get('class') in ['person', 'human', 'people']:
                    bbox = obj['bbox']  # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    # 调整边界框到目标尺寸
                    scale_x = self.target_size[0] / annotation.get('image_width', self.target_size[0])
                    scale_y = self.target_size[1] / annotation.get('image_height', self.target_size[1])
                    
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    
                    # 填充矩形掩码
                    mask[y:y+h, x:x+w] = 255
            
            return mask
            
        except Exception as e:
            print(f"⚠️ 加载标注失败: {e}")
            return None
    
    def _generate_automatic_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """自动生成掩码 (中心椭圆区域)"""
        mask = np.zeros(self.target_size[::-1], dtype=np.uint8)  # (H, W)
        
        # 在图像中心生成椭圆掩码
        center_x, center_y = self.target_size[0] // 2, self.target_size[1] // 2
        
        # 椭圆参数 (随机化增加多样性)
        radius_x = random.randint(30, 80)
        radius_y = random.randint(40, 100)
        angle = random.randint(-30, 30)
        
        cv2.ellipse(
            mask, 
            (center_x, center_y), 
            (radius_x, radius_y), 
            angle, 0, 360, 255, -1
        )
        
        return mask


def get_train_transforms(image_size: int = 256, augmentation: bool = True) -> A.Compose:
    """获取训练时的数据变换"""
    
    transforms_list = [
        A.Resize(image_size, image_size),
    ]
    
    if augmentation:
        # 几何变换
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomResizedCrop(
                height=image_size, 
                width=image_size, 
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2),
                p=0.3
            ),
        ])
        
        # 颜色变换
        transforms_list.extend([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2, 
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
        ])
        
        # 噪声和模糊
        transforms_list.extend([
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ])
    
    # 归一化和张量转换
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(
        transforms_list,
        additional_targets={'mask': 'mask'}
    )


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """获取验证时的数据变换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


def create_data_loaders(data_dir: str, 
                       config: Dict,
                       num_workers: int = 4,
                       debug: bool = False) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 获取变换
    train_transforms = get_train_transforms(
        image_size=config.get('image_size', 256),
        augmentation=True
    )
    
    val_transforms = get_val_transforms(
        image_size=config.get('image_size', 256)
    )
    
    # 创建数据集
    train_dataset = AdversarialCamouflageDataset(
        data_dir=data_dir,
        split='train',
        transforms=train_transforms,
        debug=debug
    )
    
    val_dataset = AdversarialCamouflageDataset(
        data_dir=data_dir,
        split='val', 
        transforms=val_transforms,
        debug=debug
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    ) if len(val_dataset) > 0 else None
    
    return train_loader, val_loader


class VisualizationUtils:
    """可视化工具"""
    
    @staticmethod
    def visualize_batch(batch: Dict[str, torch.Tensor], 
                       predictions: Optional[Dict[str, torch.Tensor]] = None,
                       save_path: Optional[str] = None):
        """可视化一个批次的数据"""
        import matplotlib.pyplot as plt
        
        images = batch['image']
        masks = batch['mask']
        batch_size = min(4, images.shape[0])  # 最多显示4张图
        
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        # 创建子图
        cols = 3 if predictions is None else 4
        fig, axes = plt.subplots(batch_size, cols, figsize=(cols*3, batch_size*3))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # 原图
            axes[i, 0].imshow(images_denorm[i].permute(1, 2, 0))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # 掩码
            axes[i, 1].imshow(masks[i, 0], cmap='gray')
            axes[i, 1].set_title('Target Mask')
            axes[i, 1].axis('off')
            
            # 掩码叠加
            overlay = images_denorm[i].permute(1, 2, 0).clone()
            mask_colored = torch.zeros_like(overlay)
            mask_colored[:, :, 0] = masks[i, 0]  # 红色通道
            overlay = 0.7 * overlay + 0.3 * mask_colored
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Mask Overlay')
            axes[i, 2].axis('off')
            
            # 生成结果 (如果有)
            if predictions is not None and 'generated_image' in predictions:
                gen_image = predictions['generated_image'][i]
                gen_image_denorm = gen_image * std + mean
                gen_image_denorm = torch.clamp(gen_image_denorm, 0, 1)
                
                axes[i, 3].imshow(gen_image_denorm.permute(1, 2, 0))
                axes[i, 3].set_title('Generated Image')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def test_dataset():
    """测试数据集"""
    print("🧪 测试数据集...")
    
    # 创建测试数据集
    dataset = AdversarialCamouflageDataset(
        data_dir='./data',
        split='train',
        transforms=get_train_transforms(),
        debug=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"图像形状: {sample['image'].shape}")
    print(f"掩码形状: {sample['mask'].shape}")
    print(f"图像ID: {sample['image_id']}")
    
    # 测试数据加载器
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    
    print(f"批次图像形状: {batch['image'].shape}")
    print(f"批次掩码形状: {batch['mask'].shape}")
    
    # 可视化
    VisualizationUtils.visualize_batch(batch, save_path='test_batch.png')
    print("✅ 数据集测试完成，可视化结果保存为 test_batch.png")


if __name__ == "__main__":
    test_dataset()
