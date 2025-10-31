"""
主训练脚本
Adversarial Camouflage Generation with SINet

使用方法:
    python train.py --config config/default.yaml
    python train.py --resume checkpoints/latest.pth
    python train.py --debug  # 调试模式
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import yaml
import random
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目模块
from trainer import AdversarialTrainer, create_trainer_from_config
from model_config import get_default_config, get_debug_config, get_paper_config
from data.dataset import AdversarialCamouflageDataset
from data.transforms import get_train_transforms, get_val_transforms


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Adversarial Camouflage Training')
    
    # 基础参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据集根目录')
    
    # 训练模式
    parser.add_argument('--debug', action='store_true',
                       help='调试模式 (小数据集)')
    parser.add_argument('--paper', action='store_true',
                       help='论文实验模式 (完整训练)')
    
    # 设备设置
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto/cuda/cpu)')
    parser.add_argument('--num_gpus', type=int, default=-1,
                       help='GPU数量 (-1=自动检测)')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true',
                       help='分布式训练')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='本地rank (分布式训练)')
    
    # 数据加载
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='固定内存')
    
    # 实验设置
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def setup_distributed_training(args):
    """设置分布式训练"""
    if args.distributed:
        # 初始化分布式环境
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        
        print(f"🌐 分布式训练初始化完成")
        print(f"   Rank: {dist.get_rank()}/{dist.get_world_size()}")
        print(f"   Local Rank: {args.local_rank}")


def setup_device(args):
    """设置训练设备"""
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            if args.num_gpus == -1:
                args.num_gpus = torch.cuda.device_count()
        else:
            device = 'cpu'
            args.num_gpus = 0
    else:
        device = args.device
    
    if args.distributed:
        device = f'cuda:{args.local_rank}'
    
    print(f"💻 设备设置:")
    print(f"   主设备: {device}")
    print(f"   GPU数量: {args.num_gpus}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    return device


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 随机种子设置: {seed}")


def load_config(args):
    """加载配置"""
    if args.config is not None:
        # 从YAML文件加载配置
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📄 从文件加载配置: {args.config}")
    
    elif args.debug:
        config = get_debug_config()
        print("🐛 使用调试配置")
    
    elif args.paper:
        config = get_paper_config()
        print("📝 使用论文实验配置")
    
    else:
        config = get_default_config()
        print("⚙️ 使用默认配置")
    
    # 从命令行参数覆盖配置
    if args.experiment_name is not None:
        config['experiment_name'] = args.experiment_name
    
    return config


def create_datasets(args, config):
    """创建数据集"""
    print("📊 创建数据集...")
    
    # 训练变换
    train_transforms = get_train_transforms(
        image_size=config['training'].get('image_size', 256),
        augmentation=True
    )
    
    # 验证变换
    val_transforms = get_val_transforms(
        image_size=config['training'].get('image_size', 256)
    )
    
    # 训练数据集
    train_dataset = AdversarialCamouflageDataset(
        data_dir=args.data_dir,
        split='train',
        transforms=train_transforms,
        debug=args.debug
    )
    
    # 验证数据集
    val_dataset = AdversarialCamouflageDataset(
        data_dir=args.data_dir,
        split='val',
        transforms=val_transforms,
        debug=args.debug
    )
    
    print(f"   训练样本数: {len(train_dataset)}")
    print(f"   验证样本数: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config, args):
    """创建数据加载器"""
    print("🔄 创建数据加载器...")
    
    # 分布式采样器
    train_sampler = None
    val_sampler = None
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )
    
    # 验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    ) if len(val_dataset) > 0 else None
    
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   验证批次数: {len(val_loader) if val_loader else 0}")
    
    return train_loader, val_loader


def create_trainer(config, device, args):
    """创建训练器"""
    print("🏗️ 创建训练器...")
    
    trainer = AdversarialTrainer(
        config=config,
        device=device,
        distributed=args.distributed
    )
    
    return trainer


def resume_training(trainer, resume_path):
    """恢复训练"""
    print(f"🔄 恢复训练: {resume_path}")
    
    checkpoint = torch.load(resume_path, map_location=trainer.device)
    
    # 加载模型状态
    trainer.generator.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 恢复训练状态
    trainer.current_epoch = checkpoint['epoch'] + 1
    trainer.best_metric = checkpoint.get('best_metric', 0.0)
    
    print(f"   恢复到epoch {trainer.current_epoch}")
    print(f"   最佳指标: {trainer.best_metric:.3f}")


def print_training_info(config, train_loader, val_loader):
    """打印训练信息"""
    print("\n" + "="*60)
    print("🚀 训练配置总览")
    print("="*60)
    
    print(f"📋 模型配置:")
    print(f"   生成器: PCTG ({config['pctg'].encoder_name})")
    print(f"   检测器: SINet + CLIP")
    print(f"   物理约束: {config['pctg'].printable_colors_only}")
    
    print(f"\n🎯 训练参数:")
    print(f"   总轮数: {config['training'].epochs}")
    print(f"   批次大小: {config['training'].batch_size}")
    print(f"   学习率: {config['training'].learning_rate}")
    print(f"   优化器: {config['training'].optimizer}")
    
    print(f"\n📊 数据信息:")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader) if val_loader else 0}")
    
    print(f"\n💾 损失权重:")
    print(f"   对抗损失: {config['training'].adversarial_weight}")
    print(f"   内容损失: {config['training'].content_weight}")
    print(f"   感知损失: {config['training'].perceptual_weight}")
    print(f"   纹理损失: {config['training'].texture_weight}")
    print(f"   物理约束: {config['training'].physical_constraint_weight}")
    
    print("="*60 + "\n")


def main():
    """主函数"""
    print("🎭 Adversarial Camouflage Training with SINet")
    print("="*60)
    
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置分布式训练
    setup_distributed_training(args)
    
    # 设置设备
    device = setup_device(args)
    
    # 加载配置
    config = load_config(args)
    
    # 创建数据集
    train_dataset, val_dataset = create_datasets(args, config)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, config, args
    )
    
    # 创建训练器
    trainer = create_trainer(config, device, args)
    
    # 恢复训练 (如果需要)
    if args.resume is not None:
        resume_training(trainer, args.resume)
    
    # 打印训练信息
    print_training_info(config, train_loader, val_loader)
    
    try:
        # 开始训练
        trainer.train(train_loader, val_loader)
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        # 保存当前状态
        checkpoint_path = trainer.checkpoint_dir / "interrupted_checkpoint.pth"
        trainer._save_checkpoint({}, {})
        print(f"💾 已保存中断检查点: {checkpoint_path}")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        if args.distributed:
            dist.destroy_process_group()
        
        print("\n🏁 训练脚本结束")


if __name__ == "__main__":
    main()
