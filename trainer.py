"""
训练器模块
Adversarial Camouflage Training with SINet

核心功能:
1. 端到端对抗训练流程
2. 多检测器联合优化  
3. 物理约束下的纹理生成
4. 可视化与评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# 导入项目模块
from pctg_generator import PCTGGenerator
from sinet_detector import SINetDetector, load_pretrained_sinet
from losses import CompositeLoss
from model_config import TrainingConfig, get_default_config


class AdversarialTrainer:
    """
    对抗性迷彩训练器
    
    核心训练流程:
    1. 生成迷彩纹理
    2. 多检测器评估  
    3. 反向传播优化
    4. 物理约束验证
    """
    
    def __init__(self, config=None, device='cuda', distributed=False):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            device: 训练设备
            distributed: 是否分布式训练
        """
        self.device = device
        self.distributed = distributed
        
        # 配置
        if config is None:
            config = get_default_config()
        self.config = config
        self.training_config = config['training']
        
        # 创建输出目录
        self.output_dir = Path("./outputs")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.vis_dir = self.output_dir / "visualizations"
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.vis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self._initialize_models()
        
        # 初始化优化器
        self._initialize_optimizers()
        
        # 初始化损失函数
        self._initialize_losses()
        
        # 初始化记录器
        self._initialize_logging()
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_stats = []
        
        print(f"✅ 训练器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   分布式: {self.distributed}")
        print(f"   输出目录: {self.output_dir}")
    
    def _initialize_models(self):
        """初始化模型"""
        print("📦 初始化模型...")
        
        # 1. PCTG生成器
        self.generator = PCTGGenerator(self.config['pctg']).to(self.device)
        
        # 2. SINet检测器
        self.sinet_detector = load_pretrained_sinet(
            self.config['sinet'].pretrained_path,
            self.device
        )
        self.sinet_detector.eval()  # 检测器保持评估模式
        
        # 3. CLIP检测器 (可选)
        self.clip_detector = None
        if 'clip' in self.config['multi_detector'].detector_types:
            self.clip_detector = self._load_clip_detector()
        
        # 分布式训练设置
        if self.distributed:
            self.generator = DDP(self.generator, device_ids=[self.device])
        
        # 计算参数量
        generator_params = sum(p.numel() for p in self.generator.parameters())
        print(f"   生成器参数量: {generator_params / 1e6:.1f}M")
    
    def _load_clip_detector(self):
        """加载CLIP检测器"""
        try:
            import clip
            model, preprocess = clip.load(
                self.config['clip'].model_name, 
                device=self.device
            )
            model.eval()
            
            # 包装CLIP检测器
            clip_detector = CLIPDetectorWrapper(
                model, preprocess, self.config['clip']
            )
            
            print(f"   ✅ CLIP检测器加载成功: {self.config['clip'].model_name}")
            return clip_detector
            
        except ImportError:
            print("   ⚠️ CLIP未安装，跳过CLIP检测器")
            return None
    
    def _initialize_optimizers(self):
        """初始化优化器"""
        print("⚙️ 初始化优化器...")
        
        # 生成器优化器
        if self.training_config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.generator.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                betas=(self.training_config.beta1, self.training_config.beta2)
            )
        elif self.training_config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.generator.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                betas=(self.training_config.beta1, self.training_config.beta2)
            )
        else:
            raise ValueError(f"不支持的优化器: {self.training_config.optimizer}")
        
        # 学习率调度器
        self._initialize_scheduler()
        
        # 混合精度训练
        if self.training_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        print(f"   优化器: {self.training_config.optimizer}")
        print(f"   学习率: {self.training_config.learning_rate}")
        print(f"   混合精度: {self.training_config.mixed_precision}")
    
    def _initialize_scheduler(self):
        """初始化学习率调度器"""
        if self.training_config.lr_scheduler.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.epochs,
                eta_min=self.training_config.learning_rate * self.training_config.min_lr_ratio
            )
        elif self.training_config.lr_scheduler.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.epochs // 3,
                gamma=0.5
            )
        else:
            self.scheduler = None
    
    def _initialize_losses(self):
        """初始化损失函数"""
        print("🎯 初始化损失函数...")
        
        self.criterion = CompositeLoss(self.config['training']).to(self.device)
        
        print(f"   损失组件数: {len(self.criterion.weights)}")
    
    def _initialize_logging(self):
        """初始化日志记录"""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.step_count = 0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        开始训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print(f"🚀 开始训练 - 总轮数: {self.training_config.epochs}")
        
        # 训练循环
        for epoch in range(self.current_epoch, self.training_config.epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader)
            
            # 验证 (如果需要)
            val_metrics = {}
            if val_loader is not None and epoch % self.training_config.val_interval == 0:
                val_metrics = self._validate_epoch(val_loader)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录日志
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # 保存检查点
            if epoch % self.training_config.save_interval == 0:
                self._save_checkpoint(train_metrics, val_metrics)
            
            # 早停检查
            if self._should_early_stop(val_metrics):
                print(f"⏹️ 早停: epoch {epoch}")
                break
        
        print("✅ 训练完成!")
        self._finalize_training()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.generator.train()
        
        epoch_losses = {}
        epoch_metrics = {}
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {self.current_epoch+1}/{self.training_config.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 训练一个batch
            batch_losses, batch_metrics = self._train_step(batch)
            
            # 累积损失和指标
            for key, value in batch_losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)
            
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{batch_losses.get('total_loss', 0):.4f}",
                'sinet_succ': f"{batch_metrics.get('sinet_attack_success_rate', 0):.3f}"
            })
            
            # 记录步骤级别的指标
            if batch_idx % 50 == 0:  # 每50步记录一次
                self._log_step_metrics(batch_losses, batch_metrics)
        
        # 计算epoch平均值
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """训练一步"""
        # 数据移到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # 梯度清零
        self.optimizer.zero_grad()
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # 1. 生成迷彩纹理
            generator_output = self.generator(
                batch['image'], 
                batch['mask'],
                adversarial_gradients=None  # 第一次前向传播
            )
            
            # 2. 检测器评估
            detectors_output = self._evaluate_detectors(
                generator_output['blended_result']
            )
            
            # 3. 计算损失
            predictions = {
                'generated_image': generator_output['blended_result'],
                'generated_texture': generator_output['generated_texture'],
                'detectors_output': detectors_output,
                'constraint_outputs': {
                    'constraint_loss': generator_output['constraint_loss']
                }
            }
            
            targets = {
                'original_image': batch['image'],
                'mask': batch['mask']
            }
            
            losses = self.criterion(predictions, targets, self.generator)
            total_loss = losses['total_loss']
        
        # 反向传播
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # 计算评估指标
        metrics = self.criterion.compute_metrics(predictions, targets)
        
        # 转换为标量
        scalar_losses = {k: v.item() if isinstance(v, torch.Tensor) else v 
                        for k, v in losses.items()}
        
        return scalar_losses, metrics
    
    def _evaluate_detectors(self, generated_images: torch.Tensor) -> Dict[str, Any]:
        """评估检测器"""
        detectors_output = {}
        
        # SINet检测
        with torch.no_grad():
            sinet_saliency = self.sinet_detector(generated_images)
            detectors_output['sinet'] = sinet_saliency
        
        # CLIP检测 (如果可用)
        if self.clip_detector is not None:
            with torch.no_grad():
                clip_output = self.clip_detector(generated_images)
                detectors_output['clip'] = clip_output
        
        return detectors_output
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.generator.eval()
        
        val_losses = {}
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # 数据移到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                generator_output = self.generator(batch['image'], batch['mask'])
                detectors_output = self._evaluate_detectors(generator_output['blended_result'])
                
                # 计算损失和指标
                predictions = {
                    'generated_image': generator_output['blended_result'],
                    'generated_texture': generator_output['generated_texture'],
                    'detectors_output': detectors_output,
                    'constraint_outputs': {
                        'constraint_loss': generator_output['constraint_loss']
                    }
                }
                
                targets = {
                    'original_image': batch['image'],
                    'mask': batch['mask']
                }
                
                batch_losses = self.criterion(predictions, targets)
                batch_metrics = self.criterion.compute_metrics(predictions, targets)
                
                # 累积
                for key, value in batch_losses.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item() if isinstance(value, torch.Tensor) else value)
                
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)
        
        # 计算平均值
        avg_losses = {f"val_{k}": np.mean(v) for k, v in val_losses.items()}
        avg_metrics = {f"val_{k}": np.mean(v) for k, v in val_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    def _log_step_metrics(self, losses: Dict[str, float], metrics: Dict[str, float]):
        """记录步骤级别指标"""
        self.step_count += 1
        
        # 记录损失
        for key, value in losses.items():
            self.writer.add_scalar(f"step_loss/{key}", value, self.step_count)
        
        # 记录指标
        for key, value in metrics.items():
            self.writer.add_scalar(f"step_metrics/{key}", value, self.step_count)
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]):
        """记录epoch级别指标"""
        # 训练指标
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"epoch_train/{key}", value, self.current_epoch)
        
        # 验证指标
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"epoch_val/{key}", value, self.current_epoch)
        
        # 学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("training/learning_rate", current_lr, self.current_epoch)
        
        # 打印日志
        print(f"Epoch {self.current_epoch+1}: "
              f"Loss={train_metrics.get('total_loss', 0):.4f}, "
              f"SINet_Success={train_metrics.get('sinet_attack_success_rate', 0):.3f}")
        
        if val_metrics:
            print(f"           Val_Loss={val_metrics.get('val_total_loss', 0):.4f}, "
                  f"Val_SINet_Success={val_metrics.get('val_sinet_attack_success_rate', 0):.3f}")
    
    def _save_checkpoint(self, train_metrics: Dict[str, float], 
                        val_metrics: Dict[str, float]):
        """保存检查点"""
        # 当前检查点
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存当前checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        current_metric = val_metrics.get('val_sinet_attack_success_rate', 
                                       train_metrics.get('sinet_attack_success_rate', 0))
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: 攻击成功率 {current_metric:.3f}")
    
    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        if not self.training_config.early_stopping:
            return False
        
        # 这里可以实现更复杂的早停逻辑
        # 暂时返回False
        return False
    
    def _finalize_training(self):
        """完成训练的清理工作"""
        self.writer.close()
        print(f"📊 最佳攻击成功率: {self.best_metric:.3f}")
        print(f"💾 模型保存在: {self.checkpoint_dir}")


class CLIPDetectorWrapper(nn.Module):
    """CLIP检测器包装器"""
    
    def __init__(self, clip_model, preprocess, config):
        super().__init__()
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.config = config
        
        # 预处理文本提示
        positive_text = clip.tokenize(config.positive_prompts).to(clip_model.device)
        negative_text = clip.tokenize(config.negative_prompts).to(clip_model.device)
        
        with torch.no_grad():
            self.positive_features = clip_model.encode_text(positive_text)
            self.negative_features = clip_model.encode_text(negative_text)
            
            # 归一化
            self.positive_features = self.positive_features / self.positive_features.norm(dim=-1, keepdim=True)
            self.negative_features = self.negative_features / self.negative_features.norm(dim=-1, keepdim=True)
    
    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] 输入图像 (-1 to 1 or 0 to 1)
            
        Returns:
            output: dict with similarities
        """
        # 转换图像格式
        if images.min() < 0:
            images = (images + 1) / 2  # [-1,1] -> [0,1]
        
        # CLIP图像编码
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            positive_sim = (image_features @ self.positive_features.T).mean(dim=-1)
            negative_sim = (image_features @ self.negative_features.T).mean(dim=-1)
        
        return {
            'positive_similarity': positive_sim,
            'negative_similarity': negative_sim
        }


def create_trainer_from_config(config_path: str = None) -> AdversarialTrainer:
    """从配置文件创建训练器"""
    if config_path is not None:
        # 从文件加载配置
        # 这里可以添加配置文件加载逻辑
        pass
    
    # 使用默认配置
    config = get_default_config()
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = AdversarialTrainer(config, device)
    
    return trainer


if __name__ == "__main__":
    # 测试训练器
    print("🧪 测试训练器...")
    
    trainer = create_trainer_from_config()
    
    print("✅ 训练器测试完成")
    print(f"   生成器参数: {sum(p.numel() for p in trainer.generator.parameters()) / 1e6:.1f}M")
    print(f"   设备: {trainer.device}")
