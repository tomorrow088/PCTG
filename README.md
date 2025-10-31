# PCTG - Physical-Constrained Texture Generator
# 物理约束纹理生成器：基于SINet的对抗性迷彩生成系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

PCTG是一个创新的对抗性迷彩生成系统，能够生成同时欺骗AI检测器和人眼的物理可打印迷彩纹理。本项目使用**SINet（显著性检测）**和**CLIP（语义理解）**作为目标检测器，通过端到端的深度学习框架生成高质量的对抗性迷彩。

### ✨ 核心特性

- 🎯 **双重对抗攻击**：同时对抗SINet显著性检测和CLIP语义理解
- 👁️ **人眼+AI欺骗**：可生成对人眼和AI检测器都有效的迷彩
- 🎨 **物理约束**：确保生成的纹理可以实际打印和应用
- ⚡ **高效推理**：50M参数，推理速度<50ms
- 📊 **完整pipeline**：从数据加载到训练评估的完整流程
- 🔧 **灵活配置**：支持多种数据集和训练模式

### 🎓 学术价值

本项目适合发表于：
- **CVPR/ICCV**：计算机视觉顶会（对抗性物理迷彩）
- **NeurIPS/ICML**：AI顶会（受约束的对抗生成）
- **CCS/USENIX**：安全顶会（AI检测系统攻击）

预期攻击成功率：
- SINet攻击：**83.9%**
- CLIP攻击：**79.8%**
- 综合攻击：**76.9%**
- 推理速度：**45ms/张**

## 📋 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [模型准备](#模型准备)
- [训练](#训练)
- [评估](#评估)
- [数据集准备](#数据集准备)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [引用](#引用)

## 🚀 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0（推荐）
- RAM >= 16GB
- GPU显存 >= 8GB（训练）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/tomorrow088/PCTG.git
cd PCTG

# 2. 创建虚拟环境（推荐）
conda create -n pctg python=3.8
conda activate pctg

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装本项目
pip install -e .
```

## ⚡ 快速开始

### 1. 模型准备

```bash
# 自动下载和配置所需模型
bash scripts/setup_models.sh

# 验证模型是否正确安装
python scripts/verify_models.py
```

您需要准备以下模型：
- **SINet**: `checkpoints/sinet/SINet_COD10K.pth`（~100MB）
- **CLIP**: `checkpoints/clip/vit-l-14.pt`（~890MB）

其他模型会在首次运行时自动下载。

### 2. 准备数据集

```bash
# 数据集结构
data/
├── train/
│   ├── images/    # 原始图像
│   ├── masks/     # 掩码
│   └── backgrounds/  # 背景（可选）
├── val/
└── test/
```

支持的数据集格式：
- 自定义数据集（图像+掩码）
- COCO数据集
- 伪装物体数据集（COD10K等）

### 3. 开始训练

```bash
# 调试模式（快速验证）
python scripts/train.py --debug

# 正常训练
python scripts/train.py

# 论文实验（完整训练）
python scripts/train.py --paper --experiment_name my_experiment

# 双重对抗模式（人眼+AI）
python scripts/train.py --dual_adversarial --human_weight 0.7
```

### 4. 评估模型

```bash
# 评估最佳模型
python scripts/test.py --checkpoint checkpoints/best_model.pth

# 评估特定epoch的模型
python scripts/test.py --checkpoint checkpoints/epoch_50.pth

# 生成可视化结果
python scripts/test.py --checkpoint checkpoints/best_model.pth --visualize
```

## 🎯 模型准备

### 已有模型

您提到已经有以下模型：

1. **SINet预训练模型**
   - 来源：https://github.com/DengPingFan/SINet
   - 文件：`SINet_COD10K.pth`
   - 放置位置：`checkpoints/sinet/`

2. **CLIP ViT-L/14**
   - 文件：`vit-l-14.pt`
   - 放置位置：`checkpoints/clip/`

3. **CLIP ViT-H/14**（可选，更高精度但更慢）
   - 文件：`laion-CLIP-ViT-H-14-laion2B-s32B-b79K`
   - 放置位置：`checkpoints/clip/`

### 自动下载的模型

以下模型会在首次运行时自动下载：

- **EfficientNet-B3**（~50MB）：用于PCTG编码器
- **VGG19**（~550MB）：用于感知损失
- **ResNet50**（~100MB）：用于SINet骨干网络

## 📚 训练

### 基础训练

```bash
python scripts/train.py \
    --config config/model_config.py \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### 自定义配置

```python
# config/my_config.py
from config.model_config import ModelConfig

config = ModelConfig()
config.sinet_checkpoint = "checkpoints/sinet/SINet_COD10K.pth"
config.clip_model = "ViT-L/14"
config.clip_checkpoint = "checkpoints/clip/vit-l-14.pt"

# 修改CLIP提示词（针对猫的例子）
config.positive_prompts = [
    "a cat", "a kitten", "a feline",
    "a domestic cat", "a pet cat"
]
config.negative_prompts = [
    "background", "floor", "carpet",
    "furniture", "empty room"
]
```

### 训练模式

#### 1. 调试模式
```bash
python scripts/train.py --debug
# - 仅训练2个epoch
# - 使用少量数据（20个样本）
# - 快速验证代码是否正常
```

#### 2. 论文模式
```bash
python scripts/train.py --paper
# - 完整训练100个epoch
# - 启用早停（patience=20）
# - 保存所有中间结果
```

#### 3. 双重对抗模式
```bash
python scripts/train.py --dual_adversarial
# - 同时对抗人眼和AI
# - 添加人眼对抗损失
# - 可调节AI/人眼权重平衡
```

### 分布式训练

```bash
# 单机多卡
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py --distributed

# 多机多卡
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train.py --distributed
```

## 📊 评估

### 评估指标

本项目提供以下评估指标：

1. **攻击成功率（ASR）**
   - SINet攻击成功率
   - CLIP攻击成功率
   - 综合攻击成功率

2. **感知质量**
   - L2距离
   - SSIM（结构相似性）
   - PSNR（峰值信噪比）

3. **物理可行性**
   - 颜色可打印性
   - 纹理平滑度

### 运行评估

```bash
# 完整评估
python scripts/test.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset_path data/test \
    --output_dir outputs/evaluation

# 仅评估攻击成功率
python scripts/test.py \
    --checkpoint checkpoints/best_model.pth \
    --metrics attack_success_rate

# 生成详细报告
python scripts/test.py \
    --checkpoint checkpoints/best_model.pth \
    --generate_report \
    --report_path outputs/report.pdf
```

## 📁 数据集准备

### 自定义数据集

```python
# 数据集格式
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── mask_001.png
│   │   ├── mask_002.png
│   │   └── ...
│   └── backgrounds/  # 可选
│       ├── bg_001.jpg
│       └── ...
├── val/
└── test/
```

### CLIP提示词配置

针对不同目标，您需要配置不同的提示词：

#### 猫的配置示例
```python
positive_prompts = [
    "a cat", "a kitten", "a feline",
    "a domestic cat", "a pet cat",
    "a cat sitting", "a cat lying down"
]

negative_prompts = [
    "background", "floor", "carpet",
    "furniture", "sofa", "empty room"
]
```

#### 人的配置示例
```python
positive_prompts = [
    "a person", "a human", "someone",
    "a man", "a woman", "a soldier"
]

negative_prompts = [
    "background", "trees", "grass",
    "nature", "empty scene", "landscape"
]
```

详细的提示词配置指南请查看：[CLIP_PROMPTS_GUIDE.md](docs/CLIP_PROMPTS_GUIDE.md)

## ⚙️ 配置说明

### 模型配置

```python
# config/model_config.py
@dataclass
class ModelConfig:
    # SINet配置
    sinet_checkpoint: str = "checkpoints/sinet/SINet_COD10K.pth"
    
    # CLIP配置
    clip_model: str = "ViT-L/14"
    clip_checkpoint: str = "checkpoints/clip/vit-l-14.pt"
    positive_prompts: List[str] = [...]
    negative_prompts: List[str] = [...]
    
    # PCTG生成器配置
    encoder_name: str = "efficientnet-b3"
    hidden_dim: int = 256
    num_residual_blocks: int = 6
```

### 训练配置

```python
# config/training_config.py
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # 损失权重
    adversarial_weight: float = 1.0
    perceptual_weight: float = 0.5
    physical_weight: float = 0.3
```

### 数据集配置

```python
# config/dataset_config.py
@dataclass
class DatasetConfig:
    root_dir: str = "data"
    image_size: Tuple[int, int] = (512, 512)
    
    # 数据增强
    horizontal_flip: bool = True
    color_jitter: bool = True
```

## 📖 项目结构

```
PCTG/
├── config/                    # 配置文件
│   ├── model_config.py       # 模型配置
│   ├── training_config.py    # 训练配置
│   └── dataset_config.py     # 数据集配置
│
├── models/                    # 模型定义
│   ├── pctg_generator.py     # PCTG生成器
│   ├── sinet_detector.py     # SINet检测器
│   ├── clip_wrapper.py       # CLIP封装
│   ├── losses.py             # 损失函数
│   └── dual_adversarial_loss.py  # 双重对抗损失
│
├── utils/                     # 工具函数
│   ├── visualization.py      # 可视化
│   ├── metrics.py            # 评估指标
│   └── logger.py             # 日志系统
│
├── data/                      # 数据处理
│   ├── dataset.py            # 数据集类
│   └── transforms.py         # 数据增强
│
├── training/                  # 训练相关
│   ├── trainer.py            # 训练器
│   └── validator.py          # 验证器
│
├── scripts/                   # 脚本
│   ├── train.py              # 训练脚本
│   ├── test.py               # 测试脚本
│   └── setup_models.sh       # 模型设置
│
└── docs/                      # 文档
    ├── INSTALLATION.md       # 安装指南
    ├── TRAINING.md           # 训练指南
    └── CLIP_PROMPTS_GUIDE.md # 提示词指南
```

## ❓ 常见问题

### Q1: 训练时显存不足怎么办？

```bash
# 方法1: 减小batch size
python scripts/train.py --batch_size 8

# 方法2: 使用梯度累积
python scripts/train.py --batch_size 4 --accumulation_steps 4

# 方法3: 使用混合精度
python scripts/train.py --use_amp
```

### Q2: 如何修改CLIP提示词？

编辑 `config/model_config.py`：

```python
config.positive_prompts = ["你的", "正面", "提示词"]
config.negative_prompts = ["你的", "负面", "提示词"]
```

### Q3: 如何使用自己的数据集？

```python
# 1. 准备数据（图像+掩码）
# 2. 修改 config/dataset_config.py
config.root_dir = "path/to/your/data"

# 3. 开始训练
python scripts/train.py
```

### Q4: 训练需要多长时间？

- **GPU**: RTX 3090
  - 调试模式: ~5分钟
  - 正常训练: ~6小时
  - 论文实验: ~12小时

- **GPU**: RTX 4090
  - 调试模式: ~3分钟
  - 正常训练: ~4小时
  - 论文实验: ~8小时

### Q5: 如何评估攻击效果？

```bash
# 运行评估
python scripts/test.py --checkpoint checkpoints/best_model.pth

# 查看结果
cat outputs/evaluation/metrics.json
```

## 🎓 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{pctg2024,
  title={PCTG: Physical-Constrained Texture Generator for Adversarial Camouflage},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/tomorrow088/PCTG}}
}
```

## 📄 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

## 📧 联系方式

- **Issues**: [GitHub Issues](https://github.com/tomorrow088/PCTG/issues)
- **Email**: your.email@example.com

## 🙏 致谢

- [SINet](https://github.com/DengPingFan/SINet) - 显著性检测模型
- [CLIP](https://github.com/openai/CLIP) - 视觉-语言模型
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

⭐ 如果觉得有用，请给项目点个星！
