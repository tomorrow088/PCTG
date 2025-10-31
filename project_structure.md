# PCTG - 物理约束纹理生成器 完整项目结构

## 📁 项目目录树

```
PCTG/
├── README.md                      # 项目主文档
├── requirements.txt               # Python依赖
├── setup.py                       # 安装脚本
├── .gitignore                     # Git忽略文件
│
├── config/                        # 配置文件目录
│   ├── __init__.py
│   ├── model_config.py           # 模型配置
│   ├── training_config.py        # 训练配置
│   └── dataset_config.py         # 数据集配置
│
├── models/                        # 模型定义目录
│   ├── __init__.py
│   ├── pctg_generator.py         # PCTG生成器核心
│   ├── sinet_detector.py         # SINet检测器封装
│   ├── clip_wrapper.py           # CLIP模型封装
│   ├── losses.py                 # 损失函数模块
│   └── dual_adversarial_loss.py  # 双重对抗损失
│
├── utils/                         # 工具函数目录
│   ├── __init__.py
│   ├── visualization.py          # 可视化工具
│   ├── metrics.py                # 评估指标
│   ├── image_processing.py       # 图像处理
│   └── logger.py                 # 日志系统
│
├── data/                          # 数据处理目录
│   ├── __init__.py
│   ├── dataset.py                # 数据集加载
│   ├── transforms.py             # 数据增强
│   └── dataloader.py             # 数据加载器
│
├── training/                      # 训练相关目录
│   ├── __init__.py
│   ├── trainer.py                # 训练器核心
│   ├── validator.py              # 验证器
│   └── optimizer.py              # 优化器配置
│
├── inference/                     # 推理相关目录
│   ├── __init__.py
│   ├── predictor.py              # 预测器
│   └── demo.py                   # 演示脚本
│
├── scripts/                       # 脚本目录
│   ├── train.py                  # 主训练脚本
│   ├── test.py                   # 测试脚本
│   ├── setup_models.sh           # 模型设置脚本
│   ├── verify_models.py          # 模型验证脚本
│   └── download_models.py        # 模型下载脚本
│
├── third_party/                   # 第三方代码
│   └── SINet/                    # SINet源码
│
├── checkpoints/                   # 模型权重目录
│   ├── sinet/                    # SINet预训练权重
│   │   └── SINet_COD10K.pth
│   └── clip/                     # CLIP预训练权重
│       ├── vit-l-14.pt
│       └── ViT-H-14.pt
│
├── outputs/                       # 输出目录
│   ├── experiments/              # 实验结果
│   ├── visualizations/           # 可视化结果
│   └── logs/                     # 训练日志
│
├── tests/                         # 测试目录
│   ├── __init__.py
│   ├── test_models.py            # 模型测试
│   ├── test_losses.py            # 损失函数测试
│   └── test_dataset.py           # 数据集测试
│
└── docs/                          # 文档目录
    ├── INSTALLATION.md           # 安装指南
    ├── TRAINING.md               # 训练指南
    ├── MODEL_REQUIREMENTS.md     # 模型需求
    ├── CLIP_PROMPTS_GUIDE.md     # CLIP提示词指南
    └── API.md                    # API文档
```

## 🔑 核心文件说明

### 配置文件
- **model_config.py**: 模型架构配置（PCTG、SINet、CLIP）
- **training_config.py**: 训练超参数配置
- **dataset_config.py**: 数据集路径和处理配置

### 模型文件
- **pctg_generator.py**: 物理约束纹理生成器（50M参数）
- **sinet_detector.py**: SINet显著性检测器封装
- **clip_wrapper.py**: CLIP语义检测器封装
- **losses.py**: 综合损失函数（对抗损失、感知损失、物理约束）
- **dual_adversarial_loss.py**: 人眼+AI双重对抗损失

### 训练脚本
- **trainer.py**: 完整训练流程（训练、验证、保存）
- **train.py**: 主训练入口脚本

### 数据处理
- **dataset.py**: 自定义数据集类
- **transforms.py**: 数据增强策略
- **dataloader.py**: 数据加载器配置

### 工具函数
- **visualization.py**: 训练过程可视化
- **metrics.py**: 攻击成功率等评估指标
- **image_processing.py**: 图像预处理工具

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置模型
bash scripts/setup_models.sh

# 3. 验证模型
python scripts/verify_models.py

# 4. 开始训练
python scripts/train.py

# 5. 运行测试
python scripts/test.py
```

## 📊 文件功能矩阵

| 功能模块 | 核心文件 | 作用 |
|---------|---------|------|
| **模型架构** | `models/pctg_generator.py` | 生成迷彩纹理 |
| **检测器** | `models/sinet_detector.py` | 显著性检测 |
| **语义理解** | `models/clip_wrapper.py` | 文本-图像匹配 |
| **损失计算** | `models/losses.py` | 多模态损失 |
| **训练流程** | `training/trainer.py` | 端到端训练 |
| **数据加载** | `data/dataset.py` | 图像+掩码加载 |
| **可视化** | `utils/visualization.py` | 结果展示 |
| **评估** | `utils/metrics.py` | 性能指标 |

## 🎯 关键依赖关系

```
train.py
  ├── trainer.py
  │   ├── pctg_generator.py
  │   ├── sinet_detector.py
  │   ├── clip_wrapper.py
  │   ├── losses.py
  │   └── dual_adversarial_loss.py
  ├── dataset.py
  │   └── transforms.py
  └── visualization.py
```

## 📝 配置文件关系

```
model_config.py       → 定义模型参数
training_config.py    → 定义训练参数
dataset_config.py     → 定义数据参数
    ↓
train.py             → 读取所有配置
    ↓
trainer.py           → 执行训练流程
```

## 🔧 扩展性设计

- **新检测器**: 在 `models/` 添加新的检测器类
- **新损失函数**: 在 `models/losses.py` 添加新的损失项
- **新数据集**: 在 `data/dataset.py` 添加新的数据集类
- **新指标**: 在 `utils/metrics.py` 添加新的评估指标

## 📖 文档索引

- [安装指南](docs/INSTALLATION.md)
- [训练教程](docs/TRAINING.md)
- [模型需求](docs/MODEL_REQUIREMENTS.md)
- [CLIP提示词](docs/CLIP_PROMPTS_GUIDE.md)
- [API文档](docs/API.md)
