#!/bin/bash

# ============================================================
# 模型一键设置脚本
# Adversarial Camouflage Project - Model Setup Script
# ============================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 分隔线
print_separator() {
    echo "============================================================"
}

# ============================================================
# 主脚本开始
# ============================================================

clear
print_separator
echo "🚀 对抗性迷彩生成项目 - 模型自动设置"
print_separator
echo ""

# 1. 创建必要的目录结构
print_info "创建目录结构..."
mkdir -p checkpoints/{sinet,clip,efficientnet,vgg,resnet}
mkdir -p data/{images,masks,annotations}/{train,val}
mkdir -p outputs/{checkpoints,logs,visualizations}
print_success "目录创建完成"
echo ""

# 2. 检查并移动用户已有的模型
print_separator
print_info "检查用户已有的模型..."
print_separator
echo ""

# 检查SINet模型
SINET_FILES=(
    "SINet_COD10K.pth"
    "SINet_CAMO.pth"
    "SINet_CHAMELEON.pth"
)

found_sinet=false
for file in "${SINET_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "找到SINet模型: $file"
        mv "$file" checkpoints/sinet/
        print_success "已移动到 checkpoints/sinet/"
        found_sinet=true
        break
    fi
done

if [ "$found_sinet" = false ]; then
    print_warning "未找到SINet模型文件"
    print_info "请手动下载："
    print_info "  1. 访问: https://github.com/DengPingFan/SINet"
    print_info "  2. 下载预训练模型（如 SINet_COD10K.pth）"
    print_info "  3. 放到: checkpoints/sinet/"
    echo ""
fi

# 检查CLIP模型
CLIP_FILES=(
    "vit-l-14.pt"
    "ViT-L-14.pt"
    "vit_l_14.pt"
)

found_clip_l=false
for file in "${CLIP_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "找到CLIP ViT-L/14: $file"
        mv "$file" checkpoints/clip/vit-l-14.pt
        print_success "已移动到 checkpoints/clip/"
        found_clip_l=true
        break
    fi
done

if [ "$found_clip_l" = false ]; then
    print_warning "未找到CLIP ViT-L/14模型"
    print_info "请确保您有 vit-l-14.pt 文件"
fi

# CLIP ViT-H/14 (可选)
CLIP_H_FILES=(
    "laion-CLIP-ViT-H-14-laion2B-s32B-b79K"
    "ViT-H-14.pt"
    "vit-h-14.pt"
)

found_clip_h=false
for file in "${CLIP_H_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "找到CLIP ViT-H/14: $file (可选)"
        mv "$file" checkpoints/clip/
        print_success "已移动到 checkpoints/clip/"
        found_clip_h=true
        break
    fi
done

echo ""

# 3. 下载PyTorch模型
print_separator
print_info "准备下载PyTorch预训练模型..."
print_separator
echo ""

print_info "需要下载以下模型:"
echo "  1. EfficientNet-B3 (~50MB)"
echo "  2. VGG19 (~550MB)"
echo "  3. ResNet50 (~100MB)"
echo ""
echo "总下载大小: ~700MB"
echo ""

read -p "是否立即下载? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "开始下载模型..."
    
    # 使用Python下载模型
    python3 << 'EOF'
import sys
import torch
import torchvision.models as models
from tqdm import tqdm

print("\n📥 下载 EfficientNet-B3...")
try:
    import timm
    model = timm.create_model('efficientnet_b3', pretrained=True)
    del model
    print("✅ EfficientNet-B3 下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("请确保已安装 timm: pip install timm")

print("\n📥 下载 VGG19...")
try:
    vgg = models.vgg19(pretrained=True)
    del vgg
    print("✅ VGG19 下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")

print("\n📥 下载 ResNet50...")
try:
    resnet = models.resnet50(pretrained=True)
    del resnet
    print("✅ ResNet50 下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")

print("\n🎉 PyTorch模型下载完成！")
print(f"缓存位置: ~/.cache/torch/hub/checkpoints/")
EOF

    print_success "PyTorch模型下载完成"
else
    print_warning "跳过模型下载"
    print_info "稍后可以运行: python verify_models.py"
fi

echo ""

# 4. 验证安装
print_separator
print_info "验证模型安装..."
print_separator
echo ""

if command -v python3 &> /dev/null; then
    python3 verify_models.py --no-download
else
    print_error "未找到Python3"
fi

echo ""

# 5. 生成配置文件
print_separator
print_info "生成配置文件..."
print_separator
echo ""

# 创建配置文件
cat > config/paths_config.yaml << 'EOF'
# 路径配置文件
# Paths Configuration

# 模型路径
models:
  sinet:
    pretrained: "checkpoints/sinet/SINet_COD10K.pth"
  
  clip:
    vit_l_14: "checkpoints/clip/vit-l-14.pt"
    vit_h_14: "checkpoints/clip/laion-CLIP-ViT-H-14-laion2B-s32B-b79K"
  
  pctg:
    encoder: "efficientnet_b3"  # 自动从缓存加载

# 数据路径
data:
  root: "data"
  train_images: "data/images/train"
  train_masks: "data/masks/train"
  val_images: "data/images/val"
  val_masks: "data/masks/val"

# 输出路径
outputs:
  checkpoints: "outputs/checkpoints"
  logs: "outputs/logs"
  visualizations: "outputs/visualizations"
EOF

print_success "配置文件已生成: config/paths_config.yaml"

echo ""

# 6. 创建快速启动脚本
print_separator
print_info "创建快速启动脚本..."
print_separator
echo ""

cat > quick_start.sh << 'EOF'
#!/bin/bash

echo "🚀 对抗性迷彩生成 - 快速启动"
echo ""

# 检查模型
echo "🔍 检查模型状态..."
python3 verify_models.py --no-download --test-only

echo ""
echo "选择运行模式:"
echo "  1) 调试模式 (快速验证)"
echo "  2) 正常训练"
echo "  3) 论文实验模式 (完整训练)"
echo "  4) 退出"
echo ""

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "🐛 启动调试模式..."
        python3 train.py --debug
        ;;
    2)
        echo "🏃 启动正常训练..."
        python3 train.py
        ;;
    3)
        echo "📝 启动论文实验模式..."
        python3 train.py --paper
        ;;
    4)
        echo "👋 退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
EOF

chmod +x quick_start.sh
print_success "快速启动脚本已创建: ./quick_start.sh"

echo ""

# 7. 显示目录结构
print_separator
print_info "当前目录结构:"
print_separator
echo ""

if command -v tree &> /dev/null; then
    tree -L 2 checkpoints/ config/ data/ outputs/ 2>/dev/null || true
else
    ls -lR checkpoints/ config/ data/ outputs/ 2>/dev/null || true
fi

echo ""

# 8. 最终总结
print_separator
print_success "设置完成！"
print_separator
echo ""

echo "📋 下一步操作:"
echo ""
echo "1️⃣  验证模型:"
echo "   python verify_models.py"
echo ""
echo "2️⃣  快速启动:"
echo "   ./quick_start.sh"
echo ""
echo "3️⃣  或者直接训练:"
echo "   python train.py --debug          # 调试模式"
echo "   python train.py                  # 正常训练"
echo "   python train.py --paper          # 论文实验"
echo ""

if [ "$found_sinet" = false ]; then
    print_warning "提醒: 请确保下载SINet预训练模型"
    echo "   下载地址: https://github.com/DengPingFan/SINet"
    echo ""
fi

if [ "$found_clip_l" = false ]; then
    print_warning "提醒: 请确保有CLIP ViT-L/14模型"
    echo "   文件名: vit-l-14.pt"
    echo ""
fi

print_separator
print_success "🎉 设置脚本运行完成！"
print_separator
