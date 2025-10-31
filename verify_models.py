"""
模型验证和自动下载脚本
Model Verification and Auto-Download Script

功能:
1. 检查所有必需模型是否存在
2. 自动下载缺失的模型
3. 验证模型可以正常加载
4. 生成模型配置报告
"""

import torch
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm


class ModelVerifier:
    """模型验证器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
        
        # 创建目录
        for dir_path in [
            self.checkpoint_dir / "sinet",
            self.checkpoint_dir / "clip",
            self.checkpoint_dir / "efficientnet",
            self.checkpoint_dir / "vgg",
            self.checkpoint_dir / "resnet"
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 模型清单
        self.models_info = {
            'project_models': {
                'SINet': {
                    'path': 'checkpoints/sinet/SINet_COD10K.pth',
                    'size_mb': 100,
                    'required': True,
                    'auto_download': False,
                    'note': '需要从GitHub手动下载'
                },
                'CLIP-ViT-L-14': {
                    'path': 'checkpoints/clip/vit-l-14.pt',
                    'size_mb': 890,
                    'required': True,
                    'auto_download': False,
                    'note': '您已有此文件'
                },
                'CLIP-ViT-H-14': {
                    'path': 'checkpoints/clip/laion-CLIP-ViT-H-14-laion2B-s32B-b79K',
                    'size_mb': 2500,
                    'required': False,
                    'auto_download': False,
                    'note': '可选，您已有此文件'
                }
            },
            'pytorch_models': {
                'EfficientNet-B3': {
                    'cache_name': 'efficientnet_b3_ra2-cf984f9c.pth',
                    'size_mb': 50,
                    'required': True,
                    'auto_download': True
                },
                'VGG19': {
                    'cache_name': 'vgg19-dcbb9e9d.pth',
                    'size_mb': 550,
                    'required': True,
                    'auto_download': True
                },
                'ResNet50': {
                    'cache_name': 'resnet50-0676ba61.pth',
                    'size_mb': 100,
                    'required': True,
                    'auto_download': True
                }
            }
        }
    
    def verify_project_models(self) -> Dict[str, bool]:
        """验证项目模型"""
        print("\n" + "="*60)
        print("📁 检查项目模型")
        print("="*60)
        
        results = {}
        for name, info in self.models_info['project_models'].items():
            path = self.project_root / info['path']
            exists = path.exists()
            
            # 图标
            icon = "✅" if exists else ("⚠️" if not info['required'] else "❌")
            required_text = "必需" if info['required'] else "可选"
            
            print(f"\n{icon} {name} ({required_text})")
            print(f"   路径: {info['path']}")
            
            if exists:
                size_mb = path.stat().st_size / (1024**2)
                print(f"   大小: {size_mb:.1f} MB")
                print(f"   状态: 已找到")
            else:
                print(f"   预期大小: ~{info['size_mb']} MB")
                print(f"   状态: 未找到")
                if info['note']:
                    print(f"   备注: {info['note']}")
            
            results[name] = exists
        
        return results
    
    def verify_pytorch_models(self) -> Dict[str, bool]:
        """验证PyTorch模型缓存"""
        print("\n" + "="*60)
        print("📦 检查PyTorch模型缓存")
        print("="*60)
        
        results = {}
        for name, info in self.models_info['pytorch_models'].items():
            cache_path = self.cache_dir / info['cache_name']
            exists = cache_path.exists()
            
            icon = "✅" if exists else "⬇️"
            
            print(f"\n{icon} {name} ({'必需' if info['required'] else '可选'})")
            print(f"   缓存文件: {info['cache_name']}")
            
            if exists:
                size_mb = cache_path.stat().st_size / (1024**2)
                print(f"   大小: {size_mb:.1f} MB")
                print(f"   状态: 已缓存")
            else:
                print(f"   预期大小: ~{info['size_mb']} MB")
                print(f"   状态: 未缓存 (首次运行时自动下载)")
            
            results[name] = exists
        
        return results
    
    def auto_download_pytorch_models(self):
        """自动下载PyTorch模型"""
        print("\n" + "="*60)
        print("⬇️ 自动下载PyTorch模型")
        print("="*60)
        
        # EfficientNet-B3
        print("\n📥 下载 EfficientNet-B3...")
        try:
            import timm
            model = timm.create_model('efficientnet_b3', pretrained=True)
            del model
            print("✅ EfficientNet-B3 下载完成")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
        
        # VGG19
        print("\n📥 下载 VGG19...")
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True)
            del vgg
            print("✅ VGG19 下载完成")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
        
        # ResNet50
        print("\n📥 下载 ResNet50...")
        try:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            del resnet
            print("✅ ResNet50 下载完成")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
    
    def test_model_loading(self):
        """测试模型加载"""
        print("\n" + "="*60)
        print("🧪 测试模型加载")
        print("="*60)
        
        tests = []
        
        # 测试EfficientNet
        print("\n🔧 测试 EfficientNet-B3...")
        try:
            import timm
            model = timm.create_model('efficientnet_b3', pretrained=True)
            # 测试前向传播
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            del model, dummy_input, output
            print("✅ EfficientNet-B3 加载和推理成功")
            tests.append(('EfficientNet-B3', True))
        except Exception as e:
            print(f"❌ 失败: {e}")
            tests.append(('EfficientNet-B3', False))
        
        # 测试VGG19
        print("\n🔧 测试 VGG19...")
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True)
            vgg.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = vgg(dummy_input)
            del vgg, dummy_input, output
            print("✅ VGG19 加载和推理成功")
            tests.append(('VGG19', True))
        except Exception as e:
            print(f"❌ 失败: {e}")
            tests.append(('VGG19', False))
        
        # 测试ResNet50
        print("\n🔧 测试 ResNet50...")
        try:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            resnet.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = resnet(dummy_input)
            del resnet, dummy_input, output
            print("✅ ResNet50 加载和推理成功")
            tests.append(('ResNet50', True))
        except Exception as e:
            print(f"❌ 失败: {e}")
            tests.append(('ResNet50', False))
        
        # 测试CLIP
        print("\n🔧 测试 CLIP...")
        try:
            import clip
            # 尝试加载本地模型
            clip_path = self.checkpoint_dir / "clip" / "vit-l-14.pt"
            if clip_path.exists():
                model, preprocess = clip.load(str(clip_path), device="cpu", jit=False)
                print("✅ CLIP (本地) 加载成功")
            else:
                model, preprocess = clip.load("ViT-B/32", device="cpu")
                print("✅ CLIP (在线) 加载成功")
            
            # 测试
            dummy_image = torch.randn(1, 3, 224, 224)
            dummy_text = clip.tokenize(["a cat"])
            with torch.no_grad():
                image_features = model.encode_image(dummy_image)
                text_features = model.encode_text(dummy_text)
            
            del model, preprocess, dummy_image, dummy_text, image_features, text_features
            tests.append(('CLIP', True))
        except Exception as e:
            print(f"❌ 失败: {e}")
            tests.append(('CLIP', False))
        
        # 测试SINet (如果存在)
        print("\n🔧 测试 SINet...")
        sinet_path = self.checkpoint_dir / "sinet" / "SINet_COD10K.pth"
        if sinet_path.exists():
            try:
                # 简单的权重加载测试
                checkpoint = torch.load(sinet_path, map_location='cpu')
                print(f"✅ SINet 权重文件加载成功")
                print(f"   权重键数量: {len(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
                tests.append(('SINet', True))
            except Exception as e:
                print(f"❌ 失败: {e}")
                tests.append(('SINet', False))
        else:
            print("⚠️ SINet模型文件不存在，跳过测试")
            tests.append(('SINet', None))
        
        return tests
    
    def generate_report(self, project_results, pytorch_results, test_results):
        """生成验证报告"""
        print("\n" + "="*60)
        print("📊 模型验证报告")
        print("="*60)
        
        # 统计
        project_total = len(project_results)
        project_ready = sum(1 for v in project_results.values() if v)
        project_required = sum(1 for name, info in self.models_info['project_models'].items() 
                              if info['required'])
        project_required_ready = sum(1 for name, v in project_results.items() 
                                    if v and self.models_info['project_models'][name]['required'])
        
        pytorch_total = len(pytorch_results)
        pytorch_ready = sum(1 for v in pytorch_results.values() if v)
        
        test_passed = sum(1 for _, result in test_results if result == True)
        test_total = len([t for t in test_results if t[1] is not None])
        
        print(f"\n📁 项目模型: {project_ready}/{project_total} 就绪")
        print(f"   必需模型: {project_required_ready}/{project_required} 就绪")
        
        print(f"\n📦 PyTorch模型: {pytorch_ready}/{pytorch_total} 已缓存")
        
        print(f"\n🧪 加载测试: {test_passed}/{test_total} 通过")
        
        # 状态
        all_required_ready = (project_required_ready == project_required)
        
        print("\n" + "="*60)
        if all_required_ready:
            print("✅ 所有必需模型已就绪，可以开始训练！")
            print("\n🚀 启动训练命令:")
            print("   python train.py --debug")
        else:
            print("⚠️ 部分必需模型缺失")
            print("\n📋 缺失的必需模型:")
            for name, ready in project_results.items():
                info = self.models_info['project_models'][name]
                if info['required'] and not ready:
                    print(f"   ❌ {name}")
                    print(f"      {info['note']}")
        
        print("="*60)
        
        return all_required_ready
    
    def run(self, auto_download=True):
        """运行完整验证流程"""
        print("🔍 对抗性迷彩生成项目 - 模型验证工具")
        print(f"📂 项目路径: {self.project_root}")
        
        # 1. 验证项目模型
        project_results = self.verify_project_models()
        
        # 2. 验证PyTorch模型
        pytorch_results = self.verify_pytorch_models()
        
        # 3. 自动下载缺失的PyTorch模型
        if auto_download:
            missing_pytorch = [name for name, exists in pytorch_results.items() if not exists]
            if missing_pytorch:
                print(f"\n⬇️ 检测到 {len(missing_pytorch)} 个缺失的PyTorch模型")
                response = input("是否立即下载? (y/n): ")
                if response.lower() == 'y':
                    self.auto_download_pytorch_models()
                    # 重新验证
                    pytorch_results = self.verify_pytorch_models()
        
        # 4. 测试模型加载
        test_results = self.test_model_loading()
        
        # 5. 生成报告
        all_ready = self.generate_report(project_results, pytorch_results, test_results)
        
        return all_ready


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型验证和下载工具')
    parser.add_argument('--no-download', action='store_true',
                       help='不自动下载缺失的模型')
    parser.add_argument('--test-only', action='store_true',
                       help='仅测试模型加载，不下载')
    
    args = parser.parse_args()
    
    verifier = ModelVerifier()
    
    if args.test_only:
        verifier.test_model_loading()
    else:
        all_ready = verifier.run(auto_download=not args.no_download)
        
        # 返回状态码
        sys.exit(0 if all_ready else 1)


if __name__ == "__main__":
    main()
