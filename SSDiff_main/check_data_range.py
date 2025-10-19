#!/usr/bin/env python3
"""
检查数据范围和归一化的统一性
"""
import sys
import torch
import numpy as np

sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')

from configs.option_DPM_pansharpening import parser_args as ssdiff_parser_args
from pancollection.common.psdata import PansharpeningSession as DataSession
from ssdiff_distill import SSDiff_gen

def check_dataloader_range():
    """检查数据加载器的数据范围"""
    print("="*80)
    print("🔍 检查数据加载器的数据范围")
    print("="*80)
    
    # 获取配置
    args = ssdiff_parser_args()
    
    # 创建数据加载器
    session = DataSession(args)
    train_dataloader, _, _ = session.get_dataloader(args.dataset['train'], False, None)
    
    # 从数据加载器中采样一个batch
    batch = next(iter(train_dataloader))
    
    pan = batch['pan']
    lms = batch['lms']
    ms = batch['ms']
    gt = batch['gt']
    
    # 调整GT维度（如果需要）
    if gt.shape[1] > gt.shape[2]:
        import einops
        gt = einops.rearrange(gt, 'b h w c -> b c h w')
    
    print("\n📊 数据维度:")
    print(f"   pan: {pan.shape}")
    print(f"   lms: {lms.shape}")
    print(f"   ms: {ms.shape}")
    print(f"   gt: {gt.shape}")
    
    print("\n📊 数据范围 (min, max, mean, std):")
    print(f"   pan: [{pan.min():.4f}, {pan.max():.4f}], mean={pan.mean():.4f}, std={pan.std():.4f}")
    print(f"   lms: [{lms.min():.4f}, {lms.max():.4f}], mean={lms.mean():.4f}, std={lms.std():.4f}")
    print(f"   ms: [{ms.min():.4f}, {ms.max():.4f}], mean={ms.mean():.4f}, std={ms.std():.4f}")
    print(f"   gt: [{gt.min():.4f}, {gt.max():.4f}], mean={gt.mean():.4f}, std={gt.std():.4f}")
    
    # 判断数据范围
    if pan.min() >= 0 and pan.max() <= 1.1:
        print("\n✅ 数据范围看起来是 [0, 1]")
        data_range = "0_to_1"
    elif pan.min() >= -1.1 and pan.max() <= 1.1:
        print("\n⚠️  数据范围看起来是 [-1, 1]")
        data_range = "-1_to_1"
    else:
        print(f"\n❌ 数据范围异常: [{pan.min()}, {pan.max()}]")
        data_range = "unknown"
    
    return data_range, batch


def check_model_expectation(args):
    """检查模型期望的输入范围"""
    print("\n" + "="*80)
    print("🔍 检查模型期望的输入范围")
    print("="*80)
    
    # 查看配置中的相关参数
    print(f"\n📊 模型配置:")
    print(f"   diffusion_steps: {args.diffusion_steps}")
    print(f"   noise_schedule: {args.noise_schedule}")
    print(f"   timestep_respacing: {args.timestep_respacing}")
    
    # 检查是否有rescale相关的配置
    if hasattr(args, 'rescale_timesteps'):
        print(f"   rescale_timesteps: {args.rescale_timesteps}")
    if hasattr(args, 'rescale_learned_sigmas'):
        print(f"   rescale_learned_sigmas: {args.rescale_learned_sigmas}")
    
    # 创建模型并检查
    print("\n📦 创建模型...")
    try:
        model_gen = SSDiff_gen(args)
        print("✅ 模型创建成功")
        
        # 检查模型的第一层和最后一层
        print("\n📊 模型结构检查:")
        print(f"   UNet类型: {type(model_gen.unet)}")
        
        return "model_ok"
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return "model_error"


def check_training_script():
    """检查训练脚本中的数据处理"""
    print("\n" + "="*80)
    print("🔍 检查训练脚本中的数据处理")
    print("="*80)
    
    import re
    
    # 读取训练脚本
    with open('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/train_ssdiff_distill.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 搜索归一化相关的代码
    patterns = [
        r'2\s*\*.*-\s*1',  # 2 * x - 1
        r'normalize',
        r'\[-1,\s*1\]',
        r'\[0,\s*1\]',
    ]
    
    found_patterns = {}
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            found_patterns[pattern] = matches
    
    if found_patterns:
        print("\n📝 找到的归一化相关代码:")
        for pattern, matches in found_patterns.items():
            print(f"   模式 '{pattern}': {len(matches)} 处")
    else:
        print("\n⚠️  没有找到明显的归一化代码")
    
    # 检查注释
    if '[0, 1]' in content:
        print("\n✅ 代码中提到数据范围是 [0, 1]")
        return "0_to_1"
    elif '[-1, 1]' in content:
        print("\n⚠️  代码中提到数据范围是 [-1, 1]")
        return "-1_to_1"
    else:
        print("\n❓ 代码中没有明确说明数据范围")
        return "unclear"


def check_ssdiff_model():
    """检查SSDiff模型的forward函数"""
    print("\n" + "="*80)
    print("🔍 检查SSDiff模型代码")
    print("="*80)
    
    with open('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/ssdiff_distill.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 搜索forward函数中对输入的处理
    import re
    
    # 查找是否有 2*x-1 这样的转换
    if re.search(r'2\s*\*.*-\s*1', content):
        print("⚠️  模型代码中发现 2*x-1 转换，可能期望输入 [0,1] 转为 [-1,1]")
        return "-1_to_1_expected"
    
    # 查找注释中的说明
    if '[-1, 1]' in content:
        print("⚠️  模型代码注释中提到 [-1, 1] 范围")
        return "-1_to_1_expected"
    elif '[0, 1]' in content:
        print("✅ 模型代码注释中提到 [0, 1] 范围")
        return "0_to_1_expected"
    
    print("❓ 模型代码中没有明确说明期望的输入范围")
    return "unclear"


def main():
    print("\n")
    print("🔍 " + "="*76)
    print("🔍 数据范围统一性检查工具")
    print("🔍 " + "="*76)
    print("\n")
    
    # 1. 检查数据加载器
    data_range, batch = check_dataloader_range()
    
    # 2. 检查训练脚本
    script_range = check_training_script()
    
    # 3. 检查SSDiff模型
    model_range = check_ssdiff_model()
    
    # 4. 检查模型期望
    args = ssdiff_parser_args()
    model_status = check_model_expectation(args)
    
    # 综合判断
    print("\n" + "="*80)
    print("📋 综合分析结果")
    print("="*80)
    
    print(f"\n📊 检查结果汇总:")
    print(f"   1. 数据加载器范围: {data_range}")
    print(f"   2. 训练脚本说明: {script_range}")
    print(f"   3. SSDiff模型期望: {model_range}")
    print(f"   4. 模型创建状态: {model_status}")
    
    # 给出建议
    print("\n" + "="*80)
    print("💡 结论和建议")
    print("="*80)
    
    if data_range == "0_to_1" and script_range == "0_to_1":
        print("\n✅ 数据范围统一性检查通过！")
        print("   - 数据加载器输出: [0, 1]")
        print("   - 训练脚本期望: [0, 1]")
        print("   - 模型可以直接使用数据")
    elif data_range == "-1_to_1":
        print("\n⚠️  数据范围是 [-1, 1]")
        if model_range == "0_to_1_expected":
            print("   ❌ 但模型期望 [0, 1]，需要转换!")
            print("\n   🔧 修复方案: 在模型输入前添加转换")
            print("      # 从 [-1, 1] 转换到 [0, 1]")
            print("      data = (data + 1) / 2")
        else:
            print("   ✅ 模型也期望 [-1, 1]，范围匹配")
    elif data_range == "0_to_1":
        if model_range == "-1_to_1_expected":
            print("\n⚠️  数据范围不匹配!")
            print("   - 数据加载器输出: [0, 1]")
            print("   - 模型期望: [-1, 1]")
            print("\n   🔧 修复方案: 在模型输入前添加转换")
            print("      # 从 [0, 1] 转换到 [-1, 1]")
            print("      data = data * 2 - 1")
        else:
            print("\n✅ 数据范围应该是统一的 [0, 1]")
            print("   - 训练脚本注释确认数据已归一化到 [0, 1]")
            print("   - 无需额外转换")
    
    # 实际测试模型forward
    print("\n" + "="*80)
    print("🧪 实际测试模型forward")
    print("="*80)
    
    try:
        model_gen = SSDiff_gen(args)
        model_gen.eval()
        
        # 使用真实数据测试
        with torch.no_grad():
            pan = batch['pan'][:1].cuda() if torch.cuda.is_available() else batch['pan'][:1]
            lms = batch['lms'][:1].cuda() if torch.cuda.is_available() else batch['lms'][:1]
            ms = batch['ms'][:1].cuda() if torch.cuda.is_available() else batch['ms'][:1]
            gt = batch['gt'][:1]
            
            if gt.shape[1] > gt.shape[2]:
                import einops
                gt = einops.rearrange(gt, 'b h w c -> b c h w')
            
            gt = gt.cuda() if torch.cuda.is_available() else gt
            
            print(f"\n📊 输入数据范围:")
            print(f"   pan: [{pan.min():.4f}, {pan.max():.4f}]")
            print(f"   lms: [{lms.min():.4f}, {lms.max():.4f}]")
            
            output, residual = model_gen(lms, pan, ms, gt)
            
            print(f"\n📊 输出数据范围:")
            print(f"   output: [{output.min():.4f}, {output.max():.4f}]")
            print(f"   residual: [{residual.min():.4f}, {residual.max():.4f}]")
            
            if output.min() >= -0.1 and output.max() <= 1.1:
                print("\n✅ 模型输出在合理范围 [0, 1] 内")
            elif output.min() >= -1.1 and output.max() <= 1.1:
                print("\n✅ 模型输出在合理范围 [-1, 1] 内")
            else:
                print(f"\n⚠️  模型输出范围异常: [{output.min():.4f}, {output.max():.4f}]")
                
    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ 检查完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

