#!/usr/bin/env python3
"""
对比不同训练策略的损失分布
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_distribution_comparison():
    """可视化对比原始配置和改进配置的损失分布"""
    
    # 原始配置（当前）
    original_config = {
        'name': '原始配置\n(SSIM=0.16)',
        'lambda_l2': 1.0,
        'lambda_vsd': 1.0,
        'lambda_vsd_lora': 1.0,
        'loss_l2_value': 0.011,
        'loss_vsd_value': 0.195,
        'loss_vsd_reg': 0.0007,
        'loss_diff': 0.010,
    }
    
    # 计算原始配置的加权损失
    original_total = (
        original_config['loss_l2_value'] * original_config['lambda_l2'] +
        original_config['loss_vsd_value'] * original_config['lambda_vsd'] +
        original_config['loss_vsd_reg'] * original_config['lambda_vsd'] +
        original_config['loss_diff'] * original_config['lambda_vsd_lora']
    )
    
    original_contributions = {
        'L1 Loss': original_config['loss_l2_value'] * original_config['lambda_l2'],
        'VSD Teacher': original_config['loss_vsd_value'] * original_config['lambda_vsd'],
        'VSD Reg': original_config['loss_vsd_reg'] * original_config['lambda_vsd'],
        'Diff Loss': original_config['loss_diff'] * original_config['lambda_vsd_lora'],
    }
    
    # 方案A配置
    plan_a_config = {
        'name': '方案A\n(预期SSIM>0.4)',
        'lambda_l2': 10.0,
        'lambda_vsd': 0.05,
        'lambda_vsd_lora': 0.5,
        'loss_l2_value': 0.011,
        'loss_vsd_value': 0.195,
        'loss_vsd_reg': 0.0007,
        'loss_diff': 0.010,
    }
    
    plan_a_total = (
        plan_a_config['loss_l2_value'] * plan_a_config['lambda_l2'] +
        plan_a_config['loss_vsd_value'] * plan_a_config['lambda_vsd'] +
        plan_a_config['loss_vsd_reg'] * plan_a_config['lambda_vsd'] +
        plan_a_config['loss_diff'] * plan_a_config['lambda_vsd_lora']
    )
    
    plan_a_contributions = {
        'L1 Loss': plan_a_config['loss_l2_value'] * plan_a_config['lambda_l2'],
        'VSD Teacher': plan_a_config['loss_vsd_value'] * plan_a_config['lambda_vsd'],
        'VSD Reg': plan_a_config['loss_vsd_reg'] * plan_a_config['lambda_vsd'],
        'Diff Loss': plan_a_config['loss_diff'] * plan_a_config['lambda_vsd_lora'],
    }
    
    # 方案B配置
    plan_b_config = {
        'name': '方案B\n(保守)',
        'lambda_l2': 5.0,
        'lambda_vsd': 0.1,
        'lambda_vsd_lora': 0.8,
        'loss_l2_value': 0.011,
        'loss_vsd_value': 0.195,
        'loss_vsd_reg': 0.0007,
        'loss_diff': 0.010,
    }
    
    plan_b_total = (
        plan_b_config['loss_l2_value'] * plan_b_config['lambda_l2'] +
        plan_b_config['loss_vsd_value'] * plan_b_config['lambda_vsd'] +
        plan_b_config['loss_vsd_reg'] * plan_b_config['lambda_vsd'] +
        plan_b_config['loss_diff'] * plan_b_config['lambda_vsd_lora']
    )
    
    plan_b_contributions = {
        'L1 Loss': plan_b_config['loss_l2_value'] * plan_b_config['lambda_l2'],
        'VSD Teacher': plan_b_config['loss_vsd_value'] * plan_b_config['lambda_vsd'],
        'VSD Reg': plan_b_config['loss_vsd_reg'] * plan_b_config['lambda_vsd'],
        'Diff Loss': plan_b_config['loss_diff'] * plan_b_config['lambda_vsd_lora'],
    }
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Training Strategy Comparison', fontsize=16, fontweight='bold')
    
    configs = [
        (original_config, original_contributions, original_total),
        (plan_a_config, plan_a_contributions, plan_a_total),
        (plan_b_config, plan_b_contributions, plan_b_total),
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, (config, contributions, total) in enumerate(configs):
        # 饼图 - 损失占比
        ax1 = axes[0, idx]
        values = list(contributions.values())
        labels = list(contributions.keys())
        percentages = [v/total*100 for v in values]
        
        ax1.pie(values, labels=[f'{l}\n{p:.1f}%' for l, p in zip(labels, percentages)],
                autopct='', colors=colors, startangle=90)
        ax1.set_title(f"{config['name']}\nLoss Distribution", fontweight='bold')
        
        # 柱状图 - 权重配置
        ax2 = axes[1, idx]
        weights = {
            'λ_L1': config['lambda_l2'],
            'λ_VSD': config['lambda_vsd'],
            'λ_Lora': config['lambda_vsd_lora'],
        }
        
        bars = ax2.bar(weights.keys(), weights.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_ylabel('Weight Value')
        ax2.set_title('Loss Weights Configuration', fontweight='bold')
        ax2.set_ylim(0, 12)
        
        # 在柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 添加总损失信息
        ax2.text(0.5, 0.95, f'Total Loss: {total:.4f}',
                transform=ax2.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'experiments/ssdiff_distill_20251017_232238/eval/strategy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 策略对比图已保存: {output_path}")
    
    # 打印详细对比
    print("\n" + "="*80)
    print("📊 训练策略详细对比")
    print("="*80)
    
    print("\n【原始配置】- 当前状态")
    print(f"  权重配置: λ_L1={original_config['lambda_l2']}, λ_VSD={original_config['lambda_vsd']}, λ_Lora={original_config['lambda_vsd_lora']}")
    print(f"  损失占比:")
    for name, value in original_contributions.items():
        print(f"    {name:12s}: {value:.6f} ({value/original_total*100:.1f}%)")
    print(f"  总损失: {original_total:.6f}")
    print(f"  测试结果: SSIM=0.16, PSNR=18.23 ❌")
    
    print("\n【方案A】- 激进修正（推荐）")
    print(f"  权重配置: λ_L1={plan_a_config['lambda_l2']}, λ_VSD={plan_a_config['lambda_vsd']}, λ_Lora={plan_a_config['lambda_vsd_lora']}")
    print(f"  损失占比:")
    for name, value in plan_a_contributions.items():
        print(f"    {name:12s}: {value:.6f} ({value/plan_a_total*100:.1f}%)")
    print(f"  总损失: {plan_a_total:.6f}")
    print(f"  预期结果: SSIM>0.4, PSNR>22 ✅")
    print(f"  优势: L1损失占主导({plan_a_contributions['L1 Loss']/plan_a_total*100:.1f}%)，强调像素重建")
    
    print("\n【方案B】- 保守调整")
    print(f"  权重配置: λ_L1={plan_b_config['lambda_l2']}, λ_VSD={plan_b_config['lambda_vsd']}, λ_Lora={plan_b_config['lambda_vsd_lora']}")
    print(f"  损失占比:")
    for name, value in plan_b_contributions.items():
        print(f"    {name:12s}: {value:.6f} ({value/plan_b_total*100:.1f}%)")
    print(f"  总损失: {plan_b_total:.6f}")
    print(f"  预期结果: SSIM>0.3, PSNR>20")
    
    print("\n" + "="*80)
    print("🎯 推荐: 使用方案A，L1损失占主导地位更适合图像重建任务")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    plot_loss_distribution_comparison()

