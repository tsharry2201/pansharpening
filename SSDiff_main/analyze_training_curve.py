#!/usr/bin/env python3
"""
分析SSDiff蒸馏训练曲线
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def load_tensorboard_logs(log_dir):
    """加载tensorboard日志"""
    # 查找events文件
    events_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not events_files:
        print(f"❌ 未找到events文件在: {log_dir}")
        return None
    
    events_file = os.path.join(log_dir, events_files[0])
    print(f"📂 读取日志文件: {events_file}")
    
    # 加载events
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()
    
    # 获取所有标量
    tags = ea.Tags()['scalars']
    print(f"📊 找到的指标: {tags}")
    
    return ea, tags

def extract_scalar_data(ea, tag):
    """提取标量数据"""
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except:
        return [], []

def moving_average(data, window_size=100):
    """计算移动平均"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def analyze_loss_trend(steps, values, window_size=100):
    """分析损失趋势"""
    if len(values) < window_size:
        return {
            'current': values[-1] if values else 0,
            'mean': np.mean(values) if values else 0,
            'std': np.std(values) if values else 0,
            'trend': 'insufficient_data'
        }
    
    # 最近N步的统计
    recent_values = values[-window_size:]
    earlier_values = values[-2*window_size:-window_size] if len(values) >= 2*window_size else values[:window_size]
    
    recent_mean = np.mean(recent_values)
    earlier_mean = np.mean(earlier_values)
    
    # 判断趋势
    if recent_mean < earlier_mean * 0.95:
        trend = '下降 ↓'
    elif recent_mean > earlier_mean * 1.05:
        trend = '上升 ↑'
    else:
        trend = '平稳 →'
    
    # 计算波动性
    recent_std = np.std(recent_values)
    volatility = recent_std / recent_mean if recent_mean > 0 else 0
    
    return {
        'current': values[-1],
        'recent_mean': recent_mean,
        'earlier_mean': earlier_mean,
        'std': recent_std,
        'volatility': volatility,
        'trend': trend,
        'improvement': (earlier_mean - recent_mean) / earlier_mean * 100 if earlier_mean > 0 else 0
    }

def main():
    log_dir = "/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/experiments/ssdiff_distill_20251017_232238/logs/ssdiff_distill"
    output_dir = "/data2/user/zelilin/ARConv_SSDiff/SSDiff_main/experiments/ssdiff_distill_20251017_232238/eval"
    
    # 加载日志
    result = load_tensorboard_logs(log_dir)
    if result is None:
        return
    
    ea, tags = result
    
    print("\n" + "="*80)
    print("📈 训练曲线分析报告")
    print("="*80)
    
    # 分析关键指标
    metrics_to_analyze = [
        'loss_total', 'loss_l2', 'loss_vsd_teacher', 'loss_vsd_reg', 'loss_diff'
    ]
    
    data_dict = {}
    analysis_results = {}
    
    for metric in metrics_to_analyze:
        for tag in tags:
            if metric in tag:
                steps, values = extract_scalar_data(ea, tag)
                if steps:
                    data_dict[metric] = (steps, values)
                    analysis = analyze_loss_trend(steps, values)
                    analysis_results[metric] = analysis
                    
                    print(f"\n📊 {metric.upper()}")
                    print(f"   当前值: {analysis['current']:.6f}")
                    print(f"   最近100步均值: {analysis['recent_mean']:.6f}")
                    print(f"   前100步均值: {analysis['earlier_mean']:.6f}")
                    print(f"   标准差: {analysis['std']:.6f}")
                    print(f"   波动率: {analysis['volatility']*100:.2f}%")
                    print(f"   趋势: {analysis['trend']}")
                    print(f"   改善: {analysis['improvement']:.2f}%")
                break
    
    if not data_dict:
        print("❌ 未找到损失数据")
        return
    
    # 绘制训练曲线
    print("\n🎨 正在生成训练曲线图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('SSDiff Distillation Training Curves', fontsize=16, fontweight='bold')
    
    plot_configs = [
        ('loss_total', 'Total Loss', axes[0, 0], 'red'),
        ('loss_l2', 'L1 Loss', axes[0, 1], 'blue'),
        ('loss_vsd_teacher', 'VSD Teacher Loss', axes[1, 0], 'green'),
        ('loss_vsd_reg', 'VSD Reg Loss', axes[1, 1], 'orange'),
        ('loss_diff', 'Diff Loss', axes[2, 0], 'purple'),
    ]
    
    for metric, title, ax, color in plot_configs:
        if metric in data_dict:
            steps, values = data_dict[metric]
            
            # 原始曲线（半透明）
            ax.plot(steps, values, alpha=0.3, color=color, linewidth=0.5, label='Raw')
            
            # 移动平均（粗线）
            if len(values) > 100:
                ma_values = moving_average(values, window_size=100)
                ma_steps = steps[len(steps)-len(ma_values):]
                ax.plot(ma_steps, ma_values, color=color, linewidth=2, label='MA-100')
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 标注当前值
            if analysis_results.get(metric):
                analysis = analysis_results[metric]
                textstr = f"Current: {analysis['current']:.4f}\nTrend: {analysis['trend']}"
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 删除空白子图
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'training_curves_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线图已保存: {output_path}")
    
    # 生成综合分析
    print("\n" + "="*80)
    print("🔍 综合诊断")
    print("="*80)
    
    # 检查总损失
    if 'loss_total' in analysis_results:
        total_analysis = analysis_results['loss_total']
        print(f"\n1️⃣ 总损失状态:")
        print(f"   当前值: {total_analysis['current']:.4f}")
        print(f"   趋势: {total_analysis['trend']}")
        
        if total_analysis['volatility'] > 0.1:
            print(f"   ⚠️  波动率较高 ({total_analysis['volatility']*100:.1f}%)，训练不稳定")
        
        if '平稳' in total_analysis['trend']:
            print(f"   ⚠️  损失已趋于平稳，可能需要调整学习率或损失权重")
    
    # 检查各个损失分量
    print(f"\n2️⃣ 损失分量分析:")
    
    if 'loss_vsd_teacher' in analysis_results:
        vsd_teacher = analysis_results['loss_vsd_teacher']
        print(f"\n   VSD Teacher Loss (占主导):")
        print(f"   - 当前值: {vsd_teacher['current']:.4f}")
        print(f"   - 波动率: {vsd_teacher['volatility']*100:.1f}%")
        if vsd_teacher['volatility'] > 0.15:
            print(f"   - ⚠️  VSD Teacher Loss波动很大，建议降低其权重")
    
    if 'loss_l2' in analysis_results:
        l2 = analysis_results['loss_l2']
        print(f"\n   L1 Loss:")
        print(f"   - 当前值: {l2['current']:.4f}")
        if l2['current'] < 0.01:
            print(f"   - ✅ L1损失已经很小，像素级重建效果好")
    
    # 给出建议
    print(f"\n3️⃣ 优化建议:")
    suggestions = []
    
    # 检查学习率
    suggestions.append("   📌 当前使用固定学习率(constant scheduler)，建议改用cosine衰减")
    
    # 检查VSD损失
    if 'loss_vsd_teacher' in analysis_results:
        vsd = analysis_results['loss_vsd_teacher']
        if vsd['volatility'] > 0.15:
            suggestions.append("   📌 VSD Teacher Loss波动大，建议将lambda_vsd从1.0降到0.5")
    
    # 检查收敛
    if 'loss_total' in analysis_results:
        total = analysis_results['loss_total']
        if '平稳' in total['trend'] and total['volatility'] < 0.05:
            suggestions.append("   📌 训练已基本收敛，可以尝试降低学习率继续fine-tune")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()

