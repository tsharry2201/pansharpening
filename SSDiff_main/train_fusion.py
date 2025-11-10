"""
融合模型训练脚本
训练OTPNet和SSDiff的后融合网络
"""
import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm

# 添加路径
sys.path.append('/data2/user/zelilin/pansharpening/SSDiff_main')
sys.path.append('/data2/user/zelilin/OTPNet')

from fusion_model import create_fusion_model
from configs.option_DPM_pansharpening import parser_args as ssdiff_parser_args
from pancollection.common.psdata import PansharpeningSession as DataSession


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练OTPNet-SSDiff融合模型')
    
    # 模型路径
    parser.add_argument("--otpnet_checkpoint", type=str, required=True,
                       help="OTPNet模型路径")
    parser.add_argument("--ssdiff_checkpoint", type=str, required=True,
                       help="SSDiff模型路径")
    parser.add_argument("--pretrained_ssdiff_path", type=str, required=True,
                       help="预训练SSDiff基础模型路径（蒸馏模式需要）")
    
    # 融合网络配置
    parser.add_argument("--fusion_channels", type=int, default=64,
                       help="融合网络隐藏通道数")
    parser.add_argument("--num_fusion_blocks", type=int, default=3,
                       help="融合块数量")
    
    # 训练参数
    parser.add_argument("--train_dataset", type=str, default='WV3/train_wv3.h5',
                       help="训练数据集")
    parser.add_argument("--val_dataset", type=str, default='WV3/valid_wv3.h5',
                       help="验证数据集")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批大小")
    parser.add_argument("--num_epochs", type=int, default=200,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--scheduler", type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help="学习率调度器")
    
    # 损失权重
    parser.add_argument("--loss_l1_weight", type=float, default=1.0,
                       help="L1损失权重")
    parser.add_argument("--loss_l2_weight", type=float, default=1.0,
                       help="L2损失权重")
    parser.add_argument("--loss_consistency_weight", type=float, default=0.1,
                       help="一致性损失权重（与OTPNet/SSDiff输出的差异）")
    
    # 设备和保存
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--save_dir", type=str, default='checkpoints/fusion',
                       help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="保存间隔（epoch）")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="日志打印间隔（步）")
    
    return parser.parse_args()


def compute_loss(pred, gt, otpnet_out, ssdiff_out, args):
    """
    计算损失函数
    
    Args:
        pred: 融合模型输出 [B, 8, H, W]
        gt: Ground truth [B, 8, H, W]
        otpnet_out: OTPNet输出 [B, 8, H, W]
        ssdiff_out: SSDiff输出 [B, 8, H, W]
        args: 参数
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    loss_dict = {}
    
    # 1. L1损失（主要损失）
    loss_l1 = F.l1_loss(pred, gt)
    loss_dict['l1'] = loss_l1.item()
    
    # 2. L2损失（MSE）
    loss_l2 = F.mse_loss(pred, gt)
    loss_dict['l2'] = loss_l2.item()
    
    # 3. 一致性损失：融合结果应该与两个基础模型的输出保持一定一致性
    # 避免过度偏离基础模型
    loss_consistency = (
        F.l1_loss(pred, otpnet_out.detach()) + 
        F.l1_loss(pred, ssdiff_out.detach())
    ) / 2.0
    loss_dict['consistency'] = loss_consistency.item()
    
    # 总损失
    total_loss = (
        args.loss_l1_weight * loss_l1 +
        args.loss_l2_weight * loss_l2 +
        args.loss_consistency_weight * loss_consistency
    )
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


def train_epoch(model, dataloader, optimizer, args, epoch):
    """训练一个epoch"""
    model.train()
    # 注意：只训练融合网络部分
    for name, param in model.named_parameters():
        if 'fusion_net' in name or 'output_conv' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    total_loss = 0
    loss_accum = {
        'total': 0, 'l1': 0, 'l2': 0, 'consistency': 0
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")
    for step, batch in enumerate(pbar):
        # 获取数据
        pan = batch['pan'].to(args.device)  # [B, 1, H, W]
        lms = batch['lms'].to(args.device)  # [B, 8, H, W]
        ms = batch['ms'].to(args.device)    # [B, 8, H/4, W/4]
        gt = batch['gt'].to(args.device)    # [B, H, W, 8]
        
        # 调整GT维度：[B, H, W, 8] -> [B, 8, H, W]
        if gt.dim() == 4 and gt.shape[-1] == 8:
            gt = gt.permute(0, 3, 1, 2)
        
        # 前向传播
        pred, otpnet_out, ssdiff_out = model(pan, lms, ms)
        
        # 计算损失
        loss, loss_dict = compute_loss(pred, gt, otpnet_out, ssdiff_out, args)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        for key in loss_accum:
            loss_accum[key] += loss_dict[key]
        
        # 实时更新进度条 - 显示当前batch的loss
        current_avg_loss = total_loss / (step + 1)
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg': f"{current_avg_loss:.4f}",
            'l1': f"{loss_dict['l1']:.4f}",
            'l2': f"{loss_dict['l2']:.4f}"
        })
    
    # 返回平均损失
    num_batches = len(dataloader)
    avg_losses = {key: val / num_batches for key, val in loss_accum.items()}
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, args):
    """验证"""
    model.eval()
    
    total_loss = 0
    loss_accum = {
        'total': 0, 'l1': 0, 'l2': 0, 'consistency': 0
    }
    
    for batch in tqdm(dataloader, desc="Validating"):
        # 获取数据
        pan = batch['pan'].to(args.device)
        lms = batch['lms'].to(args.device)
        ms = batch['ms'].to(args.device)
        gt = batch['gt'].to(args.device)
        
        # 调整GT维度
        if gt.dim() == 4 and gt.shape[-1] == 8:
            gt = gt.permute(0, 3, 1, 2)
        
        # 前向传播
        pred, otpnet_out, ssdiff_out = model(pan, lms, ms)
        
        # 计算损失
        loss, loss_dict = compute_loss(pred, gt, otpnet_out, ssdiff_out, args)
        
        # 累积损失
        for key in loss_accum:
            loss_accum[key] += loss_dict[key]
    
    # 返回平均损失
    num_batches = len(dataloader)
    avg_losses = {key: val / num_batches for key, val in loss_accum.items()}
    return avg_losses


def main():
    """主函数"""
    args = parse_args()
    
    # 生成时间戳并创建带时间戳的保存目录
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    timestamped_save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(timestamped_save_dir, exist_ok=True)
    
    # 更新保存路径
    args.save_dir = timestamped_save_dir
    
    # 设置设备
    torch.cuda.set_device(args.device)
    print(f"使用设备: {args.device}")
    print(f"保存目录: {timestamped_save_dir}")
    
    print("\n" + "="*60)
    print("OTPNet-SSDiff融合模型训练")
    print("="*60)
    
    # 创建融合模型
    print("\n创建融合模型...")
    
    # 获取SSDiff配置
    ssdiff_args = ssdiff_parser_args()
    ssdiff_args.model_path = args.ssdiff_checkpoint
    ssdiff_args.pretrained_ssdiff_path = args.pretrained_ssdiff_path
    ssdiff_args.use_distillation = True
    ssdiff_args.device = args.device
    
    # 确保使用正确的数据集配置（不使用配置文件中的默认值）
    if hasattr(ssdiff_args, 'dataset'):
        # 提取文件名（去掉.h5后缀）
        train_name = args.train_dataset.replace('.h5', '') if args.train_dataset.endswith('.h5') else args.train_dataset
        val_name = args.val_dataset.replace('.h5', '') if args.val_dataset.endswith('.h5') else args.val_dataset
        ssdiff_args.dataset['train'] = train_name
        ssdiff_args.dataset['valid'] = val_name
        print(f"   使用数据集: train={train_name}, valid={val_name}")
    
    from fusion_model import OTPNet_SSDiff_Fusion
    model = OTPNet_SSDiff_Fusion(
        otpnet_checkpoint=args.otpnet_checkpoint,
        ssdiff_checkpoint=args.ssdiff_checkpoint,
        ssdiff_args=ssdiff_args,
        fusion_channels=args.fusion_channels,
        num_fusion_blocks=args.num_fusion_blocks,
        device=args.device
    )
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    session = DataSession(ssdiff_args)
    
    # 使用正确的DataSession API
    train_data, _, _ = session.get_dataloader(args.train_dataset, False, None)
    val_data, _, _ = session.get_dataloader(args.val_dataset, False, None)
    
    print(f"训练集批次数: {len(train_data)}")
    print(f"验证集批次数: {len(val_data)}")
    
    # 创建优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    print(f"\n优化器: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    if scheduler:
        print(f"调度器: {args.scheduler}")
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 60)
        
        # 训练
        train_losses = train_epoch(model, train_data, optimizer, args, epoch)
        
        # 验证
        val_losses = validate(model, val_data, args)
        
        # 打印损失
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} 结果:")
        print(f"{'='*60}")
        print(f"训练 - Total: {train_losses['total']:.6f}, L1: {train_losses['l1']:.6f}, L2: {train_losses['l2']:.6f}, Cons: {train_losses['consistency']:.6f}")
        print(f"验证 - Total: {val_losses['total']:.6f}, L1: {val_losses['l1']:.6f}, L2: {val_losses['l2']:.6f}, Cons: {val_losses['consistency']:.6f}")
        
        # 学习率调度
        if scheduler:
            scheduler.step()
            print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")
        
        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_path = os.path.join(args.save_dir, 'fusion_best.pth')
            model.save_fusion_weights(save_path)
            print(f"保存最佳模型 (val_loss={best_val_loss:.4f})")
        
        # 定期保存
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'fusion_epoch{epoch}.pth')
            model.save_fusion_weights(save_path)
            print(f"检查点: epoch {epoch}")
    
    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'fusion_final.pth')
    model.save_fusion_weights(final_path)
    
    print("\n" + "="*60)
    print("训练完成")
    print(f"最佳模型: {os.path.join(args.save_dir, 'fusion_best.pth')}")
    print(f"最终模型: {final_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

