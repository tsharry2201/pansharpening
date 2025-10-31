"""
SSDiff统一训练模型
结合VAE/ControlNet/CLIP/RAM模块和蒸馏学习（VSD loss）
支持：
1. 直接监督学习（L1 loss）
2. 知识蒸馏（VSD loss from teacher）
3. 混合训练（L1 + VSD）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import wandb

# 添加SSDiff路径
sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler

from ssdiff_distill import SSDiff_gen, SSDiff_reg
from configs.option_DPM_pansharpening import parser_args as ssdiff_parser_args
from pancollection.common.psdata import PansharpeningSession as DataSession


def parse_args(input_args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train SSDiff Distillation')
    
    # 基础路径
    parser.add_argument("--pretrained_ssdiff_path", type=str, required=True,
                       help="预训练SSDiff模型路径")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="experiments/ssdiff_distill",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    
    # 学习率调度
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       choices=["linear", "cosine", "cosine_with_restarts", 
                               "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    
    # 优化器参数
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # 损失权重
    parser.add_argument("--lambda_l2", type=float, default=1.0,
                       help="L2重建损失权重")
    parser.add_argument("--lambda_vsd", type=float, default=1.0,
                       help="分布匹配损失权重")
    parser.add_argument("--lambda_vsd_lora", type=float, default=1.0,
                       help="扩散损失权重")
    
    # LoRA配置
    parser.add_argument("--lora_rank", type=int, default=4)
    
    # 混合精度和加速
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    
    # 日志和报告
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="train_ssdiff_distill")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # SSDiff特定参数
    parser.add_argument("--ms_dim", type=int, default=8, help="多光谱通道数")
    parser.add_argument("--pan_dim", type=int, default=1, help="全色通道数")
    parser.add_argument("--image_size", type=int, default=64, help="Patch大小")
    
    # 恢复训练
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_step", type=int, default=0)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main(args):
    """主训练函数"""
    
    # 初始化Accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    # 获取SSDiff的配置
    ssdiff_args = ssdiff_parser_args()
    # 合并参数
    for key, value in vars(args).items():
        # 强制添加所有新参数，不管ssdiff_args是否已有
        setattr(ssdiff_args, key, value)
    
    # 创建模型
    print("Creating SSDiff distillation models...")
    model_gen = SSDiff_gen(ssdiff_args)
    model_gen.set_train()
    
    model_reg = SSDiff_reg(ssdiff_args, accelerator)
    model_reg.set_train()
    
    print("✅ Models created successfully!")
    
    # 设置优化器 - 生成器
    layers_to_opt = []
    for n, p in model_gen.unet.named_parameters():
        if ("lora_down" in n or "lora_up" in n) and p.requires_grad:
            layers_to_opt.append(p)
    
    print(f"Optimizing {len(layers_to_opt)} LoRA parameters in generator")
    
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # 设置优化器 - 正则化器
    layers_to_opt_reg = []
    for n, p in model_reg.unet_update.named_parameters():
        if ("lora_down" in n or "lora_up" in n) and p.requires_grad:
            layers_to_opt_reg.append(p)
    
    print(f"Optimizing {len(layers_to_opt_reg)} LoRA parameters in regularizer")
    
    optimizer_reg = torch.optim.AdamW(
        layers_to_opt_reg,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler_reg = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_reg,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # 创建数据加载器
    print("Creating dataloaders...")
    session = DataSession(ssdiff_args)
    # 使用正确的方法名称和参数格式
    train_dataloader, _, _ = session.get_dataloader(ssdiff_args.dataset['train'], False, None)
    
    print(f"Train batches: {len(train_dataloader)}")
    
    # Prepare everything with accelerator
    model_gen, model_reg, optimizer, optimizer_reg, train_dataloader, \
    lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, train_dataloader,
        lr_scheduler, lr_scheduler_reg
    )
    
    # 初始化trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    # 恢复checkpoint
    global_step = 0
    resume_step = 0
    
    if args.resume_from_checkpoint is not None:
        print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
        try:
            ckpt = torch.load(args.resume_from_checkpoint, map_location='cpu')
            
            unwrapped_model_gen = accelerator.unwrap_model(model_gen)
            if 'unet_state_dict' in ckpt:
                for name, param in unwrapped_model_gen.unet.named_parameters():
                    if 'lora' in name and name in ckpt['unet_state_dict']:
                        param.data.copy_(ckpt['unet_state_dict'][name])
            
            resume_step = args.resume_step
            global_step = resume_step
            print(f"✅ Resumed from step {resume_step}")
        except Exception as e:
            print(f"❌ Failed to resume: {e}")
            resume_step = 0
            global_step = 0
    
    # 训练循环
    progress_bar = tqdm(
        range(resume_step, args.max_train_steps),
        initial=resume_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model_gen, model_reg):
                # 获取数据
                # 数据说明：
                # - pan: 高分辨率全色图像 [B, 1, 64, 64]
                # - lms: 上采样后的低分辨率多光谱 [B, 8, 64, 64]  
                # - ms: 原始低分辨率多光谱 [B, 8, 16, 16]
                # - gt: 目标高分辨率多光谱 [B, H, W, 8] 或 [B, 8, 64, 64]
                pan = batch['pan']
                lms = batch['lms']
                ms = batch['ms']
                gt = batch['gt']
                
                # 调试：打印数据维度和范围（仅第一次）
                if global_step == 0:
                    print(f"Data shapes - pan: {pan.shape}, lms: {lms.shape}, ms: {ms.shape}, gt: {gt.shape}")
                    print(f"Data ranges BEFORE norm - pan: [{pan.min():.2f}, {pan.max():.2f}], lms: [{lms.min():.2f}, {lms.max():.2f}], ms: [{ms.min():.2f}, {ms.max():.2f}], gt: [{gt.min():.2f}, {gt.max():.2f}]")
                
                # 调整GT维度：检查是否需要转换
                # 数据集中的gt可能是 [B, 8, H, W] 或 [B, H, W, 8]
                # 只有当gt的第二个维度大于第一个维度时才需要转换
                if gt.shape[1] > gt.shape[2]:  # 如果是 [B, H, W, C] 格式
                    import einops
                    gt = einops.rearrange(gt, 'b h w c -> b c h w')
                    if global_step == 0:
                        print(f"GT rearranged to: {gt.shape}")
                else:
                    if global_step == 0:
                        print(f"GT already in correct format: {gt.shape}")
                
                # 注意：数据加载器已经将数据归一化到[0, 1]，不需要再次归一化！
                # 调试：打印最终数据范围（仅第一次）
                if global_step == 0:
                    print(f"Final data ranges - pan: [{pan.min():.4f}, {pan.max():.4f}], lms: [{lms.min():.4f}, {lms.max():.4f}], ms: [{ms.min():.4f}, {ms.max():.4f}], gt: [{gt.min():.4f}, {gt.max():.4f}]")
                
                # 检查是否有无效值
                if torch.isnan(lms).any() or torch.isinf(lms).any():
                    print(f"Warning: lms contains NaN or Inf, skipping batch")
                    continue
                if torch.isnan(pan).any() or torch.isinf(pan).any():
                    print(f"Warning: pan contains NaN or Inf, skipping batch")
                    continue
                if torch.isnan(ms).any() or torch.isinf(ms).any():
                    print(f"Warning: ms contains NaN or Inf, skipping batch")
                    continue
                if torch.isnan(gt).any() or torch.isinf(gt).any():
                    print(f"Warning: gt contains NaN or Inf, skipping batch")
                    continue
                
                # 前向传播（现在返回output和residual）
                try:
                    output_pred, residual_pred = model_gen(lms, pan, ms, gt)
                except RuntimeError as e:
                    print(f"Error in forward pass: {e}")
                    print(f"Shapes - lms: {lms.shape}, pan: {pan.shape}, ms: {ms.shape}, gt: {gt.shape}")
                    raise
                
                # 调试：打印输出维度（仅第一次）
                if global_step == 0:
                    print(f"Model output - output_pred: {output_pred.shape}, residual_pred: {residual_pred.shape}")
                    print(f"Before residual calc - gt: {gt.shape}, lms: {lms.shape}")
                
                # 🔥 OSEDiff风格修改：组合损失
                # 1. L1重建损失（像素级监督）
                gt_residual = gt - lms
                loss_l2 = F.l1_loss(
                    residual_pred.float(), 
                    gt_residual.float(), 
                    reduction="mean"
                ) * args.lambda_l2
                
                # 2. VSD分布匹配损失（Teacher指导，新增）
                # 处理DDP包装的情况
                if torch.cuda.device_count() > 1:
                    loss_vsd_teacher = model_gen.module.distribution_matching_loss(lms, pan, ms, gt) * args.lambda_vsd
                else:
                    loss_vsd_teacher = model_gen.distribution_matching_loss(lms, pan, ms, gt) * args.lambda_vsd
                
                # 3. 原有的分布匹配损失（使用model_reg）
                if torch.cuda.device_count() > 1:
                    loss_vsd_reg = model_reg.module.distribution_matching_loss(
                        lms, pan, ms, output_pred
                    ) * args.lambda_vsd
                else:
                    loss_vsd_reg = model_reg.distribution_matching_loss(
                        lms, pan, ms, output_pred
                    ) * args.lambda_vsd
                
                # 4. 总损失
                loss = loss_l2 + loss_vsd_teacher + loss_vsd_reg
                
                # 打印损失（每100步）
                if global_step % 100 == 0:
                    print(f"\n📊 Losses at step {global_step}:")
                    print(f"   L1 Loss: {loss_l2.item():.6f}")
                    print(f"   VSD Teacher Loss: {loss_vsd_teacher.item():.6f}")
                    print(f"   VSD Reg Loss: {loss_vsd_reg.item():.6f}")
                    print(f"   Total Loss: {loss.item():.6f}")
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                # 计算扩散损失（正则化器）
                if torch.cuda.device_count() > 1:
                    loss_diff = model_reg.module.diff_loss(
                        lms, pan, ms, gt
                    ) * args.lambda_vsd_lora
                else:
                    loss_diff = model_reg.diff_loss(
                        lms, pan, ms, gt
                    ) * args.lambda_vsd_lora
                
                accelerator.backward(loss_diff)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model_reg.parameters(), 
                        args.max_grad_norm
                    )
                
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)
            
            # 更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    # 记录日志 - 只显示最重要的指标在进度条上
                    progress_logs = {
                        "loss_total": loss.detach().item(),
                        "vsd_teacher": loss_vsd_teacher.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                    }
                    progress_bar.set_postfix(**progress_logs)
                    
                    # 完整日志用于accelerator
                    full_logs = {
                        "loss_l2": loss_l2.detach().item(),
                        "loss_vsd_teacher": loss_vsd_teacher.detach().item(),
                        "loss_vsd_reg": loss_vsd_reg.detach().item(),
                        "loss_diff": loss_diff.detach().item(),
                        "loss_total": loss.detach().item(),
                    }
                    
                    # 保存checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(
                            args.output_dir, 
                            "checkpoints", 
                            f"model_{global_step}.pkl"
                        )
                        accelerator.unwrap_model(model_gen).save_model(outf)
                        print(f"💾 Checkpoint saved at step {global_step}")
                    
                    # 记录到wandb/tensorboard
                    if global_step % 10 == 0:
                        # 获取当前实际学习率
                        current_lr = optimizer.param_groups[0]['lr']
                        current_lr_reg = optimizer_reg.param_groups[0]['lr']
                        
                        wandb_logs = {
                            "train/loss_total": loss.item(),
                            "train/loss_l2": loss_l2.item(),
                            "train/loss_vsd_teacher": loss_vsd_teacher.item(),
                            "train/loss_vsd_reg": loss_vsd_reg.item(),
                            "train/loss_diff": loss_diff.item(),
                            "train/step": global_step,
                            "train/learning_rate": current_lr,
                            "train/learning_rate_reg": current_lr_reg,
                            "train/epoch": epoch,
                            "train/lambda_l2": args.lambda_l2,
                            "train/lambda_vsd": args.lambda_vsd,
                            "train/lambda_vsd_lora": args.lambda_vsd_lora,
                        }
                        wandb.log(wandb_logs, step=global_step)
                    
                    accelerator.log(full_logs, step=global_step)
                
                # 早停
                if global_step >= args.max_train_steps:
                    print(f"\n🏁 Training completed! Reached max steps: {args.max_train_steps}")
                    return
        
        # 检查是否应该停止
        if global_step >= args.max_train_steps:
            wandb.finish()
            break
    
    print("🎉 Training finished!")


if __name__ == "__main__":
    args = parse_args()
    
    # 初始化wandb
    run_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    wandb.init(
        config=vars(args),
        project="ssdiff-distillation",
        entity="tszharry-xi-an-jiaotong-university-",  # 设置您的wandb entity
        notes=socket.gethostname(),
        name=f"distill_{args.lora_rank}",
        dir=run_dir,
        job_type="training",
        mode="offline",  # 改为"online"以同步到wandb服务器
        reinit=True
    )
    
    main(args)

