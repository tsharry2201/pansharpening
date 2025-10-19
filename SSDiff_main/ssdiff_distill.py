"""
SSDiff一步蒸馏模型
模仿OSEDiff的训练范式，将多步SSDiff蒸馏为单步模型
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
import copy

# 添加SSDiff路径
sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')
from utils.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults


def initialize_ssdiff_unet_with_lora(args, pretrained_path=None):
    """
    初始化SSDiff的UNet并添加LoRA层
    
    Args:
        args: 配置参数
        pretrained_path: 预训练模型路径
    
    Returns:
        unet: 带LoRA的UNet模型
        lora_target_modules: LoRA目标模块列表
    """
    # 创建模型
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # 加载预训练权重
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"Loading pretrained SSDiff from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Pretrained SSDiff loaded successfully!")
    
    # 冻结所有参数
    model.requires_grad_(False)
    model.train()
    
    # 找到所有可以添加LoRA的层
    lora_target_modules = []
    for name, module in model.named_modules():
        # 为Conv2d和Linear层添加LoRA
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 跳过输入输出层和一些特殊层
            if any(skip in name for skip in ['time_embed', 'label_emb']):
                continue
            lora_target_modules.append(name)
    
    # 只保留部分关键层（避免过度参数化）
    # 优先选择attention和残差块中的层
    filtered_modules = []
    for name in lora_target_modules:
        if any(key in name for key in ['attn', 'in_layers', 'out_layers', 'skip_connection']):
            filtered_modules.append(name)
    
    if len(filtered_modules) == 0:
        filtered_modules = lora_target_modules  # 如果过滤后为空，使用全部
    
    print(f"Found {len(filtered_modules)} modules for LoRA")
    print(f"Sample modules: {filtered_modules[:5]}")
    
    # 手动为Conv2d层添加LoRA参数
    # 使用简单的LoRA实现，避免PEFT库的复杂性
    class LoRALayer(nn.Module):
        def __init__(self, base_layer, rank=4, alpha=8):
            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # 为Conv2d创建LoRA参数
            if isinstance(base_layer, nn.Conv2d):
                self.lora_down = nn.Conv2d(
                    base_layer.in_channels, 
                    rank, 
                    kernel_size=base_layer.kernel_size,
                    stride=base_layer.stride,
                    padding=base_layer.padding,
                    bias=False
                )
                self.lora_up = nn.Conv2d(
                    rank, 
                    base_layer.out_channels, 
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                )
                # 初始化
                nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
                nn.init.zeros_(self.lora_up.weight)
        
        def forward(self, x):
            base_out = self.base_layer(x)
            if hasattr(self, 'lora_down'):
                # 确保 LoRA 层使用与输入相同的 dtype
                x_dtype = x.dtype
                lora_down_out = self.lora_down(x.to(self.lora_down.weight.dtype))
                lora_out = self.lora_up(lora_down_out) * self.scaling
                # 转换回原始 dtype
                lora_out = lora_out.to(x_dtype)
                return base_out + lora_out
            return base_out
    
    # 为选定的模块添加LoRA
    lora_count = 0
    for name, module in model.named_modules():
        if name in filtered_modules and isinstance(module, nn.Conv2d):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            
            # 创建LoRA包装层
            lora_layer = LoRALayer(module, rank=args.lora_rank, alpha=args.lora_rank * 2)
            setattr(parent, child_name, lora_layer)
            lora_count += 1
    
    print(f"✅ Added LoRA to {lora_count} Conv2d layers")
    return model, filtered_modules


class SSDiff_gen(nn.Module):
    """
    单步生成器模型
    类似于OSEDiff_gen，但针对SSDiff的全景锐化任务
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 🔥 OSEDiff风格修改：添加Teacher模型（冻结）
        # Teacher：原始预训练的SSDiff（不添加LoRA）
        print("📚 Loading Teacher model (frozen)...")
        # 使用create_model_and_diffusion创建teacher（与student一致的结构）
        self.unet_teacher, _ = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        # 加载teacher权重
        teacher_state = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
        self.unet_teacher.load_state_dict(teacher_state, strict=True)
        # 冻结teacher
        self.unet_teacher.eval()
        for p in self.unet_teacher.parameters():
            p.requires_grad = False
        print("✅ Teacher model loaded and frozen")
        
        # Student：添加LoRA的可训练模型
        print("🎓 Loading Student model (with LoRA)...")
        self.unet, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args, 
            pretrained_path=args.pretrained_ssdiff_path
        )
        print("✅ Student model loaded with LoRA")
        
        # 创建diffusion（用于单步去噪，使用顶部已导入的函数）
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 打印predict_xstart设置（训练时）
        print(f"🔍 [SSDiff_gen 训练] predict_xstart = {args.predict_xstart}")
        
        # 🔥 OSEDiff风格修改：不再固定timestep，训练时动态生成
        # 推理时使用t=999，训练时使用随机timestep [100, 999)
        self.inference_timestep = 999
        
        self.lora_rank = args.lora_rank
        self.training = True  # 添加training标志
    
    def set_train(self):
        """设置训练模式，只有LoRA层可训练"""
        self.training = True  # 设置训练标志
        self.unet.train()
        for n, p in self.unet.named_parameters():
            # 只有lora_down和lora_up参数可训练，base_layer参数冻结
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def forward(self, lms, pan, ms, gt=None):
        """
        前向传播（残差学习版本，与原SSDiff对齐）
        
        Args:
            lms: 低分辨率多光谱图像 [B, 8, H, W]
            pan: 全色图像 [B, 1, H, W]
            ms: 上采样的多光谱图像 [B, 8, H, W]
            gt: Ground truth (训练时使用) [B, 8, H, W]
        
        Returns:
            output: 预测的高分辨率多光谱图像 [B, 8, H, W]
            residual_pred: UNet预测的残差
        """
        device = lms.device
        batch_size = lms.shape[0]
        
        # 🔥 训练时使用随机timestep
        if self.training:
            # 训练：随机timestep [100, 999)，覆盖多种噪声水平
            timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        else:
            # 推理：固定t=999单步
            timesteps = torch.full((batch_size,), self.inference_timestep, device=device, dtype=torch.long)
        
        # 🔥 核心修改：与原SSDiff完全一致，使用q_sample_xt
        if self.training and gt is not None:
            # 训练时：计算真残差，使用原SSDiff的扩散过程
            gt_residual = gt - lms  # 真残差
            
            # 使用原SSDiff的q_sample_xt进行加噪（与原SSDiff完全一致）
            noise = torch.randn_like(gt_residual)
            x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        else:
            # 推理时：输入为0（初始噪声残差的近似）
            x_t = torch.zeros_like(lms)
        
        # UNet预测残差（使用forward_impl方法）
        # forward_impl(self, lms, pan, ms, x_t, timesteps)
        # 参数说明：lms(64x64), pan(64x64), ms(16x16会被upsample), x_t(噪声残差, 64x64)
        # 注意：x_t现在是噪声残差，与原SSDiff的输入一致
        residual_pred = self.unet.forward_impl(lms, pan, ms, x_t, timesteps)
        
        # 检查 UNet 输出
        if torch.isnan(residual_pred).any():
            print(f"[SSDiff_gen] residual_pred contains NaN after forward_impl!")
            print(f"  Input ranges - lms: [{lms.min():.4f}, {lms.max():.4f}], x_t: [{x_t.min():.4f}, {x_t.max():.4f}]")
            # 将 NaN 替换为 0
            residual_pred = torch.nan_to_num(residual_pred, nan=0.0)
        
        # 单步去噪得到残差
        if self.args.predict_xstart:
            # 如果模型预测x0，直接使用（这里x0就是残差）
            residual = residual_pred
        else:
            # 从噪声预测计算x0（残差）
            residual = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
        
        # 最终输出 = LMS + 残差
        output = lms + residual
        
        # 检查最终输出
        if torch.isnan(output).any():
            print(f"[SSDiff_gen] output contains NaN!")
            output = torch.nan_to_num(output, nan=0.0)
        
        # 裁剪到有效范围
        output = output.clamp(0, 1)
        
        return output, residual_pred
    
    def distribution_matching_loss(self, lms, pan, ms, gt):
        """
        🔥 OSEDiff风格的VSD (Variational Score Distillation) 损失
        
        核心思想：Student学习Teacher在不同噪声水平下的预测分布，而不是直接学GT
        
        Args:
            lms: 低分辨率多光谱图像 [B, 8, H, W]
            pan: 全色图像 [B, 1, H, W]
            ms: 上采样的多光谱图像 [B, 8, H, W]
            gt: Ground truth [B, 8, H, W]
        
        Returns:
            loss_vsd: VSD损失标量
        """
        batch_size = lms.shape[0]
        device = lms.device
        
        # 随机timestep（与forward中训练时的范围一致）
        timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        
        # 🔥 核心修改：使用原SSDiff的q_sample_xt（完全一致）
        gt_residual = gt - lms  # 计算真残差
        noise = torch.randn_like(gt_residual)
        # 使用原SSDiff的扩散过程进行加噪
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        with torch.no_grad():
            # Teacher预测（冻结，不计算梯度）
            residual_teacher = self.unet_teacher.forward_impl(lms, pan, ms, x_t, timesteps)
            if self.args.predict_xstart:
                residual_teacher_final = residual_teacher
            else:
                residual_teacher_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_teacher)
            output_teacher = (lms + residual_teacher_final).clamp(0, 1)
        
        # Student预测（需要梯度）
        residual_student = self.unet.forward_impl(lms, pan, ms, x_t, timesteps)
        if self.args.predict_xstart:
            residual_student_final = residual_student
        else:
            residual_student_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_student)
        output_student = (lms + residual_student_final).clamp(0, 1)
        
        # VSD损失计算（模仿OSEDiff的实现）
        # 加权因子：基于GT和Teacher预测的差异
        weighting_factor = torch.abs(gt - output_teacher).mean(dim=[1, 2, 3], keepdim=True) + 1e-8
        
        # 梯度：Student和Teacher的差异，经过weighting标准化
        grad = (output_student - output_teacher) / weighting_factor
        
        # VSD损失：让Student接近Teacher的分布
        # 使用stop_gradient技巧：gt - grad作为target，但grad不传梯度
        loss_vsd = F.mse_loss(gt, (gt - grad).detach(), reduction="mean")
        
        return loss_vsd
    
    def save_model(self, save_path):
        """保存模型（只保存LoRA权重）"""
        state_dict = {
            'lora_target_modules': self.lora_target_modules,
            'lora_rank': self.lora_rank,
            'unet_state_dict': {},
        }
        
        # 只保存LoRA参数
        for name, param in self.unet.named_parameters():
            if 'lora' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
        
        torch.save(state_dict, save_path)
        print(f"💾 Model saved to: {save_path}")


class SSDiff_reg(nn.Module):
    """
    正则化模型
    类似于OSEDiff_reg，包含固定的UNet和可更新的UNet
    """
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        
        # 创建固定的UNet（作为教师模型）
        self.unet_fix, _ = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        if args.pretrained_ssdiff_path and os.path.exists(args.pretrained_ssdiff_path):
            state_dict = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
            self.unet_fix.load_state_dict(state_dict)
        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()
        
        # 创建可更新的UNet（带LoRA）
        self.unet_update, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args,
            pretrained_path=args.pretrained_ssdiff_path
        )
        
        # 创建diffusion（使用顶部已导入的函数）
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 打印predict_xstart设置（正则化训练）
        print(f"🔍 [SSDiff_reg 训练] predict_xstart = {args.predict_xstart}")
        
        # 设置权重类型
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
    
    def set_train(self):
        """设置训练模式"""
        self.unet_update.train()
        for n, p in self.unet_update.named_parameters():
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def diff_loss(self, lms, pan, ms, gt):
        """
        扩散损失：让LoRA学习残差（与原始SSDiff保持一致）
        改进方案：从MS开始预测残差（GT - LMS）
        """
        device = gt.device
        bsz = gt.shape[0]
        
        # 计算真实残差
        gt_residual = gt - lms
        
        # 固定在timestep=999（与主模型一致）
        timesteps = torch.full((bsz,), 999, device=device, dtype=torch.long)
        
        # 🔥 核心修改：使用原SSDiff的q_sample_xt（完全一致）
        noise = torch.randn_like(gt_residual)
        # 使用原SSDiff的扩散过程进行加噪
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        # 预测残差（使用forward_impl方法）
        # 参数：lms(64x64), pan(64x64), ms(16x16会被upsample), x_t(64x64)
        residual_pred = self.unet_update.forward_impl(lms, pan, ms, x_t, timesteps)
        
        # 计算损失：预测残差与真实残差的差异（使用L1，与原始SSDiff一致）
        if self.args.predict_xstart:
            # 如果预测x0，这里x0就是残差
            loss = F.l1_loss(residual_pred.float(), gt_residual.float(), reduction="mean")
        else:
            # 如果预测噪声，需要转换为x0（残差）再比较
            residual_pred_x0 = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
            loss = F.l1_loss(residual_pred_x0.float(), gt_residual.float(), reduction="mean")
        
        return loss
    
    def distribution_matching_loss(self, lms, pan, ms, x_pred):
        """
        分布匹配损失：让单步模型的输出接近多步SSDiff的分布
        
        Args:
            lms, pan, ms: 条件输入
            x_pred: 学生模型（单步）的预测
        
        Returns:
            loss: 分布匹配损失
        """
        device = x_pred.device
        bsz = x_pred.shape[0]
        
        # 检查输入数据
        if torch.isnan(lms).any():
            print(f"[distribution_matching_loss] lms contains NaN!")
        if torch.isnan(pan).any():
            print(f"[distribution_matching_loss] pan contains NaN!")
        if torch.isnan(ms).any():
            print(f"[distribution_matching_loss] ms contains NaN!")
        if torch.isnan(x_pred).any():
            print(f"[distribution_matching_loss] x_pred contains NaN!")
        
        # 随机采样中间时间步
        timesteps = torch.randint(20, 980, (bsz,), device=device).long()
        
        # 🔥 核心修改：对残差添加噪声（与原SSDiff一致）
        x_pred_residual = x_pred - lms  # 预测的残差
        noise = torch.randn_like(x_pred_residual)
        # 使用diffusion的q_sample_xt对残差添加噪声
        noisy_x = self.diffusion.q_sample_xt(x_pred_residual, timesteps, noise=noise)
        
        if torch.isnan(noisy_x).any():
            print(f"[distribution_matching_loss] noisy_x contains NaN after q_sample_xt!")
        
        with torch.no_grad():
            # 学生模型（可更新）的预测（使用forward_impl）
            # 参数：lms(64x64), pan(64x64), ms(16x16会被upsample), noisy_x(64x64)
            noise_pred_update = self.unet_update.forward_impl(lms, pan, ms, noisy_x, timesteps)
            x0_pred_update = self.diffusion._predict_xstart_from_eps(
                noisy_x, timesteps, noise_pred_update
            )
            
            # 教师模型（固定）的预测（使用forward_impl）
            # 参数：lms(64x64), pan(64x64), ms(16x16会被upsample), noisy_x(64x64)
            noise_pred_fix = self.unet_fix.forward_impl(
                lms.to(self.weight_dtype), 
                pan.to(self.weight_dtype),
                ms.to(self.weight_dtype),
                noisy_x.to(self.weight_dtype), 
                timesteps
            )
            x0_pred_fix = self.diffusion._predict_xstart_from_eps(
                noisy_x, timesteps, noise_pred_fix.float()
            )
        
        # 计算权重因子
        weighting_factor = torch.abs(x_pred - x0_pred_fix).mean(
            dim=[1, 2, 3], keepdim=True
        ) + 1e-5
        
        # 计算梯度
        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        
        # VSD损失
        loss = F.mse_loss(x_pred, (x_pred - grad).detach())
        
        return loss


class SSDiff_test(nn.Module):
    """
    测试/推理模型
    支持两种模式：
    1. 原始SSDiff多步采样（use_distillation=False）
    2. 蒸馏后的单步采样（use_distillation=True）
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 对于蒸馏模式，不使用SpacedDiffusion，直接使用完整的1000步空间
        if args.use_distillation:
            # 临时移除 timestep_respacing，使用完整的 Gaussian Diffusion
            original_respacing = args.timestep_respacing
            args.timestep_respacing = ""  # 空字符串表示不使用spacing
            print(f"📊 Using full 1000-step space for distillation (训练时使用timestep=999)")
        
        # 创建模型和diffusion
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 恢复原始设置
        if args.use_distillation:
            args.timestep_respacing = original_respacing
            
        # 打印predict_xstart设置（测试时）
        print(f"🔍 [SSDiff_test 测试] predict_xstart = {args.predict_xstart}")
        
        # 加载预训练权重
        if hasattr(args, 'model_path') and args.model_path:
            print(f"Loading model from: {args.model_path}")
            state_dict = torch.load(args.model_path, map_location='cpu')
            
            if args.use_distillation and 'unet_state_dict' in state_dict:
                # 蒸馏模型：先加载基础权重，再加载LoRA
                if hasattr(args, 'pretrained_ssdiff_path'):
                    base_state = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
                    self.model.load_state_dict(base_state)
                    print("✅ Loaded base SSDiff weights")
                
                # 添加LoRA层
                if 'lora_target_modules' in state_dict:
                    # 从checkpoint中读取lora_rank
                    if 'lora_rank' in state_dict:
                        args.lora_rank = state_dict['lora_rank']
                        print(f"📊 Using lora_rank={args.lora_rank} from checkpoint")
                    else:
                        # 默认值
                        args.lora_rank = 4
                        print(f"⚠️  lora_rank not found in checkpoint, using default: {args.lora_rank}")
                    
                    # 添加LoRA层并加载权重（在基础SSDiff权重之上）
                    self.model, _ = initialize_ssdiff_unet_with_lora(
                        args, pretrained_path=args.pretrained_ssdiff_path
                    )
                    
                    # 加载LoRA权重
                    for name, param in self.model.named_parameters():
                        if 'lora' in name and name in state_dict['unet_state_dict']:
                            param.data.copy_(state_dict['unet_state_dict'][name])
                    print("✅ Loaded LoRA weights")
            else:
                # 原始SSDiff模型
                self.model.load_state_dict(state_dict)
                print("✅ Loaded original SSDiff weights")
        
        # 🔥 简化模型加载，避免复杂的权重类型设置
        self.model.to(self.device)
        
        # 设置 epoch > 1000，确保 ARConv 使用训练好的固定卷积核（reserved_NXY）
        self.model.set_epoch(10001)
        print("✅ Set epoch to 10001 for using fixed ARConv kernel size")
        
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, lms, pan, ms):
        """
        推理前向传播
        
        Args:
            lms: 低分辨率多光谱 [B, 8, H, W]
            pan: 全色图像 [B, 1, H, W]  
            ms: 上采样多光谱 [B, 8, H, W]
        
        Returns:
            output: 锐化后的多光谱图像 [B, 8, H, W]
        """
        # 🔥 确保推理模式
        self.model.training = False
        self.model.eval()
        
        # 数据类型转换（去掉混合精度）
        lms = lms.to(self.device)
        pan = pan.to(self.device)
        ms = ms.to(self.device)
        
        model_kwargs = {"lms": lms, "pan": pan, "ms": ms}
        
        if self.args.use_distillation:
            # 单步蒸馏模式 - 直接调用forward_impl，不使用采样过程
            # 与训练时的逻辑完全一致：直接前向传播，无需ddim_sample
            
            # 使用forward_chop处理大图像
            # 需要定义一个简单的包装函数，不使用采样
            
            # 用于记录第一个patch的调试信息
            first_patch = [True]
            
            def direct_forward_fn(model_output, t, lms_input, **kwargs):
                # 训练时直接使用forward_impl的输出作为残差
                # model_output是forward_impl的输出（残差预测）
                # lms_input是切分后的patch（64x64）
                
                # 打印调试信息（仅第一次）
               
                # 如果predict_xstart=True，model_output就是预测的x0（残差）
                # 否则需要从eps转换为x0
                if self.args.predict_xstart:
                    pred_xstart = model_output
                else:
                    # 从kwargs稳健获取当前patch对应的x_t；若无则用0
                    x_t = kwargs.get('noise', None)
                    if x_t is None:
                        x_t = torch.zeros_like(lms_input)
                    
                    # 确保x_t与lms_input形状匹配
                    if x_t.shape != lms_input.shape:
                        print(f"⚠️ 形状不匹配! x_t: {x_t.shape}, lms: {lms_input.shape}")
                        # 重新创建与当前patch匹配的x_t
                        x_t = torch.zeros_like(lms_input)
                    
                    # 构造与当前patch批大小匹配的t_batch（优先使用回调传入的t）
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 0:
                            t_batch = t.to(device=lms_input.device, dtype=torch.long).expand(lms_input.shape[0])
                        else:
                            t_batch = t.to(device=lms_input.device, dtype=torch.long)
                    else:
                        t_batch = torch.full((lms_input.shape[0],), 999, device=lms_input.device, dtype=torch.long)
                    pred_xstart = self.diffusion._predict_xstart_from_eps(x_t, t_batch, model_output)
                
                # 最终输出 = lms_patch + 残差
                output = lms_input + pred_xstart
                
                # 打印统计信息
                print(f"   LMS patch: [{lms_input.min():.4f}, {lms_input.max():.4f}], mean={lms_input.mean():.4f}")
                print(f"   Model output (residual): [{model_output.min():.4f}, {model_output.max():.4f}], mean={model_output.mean():.4f}")
                print(f"   Predicted residual: [{pred_xstart.min():.4f}, {pred_xstart.max():.4f}], mean={pred_xstart.mean():.4f}")
                print(f"   Output before clamp: [{output.min():.4f}, {output.max():.4f}], mean={output.mean():.4f}")
                
                return {"sample": output, "pred_xstart": pred_xstart}
            
            # 与训练/推理逻辑保持一致：单步蒸馏推理时使用 x_t = 0
            xt = torch.zeros_like(lms)
            
            # 再次确认测试时predict_xstart设置
            print(f"🔍 [SSDiff_test 单步蒸馏推理] predict_xstart = {self.args.predict_xstart}")
            
            # 可选：添加对比实验 - 使用带噪声的x_t（与训练更接近）
            if hasattr(self.args, 'test_with_noise') and self.args.test_with_noise:
                print("🔬 使用带噪声的x_t进行测试（更接近训练分布）")
                # 对整张图生成一致的噪声，避免patch边界问题
                noise = torch.randn_like(lms)
                # 时间步固定t=999
                t_full = torch.full((lms.shape[0],), 999, device=self.device, dtype=torch.long)
                # 对零残差加噪
                xt = self.diffusion.q_sample_xt(torch.zeros_like(lms), t_full, noise=noise)
                print(f"   带噪x_t范围: [{xt.min():.4f}, {xt.max():.4f}], 均值={xt.mean():.4f}, 标准差={xt.std():.4f}")
                        
            # 🔥 统一使用 forward_chop 处理（支持大图像patch切分）
            batch_size = lms.shape[0]
            timesteps = torch.full((batch_size,), 999, device=self.device, dtype=torch.long)
            
            print(f"使用自定义的 forward_chop_distill 进行单步蒸馏推理")
            
            # 🔥 使用我们在 SSNet.py 中新添加的 forward_chop_distill 方法
            # 这个方法直接处理输入，不需要经过复杂的 module.py 逻辑
            output = self.model.forward_chop_distill(
                lms, pan, ms, xt,
                sample_fn=direct_forward_fn,
                noise=xt,
            )
        else:
            # 原始多步采样模式
            sample_fn = (
                self.diffusion.p_sample_loop 
                if not self.args.use_ddim 
                else self.diffusion.ddim_sample_loop
            )
            
            output = sample_fn(
                self.model,
                shape=ms.shape,
                model_kwargs=model_kwargs,
                clip_denoised=self.args.clip_denoised,
                progress=False
            )
        
        # 裁剪到有效范围
        output = output.clamp(0, 1)
        
        return output