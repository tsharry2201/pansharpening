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
        
        # 创建UNet和diffusion
        self.unet, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args, 
            pretrained_path=args.pretrained_ssdiff_path
        )
        
        # 创建diffusion（用于单步去噪，使用顶部已导入的函数）
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 固定在最后一个时间步（单步去噪）
        self.timesteps = torch.tensor([999], dtype=torch.long)
        
        self.lora_rank = args.lora_rank
    
    def set_train(self):
        """设置训练模式，只有LoRA层可训练"""
        self.unet.train()
        for n, p in self.unet.named_parameters():
            # 只有lora_down和lora_up参数可训练，base_layer参数冻结
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def forward(self, lms, pan, ms, gt=None):
        """
        前向传播（残差学习版本）
        
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
        self.timesteps = self.timesteps.to(device)
        
        # 残差学习方案：与原始SSDiff保持一致
        # 输入: 上采样的 LMS (作为基础)
        # 输出: 残差 (GT - upsampled_LMS)  
        # 最终: GT = upsampled_LMS + 残差
        #
        # 数据说明：
        # - lms: 上采样后的低分辨率MS (64x64)
        # - ms: 原始低分辨率MS (16x16)，会在forward_impl内部被upsample
        # - x_t: 高分辨率的噪声/初始图像 (64x64)
        
        # x_t从上采样的lms开始（64x64）
        x_t = lms.clone()
        
        # UNet预测残差（使用forward_impl方法）
        # forward_impl(self, lms, pan, ms, x_t, timesteps)
        # 参数说明：lms(64x64), pan(64x64), ms(16x16会被upsample), x_t(64x64)
        residual_pred = self.unet.forward_impl(lms, pan, ms, x_t, self.timesteps)
        
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
            residual = self.diffusion._predict_xstart_from_eps(x_t, self.timesteps, residual_pred)
        
        # 最终输出 = LMS + 残差
        output = lms + residual
        
        # 检查最终输出
        if torch.isnan(output).any():
            print(f"[SSDiff_gen] output contains NaN!")
            output = torch.nan_to_num(output, nan=0.0)
        
        # 裁剪到有效范围
        output = output.clamp(0, 1)
        
        return output, residual_pred
    
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
        
        # 直接从上采样的lms开始（不加噪声）
        x_t = lms.clone()
        
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
        
        # 对预测结果添加噪声
        noise = torch.randn_like(x_pred)
        noisy_x = self.diffusion.q_sample_xt(x_pred, timesteps, noise=noise)
        
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
        
        # 创建模型和diffusion
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
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
                    # TODO: 添加LoRA层并加载权重
                    self.model, _ = initialize_ssdiff_unet_with_lora(args, None)
                    
                    # 加载LoRA权重
                    for name, param in self.model.named_parameters():
                        if 'lora' in name and name in state_dict['unet_state_dict']:
                            param.data.copy_(state_dict['unet_state_dict'][name])
                    print("✅ Loaded LoRA weights")
            else:
                # 原始SSDiff模型
                self.model.load_state_dict(state_dict)
                print("✅ Loaded original SSDiff weights")
        
        # 设置权重类型
        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        
        self.model.to(self.device, dtype=self.weight_dtype)
        
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
        lms = lms.to(self.device, dtype=self.weight_dtype)
        pan = pan.to(self.device, dtype=self.weight_dtype)
        ms = ms.to(self.device, dtype=self.weight_dtype)
        
        model_kwargs = {"lms": lms, "pan": pan, "ms": ms}
        
        if self.args.use_distillation:
            # 单步蒸馏模式
            timesteps = torch.tensor([999], device=self.device, dtype=torch.long)
            # 直接从MS图像开始（改进方案）
            x_t = ms.clone()
            
            # 单步去噪
            noise_pred = self.model(x_t, timesteps, **model_kwargs)
            if self.args.predict_xstart:
                output = noise_pred
            else:
                output = self.diffusion._predict_xstart_from_eps(
                    x_t, timesteps, noise_pred
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

