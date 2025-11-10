"""
后融合模型：融合OTPNet和SSDiff的输出
将两个预训练模型的输出拼接后，通过轻量级卷积网络进行融合
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加路径
sys.path.append('/data2/user/zelilin/pansharpening/SSDiff_main')
sys.path.append('/data2/user/zelilin/OTPNet')

from ssdiff_distill import SSDiff_test
from otpnet import OTPNet


class FusionBlock(nn.Module):
    """融合卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 确保输入和权重在同一设备和类型
        x = x.float()  # 转换为float32
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class OTPNet_SSDiff_Fusion(nn.Module):
    """
    OTPNet和SSDiff的后融合模型
    
    架构：
    1. OTPNet输出 [B, 8, H, W] (冻结)
    2. SSDiff输出 [B, 8, H, W] (冻结)
    3. 拼接 -> [B, 16, H, W]
    4. 融合网络 -> [B, 8, H, W] (可训练)
    """
    def __init__(
        self,
        otpnet_checkpoint,
        ssdiff_checkpoint,
        ssdiff_args,
        ms_channels=8,
        fusion_channels=64,
        num_fusion_blocks=3,
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.ms_channels = ms_channels
        
        print("="*60)
        print("初始化OTPNet-SSDiff融合模型")
        print("="*60)
        
        self.otpnet = self._load_otpnet(otpnet_checkpoint)
        self.otpnet.eval()
        for param in self.otpnet.parameters():
            param.requires_grad = False
        print("OTPNet加载完成并冻结")
        
        self.ssdiff = SSDiff_test(ssdiff_args)
        self.ssdiff.eval()
        for param in self.ssdiff.parameters():
            param.requires_grad = False
        print("SSDiff加载完成并冻结")
        
        # 3. 融合网络（可训练）
        print("\n创建融合网络...")
        self.fusion_net = nn.ModuleList()
        
        # 输入层：16通道 -> fusion_channels
        self.fusion_net.append(
            FusionBlock(ms_channels * 2, fusion_channels, kernel_size=3)
        )
        
        # 中间层
        for _ in range(num_fusion_blocks - 1):
            self.fusion_net.append(
                FusionBlock(fusion_channels, fusion_channels, kernel_size=3)
            )
        
        # 输出层：fusion_channels -> ms_channels
        self.output_conv = nn.Conv2d(fusion_channels, ms_channels, kernel_size=1)
        
        # 权重初始化
        self._initialize_fusion_weights()
        
        # 将融合网络移到设备
        self.fusion_net.to(device)
        self.output_conv.to(device)
        
        print(f" 融合网络创建完成：{num_fusion_blocks}个块 + 输出层")
        print(f"   可训练参数：{self.count_trainable_params():,}")
        print("="*60)
    
    def _load_otpnet(self, checkpoint_path):
        """加载OTPNet模型"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 从checkpoint中提取训练参数
        ckpt_args = {}
        if isinstance(checkpoint, dict) and "args" in checkpoint:
            stored = checkpoint.get("args")
            if isinstance(stored, dict):
                ckpt_args = stored
        
        # 构建模型配置（优先使用checkpoint中的参数）
        model_kwargs = {
            'pan_channels': 1,
            'ms_channels': self.ms_channels,
            'hidden_channels': ckpt_args.get('hidden_channels', 64),
            'num_stages': ckpt_args.get('num_stages', 4),
            'proximal_layers': ckpt_args.get('proximal_layers', 3),
            'norm': ckpt_args.get('norm', 'layer'),
            'upsample_mode': ckpt_args.get('upsample_mode', 'bicubic'),
        }
        
        print(f"   OTPNet配置: num_stages={model_kwargs['num_stages']}, "
              f"hidden_channels={model_kwargs['hidden_channels']}, "
              f"proximal_layers={model_kwargs['proximal_layers']}")
        
        model = OTPNet(**model_kwargs)
        
        # 加载权重（OTPNet的checkpoint使用'model_state'键）
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=True)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        return model.to(self.device)
    
    def _initialize_fusion_weights(self):
        """初始化融合网络权重"""
        for m in self.fusion_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 输出层使用较小的初始化
        nn.init.xavier_normal_(self.output_conv.weight, gain=0.01)
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0)
    
    def count_trainable_params(self):
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, pan, lms, ms=None):
        """
        前向传播
        
        Args:
            pan: 全色图像 [B, 1, H, W] (归一化到[0,1])
            lms: 上采样后的低分辨率多光谱 [B, 8, H, W] (由ms上采样得到，归一化到[0,1])
            ms: 原始低分辨率多光谱 [B, 8, H/4, W/4] (归一化到[0,1])
        
        Returns:
            output: 融合后的高分辨率多光谱 [B, 8, H, W]
            otpnet_out: OTPNet的输出
            ssdiff_out: SSDiff的输出
        """
        # 1. OTPNet推理（输入：pan, lr_ms）
        with torch.no_grad():
            lr_ms = ms
            otpnet_out = self.otpnet(pan, lr_ms)
        
        # 2. SSDiff推理（输入：lms, pan, ms）
        with torch.no_grad():
            ssdiff_out = self.ssdiff(lms, pan, ms)
        
        # 3. 拼接[B, 16, H, W]
        concat_features = torch.cat([otpnet_out, ssdiff_out], dim=1)
        
        # 4. 融合网络 - 确保类型一致
        x = concat_features.to(self.device)
        for block in self.fusion_net:
            x = block(x)
        
        # 5. 输出层（残差学习）
        residual = self.output_conv(x)
        
        base = (otpnet_out + ssdiff_out) / 2.0
        output = base + residual
        
        output = output.clamp(0, 1)
        
        return output, otpnet_out, ssdiff_out
    
    def save_fusion_weights(self, save_path):
        """只保存融合网络的权重"""
        state_dict = {
            'fusion_net': self.fusion_net.state_dict(),
            'output_conv': self.output_conv.state_dict(),
            'ms_channels': self.ms_channels,
        }
        torch.save(state_dict, save_path)
        print(f"[SAVE] 融合网络权重已保存到: {save_path}")
    
    def load_fusion_weights(self, load_path):
        """加载融合网络的权重"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.fusion_net.load_state_dict(checkpoint['fusion_net'])
        self.output_conv.load_state_dict(checkpoint['output_conv'])
        print(f" 融合网络权重已从 {load_path} 加载")


def create_fusion_model(
    otpnet_checkpoint,
    ssdiff_checkpoint,
    ssdiff_config_fn,
    fusion_channels=64,
    num_fusion_blocks=3,
    device='cuda'
):
    """
    创建融合模型的辅助函数
    
    
    Returns:
        model: 融合模型
    """
    # 获取SSDiff配置
    ssdiff_args = ssdiff_config_fn()
    ssdiff_args.model_path = ssdiff_checkpoint
    ssdiff_args.use_distillation = True  # 假设使用蒸馏模型
    ssdiff_args.device = device
    
    model = OTPNet_SSDiff_Fusion(
        otpnet_checkpoint=otpnet_checkpoint,
        ssdiff_checkpoint=ssdiff_checkpoint,
        ssdiff_args=ssdiff_args,
        fusion_channels=fusion_channels,
        num_fusion_blocks=num_fusion_blocks,
        device=device
    )
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("测试融合模型...")
    

    print("请使用train_fusion.py进行训练")

