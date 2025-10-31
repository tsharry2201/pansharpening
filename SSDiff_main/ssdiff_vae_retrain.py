"""
SSDiff ControlNet二次训练模型
基于一步蒸馏后的预训练模型和LoRA层，使用完整ControlNet进行二次训练
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
sys.path.append('/home/zelilin/data/pansharpening/SSDiff_main')
from utils.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults
from model.ARConv import ARConv
from model.fusformer import Fusformer
from model.SSNet import ResBlock, Down, Up
from model.nn import (
    SiLU,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from transformers import AutoTokenizer, CLIPTextModel
import torchvision.transforms as T
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from ram.models.ram_lora import ram
from ram import inference_ram as inference

class RAMCaptionGenerator:
    
    def __init__(self, model_path, device='cuda'):
        """
        初始化RAM模型
        Args:
            model_path: 预训练RAM权重路径，例如 'ram_swin_large_14m.pth'
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path

        # 加载模型
        print(f"🔧 正在加载 RAM 模型权重: {model_path}")
        self.model = ram(pretrained=model_path, image_size=384, vit='swin_l').to(device)
        self.model.eval()

        # 图像预处理
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("✓ RAM Caption Generator 已初始化（完整版）")
        print("  - 模型结构: Swin-Large")
        print("  - 权重文件:", model_path)
        print("  - 功能: 自动从PAN图像生成场景caption")

    def preprocess_image(self, image_tensor):
        """
        将输入的PAN图像张量转为模型可接受的格式
        Args:
            image_tensor: [1, H, W] 或 [B, 1, H, W]
        Returns:
            PIL.Image 格式
        """
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        if image_tensor.shape[0] == 1:
            zeros = torch.zeros_like(image_tensor)
            img = image_tensor.repeat(3, 1, 1)
            #print("zero")
        elif image_tensor.shape[0] > 3:
            img = image_tensor[[1,2,4],:,:]  # 多通道取平均

        img = img.clamp(0, 1)
        img_pil = T.ToPILImage()(img.cpu())
        return img_pil

    @torch.no_grad()
    def generate_caption(self, image_tensor):
        """
        从PAN图像生成语义caption
        Args:
            image_tensor: [B, 1, H, W] 或 [1, H, W]
        Returns:
            caption: 生成的文本描述字符串
        """
        img_pil = self.preprocess_image(image_tensor)
        img_input = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 模型前向推理，得到标签
        tags = self.model.generate_tag(img_input)
        if isinstance(tags, list):
        # 如果是嵌套列表，就展开成一维
            flat_tags = []
            for t in tags:
                if isinstance(t, list):
                    flat_tags.extend(t)
                elif isinstance(t, str):
                    flat_tags.append(t)
            tags = flat_tags
        elif isinstance(tags, str):
            tags = [tags]
        else:
            tags = [str(tags)]
        caption = ", ".join(tags)
        return caption

def _extract_effective_weight(conv_layer):
    """提取Conv层的有效权重"""
    if hasattr(conv_layer, 'base_layer') and hasattr(conv_layer, 'lora_down'):
        base_weight = conv_layer.base_layer.weight.data.clone()
        lora_down_weight = conv_layer.lora_down.weight.data
        lora_up_weight = conv_layer.lora_up.weight.data
        scaling = conv_layer.scaling
        
        rank, inc, kh, kw = lora_down_weight.shape
        outc, rank2, _, _ = lora_up_weight.shape
        
        lora_down_flat = lora_down_weight.reshape(rank, inc * kh * kw)
        lora_up_flat = lora_up_weight.squeeze(-1).squeeze(-1)
        lora_delta_flat = torch.matmul(lora_up_flat, lora_down_flat) * scaling
        lora_delta = lora_delta_flat.reshape(outc, inc, kh, kw)
        
        fused_weight = base_weight + lora_delta
        return fused_weight
    else:
        return conv_layer.weight.data.clone()


def switch_conv2d_to_arconv_with_interpolation(model, args):
    """动态切换：将模型中的Conv2d替换为ARConv"""
    def replace_conv_in_module(module, module_name=""):
        from model.SSNet import ResBlock
        
        if isinstance(module, ResBlock):
            if isinstance(module.conv0, ARConv):
                return
            
            conv0_weight = _extract_effective_weight(module.conv0)
            conv1_weight = _extract_effective_weight(module.conv1)
            
            in_channels = conv0_weight.shape[1]
            hidden_channels = conv0_weight.shape[0]
            out_channels = conv1_weight.shape[0]
            
            new_conv0 = ARConv(in_channels, hidden_channels, 3, 1, 1)
            new_conv1 = ARConv(hidden_channels, out_channels, 3, 1, 1)
            
            _init_arconv_from_conv2d_interpolation(new_conv0, conv0_weight)
            _init_arconv_from_conv2d_interpolation(new_conv1, conv1_weight)
            
            module.conv0 = new_conv0
            module.conv1 = new_conv1
            module.use_arconv = True
            
            if hasattr(args, 'arconv_hw_range'):
                hw_range = args.arconv_hw_range
                if isinstance(hw_range, str):
                    hw_range = eval(hw_range)
                module.arconv_hw_range = hw_range
            else:
                module.arconv_hw_range = [1, 9]
    
        for name, child in module.named_children():
            replace_conv_in_module(child, f"{module_name}.{name}" if module_name else name)
    
    replace_conv_in_module(model, "unet")


def _init_arconv_from_conv2d_interpolation(arconv_model, conv2d_weight):
    """ARConv恒等起步初始化"""
    kernel_sizes = [(3,3), (3,5), (5,3), (3,7), (7,3), (5,5), (5,7), (7,5), (7,7)]
    
    for i, (h, w) in enumerate(kernel_sizes):
        if h == 3 and w == 3:
            arconv_model.convs[i].weight.data = conv2d_weight.clone()
        else:
            expanded_weight = F.interpolate(
                conv2d_weight, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=True
            )
            arconv_model.convs[i].weight.data = expanded_weight
    
    nn.init.zeros_(arconv_model.m_conv[6].weight)
    if arconv_model.m_conv[6].bias is not None:
        nn.init.constant_(arconv_model.m_conv[6].bias, 3.0)
    
    nn.init.zeros_(arconv_model.b_conv[6].weight)
    if arconv_model.b_conv[6].bias is not None:
        nn.init.zeros_(arconv_model.b_conv[6].bias)
    
    nn.init.zeros_(arconv_model.p_conv[4].weight)
    if arconv_model.p_conv[4].bias is not None:
        nn.init.zeros_(arconv_model.p_conv[4].bias)
    
    nn.init.zeros_(arconv_model.l_conv[4].weight)
    if arconv_model.l_conv[4].bias is not None:
        nn.init.constant_(arconv_model.l_conv[4].bias, 0.0)
    
    nn.init.zeros_(arconv_model.w_conv[4].weight)
    if arconv_model.w_conv[4].bias is not None:
        nn.init.constant_(arconv_model.w_conv[4].bias, 0.0)


def initialize_ssdiff_unet_with_lora(args, pretrained_path=None):
    """初始化SSDiff的UNet并添加LoRA层到指定的Conv2d和Linear层
    
    Args:
        args: 参数配置
        pretrained_path: 预训练权重路径
        
    args中的lora_target选项:
        'all': 对所有Conv2d和Linear层应用LoRA（默认）
        'resblock': 仅对ResBlock应用LoRA
        'fusformer': 仅对Fusformer应用LoRA
        'resblock+fusformer': 对ResBlock和Fusformer应用LoRA
    """
    student_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    # 只设置合法的参数，使用getattr避免AttributeError
    if 'use_arconv' in student_args_dict:
        student_args_dict['use_arconv'] = getattr(args, 'use_arconv', False)
    if 'use_scene_token' in student_args_dict:
        student_args_dict['use_scene_token'] = getattr(args, 'use_scene_token', False)
    model, _ = create_model_and_diffusion(**student_args_dict)
    
    if pretrained_path is not None and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
    
    model.requires_grad_(False)
    model.train()
    
    # 获取LoRA目标配置
    lora_target = getattr(args, 'lora_target', 'all')
    
    # 扩展的LoRA应用策略：根据lora_target选择应用范围
    lora_target_modules = []
    excluded_patterns = [
        'time_embed',      # 时间嵌入层（保持冻结）
        'label_emb',       # 标签嵌入层
        'scene_embed',     # Scene token嵌入层（独立训练）
        'cross_attn.to_out.0',  # CrossAttention的输出层（使用Linear但已包含在to_out中）
    ]
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 检查是否应该排除
            should_exclude = any(pattern in name for pattern in excluded_patterns)
            if should_exclude:
                continue
            
            # 根据lora_target过滤模块
            if lora_target == 'all':
                # 所有层都添加LoRA
                lora_target_modules.append(name)
            elif lora_target == 'resblock':
                # 仅ResBlock
                if 'resblock' in name:
                    lora_target_modules.append(name)
            elif lora_target == 'fusformer':
                # 仅Fusformer
                if 'fusformer' in name:
                    lora_target_modules.append(name)
            elif lora_target == 'resblock+fusformer':
                # ResBlock和Fusformer
                if 'resblock' in name or 'fusformer' in name:
                    lora_target_modules.append(name)
            else:
                # 默认所有层
                lora_target_modules.append(name)
    
    # 使用所有候选模块（不再过滤）
    filtered_modules = lora_target_modules
    
    print(f"\n{'='*70}")
    print(f"LoRA配置:")
    print(f"  LoRA目标范围: {lora_target}")
    print(f"  候选模块总数: {len(lora_target_modules)}")
    print(f"  将添加LoRA的模块: {len(filtered_modules)}")
    print(f"  LoRA Rank: {args.lora_rank}")
    print(f"{'='*70}\n")
    
    class LoRALayer(nn.Module):
        def __init__(self, base_layer, rank=4, alpha=8):
            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
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
                nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
                nn.init.zeros_(self.lora_up.weight)
            elif isinstance(base_layer, nn.Linear):
                self.lora_down = nn.Linear(
                    base_layer.in_features,
                    rank,
                    bias=False
                )
                self.lora_up = nn.Linear(
                    rank,
                    base_layer.out_features,
                    bias=False
                )
                nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
                nn.init.zeros_(self.lora_up.weight)
        
        def forward(self, x):
            base_out = self.base_layer(x)
            if hasattr(self, 'lora_down'):
                x_dtype = x.dtype
                lora_down_out = self.lora_down(x.to(self.lora_down.weight.dtype))
                lora_out = self.lora_up(lora_down_out) * self.scaling
                lora_out = lora_out.to(x_dtype)
                return base_out + lora_out
            return base_out
    
    lora_count = 0
    lora_stats = {'resblock': 0, 'fusformer': 0, 'down': 0, 'up': 0, 'other': 0}
    
    for name, module in model.named_modules():
        if name in filtered_modules and isinstance(module, (nn.Conv2d, nn.Linear)):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            
            lora_layer = LoRALayer(module, rank=args.lora_rank, alpha=args.lora_rank * 2)
            setattr(parent, child_name, lora_layer)
            lora_count += 1
            
            # 统计各模块的LoRA数量
            if 'resblock' in name:
                lora_stats['resblock'] += 1
            elif 'fusformer' in name:
                lora_stats['fusformer'] += 1
            elif 'down' in name:
                lora_stats['down'] += 1
            elif 'up' in name:
                lora_stats['up'] += 1
            else:
                lora_stats['other'] += 1
    
    print(f"✅ 成功为 {lora_count} 个层添加LoRA")
    print(f"   模块分布:")
    print(f"     - ResBlock:  {lora_stats['resblock']} 层")
    print(f"     - Fusformer: {lora_stats['fusformer']} 层")
    print(f"     - Down/Up:   {lora_stats['down'] + lora_stats['up']} 层")
    print(f"     - 其他:      {lora_stats['other']} 层\n")
    
    return model, filtered_modules


class PerceptualLoss(nn.Module):
    """感知损失，使用VGG16提取特征，更加稳定的版本"""
    def __init__(self, style_weight=0):
        super().__init__()
        vgg = models.vgg16(pretrained=True).eval()
        self.feature_extractor = create_feature_extractor(
            vgg, 
            return_nodes={
                'features.4': 'relu1_2',  # 浅层特征
                'features.9': 'relu2_2',  # 中层特征
                # 减少深层特征，它们可能对于遥感图像不太适用
                # 'features.16': 'relu3_3',
                # 'features.23': 'relu4_3'
            }
        )
        self.feature_extractor.requires_grad_(False)
        
        # 调整特征层权重，更关注浅层特征
        self.content_weights = {
            'relu1_2': 0.7,  # 增加浅层特征权重
            'relu2_2': 0.3,  # 减少中层特征权重
        }
        
        # 注册缓冲区以保存均值和标准差
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def _normalize(self, x):
        """将图像归一化到VGG预期的范围，增强稳定性"""
        # 确保输入是3通道的，如果不是则复制通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            # 如果是多光谱图像，取前3个通道
            x = x[:, :3, :, :]
        
        # 强制归一化到[0, 1]范围
        x = x.clamp(0, 1)
        
        # 应用ImageNet归一化
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        """计算x和y之间的感知损失，添加梯度裁剪以增强稳定性"""
        # 确保输入在合理范围内
        x = x.clamp(0, 1)
        y = y.clamp(0, 1)
        
        x = self._normalize(x)
        y = self._normalize(y)
        
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        
        content_loss = 0.0
        for layer, weight in self.content_weights.items():
            # 使用L1损失代替MSE，可能更稳定
            layer_loss = weight * F.l1_loss(x_features[layer], y_features[layer])
            content_loss += layer_loss
        
        # 裁剪异常大的损失值
        if content_loss > 10.0:
            content_loss = torch.log(content_loss + 1.0)
        
        return content_loss


class VAEEncoder(nn.Module):
    """VAE编码器，用于提取PAN图像的特征表示，同时支持ControlNet模式"""
    def __init__(self, input_channels=1, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器主干网络
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # 全局特征提取（用于VAE潜在向量）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # 存储中间特征图（用于ControlNet）
        self.feature_maps = {}
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        前向传播，支持返回中间特征图
        Args:
            x: 输入图像
            return_features: 是否返回中间特征图（用于ControlNet）
        """
        # 清空上一次的特征图
        self.feature_maps = {}
        
        # 分阶段提取特征
        features = []
        x_input = x
        
        # 提取每层特征
        for i, layer in enumerate(self.encoder):
            x_input = layer(x_input)
            if i in [1, 4, 7, 10]:  # LeakyReLU层后的特征
                features.append(x_input)
                self.feature_maps[f'level_{len(features)}'] = x_input
        
        # 提取全局特征用于VAE
        h = self.global_pool(x_input)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        if return_features:
            return mu, logvar, features
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_feature_maps(self):
        """获取中间特征图，用于ControlNet"""
        return self.feature_maps


class ZeroConv(nn.Module):
    """零初始化的1x1卷积层，用于ControlNet输出"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)


class ControlNet(nn.Module):
    """
    完整的ControlNet实现，完全复刻SSNet编码器结构并复制其权重
    用于从PAN图像中提取多尺度空间控制特征

    1. 结构与SSNet的PAN编码器完全一致（trainable copy）
    2. 从SSNet复制预训练权重（locked copy -> trainable copy）
    3. 使用zero-convolution确保训练初期不影响主模型
    """
    def __init__(self, unet_model, ms_dim=8, pan_dim=1, copy_weights=True, verbose=True):
        super().__init__()
        self.ms_dim = ms_dim
        self.pan_dim = pan_dim
        self.verbose = verbose  # 控制是否输出详细信息
        
        # 从UNet中提取必要的配置
        self.model_channels = unet_model.model_channels
        self.use_scale_shift_norm = unet_model.use_scale_shift_norm
        dim = 32  # 基础维度

        
        # 时间嵌入层（复制自SSNet）
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # 输入端zero-conv（原始ControlNet设计）
        # 将PAN+x_t映射到初始特征，但训练初期输出为0
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(ms_dim + pan_dim, 16, 3, 1, 1),
            SiLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            SiLU(),
            zero_module(nn.Conv2d(32, dim, 3, 1, 1))  # 零初始化最后一层
        )
        
        # 时间嵌入投影层（复制自SSNet）
        self.out_layers_pan = nn.Sequential(
            nn.GroupNorm(32, dim),
            SiLU(),
            nn.Dropout(p=0.0),
            zero_module(conv_nd(2, dim, dim, 3, padding=1)),
        )
        
        self.emb_layers_pan = nn.Sequential(
            SiLU(),
            linear(time_embed_dim, 2 * dim if self.use_scale_shift_norm else dim),
        )
        
        # 定义各层维度（与SSNet完全一致）
        dim0 = dim       # 32
        dim1 = dim * 2   # 64
        dim2 = dim * 4   # 128
        dim3 = dim * 2   # 64
        dim4 = dim       # 32
        
        dim_head = 16
        se_ratio_mlp = 0.5
        se_ratio_rb = 0.5
        
        # Level 0: 64x64 分辨率（与SSNet完全一致）
        self.fusformer0 = Fusformer(dim0, dim0//dim_head, dim_head, int(dim0*se_ratio_mlp))
        self.resblock0 = ResBlock(
            dim0, int(se_ratio_rb*dim0), dim0,
            model_channels=self.model_channels,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_arconv=False,  # ControlNet不使用ARConv
        )
        self.down0 = Down(dim0, dim1)
        
        # Level 1: 32x32 分辨率
        self.fusformer1 = Fusformer(dim1, dim1//dim_head, dim_head, int(dim1*se_ratio_mlp))
        self.resblock1 = ResBlock(
            dim1, int(se_ratio_rb*dim1), dim1,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_arconv=False,
        )
        self.down1 = Down(dim1, dim2)
        
        # Level 2: 16x16 分辨率（最深层）
        self.fusformer2 = Fusformer(dim2, dim2//dim_head, dim_head, int(dim2*se_ratio_mlp))
        self.resblock2 = ResBlock(
            dim2, int(se_ratio_rb*dim2), dim2,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_arconv=False,
        )
        self.up0 = Up(dim2, dim3)
        
        # Level 3: 32x32 分辨率（上采样）
        self.fusformer3 = Fusformer(dim3, dim3//dim_head, dim_head, int(dim3*se_ratio_mlp))
        self.resblock3 = ResBlock(
            dim3, int(se_ratio_rb*dim3), dim3,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_arconv=False,
        )
        self.up1 = Up(dim3, dim4)
        
        self.fusformer4 = Fusformer(dim4, dim4//dim_head, dim_head, int(dim4*se_ratio_mlp))
        
        # ========================================
        # 从UNet复制权重作为初始化（训练时必要，测试时会被checkpoint覆盖）
        if copy_weights:
            self._copy_weights_from_unet(unet_model)
        
        self.zero_convs = nn.ModuleDict({
            'level_0': ZeroConv(dim0, dim0),
            'level_1': ZeroConv(dim1, dim1),
            'level_2': ZeroConv(dim2, dim2),
            'level_3': ZeroConv(dim3, dim3),
            'level_4': ZeroConv(dim4, dim4),
        })
        
        # 门控参数
        self.control_gate = nn.ParameterDict({
            'level_0': nn.Parameter(torch.tensor(-0.5)),
            'level_1': nn.Parameter(torch.tensor(-0.5)),
            'level_2': nn.Parameter(torch.tensor(-0.5)),
            'level_3': nn.Parameter(torch.tensor(-0.5)),
            'level_4': nn.Parameter(torch.tensor(-0.5))
        })
        
        print(f"\n{'='*70}")
        print(f"ControlNet初始化完成")
    
    def _copy_weights_from_unet(self, unet_model):
        """
        从SSNet复制对应模块的预训练权重作为初始化
        """
        if self.verbose:
            print(f"\n📦 正在从UNet复制权重进行初始化...")
            print(f"   (注：测试时此权重会被checkpoint覆盖)")
        
        # 映射表：ControlNet模块 -> SSNet模块
        module_mapping = [
            # 时间嵌入
            ('time_embed', 'time_embed'),
            ('emb_layers_pan', 'emb_layers_pan'),
            ('out_layers_pan', 'out_layers_pan'),
            # Level 0
            ('fusformer0', 'fusformer0'),
            ('resblock0', 'resblock0'),
            ('down0', 'down0'),
            # Level 1
            ('fusformer1', 'fusformer1'),
            ('resblock1', 'resblock1'),
            ('down1', 'down1'),
            # Level 2
            ('fusformer2', 'fusformer2'),
            ('resblock2', 'resblock2'),
            ('up0', 'up0'),
            # Level 3
            ('fusformer3', 'fusformer3'),
            ('resblock3', 'resblock3'),
            ('up1', 'up1'),
            # Level 4
            ('fusformer4', 'fusformer4'),
        ]
        
        copied_modules = 0
        total_params = 0
        
        for ctrl_name, unet_name in module_mapping:
            if hasattr(self, ctrl_name) and hasattr(unet_model, unet_name):
                ctrl_module = getattr(self, ctrl_name)
                unet_module = getattr(unet_model, unet_name)
                
                # 复制权重
                try:
                    ctrl_state = ctrl_module.state_dict()
                    unet_state = unet_module.state_dict()
                    
                    # 只复制形状匹配的参数
                    for key in ctrl_state.keys():
                        if key in unet_state and ctrl_state[key].shape == unet_state[key].shape:
                            ctrl_state[key].copy_(unet_state[key])
                            total_params += ctrl_state[key].numel()
                    
                    ctrl_module.load_state_dict(ctrl_state)
                    copied_modules += 1
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ 跳过 {ctrl_name}: {e}")
        
        if self.verbose:
            print(f"✓ 成功复制 {copied_modules}/{len(module_mapping)} 个模块")
            print(f"✓ ControlNet权重初始化完成")
    
    def time_emb_pan(self, h, emb):
        """应用时间嵌入到PAN特征"""
        emb_out = self.emb_layers_pan(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers_pan[0], self.out_layers_pan[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers_pan(h)
        return h
    
    def forward(self, pan, x_t, timesteps):
        """
        前向传播，提取多尺度控制特征（完全遵循原始ControlNet设计）
        
        Args:
            pan: PAN图像 [B, 1, H, W]
            x_t: 噪声残差 [B, 8, H, W]
            timesteps: 时间步 [B]
        
        Returns:
            control_features: 字典，包含5个层级的控制特征
                - 'level_0': [B, 32, 64, 64]
                - 'level_1': [B, 64, 32, 32]
                - 'level_2': [B, 128, 16, 16]
                - 'level_3': [B, 64, 32, 32]
                - 'level_4': [B, 32, 64, 64]
        """
        # 时间嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # 原始ControlNet：输入条件通过hint block处理
        # 输入: PAN+x_t -> hint block -> 初始特征
        # 训练初期：hint block的最后一层是零初始化，输出≈0
        pan_xt = torch.cat([pan, x_t], dim=1)  # [B, 9, H, W]
        y = self.input_hint_block(pan_xt)  # [B, 32, H, W]
        
        # 时间嵌入注入
        y = self.time_emb_pan(y, emb)
        
        control_features = {}
        
        # Level 0: 64x64
        y = self.fusformer0(y, y)  # 自注意力
        y = self.resblock0(y, emb, epoch=0)
        # 原始ControlNet：输出通过zero-conv + 门控
        # zero_conv: 训练初期输出为0
        # control_gate: sigmoid(-3.0) ≈ 0.047，进一步抑制 现在不用了
        control_features['level_0'] = self.zero_convs['level_0'](y) 
        skip_0 = y
        y = self.down0(y)  # 下采样到32x32
        
        # Level 1: 32x32
        y = self.fusformer1(y, y)
        y = self.resblock1(y, emb, epoch=0)
        control_features['level_1'] = self.zero_convs['level_1'](y) 
        skip_1 = y
        y = self.down1(y)  # 下采样到16x16
        
        # Level 2: 16x16（最深层）
        y = self.fusformer2(y, y)
        y = self.resblock2(y, emb, epoch=0)
        control_features['level_2'] = self.zero_convs['level_2'](y)
        y = self.up0(y, skip_1)  # 上采样到32x32
        
        # Level 3: 32x32（上采样）
        y = self.fusformer3(y, y)
        y = self.resblock3(y, emb, epoch=0)
        control_features['level_3'] = self.zero_convs['level_3'](y) 
        y = self.up1(y, skip_0)  # 上采样到64x64
        
        # Level 4: 64x64（最终层）
        y = self.fusformer4(y, y)
        control_features['level_4'] = self.zero_convs['level_4'](y) 
        
        return control_features


class SSDiff_VAE_gen(nn.Module):
    """基于VAE的单步生成器模型，支持ControlNet"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_step = 0
        
        # 加载预训练的蒸馏模型（带LoRA）- 需要先初始化UNet
        self.unet, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args, 
            pretrained_path=args.pretrained_ssdiff_path
        )
        
        # 加载预训练的LoRA权重
        if hasattr(args, 'lora_checkpoint_path') and args.lora_checkpoint_path:
            lora_state = torch.load(args.lora_checkpoint_path, map_location='cpu')
            if 'unet_state_dict' in lora_state:
                for name, param in self.unet.named_parameters():
                    if 'lora' in name and name in lora_state['unet_state_dict']:
                        param.data.copy_(lora_state['unet_state_dict'][name])
        
        # 初始化各个模块（互相独立）
        self.vae_encoder = None
        self.controlnet_full = None
        self.tokenizer = None
        self.text_encoder = None
        self.use_perceptual_loss = False
        self.use_kl_loss = False
        self.use_clip = False
        self.use_ram = False  
        self.ram_caption_generator = None
        
        # 选项1：VAE编码器（用于提取全局场景特征，作为scene_token）
        if hasattr(args, 'use_vae') and args.use_vae:
            latent_dim = getattr(args, 'vae_latent_dim', 256)
            self.vae_encoder = VAEEncoder(
                input_channels=1,
                latent_dim=latent_dim
            )
            print(f"✓ 已启用VAE编码器（全局场景特征）")
            
            # VAE相关的损失函数
            self.use_perceptual_loss = getattr(args, 'use_perceptual_loss', False)
            if self.use_perceptual_loss:
                self.perceptual_loss = PerceptualLoss()
                self.lambda_perceptual = getattr(args, 'lambda_perceptual', 0.1)
                print(f"  - 感知损失: 启用 (λ={self.lambda_perceptual})")
            
            self.use_kl_loss = getattr(args, 'use_kl_loss', False)
            self.lambda_kl = getattr(args, 'lambda_kl', 0.001)
            if self.use_kl_loss:
                print(f"  - KL散度损失: 启用 (λ={self.lambda_kl})")
        
        # 选项2：完整ControlNet（用于提取多尺度空间特征，作为control_features）
        if hasattr(args, 'use_controlnet') and args.use_controlnet:
            ms_dim = getattr(args, 'ms_channels', 8)
            pan_dim = 1
            self.controlnet_full = ControlNet(
                unet_model=self.unet,
                ms_dim=ms_dim,
                pan_dim=pan_dim
            )
            print(f"✓ 已启用完整ControlNet（多尺度空间特征）")
        
        # 选项3：CLIP文本编码器（用于从文本提示中提取语义特征）
        if hasattr(args, 'use_clip') and args.use_clip:
            # 使用SD21Base作为CLIP基础模型路径
            clip_base_path = getattr(args, 'clip_model_path', None)
            if clip_base_path is None:
                # 默认使用SD21Base路径
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                clip_base_path = os.path.join(script_dir, "model", "SD21Base")
            
            print(f"try to use clip from: {clip_base_path}")
            try:
                # 从SD21Base加载tokenizer和text_encoder基础结构
                self.tokenizer = AutoTokenizer.from_pretrained(
                    clip_base_path, 
                    subfolder="tokenizer"
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    clip_base_path, 
                    subfolder="text_encoder"
                ).to(next(self.unet.parameters()).device)
                
                # 冻结CLIP模型（训练时不更新CLIP权重）
                self.text_encoder.requires_grad_(False)
                
                self.use_clip = True
                self.default_prompt = getattr(args, 'prompt', "A high resolution satellite image")
                
                print(f"✓ 已从SD21Base加载CLIP文本编码器")
                print(f"  - 基础模型路径: {clip_base_path}")
                print(f"  - 默认提示: \"{self.default_prompt}\"")
                print(f"  - CLIP模型已冻结（不参与训练）")
                
            except Exception as e:
                print(f"⚠ CLIP模型初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self.use_clip = False
                self.tokenizer = None
                self.text_encoder = None
        
        #  选项4：RAM Caption生成器（用于从PAN图像自动生成caption）
        if hasattr(args, 'use_ram') and args.use_ram:
            self.use_ram = True
            ram_model_path = getattr(args, 'ram_model_path', None)

            try:
                # =============================
                # 初始化 RAM caption 生成器（使用自定义封装）
                # =============================
                device = next(self.unet.parameters()).device

                # 初始化生成器
                self.ram_caption_generator = RAMCaptionGenerator(
                    model_path=ram_model_path,
                    device=device
                )

                print(f"✓ 已启用RAM Caption生成器（Swin-Large 模型）")
                print(f"  - 自动从PAN图像生成caption")
                print(f"  - 模型路径: {ram_model_path}")

            except Exception as e:
                print(f"⚠ RAM模型加载失败: {e}")
                self.use_ram = False
                self.ram_caption_generator = None
        
        # 打印最终配置
        if self.vae_encoder is None and self.controlnet_full is None and not self.use_clip and not self.use_ram:
            print(f"⚠ 未启用任何条件模块，使用原始UNet+LoRA")
        
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        self.inference_timestep = 999
        self.lora_rank = args.lora_rank
        self.training = True
    
    def set_step(self, step):
        self.current_step = step
        if hasattr(self.unet, 'set_epoch'):
            self.unet.set_epoch(step)
    
    def _build_color_subspace(self):
        """
        构建颜色子空间的正交基（用于颜色去偏）
        """
        color_words = [
            "green", "blue", "red", "brown", "yellow", "white", "gray", "grey",
            "black", "purple", "pink", "orange", "cyan", "magenta",
            "light", "dark", "pale", "bright", "deep"
        ]
        
        device = next(self.text_encoder.parameters()).device
        
        # 编码颜色词
        with torch.no_grad():
            color_embeds_list = []
            for color in color_words:
                text_input = self.tokenizer(
                    color,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                color_embed = self.text_encoder(text_input.input_ids)[0]  # [1, 77, 768]
                color_embed = color_embed.mean(dim=1)  # [1, 768]
                color_embeds_list.append(color_embed)
            
            # 堆叠所有颜色embedding: [K, D]
            color_embeds = torch.cat(color_embeds_list, dim=0)
            # 归一化
            color_embeds = F.normalize(color_embeds, dim=-1)
            
            # QR分解构建正交基: Q @ R = color_embeds.T
            Q, _ = torch.linalg.qr(color_embeds.T)  # Q: [D, K]
        
        return Q
    
    def _apply_color_debiasing(self, text_embeds, Q):
        """
        对text embedding应用颜色子空间去偏
        """
        # 投影到颜色子空间: proj_color = (Q @ Q.T) @ text_embeds.T
        proj_color = Q @ (Q.T @ text_embeds.T)  # [D, B]
        
        # 去除颜色分量
        text_debiased = text_embeds.T - proj_color  # [D, B]
        text_debiased = text_debiased.T  # [B, D]
        
        # 归一化到unit sphere
        text_debiased = F.normalize(text_debiased, dim=-1)
        
        return text_debiased
    
    def select_clip_semantic_label(self, ram_caption, top_k=3, max_ram_tags=12):
        """
        使用CLIP对RAM生成的caption进行语义过滤（标准版本）
        Args:
            ram_caption: RAM生成的原始caption（逗号分隔的标签列表）
            top_k: 选择前k个最相关的标签（默认3）
            max_ram_tags: RAM输出中保留的最大标签数（避免过长导致截断）
        Returns:
            topk_labels: 过滤后的标签列表
        """
        labels = [
            "urban area", "residential zone", "industrial zone", "forest", "grassland",
            "mountain", "river", "lake", "sea", "beach", "harbor", "farmland",
            "road network", "bare soil", "wetland", "desert",
            "bridge", "building", "highway", "stadium",
            "coastline", "cliff", "hill slope", "dense vegetation", "open water",
            "cloudy", "coastal city", "harbor town"
        ]
        
        # 预处理RAM的输出：RAM按概率从高到低输出标签，我们只保留前N个
        if isinstance(ram_caption, str):
            ram_tags = [tag.strip() for tag in ram_caption.split(',')]
            ram_tags = ram_tags[:max_ram_tags]
            ram_caption_short = ', '.join(ram_tags)
        else:
            ram_caption_short = ram_caption
        
        # 将RAM输出与标签一并tokenize
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            labels + [ram_caption_short],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # 提取文本特征
            text_embeds = self.text_encoder(text_inputs.input_ids)[0]  # [len(labels)+1, 77, 768]
            text_embeds = text_embeds.mean(dim=1)  # 全局平均池化 -> [len(labels)+1, 768]
            text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)
            
            # 计算余弦相似度：labels与ram_caption的相似度
            label_embeds = text_embeds[:-1]  # [num_labels, 768]
            caption_embed = text_embeds[-1:]  # [1, 768]
            scores = (label_embeds * caption_embed).sum(dim=-1)  # [num_labels]
            
        # 取前 top_k 个标签
        topk_idx = torch.topk(scores, k=min(top_k, len(labels))).indices.cpu().numpy()
        topk_labels = [labels[i] for i in topk_idx]
        
        return topk_labels
    
    def select_clip_semantic_label_with_color_debiasing(self, ram_caption, top_k=5):
        """
        对RAM生成的caption进行颜色去偏，直接返回去偏后的标签
        
        """
        # 解析RAM输出
        if isinstance(ram_caption, str):
            ram_tags = [tag.strip() for tag in ram_caption.split(',')]
        else:
            ram_tags = ram_caption
        
        # 如果RAM输出少于等于top_k个，直接返回全部
        if len(ram_tags) <= top_k:
            return ram_tags[:top_k]
        
        # 编码所有RAM标签
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            ram_tags,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # 提取文本特征
            text_embeds = self.text_encoder(text_inputs.input_ids)[0]  # [num_tags, 77, 768]
            text_embeds = text_embeds.mean(dim=1)  # 全局平均池化 -> [num_tags, 768]
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            Q = self._build_color_subspace()  # [768, K]
            
            tag_embeds_debiased = self._apply_color_debiasing(text_embeds, Q)  # [num_tags, 768]
            
            color_scores = (text_embeds * tag_embeds_debiased).sum(dim=-1)  # [num_tags]
            
            topk_idx = torch.topk(color_scores, k=min(top_k, len(ram_tags))).indices.cpu().numpy()
            
        debiased_labels = [ram_tags[i] for i in topk_idx]
        
        return debiased_labels
    
    def encode_prompt(self, prompt_batch):
        """
        编码文本提示为文本嵌入
        Args:
            prompt_batch: 文本提示列表
        Returns:
            prompt_embeds: 文本嵌入 [B, 77, 768]
        """
        if not self.use_clip or self.tokenizer is None or self.text_encoder is None:
            return None 
        prompt_embeds_list = []
        with torch.no_grad():  # 不计算梯度，确认不训练
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, 
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device)
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        
        if prompt_embeds_list:
            prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
            prompt_embeds = F.layer_norm(prompt_embeds, prompt_embeds.shape[-1:])
            ## 修改
            if self.current_step % 500 == 0:
                with torch.no_grad():
                    mean_var = prompt_embeds.var(dim=-1).mean().item()
                    mean_norm = prompt_embeds.norm(dim=-1).mean().item()
                    print(f"[CLIP monitor] norm={mean_norm:.3f}, var={mean_var:.5f}")
            return prompt_embeds
        return None
    
    def set_train(self, enable_arconv=False, train_vae=False, train_lora=True, train_controlnet=True, train_clip=False):
        """
        设置训练模式
        Args:
            enable_arconv: 是否启用ARConv训练
            train_vae: 是否训练VAE编码器
            train_lora: 是否训练LoRA层（默认True）
            train_controlnet: 是否训练完整ControlNet（默认True）
            train_clip: 是否训练CLIP文本编码器（默认False）
        """
        self.training = True
        self.unet.train()
        
        trainable_count = {'lora': 0, 'vae': 0, 'arconv': 0, 'controlnet': 0, 'clip': 0, 'gate': 0, 'proj': 0}
        
        # 设置UNet参数
        for n, p in self.unet.named_parameters():
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = train_lora
                if train_lora:
                    trainable_count['lora'] += 1
            elif any(key in n.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'kernel_gen']):
                p.requires_grad = enable_arconv
                if enable_arconv:
                    trainable_count['arconv'] += 1
            #
            elif 'gate' in n and 'attn_gate' not in n:  # attn_gate是buffer，不训练
                p.requires_grad = True
                trainable_count['gate'] += 1
            #
            elif 'text_proj' in n:
                p.requires_grad = True
                trainable_count['proj'] += 1
            else:
                p.requires_grad = False
        
        # 设置VAE编码器参数（独立）
        if self.vae_encoder is not None:
            self.vae_encoder.train()
            for p in self.vae_encoder.parameters():
                p.requires_grad = train_vae
                if train_vae:
                    trainable_count['vae'] += 1
        
        # 设置完整ControlNet参数（独立）
        if self.controlnet_full is not None:
            self.controlnet_full.train()
            for p in self.controlnet_full.parameters():
                p.requires_grad = train_controlnet
                if train_controlnet:
                    trainable_count['controlnet'] += 1
        
        # 设置CLIP文本编码器参数（独立）
        if self.use_clip and self.text_encoder is not None:
            if train_clip:
                self.text_encoder.train()
                for p in self.text_encoder.parameters():
                    p.requires_grad = True
                    trainable_count['clip'] += 1
            else:
                self.text_encoder.eval()
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
        
        if not hasattr(self, '_last_train_state') or self._last_train_state != (enable_arconv, train_vae, train_lora, train_controlnet, train_clip):
            arconv_status = "训练中" if enable_arconv else "冻结"
            vae_status = "训练中" if train_vae else "冻结"
            lora_status = "训练中" if train_lora else "冻结"
            controlnet_status = "训练中" if train_controlnet else "冻结"
            clip_status = "训练中" if train_clip else "冻结"
            print(f"可训练参数: LoRA={trainable_count['lora']} ({lora_status}), VAE={trainable_count['vae']} ({vae_status}), ARConv={trainable_count['arconv']} ({arconv_status}), ControlNet={trainable_count['controlnet']} ({controlnet_status}), CLIP={trainable_count['clip']} ({clip_status})")
            # 显示门控和投影层参数（始终训练）
            if trainable_count['gate'] > 0 or trainable_count['proj'] > 0:
                print(f" CLIP相关参数: Gate={trainable_count['gate']} (始终训练), Projection={trainable_count['proj']} (始终训练)")
            self._last_train_state = (enable_arconv, train_vae, train_lora, train_controlnet, train_clip)
    
    def forward(self, lms, pan, ms, gt=None, prompt=None):
        device = lms.device
        batch_size = lms.shape[0]
        
        # VAE编码器：提取全局场景特征 -> scene_token
        vae_features = None
        if self.vae_encoder is not None:
            mu, logvar = self.vae_encoder(pan)
            vae_features = mu  # 使用mu作为scene_token
        
        # CLIP文本编码器：提取文本语义特征 -> encoder_hidden_states
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            # 如果启用RAM且没有提供prompt，为当前batch生成caption
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]  # 使用PAN图像生成caption
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label_with_color_debiasing(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions
                    
                    # 定期监控caption多样性
                    if self.current_step % 500 == 0:
                        diversity = len(set(captions)) / max(len(captions), 1)
                        avg_len = sum(len(str(c)) for c in captions) / max(len(captions), 1)
                        print(f"[RAM monitor] batch_size={len(captions)}, unique_rate={diversity:.3f}, avg_len={avg_len:.1f}")
                        print(f"  Sample captions: {captions[:3]}")
                        
            # 如果仍然没有prompt，使用默认prompt
            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")
                
            # 将单个提示转换为列表
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size
                
            # 编码文本提示
            text_embeds = self.encode_prompt(prompt)
        
        # 生成时间步和噪声
        if self.training:
            timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        else:
            timesteps = torch.full((batch_size,), self.inference_timestep, device=device, dtype=torch.long)
        
        if self.training and gt is not None:
            gt_residual = gt - lms
            noise = torch.randn_like(gt_residual)
            x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        else:
            x_t = torch.zeros_like(lms)
        
        # ControlNet：提取多尺度空间特征 -> control_features
        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)
            
            # 在forward中也监控 ControlNet 特征（每500步）
            if not self.training and hasattr(self, 'current_step') and self.current_step % 500 == 0:
                print(f"\n[ControlNet 特征监控 - Forward]")
                print(f"{'层级':<10} {'Mean':<12} {'Std':<12} {'Abs Mean':<12} {'Shape'}")
                print("-" * 70)
                for level_name, features in control_features.items():
                    mean_val = features.mean().item()
                    std_val = features.std().item()
                    abs_mean_val = features.abs().mean().item()
                    shape_str = 'x'.join(map(str, features.shape[1:]))
                    print(f"{level_name:<10} {mean_val:>11.6f} {std_val:>11.6f} {abs_mean_val:>11.6f} [{shape_str}]")
        
        # UNet前向传播（可以同时使用scene_token、control_features和text_embeds）
        residual_pred = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps, 
            scene_token=vae_features,      # VAE的全局特征
            control_features=control_features,  # ControlNet的空间特征
            encoder_hidden_states=text_embeds,  # CLIP的文本特征
            epoch=self.current_step
        )
        
        if torch.isnan(residual_pred).any():
            residual_pred = torch.nan_to_num(residual_pred, nan=0.0)
        
        if self.args.predict_xstart:
            residual = residual_pred
        else:
            residual = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
        
        output = lms + residual
        output = output.clamp(0, 1)
        
        return output, residual_pred, vae_features
    
    def l1_loss(self, lms, pan, ms, gt, prompt=None):
        batch_size = lms.shape[0]
        device = lms.device
        
        # VAE编码器：提取全局场景特征
        vae_features = None
        mu = None
        logvar = None
        if self.vae_encoder is not None:
            mu, logvar = self.vae_encoder(pan)
            vae_features = mu
        
        # CLIP文本编码器：提取文本语义特征
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            # 如果启用RAM且没有提供prompt，为当前batch生成caption
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]  # 使用PAN图像生成caption
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label_with_color_debiasing(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions
            
            # 如果仍然没有prompt，使用默认prompt
            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")
            
            # 将单个提示转换为列表
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size
                
            # 编码文本提示
            text_embeds = self.encode_prompt(prompt)
        
        timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        
        gt_residual = gt - lms
        noise = torch.randn_like(gt_residual)
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        # ControlNet：提取多尺度空间特征
        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)
        
        # UNet前向传播
        residual_pred = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps, 
            scene_token=vae_features,      # VAE的全局特征
            control_features=control_features,  # ControlNet的空间特征
            encoder_hidden_states=text_embeds,  # CLIP的文本特征
            epoch=self.current_step
        )
        
        if self.args.predict_xstart:
            residual_final = residual_pred
        else:
            residual_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
        
        # 计算L1重建损失
        l1_loss = F.l1_loss(residual_final, gt_residual, reduction="mean")
        
        # 初始化总损失为L1损失
        total_loss = l1_loss
        
        # 损失日志
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        # 添加KL散度损失
        kl_loss = 0.0
        if self.use_kl_loss and mu is not None and logvar is not None:
            # KL散度: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            total_loss = total_loss + self.lambda_kl * kl_loss
            loss_dict['kl_loss'] = kl_loss.item()
            loss_dict['total_loss'] = total_loss.item()
        
            # 添加感知损失
            perceptual_loss = 0.0
            if self.use_perceptual_loss and hasattr(self, 'perceptual_loss'):
                try:
     
                    pred_image = lms + residual_final
          
                    pred_image = pred_image.clamp(0, 1)
                    gt = gt.clamp(0, 1)
                    
          
                    perceptual_loss = self.perceptual_loss(pred_image, gt)
                    
             
                    if perceptual_loss > 5.0:
                        perceptual_loss = torch.log(perceptual_loss + 1.0)
                        
     
                    weighted_perceptual_loss = self.lambda_perceptual * perceptual_loss
                    total_loss = total_loss + weighted_perceptual_loss
                    loss_dict['perceptual_loss'] = perceptual_loss.item()
                    loss_dict['weighted_perceptual_loss'] = weighted_perceptual_loss.item()
                    loss_dict['total_loss'] = total_loss.item()
                except Exception as e:
                    print(f"计算感知损失时出错: {e}")
                    # 出错时不添加感知损失
                    loss_dict['perceptual_loss_error'] = str(e)
        if text_embeds is not None:
            clip_var = text_embeds.var(dim=-1).mean()
            clip_loss = 1e-3 * (clip_var - 1.0).pow(2)   # 让方差接近 1
            total_loss = total_loss + clip_loss
            loss_dict['clip_loss'] = clip_loss.item()
        # 每100步打印损失详情
        if hasattr(self, 'current_step') and self.current_step % 500 == 0:
            print(f"\n📊 损失详情 (步数 {self.current_step}):")
            print(f"   L1损失: {l1_loss.item():.6f}")
            
            if self.use_kl_loss and 'kl_loss' in loss_dict:
                kl_value = loss_dict['kl_loss']
                print(f"   KL散度损失: {kl_value:.6f} (λ={self.lambda_kl})")
                
            if self.use_perceptual_loss and 'perceptual_loss' in loss_dict:
                p_value = loss_dict['perceptual_loss']
                wp_value = loss_dict.get('weighted_perceptual_loss', p_value * self.lambda_perceptual)
                print(f"   感知损失: {p_value:.6f} (原始)")
                print(f"   加权感知损失: {wp_value:.6f} (λ={self.lambda_perceptual})")
                
            if 'cliap_loss' in loss_dict:
                print(f"   clip_loss   : {loss_dict['clip_loss']}")
                
            print(f"   总损失: {total_loss.item():.6f}")
            
            
            if text_embeds is not None:
                mean_norm = text_embeds.norm(dim=-1).mean().item()
                cos_sim = F.cosine_similarity(text_embeds[:-1], text_embeds[1:]).mean().item()
                print(f"\n[CLIP monitor] mean_norm={mean_norm:.3f}, cos_sim={cos_sim:.3f}")
            
            # 监控 ControlNet 注入特征的统计信息
            if control_features is not None:
                print(f"\n[ControlNet 特征监控]")
                print(f"{'层级':<10} {'Mean':<12} {'Std':<12} {'Abs Mean':<12} {'Shape'}")
                print("-" * 70)
                for level_name, features in control_features.items():
                    mean_val = features.mean().item()
                    std_val = features.std().item()
                    abs_mean_val = features.abs().mean().item()
                    shape_str = 'x'.join(map(str, features.shape[1:]))  # 不显示batch维度
                    print(f"{level_name:<10} {mean_val:>11.6f} {std_val:>11.6f} {abs_mean_val:>11.6f} [{shape_str}]")
                
                # 计算整体统计
                all_means = [f.mean().item() for f in control_features.values()]
                all_stds = [f.std().item() for f in control_features.values()]
                all_abs_means = [f.abs().mean().item() for f in control_features.values()]
                print("-" * 70)
                print(f"{'Overall':<10} {sum(all_means)/len(all_means):>11.6f} {sum(all_stds)/len(all_stds):>11.6f} {sum(all_abs_means)/len(all_abs_means):>11.6f} [avg]")
                print("")
        
        return total_loss, vae_features, loss_dict
    
    def save_model(self, save_path):
        state_dict = {
            'lora_target_modules': self.lora_target_modules,
            'lora_rank': self.lora_rank,
            'unet_state_dict': {},
            'vae_state_dict': {},
            'controlnet_state_dict': {},
            'clip_state_dict': {},
        }
        
        # 保存LoRA参数 + 门控参数 + CLIP投影层
        saved_counts = {'lora': 0, 'text_gate': 0, 'scene_gate': 0, 'control_gate': 0, 'proj': 0}
        print(f"\n{'='*70}")
        print(f"保存UNet参数:")
        print(f"{'='*70}")
        for name, param in self.unet.named_parameters():
            # LoRA参数
            if 'lora' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
                saved_counts['lora'] += 1
            # 门控参数（text_gate, scene_gate, control_gate等）
            elif 'gate' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
                if 'text_gate' in name:
                    saved_counts['text_gate'] += 1
                    print(f"  CLIP门控: {name} = {param.item():.6f}")
                elif 'scene_gate' in name:
                    saved_counts['scene_gate'] += 1
                    print(f"  Scene门控: {name} = {param.item():.6f}")
                elif 'control_gate' in name:
                    saved_counts['control_gate'] += 1
                    print(f"  Control门控: {name} = {param.item():.6f}")
            elif 'text_proj' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
                saved_counts['proj'] += 1
                print(f"  CLIP投影层: {name}, shape={param.shape}")
        
        print(f"\n{'='*70}")
        print(f"保存统计:")
        print(f"{'='*70}")
        print(f"  ✅ LoRA参数: {saved_counts['lora']} 个")
        total_gates = saved_counts['text_gate'] + saved_counts['scene_gate'] + saved_counts['control_gate']
        if total_gates > 0:
            print(f"  ✅ 门控参数: {total_gates} 个 (Text={saved_counts['text_gate']}, Scene={saved_counts['scene_gate']}, Control={saved_counts['control_gate']})")
        if saved_counts['proj'] > 0:
            print(f"  ✅ 投影层参数: {saved_counts['proj']} 个")
        print(f"  📦 总参数: {len(state_dict['unet_state_dict'])} 个")
        print(f"{'='*70}")
        
        # 保存VAE参数（如果有）
        if self.vae_encoder is not None:
            state_dict['vae_state_dict'] = self.vae_encoder.state_dict()
        
        # 保存完整ControlNet参数（如果有）
        if self.controlnet_full is not None:
            state_dict['controlnet_state_dict'] = self.controlnet_full.state_dict()
            # 显示ControlNet的gate参数值
            print(f"\nControlNet参数:")
            controlnet_gate_count = 0
            for name, param in self.controlnet_full.named_parameters():
                if 'control_gate' in name:
                    controlnet_gate_count += 1
                    print(f"   ControlNet门控: {name} = {param.item():.6f}")
            print(f" ControlNet总参数: {len(state_dict['controlnet_state_dict'])}, 其中gate参数: {controlnet_gate_count}")
            
        # 保存CLIP相关参数（如果启用）
        if self.use_clip:
            state_dict['use_clip'] = True
            state_dict['default_prompt'] = getattr(self, 'default_prompt', "A high resolution satellite image")
            
            # 如果CLIP模型被微调了，保存其状态
            if self.text_encoder is not None and any(p.requires_grad for p in self.text_encoder.parameters()):
                state_dict['clip_state_dict'] = self.text_encoder.state_dict()
        
        torch.save(state_dict, save_path)


class SSDiff_VAE_test(nn.Module):
    """测试/推理模型，支持VAE和ControlNet"""
    def __init__(self, args, use_multi_gpu=False):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_data_parallel = False  # 始终单卡
        print(f"当前运行设备: {self.device}")

        print(args.use_clip)
        if args.use_distillation:
            original_respacing = args.timestep_respacing
            args.timestep_respacing = ""
        
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        if args.use_distillation:
            args.timestep_respacing = original_respacing
        
        # 加载模型权重
        if hasattr(args, 'model_path') and args.model_path:
            state_dict = torch.load(args.model_path, map_location='cpu')
            
            if args.use_distillation and 'unet_state_dict' in state_dict:
                if hasattr(args, 'pretrained_ssdiff_path'):
                    base_state = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
                    missing_keys, _ = self.model.load_state_dict(base_state, strict=False)
                
                if 'lora_target_modules' in state_dict:
                    if 'lora_rank' in state_dict:
                        args.lora_rank = state_dict['lora_rank']
                    else:
                        args.lora_rank = 4
                    
                    self.model, _ = initialize_ssdiff_unet_with_lora(
                        args, pretrained_path=args.pretrained_ssdiff_path
                    )
                    
                    # 加载LoRA、门控参数和CLIP投影层
                    loaded_counts = {'lora': 0, 'text_gate': 0, 'scene_gate': 0, 'control_gate': 0, 'proj': 0}
                    print(f"\n{'='*70}")
                    print(f"加载UNet参数:")
                    print(f"{'='*70}")
                    for name, param in self.model.named_parameters():
                        if name in state_dict['unet_state_dict']:
                            param.data.copy_(state_dict['unet_state_dict'][name])
                            if 'lora' in name:
                                loaded_counts['lora'] += 1
                            elif 'text_gate' in name:
                                loaded_counts['text_gate'] += 1
                                print(f"  CLIP门控: {name} = {param.item():.6f}")
                            elif 'scene_gate' in name:
                                loaded_counts['scene_gate'] += 1
                                print(f"  Scene门控: {name} = {param.item():.6f}")
                            elif 'control_gate' in name:
                                loaded_counts['control_gate'] += 1
                                print(f"  Control门控: {name} = {param.item():.6f}")
                            elif 'text_proj' in name:
                                loaded_counts['proj'] += 1
                                print(f"  CLIP投影层: {name}, shape={param.shape}")
                    
                    print(f"\n{'='*70}")
                    print(f"加载统计:")
                    print(f"{'='*70}")
                    print(f"  ✅ LoRA参数: {loaded_counts['lora']} 个")
                    total_gates = loaded_counts['text_gate'] + loaded_counts['scene_gate'] + loaded_counts['control_gate']
                    if total_gates > 0:
                        print(f"  ✅ 门控参数: {total_gates} 个 (Text={loaded_counts['text_gate']}, Scene={loaded_counts['scene_gate']}, Control={loaded_counts['control_gate']})")
                    if loaded_counts['proj'] > 0:
                        print(f"  ✅ 投影层参数: {loaded_counts['proj']} 个")
                    print(f"{'='*70}\n")
            else:
                missing_keys, _ = self.model.load_state_dict(state_dict, strict=False)
        
        #  主模型移动到设备
        self.model.to(self.device)
        self.model.set_epoch(10001)
        self.model.eval()
        
# 直接初始化大型模块（单GPU不需要延迟）
        self.vae_encoder = None
        self.controlnet_full = None

# 初始化VAE编码器
        if getattr(args, "use_vae", False):
            print("初始化VAE编码器...")
            self.vae_encoder = VAEEncoder(
                input_channels=1,
            latent_dim=getattr(args, "vae_latent_dim", 256)
            ).to(self.device).eval()
            
            if hasattr(args, "model_path") and args.model_path:
                state_dict = torch.load(args.model_path, map_location="cpu")
                if "vae_state_dict" in state_dict:
                    self.vae_encoder.load_state_dict(state_dict["vae_state_dict"])
                    print("✓ 已加载VAE权重")
                    
            if hasattr(args, "model_path") and args.model_path:
                state_dict = torch.load(args.model_path, map_location="cpu")
                if "vae_state_dict" in state_dict:
                    self.vae_encoder.load_state_dict(state_dict["vae_state_dict"])
                print("✓ 已加载VAE权重")

        # 初始化ControlNet
        if getattr(args, "use_controlnet", False):
            print("\n初始化ControlNet...")
            # 在测试时，verbose=False避免输出过多初始化信息
            # 因为权重会从checkpoint加载而不是使用初始化的权重
            self.controlnet_full = ControlNet(
                unet_model=self.model,
                ms_dim=getattr(args, "ms_channels", 8),
                pan_dim=1,
                copy_weights=True,  # 保持复制权重（作为backup）
                verbose=False  # 测试时不输出详细的复制信息
            ).to(self.device).eval()
            if hasattr(args, "model_path") and args.model_path:
                state_dict = torch.load(args.model_path, map_location="cpu")
                if "controlnet_state_dict" in state_dict:
                    self.controlnet_full.load_state_dict(state_dict["controlnet_state_dict"])
                    print("✓ 已从checkpoint加载ControlNet训练权重")
                    # 显示加载的gate参数值
                    controlnet_gate_count = 0
                    for name, param in self.controlnet_full.named_parameters():
                        if 'control_gate' in name:
                            controlnet_gate_count += 1
                            print(f" ControlNet门控: {name} = {param.item():.6f}")
                    if controlnet_gate_count > 0:
                        print(f"  🎉 成功加载 {controlnet_gate_count} 个ControlNet门控参数")
        # 初始化CLIP文本编码器（独立）
        self.tokenizer = None
        self.text_encoder = None
        self.use_clip = False
        self.use_ram = False  #  新增：RAM支持
        self.ram_caption_generator = None
        if hasattr(args, 'use_clip') and args.use_clip:
            # 使用SD21Base作为CLIP基础模型路径
            clip_base_path = getattr(args, 'clip_model_path', None)
            if clip_base_path is None:
                # 默认使用SD21Base路径
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                clip_base_path = os.path.join(script_dir, "model", "SD21Base")
            
            print(f"正在从SD21Base加载CLIP模型: {clip_base_path}")
            try:
                # 从SD21Base加载tokenizer和text_encoder（预训练权重）
                self.tokenizer = AutoTokenizer.from_pretrained(
                    clip_base_path, subfolder="tokenizer"
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    clip_base_path, subfolder="text_encoder"
                ).to(self.device)
                self.text_encoder.eval()
                
                self.use_clip = True
                # 尝试从checkpoint加载保存的prompt配置（不加载权重，因为CLIP在训练时是冻结的）
                
                print(f"  - 注意: CLIP在训练时是冻结的，使用SD21Base预训练权重")
                    
            except Exception as e:
                print(f"⚠ CLIP模型初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self.use_clip = False
                self.tokenizer = None
                self.text_encoder = None
        
        #  RAM Caption生成器（用于从PAN图像自动生成caption）
        if hasattr(args, 'use_ram') and args.use_ram:
            self.use_ram = True
            ram_model_path = getattr(args, 'ram_model_path', None)
            
            try:
                # =============================
                # 初始化 RAM caption 生成器（使用自定义封装）
                # =============================
                device = self.device

                # 初始化生成器
                self.ram_caption_generator = RAMCaptionGenerator(
                    model_path=ram_model_path,
                    device=device
                )

                print(f"✓ 已启用RAM Caption生成器（Swin-Large 模型）")
                print(f"  - 自动从PAN图像生成caption")
                print(f"  - 模型路径: {ram_model_path}")

            except Exception as e:
                print(f"⚠ RAM模型加载失败: {e}")
                self.use_ram = False
                self.ram_caption_generator = None
    
    def _build_color_subspace(self):
        """
        构建颜色子空间的正交基（用于颜色去偏）
        Returns:
            Q: 颜色子空间的正交基 [D, K]，其中K是颜色词数量
        """
        color_words = [
            "green", "blue", "red", "brown", "yellow", "white", "gray", "grey",
            "black", "purple", "pink", "orange", "cyan", "magenta",
            "light", "dark", "pale", "bright", "deep"
        ]
        
        device = next(self.text_encoder.parameters()).device
        
        # 编码颜色词
        with torch.no_grad():
            color_embeds_list = []
            for color in color_words:
                text_input = self.tokenizer(
                    color,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                color_embed = self.text_encoder(text_input.input_ids)[0]  # [1, 77, 768]
                color_embed = color_embed.mean(dim=1)  # [1, 768]
                color_embeds_list.append(color_embed)
            
            # 堆叠所有颜色embedding: [K, D]
            color_embeds = torch.cat(color_embeds_list, dim=0)
            # 归一化
            color_embeds = F.normalize(color_embeds, dim=-1)
            
            # QR分解构建正交基: Q @ R = color_embeds.T
            Q, _ = torch.linalg.qr(color_embeds.T)  # Q: [D, K]
        
        return Q
    
    def _apply_color_debiasing(self, text_embeds, Q):
        """
        对text embedding应用颜色子空间去偏
        Args:
            text_embeds: CLIP text embeddings [B, D]
            Q: 颜色子空间的正交基 [D, K]
        Returns:
            debiased_embeds: 去除颜色分量后的embeddings [B, D]
        """
        # 投影到颜色子空间: proj_color = (Q @ Q.T) @ text_embeds.T
        proj_color = Q @ (Q.T @ text_embeds.T)  # [D, B]
        
        # 去除颜色分量
        text_debiased = text_embeds.T - proj_color  # [D, B]
        text_debiased = text_debiased.T  # [B, D]
        
        # 归一化到unit sphere
        text_debiased = F.normalize(text_debiased, dim=-1)
        
        return text_debiased
    
    def select_clip_semantic_label(self, ram_caption, top_k=3, max_ram_tags=12):
        """
        使用CLIP对RAM生成的caption进行语义过滤（标准版本）
        Args:
            ram_caption: RAM生成的原始caption（逗号分隔的标签列表）
            top_k: 选择前k个最相关的标签（默认3）
            max_ram_tags: RAM输出中保留的最大标签数（避免过长导致截断）
        Returns:
            topk_labels: 过滤后的标签列表
        """
        labels = [
            "urban area", "residential zone", "industrial zone", "forest", "grassland",
            "mountain", "river", "lake", "sea", "beach", "harbor", "farmland",
            "road network", "bare soil", "wetland", "desert",
            "bridge", "building", "highway", "stadium",
            "coastline", "cliff", "hill slope", "dense vegetation", "open water",
            "cloudy", "coastal city", "harbor town"
        ]
        
        # 预处理RAM的输出：RAM按概率从高到低输出标签，我们只保留前N个
        if isinstance(ram_caption, str):
            ram_tags = [tag.strip() for tag in ram_caption.split(',')]
            ram_tags = ram_tags[:max_ram_tags]
            ram_caption_short = ', '.join(ram_tags)
        else:
            ram_caption_short = ram_caption
        
        # 将RAM输出与标签一并tokenize
        device = self.device
        text_inputs = self.tokenizer(
            labels + [ram_caption_short],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # 提取文本特征
            text_embeds = self.text_encoder(text_inputs.input_ids)[0]  # [len(labels)+1, 77, 768]
            text_embeds = text_embeds.mean(dim=1)  # 全局平均池化 -> [len(labels)+1, 768]
            text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)
            
            # 计算余弦相似度：labels与ram_caption的相似度
            label_embeds = text_embeds[:-1]  # [num_labels, 768]
            caption_embed = text_embeds[-1:]  # [1, 768]
            scores = (label_embeds * caption_embed).sum(dim=-1)  # [num_labels]
            
        # 取前 top_k 个标签
        topk_idx = torch.topk(scores, k=min(top_k, len(labels))).indices.cpu().numpy()
        topk_labels = [labels[i] for i in topk_idx]
        
        return topk_labels
    
    def select_clip_semantic_label_with_color_debiasing(self, ram_caption, top_k=5):
        """
        对RAM生成的caption进行颜色去偏，直接返回去偏后的标签
        
        实现原理（Color Debiasing）：
        1. 构建颜色词的CLIP embedding构成的颜色子空间
        2. 通过QR分解得到颜色子空间的正交基 Q
        3. 对RAM每个标签的embedding去除其在颜色子空间的投影：e' = e - Q @ (Q.T @ e)
        4. 根据去偏后的embedding之间的相似度，保留最不相关颜色的前k个标签
        5. 返回去偏后的标签（保持原始文本）
        
        Args:
            ram_caption: RAM生成的原始caption（逗号分隔的标签列表）
            top_k: 返回前k个标签（默认5）
        Returns:
            debiased_labels: 去偏后的前k个标签列表
        """
        # 解析RAM输出
        if isinstance(ram_caption, str):
            ram_tags = [tag.strip() for tag in ram_caption.split(',')]
        else:
            ram_tags = ram_caption
        
        # 如果RAM输出少于等于top_k个，直接返回全部
        if len(ram_tags) <= top_k:
            return ram_tags[:top_k]
        
        # 编码所有RAM标签
        device = self.device
        text_inputs = self.tokenizer(
            ram_tags,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # 提取文本特征
            text_embeds = self.text_encoder(text_inputs.input_ids)[0]  # [num_tags, 77, 768]
            text_embeds = text_embeds.mean(dim=1)  # 全局平均池化 -> [num_tags, 768]
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            # 构建颜色子空间
            Q = self._build_color_subspace()  # [768, K]
            
            # 对每个标签进行颜色去偏
            tag_embeds_debiased = self._apply_color_debiasing(text_embeds, Q)  # [num_tags, 768]
            
            # 计算每个标签去偏前后的相似度（相似度越低说明颜色成分越多）
            color_scores = (text_embeds * tag_embeds_debiased).sum(dim=-1)  # [num_tags]
            
            # 选择颜色成分最少的前k个标签（相似度最高的）
            # 或者直接按原始顺序返回前k个
            topk_idx = torch.topk(color_scores, k=min(top_k, len(ram_tags))).indices.cpu().numpy()
            
        # 返回去偏后的标签（保持原始文本）
        debiased_labels = [ram_tags[i] for i in topk_idx]
        
        return debiased_labels
        
    def encode_prompt(self, prompt_batch):
        """
        编码文本提示为文本嵌入
        Args:
            prompt_batch: 文本提示列表
        Returns:
            prompt_embeds: 文本嵌入 [B, 77, 768]
        """
        print("[DEBUG: encode_prompt]")
        if hasattr(self, "ram_caption_generator") and self.ram_caption_generator is not None:
            print("RAM Caption Generator 已加载，当前批次输入:")
            for i, p in enumerate(prompt_batch[:3]):
                print(f"  [{i}] prompt: {p}")
        if not self.use_clip or self.tokenizer is None or self.text_encoder is None:
            print("[DEBUG: encode_prompt]")
            print("  self.use_clip:", self.use_clip)
            print("  self.tokenizer:", type(self.tokenizer))
            print("  self.text_encoder:", type(self.text_encoder))
            print("  self.ram_caption_generator:", type(self.ram_caption_generator))
            return None
            
        prompt_embeds_list = []
        with torch.no_grad():  # 不计算梯度，确认不训练
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, 
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device)
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        
        if prompt_embeds_list:
            prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
            prompt_embeds = F.layer_norm(prompt_embeds, prompt_embeds.shape[-1:])
            return prompt_embeds
        return None
    
    @torch.no_grad()
    def forward(self, lms, pan, ms, prompt=None):
        self.model.training = False
        self.model.eval()
        
        lms = lms.to(self.device)
        pan = pan.to(self.device)
        ms = ms.to(self.device)
        
        # 处理文本提示
        text_embeds = None
        batch_size = lms.shape[0]
        if self.use_clip and self.text_encoder is not None:
            # 如果启用RAM，为当前batch的每张图像生成caption
            if self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]  # 使用PAN图像生成caption
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions
                    print(f"[DEBUG] 为当前batch生成的captions: {prompt[:3] if len(prompt) > 3 else prompt}")
            elif prompt is None:
                # 如果没有提供prompt且没有启用RAM，使用默认prompt
                prompt = [self.default_prompt] * batch_size
                
            # 编码文本提示
            text_embeds = self.encode_prompt(prompt)
        
        model_kwargs = {"lms": lms, "pan": pan, "ms": ms, "encoder_hidden_states": text_embeds}
        
        if self.args.use_distillation:
            first_patch = [True]
            
            def direct_forward_fn(model_output, t, lms_input, **kwargs):
                if self.args.predict_xstart:
                    pred_xstart = model_output
                else:
                    x_t = kwargs.get('noise', None)
                    if x_t is None:
                        x_t = torch.zeros_like(lms_input)
                    
                    if x_t.shape != lms_input.shape:
                        x_t = torch.zeros_like(lms_input)
                    
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 0:
                            t_batch = t.to(device=lms_input.device, dtype=torch.long).expand(lms_input.shape[0])
                        else:
                            t_batch = t.to(device=lms_input.device, dtype=torch.long)
                    else:
                        t_batch = torch.full((lms_input.shape[0],), 999, device=lms_input.device, dtype=torch.long)
                    pred_xstart = self.diffusion._predict_xstart_from_eps(x_t, t_batch, model_output)
                
                output = lms_input + pred_xstart
                return {"sample": output, "pred_xstart": pred_xstart}
            
            xt = torch.zeros_like(lms)
            
            if hasattr(self.args, 'test_with_noise') and self.args.test_with_noise:
                noise = torch.randn_like(lms)
                t_full = torch.full((lms.shape[0],), 99, device=self.device, dtype=torch.long)
                xt = self.diffusion.q_sample_xt(torch.zeros_like(lms), t_full, noise=noise)
            
            # VAE编码器：提取全局场景特征（轻量级，可以在整图上运行）
            vae_features = None
            if self.vae_encoder is not None:
                mu, logvar = self.vae_encoder(pan)
                vae_features = mu
            
            batch_size = lms.shape[0]
            timesteps = torch.full((batch_size,), 99, device=self.device, dtype=torch.long)
            
            #  ControlNet：不在这里提取特征，而是在forward_chop_distill内部逐patch提取
            # 这样可以避免对整图计算ControlNet导致OOM
            # control_features = None
            # if self.controlnet_full is not None:
            #     control_features = self.controlnet_full(pan, xt, timesteps)
            
            #  将controlnet_full传递给forward_chop_distill，让它内部逐patch计算
            output = self.model.forward_chop_distill(
                lms, pan, ms, xt,
                sample_fn=direct_forward_fn,
                scene_token=vae_features,        # VAE的全局特征
                controlnet=self.controlnet_full,  #  传递ControlNet对象而不是特征
                encoder_hidden_states=text_embeds,  # CLIP的文本特征
                noise=xt,
            )
        else:
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
        
        output = output.clamp(0, 1)
        
        return output
