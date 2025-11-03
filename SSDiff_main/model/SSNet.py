import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from model.fusformer import Fusformer
from model.nn import (
    SiLU,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from model.ARConv import ARConv
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        model_channels=128,
        norm_type="gn",
        dropout=0.0,
        dims=2,
        use_scale_shift_norm=False,
        use_arconv=False,
        arconv_hw_range=[1,9]
        
    ):
        super().__init__()
        self.use_arconv = use_arconv
        self.arconv_hw_range = arconv_hw_range
        if use_arconv:
            self.conv0 = ARConv(in_channels, hidden_channels, 3, 1, 1)
            self.conv1 = ARConv(hidden_channels, out_channels, 3, 1, 1)
        else:
            self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
            self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        emb_channels = model_channels * 4     # model_channel * 4
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                # out_channels,
                2 * out_channels if use_scale_shift_norm else out_channels
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(norm_type, out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, out_channels, out_channels, 3, padding=1)
            ),
        )
        
    def time_emb(self, h, emb):
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return h
    
    
    def forward(self, x, emb,epoch=0):         # 32 64 64
        if self.use_arconv:
            rs1 = self.relu(self.conv0(x, epoch, self.arconv_hw_range))
            rs1 = self.conv1(rs1, epoch, self.arconv_hw_range)
        else:
            rs1 = self.relu(self.conv0(x))
            rs1 = self.conv1(rs1)
            
        rs1 = self.time_emb(rs1, emb)
        rs = torch.add(x, rs1)
        return rs


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x = x1 + x2
        return self.relu(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


from patch_merge_module.import_module import PatchMergeModule
class SSNet(PatchMergeModule):
    def __init__(
        self,
        dim,        # 32
        model_channels=128,
        dim_head=16,
        ms_dim=8,
        pan_dim=1,
        se_ratio_mlp=0.5,
        se_ratio_rb=0.5,
        dropout=0,
        device='cpu',
        norm_type="bn",
        crop_batch_size=1,
        use_scale_shift_norm=False
    ):
        super().__init__(True, crop_batch_size, [64, 64, 16], device=device)
        ms_dim = ms_dim  # qb:4, gf2,  wv3:8
        # lms_dim = ms_dim
        pan_dim = pan_dim
        self.model_channels = model_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.device = device
        self.current_epoch = 0
        self.relu = nn.LeakyReLU()
        self.upsample = Upsample(ms_dim)
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),    # in_channels, out_channels, kernel_size, stride,padding
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.conv_lms2x_t = nn.Conv2d(ms_dim*2, dim, 3, 1, 1)    # 16 32
        self.conv_pan2x_t = nn.Conv2d(ms_dim+pan_dim, dim, 3, 1, 1)    # 9 32
        self.conv_x_t = nn.Conv2d(ms_dim, dim, 3, 1, 1)  # 8 32
        
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        # layer 0
        self.fusformer0 = Fusformer(dim0, dim0//dim_head, dim_head, int(dim0*se_ratio_mlp))
        self.down0 = Down(dim0, dim1)
        self.resblock0 = ResBlock(dim0, int(se_ratio_rb*dim0), dim0,
                                  model_channels=self.model_channels, use_scale_shift_norm=use_scale_shift_norm,use_arconv=True,arconv_hw_range=[1,9]) # 32 16 32 128
 
        # layer 1
        self.fusformer1 = Fusformer(dim1, dim1//dim_head, dim_head, int(dim1*se_ratio_mlp))
        self.down1 = Down(dim1, dim2)
        self.resblock1 = ResBlock(dim1, int(se_ratio_rb*dim1), dim1, use_scale_shift_norm=use_scale_shift_norm,use_arconv=True,arconv_hw_range=[1,9])

        # layer 2
        self.fusformer2 = Fusformer(dim2, dim2//dim_head, dim_head, int(dim2*se_ratio_mlp))
        self.up0 = Up(dim2, dim3)
        self.resblock2 = ResBlock(dim2, int(se_ratio_rb*dim2), dim2, use_scale_shift_norm=use_scale_shift_norm,use_arconv=True,arconv_hw_range=[1,9])

        # layer 3
        self.fusformer3 = Fusformer(dim3, dim3//dim_head, dim_head, int(dim3*se_ratio_mlp))
        self.up1 = Up(dim3, dim4)
        self.resblock3 = ResBlock(dim3, int(se_ratio_rb*dim3), dim3, use_scale_shift_norm=use_scale_shift_norm,use_arconv=True,arconv_hw_range=[1,9])

        # layer 4
        self.fusformer4 = Fusformer(dim4, dim4//dim_head, dim_head, int(dim4*se_ratio_mlp))
        # self.fusformer4 = Fusformer(dim4, dim4//8, 8, int(dim4*se_ratio_mlp))

        self.out_layers_pan = nn.Sequential(
            normalization(norm_type, dim),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(2, dim, dim, 3, padding=1)
            ),
        )
        
        self.out_layers_lms = nn.Sequential(
            normalization(norm_type, dim),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(2, dim, dim, 3, padding=1)
            ),
        )
        
        emb_channels = self.model_channels * 4
        self.emb_layers_pan = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                # 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                2 * dim if use_scale_shift_norm else dim
            ),
        )
        
        self.emb_layers_lms = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                # 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                2 * dim if use_scale_shift_norm else dim,
            ),
        )
        
        
    def time_emb_pan(self, h, emb): # 24 32 64 64, 24 512
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
    
    def time_emb_lms(self, h, emb):
        emb_out = self.emb_layers_lms(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers_lms[0], self.out_layers_lms[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers_lms(h)
        return h
    
    def Fourier_filter(self, x, threshold, scale):
        import torch.fft as fft
        dtype = x.dtype
        x = x.type(torch.float32)
        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        
        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W)).to(x.device)

        crow, ccol = H // 2, W //2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
        
        x_filtered = x_filtered.type(dtype)
        return x_filtered

    def forward_impl(self, lms, pan, ms, x_t, timesteps):    # x:lms , y:pan
        

        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        
        x = self.upsample(ms)
        skip_c0 = x
        
        # 🔥 在cat之前打印形状
        if not hasattr(self, '_debug_cat_printed'):
            self._debug_cat_printed = True
            print(f"\n[Step 4] 准备 cat 操作:")
            print(f"  pan.shape (before cat) = {pan.shape}")
            print(f"  x_t.shape (before cat) = {x_t.shape}")
            print(f"  尝试执行: torch.cat([pan, x_t], dim=1)")
        
        pan = torch.cat([pan, x_t], dim=1)  # 9 64 64
        pan = self.conv_pan2x_t(pan)    # 32 64 64
        
        lms = torch.cat([lms, x_t], dim=1)  # 16 64 64
        lms = self.conv_lms2x_t(lms)    # 32 64 64
        
        y = self.time_emb_pan(pan, emb)   # 32 64 64
        x = self.time_emb_lms(lms, emb)   # 32 64 64
        
        # gf5
        b1=1.2
        b2=1.4
        b3=1.6
        s1=0.9
        s2=0.6
        s3=0.3
        # layer 0
        x[:,:16] = x[:,:16] * b1
        y0 = self.Fourier_filter(y, threshold=1, scale=s1)
        # y0 = y
        x = self.fusformer0(x, y0)  # 32 64 64
        
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        
        y = self.resblock0(y, emb, epoch=self.current_epoch)  # 32 64 64
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x[:,:32] = x[:,:32] * b2
        y1 = self.Fourier_filter(y, threshold=1, scale=s2)
        # y1 = y
        x = self.fusformer1(x, y1)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        
        
        y = self.resblock1(y, emb,epoch=self.current_epoch)  # 64 32 32
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x[:,:64] = x[:,:64] * b3   # 128
        y2 = self.Fourier_filter(y, threshold=1, scale=s3)
        x = self.fusformer2(x, y2)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        
        y = self.resblock2(y, emb,epoch=self.current_epoch)  # 128 16 16
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x[:,:32] = x[:,:32] * b2
        y3 = self.Fourier_filter(y, threshold=1, scale=s2)
        x = self.fusformer3(x, y3)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        
        y = self.resblock3(y, emb,epoch=self.current_epoch)  # 64 32 32
        y = self.up1(y, skip_c11)  # 32 64 64

        # layer 4
        x[:,:16] = x[:,:16] * b1
        y4 = self.Fourier_filter(y, threshold=1, scale=s1)
        x = self.fusformer4(x, y4)  # 32 64 64


        output = self.to_hrms(x) + skip_c0  # 8 64 64

        
        return output
    def set_epoch(self, epoch):
        """供训练循环调用来更新当前epoch"""
        self.current_epoch = epoch
    def sample(self, 
               *args,
               **kwargs
               ):
        sample_fn = kwargs.pop('sample_fn')

        noise = kwargs['noise']
        shape = kwargs.pop('shape')
        device = args[0].device
        num_batch = args[0].shape[0]
        lms = args[0]  # 获取lms patch
        
        if noise is not None:
            img = noise
        else:
            # 默认从随机噪声开始，但对于单步蒸馏，应该从lms开始
            # 如果num_timesteps=1，从lms开始（用于蒸馏）
            if kwargs.get('num_timesteps', 1000) == 1:
                img = lms.clone()  # 从lms patch开始（与训练一致）
            else:
                img = torch.randn(*shape, device=device)  # 随机噪声
        img = img[:num_batch, ...]
        kwargs['noise'] = img
        
        self.indices = list(range(kwargs.pop('num_timesteps')))[::-1]
        if kwargs['progress'] and not hasattr(self, 'indices'):
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            self.indices = tqdm(self.indices)
        
        # lms已经在前面定义了，不需要再定义
        for i in self.indices:
            t = torch.tensor([i] * shape[0], device=device)
            t = t[:num_batch, ...]          
            # out_patch[:, [4,2,0]]  
            out_patch = self.forward_impl(*args, x_t=img, timesteps=t)
            out_patch = sample_fn(out_patch, t, lms,
                                **kwargs)
            img = out_patch['sample']
            kwargs['noise'] = img

        return img.contiguous()
    
    def forward_chop_distill(self, lms, pan, ms, xt, sample_fn, patch_size=64, batch_size=8, **kwargs):
        """
        专门用于单步蒸馏的 forward_chop 实现，支持大图像的 patch 切分
        
        Args:
            lms: [B, 8, H, W] 低分辨率MS上采样
            pan: [B, 1, H, W] 全色图像
            ms: [B, 8, h, w] 原始低分辨率MS（h=H/4, w=W/4）
            xt: [B, 8, H, W] 噪声残差
            sample_fn: 回调函数
            patch_size: patch 大小（默认64）
            batch_size: 每次处理的 patch 数量（默认8）
        """
        import torch.nn.functional as F
        
        print(f"\n[DEBUG] 进入 SSNet.forward_chop_distill")
        print(f"  输入形状: lms={lms.shape}, pan={pan.shape}, ms={ms.shape}, xt={xt.shape}")
        
        B, C_lms, H, W = lms.shape
        device = lms.device
        
        # 如果图像小于等于 patch_size，直接处理
        if H <= patch_size and W <= patch_size:
            print(f"  小图像，直接处理")
            timesteps = torch.full((B,), 999, device=device, dtype=torch.long)
            model_output = self.forward_impl(lms, pan, ms, x_t=xt, timesteps=timesteps)
            # 更新 kwargs 中的 noise
            kwargs_copy = kwargs.copy()
            kwargs_copy['noise'] = xt
            result = sample_fn(model_output, timesteps, lms, **kwargs_copy)
            return result['sample']
        
        # 大图像：进行 patch 切分
        print(f"  大图像 ({H}x{W})，进行 patch 切分（patch_size={patch_size}）")
        
        # 计算 padding
        stride = patch_size // 2
        pad_h = (stride - (H - patch_size) % stride) % stride
        pad_w = (stride - (W - patch_size) % stride) % stride
        
        # Padding 输入
        lms_pad = F.pad(lms, (0, pad_w, 0, pad_h), mode='reflect')
        pan_pad = F.pad(pan, (0, pad_w, 0, pad_h), mode='reflect')
        xt_pad = F.pad(xt, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_pad, W_pad = lms_pad.shape[2:]
        print(f"  Padding后: {H_pad}x{W_pad}")
        
        # 🔥 关键修复：ms 也需要切分！
        # ms 的尺寸是 lms 的 1/4，所以 patch_size 也要除以 4
        ms_patch_size = patch_size // 4
        ms_stride = stride // 4
        
        # Padding ms
        _, _, h, w = ms.shape
        ms_pad_h = (ms_stride - (h - ms_patch_size) % ms_stride) % ms_stride
        ms_pad_w = (ms_stride - (w - ms_patch_size) % ms_stride) % ms_stride
        ms_pad = F.pad(ms, (0, ms_pad_w, 0, ms_pad_h), mode='reflect')
        
        # 使用 unfold 切分 patches
        lms_patches = F.unfold(lms_pad, kernel_size=patch_size, stride=stride)  # [B, C*ps*ps, num_patches]
        pan_patches = F.unfold(pan_pad, kernel_size=patch_size, stride=stride)
        xt_patches = F.unfold(xt_pad, kernel_size=patch_size, stride=stride)
        ms_patches = F.unfold(ms_pad, kernel_size=ms_patch_size, stride=ms_stride)  # ms 的 patch 更小
        
        num_patches = lms_patches.shape[2]
        print(f"  切分成 {num_patches} 个 patches (lms: {patch_size}x{patch_size}, ms: {ms_patch_size}x{ms_patch_size})")
        
        # Reshape patches: [B, C, ps, ps, num_patches] -> [num_patches, C, ps, ps]
        lms_patches = lms_patches.view(B, C_lms, patch_size, patch_size, num_patches).permute(4, 0, 1, 2, 3).squeeze(1)
        pan_patches = pan_patches.view(B, 1, patch_size, patch_size, num_patches).permute(4, 0, 1, 2, 3).squeeze(1)
        xt_patches = xt_patches.view(B, C_lms, patch_size, patch_size, num_patches).permute(4, 0, 1, 2, 3).squeeze(1)
        ms_patches = ms_patches.view(B, C_lms, ms_patch_size, ms_patch_size, num_patches).permute(4, 0, 1, 2, 3).squeeze(1)
        
        # 处理每个 patch
        output_patches = []
        timesteps = torch.full((batch_size,), 999, device=device, dtype=torch.long)
        
        for i in range(0, num_patches, batch_size):
            end_idx = min(i + batch_size, num_patches)
            curr_batch = end_idx - i
            
            lms_batch = lms_patches[i:end_idx]
            pan_batch = pan_patches[i:end_idx]
            xt_batch = xt_patches[i:end_idx]
            ms_batch = ms_patches[i:end_idx]  # 🔥 使用对应的 ms patch
            
            # 调整 timesteps 大小
            ts_batch = timesteps[:curr_batch]
            
            # 调用 forward_impl，传入对应的 ms patch
            model_output = self.forward_impl(lms_batch, pan_batch, ms_batch, x_t=xt_batch, timesteps=ts_batch)
            
            # 调用回调（注意：不要重复传 noise，已经在 kwargs 中了）
            # 临时更新 kwargs 中的 noise 为当前 batch 的 xt
            kwargs_batch = kwargs.copy()
            kwargs_batch['noise'] = xt_batch
            result = sample_fn(model_output, ts_batch, lms_batch, **kwargs_batch)
            output_patches.append(result['sample'])
        
        # 合并 patches
        output_patches = torch.cat(output_patches, dim=0)  # [num_patches, C, ps, ps]
        
        # Reshape回去: [num_patches, C, ps, ps] -> [B, C*ps*ps, num_patches]
        output_patches = output_patches.unsqueeze(1).permute(1, 2, 3, 4, 0).reshape(B, -1, num_patches)
        
        # 使用 fold 重建图像
        output = F.fold(output_patches, output_size=(H_pad, W_pad), kernel_size=patch_size, stride=stride)
        
        # 计算重叠次数并平均
        ones = torch.ones_like(lms_pad)
        ones_patches = F.unfold(ones, kernel_size=patch_size, stride=stride)
        divisor = F.fold(ones_patches, output_size=(H_pad, W_pad), kernel_size=patch_size, stride=stride)
        output = output / divisor
        
        # 裁剪回原始尺寸
        output = output[:, :, :H, :W]
        
        print(f"  输出形状: {output.shape}")
        return output

    
def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


def split_tensor(tensor, len):
    
    split_len = int(len/2)
    b = torch.split(tensor, split_len, dim=2)
    c1 = torch.split(b[0], split_len, dim=3)
    c2 = torch.split(b[1], split_len, dim=3)
    return [split_tensor(c1[0], split_len), split_tensor(c1[1], split_len),
            split_tensor(c2[0], split_len), split_tensor(c2[1], split_len)]

def concat_tensor(tensors):
    c1 = torch.cat((tensors[0], tensors[1]), 3)
    c2 = torch.cat((tensors[2], tensors[3]), 3)
    
    print(c1)
    print(c2)
    c = torch.cat((c1, c2), 2)
    print(c)
    return c
    

if __name__ == "__main__":
    import torch.nn.functional as F
    import einops
    import time
    from torch.cuda import max_memory_allocated
    device = 'cuda:1'
    model = SSNet(32).to(device)
    model.forward = model.forward_impl

    lms = torch.randn(1, 8, 256, 256).to(device)
    pan = torch.randn(1, 1, 256, 256).to(device)
    ms = torch.randn(1, 8, 64, 64).to(device)
    x_t = torch.randn(1, 8, 256, 256).to(device)
    t = torch.tensor([1.]).to(device)

    tic = time.time()
    model.eval()
    for _ in range(2000):
        sr = model.forward(lms, pan, ms, x_t, t)
    print(time.time() - tic)