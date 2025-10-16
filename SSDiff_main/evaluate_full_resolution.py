#!/usr/bin/env python3
"""
评估 Full-Resolution 全景锐化结果
支持评估 D_lambda、D_s 和 HQNR 指标（无参考评估）

参考文献:
[Alparone08] L. Alparone, et al., "Multispectral and panchromatic data fusion assessment without reference,"
             Photogrammetric Engineering and Remote Sensing, 2008.
[Khan09]     M. M. Khan, et al., "Pansharpening quality assessment using the modulation transfer functions of instruments",
             IEEE Trans. Geosci. Remote Sens., 2009.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
from scipy.ndimage import convolve, zoom
from scipy.signal import convolve2d
import cv2


def uqi(img1, img2):
    """
    Universal Quality Index (Q-index)
    Wang, Z., & Bovik, A. C. (2002). A universal image quality index.
    
    Args:
        img1: 第一张图像 (2D)
        img2: 第二张图像 (2D)
    
    Returns:
        Q-index 值
    """
    img1 = img1.astype(np.float64).flatten()
    img2 = img2.astype(np.float64).flatten()
    
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    
    var1 = np.var(img1, ddof=1)
    var2 = np.var(img2, ddof=1)
    cov12 = np.cov(img1, img2)[0, 1]
    
    Q = 4 * cov12 * m1 * m2 / ((var1 + var2) * (m1**2 + m2**2) + np.finfo(np.float64).eps)
    
    return Q


def blockproc_uqi(img1, img2, block_size):
    """
    对图像进行分块处理，计算每个块的 UQI
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        block_size: 块大小
    
    Returns:
        Q-index map
    """
    H, W = img1.shape
    
    # 确保图像尺寸是块大小的倍数
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(f"图像尺寸必须是块大小 {block_size} 的倍数")
    
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    
    q_map = np.zeros((num_blocks_h, num_blocks_w))
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block1 = img1[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block2 = img2[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            q_map[i, j] = uqi(block1, block2)
    
    return q_map


def gen_mtf_filter(ratio, sensor, num_bands):
    """
    生成 MTF 滤波器
    
    Args:
        ratio: MS 和 PAN 之间的缩放比例
        sensor: 传感器类型 ('WV3', 'WV2', 'QB', 'GF2', etc.)
        num_bands: 光谱波段数
    
    Returns:
        MTF 滤波器组 (N, N, num_bands)
    """
    # MTF GNyq 值
    if sensor == 'QB':
        GNyq = np.array([0.34, 0.32, 0.30, 0.22])
    elif sensor == 'IKONOS':
        GNyq = np.array([0.26, 0.28, 0.29, 0.28])
    elif sensor in ['GeoEye1', 'WV4']:
        GNyq = np.array([0.23, 0.23, 0.23, 0.23])
    elif sensor == 'WV2':
        GNyq = np.concatenate([0.35 * np.ones(7), [0.27]])
    elif sensor == 'WV3':
        GNyq = np.array([0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315])
    elif sensor == 'GF2':
        GNyq = 0.3 * np.ones(num_bands)
    else:
        GNyq = 0.3 * np.ones(num_bands)
    
    # 生成高斯滤波器
    N = 41
    h = np.zeros((N, N, num_bands))
    fcut = 1.0 / ratio
    
    for i in range(num_bands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[i])))
        # 创建高斯核
        ax = np.arange(-N // 2 + 1, N // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * alpha ** 2))
        kernel = kernel / np.max(kernel)
        h[:, :, i] = kernel
    
    return h


def mtf_filter(img, sensor, ratio):
    """
    使用 MTF 滤波器对图像进行滤波
    
    Args:
        img: 输入图像 (H, W, C)
        sensor: 传感器类型
        ratio: 缩放比例
    
    Returns:
        滤波后的图像
    """
    H, W, C = img.shape
    h = gen_mtf_filter(ratio, sensor, C)
    
    img_filtered = np.zeros_like(img, dtype=np.float64)
    for i in range(C):
        img_filtered[:, :, i] = convolve2d(img[:, :, i], h[:, :, i], mode='same', boundary='wrap')
    
    return img_filtered


def interp23tap(img, ratio):
    """
    使用 23-tap 多项式插值器对图像进行插值
    简化版本：使用双三次插值代替
    
    Args:
        img: 输入图像 (H, W) 或 (H, W, C)
        ratio: 上采样比例
    
    Returns:
        插值后的图像
    """
    if img.ndim == 2:
        return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    else:
        H, W, C = img.shape
        img_interp = np.zeros((H * ratio, W * ratio, C), dtype=img.dtype)
        for i in range(C):
            img_interp[:, :, i] = cv2.resize(img[:, :, i], None, fx=ratio, fy=ratio, 
                                            interpolation=cv2.INTER_CUBIC)
        return img_interp


def d_lambda(sr, ms_lr, ms, block_size=32, ratio=4, p=1):
    """
    D_lambda 指标：光谱失真指数
    
    Args:
        sr: 全景锐化结果 (H, W, C)
        ms_lr: 原始低分辨率 MS 图像 (H/ratio, W/ratio, C)
        ms: MS 图像上采样到 PAN 尺度 (H, W, C)
        block_size: 块大小
        ratio: 分辨率比例
        p: 指数参数
    
    Returns:
        D_lambda 指标值
    """
    if sr.shape != ms.shape:
        raise ValueError("SR 和 MS 图像必须具有相同的尺寸")
    
    H, W, C = sr.shape
    
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(f"图像尺寸必须是块大小 {block_size} 的倍数")
    
    d_lambda_sum = 0.0
    count = 0
    
    # 计算所有波段对之间的 Q-index 差异
    for i in range(C - 1):
        for j in range(i + 1, C):
            # MS 图像的波段对
            q_map_ms = blockproc_uqi(ms[:, :, i], ms[:, :, j], block_size)
            q_ms = np.mean(q_map_ms)
            
            # SR 图像的波段对
            q_map_sr = blockproc_uqi(sr[:, :, i], sr[:, :, j], block_size)
            q_sr = np.mean(q_map_sr)
            
            d_lambda_sum += np.abs(q_sr - q_ms) ** p
            count += 1
    
    d_lambda_index = (d_lambda_sum / count) ** (1.0 / p)
    
    return d_lambda_index


def d_s(sr, ms, ms_lr, pan, ratio=4, block_size=32, q=1):
    """
    D_s 指标：空间失真指数
    
    Args:
        sr: 全景锐化结果 (H, W, C)
        ms: MS 图像上采样到 PAN 尺度 (H, W, C)
        ms_lr: 原始低分辨率 MS 图像 (H/ratio, W/ratio, C)
        pan: PAN 图像 (H, W) 或 (H, W, 1)
        ratio: 分辨率比例
        block_size: 块大小
        q: 指数参数
    
    Returns:
        D_s 指标值
    """
    if sr.shape != ms.shape:
        raise ValueError("SR 和 MS 图像必须具有相同的尺寸")
    
    # 确保 PAN 是 2D
    if pan.ndim == 3:
        pan = pan.squeeze()
    
    H, W, C = sr.shape
    
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(f"图像尺寸必须是块大小 {block_size} 的倍数")
    
    # 对 PAN 进行降采样再上采样
    pan_lr = cv2.resize(pan, (W // ratio, H // ratio), interpolation=cv2.INTER_CUBIC)
    pan_filtered = interp23tap(pan_lr, ratio)
    
    d_s_sum = 0.0
    
    # 对每个波段计算 Q-index 差异
    for i in range(C):
        # SR 与 PAN 的 Q-index
        q_map_high = blockproc_uqi(sr[:, :, i], pan, block_size)
        q_high = np.mean(q_map_high)
        
        # MS 与 filtered PAN 的 Q-index
        q_map_low = blockproc_uqi(ms[:, :, i], pan_filtered, block_size)
        q_low = np.mean(q_map_low)
        
        d_s_sum += np.abs(q_high - q_low) ** q
    
    d_s_index = (d_s_sum / C) ** (1.0 / q)
    
    return d_s_index


def hqnr(sr, ms, ms_lr, pan, sensor='WV3', ratio=4, block_size=32):
    """
    HQNR 指标：混合质量无参考指标
    
    Args:
        sr: 全景锐化结果 (H, W, C)
        ms: MS 图像上采样到 PAN 尺度 (H, W, C)
        ms_lr: 原始低分辨率 MS 图像 (H/ratio, W/ratio, C)
        pan: PAN 图像 (H, W) 或 (H, W, 1)
        sensor: 传感器类型
        ratio: 分辨率比例
        block_size: 块大小
    
    Returns:
        (HQNR, D_lambda, D_s) 元组
    """
    # 使用 MTF 滤波计算 D_lambda
    sr_degraded = mtf_filter(sr, sensor, ratio)
    sr_degraded_lr = cv2.resize(sr_degraded, (ms_lr.shape[1], ms_lr.shape[0]), 
                                 interpolation=cv2.INTER_CUBIC)
    
    # 计算 D_lambda（使用退化后的图像）
    # 这里简化实现，直接使用上采样的 MS
    dl = d_lambda(sr, ms_lr, ms, block_size, ratio)
    
    # 计算 D_s
    ds = d_s(sr, ms, ms_lr, pan, ratio, block_size)
    
    # 计算 HQNR
    hqnr_value = (1 - dl) * (1 - ds)
    
    return hqnr_value, dl, ds


def evaluate_single_image(sr, ms, ms_lr, pan, sensor='WV3', ratio=4, block_size=32):
    """
    评估单张图像的 full-resolution 指标
    
    Args:
        sr: 超分辨率结果，shape (C, H, W) 或 (H, W, C)
        ms: 上采样的 MS 图像，shape (C, H, W) 或 (H, W, C)
        ms_lr: 低分辨率 MS 图像，shape (C, H/ratio, W/ratio) 或 (H/ratio, W/ratio, C)
        pan: PAN 图像，shape (1, H, W) 或 (H, W, 1) 或 (H, W)
        sensor: 传感器类型
        ratio: 分辨率比例
        block_size: 块大小
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 确保形状是 (H, W, C)
    if sr.ndim == 3 and sr.shape[0] < sr.shape[2]:
        sr = np.transpose(sr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    if ms.ndim == 3 and ms.shape[0] < ms.shape[2]:
        ms = np.transpose(ms, (1, 2, 0))
    if ms_lr.ndim == 3 and ms_lr.shape[0] < ms_lr.shape[2]:
        ms_lr = np.transpose(ms_lr, (1, 2, 0))
    if pan.ndim == 3 and pan.shape[0] == 1:
        pan = pan.squeeze(0)  # (1, H, W) -> (H, W)
    elif pan.ndim == 3 and pan.shape[2] == 1:
        pan = pan.squeeze(2)  # (H, W, 1) -> (H, W)
    
    # 计算指标
    hqnr_val, dl, ds = hqnr(sr, ms, ms_lr, pan, sensor, ratio, block_size)
    
    metrics = {
        'D_lambda': dl,
        'D_s': ds,
        'HQNR': hqnr_val
    }
    
    return metrics


def evaluate_mat_file(mat_path, sensor='WV3', ratio=4, block_size=32):
    """
    评估 .mat 文件中的 full-resolution 结果
    
    Args:
        mat_path: .mat 文件路径，应包含 'sr', 'lms', 'ms', 'pan' 字段
        sensor: 传感器类型
        ratio: 分辨率比例
        block_size: 块大小
    """
    print(f"正在加载结果文件: {mat_path}")
    data = sio.loadmat(mat_path)
    
    # 检查必需的字段
    required_fields = ['sr', 'lms', 'ms', 'pan']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Mat 文件必须包含 '{field}' 字段")
    
    sr_images = data['sr']
    lms_images = data['lms']  # 上采样的 MS
    ms_lr_images = data['ms']  # 低分辨率 MS
    pan_images = data['pan']
    
    print(f"SR shape: {sr_images.shape}")
    print(f"LMS shape: {lms_images.shape}")
    print(f"MS_LR shape: {ms_lr_images.shape}")
    print(f"PAN shape: {pan_images.shape}")
    
    # 处理数据格式
    if isinstance(sr_images, np.ndarray):
        if sr_images.ndim == 4:  # (N, C, H, W) or (N, H, W, C)
            num_images = sr_images.shape[0]
        else:
            num_images = 1
            sr_images = sr_images[np.newaxis, ...]
            lms_images = lms_images[np.newaxis, ...]
            ms_lr_images = ms_lr_images[np.newaxis, ...]
            pan_images = pan_images[np.newaxis, ...]
    else:
        sr_images = np.array(sr_images)
        lms_images = np.array(lms_images)
        ms_lr_images = np.array(ms_lr_images)
        pan_images = np.array(pan_images)
        num_images = len(sr_images)
    
    print(f"\n正在评估 {num_images} 张图像...")
    print("=" * 80)
    
    all_metrics = {
        'D_lambda': [],
        'D_s': [],
        'HQNR': []
    }
    
    for i in range(num_images):
        sr = sr_images[i] if num_images > 1 else sr_images[0]
        lms = lms_images[i] if num_images > 1 else lms_images[0]
        ms_lr = ms_lr_images[i] if num_images > 1 else ms_lr_images[0]
        pan = pan_images[i] if num_images > 1 else pan_images[0]
        
        # 去除多余的维度
        sr = np.squeeze(sr)
        lms = np.squeeze(lms)
        ms_lr = np.squeeze(ms_lr)
        pan = np.squeeze(pan)
        
        try:
            metrics = evaluate_single_image(sr, lms, ms_lr, pan, sensor, ratio, block_size)
            
            print(f"\n图像 {i+1}/{num_images}:")
            print(f"  D_lambda: {metrics['D_lambda']:.6f}")
            print(f"  D_s:      {metrics['D_s']:.6f}")
            print(f"  HQNR:     {metrics['HQNR']:.6f}")
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        except Exception as e:
            print(f"\n图像 {i+1}/{num_images} 评估失败: {str(e)}")
            continue
    
    # 计算平均值和标准差
    print("\n" + "=" * 80)
    print("平均指标:")
    print("=" * 80)
    for key in all_metrics:
        if len(all_metrics[key]) > 0:
            values = np.array(all_metrics[key])
            mean_val = values.mean()
            std_val = values.std()
            print(f"{key:10s}: {mean_val:.6f} ± {std_val:.6f}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='评估 Full-Resolution 全景锐化结果')
    parser.add_argument('--mat_file', type=str, required=True,
                        help='.mat 文件路径，包含 sr, lms, ms, pan 字段')
    parser.add_argument('--sensor', type=str, default='WV3',
                        choices=['WV2', 'WV3', 'WV4', 'QB', 'IKONOS', 'GeoEye1', 'GF2'],
                        help='传感器类型')
    parser.add_argument('--ratio', type=int, default=4,
                        help='分辨率比例')
    parser.add_argument('--block_size', type=int, default=32,
                        help='Q-index 的块大小')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mat_file):
        print(f"错误: 文件不存在: {args.mat_file}")
        return
    
    evaluate_mat_file(args.mat_file, args.sensor, args.ratio, args.block_size)


if __name__ == '__main__':
    main()

