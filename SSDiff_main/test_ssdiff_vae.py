"""
统一的SSDiff VAE测试脚本
支持两种模式：
1. 原始SSDiff多步采样 (use_distillation=False)
2. 蒸馏后的单步采样 (use_distillation=True)
"""
import os
import sys
import argparse
import time
import torch
import numpy as np
from scipy.io import savemat
import einops
import datetime

# 添加SSDiff路径
sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')

from configs.option_DPM_pansharpening import parser_args as ssdiff_parser_args
from pancollection.common.psdata import PansharpeningSession as DataSession
from ssdiff_vae_retrain import SSDiff_VAE_test
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Unified SSDiff VAE Testing')
    
    # 模型路径
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径（原始SSDiff或蒸馏后的checkpoint）")
    parser.add_argument("--pretrained_ssdiff_path", type=str, default=None,
                       help="预训练SSDiff路径（蒸馏模式需要）")
    
    # 模式选择
    parser.add_argument("--use_distillation", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用蒸馏模型（True=单步，False=多步）")
    parser.add_argument("--test_with_noise", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用带噪声的x_t（仅蒸馏模式有效）")
    
    # 测试参数
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--crop_batch_size", type=int, default=8)
    parser.add_argument("--timestep_respacing", type=str, default="ddim10",
                       help="采样步数（ddim1/ddim10/ddim50等）")
    parser.add_argument("--test_dataset", type=str, default='test_wv3_multiExm1.h5',
                       help="测试数据集")
    parser.add_argument("--num_test_images", type=int, default=20,
                       help="测试图像数量")
    
    # 其他参数
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, default="test_results")
    
    # VAE配置（独立）
    parser.add_argument("--use_vae", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用VAE编码器（提取全局场景特征）")
    parser.add_argument("--vae_latent_dim", type=int, default=256,
                       help="VAE潜在维度")
    parser.add_argument("--use_kl_loss", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用KL散度损失（仅VAE相关）")
    parser.add_argument("--use_perceptual_loss", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用感知损失（仅VAE相关）")
    
    # ControlNet配置（独立）
    parser.add_argument("--use_controlnet", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用完整ControlNet（提取多尺度空间特征）")
    parser.add_argument("--ms_channels", type=int, default=8,
                       help="多光谱通道数（ControlNet需要）")
    
    # Scene Token配置（保留兼容性，但VAE和ControlNet是独立的）
    parser.add_argument("--use_scene_token", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用场景token（旧版，已被VAE替代）")
    parser.add_argument("--scene_token_freeze_steps", type=int, default=0,
                       help="场景token冻结步数")
    
    # ARConv配置
    parser.add_argument("--use_arconv", type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help="是否使用ARConv")
    parser.add_argument("--arconv_hw_range", type=str, default="[1,9]",
                       help="ARConv卷积核范围")
    parser.add_argument("--arconv_fixstep", type=int, default=10001,
                       help="ARConv fixstep参数（测试时无影响，会强制使用epoch=10001）")
    # CLIP文本编码器配置（独立）
    parser.add_argument("--use_clip", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用CLIP文本编码器（提取文本语义特征）")
    parser.add_argument("--clip_model_path", type=str,
                       default=None,
                       help="CLIP模型路径，如果为None则使用预训练模型路径")
    parser.add_argument("--prompt", type=str,
                       default="A high resolution satellite image with clear details",
                       help="默认文本提示")
    parser.add_argument("--train_clip", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否训练CLIP文本编码器（默认冻结）")
    
    # 🔥 RAM Caption生成器配置（独立）
    parser.add_argument("--use_ram", type=lambda x: str(x).lower() == 'true',
                       default=False,
                       help="是否使用RAM caption生成器（从PAN图像自动生成caption）")
    parser.add_argument("--ram_model_path", type=str,
                       default=None,
                       help="RAM模型路径")
    
    return parser.parse_args()


def test_original_ssdiff(args, ssdiff_args):
    """
    测试原始SSDiff（多步采样）
    """
    print("=" * 60)
    print("🔵 Testing Original SSDiff (Multi-step Sampling)")
    print(f"   Model: {args.model_path}")
    print(f"   Sampling: {args.timestep_respacing}")
    print("=" * 60)
    
    # 创建模型
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(ssdiff_args, model_and_diffusion_defaults().keys())
    )
    
    # 加载权重（先加载到CPU）
    model.load_state_dict(
        torch.load(args.model_path, map_location='cpu')
    )
    
    # 设置 epoch > 1000，确保 ARConv 使用训练好的固定卷积核
    model.set_epoch(10001)
    print("   Set epoch to 10001 for using fixed ARConv kernel size")
    
    model.eval()
    
    # 🔥 多GPU支持：DataParallel包装后再移动到GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n🚀 检测到 {num_gpus} 个GPU，使用DataParallel进行并行推理")
        print("   模型将自动分布到所有GPU上")
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        print(f"✅ 模型已加载到 {num_gpus} 个GPU")
    else:
        print("📍 使用单GPU模式")
        model = model.cuda()
    
    # 创建数据加载器
    session = DataSession(ssdiff_args)
    data, _ = session.get_eval_dataloader(args.test_dataset, False)
    dl = iter(data)
    
    # 测试
    all_images = []
    data4gt = []
    image_num = min(args.num_test_images, len(data))
    
    print(f"Testing on {image_num} images...")
    
    tic = time.time()
    for i in range(image_num):
        batch = next(dl)
        pan, lms, ms, gt = batch['pan'], batch['lms'], batch['ms'], batch['gt']
        gt = einops.rearrange(gt, 'b k1 k2 c -> b c k1 k2')
        data4gt.append(gt[0])
        
        pan, lms, ms = map(lambda x: x.cuda(), (pan, lms, ms))
        
        # 多步采样
        sample_fn = (
            diffusion.p_sample_loop 
            if not ssdiff_args.use_ddim 
            else diffusion.ddim_sample_loop
        )
        
        kwargs_data = {"lms": lms, "pan": pan, "ms": ms}
        
        with torch.no_grad():
            sample = sample_fn(
                model,
                shape=(ssdiff_args.crop_batch_size, ssdiff_args.ms_dim, 
                      ssdiff_args.image_size, ssdiff_args.image_size),
                model_kwargs=kwargs_data,
                clip_denoised=ssdiff_args.clip_denoised,
                progress=False
            )
        
        sample = sample.contiguous()
        sample = (sample * 2047.).clamp(0, 2047)
        all_images.extend([sample.cpu().numpy()])
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{image_num} images")
    
    inference_time = time.time() - tic
    print(f"\n⏱️  Total time: {inference_time:.2f}s")
    print(f"⏱️  Average time per image: {inference_time/image_num:.3f}s")
    
    return all_images, data4gt, inference_time


def test_distilled_ssdiff(args, ssdiff_args):
    """
    测试蒸馏后的SSDiff（单步采样） 
    """
    print("=" * 60)
    print(" Testing Distilled SSDiff VAE (One-step Sampling)")
    print(f"   Model: {args.model_path}")
    print(f"   Base: {args.pretrained_ssdiff_path}")
    if hasattr(ssdiff_args, 'test_with_noise') and ssdiff_args.test_with_noise:
        print(f"   🔬 x_t mode: Noisy (q_sample_xt)")
    else:
        print(f"   ✓ x_t mode: Zero tensor (stable)")
    print("=" * 60)

    # 🔹 固定单GPU模式
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"📍 使用单GPU模式: {device}")

    # 直接创建模型（不使用DataParallel）
    test_model = SSDiff_VAE_test(ssdiff_args, use_multi_gpu=False)

    # 立即初始化大型模块（确保CLIP/VAE/RAM加载）
    if hasattr(test_model, '_initialize_large_modules'):
        test_model._initialize_large_modules()
        print("✅ 所有模块初始化完成 (VAE/ControlNet/CLIP/RAM)")

    # 将模型放到GPU
    test_model.to(device)
    test_model.eval()

    # 打印CLIP / RAM 加载状态
    print(f"use_clip: {getattr(test_model, 'use_clip', None)}")
    print(f"use_ram: {getattr(test_model, 'use_ram', None)}")
    print(f"text_encoder: {type(getattr(test_model, 'text_encoder', None))}")
    print(f"ram_model: {type(getattr(test_model, 'ram_model', None))}")

    #  创建数据加载器
    session = DataSession(ssdiff_args)
    data, _ = session.get_eval_dataloader(args.test_dataset, False)
    dl = iter(data)

    all_images = []
    data4gt = []
    image_num = min(args.num_test_images, len(data))
    print(f"Testing on {image_num} images...")

    tic = time.time()
    for i in range(image_num):
        batch = next(dl)
        pan, lms, ms, gt = batch['pan'], batch['lms'], batch['ms'], batch['gt']
        gt = einops.rearrange(gt, 'b k1 k2 c -> b c k1 k2')
        data4gt.append(gt[0])

        pan, lms, ms = pan.to(device), lms.to(device), ms.to(device)

        if i == 0:
            print("\n" + "=" * 60)
            print("🔍 DEBUG INFO - First Image (normalized inputs)")
            print("=" * 60)
            print(f"LMS: [{lms.min():.4f}, {lms.max():.4f}], mean={lms.mean():.4f}")
            print(f"PAN: [{pan.min():.4f}, {pan.max():.4f}], mean={pan.mean():.4f}")
            print(f"MS : [{ms.min():.4f}, {ms.max():.4f}], mean={ms.mean():.4f}")

        with torch.no_grad():
            sample = test_model(lms, pan, ms)

        if i == 0:
            print(f"\n📤 Model output range [0,1]: [{sample.min():.4f}, {sample.max():.4f}], mean={sample.mean():.4f}")

        sample = (sample * 2047.).clamp(0, 2047)
        all_images.append(sample.cpu().numpy())

        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{image_num} images")

    inference_time = time.time() - tic
    print(f"\n⏱️ Total time: {inference_time:.2f}s")
    print(f"⏱️ Avg per image: {inference_time / image_num:.3f}s")

    return all_images, data4gt, inference_time


def main():
    """主函数"""
    args = parse_args()
    
    # 获取SSDiff配置
    ssdiff_args = ssdiff_parser_args()
    
    # 合并参数
    ssdiff_args.device = args.device
    ssdiff_args.crop_batch_size = args.crop_batch_size
    ssdiff_args.timestep_respacing = args.timestep_respacing
    ssdiff_args.model_path = args.model_path
    ssdiff_args.use_distillation = args.use_distillation
    ssdiff_args.test_with_noise = args.test_with_noise
    ssdiff_args.pretrained_ssdiff_path = args.pretrained_ssdiff_path
    ssdiff_args.mixed_precision = args.mixed_precision
    ssdiff_args.dataset['test'] = args.test_dataset
    
    # 添加VAE参数（独立）
    ssdiff_args.use_vae = args.use_vae.lower() == 'true' if isinstance(args.use_vae, str) else args.use_vae
    ssdiff_args.vae_latent_dim = args.vae_latent_dim
    ssdiff_args.use_kl_loss = args.use_kl_loss.lower() == 'true' if isinstance(args.use_kl_loss, str) else args.use_kl_loss
    ssdiff_args.use_perceptual_loss = args.use_perceptual_loss.lower() == 'true' if isinstance(args.use_perceptual_loss, str) else args.use_perceptual_loss
    
    # 添加ControlNet参数（独立）
    ssdiff_args.use_controlnet = args.use_controlnet.lower() == 'true' if isinstance(args.use_controlnet, str) else args.use_controlnet
    ssdiff_args.ms_channels = args.ms_channels
    
    # 添加Scene Token参数（保留兼容性）
    ssdiff_args.use_scene_token = args.use_scene_token.lower() == 'true' if isinstance(args.use_scene_token, str) else args.use_scene_token
    ssdiff_args.scene_token_freeze_steps = args.scene_token_freeze_steps
    
    # 添加ARConv参数
    ssdiff_args.use_arconv = args.use_arconv.lower() == 'true' if isinstance(args.use_arconv, str) else args.use_arconv
    if isinstance(args.arconv_hw_range, str):
        ssdiff_args.arconv_hw_range = eval(args.arconv_hw_range)
    else:
        ssdiff_args.arconv_hw_range = args.arconv_hw_range
    ssdiff_args.arconv_fixstep = args.arconv_fixstep
    
    # 添加RAM Caption生成器参数（独立）
    ssdiff_args.use_ram = args.use_ram.lower() == 'true' if isinstance(args.use_ram, str) else args.use_ram
    ssdiff_args.ram_model_path = args.ram_model_path
    
    # 添加CLIP文本编码器参数（独立）
    ssdiff_args.use_clip = args.use_clip.lower() == 'true' if isinstance(args.use_clip, str) else args.use_clip
    ssdiff_args.clip_model_path = args.clip_model_path
    ssdiff_args.prompt = args.prompt
    ssdiff_args.train_clip = args.train_clip.lower() == 'true' if isinstance(args.train_clip, str) else args.train_clip
    
    
    # 打印配置信息
    print("\n" + "="*60)
    print("🔧 模型配置:")
    print("="*60)
    print(f"  VAE编码器: {'✓ 启用' if ssdiff_args.use_vae else '✗ 禁用'}")
    if ssdiff_args.use_vae:
        print(f"    - 潜在维度: {ssdiff_args.vae_latent_dim}")
    print(f"  完整ControlNet: {'✓ 启用' if ssdiff_args.use_controlnet else '✗ 禁用'}")
    if ssdiff_args.use_controlnet:
        print(f"    - 多光谱通道: {ssdiff_args.ms_channels}")
    print(f"  ARConv: {'✓ 启用' if ssdiff_args.use_arconv else '✗ 禁用'}")
    print(f"  RAM Caption生成器: {'✓ 启用' if ssdiff_args.use_ram else '✗ 禁用'}")
    if ssdiff_args.use_ram:
        print(f"    - 模型路径: {ssdiff_args.ram_model_path}")
    print("="*60 + "\n")
    
    # 设置GPU - 多GPU模式下不设置特定设备
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🚀 多GPU模式：将使用 {num_gpus} 个GPU")
    else:
        torch.cuda.set_device(ssdiff_args.device)
        print(f"📍 单GPU模式：{ssdiff_args.device}")
    
    # 配置日志
    rootPath = os.path.abspath(os.path.dirname(__file__))
    logger.configure(dir=os.path.join(rootPath, 'logs/sample_logs/'))
    
    # 执行测试
    if args.use_distillation:
        # 蒸馏模式（单步）
        if args.pretrained_ssdiff_path is None:
            raise ValueError("蒸馏模式需要提供 --pretrained_ssdiff_path")
        all_images, data4gt, inference_time = test_distilled_ssdiff(args, ssdiff_args)
        mode_name = "distilled_vae_1step"
    else:
        # 原始模式（多步）
        all_images, data4gt, inference_time = test_original_ssdiff(args, ssdiff_args)
        mode_name = f"original_{args.timestep_respacing}"
    
    # 保存结果
    arr = np.concatenate(all_images, axis=0)
    
    # 提取模型信息
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model_dir = os.path.basename(os.path.dirname(args.model_path))
    
    # 判断分辨率
    dataset_name = args.test_dataset
    if 'OrigScale' in dataset_name or 'origscale' in dataset_name.lower():
        resolution_type = 'full'
        img_size = arr[0].shape[0] if len(arr) > 0 else 512
    else:
        resolution_type = 'reduced'
        img_size = arr[0].shape[0] if len(arr) > 0 else 256
    
    # 准备保存数据
    d = dict(
        gt=[sample.cpu().numpy() * 2047 for sample in data4gt],  # 反归一化到[0,2047]
        sr=[sample for sample in arr],  # 已经在[0,2047]范围
        model_name=model_name,
        model_dir=model_dir,
        model_path=args.model_path,
        mode=mode_name,
        inference_time=inference_time,
        avg_time_per_image=inference_time / len(arr),
    )
    
    # 保存到文件
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    out_path = os.path.join(
        args.output_dir,
        f'samp_{mode_name}_{resolution_type}_{len(arr)}_{img_size}_{model_name}_{timestamp}.mat'
    )
    
    savemat(out_path, d)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)
    print(f"📁 Results saved to: {out_path}")
    print(f"🖼️  Images processed: {len(arr)}")
    print(f"⏱️  Total time: {inference_time:.2f}s")
    print(f"⏱️  Avg time/image: {inference_time/len(arr):.3f}s")
    print(f"🚀 Speed: {len(arr)/inference_time:.2f} images/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
