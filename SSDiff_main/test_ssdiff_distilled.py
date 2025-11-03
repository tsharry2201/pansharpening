"""
统一的SSDiff测试脚本
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
from ssdiff_distill import SSDiff_test
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Unified SSDiff Testing')
    
    # 模型路径
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径（原始SSDiff或蒸馏后的checkpoint）")
    parser.add_argument("--pretrained_ssdiff_path", type=str, default=None,
                       help="预训练SSDiff路径（蒸馏模式需要）")
    
    # 模式选择
    parser.add_argument("--use_distillation", type=bool, default=False,
                       help="是否使用蒸馏模型（True=单步，False=多步）")
    parser.add_argument("--test_with_noise", type=bool, default=False,
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
    
    # 加载权重
    model.load_state_dict(
        torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
    )
    model.cuda()
    
    # 设置 epoch > 1000，确保 ARConv 使用训练好的固定卷积核（reserved_NXY）
    model.set_epoch(10001)
    print("   Set epoch to 10001 for using fixed ARConv kernel size")
    
    model.eval()
    
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
    print("🟢 Testing Distilled SSDiff (One-step Sampling)")
    print(f"   Model: {args.model_path}")
    print(f"   Base: {args.pretrained_ssdiff_path}")
    if hasattr(ssdiff_args, 'test_with_noise') and ssdiff_args.test_with_noise:
        print(f"   🔬 x_t mode: Noisy (q_sample_xt)")
    else:
        print(f"   ✓ x_t mode: Zero tensor (stable)")
    print("=" * 60)
    
    # 创建蒸馏模型
    test_model = SSDiff_test(ssdiff_args)
    
    # 设置 epoch > 1000，确保 ARConv 使用训练好的固定卷积核（reserved_NXY）
    #test_model.model.set_epoch(10001)
    print("   Set epoch to 10001 for using fixed ARConv kernel size")
    
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
        
        # ⚠️ 数据加载器已经归一化到[0,1]了（用img_scale=2047.0），不需要再次归一化！
        # 第一张图像：检查数据范围
        if i == 0:
            print("\n" + "="*60)
            print("🔍 DEBUG INFO - First Image (数据已被加载器归一化)")
            print("="*60)
            print(f"📥 Input ranges (already normalized to [0,1] by dataloader):")
            print(f"   LMS: [{lms.min():.4f}, {lms.max():.4f}], mean={lms.mean():.4f}, std={lms.std():.4f}")
            print(f"   PAN: [{pan.min():.4f}, {pan.max():.4f}], mean={pan.mean():.4f}, std={pan.std():.4f}")
            print(f"   MS:  [{ms.min():.4f}, {ms.max():.4f}], mean={ms.mean():.4f}, std={ms.std():.4f}")
            
            # GT信息（已归一化）
            gt_normalized = gt[0].cpu().numpy()
            print(f"   GT:  [{gt_normalized.min():.4f}, {gt_normalized.max():.4f}], mean={gt_normalized.mean():.4f}, std={gt_normalized.std():.4f}")
      
        # 单步推理
        with torch.no_grad():
            sample = test_model(lms, pan, ms)
        
        # 第一张图像时打印输出信息
        if i == 0:
            print(f"\n📤 Model output ([0,1] range):")
            print(f"   Output: [{sample.min():.4f}, {sample.max():.4f}], mean={sample.mean():.4f}, std={sample.std():.4f}")
            
            # 计算与GT的差异
            sample_cpu = sample[0].cpu()
            lms_cpu = lms[0].cpu()
            # gt[0]已经是[C,H,W]格式且已归一化
            gt_tensor = torch.from_numpy(gt_normalized)
            
            # 估计的残差（模型预测的）
            predicted_residual = sample_cpu - lms_cpu
            # 实际的残差（真实需要的）
            actual_residual = gt_tensor - lms_cpu
            # 残差预测误差
            residual_error = predicted_residual - actual_residual
            
            print(f"\n📊 残差分析 (Residual Analysis):")
            print(f"   预测残差 (Predicted): [{predicted_residual.min():.4f}, {predicted_residual.max():.4f}], mean={predicted_residual.mean():.4f}, std={predicted_residual.std():.4f}")
            print(f"   实际残差 (Actual):    [{actual_residual.min():.4f}, {actual_residual.max():.4f}], mean={actual_residual.mean():.4f}, std={actual_residual.std():.4f}")
            print(f"   残差误差 (Error):     [{residual_error.min():.4f}, {residual_error.max():.4f}], mean={residual_error.mean():.4f}, std={residual_error.std():.4f}")
            print(f"   残差MSE: {(residual_error**2).mean():.6f}")
            
            # 计算与GT的差异
            output_diff = torch.abs(sample_cpu - gt_tensor)
            print(f"\n📊 输出与GT差异:")
            print(f"   Abs diff: [{output_diff.min():.4f}, {output_diff.max():.4f}], mean={output_diff.mean():.4f}")
            print(f"   MSE: {(output_diff**2).mean():.6f}")
        
        # 反归一化到[0, 2047]用于评估指标，与原始SSDiff保持一致
        sample = (sample * 2047.).clamp(0, 2047)
        all_images.extend([sample.cpu().numpy()])
        
        if i == 0:
            print(f"\n📤 Final output (denormalized to [0,2047] for metrics):")
            print(f"   Output: [{sample.min():.4f}, {sample.max():.4f}], mean={sample.mean():.4f}")
            print("="*60)
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{image_num} images")
    
    inference_time = time.time() - tic
    print(f"\n⏱️  Total time: {inference_time:.2f}s")
    print(f"⏱️  Average time per image: {inference_time/image_num:.3f}s")
    
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
    ssdiff_args.test_with_noise = args.test_with_noise  #  添加噪声测试选项
    ssdiff_args.pretrained_ssdiff_path = args.pretrained_ssdiff_path
    ssdiff_args.mixed_precision = args.mixed_precision
    ssdiff_args.dataset['test'] = args.test_dataset
    
    # 设置GPU
    torch.cuda.set_device(ssdiff_args.device)
    
    # 配置日志
    rootPath = os.path.abspath(os.path.dirname(__file__))
    logger.configure(dir=os.path.join(rootPath, 'logs/sample_logs/'))
    
    # 执行测试
    if args.use_distillation:
        # 蒸馏模式（单步）
        if args.pretrained_ssdiff_path is None:
            raise ValueError("蒸馏模式需要提供 --pretrained_ssdiff_path")
        all_images, data4gt, inference_time = test_distilled_ssdiff(args, ssdiff_args)
        mode_name = "distilled_1step"
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
    
    # 准备保存数据 - 与原始SSDiff保持一致
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