"""
融合模型测试脚本
测试OTPNet和SSDiff的后融合模型
"""
import os
import sys
import argparse
import datetime
import time
import torch
import numpy as np
from scipy.io import savemat
from tqdm import tqdm
import einops

# 添加路径
sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')
sys.path.append('/data2/user/zelilin/OTPNet')

from fusion_model import OTPNet_SSDiff_Fusion
from configs.option_DPM_pansharpening import parser_args as ssdiff_parser_args
from pancollection.common.psdata import PansharpeningSession as DataSession


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试OTPNet-SSDiff融合模型')
    
    # 模型路径
    parser.add_argument("--otpnet_checkpoint", type=str, required=True,
                       help="OTPNet模型路径")
    parser.add_argument("--ssdiff_checkpoint", type=str, required=True,
                       help="SSDiff模型路径")
    parser.add_argument("--pretrained_ssdiff_path", type=str, required=True,
                       help="预训练SSDiff基础模型路径")
    parser.add_argument("--fusion_weights", type=str, required=True,
                       help="融合网络权重路径")
    
    # 融合网络配置（需与训练时一致）
    parser.add_argument("--fusion_channels", type=int, default=64,
                       help="融合网络隐藏通道数")
    parser.add_argument("--num_fusion_blocks", type=int, default=3,
                       help="融合块数量")
    
    # 测试参数
    parser.add_argument("--test_dataset", type=str, default='test_wv3_multiExm1.h5',
                       help="测试数据集")
    parser.add_argument("--num_test_images", type=int, default=20,
                       help="测试图像数量")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--output_dir", type=str, default="test_results")
    
    # Self-ensemble选项
    parser.add_argument("--use_self_ensemble", action='store_true',
                       help="是否使用self-ensemble（8种几何变换集成）")
    
    # 保存单个模型的输出用于对比
    parser.add_argument("--save_individual_outputs", action='store_true',
                       help="是否保存OTPNet和SSDiff的单独输出")
    
    return parser.parse_args()


def apply_self_ensemble(model, pan, lms, ms):
    """
    应用self-ensemble进行推理
    
    Args:
        model: 融合模型
        pan, lms, ms: 输入数据
    
    Returns:
        ensemble_output: 平均后的预测结果
    """
    transforms = [
        # (transform_fn, inverse_fn, name)
        (lambda x: x, lambda x: x, "original"),
        (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1]), "h_flip"),
        (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2]), "v_flip"),
        (lambda x: torch.rot90(x, k=1, dims=[-2, -1]), lambda x: torch.rot90(x, k=-1, dims=[-2, -1]), "rot90"),
        (lambda x: torch.rot90(x, k=2, dims=[-2, -1]), lambda x: torch.rot90(x, k=2, dims=[-2, -1]), "rot180"),
        (lambda x: torch.rot90(x, k=3, dims=[-2, -1]), lambda x: torch.rot90(x, k=-3, dims=[-2, -1]), "rot270"),
        (lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1]), 
         lambda x: torch.flip(torch.rot90(x, k=-1, dims=[-2, -1]), dims=[-1]), "h_flip_rot90"),
        (lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=1, dims=[-2, -1]), 
         lambda x: torch.flip(torch.rot90(x, k=-1, dims=[-2, -1]), dims=[-2]), "v_flip_rot90"),
    ]
    
    ensemble_outputs = []
    
    for transform_fn, inverse_fn, name in transforms:
        # 1. 对输入应用变换
        pan_t = transform_fn(pan)
        lms_t = transform_fn(lms)
        ms_t = transform_fn(ms)
        
        # 2. 模型推理
        with torch.no_grad():
            output_t, _, _ = model(pan_t, lms_t, ms_t)
        
        # 3. 对输出应用逆变换
        output_inv = inverse_fn(output_t)
        ensemble_outputs.append(output_inv)
    
    # 4. 平均所有变换的结果
    ensemble_output = torch.stack(ensemble_outputs, dim=0).mean(dim=0)
    
    return ensemble_output


@torch.no_grad()
def test(args):
    """测试函数"""
    print("=" * 60)
    print("🎯 测试OTPNet-SSDiff融合模型")
    print("=" * 60)
    print(f"   OTPNet: {args.otpnet_checkpoint}")
    print(f"   SSDiff: {args.ssdiff_checkpoint}")
    print(f"   融合权重: {args.fusion_weights}")
    if args.use_self_ensemble:
        print(f"   🔄 Self-Ensemble: Enabled (8 transforms)")
    print("=" * 60)
    
    # 设置设备
    torch.cuda.set_device(args.device)
    
    # 获取SSDiff配置
    ssdiff_args = ssdiff_parser_args()
    ssdiff_args.model_path = args.ssdiff_checkpoint
    ssdiff_args.pretrained_ssdiff_path = args.pretrained_ssdiff_path
    ssdiff_args.use_distillation = True
    ssdiff_args.device = args.device
    
    # 创建融合模型
    print("\n创建融合模型...")
    model = OTPNet_SSDiff_Fusion(
        otpnet_checkpoint=args.otpnet_checkpoint,
        ssdiff_checkpoint=args.ssdiff_checkpoint,
        ssdiff_args=ssdiff_args,
        fusion_channels=args.fusion_channels,
        num_fusion_blocks=args.num_fusion_blocks,
        device=args.device
    )
    
    # 加载融合网络权重
    model.load_fusion_weights(args.fusion_weights)
    model.eval()
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    session = DataSession(ssdiff_args)
    data, _ = session.get_eval_dataloader(args.test_dataset, False)
    dl = iter(data)
    
    # 测试
    all_images_fusion = []
    all_images_otpnet = []
    all_images_ssdiff = []
    data4gt = []
    image_num = min(args.num_test_images, len(data))
    
    print(f"\n测试 {image_num} 张图像...")
    
    tic = time.time()
    for i in tqdm(range(image_num), desc="Testing"):
        batch = next(dl)
        pan, lms, ms, gt = batch['pan'], batch['lms'], batch['ms'], batch['gt']
        
        # 调整GT维度
        gt = einops.rearrange(gt, 'b k1 k2 c -> b c k1 k2')
        data4gt.append(gt[0])
        
        # 移动到设备
        pan = pan.to(args.device)
        lms = lms.to(args.device)
        ms = ms.to(args.device)
        
        # 推理
        if args.use_self_ensemble:
            # Self-ensemble
            sample_fusion = apply_self_ensemble(model, pan, lms, ms)
            # 单独模型输出（使用第一个变换，即原图）
            with torch.no_grad():
                _, sample_otpnet, sample_ssdiff = model(pan, lms, ms)
        else:
            # 普通推理
            with torch.no_grad():
                sample_fusion, sample_otpnet, sample_ssdiff = model(pan, lms, ms)
        
        # 反归一化到[0, 2047]用于评估指标
        sample_fusion = (sample_fusion * 2047.).clamp(0, 2047)
        sample_otpnet = (sample_otpnet * 2047.).clamp(0, 2047)
        sample_ssdiff = (sample_ssdiff * 2047.).clamp(0, 2047)
        
        # 保存结果
        all_images_fusion.append(sample_fusion.cpu().numpy())
        if args.save_individual_outputs:
            all_images_otpnet.append(sample_otpnet.cpu().numpy())
            all_images_ssdiff.append(sample_ssdiff.cpu().numpy())
        
        # 第一张图像时打印详细信息
        if i == 0:
            print(f"\n第一张图像统计信息：")
            print(f"   融合输出: [{sample_fusion.min():.2f}, {sample_fusion.max():.2f}], mean={sample_fusion.mean():.2f}")
            print(f"   OTPNet输出: [{sample_otpnet.min():.2f}, {sample_otpnet.max():.2f}], mean={sample_otpnet.mean():.2f}")
            print(f"   SSDiff输出: [{sample_ssdiff.min():.2f}, {sample_ssdiff.max():.2f}], mean={sample_ssdiff.mean():.2f}")
            gt_denorm = gt[0].cpu().numpy() * 2047
            print(f"   GT: [{gt_denorm.min():.2f}, {gt_denorm.max():.2f}], mean={gt_denorm.mean():.2f}")
    
    inference_time = time.time() - tic
    print(f"\n⏱️  总时间: {inference_time:.2f}s")
    print(f"⏱️  平均每张: {inference_time/image_num:.3f}s")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 拼接结果
    arr_fusion = np.concatenate(all_images_fusion, axis=0)
    
    # 准备保存数据
    save_data = {
        'sr': arr_fusion,  # 融合模型输出
        'gt': [sample.cpu().numpy() * 2047 for sample in data4gt],
        'model_name': 'fusion',
        'fusion_weights': args.fusion_weights,
        'inference_time': inference_time,
        'avg_time_per_image': inference_time / len(arr_fusion),
    }
    
    # 如果保存单独输出
    if args.save_individual_outputs:
        arr_otpnet = np.concatenate(all_images_otpnet, axis=0)
        arr_ssdiff = np.concatenate(all_images_ssdiff, axis=0)
        save_data['sr_otpnet'] = arr_otpnet
        save_data['sr_ssdiff'] = arr_ssdiff
    
    # 生成文件名
    timestamp = datetime.datetime.now().strftime('%m-%d-%H-%M')
    ensemble_tag = '_ensemble' if args.use_self_ensemble else ''
    img_size = arr_fusion[0].shape[0] if len(arr_fusion) > 0 else 256
    
    # 判断分辨率类型
    dataset_name = args.test_dataset
    if 'OrigScale' in dataset_name or 'origscale' in dataset_name.lower():
        resolution_type = 'full'
    else:
        resolution_type = 'reduced'
    
    out_path = os.path.join(
        args.output_dir,
        f'samp_fusion{ensemble_tag}_{resolution_type}_{len(arr_fusion)}_{img_size}_{timestamp}.mat'
    )
    
    savemat(out_path, save_data)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print(f"📁 结果保存到: {out_path}")
    print(f"🖼️  处理图像数: {len(arr_fusion)}")
    print(f"⏱️  总时间: {inference_time:.2f}s")
    print(f"⏱️  平均每张: {inference_time/len(arr_fusion):.3f}s")
    print(f"🚀 速度: {len(arr_fusion)/inference_time:.2f} images/sec")
    if args.save_individual_outputs:
        print(f"📊 同时保存了OTPNet和SSDiff的单独输出用于对比")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()

