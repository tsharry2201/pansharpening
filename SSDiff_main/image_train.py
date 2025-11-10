"""
Train a diffusion model on images.

Matlab test code for WV3:

cd Matlab-Test-Package
analysis_ref_batched_images('*.mat', 4, 0, 2047)

"""
import os
import torch.cuda
import torch as th
import numpy as np
import datetime
import socket
import wandb

rootPath = os.path.abspath(os.path.dirname(__file__))

from improved_diffusion import logger
from improved_diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from utils.train_util import TrainLoop
from pancollection.common.psdata import PansharpeningSession as DataSession
from configs.option_DPM_pansharpening import parser_args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def main(
    device='cuda:5',
    Resume = False,
    resume_epoch = 0  # 继续训练时设置正确的epoch值
    ):
    
    args = parser_args()
    if isinstance(args.dataset, dict):
        args.dataset['train'] = 'wv3_otpnet'
        args.dataset['valid'] = 'wv3_otpnet'
        args.dataset['test'] = 'test_wv3_multiExm1_otpnet.h5'

    print("========== Training Configuration ==========")
    if isinstance(args.dataset, dict):
        train_ds = args.dataset.get('train', 'N/A')
        valid_ds = args.dataset.get('valid', 'N/A')
        test_ds = args.dataset.get('test', 'N/A')
        print(f"dataset(train/valid/test): {train_ds} / {valid_ds} / {test_ds}")
    else:
        print(f"dataset: {args.dataset}")
    print(f"data_dir: {args.data_dir}")
    print("============================================")
    set_seed(2024)
    if device is not None:
        args.device = device
    torch.cuda.set_device(args.device)
    
    # 设置最大训练步数为12万
    args.lr_anneal_steps = 1200000
    
    # 初始化wandb
    run_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    wandb.init(
        config=args,
        project="ssdiff-pansharpening",
        entity="tszharry-xi-an-jiaotong-university-",
        notes=socket.gethostname(),
        name="WV3-SSDiff_ARConv",
        dir=run_dir,
        job_type="training",
        mode="offline",  # 可以改为"offline"如果不想上传到wandb
        reinit=True
    )
    
    logger.configure(dir='/'.join([rootPath, 'logs/train_logs/']))

    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    
    if Resume:
        path_checkpoint = args.resume_from
        logger.log("path_checkpoint: ", path_checkpoint)
        model.load_state_dict(torch.load(path_checkpoint, map_location=lambda storage, loc: storage.cuda()), strict=False)

    session = DataSession(args)
    
    """  pansharpening  """
    # 再次强制设置，确保使用 OTPNet 数据集
    if isinstance(args.dataset, dict):
        args.dataset['train'] = 'wv3_otpnet'
        args.dataset['valid'] = 'wv3_otpnet'
        args.dataset['test'] = 'test_wv3_multiExm1_otpnet.h5'
    
    print(f"[DEBUG] About to call get_dataloader with args.dataset['train'] = '{args.dataset['train']}'")
    print(f"[DEBUG] args.dataset = {args.dataset}")
    data, _ , _= session.get_dataloader(args.dataset['train'], False, None)    
    
    # 计算 lms 和 gt 的 PSNR，确认使用的是 OTPNet 数据集
    print("\n========== Verifying Dataset (LMS vs GT PSNR) ==========")
    psnr_list = []
    sample_count = min(10, len(data))  # 采样前10个样本
    
    data_iter = iter(data)
    for i in range(sample_count):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
            
        lms = batch['lms'].cpu().numpy().astype(np.float64)
        gt = batch['gt'].cpu().numpy().astype(np.float64)
        
        # 打印形状信息（仅第一次）
        if i == 0:
            print(f"Batch shapes - lms: {lms.shape}, gt: {gt.shape}")
        
        # 统一转换为 [B, H, W, C] 格式
        if len(gt.shape) == 4:
            if gt.shape[1] > 1 and gt.shape[1] <= 16:  # [B, C, H, W]
                gt = np.transpose(gt, (0, 2, 3, 1))  # 转为 [B, H, W, C]
        
        if len(lms.shape) == 4:
            if lms.shape[1] > 1 and lms.shape[1] <= 16:  # [B, C, H, W]
                lms = np.transpose(lms, (0, 2, 3, 1))  # 转为 [B, H, W, C]
        
        # 确保形状匹配
        if lms.shape != gt.shape:
            print(f"Warning: Shape mismatch - lms: {lms.shape}, gt: {gt.shape}")
            continue
        
        # 确定数据范围
        max_val = 1.0 if (lms.max() <= 1.1 and gt.max() <= 1.1) else 2047.0
        
        # 计算每个样本的 PSNR
        for b in range(lms.shape[0]):
            mse = np.mean((lms[b] - gt[b]) ** 2)
            psnr = float('inf') if mse == 0 else 10 * np.log10((max_val ** 2) / mse)
            psnr_list.append(psnr)
    
    if psnr_list:
        psnr_array = np.array(psnr_list)
        print(f"Sampled {len(psnr_list)} patches from training dataset")
        print(f"LMS vs GT PSNR: mean={psnr_array.mean():.4f} dB, "
              f"std={psnr_array.std():.4f}, min={psnr_array.min():.4f}, max={psnr_array.max():.4f}")
        print(f"Expected OTPNet PSNR: ~38-39 dB (if using OTPNet dataset)")
        print(f"Expected original PSNR: ~27-29 dB (if using original dataset)")
        print("=" * 55 + "\n")
    
    # 重新创建 dataloader 用于训练（因为上面已经遍历了一次）
    data, _ , _= session.get_dataloader(args.dataset['train'], False, None)
 
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        device=args.device,
        batch_size=args.samples_per_gpu,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        rootPath=rootPath,
        resume_epoch=resume_epoch,  # 传递resume_epoch
    ).run_loop()
    
    # 完成wandb记录
    wandb.finish()


if __name__ == "__main__":
    # ============ 继续训练模式 ============
    # 1. 在 configs/option_DPM_pansharpening.py 中设置 ckpt_model_path
    # 2. 设置 resume_epoch > 5000 以保护ARConv的reserved_NXY不被覆盖
    # 3. 建议降低学习率 (lr=1e-4 或 1e-5)
    # 示例：
    main(device='cuda:4', Resume=True, resume_epoch=80000)
    
    # ============ 正常训练模式 ============
    #main()
