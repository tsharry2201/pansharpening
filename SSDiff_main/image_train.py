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
    device='cuda:1',
    Resume = False,
    resume_epoch = 0  # 继续训练时设置正确的epoch值
    ):
    
    args = parser_args()
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
    main(device='cuda:1', Resume=True, resume_epoch=52000)
    
    # ============ 正常训练模式 ============
    #main()
