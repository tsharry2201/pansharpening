# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/6/4 15:08
# @Author  : Xiao Wu
# @reference: 
#
import argparse
import sys
from udl_vis.Basis.option import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='DPM_ps'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from pancollection.configs.configs import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0].replace('\\', '/')
        
        # 硬编码数据集路径
        #data_dir = '/home/zelilin/data/pansharpening/SSDiff_main/dataset'
        data_dir = '/data2/user/zelilin/pansharpening/SSDiff_main/dataset'
        # 继续训练时，设置checkpoint路径
        # 例如：ckpt_model_path = "/home/zelilin/data/pansharpening/SSDiff_main/results/10-17-03-13/model065000.pt"
        ckpt_model_path = "/data2/user/zelilin/pansharpening/SSDiff_main/results/11-07-23-43/model080000.pt"
        
        # 修改为您的模型路径
        # EMA模型（推荐）：results/MM-DD-HH-MM/ema_0.9999_XXXXXX.pt
        # 或主模型：results/MM-DD-HH-MM/modelXXXXXX.pt
        #test_model_path = "/home/zelilin/data/pansharpening/SSDiff_main/results/11-05-22-52/model120000.pt"
        #test_model_path = "/data2/user/zelilin/pansharpening/SSDiff_main/results/11-07-23-43/model070000.pt"       
        test_model_path = "/data2/user/zelilin/pansharpening/SSDiff_main/results/11-07-23-39/model110000.pt"       
        #10-14-22-28代表是+ARConv的    10-15-17-45是没有ARConv的  10-15-22-29是调整ARConv中fix为1e4的 10-17-03-13是fix为1e3的  10-17-12-19是fix为5e3的 1600是继续训0313的
        #11042121 是新版在OTPNet下进行的，fix 为5e3  2129fix为2e3
        #11052252 是SSDiff无OTPNet下训练的  11061941 是使用了ARConv+OPTNet数据集 fix为2e3 都是有问题的
        #11071332 是fix为2e3的 11071334是fix为5e3 搞错了，这个是用原来的训的
        #增加了skip_connection lms的模型 11072339 fix 为5e3   2343 是fix为2e3的 
        #11090005 是继续训练2343 从8e4 epoch开始
        parser = argparse.ArgumentParser(description='PyTorch Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        parser.add_argument('--data_dir', metavar='DIR', default=data_dir,
                            help='path to dataset')
        # * Training
        parser.add_argument('--lr', default=2e-4, type=float)  # 1e-4 2e-4 原来使用的是1e-3
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--crop_batch_size', default=24, type=int)  # 减小以适应24GB显存
        parser.add_argument('--samples_per_gpu', default=16, type=int,              # batch_size 从20降至8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=500, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=3407, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=12000, type=int) # 12000
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--device', default='cpu', type=str)
        parser.add_argument('--resume_from',
                            default=ckpt_model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='iDPM', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
    # 使用本地 WV3 的 otpnet 版本作为默认 dataset（train/valid/test）
    # 注意：本 repo dataset 目录下存在文件 train_wv3_otpnet.h5 / valid_wv3_otpnet.h5
        parser.add_argument('--dataset', default={'train': 'wv3_otpnet', 'valid': 'wv3_otpnet', 'test': 'test_wv3_multiExm1_otpnet.h5'},
                choices=[None, 'wv2', 'wv3', 'wv3_otpnet', 'wv4', 'qb', 'gf2',
                     'wv3_OrigScale_multiExm1.h5', 'wv3_multiExm1.h5'],
                help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
            help="performing evalution for patch2entire")
        parser.add_argument('--dim', default=32, type=int)
        parser.add_argument('--dim_head', default=16, type=int)
        parser.add_argument('--se_ratio_mlp', default=0.5, type=float)
        parser.add_argument('--se_ratio_rb', default=0.5, type=float)
        parser.add_argument('--ms_dim', default=8, type=int) # qb/gf2:4,  wv3:8
        parser.add_argument('--pan_dim', default=1, type=int)
        parser.add_argument('--model_channels', default=128, type=int)
        
        
        # * DPM
        parser.add_argument('--schedule_sampler', default="uniform"),
        parser.add_argument('--lr_anneal_steps', default=0),
        parser.add_argument('--microbatch', default=-1),  # -1 disables microbatches
        parser.add_argument('--ema_rate', default="0.9999"),  # comma-separated list of EMA values
        parser.add_argument('--use_fp16', default=False),
        parser.add_argument('--fp16_scale_growth', default=1e-3),

        parser.add_argument('--image_size', default=64)
        parser.add_argument('--num_channels', default=128)  # 128

        parser.add_argument('--num_res_blocks', default=2)
        parser.add_argument('--num_heads', default=4)
        parser.add_argument('--num_heads_upsample', default=-1)
        parser.add_argument('--attention_resolutions', default="16,8")
        parser.add_argument('--dropout', default=0.0)
        parser.add_argument('--learn_sigma', default=False)
        parser.add_argument('--sigma_small', default=False)
        parser.add_argument('--class_cond', default=False)
        parser.add_argument('--diffusion_steps', default=1000)
        parser.add_argument('--noise_schedule', default="cosine")
        parser.add_argument('--timestep_respacing', default="ddim1000")
        parser.add_argument('--use_kl', default=False)
        parser.add_argument('--predict_xstart', default=True)
        parser.add_argument('--rescale_timesteps', default=True)
        parser.add_argument('--rescale_learned_sigmas', default=True)
        parser.add_argument('--use_scale_shift_norm', default=False)
        parser.add_argument('--use_checkpoint', default=False)

        # *
        parser.add_argument('--log_interval', default=100)
        parser.add_argument('--save_interval', default=200)
        parser.add_argument('--resume_checkpoint', default="")
        parser.add_argument('--weight_decay', default=0)

        # * sample
        parser.add_argument('--clip_denoised', default=True)
        parser.add_argument('--num_samples', default=2047)
        parser.add_argument('--test_samples_per_gpu', default=16)
        parser.add_argument('--use_ddim', default=True)
        parser.add_argument('--model_path', default=test_model_path) # model_path

        # 修改：使用空参数列表避免与train_ssdiff_distill.py的参数解析冲突
        # 仅使用默认值，不从命令行读取
        args = parser.parse_args(args=[])
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = 'Test'
        cfg.merge_args2cfg(args)
        cfg.workflow = [('train', 1)]
        cfg.img_range = 2047.0
        cfg.dataloader_name = "PanCollection_dataloader"
        
        # 强制设置使用 OTPNet 数据集
        if hasattr(cfg, 'dataset'):
            if isinstance(cfg.dataset, dict):
                cfg.dataset['train'] = 'wv3_otpnet'
                cfg.dataset['valid'] = 'wv3_otpnet'
                cfg.dataset['test'] = 'test_wv3_multiExm1_otpnet.h5'
        if hasattr(args, 'dataset'):
            if isinstance(args.dataset, dict):
                args.dataset['train'] = 'wv3_otpnet'
                args.dataset['valid'] = 'wv3_otpnet'
                args.dataset['test'] = 'test_wv3_multiExm1_otpnet.h5'

        self.merge_from_dict(cfg)
