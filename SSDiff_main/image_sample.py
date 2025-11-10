from functools import partial
import os
import time
import torch as th

from torch.utils.data import DataLoader

from configs.option_DPM_pansharpening import parser_args
from scipy.io import savemat
import einops
import numpy as np
import datetime
import torch.distributed as dist
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
# import spectral as spy
from improved_diffusion import logger
from utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pancollection.common.psdata import PansharpeningSession as DataSession

rootPath = os.path.abspath(os.path.dirname(__file__))


def main(
    device='cuda:5',
    crop_batch_size=8,
    timestep_respacing="ddim10",
    test_dataset=None  # 新增：指定测试数据集
    ):


    args = parser_args()

    if device is not None:
        args.device = device
    th.cuda.set_device(args.device)
    
    if crop_batch_size is not None:
        args.crop_batch_size = crop_batch_size
    if timestep_respacing is not None:
        args.timestep_respacing = timestep_respacing
    if test_dataset is not None:
        args.dataset['test'] = test_dataset
    

    logger.configure(dir='/'.join([rootPath, 'logs/sample_logs/']))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(th.load(args.model_path, map_location=lambda storage, loc: storage.cuda()))
    model.cuda()
    
    # 设置 epoch > 1000，确保 ARConv 使用训练好的固定卷积核（reserved_NXY）
    # 而不是动态计算卷积核大小
    model.set_epoch(10001)
    logger.log("Set epoch to 10001 for using fixed ARConv kernel size")
    
    model.eval()

    logger.log("sampling...")
    logger.log("model_path: ", args.model_path)
    all_images = []
    session = DataSession(args)
    data, _ = session.get_eval_dataloader(args.dataset['test'], False)    
    dl = iter(data)
    print("batch_size: ", args.crop_batch_size)


    data4gt = []
    data4lms = []
    data4ms = []
    data4pan = []
    psnr_list = []
    # image_num = len(data)
    image_num = 20
    print("image_num:", image_num)
    
    tic = time.time()
    for i in range(image_num):
        batch = next(dl)
        pan_ori, lms_ori, ms_ori, gt = batch['pan'], batch['lms'], batch['ms'], batch['gt']
        # 动态获取图像尺寸，支持 Reduced-Resolution (256x256) 和 Full-Resolution (512x512)
        gt =  einops.rearrange(gt, 'b k1 k2 c -> b c k1 k2')
        
        data4gt.append(gt[0])
        data4lms.append(lms_ori[0])
        data4ms.append(ms_ori[0])
        data4pan.append(pan_ori[0])

        lms_np = lms_ori[0].cpu().numpy().astype(np.float64)
        gt_np = gt[0].cpu().numpy().astype(np.float64)
        if lms_np.max() <= 1.1 and gt_np.max() <= 1.1:
            max_val = 1.0
        else:
            max_val = 2047.0
        mse = np.mean((lms_np - gt_np) ** 2)
        psnr = float('inf') if mse == 0 else 10 * np.log10((max_val ** 2) / mse)
        psnr_list.append(psnr)
        print(f"[LMS vs GT] image {i}: PSNR={psnr:.4f} dB, MSE={mse:.4f}")

        pan, lms, ms = map(lambda x: x.cuda(), (pan_ori, lms_ori, ms_ori))
        logger.log(f"test [{i}]/[{image_num}],  {args.timestep_respacing}", pan.shape, lms.shape, ms.shape)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        kwargs_data = {"lms": lms, "pan": pan, "ms": ms}
        
        # 创建与 shape 参数匹配的噪声初始化（不是与 lms 匹配！）
        shape = (args.crop_batch_size, args.ms_dim, args.image_size, args.image_size)
        noise_init = th.zeros(*shape, device=lms.device)  # 从零开始（无残差）
        
        # 如果想要小幅度随机噪声，可以改为：
        #noise_init = th.randn(*shape, device=lms.device) * 0.01
        
        sample = sample_fn(
                    model,
                    shape=shape,
                    noise=noise_init,
                    model_kwargs=kwargs_data,
                    clip_denoised=args.clip_denoised,
                    progress=False)

        # 动态获取输出尺寸，支持不同分辨率
        sample_d =  einops.rearrange(sample, '1 c k1 k2 -> k1 k2 c')
        sample_d = sample_d.contiguous()  # sample[:, [4,2,0]]
        sample_d = (sample_d * 2047.).clamp(0, 2047)
        d = dict(  # [b, h, w, c], wv3 [0, 2047]
            sr=[sample_d.cpu().numpy()],
        )

        sample = sample.contiguous()  # sample[:, [4,2,0]]
        sample = (sample * 2047.).clamp(0, 2047)
        gathered_samples = [sample]
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_images) * args.crop_batch_size} samples")

    print(time.time() - tic)
    print(len(all_images))
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if psnr_list:
        psnr_array = np.array(psnr_list)
        print(
            f"[LMS vs GT] PSNR stats -> mean: {psnr_array.mean():.4f} dB, "
            f"std: {psnr_array.std():.4f}, min: {psnr_array.min():.4f}, max: {psnr_array.max():.4f}"
        )

    # 从模型路径中提取模型名称
    model_path = args.model_path
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # 例如: "model016000" 或 "ema_0.9999_016000"
    model_dir = os.path.basename(os.path.dirname(model_path))  # 例如: "10-14-17-18"
    
    # 自动判断分辨率类型和图像尺寸
    dataset_name = args.dataset['test']
    if 'OrigScale' in dataset_name or 'origscale' in dataset_name.lower():
        resolution_type = 'full'
        img_size = arr[0].shape[0] if len(arr) > 0 else 512  # 从实际数据获取尺寸
    else:
        resolution_type = 'reduced'
        img_size = arr[0].shape[0] if len(arr) > 0 else 256
    
    d = dict(  # [b, h, w, c], wv3 [0, 2047]
            gt=[sample.cpu().numpy()*2047 for sample in data4gt],
            sr=[sample for sample in arr],
            lms=[sample.cpu().numpy() for sample in data4lms],
            ms=[sample.cpu().numpy() for sample in data4ms],
            pan=[sample.cpu().numpy() for sample in data4pan],
            model_name=model_name,
            model_dir=model_dir,
            model_path=model_path,
        )

    
    loca=datetime.datetime.now().strftime('%m-%d-%H-%M')
    out_path = '/'.join([rootPath, f'logs/samp_{resolution_type}_{len(arr)}_{img_size}_{model_name}_{str(loca)}.mat'])
    
    savemat(out_path, d)
    logger.log(f"saving to {out_path}")
    logger.log(f"model: {model_name} from {model_dir}")
    print("save result")
    logger.log("sampling complete")
    
    return out_path


if __name__ == "__main__":
    # 选择测试数据集类型
    # Reduced-Resolution: 'test_wv3_multiExm1.h5' (默认)
    # Full-Resolution: 'test_wv3_OrigScale_multiExm1.h5'
    
    # Reduced-Resolution 评估
    #out_path = main()
    out_path = main(test_dataset='test_wv3_multiExm1_otpnet.h5')
    # Full-Resolution 评估（取消注释以使用）
    #out_path = main(test_dataset='test_wv3_OrigScale_multiExm1.h5')

