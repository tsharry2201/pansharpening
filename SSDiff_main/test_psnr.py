import h5py, numpy as np
path = '/data2/user/zelilin/pansharpening/SSDiff_main/dataset/PanCollection/test_data/test_wv3_multiExm1_otpnet.h5'
with h5py.File(path, 'r') as f:
    lms = f['lms'][:].astype(np.float64)
    gt = f['gt'][:].astype(np.float64)
max_val = 2047.0
mse = ((lms - gt) ** 2).reshape(lms.shape[0], -1).mean(axis=1)
psnr = 10 * np.log10((max_val ** 2) / mse)
print('count', psnr.size)
print('mean {:.4f} dB std {:.4f} min {:.4f} max {:.4f}'.format(psnr.mean(), psnr.std(), psnr.min(), psnr.max()))