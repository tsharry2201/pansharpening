import torch.nn as nn

try:
    from ..third_party.ARConv.ARConv import ARConv2d
except Exception:
    ARConv2d = None

def make_conv2d(in_ch, out_ch, k=3, s=1, p=None, bias=True, use_arconv=False, **kw):
    if p is None:
        p = k // 2
    if use_arconv and ARConv2d is not None and s == 1:
        # ARConv 通常先从 stride=1 的特征提取卷积替换起步
        return ARConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias, **kw)
    else:
        return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
