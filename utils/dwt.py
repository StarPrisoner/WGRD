# utils/dwt.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTForward(nn.Module):
    """优化后的Haar小波分解"""
    def __init__(self, pad_mode='reflect'):
        super().__init__()
        # 定义可学习的小波核
        self.ll = nn.Parameter(torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2)
        self.lh = nn.Parameter(torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 2)
        self.hl = nn.Parameter(torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 2)
        self.hh = nn.Parameter(torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2)
        self.pad_mode = pad_mode

    def forward(self, x):
        # 自动填充保证可分解
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 1), mode=self.pad_mode)

        # 分离通道计算
        ll = F.conv2d(x, self.ll.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)
        lh = F.conv2d(x, self.lh.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)
        hl = F.conv2d(x, self.hl.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)
        hh = F.conv2d(x, self.hh.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)

        return ll, (lh, hl, hh)


class DWTInverse(nn.Module):
    """可微分的小波重建"""
    def __init__(self, in_channels):
        super().__init__()
        # 动态转置卷积
        self.conv_t = nn.ConvTranspose2d(
            in_channels * 4, in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=in_channels
        )
        # 初始化核参数
        kernel = torch.zeros(4, 1, 4, 4)
        kernel[0, 0, 1::2, 1::2] = 1  # LL
        kernel[1, 0, 1::2, ::2] = 1   # LH
        kernel[2, 0, ::2, 1::2] = 1   # HL
        kernel[3, 0, ::2, ::2] = 1    # HH
        self.conv_t.weight.data = kernel.repeat(in_channels, 1, 1, 1) / 2
        self.conv_t.weight.requires_grad = False

    def forward(self, coeffs):
        ll, highs = coeffs
        lh, hl, hh = highs
        # 通道维度合并
        x = torch.cat([ll, lh, hl, hh], dim=1)
        return self.conv_t(x)