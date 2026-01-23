import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .WGRD_model import WGRD_Unet
from .WGRD_parts import *
from utils.dwt import DWTForward, DWTInverse


# 1. Basic Components

class HybridAttention(nn.Module):
    """Combines channel and spatial attention to reduce redundancy."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(channels, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.shape
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x = x * y
        # Spatial Attention
        return x * self.sigmoid(self.spatial_conv(x))


class DirectionalConvBranch(nn.Module):
    """Directional enhancement branch for rain streaks."""

    def __init__(self, dim, direction_type, reduction=4):
        super().__init__()
        # Physical property mapping
        kernels = {"horizontal": (1, 3), "vertical": (3, 1), "diagonal": (3, 3)}
        paddings = {"horizontal": (0, 1), "vertical": (1, 0), "diagonal": (1, 1)}

        self.conv = nn.Conv2d(dim, dim, kernels[direction_type], padding=paddings[direction_type])
        self.attention = HybridAttention(dim, reduction)
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.attention(self.conv(x))))


# 2. Core Algorithm Blocks

class HierarchicalWaveBlock(nn.Module):
    """Hierarchical Wavelet Block (HWB)."""

    def __init__(self, dim, time_emb_dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.dwt = DWTForward()

        # Subband branches
        self.low_freq_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.high_freq_branches = nn.ModuleList([
            DirectionalConvBranch(dim, d, reduction) for d in ["horizontal", "vertical", "diagonal"]
        ])

        # Fusion logic
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(4 * dim, 4 * dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(4 * dim, 4 * dim, 1),
            ChannelSplitActivation(dim)
        )
        self.rain_attn = nn.Sequential(
            nn.Conv2d(4 * dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, 4 * dim, 1), nn.Sigmoid()
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 4 * dim)) if time_emb_dim else None
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape
        ll, hf = self.dwt(x)
        lh, hl, hh = [h.squeeze(2) for h in hf[0].chunk(3, dim=2)]

        # Feature extraction and alignment
        feats = [self.low_freq_conv(ll)] + [branch(h) for branch, h in zip(self.high_freq_branches, [lh, hl, hh])]
        feats = [F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False) for f in feats]

        concated = torch.cat(feats, dim=1)
        if self.time_proj and time_emb is not None:
            concated += self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)

        gates = self.fusion_gate(concated)
        mask = self.rain_attn(concated)

        fused = 0
        for i in range(4):
            scale = 1.5 if i > 0 else 1.0  # High-frequency enhancement
            fused += feats[i] * gates[:, i * C:(i + 1) * C] * mask[:, i * C:(i + 1) * C] * scale

        return x + self.res_scale * fused


class CrossScaleFusion(nn.Module):
    """Cross-Scale Fusion Module (CSF)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attn = HybridAttention(out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )
        self.fusion_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, prev_feat, curr_feat):
        if prev_feat.shape[2:] != curr_feat.shape[2:]:
            prev_feat = F.interpolate(prev_feat, size=curr_feat.shape[2:], mode='bilinear')
        fused = self.attn(self.conv(torch.cat([prev_feat, curr_feat], dim=1)))
        return curr_feat + self.fusion_scale * fused


# 3. Helpers

class ChannelSplitActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([torch.softmax(x[:, :self.dim], dim=1), torch.sigmoid(x[:, self.dim:])], dim=1)


# 4. Main Architecture

class WGRD_WaveUnet(WGRD_Unet):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), **kwargs):
        super().__init__(dim=dim, dim_mults=dim_mults, **kwargs)
        self.time_emb_dim = dim * 4
        self.TIME_DEPENDENT_MODULES = (ResnetBlock, HierarchicalWaveBlock)

        # Initialize custom components
        self.wave_blocks = nn.ModuleList([HierarchicalWaveBlock(dim * m, self.time_emb_dim) for m in dim_mults])

        down_dims = [dim * m for m in dim_mults]
        up_dims = list(reversed(down_dims))[1:]
        self.cross_fusions = nn.ModuleList([
            CrossScaleFusion(in_channels=down_dims[-1 - i] + up_dims[i], out_channels=up_dims[i])
            for i in range(len(up_dims))
        ])

    def _align_size(self, x, target_size=None, mode='pad'):
        """Unified handling for size padding and cropping."""
        factor = 32  # Factor set to 32

        if mode == 'pad':
            h, w = x.shape[2:]
            # Calculate padding height and width to be divisible by 32
            ph = (factor - h % factor) % factor
            pw = (factor - w % factor) % factor
            # Use reflection padding to avoid boundary artifacts
            return F.pad(x, (0, pw, 0, ph), mode='reflect'), (ph, pw)

        # Crop back to original size
        return x[:, :, :target_size[0], :target_size[1]]

    def forward(self, x, time, input_cond=None, mask_cond=None):
        # 0. Preparation and Padding
        h_orig, w_orig = x.shape[2:]
        x, (ph, pw) = self._align_size(x)
        if input_cond is not None:
            # Note: input_cond pad size is already calculated, apply padding directly
            input_cond = F.pad(input_cond, (0, pw, 0, ph), mode='reflect')
        if mask_cond is not None:
            mask_cond = F.pad(mask_cond, (0, pw, 0, ph), mode='reflect')

        # 1. Initial Convolution and Condition Injection
        if self.input_condition: x = torch.cat((x, input_cond), dim=1)
        if self.mask_condition: x = torch.cat((x, mask_cond), dim=1)
        x = self.init_conv(x)
        residual, t = x.clone(), self.time_mlp(time)

        # 2. Encoder
        down_features, pyramid_features = [], []
        for wave_block, down_stage in zip(self.wave_blocks, self.downs):
            x = down_stage[0](x, t)
            pyramid_features.append(x)
            x = down_stage[1](x, t)
            pyramid_features.append(x)
            if len(down_stage) > 3: x = down_stage[2](x)  # Attention

            x = down_stage[-1](x)  # Downsample
            x = wave_block(x, t)
            down_features.append(x)

        # 3. Bottleneck
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, t)), t)

        # 4. Decoder
        for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
            for block in [block1, block2]:
                skip = pyramid_features.pop()
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear')
                x = torch.cat((x, skip), dim=1)
                x = block(x, t)

            x = upsample(attn(x))
            if idx < len(self.cross_fusions):
                x = self.cross_fusions[idx](down_features[len(self.wave_blocks) - 1 - idx], x)

        # 5. Output Head
        if residual.shape[2:] != x.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear')
        x = self.final_res_block(torch.cat([x, residual], dim=1), t)
        output = self.final_conv(x)

        return self._align_size(output, (h_orig, w_orig), mode='crop')