# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from groksar.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
import torch
import math


class DCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, height, width):
        super(DCT2DSpatialTransformLayer, self).__init__()
        self.dct_x = DCT2DSpatialTransformLayer_x(width)
        self.dct_y = DCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # x = x.double()
        y = self.dct_x(x)
        y = self.dct_y(y)

        return y


class IDCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, height, width):
        super(IDCT2DSpatialTransformLayer, self).__init__()
        self.idct_x = IDCT2DSpatialTransformLayer_x(width)
        self.idct_y = IDCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        y = self.idct_x(x)
        y = self.idct_y(y)
        # y = y.float()

        return y


class FastDCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, height, width):
        super(FastDCT2DSpatialTransformLayer, self).__init__()
        self.dct_x = FastDCT2DSpatialTransformLayer_x(width)
        self.dct_y = FastDCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # x = x.double()
        # b,c,h,w
        y = self.dct_x(x.permute(0, 3, 2, 1))  # channel 和 width 互换 b,w,h,c
        y = self.dct_y(y.permute(0, 2, 1, 3))  # width 和 height 互换 b,h,w,c
        y = y.permute(0, 3, 1, 2)  # 复原 b,c,h,w

        return y


class FastIDCT2DSpatialTransformLayer(nn.Module):
    def __init__(self, height, width):
        super(FastIDCT2DSpatialTransformLayer, self).__init__()
        self.idct_x = FastIDCT2DSpatialTransformLayer_x(width)
        self.idct_y = FastIDCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # b,c,h,w
        y = self.idct_x(x.permute(0, 3, 2, 1))  # channel 和 width 互换 b,w,h,c
        y = self.idct_y(y.permute(0, 2, 1, 3))  # width 和 height 互换 b,h,w,c
        y = y.permute(0, 3, 1, 2)  # 复原 b,c,h,w
        # y = y.float()

        return y


class DCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(DCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + j) * v / width) / math.sqrt(width)
                if v != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, 1, x.shape[3]).expand_as(x)
            dct_components.append(dct_component.sum(3).unsqueeze(3))

        result = torch.concat(dct_components, dim=3)

        return result


class FastDCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(FastDCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + j) * v / width) / math.sqrt(width)
                if v != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result


class IDCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(IDCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + v) * j / width) / math.sqrt(width)
                if j != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, 1, x.shape[3]).expand_as(x)
            dct_components.append(dct_component.sum(3).unsqueeze(3))

        result = torch.concat(dct_components, dim=3)

        return result


class FastIDCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(FastIDCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        # dct_filter = torch.zeros(width, width, dtype=torch.float64)
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + v) * j / width) / math.sqrt(width)
                if j != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result


class DCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(DCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + i) * k / height) / math.sqrt(height)
                if k != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, x.shape[2], 1).expand_as(x)
            dct_components.append(dct_component.sum(2).unsqueeze(2))

        result = torch.concat(dct_components, dim=2)

        return result


class FastDCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(FastDCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + i) * k / height) / math.sqrt(height)
                if k != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result


class IDCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(IDCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + k) * i / height) / math.sqrt(height)
                if i != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, x):
        dct_components = []

        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, x.shape[2], 1).expand_as(x)
            dct_components.append(dct_component.sum(2).unsqueeze(2))

        result = torch.concat(dct_components, dim=2)

        return result


class FastIDCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(FastIDCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        # dct_filter = torch.zeros(height, height, dtype=torch.float64)
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + k) * i / height) / math.sqrt(height)
                if i != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y

        return dct_filter

    def forward(self, input):
        result = F.conv2d(input, self.weight.unsqueeze(2).unsqueeze(3))
        return result






class DCTFreqMoE(nn.Module):
    def __init__(self, expert_num, height, width):
        super().__init__()
        self.expert_num = expert_num
        self.H = height
        self.W = width
        self.freq_len = height * width

        # learnable band boundaries（和 FFT 版一样）
        self.band_boundaries = nn.Parameter(torch.rand(expert_num - 1))

        # gating network（和你原代码保持一致）
        self.gating_network = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len//4),
            nn.ReLU(),
            nn.Linear(self.freq_len//4,  self.expert_num)
        )

        # build radial frequency order
        sorted_idx, inv_idx = self.build_radial_order(height, width, device="cpu")
        self.register_buffer("sorted_idx", sorted_idx)
        self.register_buffer("inv_idx", inv_idx)

    def build_radial_order(self, H, W, device):
        u = torch.arange(H, device=device).view(-1, 1)
        v = torch.arange(W, device=device).view(1, -1)
        r = torch.sqrt(u ** 2 + v ** 2)
        r_flat = r.view(-1)
        sorted_idx = torch.argsort(r_flat)
        inv_idx = torch.argsort(sorted_idx)

        return sorted_idx, inv_idx

    def forward(self, x):
        """
        x: [B,C,H,W]
        """
        B, C, H, W = x.shape

        # x = x1.clone().reshape(b, h * w, c)
        # y = y.mean(2, keepdim=True) + y.max(2, keepdim=True)[0]

        X_flat = x.view(B, C, -1)  # [B,C,HW]
        X_flat = X_flat.mean(1, keepdim=True)+X_flat.max(1, keepdim=True)[0]
        # ---- 按径向频率排序（2D → 1D）----
        X_freq = X_flat[:, :, self.sorted_idx]  # [B,C,HW]
        total_freq_size = X_freq.size(-1)
        # ---- Frequency band boundaries（完全复用你的代码）----
        boundaries = torch.sigmoid(self.band_boundaries)
        boundaries, _ = torch.sort(boundaries)

        boundaries = torch.cat([
            torch.tensor([0.0], device=x.device),
            boundaries,
            torch.tensor([1.0], device=x.device)
        ])

        indices = (boundaries * total_freq_size).long()
        indices[-1] = total_freq_size

        components = []
        for i in range(self.expert_num):
            start_idx = indices[i].item()
            end_idx = indices[i + 1].item()

            freq_mask = torch.zeros_like(X_freq)
            if end_idx > start_idx:
                freq_mask[:, :, start_idx:end_idx] = 1.0

            expert_component = X_freq * freq_mask
            components.append(expert_component.unsqueeze(-1))
        # import pdb
        # pdb.set_trace()
        components = torch.cat(components, dim=-1)  # [B,C,HW,E]

        # ---- gating（和 FFT 版完全一致）----
        freq_magnitude = X_freq.abs()
        gating_input = freq_magnitude.mean(dim=1).squeeze()  # [B,HW]

        gating_scores = torch.softmax(
            self.gating_network(gating_input),
            dim=-1
        )  # [B,E]

        gating_scores = gating_scores.view(B, 1, 1, self.expert_num)

        # ---- MoE fusion ----

        # import pdb
        # pdb.set_trace()
        X_freq_out = torch.sum(components * gating_scores, dim=-1)  # [B,C,HW]

        # ---- 1D → 2D 还原 ----
        X_flat_out = torch.zeros_like(X_flat)
        X_flat_out[:, :, self.sorted_idx] = X_freq_out
        X_dct_out = X_flat_out.view(B, 1, H, W)
        x_out = X_dct_out.expand_as(x) * x
        return x_out




