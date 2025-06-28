import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, "../")  # run under the current directory
from common.network import *
from loguru import logger
from jaxtyping import jaxtyped, Float, Int64 as Long
from typeguard import typechecked as typechecker
from typing import Any, Self, Tuple, List, Dict

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}

def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def rotate_input(module, x, last_x, pad):
    pred = 0
    for r in [0,1,2,3]:
        rot_input=F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')
        rot_last_input=None if last_x is None else F.pad(torch.rot90(last_x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')
        rot_output=module(rot_input, rot_last_input)
        output=torch.rot90(rot_output, (4 - r) % 4, [2, 3])
        quantized=round_func(output * 127)
        pred+=quantized
    return pred

# --- CBAM Module ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        return ca * sa

# --- 각 branch별 SRNet (간단한 버전) ---
class SRNet_CBAM(nn.Module):
    def __init__(self, nf, scale=4, act='relu', num_blocks=8):
        super().__init__()
        layers = []
        # 첫 블록만 3채널 입력
        layers += [nn.Conv2d(3, nf, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks-1):
            layers += [nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.cbam = CBAM(nf)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(nf, 3, 3, padding=1)
        )

    def forward(self, x):
        out = self.body(x)
        out = self.cbam(out)
        out = self.upsample(out)
        return out

# --- Classifier (CNN + pooling + FC + softmax) ---
class CNNFreqClassifier(nn.Module):
    def __init__(self, num_branches=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # fc는 입력 shape을 보고 최초 forward에서 생성!
        self.fc1 = None
        self.fc2 = None
        self.num_branches = num_branches

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.conv(x)                # [B, 32, H/4, W/4]
        feat_flat = feat.reshape(feat.size(0), -1)  # [B, N]
        in_features = feat_flat.shape[1]
        # fc1/fc2 lazy init
        if self.fc1 is None or self.fc1.in_features != in_features:
            device = feat_flat.device
            self.fc1 = nn.Linear(in_features, 64).to(device)
            self.fc2 = nn.Linear(64, self.num_branches).to(device)
        x = F.relu(self.fc1(feat_flat))
        logits = self.fc2(x)
        weights = F.softmax(logits, dim=1)  # [B, 3]
        return weights.unsqueeze(-1).unsqueeze(-1)




# --- Main model ---
class MultiWeightedSRNets_FreqClassifier2(nn.Module):
    def __init__(self, num_samplers=3, sample_size=5, nf=64, scale=4, stages=1, act='relu'):
        super().__init__()
        self.num_samplers = num_samplers
        self.stages = stages
        self.sample_size = sample_size

        self.srnets = nn.ModuleList([
            SRNet_CBAM(nf=nf, scale=scale, num_blocks=4),   # A: 가볍게
            SRNet_CBAM(nf=nf, scale=scale, num_blocks=8),   # B: 기본
            SRNet_CBAM(nf=nf, scale=scale, num_blocks=12),  # C: 깊게
        ])
        self.classifier = CNNFreqClassifier(num_branches=num_samplers)

    def forward(self, x, phase='train'):
        input_x = x
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        outs = [net(x) for net in self.srnets]  # 각 [B, 3, H, W]
        outs = torch.stack(outs, dim=1)         # [B, num_samplers, 3, H, W]
        weights = self.classifier(x)            # [B, num_samplers, 1, 1]
        weights = weights.unsqueeze(-1)         # [B, num_samplers, 1, 1, 1]
        out = (outs * weights).sum(dim=1)       # [B, 3, H, W]
        input_up = F.interpolate(input_x, size=out.shape[-2:], mode='bilinear', align_corners=False)
        if input_up.size(1) == 1:
            input_up = input_up.repeat(1, 3, 1, 1)
        out = out + input_up
        return out