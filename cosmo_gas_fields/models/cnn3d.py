"""3D ResNet classifier + trivial-moment baselines.

Used as a *measurement instrument*: how well can a strong 3D CNN tell apart
crops of the true density field drawn from different simulation variants? That
number is an empirical ceiling for any downstream "which variant" claim. The
moment baselines (mean / mean+var / mean+var+skew+kurt) guard against the
ceiling being carried by low-order statistics alone: if a 1-scalar baseline
matches the ResNet, the ResNet is decorative.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    """ResNet BasicBlock in 3D (Conv3d / BN3d / ReLU, identity or 1x1x1 shortcut)."""

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """ResNet-style 3D classifier with configurable block counts and widths."""

    def __init__(
        self,
        block_counts: Tuple[int, int, int, int] = (2, 2, 2, 2),
        channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        stem_channels: Optional[int] = None,
        in_channels: int = 1,
        num_classes: int = 4,
        stem_stride: int = 2,
        stem_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if stem_channels is None:
            stem_channels = channels[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=stem_kernel_size,
                      stride=stem_stride, padding=stem_kernel_size // 2, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(stem_channels, channels[0], block_counts[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], block_counts[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], block_counts[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], block_counts[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels[3] * BasicBlock3D.expansion, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) -> logits (B, num_classes)."""
        h = self.stem(x)
        h = self.stage4(self.stage3(self.stage2(self.stage1(h))))
        return self.fc(self.avg_pool(h).flatten(1))


def resnet18_3d(in_channels: int = 1, num_classes: int = 4) -> ResNet3D:
    """ResNet-18 3D with halved channel widths (~10-12 M params at 32^3 crops)."""
    return ResNet3D(block_counts=(2, 2, 2, 2), channels=(32, 64, 128, 256),
                    stem_channels=32, in_channels=in_channels,
                    num_classes=num_classes, stem_stride=2, stem_kernel_size=7)


class MeanOverdensityBaseline(nn.Module):
    """1-scalar baseline: crop.mean() -> FC. If this matches the ResNet, the task is trivial."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(2).mean(dim=2))


class MeanVarianceBaseline(nn.Module):
    """2-scalar baseline: [mean, var] -> FC."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        return self.fc(torch.cat([flat.mean(dim=2), flat.var(dim=2, unbiased=False)], dim=1))


class MeanVarSkewKurtBaseline(nn.Module):
    """4-scalar baseline: [mean, var, skew, excess kurtosis] -> FC(4 -> 64 -> classes)."""

    _EPS = 1e-8

    def __init__(self, num_classes: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    @staticmethod
    def _moments(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        mean = flat.mean(dim=2)
        diff = flat - mean.unsqueeze(2)
        var = (diff * diff).mean(dim=2)
        std = torch.sqrt(var.clamp_min(MeanVarSkewKurtBaseline._EPS))
        mu3 = (diff ** 3).mean(dim=2)
        mu4 = (diff ** 4).mean(dim=2)
        skew = mu3 / (std ** 3)
        kurt_excess = mu4 / (var.clamp_min(MeanVarSkewKurtBaseline._EPS) ** 2) - 3.0
        return torch.cat([mean, var, skew, kurt_excess], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(self._moments(x))))
