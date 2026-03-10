from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock3D(out_ch),
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock3D(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        x = self.proj(x)
        if x.shape[-3:] != skip.shape[-3:]:
            # Safe crop if odd dimensions appear.
            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            skip = skip[
                :,
                :,
                dz // 2 : dz // 2 + x.shape[-3],
                dy // 2 : dy // 2 + x.shape[-2],
                dx // 2 : dx // 2 + x.shape[-1],
            ]
        return self.conv(torch.cat([x, skip], dim=1))


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 16,
        output_activation: str = "identity",
    ):
        super().__init__()
        self.d1 = DownBlock(in_channels, base_channels)
        self.d2 = DownBlock(base_channels, base_channels * 2)
        self.d3 = DownBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock3D(base_channels * 8),
        )

        self.u3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.u2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.u1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(base_channels // 2, out_channels, kernel_size=1),
        )
        allowed = {"identity", "relu", "softplus"}
        if output_activation not in allowed:
            raise ValueError(f"output_activation must be one of {sorted(allowed)}, got {output_activation!r}")
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial = x.shape[-3:]
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        x = self.bottleneck(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.head(x)
        if self.output_activation == "relu":
            x = F.relu(x)
        elif self.output_activation == "softplus":
            x = F.softplus(x)
        # Force exact size match with the original volume for non-divisible dimensions.
        if x.shape[-3:] != input_spatial:
            x = F.interpolate(x, size=input_spatial, mode="trilinear", align_corners=False)
        return x
