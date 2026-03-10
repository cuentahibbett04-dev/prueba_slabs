from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ALLOWED_ARCHS = ("resunet3d", "attention_unet3d", "swin_unetr")


def _pad_sizes_for_multiple_3d(spatial: tuple[int, int, int], multiple: int) -> tuple[int, int, int]:
    d, h, w = spatial
    pd = (multiple - (d % multiple)) % multiple
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    return pd, ph, pw


class SwinUNETRWithAutoPad(nn.Module):
    """Pad to valid SwinUNETR shape and crop back to original size."""

    def __init__(self, backbone: nn.Module, output_activation: str = "identity", multiple: int = 32):
        super().__init__()
        self.backbone = backbone
        self.output_activation = output_activation
        self.multiple = multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[-3:]
        pd, ph, pw = _pad_sizes_for_multiple_3d(spatial, self.multiple)
        if pd or ph or pw:
            # F.pad expects (w_left, w_right, h_left, h_right, d_left, d_right) for 3D volumes.
            x = F.pad(x, (0, pw, 0, ph, 0, pd))

        y = self.backbone(x)
        if pd or ph or pw:
            y = y[:, :, : spatial[0], : spatial[1], : spatial[2]]

        if self.output_activation == "relu":
            y = F.relu(y)
        elif self.output_activation == "softplus":
            y = F.softplus(y)
        return y


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


class AttentionGate3D(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_ch),
        )
        self.w_x = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        a = self.w_g(gate)
        b = self.w_x(skip)
        alpha = self.psi(a + b)
        return skip * alpha


class AttentionUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.gate = AttentionGate3D(out_ch, skip_ch, inter_ch=max(out_ch // 2, 8))
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
        gated_skip = self.gate(x, skip)
        return self.conv(torch.cat([x, gated_skip], dim=1))


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


class AttentionUNet3D(nn.Module):
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

        self.u3 = AttentionUpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.u2 = AttentionUpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.u1 = AttentionUpBlock(base_channels * 2, base_channels, base_channels)

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
        if x.shape[-3:] != input_spatial:
            x = F.interpolate(x, size=input_spatial, mode="trilinear", align_corners=False)
        return x


def build_model(
    arch: str,
    *,
    in_channels: int,
    out_channels: int,
    base_channels: int,
    output_activation: str,
) -> nn.Module:
    arch_l = arch.lower()
    if arch_l == "resunet3d":
        return ResUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            output_activation=output_activation,
        )
    if arch_l == "attention_unet3d":
        return AttentionUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            output_activation=output_activation,
        )
    if arch_l == "swin_unetr":
        try:
            from monai.networks.nets import SwinUNETR  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Architecture 'swin_unetr' requires MONAI. Install with: pip install monai"
            ) from exc
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=max(base_channels, 12),
            use_checkpoint=False,
            spatial_dims=3,
        )
        allowed = {"identity", "relu", "softplus"}
        if output_activation not in allowed:
            raise ValueError(f"Unsupported output_activation={output_activation!r}")
        return SwinUNETRWithAutoPad(model, output_activation=output_activation, multiple=32)
    raise ValueError(f"arch must be one of {ALLOWED_ARCHS}, got {arch!r}")


def load_model_from_checkpoint(
    ckpt: dict,
    *,
    in_channels: int = 2,
    out_channels: int = 1,
) -> nn.Module:
    arch = str(ckpt.get("arch", "resunet3d"))
    base_channels = int(ckpt.get("base_channels", 16))
    output_activation = str(ckpt.get("output_activation", "identity"))
    model = build_model(
        arch,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        output_activation=output_activation,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model
