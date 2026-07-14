"""U-Net model construction (kept separate from train and infer entry points)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two convolution, batch-normalization, and ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Down(nn.Module):
    """Spatial downsampling followed by a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Up(nn.Module):
    """Upsample, concatenate the encoder feature, and refine."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, decoder: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        decoder = self.up(decoder)
        diff_y = encoder.size(2) - decoder.size(2)
        diff_x = encoder.size(3) - decoder.size(3)
        decoder = F.pad(decoder, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([encoder, decoder], dim=1))


class UNet(nn.Module):
    """Configurable semantic U-Net returning unnormalized class logits."""

    def __init__(self, input_channels: int, num_classes: int, base_channels: int = 32, bilinear: bool = True) -> None:
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build the configured model and validate channel/class contracts."""
    model_config = config["model"]
    if str(model_config["name"]).lower() != "unet":
        raise ValueError(f"지원하지 않는 모델입니다: {model_config['name']}")
    if bool(model_config.get("pretrained", False)):
        raise ValueError("현재 U-Net은 사전학습 가중치를 제공하지 않습니다. model.pretrained=false를 사용하세요.")
    dataset_config = config.get("dataset", {})
    for key in ("input_channels", "num_classes"):
        if key in dataset_config and int(model_config[key]) != int(dataset_config[key]):
            raise ValueError(f"model.{key}와 dataset.{key}가 다릅니다.")
    return UNet(
        input_channels=int(model_config["input_channels"]),
        num_classes=int(model_config["num_classes"]),
        base_channels=int(model_config.get("base_channels", 32)),
        bilinear=bool(model_config.get("bilinear", True)),
    )


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable

