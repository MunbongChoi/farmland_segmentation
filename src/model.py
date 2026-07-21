"""Semantic segmentation models (kept separate from train and infer entry points)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .utils.tf_checkpoint import detect_tf_segformer_depths, load_tf_segformer_h5


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


class SegFormerAdapter(nn.Module):
    """Expose Hugging Face SegFormer as full-resolution logits.

    Hugging Face returns logits below the input resolution. The rest of this
    project expects every model to return ``B,C,H,W`` logits that align exactly
    with the rasterized JSON mask.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # Conv2d(768, 3, 1) may produce a logically contiguous gradient whose
        # singleton-dimension strides differ from the parameter. DDP then warns
        # and copies it into the reduction bucket. Normalize only this tiny
        # classifier gradient before DDP's reducer hook observes it.
        decode_head = getattr(model, "decode_head", None)
        classifier = getattr(decode_head, "classifier", None)
        if isinstance(classifier, nn.Conv2d) and classifier.weight.requires_grad:
            classifier.weight.register_hook(_contiguous_gradient)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=inputs).logits
        if logits.shape[-2:] != inputs.shape[-2:]:
            logits = F.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def _contiguous_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Return a fresh standard-stride gradient for DDP bucket compatibility."""
    return gradient.clone(memory_format=torch.contiguous_format)


def _load_segformer_model(
    checkpoint: str,
    revision: str | None,
    input_channels: int,
    num_classes: int,
    class_names: list[str],
    ignore_index: int,
    pretrained: bool,
    local_files_only: bool,
    use_safetensors: bool,
    h5_architecture: str,
) -> nn.Module:
    """Load a SegFormer backbone/config while keeping Transformers optional."""
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as error:
        raise RuntimeError(
            "SegFormer를 사용하려면 정상적인 transformers/safetensors 설치가 필요합니다. "
            "`python -m pip install --upgrade --force-reinstall numpy transformers safetensors`를 실행하세요."
        ) from error

    if pretrained and input_channels != 3:
        raise ValueError("사전학습 SegFormer는 RGB 3채널 입력만 지원합니다. model.input_channels=3을 사용하세요.")
    if len(class_names) != num_classes:
        raise ValueError("dataset.class_names 길이는 model.num_classes와 같아야 합니다.")

    id2label = {index: name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in id2label.items()}
    source_options = {"revision": revision, "local_files_only": local_files_only}
    try:
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.suffix.lower() in {".h5", ".hdf5"}:
            if not pretrained:
                raise ValueError("H5 checkpoint를 지정할 때는 model.pretrained=true여야 합니다.")
            architecture = h5_architecture.lower().replace("-", "")
            architecture_depths = {"b2": (3, 4, 6, 3), "b4": (3, 8, 27, 3)}
            if architecture not in architecture_depths:
                raise ValueError(f"지원하지 않는 TensorFlow SegFormer 구조입니다: {h5_architecture}")
            detected_depths = detect_tf_segformer_depths(checkpoint_path)
            expected_depths = architecture_depths[architecture]
            if detected_depths != expected_depths:
                raise ValueError(
                    f"H5 구조와 model.h5_architecture가 다릅니다: detected_depths={detected_depths}, "
                    f"configured={architecture}({expected_depths})"
                )
            hf_config = SegformerConfig(
                num_channels=input_channels,
                num_encoder_blocks=4,
                depths=list(expected_depths),
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[64, 128, 320, 512],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                classifier_dropout_prob=0.1,
                drop_path_rate=0.1,
                layer_norm_eps=1e-6,
                decoder_hidden_size=768,
                reshape_last_stage=True,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                semantic_loss_ignore_index=ignore_index,
            )
            model = SegformerForSemanticSegmentation(hf_config)
            report = load_tf_segformer_h5(model, checkpoint_path)
            logging.getLogger(__name__).info(
                "TensorFlow SegFormer H5 로드: path=%s loaded=%d skipped_classifier=%d",
                checkpoint_path,
                len(report.loaded),
                len(report.skipped_classifier),
            )
            return model

        # Load and then mutate the config explicitly. Passing num_labels next to
        # an ImageNet/ADE id2label map makes Transformers emit a false mismatch
        # warning before it applies the new three-class label contract.
        hf_config = SegformerConfig.from_pretrained(checkpoint, **source_options)
        hf_config.num_labels = num_classes
        hf_config.id2label = id2label
        hf_config.label2id = label2id
        hf_config.semantic_loss_ignore_index = ignore_index
        if pretrained:
            return SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                config=hf_config,
                ignore_mismatched_sizes=True,
                use_safetensors=use_safetensors,
                **source_options,
            )
        hf_config.num_channels = input_channels
        return SegformerForSemanticSegmentation(hf_config)
    except (OSError, ValueError) as error:
        source = "로컬 캐시" if local_files_only else "Hugging Face Hub 또는 로컬 캐시"
        raise RuntimeError(f"SegFormer checkpoint '{checkpoint}'를 {source}에서 불러오지 못했습니다: {error}") from error


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build the configured model and validate channel/class contracts."""
    model_config = config["model"]
    dataset_config = config.get("dataset", {})
    for key in ("input_channels", "num_classes"):
        if key in dataset_config and int(model_config[key]) != int(dataset_config[key]):
            raise ValueError(f"model.{key}와 dataset.{key}가 다릅니다.")

    name = str(model_config["name"]).lower().replace("-", "").replace("_", "")
    input_channels = int(model_config["input_channels"])
    num_classes = int(model_config["num_classes"])
    if name == "unet":
        if bool(model_config.get("pretrained", False)):
            raise ValueError("현재 U-Net은 사전학습 가중치를 제공하지 않습니다. model.pretrained=false를 사용하세요.")
        return UNet(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=int(model_config.get("base_channels", 32)),
            bilinear=bool(model_config.get("bilinear", True)),
        )
    if name == "segformer":
        model = _load_segformer_model(
            checkpoint=str(model_config.get("checkpoint", "nvidia/segformer-b4-finetuned-ade-512-512")),
            revision=str(model_config["revision"]) if model_config.get("revision") else None,
            input_channels=input_channels,
            num_classes=num_classes,
            class_names=list(dataset_config.get("class_names", [str(index) for index in range(num_classes)])),
            ignore_index=int(dataset_config.get("ignore_index", 255)),
            pretrained=bool(model_config.get("pretrained", True)),
            local_files_only=bool(model_config.get("local_files_only", False)),
            use_safetensors=bool(model_config.get("use_safetensors", True)),
            h5_architecture=str(model_config.get("h5_architecture", "b4")),
        )
        if bool(model_config.get("gradient_checkpointing", False)):
            model.gradient_checkpointing_enable()
        return SegFormerAdapter(model)
    raise ValueError(f"지원하지 않는 모델입니다: {model_config['name']} (지원: unet, segformer)")


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
