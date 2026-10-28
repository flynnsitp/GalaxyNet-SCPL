# Load and run the local DINOv3 ConvNeXt-Small backbone.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .artifacts import sha256_file


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    name: str
    depths: tuple[int, ...]
    dims: tuple[int, ...]
    filename: str
    sha256: str

    @property
    def feature_dim(self) -> int:
        return self.dims[-1]


BACKBONES: Mapping[str, BackboneSpec] = {
    "dinov3_convnext_small": BackboneSpec(
        "dinov3_convnext_small",
        (3, 3, 27, 3),
        (96, 192, 384, 768),
        "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
        "296db49dcbd622625befd3fc23318cbbcd98049f4c4b0cc026463de6bcd24952",
    ),
}


class _ChannelFirstLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).pow(2).mean(1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(1.0e-6 * torch.ones(channels))

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.dwconv(value).permute(0, 2, 3, 1)
        value = self.pwconv2(self.act(self.pwconv1(self.norm(value))))
        return residual + (self.gamma * value).permute(0, 3, 1, 2)


class DINOv3ConvNeXt(nn.Module):
    def __init__(self, depths: Sequence[int], dims: Sequence[int]) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, dims[0], 4, stride=4),
                    _ChannelFirstLayerNorm(dims[0]),
                ),
                *[
                    nn.Sequential(
                        _ChannelFirstLayerNorm(dims[index]),
                        nn.Conv2d(dims[index], dims[index + 1], 2, stride=2),
                    )
                    for index in range(3)
                ],
            ]
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[_ConvNeXtBlock(dims[index]) for _ in range(depths[index])]
                )
                for index in range(4)
            ]
        )
        self.norm = nn.LayerNorm(dims[-1], eps=1.0e-6)
        self.norms = nn.ModuleList(
            [nn.Identity(), nn.Identity(), nn.Identity(), self.norm]
        )
        self.head = nn.Identity()
        self.embed_dim = int(dims[-1])
        self.embed_dims = tuple(int(value) for value in dims)
        self.num_features = self.embed_dim
        self.n_storage_tokens = 0

    def forward_stage_features(self, images: Tensor) -> tuple[Tensor, ...]:
        value = images
        outputs: list[Tensor] = []
        for downsample, stage in zip(self.downsample_layers, self.stages, strict=True):
            value = stage(downsample(value))
            outputs.append(value)
        return tuple(outputs)


class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: DINOv3ConvNeXt,
        spec: BackboneSpec,
        weights_path: Path,
        train_mode: str = "frozen",
    ) -> None:
        super().__init__()
        if train_mode not in {"frozen", "full"}:
            raise ValueError("backbone.train_mode 必须为 frozen 或 full")
        self.backbone = backbone
        self.spec = spec
        self.weights_path = weights_path
        self.train_mode = train_mode
        self.backbone.requires_grad_(train_mode == "full")
        if train_mode == "frozen":
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv3FeatureExtractor":
        super().train(mode)
        if self.train_mode == "frozen":
            self.backbone.eval()
        return self

    def forward_multiscale(self, images: Tensor) -> tuple[Tensor, ...]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOv3 输入必须为 [B,3,H,W]")
        if self.train_mode == "frozen":
            with torch.no_grad():
                return tuple(
                    feature.detach()
                    for feature in self.backbone.forward_stage_features(images)
                )
        return self.backbone.forward_stage_features(images)

    def trainable_stage_parameters(self) -> tuple[tuple[nn.Parameter, ...], ...]:
        groups: list[list[nn.Parameter]] = [[] for _ in range(4)]
        assigned: set[int] = set()
        for index, modules in enumerate(
            zip(self.backbone.downsample_layers, self.backbone.stages, strict=True)
        ):
            for module in modules:
                for parameter in module.parameters():
                    if parameter.requires_grad and id(parameter) not in assigned:
                        groups[index].append(parameter)
                        assigned.add(id(parameter))
        for parameter in self.backbone.parameters():
            if parameter.requires_grad and id(parameter) not in assigned:
                groups[-1].append(parameter)
                assigned.add(id(parameter))
        expected = {
            id(parameter)
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        }
        if assigned != expected or any(not group for group in groups):
            raise RuntimeError("ConvNeXt stage参数分组不完整")
        return tuple(tuple(group) for group in groups)

    def metadata(self, input_size: int) -> dict[str, object]:
        return {
            **asdict(self.spec),
            "weights_path": str(self.weights_path),
            "input_size": int(input_size),
            "output_grid": [int(input_size) // 32, int(input_size) // 32],
            "feature_level": "four_stage_maps",
            "train_mode": self.train_mode,
            "frozen": self.train_mode == "frozen",
            "stage_grids": [
                [int(input_size) // stride, int(input_size) // stride]
                for stride in (4, 8, 16, 32)
            ],
        }


def resolve_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONES[str(name)]
    except KeyError as exc:
        raise ValueError(f"未知 backbone: {name}; 可选 {sorted(BACKBONES)}") from exc


def build_backbone(
    name: str, weights_dir: str | Path, *, train_mode: str = "frozen"
) -> DINOv3FeatureExtractor:
    spec = resolve_backbone_spec(name)
    weights_path = Path(weights_dir).expanduser().resolve() / spec.filename
    if not weights_path.is_file():
        raise FileNotFoundError(f"本地 DINOv3 权重不存在: {weights_path}")
    actual_sha = sha256_file(weights_path)
    if actual_sha != spec.sha256:
        raise RuntimeError(
            f"DINOv3 权重 SHA256 不匹配: expected={spec.sha256}, actual={actual_sha}"
        )
    backbone = DINOv3ConvNeXt(spec.depths, spec.dims)
    try:
        state = torch.load(
            weights_path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as exc:
        raise RuntimeError(f"无法读取 DINOv3 权重 {weights_path}: {exc}") from exc
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise TypeError("DINOv3 checkpoint 必须是扁平 tensor state dict")
    try:
        incompatible = backbone.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"DINOv3 strict-load 失败: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("DINOv3 strict-load 返回不兼容键")
    return DINOv3FeatureExtractor(backbone, spec, weights_path, train_mode=train_mode)
