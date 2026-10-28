# Load galaxy images and build deterministic training augmentations.

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms

from .data import SPLITS, SampleRecord


class GaussianNoise(nn.Module):
    def __init__(self, std: float) -> None:
        super().__init__()
        self.std = float(std)

    def forward(self, image: Tensor) -> Tensor:
        return (image + torch.randn_like(image) * self.std).clamp(0.0, 1.0)


def _append_morphology_augmentations(
    operations: list[Any],
    augment: Mapping[str, Any],
) -> None:
    if float(augment["random_rotation_probability"]) > 0:
        rotation = transforms.RandomRotation(
            float(augment["rotation_degrees"]),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        operations.append(
            transforms.RandomApply(
                [rotation],
                p=float(augment["random_rotation_probability"]),
            )
        )
    if float(augment["horizontal_flip_probability"]) > 0:
        operations.append(
            transforms.RandomHorizontalFlip(
                p=float(augment["horizontal_flip_probability"])
            )
        )
    if float(augment["vertical_flip_probability"]) > 0:
        operations.append(
            transforms.RandomVerticalFlip(p=float(augment["vertical_flip_probability"]))
        )


def build_image_transform(
    config: Mapping[str, Any], *, training: bool, weak: bool = False
) -> transforms.Compose:
    dataset = config["dataset"]
    size = int(dataset["input_size"])
    augment = config["augment"]
    operations: list[Any] = [
        transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.BILINEAR
        )
    ]
    if training:
        _append_morphology_augmentations(operations, augment)
        if not weak and float(augment["gaussian_blur_probability"]) > 0:
            operations.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=int(augment["gaussian_blur_kernel_size"]),
                            sigma=tuple(
                                float(value) for value in augment["gaussian_blur_sigma"]
                            ),
                        )
                    ],
                    p=float(augment["gaussian_blur_probability"]),
                )
            )
    operations.append(transforms.ToTensor())
    if training and not weak and float(augment["gaussian_noise_probability"]) > 0:
        operations.append(
            transforms.RandomApply(
                [GaussianNoise(float(augment["gaussian_noise_std"]))],
                p=float(augment["gaussian_noise_probability"]),
            )
        )
    operations.append(transforms.Normalize(dataset["mean"], dataset["std"]))
    return transforms.Compose(operations)


def build_scp_weak_transform(config: Mapping[str, Any]) -> transforms.Compose:
    return build_image_transform(config, training=True, weak=True)


class OnlineImageDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        transform: transforms.Compose,
        *,
        include_structure: bool,
        include_label: bool = True,
        transform_seed: int | None = None,
    ) -> None:
        self.records = tuple(records)
        self.transform = transform
        self.include_structure = bool(include_structure)
        self.include_label = bool(include_label)
        self.transform_seed = None if transform_seed is None else int(transform_seed)

    def set_transform_seed(self, seed: int | None) -> None:
        self.transform_seed = None if seed is None else int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB").copy()
        if self.transform_seed is None:
            transformed = self.transform(image)
        else:
            digest = hashlib.sha256(
                f"{self.transform_seed}\0{record.sample_id}".encode("utf-8")
            ).digest()
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(
                    int.from_bytes(digest[:8], "big", signed=False)
                )
                transformed = self.transform(image)
        sample = {
            "image": transformed,
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.include_label:
            sample["label"] = torch.tensor(record.label, dtype=torch.long)
        if self.include_structure:
            sample["structure"] = torch.tensor(record.structure, dtype=torch.float32)
        return sample


def build_image_dataset(
    config: Mapping[str, Any], records: Sequence[SampleRecord], split: str
) -> OnlineImageDataset:
    if split not in SPLITS:
        raise ValueError(f"未知 split: {split}")
    training = split == "train"
    return OnlineImageDataset(
        records,
        build_image_transform(config, training=training),
        include_structure=training,
    )
