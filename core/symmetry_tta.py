# Resolve symmetry rules and aggregate test-time predictions.

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import yaml
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as transform_functional
from .data import GZD5_CLASSES


def default_symmetry_tta_config(
    config: Mapping[str, Any], *, classwise: bool | None = None
) -> dict[str, Any]:
    if classwise is None:
        classwise = (
            config["train"]["mode"] == "scp_from_checkpoint"
            and float(config["dataset"]["label_fraction"]) == 0.1
        )
    rules = {
        "angles_degrees": [22.5 * index for index in range(16)],
        "horizontal_flip_values": [False, True],
    }
    if not classwise:
        return {
            **rules,
            "aggregation": "mean_probabilities",
            "original_probability_weight": 0.0,
        }
    return {
        **rules,
        "groups": {
            name: {"angle_step": step, "horizontal_flip_values": flips}
            for name, step, flips in (
                ("c4", 90.0, [False]),
                ("c4_flip", 90.0, [True]),
                ("c8_flip", 45.0, [True]),
                ("c16_flip", 22.5, [True]),
                ("d8", 45.0, [False, True]),
                ("d16", 22.5, [False, True]),
            )
        },
        "class_rules": {
            name: {"group": group, "weight": weight}
            for name, group, weight in (
                ("barred_spiral", "d16", 0.64),
                ("cigar_shaped_smooth", "d8", 0.2),
                ("disturbed", "c16_flip", 0.47),
                ("edge_on_no_bulge", "c4", 0.91),
                ("edge_on_with_bulge", "d16", 0.62),
                ("in_between_round_smooth", "c8_flip", 0.7),
                ("merging", "c8_flip", 0.76),
                ("round_smooth", "c4_flip", 1.0),
                ("unbarred_loose_spiral", "c16_flip", 0.68),
                ("unbarred_tight_spiral", "d16", 0.53),
            )
        },
        "validation_selected_calibration": True,
    }


def load_symmetry_tta_config(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = dict(source) if isinstance(source, Mapping) else yaml.safe_load(
        Path(source).expanduser().resolve().read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("TTA配置顶层必须是mapping")
    uniform_fields = {
        "angles_degrees",
        "horizontal_flip_values",
        "aggregation",
        "original_probability_weight",
    }
    r30_fields = {
        "angles_degrees",
        "horizontal_flip_values",
        "groups",
        "class_rules",
        "validation_selected_calibration",
    }
    if set(payload) == uniform_fields:
        if payload["angles_degrees"] != [22.5 * index for index in range(16)]:
            raise ValueError("通用TTA仅支持D16固定旋转集合")
        if payload["horizontal_flip_values"] != [False, True]:
            raise ValueError("通用TTA固定使用未翻转和水平翻转")
        if payload["aggregation"] != "mean_probabilities":
            raise ValueError("通用TTA固定使用等权概率平均")
        original_weight = payload["original_probability_weight"]
        if (
            isinstance(original_weight, bool)
            or not math.isfinite(float(original_weight))
            or not 0.0 <= float(original_weight) <= 1.0
        ):
            raise ValueError("通用TTA原图概率权重必须在[0,1]")
        return payload
    if set(payload) != r30_fields:
        raise ValueError("TTA配置字段不匹配")
    if tuple(payload["angles_degrees"]) != tuple(22.5 * i for i in range(16)):
        raise ValueError("R30 TTA固定使用16角度")
    if payload["horizontal_flip_values"] != [False, True]:
        raise ValueError("R30 TTA固定使用两种翻转，共32视图")
    if payload["validation_selected_calibration"] is not True:
        raise ValueError("R30 TTA必须记录规则由validation选择")
    if set(payload["class_rules"]) != set(GZD5_CLASSES):
        raise ValueError("R30 TTA类别规则不完整")
    groups = expand_symmetry_groups(
        payload["angles_degrees"], payload["horizontal_flip_values"], payload["groups"]
    )
    for name, rule in payload["class_rules"].items():
        if set(rule) != {"group", "weight"} or rule["group"] not in groups:
            raise ValueError(f"{name}的TTA规则不正确")
        weight = rule["weight"]
        if (
            isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or not 0.0 <= float(weight) <= 1.0
        ):
            raise ValueError(f"{name}的TTA权重越界")
    return payload


def _view_key(angle: float, horizontal_flip: bool) -> str:
    return f"{float(angle):g}_{int(horizontal_flip)}"


def expand_symmetry_groups(
    angles: Sequence[float],
    available_flips: Sequence[bool],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    available = {
        _view_key(float(angle), bool(flip))
        for angle in angles
        for flip in available_flips
    }
    for name, spec in specs.items():
        if set(spec) != {"angle_step", "horizontal_flip_values"}:
            raise ValueError(f"对称群{name}字段不匹配")
        step = float(spec["angle_step"])
        flips = spec["horizontal_flip_values"]
        if not math.isfinite(step) or step <= 0.0 or not flips:
            raise ValueError(f"对称群{name}的角度步长或翻转列表无效")
        if any(type(value) is not bool for value in flips):
            raise ValueError(f"对称群{name}的翻转值必须为bool")
        keys = tuple(
            _view_key(float(angle), bool(flip))
            for angle in angles
            if math.isclose(
                float(angle) / step,
                round(float(angle) / step),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            for flip in flips
        )
        if not keys or not set(keys).issubset(available):
            raise ValueError(f"对称群{name}引用了不可用视图")
        result[str(name)] = keys
    return result


def fuse_classwise_symmetry_logits(
    view_logits: Mapping[str, Tensor],
    groups: Mapping[str, Sequence[str]],
    class_rules: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str],
) -> Tensor:
    if set(class_rules) != set(class_names) or "0_0" not in view_logits:
        raise ValueError("类别规则或原图视图不完整")
    original = view_logits["0_0"].float()
    expected_shape = (original.shape[0], len(class_names))
    if original.ndim != 2 or tuple(original.shape) != expected_shape:
        raise ValueError("原图logits形状与类别数量不匹配")
    group_logits: dict[str, Tensor] = {}
    for name, keys in groups.items():
        if not keys or any(key not in view_logits for key in keys):
            raise ValueError(f"对称群{name}缺少视图logits")
        values = [view_logits[key].float() for key in keys]
        if any(tuple(value.shape) != expected_shape for value in values):
            raise ValueError(f"对称群{name}的logits形状不一致")
        group_logits[name] = torch.stack(values).mean(dim=0)
    columns: list[Tensor] = []
    for class_index, class_name in enumerate(class_names):
        rule = class_rules[class_name]
        group = str(rule["group"])
        weight = float(rule["weight"])
        columns.append(
            (1.0 - weight) * original[:, class_index]
            + weight * group_logits[group][:, class_index]
        )
    return torch.stack(columns, dim=1)


def mean_view_log_probabilities(
    view_logits: Mapping[str, Tensor], original_probability_weight: float = 0.0
) -> Tensor:
    if not view_logits:
        raise ValueError("TTA视图不能为空")
    values = [value.float() for value in view_logits.values()]
    shape = values[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in values):
        raise ValueError("TTA视图logits必须为相同形状的[N,C]")
    weight = float(original_probability_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("原图概率权重必须在[0,1]")
    probabilities = torch.stack(values).softmax(dim=-1).mean(dim=0)
    if weight:
        if "0_0" not in view_logits:
            raise ValueError("原图概率残差需要0_0视图")
        original = view_logits["0_0"].float().softmax(dim=-1)
        probabilities = weight * original + (1.0 - weight) * probabilities
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


def _view_transform(config: Mapping[str, Any], angle: float, horizontal_flip: bool):
    size = int(config["dataset"]["input_size"])

    def rotate_and_flip(image):
        result = transform_functional.rotate(
            image,
            angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        return transform_functional.hflip(result) if horizontal_flip else result

    return transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Lambda(rotate_and_flip),
            transforms.ToTensor(),
            transforms.Normalize(config["dataset"]["mean"], config["dataset"]["std"]),
        ]
    )
