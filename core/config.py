# Resolve configuration defaults and validate training settings.

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping
import yaml
from .data import GZD5_CLASSES, resolve_structural_keys

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/scp_pct_1.yaml"


_DEFAULTS = {
    "experiment": {"seed": [2026]},
    "reproducibility": {"strict": True, "deterministic_augmentation": True},
    "dataset": {
        "input_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "classes": list(GZD5_CLASSES),
        "expected_raw_counts": {"train": 14190, "val": 1773, "test": 1773},
        "expected_valid_counts": {"train": 14186, "val": 1773, "test": 1773},
    },
    "augment": {
        "random_resized_crop_scale_min": 1.0,
        "right_angle_rotation_only": False,
        "random_rotation_probability": 0.5,
        "rotation_degrees": 180.0,
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5,
        "gaussian_blur_probability": 0.5,
        "gaussian_blur_kernel_size": 5,
        "gaussian_blur_sigma": [0.1, 2.0],
        "gaussian_noise_probability": 0.5,
        "gaussian_noise_std": 0.2,
    },
    "backbone": {"name": "dinov3_convnext_small", "train_mode": "full"},
    "adapter": {
        "type": "mma", "hidden_dim": 512, "embedding_dim": 256, "attention_heads": 8,
    },
    "train": {
        "evaluation_batch_size": 64,
        "num_workers": 16,
        "pin_memory": True,
        "weight_decay": 0.05,
        "warmup_epochs": 0,
        "gradient_clip_norm": 1.0,
        "amp_dtype": "bfloat16",
    },
    "evaluation": {"tta": False},
}
_SCP_DEFAULTS = {
    "profile": "r30",
    "class_balance_max_multiplier": 2.0,
    "generalized_cross_entropy_q": 0.0,
    "visual_temperature": 0.5,
}
_C4_DEFAULTS = {
    "prototype_mode": "labeled_c4",
    "prototype_logit_scale": 8.0,
    "loss_ramp_epochs": 3,
    "refresh_tta": "none",
    "student_classifier_initialization": "source_head",
    "labeled_resampling": "missing_pseudo_classes",
    "pseudo_train_augmentation": "weak",
    "visual_temperature": 1.0,
    "pseudo_target_mode": "joint_soft",
    "pseudo_symmetry": "hflip",
    "resampled_labeled_loss_weight": 0.5,
}


def _merge_defaults(defaults: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(defaults))
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_defaults(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_config(values: Mapping[str, Any]) -> dict[str, Any]:
    config = _merge_defaults(_DEFAULTS, values)
    train = config.get("train")
    if isinstance(train, Mapping) and train.get("mode") == "scp_from_checkpoint":
        scp = config.get("scp")
        if isinstance(scp, Mapping):
            defaults = dict(_SCP_DEFAULTS)
            default_mode = (
                "labeled_c4" if config.get("dataset", {}).get("label_fraction") == 0.01
                else "none"
            )
            if scp.get("prototype_mode", default_mode) == "labeled_c4":
                defaults.update(_C4_DEFAULTS)
            if scp.get("pseudo_target_mode", defaults.get("pseudo_target_mode", "hard")) == "joint_soft":
                defaults["pseudo_target_hard_weight"] = 0.25
            config["scp"] = _merge_defaults(defaults, scp)
    return config


class _ConfigDumper(yaml.SafeDumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        return super().represent_sequence(tag, sequence, flow_style=True)


def _keys(value: Mapping[str, Any], expected: str, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected.split()):
        raise ValueError(f"{name}字段不匹配，允许字段：{expected}")


def _keys_with_optional(
    value: Mapping[str, Any], required: str, optional: str, name: str
) -> None:
    required_keys = set(required.split())
    optional_keys = set(optional.split())
    if (
        not isinstance(value, Mapping)
        or not required_keys.issubset(value)
        or not set(value).issubset(required_keys | optional_keys)
    ):
        allowed = " ".join((required, optional)).strip()
        raise ValueError(f"{name}字段不匹配，允许字段：{allowed}")


def _number(
    value: Any,
    name: str,
    lower: float,
    upper: float = math.inf,
    *,
    integer: bool = False,
    inclusive: bool = True,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是数值") from error
    if (
        isinstance(value, bool)
        or not math.isfinite(result)
        or (result < lower if inclusive else result <= lower)
        or result > upper
        or (integer and result != int(result))
    ):
        raise ValueError(f"{name}数值越界")
    return result


def experiment_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    values = config.get("experiment", {}).get("seed")
    if not isinstance(values, list) or not values:
        raise ValueError("experiment.seed必须是非空整数列表")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("experiment.seed必须只包含非负整数")
    if len(values) != len(set(values)):
        raise ValueError("experiment.seed不能包含重复值")
    return tuple(values)


def single_experiment_seed(config: Mapping[str, Any]) -> int:
    seeds = experiment_seeds(config)
    if len(seeds) != 1:
        raise ValueError("当前操作必须使用仅含一个seed的resolved config")
    return seeds[0]


def validate_config(config: Mapping[str, Any]) -> None:
    mode = config.get("train", {}).get("mode")
    if mode not in {"staged", "scp_from_checkpoint"}:
        raise ValueError("train.mode仅支持staged/scp_from_checkpoint")
    is_scp = mode == "scp_from_checkpoint"
    top = "experiment reproducibility dataset augment backbone adapter train output evaluation"
    _keys(config, top + (" scp" if is_scp else ""), "配置顶层")
    _keys(config["experiment"], "name seed", "experiment")
    experiment_seeds(config)
    if (
        not isinstance(config["experiment"]["name"], str)
        or not config["experiment"]["name"].strip()
    ):
        raise ValueError("experiment.name必须是非空字符串")
    _keys(
        config["reproducibility"],
        "strict deterministic_augmentation",
        "reproducibility",
    )
    if any(type(v) is not bool for v in config["reproducibility"].values()):
        raise ValueError("reproducibility开关必须是bool")
    data = config["dataset"]
    _keys_with_optional(
        data,
        "root split_manifest structural_json structural_feature_count input_size mean std classes expected_raw_counts expected_valid_counts label_fraction",
        "labeled_seed",
        "dataset",
    )
    if "labeled_seed" in data and (
        type(data["labeled_seed"]) is not int or data["labeled_seed"] < 0
    ):
        raise ValueError("dataset.labeled_seed必须是非负整数")
    for key in ("root", "split_manifest", "structural_json"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"dataset.{key}必须为路径")
    if tuple(data["classes"]) != GZD5_CLASSES:
        raise ValueError("数据必须遵循固定 GZD5-10 分类顺序")
    resolve_structural_keys(data)
    size = _number(
        data["input_size"], "dataset.input_size", 0, integer=True, inclusive=False
    )
    if size % 32:
        raise ValueError("input_size必须能被32整除")
    for key in ("mean", "std"):
        if len(data[key]) != 3:
            raise ValueError(f"dataset.{key}必须包含三个数")
        for value in data[key]:
            _number(
                value,
                f"dataset.{key}",
                -math.inf if key == "mean" else 0,
                inclusive=key == "mean",
            )
    for key in ("expected_raw_counts", "expected_valid_counts"):
        _keys(data[key], "train val test", f"dataset.{key}")
        for value in data[key].values():
            _number(value, f"dataset.{key}", 0, integer=True, inclusive=False)
    _number(data["label_fraction"], "dataset.label_fraction", 0, 1, inclusive=False)
    augment = config["augment"]
    _keys(
        augment,
        "random_resized_crop_scale_min right_angle_rotation_only random_rotation_probability rotation_degrees horizontal_flip_probability vertical_flip_probability gaussian_blur_probability gaussian_blur_kernel_size gaussian_blur_sigma gaussian_noise_probability gaussian_noise_std",
        "augment",
    )
    for key, value in augment.items():
        if key.endswith("_probability"):
            _number(value, f"augment.{key}", 0, 1)
    _number(augment["rotation_degrees"], "augment.rotation_degrees", 0, 180)
    _number(
        augment["random_resized_crop_scale_min"],
        "augment.random_resized_crop_scale_min", 1.0, 1.0,
    )
    if augment["right_angle_rotation_only"] is not False:
        raise ValueError("固定连续角度旋转要求right_angle_rotation_only=false")
    kernel = _number(
        augment["gaussian_blur_kernel_size"], "blur kernel", 3, integer=True
    )
    if not int(kernel) % 2:
        raise ValueError("blur kernel必须是奇数")
    sigma = augment["gaussian_blur_sigma"]
    if len(sigma) != 2:
        raise ValueError("blur sigma必须有两个值")
    low = _number(sigma[0], "sigma min", 0, inclusive=False)
    _number(sigma[1], "sigma max", low)
    _number(augment["gaussian_noise_std"], "noise std", 0)
    _keys(config["backbone"], "name weights_dir train_mode", "backbone")
    if (
        config["backbone"]["name"] != "dinov3_convnext_small"
        or config["backbone"]["train_mode"] != "full"
    ):
        raise ValueError("当前模型固定为全量微调的DINOv3 ConvNeXt-Small")
    if (
        not isinstance(config["backbone"]["weights_dir"], str)
        or not config["backbone"]["weights_dir"]
    ):
        raise ValueError("weights_dir必须为本地路径")
    adapter = config["adapter"]
    _keys(adapter, "type hidden_dim embedding_dim attention_heads", "adapter")
    if adapter["type"] != "mma":
        raise ValueError("adapter.type必须为mma")
    for key in ("hidden_dim", "embedding_dim", "attention_heads"):
        _number(adapter[key], f"adapter.{key}", 0, integer=True, inclusive=False)
    if adapter["embedding_dim"] % adapter["attention_heads"]:
        raise ValueError("embedding_dim必须能被attention_heads整除")
    train = config["train"]
    expected = "mode epochs labeled_batch_size evaluation_batch_size num_workers pin_memory learning_rate min_learning_rate backbone_learning_rate backbone_min_learning_rate weight_decay warmup_epochs gradient_clip_norm amp_dtype"
    _keys_with_optional(
        train,
        expected + (" batch_size initial_checkpoint" if is_scp else ""),
        "backbone_layer_decay",
        "train",
    )
    for key in ("epochs", "labeled_batch_size", "evaluation_batch_size") + (
        ("batch_size",) if is_scp else ()
    ):
        _number(train[key], f"train.{key}", 0, integer=True, inclusive=False)
    for key in ("num_workers", "warmup_epochs"):
        _number(train[key], f"train.{key}", 0, integer=True)
    if train["warmup_epochs"] >= train["epochs"]:
        raise ValueError("warmup_epochs必须小于epochs")
    if type(train["pin_memory"]) is not bool or train["amp_dtype"] not in {
        "float32",
        "bfloat16",
    }:
        raise ValueError("pin_memory或amp_dtype无效")
    for key in (
        "learning_rate",
        "min_learning_rate",
        "backbone_learning_rate",
        "backbone_min_learning_rate",
        "gradient_clip_norm",
    ):
        _number(train[key], f"train.{key}", 0, inclusive=False)
    for prefix in ("", "backbone_"):
        if float(train[prefix + "min_learning_rate"]) > float(
            train[prefix + "learning_rate"]
        ):
            raise ValueError("最小学习率不能大于初始学习率")
    if "backbone_layer_decay" in train:
        _number(
            train["backbone_layer_decay"],
            "train.backbone_layer_decay",
            0,
            1,
            inclusive=False,
        )
    _number(train["weight_decay"], "weight_decay", 0)
    if is_scp:
        if (
            not isinstance(train["initial_checkpoint"], str)
            or not train["initial_checkpoint"].strip()
        ):
            raise ValueError("SCP必须指定监督Base initial_checkpoint")
        scp = config["scp"]
        _keys_with_optional(
            scp,
            "profile structural_knn max_selected_per_class max_pseudo_labels "
            "visual_confidence_threshold structural_confidence_threshold "
            "joint_margin_threshold weak_view_max_total_variation reliability_power "
            "class_balance_exponent class_balance_max_multiplier loss_weight "
            "generalized_cross_entropy_q visual_temperature",
            "refresh_epochs refresh_tta loss_ramp_epochs selection_ramp_epochs "
            "labeled_resampling pseudo_train_augmentation prototype_mode "
            "prototype_logit_scale student_classifier_initialization pseudo_symmetry "
            "pseudo_target_mode pseudo_target_hard_weight resampled_labeled_loss_weight",
            "scp",
        )
        if scp["profile"] != "r30":
            raise ValueError("scp.profile必须为r30")
        refresh_epochs = scp.get("refresh_epochs", [])
        if not isinstance(refresh_epochs, list) or any(
            type(value) is not int for value in refresh_epochs
        ):
            raise ValueError("scp.refresh_epochs必须是整数列表")
        if any(
            current <= previous
            for previous, current in zip(refresh_epochs, refresh_epochs[1:])
        ):
            raise ValueError("scp.refresh_epochs必须严格递增且无重复")
        if any(value <= 0 or value >= train["epochs"] for value in refresh_epochs):
            raise ValueError("scp.refresh_epochs节点必须大于0且小于train.epochs")
        fixed_options = {
            "refresh_tta": "none",
            "labeled_resampling": "missing_pseudo_classes",
            "pseudo_train_augmentation": "weak",
        }
        for key, expected_value in fixed_options.items():
            if scp.get(key, expected_value) != expected_value:
                raise ValueError(f"scp.{key}必须为{expected_value}")
        for key in ("loss_ramp_epochs", "selection_ramp_epochs"):
            _number(
                scp.get(key, 1), f"scp.{key}", 0, train["epochs"],
                integer=True, inclusive=False,
            )
        if scp.get("pseudo_symmetry", "none") not in {"none", "hflip"}:
            raise ValueError("scp.pseudo_symmetry仅支持none/hflip")
        if "resampled_labeled_loss_weight" in scp:
            _number(
                scp["resampled_labeled_loss_weight"],
                "scp.resampled_labeled_loss_weight", 0, 1,
            )
        pseudo_target_mode = scp.get("pseudo_target_mode", "hard")
        if pseudo_target_mode not in {"hard", "joint_soft"}:
            raise ValueError("scp.pseudo_target_mode仅支持hard/joint_soft")
        if pseudo_target_mode == "joint_soft":
            if "pseudo_target_hard_weight" not in scp:
                raise ValueError("joint_soft必须配置scp.pseudo_target_hard_weight")
            _number(scp["pseudo_target_hard_weight"], "scp.pseudo_target_hard_weight", 0, 1)
        elif "pseudo_target_hard_weight" in scp:
            raise ValueError("scp.pseudo_target_hard_weight只属于joint_soft目标")
        prototype_mode = scp.get("prototype_mode", "none")
        if prototype_mode not in {"none", "labeled_c4"}:
            raise ValueError("scp.prototype_mode仅支持none/labeled_c4")
        if prototype_mode == "labeled_c4":
            if "prototype_logit_scale" not in scp:
                raise ValueError("labeled_c4必须配置prototype_logit_scale")
            _number(
                scp["prototype_logit_scale"], "scp.prototype_logit_scale",
                0, inclusive=False,
            )
            if scp.get("student_classifier_initialization") != "source_head":
                raise ValueError("学生分类头必须保留source_head初始化")
        elif "prototype_logit_scale" in scp or "student_classifier_initialization" in scp:
            raise ValueError("原型参数仅属于labeled_c4模式")
        for key in ("structural_knn", "max_selected_per_class", "max_pseudo_labels"):
            _number(scp[key], f"scp.{key}", 0, integer=True, inclusive=False)
        _number(scp["visual_temperature"], "scp.visual_temperature", 0, inclusive=False)
        for key in (
            "visual_confidence_threshold",
            "structural_confidence_threshold",
            "joint_margin_threshold",
            "weak_view_max_total_variation",
            "class_balance_exponent",
        ):
            _number(scp[key], f"scp.{key}", 0, 1)
        _number(scp["class_balance_max_multiplier"], "class balance max multiplier", 1)
        _number(scp["loss_weight"], "scp.loss_weight", 0, inclusive=False)
        _number(
            scp["reliability_power"],
            "scp.reliability_power",
            0,
            inclusive=False,
        )
        _number(scp["generalized_cross_entropy_q"], "scp.generalized_cross_entropy_q", 0, 0)
    _keys(config["output"], "root", "output")
    evaluation = config["evaluation"]
    _keys(evaluation, "tta", "evaluation")
    tta = evaluation["tta"]
    if type(tta) is bool:
        return
    if not isinstance(tta, Mapping) or not tta:
        raise ValueError("evaluation.tta必须是bool或历史TTA规则mapping")
    if "config" in tta:
        _keys(tta, "config", "evaluation.tta")
        if not isinstance(tta["config"], str) or not tta["config"].strip():
            raise ValueError("TTA规则路径必须是非空字符串")
    else:
        from .symmetry_tta import load_symmetry_tta_config

        load_symmetry_tta_config(tta)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置顶层必须是mapping")
    try:
        validate_config(payload)
    except ValueError:
        payload = _expand_config(payload)
        validate_config(payload)
    return payload


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.dump(
            dict(config),
            Dumper=_ConfigDumper,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=1000000,
        ),
        encoding="utf-8",
    )
    return destination
