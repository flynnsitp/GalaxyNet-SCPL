# Validate checkpoint identities and preserve exact training recovery.

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import torch
from .artifacts import sha256_file
from .backbone import DINOv3FeatureExtractor
from .config import single_experiment_seed
from .data import DatasetAudit, FewShotSplit
from .model import MorphologyClassifier
from .scp import (
    SMALL_LABEL_FRACTION_PERCENT,
    SCPState,
)
from .structure import RobustScaler


def _training_contract(
    config,
    audit,
    fewshot,
    backbone,
    initial_source=None,
    *,
    max_train_batches=None,
    max_val_batches=None,
):
    is_scp = config["train"]["mode"] == "scp_from_checkpoint"
    if is_scp != (initial_source is not None):
        raise ValueError("SCP需要监督Base来源；独立Base不能暖启动")
    train_count = len(audit.records["train"])
    small_label_pool = bool(
        is_scp
        and len(fewshot.labeled_indices) * 100
        < train_count * SMALL_LABEL_FRACTION_PERCENT
    )
    refresh_epochs = (
        tuple(int(value) for value in config["scp"].get("refresh_epochs", ()))
        if is_scp
        else ()
    )
    max_pseudo_labels = int(config["scp"]["max_pseudo_labels"]) if is_scp else 0
    labeled_resampling = (
        str(config["scp"].get("labeled_resampling", "missing_pseudo_classes"))
        if is_scp
        else ""
    )
    return {
        "schema_version": 11,
        "training_seed": single_experiment_seed(config),
        "training_kind": "online_joint",
        "training_mode": config["train"]["mode"],
        "labeled_epoch_policy": (
            (
                "pseudo_coverage_with_labeled_resampling"
                if small_label_pool
                else "labeled_primary_auxiliary_single_pass"
            )
            if is_scp
            else "labeled_train_single_pass"
        ),
        **(
            {
                "small_label_pseudo_policy": {
                    "enabled": True,
                    "labeled_fraction_strictly_below": (
                        SMALL_LABEL_FRACTION_PERCENT / 100
                    ),
                    "epoch_pseudo_consumption": "exactly_once",
                    "extra_step_labeled_sampling": (
                        f"{labeled_resampling}_then_uniform_with_replacement"
                    ),
                }
            }
            if small_label_pool
            else {}
        ),
        **(
            {
                "pseudo_selection_limit": {
                    "max_pseudo_labels": max_pseudo_labels,
                    "ranking": "global_reliability_then_sample_id",
                    "at_or_below_limit": "keep_all",
                }
            }
            if is_scp
            else {}
        ),
        "student_update_scope": "configured_model",
        "scp_state_policy": (
            (
                "fixed_epoch_refreshes"
                if refresh_epochs
                else (
                    "fixed_candidate_pool_epoch_schedule"
                    if int(config["scp"].get("selection_ramp_epochs", 1)) > 1
                    else "fixed_at_start_epoch"
                )
            )
            if is_scp
            else "none"
        ),
        "initial_source": None if initial_source is None else dict(initial_source),
        "development_limits": {
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
        },
        "classes": list(audit.classes),
        "manifest_sha256": audit.manifest_sha256,
        "structural_sha256": audit.structural_sha256,
        "structural_feature_count": len(audit.structural_keys),
        "structural_keys": list(audit.structural_keys),
        "train_valid_path_sha256": audit.valid_path_sha256["train"],
        "backbone_name": backbone.spec.name,
        "backbone_sha256": backbone.spec.sha256,
        "backbone_train_mode": backbone.train_mode,
        "feature_dim": backbone.spec.feature_dim,
        "input_size": int(config["dataset"]["input_size"]),
        "preprocess": {k: config["dataset"][k] for k in ("input_size", "mean", "std")},
        "fewshot_fraction": fewshot.fraction,
        "fewshot_seed": fewshot.seed,
        "fewshot_fingerprint": fewshot.fingerprint_sha256,
        "reproducibility": dict(config["reproducibility"]),
        "augment": dict(config["augment"]),
        "adapter": dict(config["adapter"]),
        "classifier": _classifier_contract(),
        "scp": dict(config["scp"]) if is_scp else None,
        "train": {
            k: v for k, v in config["train"].items() if k != "initial_checkpoint"
        },
    }


def _canonical_training_contract(contract):
    version = contract.get("schema_version")
    if version not in (9, 10, 11):
        raise ValueError("checkpoint仅接受schema_version=9/10/11")
    result = dict(contract)
    if version in (9, 10):
        legacy_classifier = _classifier_contract()
        if result.get("classifier", legacy_classifier) != legacy_classifier:
            raise ValueError("历史checkpoint不能声明训练过分类头Dropout")
        result["classifier"] = legacy_classifier
    elif "classifier" not in result:
        raise ValueError("schema-11 checkpoint缺少classifier身份")
    elif type(result.get("training_seed")) is not int or result["training_seed"] < 0:
        raise ValueError("schema-11 checkpoint缺少有效的training_seed")
    return result


def _classifier_contract():
    return {
        "dropout_probability": 0.0,
        "dropout_location": "normalized_embedding_before_linear",
    }


def _has_development_limits(contract: Mapping[str, Any]) -> bool:
    limits = contract.get("development_limits")
    if limits is None:
        return False
    if not isinstance(limits, Mapping):
        raise ValueError("checkpoint development_limits 必须是 mapping")
    return any(value is not None for value in limits.values())


def _validate_resume_history(run_dir: Path, checkpoint: Mapping[str, Any]) -> None:
    checkpoint_epoch = int(checkpoint["epoch"])
    history_path = run_dir / "history.jsonl"
    if not history_path.is_file():
        raise ValueError("resume 运行缺少 history.jsonl，无法证明 epoch 连续性")
    try:
        rows = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise ValueError("resume history.jsonl 不是完整 JSONL") from error
    expected_epochs = list(range(1, checkpoint_epoch + 2))
    actual_epochs = [
        row.get("epoch") if isinstance(row, Mapping) else None for row in rows
    ]
    if actual_epochs != expected_epochs:
        raise ValueError(
            "resume history 与 last.pt epoch 不一致；请先恢复到同一原子边界"
        )
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("epoch") != checkpoint_epoch + 1:
        raise ValueError("resume checkpoint metrics 与 epoch 不一致")


def _structural_identity_matches(expected_sha256: Any, audit: DatasetAudit) -> bool:
    if expected_sha256 == audit.structural_sha256:
        return True
    if not isinstance(expected_sha256, str):
        return False
    payload = audit.structural_path.read_bytes()
    marker = b"/__mrd_scp_val_test_swap__/"
    if marker in payload:
        return False
    reversed_swap = (
        payload.replace(b"/val/", marker)
        .replace(b"/test/", b"/val/")
        .replace(marker, b"/test/")
    )
    return hashlib.sha256(reversed_swap).hexdigest() == expected_sha256


def _validate_initial_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    config: Mapping[str, Any],
    audit: DatasetAudit,
    fewshot: FewShotSplit,
    backbone: DINOv3FeatureExtractor,
) -> dict[str, Any]:
    validate_evaluation_role(checkpoint, "val")
    if checkpoint.get("role") != "best_validation":
        raise ValueError("--init-checkpoint 只接受 validation 选出的 best.pt")
    if not bool(checkpoint.get("training_complete")):
        raise ValueError("--init-checkpoint 必须来自已完成全部 epoch 的运行")
    if bool(checkpoint.get("development_only")):
        raise ValueError("暖启动不能使用 development-only checkpoint")
    _check_evaluation_contract(checkpoint, config, audit)
    source_contract = _canonical_training_contract(checkpoint["contract"])
    if _has_development_limits(source_contract):
        raise ValueError("暖启动不能使用带 development 截断契约的 checkpoint")
    expected_identity = {
        "train_valid_path_sha256": audit.valid_path_sha256["train"],
        "fewshot_fraction": fewshot.fraction,
        "fewshot_seed": fewshot.seed,
        "fewshot_fingerprint": fewshot.fingerprint_sha256,
        "backbone_sha256": backbone.spec.sha256,
    }
    mismatches = {
        key: (source_contract.get(key), value)
        for key, value in expected_identity.items()
        if source_contract.get(key) != value
    }
    if not _structural_identity_matches(
        source_contract.get("structural_sha256"), audit
    ):
        mismatches["structural_sha256"] = (
            source_contract.get("structural_sha256"),
            audit.structural_sha256,
        )
    if mismatches:
        raise ValueError(f"initial checkpoint 与当前训练身份不匹配: {mismatches}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    source_mode = source_contract.get("training_mode")
    source_generation_depth = 0
    root_base_sha256 = checkpoint_sha256
    if source_contract.get("schema_version") == 9:
        if (
            float(source_contract.get("mrd", {}).get("loss_weight", -1)) != 0
            or float(source_contract.get("prototype", {}).get("loss_weight", -1)) != 0
        ):
            raise ValueError("initial_checkpoint必须是不含MRD/SCP更新的监督Base")
    elif source_mode == "staged":
        if source_contract.get("scp") is not None:
            raise ValueError("监督Base checkpoint不能包含SCP配置")
        if source_contract.get("initial_source") is not None:
            raise ValueError("监督Base checkpoint不能来自暖启动")
    else:
        raise ValueError("initial_checkpoint必须是匹配的监督Base checkpoint")
    if source_contract.get("student_update_scope") not in {None, "configured_model"}:
        raise ValueError("--init-checkpoint 必须来自原始 base 训练范围")
    if source_contract.get("scp_state_policy") not in {None, "none"}:
        raise ValueError("监督Base不能包含SCP state")
    source_metrics = checkpoint.get("best_validation") or {}
    return {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha256,
        "role": str(checkpoint["role"]),
        "epoch": int(checkpoint["epoch"]) + 1,
        "backbone_state_present": checkpoint.get("backbone_state") is not None,
        "source_schema_version": source_contract.get("schema_version"),
        "source_training_mode": source_mode,
        "source_generation_depth": source_generation_depth,
        "root_base_sha256": root_base_sha256,
        "source_best_validation": {
            key: source_metrics.get(key)
            for key in ("epoch", "accuracy", "f1_macro", "f1_weighted", "nll")
        },
    }


def _checkpoint_payload(
    *,
    role: str,
    epoch: int,
    model: MorphologyClassifier,
    scp_state: SCPState | None,
    scp_refresh_count: int,
    scaler: RobustScaler,
    optimizer: torch.optim.Optimizer,
    contract: Mapping[str, Any],
    metrics: Mapping[str, Any],
    best_validation: Mapping[str, Any] | None,
    development_only: bool,
    training_complete: bool,
) -> dict[str, Any]:
    initial_source = contract.get("initial_source")
    preserve_initial_backbone = bool(
        isinstance(initial_source, Mapping)
        and initial_source.get("backbone_state_present")
    )
    backbone_state = (
        model.backbone.backbone.state_dict()
        if model.backbone.train_mode == "full" or preserve_initial_backbone
        else None
    )
    return {
        "format_version": 3,
        "training_kind": "online_joint",
        "role": role,
        "epoch": int(epoch),
        "backbone_state": backbone_state,
        "adapter_state": model.adapter.state_dict(),
        "classifier_state": model.classifier.state_dict(),
        "scp_state": None if scp_state is None else scp_state.to_checkpoint(),
        "scp_refresh_count": int(scp_refresh_count),
        "structural_scaler": {
            "structural_keys": list(contract["structural_keys"]),
            **scaler.to_checkpoint(),
        },
        "optimizer_state": optimizer.state_dict(),
        "contract": dict(contract),
        "metrics": dict(metrics),
        "best_validation": (None if best_validation is None else dict(best_validation)),
        "classification_available": True,
        "development_only": bool(development_only),
        "training_complete": bool(training_complete),
    }


def _load_model_state(
    model: MorphologyClassifier, checkpoint: Mapping[str, Any]
) -> None:
    backbone_state = checkpoint.get("backbone_state")
    contract = checkpoint.get("contract", {})
    trained_mode = contract.get("backbone_train_mode")
    initial_source = contract.get("initial_source")
    backbone_state_required = trained_mode == "full" or bool(
        isinstance(initial_source, Mapping)
        and initial_source.get("backbone_state_present")
    )
    if backbone_state_required and backbone_state is None:
        raise ValueError("full fine-tuning checkpoint 缺少 backbone_state")
    if (
        trained_mode == "frozen"
        and backbone_state is not None
        and not backbone_state_required
    ):
        raise ValueError("frozen checkpoint 不应包含 backbone_state")
    if backbone_state is not None:
        model.backbone.backbone.load_state_dict(backbone_state, strict=True)
    model.adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
    model.classifier.load_state_dict(checkpoint["classifier_state"], strict=True)


def validate_evaluation_role(
    checkpoint: Mapping[str, Any],
    split: str,
    *,
    allow_last_test: bool = False,
) -> None:
    if (
        checkpoint.get("format_version"),
        checkpoint.get("contract", {}).get("schema_version"),
    ) not in {(2, 9), (3, 10), (3, 11)}:
        raise ValueError("checkpoint format_version 不受支持")
    if checkpoint.get("training_kind") != "online_joint":
        raise ValueError("checkpoint 不是当前在线联合训练格式")
    if not bool(checkpoint.get("classification_available")):
        raise ValueError("checkpoint没有可用分类器")
    if "classifier_state" not in checkpoint:
        raise ValueError("checkpoint 缺少 classifier_state")
    if split == "test":
        role = checkpoint.get("role")
        if role != "best_validation" and not (allow_last_test and role == "last"):
            raise ValueError("test 只接受 validation 选出的 best.pt")
        if bool(checkpoint.get("development_only")):
            raise ValueError("development-only checkpoint 禁止 test")
        contract = checkpoint.get("contract")
        if isinstance(contract, Mapping) and _has_development_limits(contract):
            raise ValueError("带 development 数据截断契约的 checkpoint 禁止 test")
        if not bool(checkpoint.get("training_complete")):
            raise ValueError("尚未完成全部 epoch 的 checkpoint 禁止 test")


def _check_evaluation_contract(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    audit: DatasetAudit,
) -> None:
    contract = _canonical_training_contract(checkpoint["contract"])
    expected = {
        "train_valid_path_sha256": audit.valid_path_sha256["train"],
        "classes": list(audit.classes),
        "backbone_name": config["backbone"]["name"],
        "input_size": int(config["dataset"]["input_size"]),
        "preprocess": {
            "input_size": int(config["dataset"]["input_size"]),
            "mean": [float(value) for value in config["dataset"]["mean"]],
            "std": [float(value) for value in config["dataset"]["std"]],
        },
        "adapter": dict(config["adapter"]),
        "classifier": _classifier_contract(),
        "amp_dtype": config["train"]["amp_dtype"],
    }
    actual_adapter = contract.get("adapter")
    actual = {
        **contract,
        "adapter": (
            dict(actual_adapter)
            if isinstance(actual_adapter, Mapping)
            else actual_adapter
        ),
        "amp_dtype": contract.get("train", {}).get("amp_dtype"),
    }
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"evaluation config/checkpoint 契约不匹配: {mismatches}")
