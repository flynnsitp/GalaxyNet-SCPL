# Evaluate checkpoints and save classification metrics and predictions.

from __future__ import annotations
import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch import Tensor
from .artifacts import (
    load_torch_mapping,
    save_torch_atomic,
    setup_logger,
    sha256_file,
    state_digest,
    write_json,
)
from .config import DEFAULT_CONFIG, load_config, single_experiment_seed
from .data import SampleRecord, validate_dataset
from .engine import (
    _build_model,
    encode_records,
    configure_reproducibility,
    seed_everything,
    resolve_device,
)
from .checkpoints import (
    _check_evaluation_contract,
    _load_model_state,
    validate_evaluation_role,
)
from .metrics import classification_metrics, log_classification_metrics
from .symmetry_tta import (
    default_symmetry_tta_config,
    load_symmetry_tta_config,
    expand_symmetry_groups,
    fuse_classwise_symmetry_logits,
    mean_view_log_probabilities,
    _view_transform,
    _view_key,
)


def _write_predictions(
    path: Path,
    records: Sequence[SampleRecord],
    logits: Tensor,
    class_names: Sequence[str],
) -> None:
    probabilities = torch.softmax(logits, dim=1).cpu()
    predictions = probabilities.argmax(dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "path", "target", "prediction"] + [
        f"prob_{name}" for name in class_names
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records[: len(probabilities)]):
            row: dict[str, Any] = {
                "sample_id": record.sample_id,
                "path": record.relative_path,
                "target": class_names[record.label],
                "prediction": class_names[int(predictions[index])],
            }
            row.update(
                {
                    f"prob_{name}": float(probabilities[index, class_index])
                    for class_index, name in enumerate(class_names)
                }
            )
            writer.writerow(row)


def evaluate_checkpoint(
    config,
    checkpoint_path,
    split,
    device,
    output_dir=None,
    *,
    tta_enabled=None,
    allow_last_test=False,
    save_artifacts=True,
):
    if split not in ("val", "test"):
        raise ValueError("评估split必须是val/test")
    if tta_enabled is None:
        tta_enabled = config["evaluation"]["tta"] is True
    configure_reproducibility(config)
    seed_everything(single_experiment_seed(config))
    audit = validate_dataset(config)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_torch_mapping(checkpoint_path)
    if allow_last_test and (split != "test" or checkpoint_path.name != "last.pt"):
        raise ValueError("allow_last_test只允许test split的last.pt")
    validate_evaluation_role(
        checkpoint,
        split,
        allow_last_test=allow_last_test,
    )
    _check_evaluation_contract(checkpoint, config, audit)
    checkpoint_hash = sha256_file(checkpoint_path)
    model = _build_model(config, len(audit.classes), device, train_mode="frozen")
    _load_model_state(model, checkpoint)
    records = audit.records[split]
    _, original, labels = encode_records(model, records, config, device)
    destination = None
    if save_artifacts:
        root = (
            Path(output_dir)
            if output_dir is not None
            else checkpoint_path.parent.parent / "evaluations"
        )
        destination = (
            root / f"{split}_{checkpoint_hash[:12]}_{datetime.now():%Y%m%d-%H%M%S-%f}"
        )
        destination.mkdir(parents=True, exist_ok=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "checkpoint_role": str(checkpoint["role"]),
        "result_kind": (
            "diagnostic_non_selected_checkpoint"
            if checkpoint["role"] == "last"
            else "validation_selected_checkpoint"
        ),
        "split": split,
        "manifest_sha256": audit.manifest_sha256,
        "train_valid_path_sha256": audit.valid_path_sha256["train"],
        "split_valid_path_sha256": audit.valid_path_sha256[split],
        "output_dir": (None if destination is None else str(destination.resolve())),
        "tta_enabled": bool(tta_enabled),
        "loads_test_images": split == "test",
        "single_model_only": True,
    }

    def save_result(name, logits, metadata):
        metrics = classification_metrics(logits, labels, audit.classes)
        if destination is not None:
            folder = destination / name
            write_json(folder / "metrics.json", metrics)
            write_json(folder / "confusion_matrix.json", metrics["confusion_matrix"])
            _write_predictions(
                folder / "predictions.csv", records, logits, audit.classes
            )
            save_torch_atomic(
                folder / "logits.pt", {"logits": logits, "labels": labels}
            )
            write_json(
                folder / "summary.json", {**summary, **metadata, "metrics": metrics}
            )
        summary[name] = metrics
        print(
            f"{split}/{name}: accuracy={metrics['accuracy']:.9f}, nll={metrics['nll']:.9f}",
            flush=True,
        )

    save_result("single_view", original, {"tta_enabled": False, "view_count": 1})
    if summary["tta_enabled"]:
        tta_config = config["evaluation"]["tta"]
        if isinstance(tta_config, bool):
            rules = default_symmetry_tta_config(config)
            rule_source, rule_hash = "builtin:default", state_digest(rules)
        elif "config" in tta_config:
            rule_path = Path(tta_config["config"]).expanduser()
            bundled_classwise = {
                "configs/symmetry_tta.yaml": True,
                "configs/symmetry_tta_d16.yaml": False,
            }.get(str(rule_path))
            if not rule_path.is_absolute():
                rule_path = Path(__file__).resolve().parents[1] / rule_path
            if not rule_path.exists() and bundled_classwise is not None:
                rules = default_symmetry_tta_config(config, classwise=bundled_classwise)
                rule_source = f"builtin:{tta_config['config']}"
                rule_hash = state_digest(rules)
            else:
                rules = load_symmetry_tta_config(rule_path)
                rule_source, rule_hash = str(rule_path.resolve()), sha256_file(rule_path)
        else:
            rules = load_symmetry_tta_config(tta_config)
            rule_source, rule_hash = "evaluation.tta", state_digest(rules)
        is_r30_tta = "class_rules" in rules
        views = {} if is_r30_tta else {"0_0": original.float().cpu()}
        for angle in rules["angles_degrees"]:
            for flip in rules["horizontal_flip_values"]:
                if not is_r30_tta and angle == 0 and not flip:
                    continue
                _, logits, current_labels = encode_records(
                    model,
                    records,
                    config,
                    device,
                    transform=_view_transform(config, angle, flip),
                )
                if not torch.equal(current_labels, labels):
                    raise RuntimeError("TTA视图样本顺序不一致")
                views[_view_key(angle, flip)] = logits.float().cpu()
        if is_r30_tta:
            groups = expand_symmetry_groups(
                rules["angles_degrees"],
                rules["horizontal_flip_values"],
                rules["groups"],
            )
            fused = fuse_classwise_symmetry_logits(
                views, groups, rules["class_rules"], audit.classes
            )
            summary["tta_protocol"] = "r30_classwise_logits_32"
        else:
            original_weight = float(rules["original_probability_weight"])
            fused = mean_view_log_probabilities(views, original_weight)
            summary["tta_protocol"] = (
                f"uniform_d{len(rules['angles_degrees'])}_mean_probabilities_"
                f"original_{round(100 * original_weight):02d}"
            )
        tta_metadata = {
            "tta_enabled": True,
            "view_count": len(views),
            "tta_config": rule_source,
            "tta_config_sha256": rule_hash,
            "tta_rules": rules,
            "class_specific_rules": is_r30_tta,
            "additive_class_bias": False,
            "search_performed": False,
        }
        summary.update(
            {
                "view_count": tta_metadata["view_count"],
                "tta_config": tta_metadata["tta_config"],
                "tta_config_sha256": tta_metadata["tta_config_sha256"],
            }
        )
        if destination is not None:
            save_torch_atomic(
                destination / "symmetry_tta" / "view_logits.pt",
                {
                    "views": views,
                    "labels": labels,
                    "sample_ids": [record.sample_id for record in records],
                    "checkpoint_sha256": checkpoint_hash,
                },
            )
        save_result(
            "symmetry_tta",
            fused,
            tta_metadata,
        )
    if destination is not None:
        write_json(destination / "summary.json", summary)
    return summary


def _log_evaluation_result(
    logger: logging.Logger,
    *,
    config_path: Path,
    config_sha256: str,
    result: Mapping[str, Any],
) -> None:
    metrics = result["symmetry_tta"]
    logger.info(
        "evaluation | config=%s | checkpoint=%s | checkpoint_role=%s | "
        "checkpoint_epoch=%d | result_kind=%s | split=%s | tta_enabled=true | "
        "tta_protocol=%s | view_count=%d | checkpoint_sha256=%s | "
        "tta_config=%s | tta_config_sha256=%s",
        config_path,
        result["checkpoint"],
        result["checkpoint_role"],
        result["checkpoint_epoch"],
        result["result_kind"],
        result["split"],
        result["tta_protocol"],
        result["view_count"],
        result["checkpoint_sha256"],
        result["tta_config"],
        result["tta_config_sha256"],
    )
    log_classification_metrics(
        logger,
        split=str(result["split"]),
        loss=float(metrics["nll"]),
        metrics=metrics,
        per_class=True,
    )
    logger.info(
        "evaluation | status=ok | config_sha256=%s | checkpoint_sha256=%s | "
        "tta_config_sha256=%s",
        config_sha256,
        result["checkpoint_sha256"],
        result["tta_config_sha256"],
    )


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with optional TTA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--tta", action=argparse.BooleanOptionalAction, default=None,
        help="覆盖配置中的TTA开关；--no-tta显式关闭",
    )
    parser.add_argument(
        "--allow-last-test",
        action="store_true",
        help="仅用于完整last.pt的诊断性test，不改变正式best gate",
    )
    parser.add_argument(
        "--log-only",
        action="store_true",
        help="只写--log-file，不保存JSON/CSV/logits评估产物",
    )
    parser.add_argument("--log-file", help="log-only模式的唯一结果文件")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", help="本次评估独立子目录的父目录")
    args = parser.parse_args(argv)
    if bool(args.log_only) != bool(args.log_file):
        parser.error("--log-only与--log-file必须同时使用")
    if args.log_only and args.output_dir is not None:
        parser.error("--log-only不能同时指定--output-dir")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config = load_config(config_path)
    tta_enabled = config["evaluation"]["tta"] is True if args.tta is None else args.tta
    if args.log_only and not tta_enabled:
        parser.error("--log-only需要启用TTA，可在配置中设置或使用--tta")
    config_hash = sha256_file(config_path)
    logger = setup_logger(args.log_file, "mrd_scp.evaluate") if args.log_file else None
    if logger is not None:
        logger.info(
            "evaluation | status=started | config=%s | config_sha256=%s | "
            "checkpoint=%s",
            config_path,
            config_hash,
            checkpoint_path,
        )
    try:
        result = evaluate_checkpoint(
            config,
            checkpoint_path,
            args.split,
            resolve_device(args.device),
            None if args.log_only else args.output_dir,
            tta_enabled=tta_enabled,
            allow_last_test=args.allow_last_test,
            save_artifacts=not args.log_only,
        )
    except Exception:
        if logger is not None:
            logger.exception("evaluation | status=failed")
        raise
    if logger is not None:
        _log_evaluation_result(
            logger,
            config_path=config_path,
            config_sha256=config_hash,
            result=result,
        )
        print(f"log_file={Path(args.log_file).resolve()}", flush=True)
    else:
        print(f"evaluation_dir={result['output_dir']}", flush=True)
