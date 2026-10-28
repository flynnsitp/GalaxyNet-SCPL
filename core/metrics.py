# Compute classification metrics and select validation checkpoints.

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def confusion_matrix(targets: Tensor, predictions: Tensor, class_count: int) -> Tensor:
    encoded = targets.long() * class_count + predictions.long()
    return torch.bincount(encoded, minlength=class_count * class_count).reshape(
        class_count, class_count
    )


def classification_metrics(
    logits: Tensor, targets: Tensor, class_names: Sequence[str]
) -> dict[str, Any]:
    values = logits.detach().cpu().float()
    labels = targets.detach().cpu().long()
    class_count = len(class_names)
    if values.ndim != 2 or values.shape != (labels.numel(), class_count):
        raise ValueError("logits/targets 与类别数量不匹配")
    predictions = values.argmax(dim=1)
    matrix = confusion_matrix(labels, predictions, class_count).float()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    true_positive = matrix.diag()
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(support > 0, true_positive / support, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2.0 * precision * recall / denominator, 0.0)
    weights = support / support.sum().clamp_min(1.0)
    per_class = {
        name: {
            "tp": int(true_positive[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "accuracy": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    overall_accuracy = float((predictions == labels).float().mean())
    return {
        "num_samples": int(labels.numel()),
        "nll": float(F.cross_entropy(values, labels)),

        "accuracy": overall_accuracy,
        "oa": overall_accuracy,
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.long().tolist(),
    }


def log_classification_metrics(
    logger: logging.Logger,
    *,
    split: str,
    loss: float,
    metrics: Mapping[str, Any],
    per_class: bool,
) -> None:
    logger.info(
        "%s | loss=%.4f | acc=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        split,
        loss,
        metrics["accuracy"],
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"],
    )
    if not per_class:
        return
    logger.info("%s per-class:", split)
    logger.info(
        "  %-4s %-24s %-9s %-9s %-9s %-9s %-9s %-9s",
        "id",
        "class",
        "support",
        "TP",
        "acc",
        "prec",
        "rec",
        "f1",
    )
    for index, (name, row) in enumerate(metrics["per_class"].items()):
        logger.info(
            "  %-4d %-24s %-9d %-9d %-9.4f %-9.4f %-9.4f %-9.4f",
            index,
            name,
            row["support"],
            row["tp"],
            row["accuracy"],
            row["precision"],
            row["recall"],
            row["f1"],
        )


def is_better_validation(
    metrics: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    if best is None:
        return True
    current_value = float(metrics["accuracy"])
    best_value = float(best["accuracy"])
    if current_value != best_value:
        return current_value > best_value
    return float(metrics["nll"]) < float(best["nll"])
