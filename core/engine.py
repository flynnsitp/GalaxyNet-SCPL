# Train and resume supervised and pseudo-label galaxy classifiers.

from __future__ import annotations
import argparse
import copy
import math
import os
import platform
import random
from contextlib import nullcontext
from datetime import datetime
from itertools import chain, islice
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import importlib.metadata
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from .artifacts import (
    append_jsonl,
    create_run_dir,
    load_torch_mapping,
    save_torch_atomic,
    sha256_file,
    setup_logger,
    write_json,
    state_digest,
)
from .backbone import build_backbone
from .config import (
    DEFAULT_CONFIG,
    experiment_seeds,
    load_config,
    save_config,
    single_experiment_seed,
)
from .data import (
    FewShotSplit,
    SampleRecord,
    build_fewshot_split,
    validate_dataset,
)
from .image_data import (
    OnlineImageDataset,
    build_image_dataset,
    build_image_transform,
    build_scp_weak_transform,
)
from .metrics import (
    classification_metrics,
    is_better_validation,
    log_classification_metrics,
)
from .model import MorphologyClassifier
from .structure import RobustScaler
from .scp import (
    PROTOTYPE_VIEW_ANGLES,
    SMALL_LABEL_FRACTION_PERCENT,
    SCPState,
    build_scp_state,
    cap_scp_state_by_max_pseudo_labels,
    scheduled_scp_state,
)
from .symmetry_tta import _view_transform
from .checkpoints import (
    _canonical_training_contract,
    _training_contract,
    _validate_resume_history,
    _validate_initial_checkpoint,
    _checkpoint_payload,
    _load_model_state,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(config: Mapping[str, Any]) -> None:
    reproducibility = config["reproducibility"]
    if not bool(reproducibility["strict"]):
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {None, ":4096:8", ":16:8"}:
        raise RuntimeError(
            "strict reproducibility 要求 CUBLAS_WORKSPACE_CONFIG=" ":4096:8 或 :16:8"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 看不到 GPU")
    return device


def cosine_learning_rate(
    peak: float,
    minimum: float,
    epochs: int,
    warmup_epochs: int,
    epoch: int,
) -> float:
    if int(epochs) == 1:
        return float(peak)
    if epoch < warmup_epochs:
        return float(peak) * float(epoch + 1) / float(max(warmup_epochs, 1))
    remaining = int(epochs) - int(warmup_epochs)
    if remaining <= 1:
        return float(minimum)
    progress = float(epoch - warmup_epochs) / float(remaining - 1)
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress)
    )


def _make_loader(
    dataset: Dataset[Any],
    config: Mapping[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=bool(config["train"]["pin_memory"]),
        generator=generator,
        drop_last=False,
    )


def _make_replacement_labeled_loader(
    train_dataset: OnlineImageDataset,
    fewshot: FewShotSplit,
    config: Mapping[str, Any],
    *,
    batch_size: int,
    batch_count: int,
    seed: int,
    priority_classes: Sequence[int] = (),
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(seed))
    labeled_indices = tuple(int(index) for index in fewshot.labeled_indices)
    priority_classes = tuple(int(value) for value in priority_classes)
    if len(priority_classes) != len(set(priority_classes)):
        raise ValueError("priority_classes不能重复")
    labeled_by_class = {
        class_index: tuple(
            index
            for index in labeled_indices
            if int(train_dataset.records[index].label) == class_index
        )
        for class_index in priority_classes
    }
    if any(not indices for indices in labeled_by_class.values()):
        raise ValueError("priority_classes必须在labeled池中有样本")
    if priority_classes:
        scheduled_indices: list[int] = []
        reserved_count = min(len(priority_classes), int(batch_size))
        for batch_index in range(int(batch_count)):
            start = batch_index * reserved_count
            reserved_classes = tuple(
                priority_classes[(start + offset) % len(priority_classes)]
                for offset in range(reserved_count)
            )
            current = [
                labeled_by_class[class_index][
                    int(
                        torch.randint(
                            len(labeled_by_class[class_index]),
                            (1,),
                            generator=generator,
                        ).item()
                    )
                ]
                for class_index in reserved_classes
            ]
            remaining_count = int(batch_size) - len(current)
            if remaining_count:
                positions = torch.randint(
                    len(labeled_indices),
                    (remaining_count,),
                    generator=generator,
                ).tolist()
                current.extend(labeled_indices[position] for position in positions)
            order = torch.randperm(len(current), generator=generator).tolist()
            scheduled_indices.extend(current[position] for position in order)
    else:
        positions = torch.randint(
            len(labeled_indices),
            (int(batch_count) * int(batch_size),),
            generator=generator,
        ).tolist()
        scheduled_indices = [labeled_indices[position] for position in positions]

    return _make_loader(
        Subset(train_dataset, scheduled_indices),
        config,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 1_000_003,
    )


def _evenly_spaced_steps(total_steps: int, selected_steps: int) -> tuple[int, ...]:
    total_steps = int(total_steps)
    selected_steps = int(selected_steps)
    if total_steps < 0 or not 0 <= selected_steps <= total_steps:
        raise ValueError("selected_steps 必须位于 [0, total_steps]")
    if selected_steps == 0:
        return ()
    return tuple(
        ((2 * index + 1) * total_steps) // (2 * selected_steps)
        for index in range(selected_steps)
    )


def _fixed_batch_cross_entropy(
    logits: Tensor, targets: Tensor, reference_batch_size: int
) -> Tensor:
    if len(logits) == 0:
        raise ValueError("监督 CE 至少需要一个样本")
    return F.cross_entropy(logits, targets, reduction="sum") / int(reference_batch_size)


def _autocast(config: Mapping[str, Any], device: torch.device):
    if device.type == "cuda" and config["train"]["amp_dtype"] == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _records_structures(records: Sequence[SampleRecord]) -> Tensor:
    return torch.tensor([record.structure for record in records], dtype=torch.float32)


def _build_model(
    config: Mapping[str, Any],
    class_count: int,
    device: torch.device,
    *,
    train_mode: str | None = None,
) -> MorphologyClassifier:
    mode = str(train_mode or config["backbone"]["train_mode"])
    backbone = build_backbone(
        config["backbone"]["name"],
        config["backbone"]["weights_dir"],
        train_mode=mode,
    )
    adapter = config["adapter"]
    return MorphologyClassifier(
        backbone,
        adapter_type="mma",
        class_count=class_count,
        hidden_dim=int(adapter["hidden_dim"]),
        embedding_dim=int(adapter["embedding_dim"]),
        attention_heads=int(adapter["attention_heads"]),
    ).to(device)


def _build_optimizer(
    model: MorphologyClassifier, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    head_parameters = [
        parameter
        for module in (model.adapter, model.classifier)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not head_parameters:
        raise ValueError("当前训练范围没有可优化的MMA/classifier参数")
    groups: list[dict[str, Any]] = [
        {
            "name": "head",
            "params": head_parameters,
            "lr": float(config["train"]["learning_rate"]),
        }
    ]
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    if backbone_parameters:
        layer_decay = float(config["train"].get("backbone_layer_decay", 1.0))
        stage_group_builder = getattr(
            model.backbone, "trainable_stage_parameters", None
        )
        if layer_decay < 1.0 and callable(stage_group_builder):
            stage_groups = stage_group_builder()
            for index, parameters in enumerate(stage_groups):
                scale = layer_decay ** (len(stage_groups) - 1 - index)
                groups.append(
                    {
                        "name": f"backbone_stage_{index}",
                        "params": list(parameters),
                        "lr": (
                            float(config["train"]["backbone_learning_rate"]) * scale
                        ),
                        "lr_scale": scale,
                    }
                )
        else:
            groups.append(
                {
                    "name": "backbone",
                    "params": backbone_parameters,
                    "lr": float(config["train"]["backbone_learning_rate"]),
                    "lr_scale": 1.0,
                }
            )
    return torch.optim.AdamW(
        groups,
        weight_decay=float(config["train"]["weight_decay"]),
    )


def _set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    epoch: int,
) -> tuple[float, float | None]:
    train = config["train"]
    head_lr = cosine_learning_rate(
        float(train["learning_rate"]),
        float(train["min_learning_rate"]),
        int(train["epochs"]),
        int(train["warmup_epochs"]),
        epoch,
    )
    backbone_lr: float | None = None
    for group in optimizer.param_groups:
        if group["name"] == "head":
            group["lr"] = head_lr
        elif str(group["name"]).startswith("backbone"):
            backbone_lr = cosine_learning_rate(
                float(train["backbone_learning_rate"]),
                float(train["backbone_min_learning_rate"]),
                int(train["epochs"]),
                int(train["warmup_epochs"]),
                epoch,
            )
            group["lr"] = backbone_lr * float(group.get("lr_scale", 1.0))
    return head_lr, backbone_lr


def encode_records(
    model: MorphologyClassifier,
    records: Sequence[SampleRecord],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    max_batches: int | None = None,
    transform: Any | None = None,
    transform_seed: int | None = None,
    include_targets: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    dataset = OnlineImageDataset(
        records,
        (
            build_image_transform(config, training=False)
            if transform is None
            else transform
        ),
        include_structure=False,
        include_label=include_targets,
        transform_seed=transform_seed,
    )
    loader = _make_loader(
        dataset,
        config,
        batch_size=int(config["train"]["evaluation_batch_size"]),
        shuffle=False,
        seed=single_experiment_seed(config),
    )
    model.eval()
    embeddings: list[Tensor] = []
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            with _autocast(config, device):
                batch_embeddings, batch_logits = model(images)
            embeddings.append(batch_embeddings.float().cpu())
            logits.append(batch_logits.float().cpu())
            if include_targets:
                targets.append(batch["label"].long().cpu())
    if not embeddings:
        raise ValueError("没有可评价的图像 batch")
    return (
        torch.cat(embeddings),
        torch.cat(logits),
        torch.cat(targets) if include_targets else None,
    )


def _pseudo_label_maps(
    state: SCPState | None,
    train_count: int,
    device: torch.device,
    *,
    reliability_power: float = 1.0,
    class_balance_exponent: float = 0.0,
    class_balance_max_multiplier: float = 2.0,
) -> tuple[Tensor, Tensor]:
    targets = torch.full((train_count,), -1, dtype=torch.long, device=device)
    weights = torch.zeros(train_count, dtype=torch.float32, device=device)
    if state is not None and state.selected_indices.numel() > 0:
        state.validate_alignment()
        indices = state.selected_indices.to(device=device, dtype=torch.long)
        if bool((indices < 0).any()) or bool((indices >= train_count).any()):
            raise ValueError("SCP selected_indices 超出 train 范围")
        targets[indices] = state.selected_pseudo_labels.to(
            device=device, dtype=torch.long
        )
        selected_weights = state.selected_reliability.to(device=device).float()
        power = float(reliability_power)
        if power <= 0.0 or not math.isfinite(power):
            raise ValueError("reliability_power必须为有限正数")
        if power != 1.0:
            original_total = selected_weights.sum()
            selected_weights = selected_weights.pow(power)
            selected_weights = selected_weights * (
                original_total
                / selected_weights.sum().clamp_min(torch.finfo(torch.float32).eps)
            )
        exponent = float(class_balance_exponent)
        max_multiplier = float(class_balance_max_multiplier)
        if exponent > 0.0:
            labels = state.selected_pseudo_labels.to(device=device, dtype=torch.long)
            class_count = state.class_count
            class_mass = torch.zeros(class_count, device=device, dtype=torch.float32)
            class_mass.scatter_add_(0, labels, selected_weights)
            present = class_mass > 0.0
            reference_mass = class_mass[present].mean()
            multipliers = torch.ones_like(class_mass)
            multipliers[present] = torch.minimum(
                (reference_mass / class_mass[present]).pow(exponent),
                torch.full_like(class_mass[present], max_multiplier),
            )
            original_total = selected_weights.sum()
            selected_weights = selected_weights * multipliers[labels]
            selected_weights = selected_weights * (
                original_total
                / selected_weights.sum().clamp_min(torch.finfo(torch.float32).eps)
            )
        weights[indices] = selected_weights
    return targets, weights


def _pseudo_probability_map(
    state: SCPState | None,
    train_count: int,
    device: torch.device,
    *,
    hard_weight: float,
) -> Tensor | None:
    if state is None or state.selected_indices.numel() == 0:
        return None
    blend = float(hard_weight)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("pseudo_target_hard_weight必须位于[0,1]")
    state.validate_alignment()
    indices = state.selected_indices.to(device=device, dtype=torch.long)
    if bool((indices < 0).any()) or bool((indices >= train_count).any()):
        raise ValueError("SCP selected_indices 超出 train 范围")
    posterior = state.selected_joint_posterior.to(device=device, dtype=torch.float32)
    hard = F.one_hot(
        state.selected_pseudo_labels.to(device=device, dtype=torch.long),
        num_classes=state.class_count,
    ).float()
    probabilities = torch.zeros(
        (train_count, state.class_count), device=device, dtype=torch.float32
    )
    probabilities[indices] = blend * hard + (1.0 - blend) * posterior
    return probabilities


def _pseudo_training_logits(
    model: MorphologyClassifier, images: Tensor, symmetry: str
) -> Iterator[Tensor]:
    if symmetry not in {"none", "hflip"}:
        raise ValueError("scp.pseudo_symmetry仅支持none/hflip")
    views = (images,) if symmetry == "none" else (images, images.flip(-1))
    return (model(view)[1] for view in views)


def _scp_selection_records(
    state: SCPState,
    records: Sequence[SampleRecord],
    class_names: Sequence[str],
    unlabeled_indices: Sequence[int],
    max_selected_per_class: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state.validate_alignment()
    if len(class_names) != state.class_count:
        raise ValueError("SCP state 类别数与数据契约不一致")
    if bool((state.candidate_indices < 0).any()) or bool(
        (state.candidate_indices >= len(records)).any()
    ):
        raise ValueError("SCP candidate_indices 超出 train 范围")
    unlabeled = set(int(index) for index in unlabeled_indices)
    if state.pool_count != len(unlabeled):
        raise ValueError("SCP pool_count 与 few-shot unlabeled pool 不一致")
    if not set(state.candidate_indices.tolist()).issubset(unlabeled):
        raise ValueError("SCP candidate_indices 必须全部来自 unlabeled pool")

    selected_ranks: dict[int, int] = {}
    for class_index in range(len(class_names)):
        positions = torch.nonzero(
            state.selected_pseudo_labels == class_index, as_tuple=False
        ).flatten()
        if len(positions) > int(max_selected_per_class):
            raise ValueError("SCP selected 数量超过每类 Top-Q 契约")
        identifiers = tuple(
            records[int(state.selected_indices[position])].sample_id
            for position in positions
        )
        if identifiers != state.selected_ids[class_index]:
            raise ValueError("SCP selected_ids 与 train sample identity 不一致")
        for rank, position in enumerate(positions.tolist(), start=1):
            selected_ranks[int(state.selected_indices[position])] = rank

    candidate_rows: list[dict[str, Any]] = []
    for position, train_index in enumerate(state.candidate_indices.tolist()):
        class_index = int(state.candidate_pseudo_labels[position])
        candidate_rows.append(
            {
                "train_index": train_index,
                "sample_id": records[train_index].sample_id,
                "pseudo_label_index": class_index,
                "pseudo_label": class_names[class_index],
                "reliability": float(state.candidate_reliability[position]),
                "view_a_target_probability": float(
                    state.candidate_deterministic_visual_confidence[position]
                ),
                "view_b_target_probability": float(
                    state.candidate_weak_visual_confidence[position]
                ),
                "structural_target_probability": float(
                    state.candidate_structural_confidence[position]
                ),
                "joint_target_probability": float(
                    state.candidate_joint_confidence[position]
                ),
                "joint_margin": float(state.candidate_joint_margin[position]),
                "weak_view_total_variation": float(
                    state.candidate_view_total_variation[position]
                ),
                "selected": train_index in selected_ranks,
                "selected_rank_within_class": selected_ranks.get(train_index),
            }
        )
    selected_rows = sorted(
        (row for row in candidate_rows if row["selected"]),
        key=lambda row: (
            int(row["pseudo_label_index"]),
            int(row["selected_rank_within_class"]),
        ),
    )
    return candidate_rows, selected_rows


def _labeled_epoch_loader(train_dataset, fewshot, config, epoch):
    seed = single_experiment_seed(config) + int(epoch)
    augmentation_seed = (
        seed if config["reproducibility"]["deterministic_augmentation"] else None
    )
    train_dataset.set_transform_seed(augmentation_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader = _make_loader(
        Subset(train_dataset, list(fewshot.labeled_indices)),
        config,
        batch_size=int(config["train"]["labeled_batch_size"]),
        shuffle=True,
        seed=seed + 1_000_003,
    )
    return loader, seed, augmentation_seed


def train_supervised_epoch(
    model: MorphologyClassifier,
    train_dataset: OnlineImageDataset,
    fewshot: FewShotSplit,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    class_names: Sequence[str],
    device: torch.device,
    epoch: int,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if not fewshot.labeled_indices:
        raise ValueError("base 训练必须有 labeled train 样本")
    loader, _, augmentation_seed = _labeled_epoch_loader(
        train_dataset, fewshot, config, epoch
    )
    full_batch_count = len(loader)
    batch_count = (
        full_batch_count
        if max_batches is None
        else min(full_batch_count, int(max_batches))
    )
    model.train()
    total_loss = 0.0
    supervised_loss_sum = 0.0
    seen: list[Tensor] = []
    metric_logits: list[Tensor] = []
    metric_targets: list[Tensor] = []
    batches = islice(loader, batch_count)
    progress = tqdm(
        batches,
        total=batch_count,
        desc=f"base {epoch + 1:03d}/{int(config['train']['epochs']):03d}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    head_lr = next(
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("name") == "head"
    )
    for batch_index, batch in enumerate(progress, start=1):
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with _autocast(config, device):
            _, logits = model(images)
            loss = _fixed_batch_cross_entropy(
                logits,
                labels,
                int(config["train"]["labeled_batch_size"]),
            )
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["train"]["gradient_clip_norm"]),
        )
        optimizer.step()
        indices = batch["index"].long().cpu()
        seen.append(indices)
        metric_logits.append(logits.detach().float().cpu())
        metric_targets.append(labels.detach().cpu())
        total_loss += float(loss.detach())
        supervised_loss_sum += float(
            F.cross_entropy(logits.detach().float(), labels, reduction="sum").cpu()
        )
        progress.set_postfix(
            loss=f"{total_loss / batch_index:.4f}",
            lr=f"{head_lr:.2e}",
            refresh=False,
        )
    if not seen:
        raise ValueError("base epoch 没有 labeled batch")
    seen_tensor = torch.cat(seen)
    presentations = int(seen_tensor.numel())
    unique_seen = int(torch.unique(seen_tensor).numel())
    complete = presentations == len(fewshot.labeled_indices) == unique_seen
    if max_batches is None and not complete:
        raise RuntimeError("完整 base epoch 必须恰好一次消费全部 labeled 样本")
    return {
        "loss": total_loss / len(seen),
        "supervised_ce": supervised_loss_sum / presentations,
        "scp": 0.0,
        "epoch_driver": "labeled_train_only",
        "optimizer_steps": len(seen),
        "labeled_available": len(fewshot.labeled_indices),
        "labeled_presentations": presentations,
        "unique_labeled_seen": unique_seen,
        "labeled_repeat_presentations": presentations - unique_seen,
        "labeled_batches": len(seen),
        "labeled_single_pass_complete": complete,
        "augmentation_seed": augmentation_seed,
        "scp_state_fingerprint": None,
        "classification": classification_metrics(
            torch.cat(metric_logits), torch.cat(metric_targets), class_names
        ),
    }


def train_scp_epoch(
    model: MorphologyClassifier,
    train_dataset: OnlineImageDataset,
    fewshot: FewShotSplit,
    scp_state: SCPState | None,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    class_names: Sequence[str],
    device: torch.device,
    epoch: int,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if not fewshot.labeled_indices:
        raise ValueError("SCP 训练必须有 labeled train 样本")
    labeled_loader, seed, augmentation_seed = _labeled_epoch_loader(
        train_dataset, fewshot, config, epoch
    )
    full_labeled_batch_count = len(labeled_loader)
    active_state = scp_state
    sample_ids = tuple(record.sample_id for record in train_dataset.records)
    selection_ramp_epochs = int(config["scp"].get("selection_ramp_epochs", 1))
    if scp_state is not None and selection_ramp_epochs > 1:
        active_state = scheduled_scp_state(
            scp_state,
            sample_ids,
            max_selected_per_class=int(config["scp"]["max_selected_per_class"]),
            ramp_epochs=selection_ramp_epochs,
            epoch=epoch,
        )
    selected_before_max_pseudo_labels = (
        0 if active_state is None else int(active_state.selected_indices.numel())
    )
    max_pseudo_labels = int(config["scp"]["max_pseudo_labels"])
    if active_state is not None:
        active_state = cap_scp_state_by_max_pseudo_labels(
            active_state,
            sample_ids,
            max_pseudo_labels=max_pseudo_labels,
        )
    max_pseudo_labels_active = bool(
        active_state is not None
        and len(active_state.selected_indices) < selected_before_max_pseudo_labels
    )
    small_label_pool = bool(
        len(fewshot.labeled_indices) * 100
        < len(train_dataset) * SMALL_LABEL_FRACTION_PERCENT
    )
    small_label_expansion_active = False
    scp_weight = float(config["scp"]["loss_weight"])
    if int(config["scp"].get("loss_ramp_epochs", 1)) > 1:
        scp_weight *= min(
            1.0, (int(epoch) + 1) / int(config["scp"]["loss_ramp_epochs"])
        )
    scp_active = (
        active_state is not None
        and scp_weight > 0.0
        and active_state.selected_indices.numel() > 0
    )
    pseudo_loader: DataLoader[Any] | None = None
    pseudo_iterator: Any = None
    pseudo_steps: frozenset[int] = frozenset()
    pseudo_batch_size = int(config["train"]["batch_size"])
    full_epoch_batch_count = full_labeled_batch_count
    if scp_active:
        assert active_state is not None
        selected = active_state.selected_indices.long().tolist()
        if len(selected) != len(set(selected)):
            raise ValueError("SCP selected_indices 不能重复")
        if not set(selected).issubset(set(fewshot.unlabeled_indices)):
            raise ValueError("SCP selected_indices 必须来自 unlabeled pool")
        pseudo_dataset = OnlineImageDataset(
            train_dataset.records,
            build_scp_weak_transform(config),
            include_structure=False,
            include_label=False,
            transform_seed=seed + 8_000_003,
        )
        if small_label_pool:
            pseudo_generator = torch.Generator().manual_seed(seed + 9_000_003)
            permutation = torch.randperm(
                len(selected), generator=pseudo_generator
            ).tolist()
            selected = tuple(selected[position] for position in permutation)
        pseudo_loader = _make_loader(
            Subset(pseudo_dataset, selected),
            config,
            batch_size=pseudo_batch_size,
            shuffle=not small_label_pool,
            seed=seed + 9_000_003,
        )
        if small_label_pool:
            full_epoch_batch_count = max(full_labeled_batch_count, len(pseudo_loader))
            small_label_expansion_active = (
                full_epoch_batch_count > full_labeled_batch_count
            )
        elif len(pseudo_loader) > full_labeled_batch_count:
            raise ValueError(
                "selected pseudo batch数超过labeled step；请减小Top-Q或增大pseudo batch"
            )
    batch_count = (
        full_epoch_batch_count
        if max_batches is None
        else min(full_epoch_batch_count, int(max_batches))
    )
    if pseudo_loader is not None:
        scheduled_count = min(len(pseudo_loader), batch_count)
        if not small_label_expansion_active and batch_count < full_labeled_batch_count:
            scheduled_count = min(
                scheduled_count,
                math.ceil(scheduled_count * batch_count / full_labeled_batch_count),
            )
        pseudo_steps = frozenset(
            _evenly_spaced_steps(batch_count, min(scheduled_count, batch_count))
        )
        pseudo_iterator = iter(pseudo_loader)

    pseudo_targets, pseudo_weights = _pseudo_label_maps(
        active_state if scp_active else None,
        len(train_dataset),
        device,
        reliability_power=float(config["scp"]["reliability_power"]),
        class_balance_exponent=float(config["scp"]["class_balance_exponent"]),
        class_balance_max_multiplier=float(
            config["scp"]["class_balance_max_multiplier"]
        ),
    )
    pseudo_symmetry = str(config["scp"].get("pseudo_symmetry", "none"))
    pseudo_view_count = {"none": 1, "hflip": 2}[pseudo_symmetry]
    pseudo_probabilities = (
        _pseudo_probability_map(
            active_state if scp_active else None,
            len(train_dataset),
            device,
            hard_weight=float(config["scp"].get("pseudo_target_hard_weight", 1.0)),
        )
        if config["scp"].get("pseudo_target_mode", "hard") == "joint_soft"
        else None
    )
    model.train()
    totals = {"loss": 0.0, "scp": 0.0}
    supervised_loss_sum = 0.0
    supervised_loss_weight_sum = 0.0
    optimizer_step_count = 0
    labeled_seen: list[Tensor] = []
    pseudo_seen: list[Tensor] = []
    metric_logits: list[Tensor] = []
    metric_targets: list[Tensor] = []
    pseudo_batch_count = 0
    extra_labeled_batch_count = max(0, batch_count - full_labeled_batch_count)
    present_pseudo_classes = (
        set()
        if active_state is None
        else set(int(value) for value in active_state.selected_pseudo_labels.tolist())
    )
    priority_classes = tuple(
        class_index
        for class_index in range(len(class_names))
        if class_index not in present_pseudo_classes
    )
    if extra_labeled_batch_count:
        repeated_labeled_loader = _make_replacement_labeled_loader(
            train_dataset,
            fewshot,
            config,
            batch_size=int(config["train"]["labeled_batch_size"]),
            batch_count=extra_labeled_batch_count,
            seed=seed + 2_000_003,
            priority_classes=priority_classes,
        )
        labeled_batches = chain(labeled_loader, repeated_labeled_loader)
    else:
        labeled_batches = iter(labeled_loader)
    batches = islice(labeled_batches, batch_count)
    progress = tqdm(
        batches,
        total=batch_count,
        desc=f"scp {epoch + 1:03d}/{int(config['train']['epochs']):03d}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    head_lr = next(
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("name") == "head"
    )
    for batch_index, labeled_batch in enumerate(progress):
        pseudo_batch = None
        if batch_index in pseudo_steps:
            assert pseudo_iterator is not None
            pseudo_batch = next(pseudo_iterator)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            labeled_images = labeled_batch["image"].to(device, non_blocking=True)
            labeled_labels = labeled_batch["label"].to(device, non_blocking=True)
            labeled_indices = labeled_batch["index"].to(device, non_blocking=True)
            _, labeled_logits = model(labeled_images)
            supervised_loss = _fixed_batch_cross_entropy(
                labeled_logits,
                labeled_labels,
                int(config["train"]["labeled_batch_size"]),
            )
            supervised_loss_weight = (
                1.0
                if batch_index < full_labeled_batch_count
                else float(config["scp"].get("resampled_labeled_loss_weight", 1.0))
            )
            supervised_component = supervised_loss_weight * supervised_loss


        supervised_component.backward()
        if pseudo_batch is None:
            scp_loss = torch.zeros((), device=device)
            pseudo_count = 0
        else:
            pseudo_images = pseudo_batch["image"].to(device, non_blocking=True)
            pseudo_indices = pseudo_batch["index"].to(device, non_blocking=True)
            pseudo_logits_iterator = _pseudo_training_logits(
                model, pseudo_images, pseudo_symmetry
            )
            scp_loss = torch.zeros((), device=device)
            pseudo_count = 0
            for _ in range(pseudo_view_count):
                with _autocast(config, device):
                    pseudo_logits = next(pseudo_logits_iterator)
                    current_loss, current_count = weighted_pseudo_label_loss(
                        pseudo_logits,
                        pseudo_indices,
                        pseudo_targets,
                        pseudo_weights,
                        reference_batch_size=pseudo_batch_size,
                        pseudo_probabilities=pseudo_probabilities,
                    )
                    pseudo_component = scp_weight * current_loss / pseudo_view_count
                pseudo_component.backward()
                scp_loss += current_loss.detach() / pseudo_view_count
                if pseudo_count not in {0, current_count}:
                    raise RuntimeError("伪样本多视图的有效数量不一致")
                pseudo_count = current_count
            if pseudo_count != len(pseudo_indices):
                raise RuntimeError("selected pseudo batch 中出现未定义标签")
        total_loss = supervised_component.detach() + scp_weight * scp_loss.detach()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["train"]["gradient_clip_norm"]),
        )
        optimizer.step()
        optimizer_step_count += 1
        labeled_seen.append(labeled_indices.detach().cpu())
        metric_logits.append(labeled_logits.detach().float().cpu())
        metric_targets.append(labeled_labels.detach().cpu())
        supervised_loss_sum += float(
            F.cross_entropy(
                labeled_logits.detach().float(),
                labeled_labels,
                reduction="sum",
            ).cpu()
        )
        supervised_loss_weight_sum += supervised_loss_weight
        if pseudo_batch is not None:
            pseudo_seen.append(pseudo_indices.detach().cpu())
            pseudo_batch_count += 1
        totals["loss"] += float(total_loss.detach())
        totals["scp"] += float(scp_loss.detach())
        labeled_presentations = sum(len(value) for value in labeled_seen)
        progress.set_postfix(
            loss=f"{totals['loss'] / optimizer_step_count:.4f}",
            ce=f"{supervised_loss_sum / labeled_presentations:.4f}",
            scp=f"{totals['scp'] / optimizer_step_count:.4f}",
            pseudo=sum(len(value) for value in pseudo_seen),
            lr=f"{head_lr:.2e}",
        )

    if optimizer_step_count == 0:
        raise ValueError("SCP epoch 没有 labeled batch")
    labeled_tensor = torch.cat(labeled_seen)
    unique_labeled = int(torch.unique(labeled_tensor).numel())
    labeled_presentations = int(labeled_tensor.numel())
    labeled_repeats = labeled_presentations - unique_labeled
    labeled_single_pass_complete = labeled_presentations == len(
        fewshot.labeled_indices
    ) and unique_labeled == len(fewshot.labeled_indices)
    labeled_coverage_complete = labeled_presentations >= len(
        fewshot.labeled_indices
    ) and unique_labeled == len(fewshot.labeled_indices)
    pseudo_tensor = (
        torch.cat(pseudo_seen) if pseudo_seen else torch.empty(0, dtype=torch.long)
    )
    unique_pseudo = int(torch.unique(pseudo_tensor).numel())
    pseudo_presentations = int(pseudo_tensor.numel())
    pseudo_repeats = pseudo_presentations - unique_pseudo
    selected_available = (
        0 if active_state is None else int(active_state.selected_indices.numel())
    )
    pseudo_complete = not scp_active or (
        pseudo_presentations == selected_available
        and unique_pseudo == pseudo_presentations
    )
    if max_batches is None and not labeled_coverage_complete:
        raise RuntimeError("完整 SCP epoch 必须覆盖全部 labeled 样本")
    if max_batches is None and not pseudo_complete:
        raise RuntimeError("完整 SCP epoch 必须恰好一次消费全部 selected pseudo 样本")

    return {
        "loss": totals["loss"] / optimizer_step_count,
        "supervised_ce": supervised_loss_sum / labeled_presentations,
        "scp": totals["scp"] / optimizer_step_count,
        "epoch_driver": (
            "pseudo_coverage_with_labeled_resampling"
            if small_label_expansion_active
            else "labeled_train_primary"
        ),
        "optimizer_steps": optimizer_step_count,
        "supervised_optimizer_steps": optimizer_step_count,
        "pseudo_only_optimizer_steps": 0,
        "labeled_available": len(fewshot.labeled_indices),
        "labeled_presentations": labeled_presentations,
        "unique_labeled_seen": unique_labeled,
        "labeled_repeat_presentations": labeled_repeats,
        "labeled_batches": optimizer_step_count,
        "labeled_single_pass_complete": labeled_single_pass_complete,
        "labeled_coverage_complete": labeled_coverage_complete,
        "labeled_resampled_batches": extra_labeled_batch_count,
        "labeled_resampled_loss_weight": float(
            config["scp"].get("resampled_labeled_loss_weight", 1.0)
        ),
        "supervised_loss_weight_sum": supervised_loss_weight_sum,
        "labeled_resampling_policy": "missing_pseudo_classes",
        "labeled_priority_steps": (
            extra_labeled_batch_count if priority_classes else 0
        ),
        "labeled_priority_classes": list(priority_classes),
        "labeled_priority_presentations": (
            extra_labeled_batch_count
            * min(
                len(priority_classes),
                int(config["train"]["labeled_batch_size"]),
            )
        ),
        "scp_selected_available": selected_available,
        "scp_selected_before_max_pseudo_labels": selected_before_max_pseudo_labels,
        "scp_max_pseudo_labels_active": max_pseudo_labels_active,
        "small_label_expansion_active": small_label_expansion_active,
        "small_label_fraction_threshold": SMALL_LABEL_FRACTION_PERCENT / 100,
        "scp_max_pseudo_labels": max_pseudo_labels,
        "scp_presentations": pseudo_presentations,
        "unique_scp_seen": unique_pseudo,
        "scp_repeat_presentations": pseudo_repeats,
        "scp_batches": pseudo_batch_count,
        "scp_single_pass_complete": pseudo_complete,
        "scp_state_prepared_once": bool(
            scp_state is not None and not config["scp"].get("refresh_epochs")
        ),
        "scp_refresh_count": 0,
        "scp_selection_policy": (
            "global_reliability_top_count"
            if max_pseudo_labels_active
            else "fixed_candidate_pool_static_topq"
        ),
        "scp_selection_ramp_epochs": (
            selection_ramp_epochs if selection_ramp_epochs > 1 else 0
        ),
        "scp_active_state_fingerprint": (
            None if active_state is None else active_state.fingerprint()
        ),
        "scp_active_per_class": (
            []
            if active_state is None
            else torch.bincount(
                active_state.selected_pseudo_labels,
                minlength=len(class_names),
            ).tolist()
        ),
        "scp_effective_loss_weight": scp_weight,
        "scp_loss_ramp_epochs": int(config["scp"].get("loss_ramp_epochs", 1)),
        "scp_pseudo_train_augmentation": "weak",
        "scp_pseudo_symmetry": pseudo_symmetry,
        "scp_pseudo_view_count": pseudo_view_count,
        "scp_pseudo_target_mode": str(config["scp"].get("pseudo_target_mode", "hard")),
        "scp_pseudo_target_hard_weight": float(
            config["scp"].get("pseudo_target_hard_weight", 1.0)
        ),
        "scp_labeled_manifold_mixup_alpha": None,
        "scp_labeled_symmetry": "none",
        "scp_labeled_view_count": 1,
        "scp_labeled_c4_batches": 0,
        "scp_resampled_labeled_loss_weight": float(
            config["scp"].get("resampled_labeled_loss_weight", 1.0)
        ),
        "scp_state_fingerprint": (
            None if scp_state is None else scp_state.fingerprint()
        ),
        "augmentation_seed": augmentation_seed,
        "scp_class_balance_exponent": float(config["scp"]["class_balance_exponent"]),
        "scp_class_balance_max_multiplier": float(
            config["scp"]["class_balance_max_multiplier"]
        ),
        "classification": classification_metrics(
            torch.cat(metric_logits), torch.cat(metric_targets), class_names
        ),
    }


def _environment(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torchvision", "numpy", "Pillow", "PyYAML")
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
            for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "reproducibility": {
            **dict(config["reproducibility"]),
            "cpu_num_threads": torch.get_num_threads(),
            "cpu_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdp_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def weighted_pseudo_label_loss(
    logits,
    sample_indices,
    pseudo_targets,
    pseudo_weights,
    *,
    reference_batch_size,
    pseudo_probabilities=None,
):
    indices = sample_indices.to(device=pseudo_targets.device, dtype=torch.long)
    targets = pseudo_targets[indices]
    accepted = targets >= 0
    count = int(accepted.sum())
    if count == 0:
        return logits.sum() * 0.0, 0
    if reference_batch_size <= 0:
        raise ValueError("reference_batch_size必须为正数")
    weights = pseudo_weights[indices[accepted]].to(
        device=logits.device, dtype=logits.dtype
    )
    accepted_logits = logits[accepted.to(logits.device)]
    accepted_targets = targets[accepted].to(logits.device)
    if pseudo_probabilities is not None:
        if tuple(pseudo_probabilities.shape) != (
            len(pseudo_targets),
            accepted_logits.shape[1],
        ):
            raise ValueError("pseudo_probabilities形状必须为[train_count,class_count]")
        probabilities = pseudo_probabilities[indices[accepted]].to(
            device=logits.device, dtype=torch.float32
        )
        if (
            not torch.isfinite(probabilities).all()
            or bool((probabilities < 0).any())
            or not torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(len(probabilities), device=logits.device),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise ValueError("有效软伪目标必须为归一化非负概率")
        losses = -(probabilities * F.log_softmax(accepted_logits.float(), dim=-1)).sum(
            dim=1
        )
    else:
        losses = F.cross_entropy(
            accepted_logits.float(), accepted_targets, reduction="none"
        )
    return (losses * weights).sum() / float(reference_batch_size), count


def _labeled_prototype_logits(
    object_embeddings: Tensor,
    labeled_indices: Sequence[int],
    labeled_labels: Tensor,
    class_count: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    embeddings = F.normalize(object_embeddings.detach().float(), dim=1)
    indices = torch.as_tensor(labeled_indices, dtype=torch.long)
    labels = labeled_labels.detach().cpu().long()
    if embeddings.ndim != 2 or len(indices) != len(labels) or len(indices) == 0:
        raise ValueError("原型表征与有标签索引必须非空且一一对应")
    if float(scale) <= 0 or not math.isfinite(float(scale)):
        raise ValueError("prototype logit scale必须为有限正数")
    labeled = embeddings[indices]
    prototypes = []
    for class_index in range(int(class_count)):
        members = labeled[labels == class_index]
        if len(members) == 0:
            raise ValueError("每类必须至少有一个真实标签用于构造原型")
        prototypes.append(F.normalize(members.mean(dim=0), dim=0))
    prototype_tensor = torch.stack(prototypes)
    return float(scale) * (embeddings @ prototype_tensor.T), prototype_tensor


def _build_scp_state(
    model,
    config,
    audit,
    fewshot,
    standardized_structures,
    device,
):
    labels = torch.tensor(
        [audit.records["train"][i].label for i in fewshot.labeled_indices],
        dtype=torch.long,
    )
    if str(config["scp"].get("prototype_mode", "none")) != "none":
        original_logits, weak_logits = _labeled_prototype_visual_logits(
            model, config, audit, fewshot, labels, device
        )
    else:
        _, original_logits, _ = encode_records(
            model, audit.records["train"], config, device, include_targets=False
        )
        _, weak_logits, _ = encode_records(
            model,
            audit.records["train"],
            config,
            device,
            transform=build_scp_weak_transform(config),
            transform_seed=single_experiment_seed(config) + 7_000_003,
            include_targets=False,
        )
    selection_keys = (
        "structural_knn",
        "max_selected_per_class",
        "visual_temperature",
        "visual_confidence_threshold",
        "structural_confidence_threshold",
        "joint_margin_threshold",
        "weak_view_max_total_variation",
    )
    return build_scp_state(
        standardized_structures,
        original_logits,
        weak_logits,
        fewshot.labeled_indices,
        fewshot.unlabeled_indices,
        labels,
        audit.classes,
        tuple(row.sample_id for row in audit.records["train"]),
        **{key: config["scp"][key] for key in selection_keys},
    )


def _labeled_prototype_visual_logits(model, config, audit, fewshot, labels, device):
    views = []
    for angle in PROTOTYPE_VIEW_ANGLES[config["scp"]["prototype_mode"]]:
        embeddings, _, _ = encode_records(
            model,
            audit.records["train"],
            config,
            device,
            transform=_view_transform(config, angle, False),
            include_targets=False,
        )
        views.append(embeddings)
    object_embeddings = F.normalize(torch.stack(views).mean(dim=0), dim=1)
    original_logits, prototypes = _labeled_prototype_logits(
        object_embeddings,
        fewshot.labeled_indices,
        labels,
        len(audit.classes),
        float(config["scp"]["prototype_logit_scale"]),
    )
    weak_embeddings, _, _ = encode_records(
        model,
        audit.records["train"],
        config,
        device,
        transform=build_scp_weak_transform(config),
        transform_seed=single_experiment_seed(config) + 7_000_003,
        include_targets=False,
    )
    weak_logits = float(config["scp"]["prototype_logit_scale"]) * (
        F.normalize(weak_embeddings.float(), dim=1) @ prototypes.T
    )
    return original_logits, weak_logits


def _active_training_scp_state(state, config, audit):
    return cap_scp_state_by_max_pseudo_labels(
        state,
        tuple(record.sample_id for record in audit.records["train"]),
        max_pseudo_labels=int(config["scp"]["max_pseudo_labels"]),
    )


def _scp_refresh_transition(before, after, config, audit):
    before_active = _active_training_scp_state(before, config, audit)
    after_active = _active_training_scp_state(after, config, audit)
    if not len(after_active.selected_indices):
        raise ValueError("SCP刷新得到空伪标签集合")
    before_labels = dict(
        zip(
            before_active.selected_indices.tolist(),
            before_active.selected_pseudo_labels.tolist(),
            strict=True,
        )
    )
    after_labels = dict(
        zip(
            after_active.selected_indices.tolist(),
            after_active.selected_pseudo_labels.tolist(),
            strict=True,
        )
    )
    before_indices = set(before_labels)
    after_indices = set(after_labels)
    retained = before_indices & after_indices

    def per_class(state):
        counts = torch.bincount(
            state.selected_pseudo_labels, minlength=len(audit.classes)
        )
        return {
            class_name: int(counts[class_index])
            for class_index, class_name in enumerate(audit.classes)
        }

    return {
        "before_selected_count": len(before_indices),
        "after_selected_count": len(after_indices),
        "retained_count": len(retained),
        "added_count": len(after_indices - before_indices),
        "removed_count": len(before_indices - after_indices),
        "retained_label_changed_count": sum(
            before_labels[index] != after_labels[index] for index in retained
        ),
        "before_per_class": per_class(before_active),
        "after_per_class": per_class(after_active),
        "before_source_state_fingerprint": before.fingerprint(),
        "after_source_state_fingerprint": after.fingerprint(),
        "before_active_state_fingerprint": before_active.fingerprint(),
        "after_active_state_fingerprint": after_active.fingerprint(),
    }


def _write_scp_selection(
    run_dir,
    state,
    config,
    audit,
    fewshot,
    initial_source,
    *,
    refresh_after_epoch=None,
    refresh_count=0,
    source_last_sha256=None,
    transition=None,
):
    prototype_mode = str(config["scp"].get("prototype_mode", "none"))
    is_refresh = refresh_after_epoch is not None
    if is_refresh != (source_last_sha256 is not None and transition is not None):
        raise ValueError("SCP刷新审计缺少last.pt来源或状态变化")
    if is_refresh and int(refresh_count) <= 0:
        raise ValueError("SCP刷新计数必须为正数")
    active_state = _active_training_scp_state(state, config, audit)
    max_pseudo_labels_active = len(active_state.selected_indices) < len(
        state.selected_indices
    )
    candidates, selected = _scp_selection_records(
        active_state,
        audit.records["train"],
        audit.classes,
        fewshot.unlabeled_indices,
        config["scp"]["max_selected_per_class"],
    )
    payload = {
        **active_state.diagnostics(audit.classes),
        "prepared_once": not is_refresh,
        "refresh_count": int(refresh_count),
        "training_selection_policy": (
            "global_reliability_top_count"
            if max_pseudo_labels_active
            else (
                "fixed_candidate_pool_static_topq_ramp"
                if int(config["scp"].get("selection_ramp_epochs", 1)) > 1
                else "fixed_candidate_pool_static_topq"
            )
        ),
        "maximum_active_selected_count": len(active_state.selected_indices),
        "selected_count_before_max_pseudo_labels": len(state.selected_indices),
        "source_state_fingerprint": state.fingerprint(),
        "small_label_fraction_threshold": SMALL_LABEL_FRACTION_PERCENT / 100,
        "max_pseudo_labels": int(config["scp"]["max_pseudo_labels"]),
        "uses_unlabeled_ground_truth": False,
        "source": (
            (
                f"student_{prototype_mode}_prototypes"
                if is_refresh
                else f"initial_{prototype_mode}_prototypes"
            )
            if prototype_mode != "none"
            else (
                "student_last_checkpoint"
                if is_refresh
                else "initial_checkpoint_classifier"
            )
        ),
        "deterministic_visual_posterior": {
            "aggregation": (
                "c4_embedding_mean" if prototype_mode != "none" else "single_view"
            ),
            "view_count": (
                len(PROTOTYPE_VIEW_ANGLES[prototype_mode])
                if prototype_mode != "none"
                else 1
            ),
        },
        "initial_checkpoint_sha256": initial_source["sha256"],
        "selection": dict(config["scp"]),
        "candidate_records": candidates,
        "selected_records": selected,
    }
    if is_refresh:
        payload.update(
            refresh_after_epoch=int(refresh_after_epoch),
            source_last_checkpoint={
                "relative_path": "checkpoints/last.pt",
                "sha256": str(source_last_sha256),
                "sha256_semantics": "canonical_checkpoint_content",
            },
            transition=dict(transition),
        )
    filename = (
        f"scp_selection_refresh_after_epoch_{int(refresh_after_epoch):03d}.json"
        if is_refresh
        else "scp_selection.json"
    )
    write_json(Path(run_dir) / filename, payload)


def train_from_images(
    config,
    audit,
    run_dir,
    device,
    *,
    resume_path=None,
    labeled_seed=None,
    max_train_batches=None,
    max_val_batches=None,
    prepare_only=False,
):
    run_dir = Path(run_dir)
    is_scp = config["train"]["mode"] == "scp_from_checkpoint"
    refresh_epochs = (
        tuple(int(value) for value in config["scp"].get("refresh_epochs", ()))
        if is_scp
        else ()
    )
    if prepare_only and (not is_scp or resume_path is not None):
        raise ValueError("prepare-only只支持新建SCP运行")
    checkpoint = None
    if resume_path is not None:
        checkpoint = load_torch_mapping(resume_path)
        if (
            checkpoint.get("format_version") != 3
            or checkpoint.get("contract", {}).get("schema_version") not in (10, 11)
            or checkpoint.get("role") != "last"
        ):
            raise ValueError("精确恢复只接受新契约last.pt")
        checkpoint_labeled_seed = checkpoint["contract"].get("fewshot_seed")
        if type(checkpoint_labeled_seed) is not int or checkpoint_labeled_seed < 0:
            raise ValueError("恢复checkpoint缺少有效的fewshot_seed")
        if labeled_seed is not None and labeled_seed != checkpoint_labeled_seed:
            raise ValueError("恢复时labeled_seed必须与checkpoint契约一致")
        labeled_seed = checkpoint_labeled_seed
    elif labeled_seed is None:
        labeled_seed = config["dataset"].get(
            "labeled_seed", single_experiment_seed(config)
        )
    configured_labeled_seed = config["dataset"].get("labeled_seed")
    if configured_labeled_seed is not None and configured_labeled_seed != labeled_seed:
        raise ValueError("dataset.labeled_seed与实际标签划分seed不一致")
    if type(labeled_seed) is not int or labeled_seed < 0:
        raise ValueError("labeled_seed必须是非负整数")
    fewshot = build_fewshot_split(
        audit.records["train"],
        audit.classes,
        float(config["dataset"]["label_fraction"]),
        labeled_seed,
    )
    if not fewshot.labeled_indices:
        raise ValueError("label_fraction=0，没有可训练的标签")
    if is_scp and not fewshot.unlabeled_indices:
        raise ValueError("SCP需要非空unlabeled pool")
    train_dataset = build_image_dataset(config, audit.records["train"], "train")
    structures = _records_structures(audit.records["train"])
    scaler = RobustScaler.fit(structures)
    standardized = scaler.transform(structures)
    model = _build_model(config, len(audit.classes), device)
    development_only = (
        prepare_only or max_train_batches is not None or max_val_batches is not None
    )
    initial_source = None
    state = None
    scp_refresh_count = 0
    best_validation = None
    start_epoch = 0
    if checkpoint is not None:
        initial_source = checkpoint["contract"]["initial_source"]
    elif is_scp:
        initial_path = (
            Path(config["train"]["initial_checkpoint"]).expanduser().resolve()
        )
        initial = load_torch_mapping(initial_path)
        initial_source = _validate_initial_checkpoint(
            initial, initial_path, config, audit, fewshot, model.backbone
        )
        _load_model_state(model, initial)
    contract = _training_contract(
        config,
        audit,
        fewshot,
        model.backbone,
        initial_source,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    optimizer = None if prepare_only else _build_optimizer(model, config)
    if checkpoint is not None:
        restored_contract = _canonical_training_contract(checkpoint["contract"])
        if restored_contract["schema_version"] == 10:


            restored_contract.setdefault("training_seed", single_experiment_seed(config))
        restored_contract["schema_version"] = 11
        if (
            restored_contract != contract
            or bool(checkpoint["development_only"]) != development_only
        ):
            raise ValueError("resume checkpoint与当前数据/配置契约不一致")
        _validate_resume_history(run_dir, checkpoint)
        _load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = checkpoint["best_validation"]
        saved_refresh_count = checkpoint.get("scp_refresh_count", 0)
        if type(saved_refresh_count) is not int:
            raise ValueError("checkpoint scp_refresh_count必须是整数")
        scp_refresh_count = saved_refresh_count
        expected_refresh_count = sum(
            refresh_epoch < start_epoch for refresh_epoch in refresh_epochs
        )
        if not 0 <= scp_refresh_count <= len(refresh_epochs):
            raise ValueError("checkpoint scp_refresh_count越界")
        if scp_refresh_count != expected_refresh_count:
            raise ValueError("checkpoint SCP刷新计数与epoch边界不一致")
        if is_scp:
            if checkpoint["scp_state"] is None:
                raise ValueError("SCP checkpoint缺少固定状态")
            state = SCPState.from_checkpoint(checkpoint["scp_state"])
            if (
                state.fingerprint()
                != checkpoint["metrics"]["train"]["scp_state_fingerprint"]
            ):
                raise ValueError("SCP state fingerprint不一致")
        elif checkpoint["scp_state"] is not None:
            raise ValueError("Base checkpoint不应包含SCP状态")
        saved_scaler = checkpoint["structural_scaler"]
        if (
            not torch.equal(saved_scaler["median"], scaler.median)
            or not torch.equal(saved_scaler["iqr"], scaler.iqr)
            or saved_scaler["epsilon"] != scaler.epsilon
        ):
            raise ValueError("恢复结构标准化统计量不一致")
        if checkpoint["metrics"]["is_best"]:
            best_path = run_dir / "checkpoints/best.pt"
            if (
                not best_path.exists()
                or load_torch_mapping(best_path)["epoch"] != checkpoint["epoch"]
            ):
                repaired = dict(checkpoint)
                repaired.update(role="best_validation", optimizer_state=None)
                save_torch_atomic(best_path, repaired)
                write_json(run_dir / "best_validation.json", best_validation)
    logger = setup_logger(run_dir / "train.log", "mrd_scp.train")
    write_json(run_dir / "fewshot_split.json", fewshot.as_dict())
    write_json(
        run_dir / "structural_scaler.json",
        {"structural_keys": list(audit.structural_keys), **scaler.to_dict()},
    )
    write_json(run_dir / "checkpoint_contract.json", contract)
    write_json(
        run_dir / "backbone.json",
        model.backbone.metadata(contract["input_size"]),
    )
    write_json(
        run_dir / "run_kind.json",
        {
            "development_only": development_only,
            "training_mode": config["train"]["mode"],
            "labeled_epoch_policy": contract["labeled_epoch_policy"],
            "scp_state_policy": contract["scp_state_policy"],
            "prepare_only": prepare_only,
            "loads_test_images": not prepare_only,
            "test_observation_only": not prepare_only,
            "checkpoint_selection_split": "val",
            "validation_tta_enabled": False,
        },
    )
    if initial_source is not None:
        write_json(run_dir / "initial_checkpoint.json", initial_source)
    if prepare_only:
        state = _build_scp_state(model, config, audit, fewshot, standardized, device)
        _write_scp_selection(run_dir, state, config, audit, fewshot, initial_source)
        active_state = _active_training_scp_state(state, config, audit)
        targets, weights = _pseudo_label_maps(
            active_state,
            len(train_dataset),
            device,
            reliability_power=config["scp"].get("reliability_power", 1.0),
            class_balance_exponent=config["scp"]["class_balance_exponent"],
            class_balance_max_multiplier=config["scp"]["class_balance_max_multiplier"],
        )
        save_torch_atomic(
            run_dir / "scp_state.pt",
            {
                "contract": contract,
                "scp_state": active_state.to_checkpoint(),
                "scp_refresh_count": 0,
                "targets": targets.cpu(),
                "weights": weights.cpu(),
                "structural_scaler": scaler.to_checkpoint(),
            },
        )
        return run_dir
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        head_lr, backbone_lr = _set_epoch_learning_rates(optimizer, config, epoch)
        refresh_transition = None
        refreshed_before_epoch = False
        if is_scp and state is None:
            state = _build_scp_state(
                model, config, audit, fewshot, standardized, device
            )
            _write_scp_selection(run_dir, state, config, audit, fewshot, initial_source)
        if is_scp and epoch in refresh_epochs:
            if state is None:
                raise RuntimeError("SCP刷新前缺少上一状态")
            expected_before_refresh = refresh_epochs.index(epoch)
            if scp_refresh_count != expected_before_refresh:
                raise ValueError("SCP刷新节点已执行次数与当前epoch不一致")
            last_path = run_dir / "checkpoints/last.pt"
            if not last_path.is_file():
                raise ValueError("SCP刷新节点缺少上一epoch的last.pt")
            source_last_sha256 = state_digest(load_torch_mapping(last_path))
            refreshed_state = _build_scp_state(
                model,
                config,
                audit,
                fewshot,
                standardized,
                device,
            )
            refresh_transition = _scp_refresh_transition(
                state, refreshed_state, config, audit
            )
            _write_scp_selection(
                run_dir,
                refreshed_state,
                config,
                audit,
                fewshot,
                initial_source,
                refresh_after_epoch=epoch,
                refresh_count=scp_refresh_count + 1,
                source_last_sha256=source_last_sha256,
                transition=refresh_transition,
            )
            state = refreshed_state
            scp_refresh_count += 1
            refreshed_before_epoch = True
        if is_scp:
            train_metrics = train_scp_epoch(
                model,
                train_dataset,
                fewshot,
                state,
                optimizer,
                config,
                audit.classes,
                device,
                epoch,
                max_batches=max_train_batches,
            )
            train_metrics["scp_refresh_count"] = scp_refresh_count
            train_metrics["scp_refreshed_before_epoch"] = refreshed_before_epoch
            if refresh_transition is not None:
                train_metrics["scp_refresh_transition"] = refresh_transition
        else:
            train_metrics = train_supervised_epoch(
                model,
                train_dataset,
                fewshot,
                optimizer,
                config,
                audit.classes,
                device,
                epoch,
                max_batches=max_train_batches,
            )
        evaluation_metrics = {}
        for split in ("val", "test"):
            _, logits, targets = encode_records(
                model, audit.records[split], config, device, max_batches=max_val_batches
            )
            evaluation_metrics[split] = classification_metrics(
                logits, targets, audit.classes
            )
        val_metrics = evaluation_metrics["val"]
        improved = is_better_validation(val_metrics, best_validation)
        if improved:
            best_validation = {"epoch": epoch + 1, **val_metrics}
        metrics = {
            "epoch": epoch + 1,
            "epochs": int(config["train"]["epochs"]),
            "head_learning_rate": head_lr,
            "backbone_learning_rate": backbone_lr,
            "is_best": improved,
            "train": train_metrics,
            **evaluation_metrics,
            "state_digests": {
                "model": state_digest(model.state_dict()),
                "optimizer": state_digest(optimizer.state_dict()),
            },
        }
        append_jsonl(run_dir / "history.jsonl", metrics)
        payload = _checkpoint_payload(
            role="last",
            epoch=epoch,
            model=model,
            scp_state=state,
            scp_refresh_count=scp_refresh_count,
            scaler=scaler,
            optimizer=optimizer,
            contract=contract,
            metrics=metrics,
            best_validation=best_validation,
            development_only=development_only,
            training_complete=epoch + 1 == int(config["train"]["epochs"]),
        )
        save_torch_atomic(run_dir / "checkpoints/last.pt", payload)
        if improved:
            payload = dict(payload)
            payload.update(role="best_validation", optimizer_state=None)
            save_torch_atomic(run_dir / "checkpoints/best.pt", payload)
            write_json(run_dir / "best_validation.json", best_validation)
        logger.info(
            "epoch=%03d/%03d | validation_best=%s",
            epoch + 1,
            config["train"]["epochs"],
            improved,
        )
        for split, split_metrics in evaluation_metrics.items():
            log_classification_metrics(
                logger,
                split=split,
                loss=split_metrics["nll"],
                metrics=split_metrics,
                per_class=True,
            )
    best_path = run_dir / "checkpoints/best.pt"
    if best_path.exists():
        best = load_torch_mapping(best_path)
        best["training_complete"] = True
        save_torch_atomic(best_path, best)
    return run_dir


def _run_single_seed(config, args, device, *, labeled_seed):
    configure_reproducibility(config)
    seed_everything(single_experiment_seed(config))
    audit = validate_dataset(config)
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume is not None:
        if resume.name != "last.pt" or resume.parent.name != "checkpoints":
            raise ValueError("--resume必须指向checkpoints/last.pt")
        run_dir = resume.parent.parent
    else:
        run_dir = create_run_dir(
            config["output"]["root"], Path(args.config).stem
        )
        save_config(config, run_dir / "config_resolved.yaml")
        write_json(run_dir / "dataset_audit.json", audit.as_dict())
        write_json(run_dir / "environment.json", _environment(config))
    train_from_images(
        config,
        audit,
        run_dir,
        device,
        resume_path=resume,
        labeled_seed=labeled_seed,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        prepare_only=args.prepare_only,
    )
    if resume is not None:
        save_config(config, run_dir / "config_resolved_resume.yaml")
    return run_dir


def _batch_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train Base or SCP image classifiers.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="development模式下同时截断val与test的前向batch数",
    )
    args = parser.parse_args(argv)
    for limit in (args.max_train_batches, args.max_val_batches):
        if limit is not None and limit <= 0:
            raise ValueError("development批次数必须为正数")
    template = load_config(args.config)
    seeds = experiment_seeds(template)
    if args.resume and len(seeds) != 1:
        raise ValueError("--resume必须使用仅含一个seed的config_resolved.yaml")
    labeled_seed = template["dataset"].get("labeled_seed", seeds[0])
    device = resolve_device(args.device)
    status_path = (
        Path(template["output"]["root"]).expanduser().resolve()
        / Path(args.config).stem
        / f"seed_batch_{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    )
    status = {
        "schema_version": 1,
        "input_config": str(Path(args.config).expanduser().resolve()),
        "seeds": list(seeds),
        "labeled_seed": None if args.resume else labeled_seed,
        "status": "running",
        "started_at": _batch_time(),
        "ended_at": None,
        "runs": [],
    }
    write_json(status_path, status)
    last_run_dir = None
    failures = 0
    for seed in seeds:
        config = copy.deepcopy(template)
        config["experiment"]["seed"] = [seed]
        record = {
            "seed": seed,
            "status": "running",
            "run_dir": None,
            "started_at": _batch_time(),
            "ended_at": None,
            "error": None,
        }
        status["runs"].append(record)
        write_json(status_path, status)
        try:
            run_dir = _run_single_seed(
                config,
                args,
                device,
                labeled_seed=None if args.resume else labeled_seed,
            )
        except (KeyboardInterrupt, SystemExit):
            record["status"] = "interrupted"
            record["ended_at"] = _batch_time()
            status["status"] = "interrupted"
            status["ended_at"] = record["ended_at"]
            write_json(status_path, status)
            raise
        except Exception as error:
            record["status"] = "failed"
            record["ended_at"] = _batch_time()
            record["error"] = f"{type(error).__name__}: {error}"
            failures += 1
            write_json(status_path, status)
            continue
        record["status"] = "ok"
        record["run_dir"] = str(run_dir)
        record["ended_at"] = _batch_time()
        last_run_dir = run_dir
        write_json(status_path, status)
        print(f"seed={seed} run_dir={run_dir}", flush=True)
    status["status"] = "failed" if failures else "ok"
    status["ended_at"] = _batch_time()
    write_json(status_path, status)
    print(f"seed_batch_status={status_path}", flush=True)
    if failures:
        raise RuntimeError(f"{failures}个seed训练失败；详见{status_path}")
    assert last_run_dir is not None
    return last_run_dir
