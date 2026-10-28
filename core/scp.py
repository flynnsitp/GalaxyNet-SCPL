# Construct and schedule structural-consistency pseudo-label states.

from __future__ import annotations
import math
from dataclasses import dataclass, fields, replace
from numbers import Integral
from typing import Any, Mapping, Sequence
import torch
from torch import Tensor

_KNN_CHUNK_SIZE = 1024
SMALL_LABEL_FRACTION_PERCENT = 3

PROTOTYPE_VIEW_ANGLES = {
    "labeled_c4": (0, 90, 180, 270),
}


def _as_index_tensor(name: str, values: Sequence[int] | Tensor) -> Tensor:
    result = torch.as_tensor(values)
    if result.ndim != 1:
        raise ValueError(f"{name} 必须是一维整数序列")
    if result.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
        raise ValueError(f"{name} 必须是一维整数序列")
    return result.detach().cpu().long()


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return int(value)


def _finite_positive(name: str, value: float) -> float:
    parsed = float(value)
    if isinstance(value, bool) or not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} 必须为有限正数")
    return parsed


def _probability_threshold(name: str, value: float) -> float:
    parsed = float(value)
    if isinstance(value, bool) or not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} 必须位于 [0,1]")
    return parsed


def _validate_posteriors(name: str, values: Tensor) -> None:
    if not values.is_floating_point() or values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{name} 必须是至少两类的 [N,C]")
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError(f"{name} 必须是有限非负数")
    expected = torch.ones(values.shape[0], device=values.device, dtype=values.dtype)
    if not torch.allclose(values.sum(dim=1), expected, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError(f"{name} 每行必须归一化为 1")


def calibrated_class_posterior(logits: Tensor, *, temperature: float) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("visual logits必须是至少两类的[N,C]")
    if not torch.isfinite(logits).all():
        raise ValueError("visual logits必须为有限数")
    temperature = _finite_positive("visual temperature", temperature)
    return torch.softmax(logits.float() / temperature, dim=1)


def structural_knn_posterior(
    query_structures: Tensor,
    labeled_structures: Tensor,
    labeled_labels: Tensor,
    *,
    class_count: int,
    neighbors: int,
) -> Tensor:
    queries = query_structures.detach().cpu().float()
    references = labeled_structures.detach().cpu().float()
    labels = _as_index_tensor("labeled_labels", labeled_labels)
    if (
        isinstance(class_count, bool)
        or not isinstance(class_count, Integral)
        or int(class_count) < 2
    ):
        raise ValueError("class_count 必须是至少为 2 的整数")
    class_count = int(class_count)
    neighbors = _positive_integer("neighbors", neighbors)
    if (
        queries.ndim != 2
        or references.ndim != 2
        or queries.shape[1] == 0
        or queries.shape[1] != references.shape[1]
    ):
        raise ValueError("query/labeled structures 必须是相同特征维度的 [N,S]")
    if not torch.isfinite(queries).all() or not torch.isfinite(references).all():
        raise ValueError("query/labeled structures 必须为有限数")
    if len(references) != len(labels) or len(references) == 0:
        raise ValueError("labeled structures/labels 必须非空且一一对应")
    if bool((labels < 0).any()) or bool((labels >= class_count).any()):
        raise ValueError("labeled label 越界")
    available = len(references)
    if neighbors > available:
        raise ValueError("neighbors 超出每个 query 可用的 labeled reference 数量")

    class_counts = torch.bincount(labels, minlength=class_count).float()
    if bool((class_counts == 0).any()):
        raise ValueError("每个类别必须至少有一个 labeled reference")
    if len(queries) == 0:
        return torch.empty((0, class_count), dtype=torch.float32)

    chunks: list[Tensor] = []
    epsilon = torch.finfo(torch.float32).eps
    for start in range(0, len(queries), _KNN_CHUNK_SIZE):
        stop = start + _KNN_CHUNK_SIZE
        distances = torch.cdist(queries[start:stop], references)

        neighbor_positions = torch.argsort(distances, dim=1, stable=True)[:, :neighbors]
        neighbor_distances = distances.gather(1, neighbor_positions)
        if not torch.isfinite(neighbor_distances).all():
            raise ValueError("排除自身后没有足够的 labeled reference")
        local_scale = neighbor_distances[:, -1:].clamp_min(epsilon)
        weights = torch.exp(-neighbor_distances / local_scale)
        neighbor_labels = labels[neighbor_positions]
        scores = torch.zeros((len(distances), class_count), dtype=torch.float32)
        scores.scatter_add_(1, neighbor_labels, weights)
        prior_counts = class_counts[None, :]
        scores = scores / prior_counts.clamp_min(1.0)
        denominator = scores.sum(dim=1, keepdim=True)
        if bool((denominator <= 0).any()):
            raise RuntimeError("局部结构 posterior 无法归一化")
        chunks.append(scores / denominator)
    return torch.cat(chunks, dim=0)


@dataclass(frozen=True)
class SCPCandidateSelection:
    candidate_indices: Tensor
    candidate_pseudo_labels: Tensor
    candidate_reliability: Tensor
    candidate_deterministic_visual_confidence: Tensor
    candidate_weak_visual_confidence: Tensor
    candidate_structural_confidence: Tensor
    candidate_joint_confidence: Tensor
    candidate_joint_posterior: Tensor
    candidate_joint_margin: Tensor
    candidate_view_total_variation: Tensor
    pool_count: int
    rejection_counts: Mapping[str, int]


def select_scp_candidates(
    deterministic_visual_posterior: Tensor,
    weak_visual_posterior: Tensor,
    structural_posterior: Tensor,
    candidate_indices: Sequence[int] | Tensor,
    *,
    visual_confidence_threshold: float = 0.0,
    structural_confidence_threshold: float = 0.0,
    joint_margin_threshold: float = 0.0,
    weak_view_max_total_variation: float = 1.0,
) -> SCPCandidateSelection:
    for name, posterior in (
        ("deterministic visual posterior", deterministic_visual_posterior),
        ("weak visual posterior", weak_visual_posterior),
    ):
        _validate_posteriors(name, posterior)
    if deterministic_visual_posterior.shape != weak_visual_posterior.shape:
        raise ValueError("两个视觉 posterior 形状必须相同")
    _validate_posteriors("structural posterior", structural_posterior)
    if structural_posterior.shape != deterministic_visual_posterior.shape:
        raise ValueError("两个视觉 posterior 与结构 posterior 形状必须相同")
    indices = _as_index_tensor("candidate_indices", candidate_indices)
    if len(indices) != len(deterministic_visual_posterior):
        raise ValueError("candidate_indices 数量必须与 posterior 行数一致")
    visual_confidence_threshold = _probability_threshold(
        "visual_confidence_threshold", visual_confidence_threshold
    )
    structural_confidence_threshold = _probability_threshold(
        "structural_confidence_threshold", structural_confidence_threshold
    )
    joint_margin_threshold = _probability_threshold(
        "joint_margin_threshold", joint_margin_threshold
    )
    weak_view_max_total_variation = _probability_threshold(
        "weak_view_max_total_variation", weak_view_max_total_variation
    )

    first = deterministic_visual_posterior.detach().cpu().float()
    second = weak_visual_posterior.detach().cpu().float()
    if len(torch.unique(indices)) != len(indices):
        raise ValueError("candidate_indices 不能重复")
    first_confidence, first_top1 = first.max(dim=1)
    second_confidence, second_top1 = second.max(dim=1)
    visual = 0.5 * (first + second)
    structural = structural_posterior.detach().cpu().float()
    structural_confidence, structural_top1 = structural.max(dim=1)
    structural_confident = structural_confidence >= structural_confidence_threshold
    visual_structural_agree = first_top1 == structural_top1
    joint = torch.sqrt((visual * structural).clamp_min(0.0))
    joint = joint / joint.sum(dim=1, keepdim=True).clamp_min(torch.finfo(joint.dtype).eps)
    joint_order = torch.argsort(joint, dim=1, descending=True, stable=True)
    joint_labels = joint_order[:, :2]
    joint_values = joint.gather(1, joint_labels)
    joint_margin = joint_values[:, 0] - joint_values[:, 1]
    view_total_variation = 0.5 * (first - second).abs().sum(dim=1)

    view_labels_agree = first_top1 == second_top1
    visual_confident = (first_confidence >= visual_confidence_threshold) & (
        second_confidence >= visual_confidence_threshold
    )
    views_stable = view_total_variation <= weak_view_max_total_variation
    joint_agrees = first_top1 == joint_labels[:, 0]
    margin_sufficient = joint_margin >= joint_margin_threshold
    accepted = (
        view_labels_agree
        & visual_confident
        & structural_confident
        & views_stable
        & visual_structural_agree
        & joint_agrees
        & margin_sufficient
    )
    labels = joint_labels[:, 0]
    row = torch.arange(len(labels), device=first.device)

    reliability = torch.minimum(first[row, labels], second[row, labels])
    reliability = torch.minimum(reliability, structural[row, labels])
    structural_confidence = structural[row, labels]
    rejection_counts = {
        "weak_view_label_disagreement": int((~view_labels_agree).sum()),
        "low_visual_confidence": int((~visual_confident).sum()),
        "low_structural_confidence": int((~structural_confident).sum()),
        "weak_view_instability": int((~views_stable).sum()),
        "visual_structural_disagreement": int((~visual_structural_agree).sum()),
        "joint_label_disagreement": int((~joint_agrees).sum()),
        "low_joint_margin": int((~margin_sufficient).sum()),
    }
    return SCPCandidateSelection(
        candidate_indices=indices[accepted].cpu(),
        candidate_pseudo_labels=labels[accepted].long().cpu(),
        candidate_reliability=reliability[accepted].float().cpu(),
        candidate_deterministic_visual_confidence=first[row, labels][accepted].float().cpu(),
        candidate_weak_visual_confidence=second[row, labels][accepted].float().cpu(),
        candidate_structural_confidence=structural_confidence[accepted].float().cpu(),
        candidate_joint_confidence=joint_values[accepted, 0].float().cpu(),
        candidate_joint_posterior=joint[accepted].float().cpu(),
        candidate_joint_margin=joint_margin[accepted].float().cpu(),
        candidate_view_total_variation=view_total_variation[accepted].float().cpu(),
        pool_count=len(indices),
        rejection_counts=rejection_counts,
    )


@dataclass(frozen=True)
class SCPState(SCPCandidateSelection):
    class_count: int
    selected_indices: Tensor
    selected_pseudo_labels: Tensor
    selected_reliability: Tensor
    selected_joint_posterior: Tensor
    selected_ids: tuple[tuple[str, ...], ...]

    def validate_alignment(self) -> None:
        if self.class_count < 2 or len(self.selected_ids) != self.class_count:
            raise ValueError("SCP类别数或selected_ids不匹配")
        for prefix in ("candidate", "selected"):
            indices = getattr(self, prefix + "_indices")
            labels = getattr(self, prefix + "_pseudo_labels")
            if (
                indices.ndim != 1
                or indices.dtype != torch.long
                or len(indices.unique()) != len(indices)
                or bool((indices < 0).any())
            ):
                raise ValueError("SCP indices必须是唯一非负整数向量")
            if labels.dtype != torch.long or bool(
                ((labels < 0) | (labels >= self.class_count)).any()
            ):
                raise ValueError("SCP伪标签越界")
            for field in fields(self):
                if field.name.startswith(prefix + "_") and field.name != "selected_ids":
                    value = getattr(self, field.name)
                    if field.name.endswith("_joint_posterior"):
                        continue
                    if (
                        value.ndim != 1
                        or len(value) != len(indices)
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("SCP并行数组未对齐或包含非有限数")
                    if value.is_floating_point() and bool(((value < 0) | (value > 1)).any()):
                        raise ValueError("SCP概率或可靠度越界")
            posterior = getattr(self, prefix + "_joint_posterior")
            if tuple(posterior.shape) != (len(indices), self.class_count):
                raise ValueError("SCP联合posterior形状必须为[N,C]")
            _validate_posteriors("SCP联合posterior", posterior)
            if len(labels) and not torch.equal(posterior.argmax(dim=1), labels):
                raise ValueError("SCP联合posterior的top1必须与伪标签一致")
        if not 0 <= len(self.candidate_indices) <= self.pool_count:
            raise ValueError("SCP候选数量超过pool")
        positions = {int(v): i for i, v in enumerate(self.candidate_indices)}
        for i, index in enumerate(self.selected_indices.tolist()):
            if index not in positions:
                raise ValueError("selected必须来自candidate")
            j = positions[index]
            if (
                self.selected_pseudo_labels[i] != self.candidate_pseudo_labels[j]
                or self.selected_reliability[i] != self.candidate_reliability[j]
                or not torch.equal(
                    self.selected_joint_posterior[i],
                    self.candidate_joint_posterior[j],
                )
            ):
                raise ValueError("selected标签或权重与candidate不一致")
        for label, ids in enumerate(self.selected_ids):
            if len(ids) != int((self.selected_pseudo_labels == label).sum()):
                raise ValueError("selected_ids与类别数量不一致")

    def fingerprint(self) -> str:
        from .artifacts import state_digest

        return state_digest(self.to_checkpoint())

    def diagnostics(self, class_names: Sequence[str]) -> dict[str, Any]:
        if len(class_names) != self.class_count:
            raise ValueError("SCP类别顺序长度不匹配")
        return {
            "pool_count": self.pool_count,
            "candidate_count": len(self.candidate_indices),
            "selected_count": len(self.selected_indices),
            "state_fingerprint": self.fingerprint(),
            "rejection_counts": dict(self.rejection_counts),
            "rejection_count_semantics": "nonexclusive_failed_gate_counts",
            "classes": {
                name: {
                    "candidate_count": int((self.candidate_pseudo_labels == i).sum()),
                    "selected_count": int((self.selected_pseudo_labels == i).sum()),
                    "selected_ids": list(self.selected_ids[i]),
                }
                for i, name in enumerate(class_names)
            },
        }

    def to_checkpoint(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> SCPState:
        state = cls(**payload)
        state.validate_alignment()
        return state


def cap_scp_state_by_max_pseudo_labels(
    state: SCPState,
    sample_ids: Sequence[str],
    *,
    max_pseudo_labels: int,
) -> SCPState:
    state.validate_alignment()
    total_count = len(sample_ids)
    if total_count <= 0:
        raise ValueError("sample_ids不能为空")
    if len(state.selected_indices) and int(state.selected_indices.max()) >= total_count:
        raise ValueError("sample_ids未覆盖SCP selected索引")
    selected_limit = _positive_integer("max_pseudo_labels", max_pseudo_labels)
    if len(state.selected_indices) <= selected_limit:
        return state

    ranked = sorted(
        range(len(state.selected_indices)),
        key=lambda position: (
            -float(state.selected_reliability[position]),
            sample_ids[int(state.selected_indices[position])],
        ),
    )[:selected_limit]
    positions = torch.tensor(ranked, dtype=torch.long)
    selected_ids = tuple(
        tuple(
            sample_ids[int(state.selected_indices[position])]
            for position in ranked
            if int(state.selected_pseudo_labels[position]) == label
        )
        for label in range(state.class_count)
    )
    capped = replace(
        state,
        selected_indices=state.selected_indices[positions],
        selected_pseudo_labels=state.selected_pseudo_labels[positions],
        selected_reliability=state.selected_reliability[positions],
        selected_joint_posterior=state.selected_joint_posterior[positions],
        selected_ids=selected_ids,
    )
    capped.validate_alignment()
    return capped


def scheduled_scp_state(
    state: SCPState,
    sample_ids: Sequence[str],
    *,
    max_selected_per_class: int,
    ramp_epochs: int,
    epoch: int,
) -> SCPState:
    state.validate_alignment()
    q = _positive_integer("max_selected_per_class", max_selected_per_class)
    ramp = _positive_integer("selection_ramp_epochs", ramp_epochs)
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
        raise ValueError("epoch必须是非负整数")
    if len(state.candidate_indices) and int(state.candidate_indices.max()) >= len(sample_ids):
        raise ValueError("sample_ids未覆盖SCP候选索引")
    selected_positions: list[int] = []
    selected_ids: list[tuple[str, ...]] = []
    for label in range(state.class_count):
        pool = torch.nonzero(state.candidate_pseudo_labels == label, as_tuple=False).flatten()
        ranked = sorted(
            pool.tolist(),
            key=lambda position: (
                -float(state.candidate_reliability[position]),
                sample_ids[int(state.candidate_indices[position])],
            ),
        )
        active_q = math.ceil(q * min(epoch + 1, ramp) / ramp)
        count = min(active_q, len(ranked))
        chosen = ranked[:count]
        selected_positions.extend(chosen)
        selected_ids.append(
            tuple(sample_ids[int(state.candidate_indices[position])] for position in chosen)
        )
    positions = torch.tensor(selected_positions, dtype=torch.long)
    scheduled = replace(
        state,
        selected_indices=state.candidate_indices[positions],
        selected_pseudo_labels=state.candidate_pseudo_labels[positions],
        selected_reliability=state.candidate_reliability[positions],
        selected_joint_posterior=state.candidate_joint_posterior[positions],
        selected_ids=tuple(selected_ids),
    )
    scheduled.validate_alignment()
    return scheduled


def build_scp_state(
    standardized_structures: Tensor,
    deterministic_logits: Tensor,
    weak_logits: Tensor,
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    labeled_labels: Tensor,
    class_names: Sequence[str],
    sample_ids: Sequence[str],
    *,
    structural_knn: int,
    max_selected_per_class: int,
    visual_temperature: float,
    visual_confidence_threshold: float,
    structural_confidence_threshold: float,
    joint_margin_threshold: float,
    weak_view_max_total_variation: float,
) -> SCPState:
    labeled = _as_index_tensor("labeled_indices", labeled_indices)
    unlabeled = _as_index_tensor("unlabeled_indices", unlabeled_indices)
    labels = _as_index_tensor("labeled_labels", labeled_labels)
    n, c = len(sample_ids), len(class_names)
    if len(labeled) == 0 or len(unlabeled) == 0:
        raise ValueError("SCP需要有标签参考和非空unlabeled pool")
    all_indices = torch.cat((labeled, unlabeled))
    if len(all_indices) != n or not torch.equal(all_indices.sort().values, torch.arange(n)):
        raise ValueError("labeled/unlabeled必须互斥且完整覆盖train")
    if len(set(sample_ids)) != n or len(labels) != len(labeled):
        raise ValueError("样本ID重复或有标签索引未对齐")
    if standardized_structures.ndim != 2 or len(standardized_structures) != n:
        raise ValueError("结构数组与样本不匹配")
    for logits in (deterministic_logits, weak_logits):
        if tuple(logits.shape) != (n, c) or not torch.isfinite(logits).all():
            raise ValueError("视觉logits必须是有限的[N,C]")
    q = _positive_integer("max_selected_per_class", max_selected_per_class)
    first = calibrated_class_posterior(
        deterministic_logits[unlabeled], temperature=visual_temperature
    )
    second = calibrated_class_posterior(weak_logits[unlabeled], temperature=visual_temperature)
    structural = structural_knn_posterior(
        standardized_structures[unlabeled],
        standardized_structures[labeled],
        labels,
        class_count=c,
        neighbors=structural_knn,
    )
    candidates = select_scp_candidates(
        first,
        second,
        structural,
        unlabeled,
        visual_confidence_threshold=visual_confidence_threshold,
        structural_confidence_threshold=structural_confidence_threshold,
        joint_margin_threshold=joint_margin_threshold,
        weak_view_max_total_variation=weak_view_max_total_variation,
    )
    positions, selected_ids = [], []
    for label in range(c):
        pool = torch.nonzero(candidates.candidate_pseudo_labels == label, as_tuple=False).flatten()
        ranked = sorted(
            pool.tolist(),
            key=lambda i: (
                -float(candidates.candidate_reliability[i]),
                sample_ids[int(candidates.candidate_indices[i])],
            ),
        )[:q]
        positions.extend(ranked)
        selected_ids.append(tuple(sample_ids[int(candidates.candidate_indices[i])] for i in ranked))
    positions = torch.tensor(positions, dtype=torch.long)
    state = SCPState(
        **vars(candidates),
        class_count=c,
        selected_indices=candidates.candidate_indices[positions],
        selected_pseudo_labels=candidates.candidate_pseudo_labels[positions],
        selected_reliability=candidates.candidate_reliability[positions],
        selected_joint_posterior=candidates.candidate_joint_posterior[positions],
        selected_ids=tuple(selected_ids),
    )
    state.validate_alignment()
    return state
