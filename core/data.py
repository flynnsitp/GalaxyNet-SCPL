# Validate galaxy datasets and create deterministic labeled splits.

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import sha256_file, sha256_strings


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
STRUCTURAL_KEYS_14 = (
    "C",
    "A",
    "S",
    "G",
    "M20",
    "ellipticity",
    "sersic_n",
    "half_light_radius",
    "B/T",
    "shape_asymmetry",
    "MID_M",
    "MID_I",
    "MID_D",
    "color_gradient",
)
STRUCTURAL_KEYS_12 = tuple(
    key
    for key in STRUCTURAL_KEYS_14
    if key not in {"half_light_radius", "color_gradient"}
)
STRUCTURAL_KEYS_BY_COUNT = {
    len(STRUCTURAL_KEYS_12): STRUCTURAL_KEYS_12,
    len(STRUCTURAL_KEYS_14): STRUCTURAL_KEYS_14,
}
GZD5_CLASSES = (
    "barred_spiral",
    "cigar_shaped_smooth",
    "disturbed",
    "edge_on_no_bulge",
    "edge_on_with_bulge",
    "in_between_round_smooth",
    "merging",
    "round_smooth",
    "unbarred_loose_spiral",
    "unbarred_tight_spiral",
)


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SampleRecord:
    path: Path
    relative_path: str
    sample_id: str
    class_name: str
    label: int
    structure: tuple[float, ...]


@dataclass(frozen=True)
class DatasetAudit:
    root: Path
    manifest_path: Path
    structural_path: Path
    classes: tuple[str, ...]
    raw_counts: dict[str, int]
    valid_counts: dict[str, int]
    per_class_valid_counts: dict[str, dict[str, int]]
    records: dict[str, tuple[SampleRecord, ...]]
    dropped_invalid: dict[str, dict[str, str]]
    manifest_sha256: str
    structural_sha256: str
    valid_path_sha256: dict[str, str]
    structural_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "structural_path": str(self.structural_path),
            "classes": list(self.classes),
            "structural_feature_count": len(self.structural_keys),
            "structural_keys": list(self.structural_keys),
            "raw_counts": self.raw_counts,
            "valid_counts": self.valid_counts,
            "per_class_valid_counts": self.per_class_valid_counts,
            "dropped_invalid": self.dropped_invalid,
            "manifest_sha256": self.manifest_sha256,
            "structural_sha256": self.structural_sha256,
            "valid_path_sha256": self.valid_path_sha256,
            "val_test_structures_used_for_classification": False,
        }


@dataclass(frozen=True)
class FewShotSplit:
    fraction: float
    seed: int
    labeled_indices: tuple[int, ...]
    unlabeled_indices: tuple[int, ...]
    per_class_counts: dict[str, int]
    labeled_ids: tuple[str, ...]
    fingerprint_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["labeled_indices"] = list(self.labeled_indices)
        payload["unlabeled_indices"] = list(self.unlabeled_indices)
        payload["labeled_ids"] = list(self.labeled_ids)
        return payload


def _relative_image_path(value: str | Path) -> str | None:
    parts = Path(value).parts
    for index, part in enumerate(parts):
        if part in SPLITS and len(parts) - index >= 3:
            return Path(*parts[index:]).as_posix()
    return None


def resolve_structural_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    dataset = config["dataset"] if "dataset" in config else config
    value = dataset["structural_feature_count"]
    try:
        count = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("dataset.structural_feature_count仅支持12或14") from error
    if isinstance(value, bool) or not math.isfinite(count) or count != int(count):
        raise ValueError("dataset.structural_feature_count仅支持12或14")
    structural_keys = STRUCTURAL_KEYS_BY_COUNT.get(int(count))
    if structural_keys is None:
        raise ValueError("dataset.structural_feature_count仅支持12或14")
    return structural_keys


def _load_structures(
    path: Path, structural_keys: Sequence[str]
) -> dict[str, tuple[tuple[float, ...] | None, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DatasetValidationError("structural.json 顶层必须是 list")
    result: dict[str, tuple[tuple[float, ...] | None, str]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise DatasetValidationError(f"structural.json 第 {index} 行缺少 path")
        relative = _relative_image_path(row["path"])
        if relative is None:
            raise DatasetValidationError(f"无法解析结构量路径: {row['path']}")
        if relative in result:
            raise DatasetValidationError(f"重复结构量路径: {relative}")
        reason = str(row.get("error_message") or "结构量无效")
        features = row.get("features")
        values: tuple[float, ...] | None = None
        if str(row.get("status", "")).lower() == "ok" and isinstance(features, Mapping):
            raw_values = [features.get(key) for key in structural_keys]
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in raw_values
            ):
                values = tuple(float(value) for value in raw_values)
            else:
                missing = [
                    key
                    for key, value in zip(structural_keys, raw_values, strict=True)
                    if not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ]
                reason = f"缺失或非有限结构量: {', '.join(missing)}"
        result[relative] = (values, reason)
    return result


def _manifest_paths(
    manifest: Mapping[str, Any], split: str, classes: Sequence[str]
) -> list[str]:
    per_class = manifest.get("per_class")
    if not isinstance(per_class, Mapping):
        raise DatasetValidationError("manifest 缺少 per_class")
    paths: list[str] = []
    for class_name in classes:
        row = per_class.get(class_name)
        if not isinstance(row, Mapping) or not isinstance(row.get(split), list):
            raise DatasetValidationError(f"manifest 缺少 {class_name}.{split}")
        class_paths = [Path(value).as_posix().lstrip("./") for value in row[split]]
        if int(row.get(f"{split}_count", -1)) != len(class_paths):
            raise DatasetValidationError(f"manifest 数量不匹配: {class_name}.{split}")
        paths.extend(class_paths)
    if len(paths) != len(set(paths)):
        raise DatasetValidationError(f"manifest 的 {split} 含重复路径")
    return paths


def validate_dataset(config: Mapping[str, Any]) -> DatasetAudit:
    dataset = config["dataset"] if "dataset" in config else config
    root = Path(dataset["root"]).expanduser().resolve()
    manifest_path = Path(dataset["split_manifest"]).expanduser().resolve()
    structural_path = Path(dataset["structural_json"]).expanduser().resolve()
    if (
        not root.is_dir()
        or not manifest_path.is_file()
        or not structural_path.is_file()
    ):
        raise FileNotFoundError("数据根目录、manifest 或 structural.json 不存在")

    classes = tuple(str(value) for value in dataset["classes"])
    if classes != GZD5_CLASSES:
        raise DatasetValidationError("类别顺序与冻结的 GZD5-10 协议不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("classes") != list(classes):
        raise DatasetValidationError("manifest 类别顺序不一致")
    structural_keys = resolve_structural_keys(dataset)
    structures = _load_structures(structural_path, structural_keys)

    raw_counts: dict[str, int] = {}
    valid_counts: dict[str, int] = {}
    per_class_counts: dict[str, dict[str, int]] = {}
    records: dict[str, tuple[SampleRecord, ...]] = {}
    dropped: dict[str, dict[str, str]] = {}
    path_fingerprints: dict[str, str] = {}
    seen_ids: set[str] = set()

    for split in SPLITS:
        relative_paths = _manifest_paths(manifest, split, classes)
        disk_paths = {
            path.relative_to(root).as_posix()
            for class_name in classes
            for path in (root / split / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if disk_paths != set(relative_paths):
            raise DatasetValidationError(
                f"{split} 磁盘文件与 manifest 不一致: "
                f"missing={sorted(set(relative_paths)-disk_paths)[:5]}, "
                f"extra={sorted(disk_paths-set(relative_paths))[:5]}"
            )
        raw_counts[split] = len(relative_paths)
        accepted: list[SampleRecord] = []
        invalid: dict[str, str] = {}
        class_counts = {name: 0 for name in classes}
        for relative in relative_paths:
            image_path = (root / relative).resolve()
            class_name = Path(relative).parent.name
            sample_id = image_path.stem
            if sample_id in seen_ids:
                raise DatasetValidationError(f"对象 ID 跨划分重复: {sample_id}")
            seen_ids.add(sample_id)
            structure_entry = structures.get(relative)
            if structure_entry is None:
                raise DatasetValidationError(f"缺少结构量记录: {relative}")
            structure, reason = structure_entry
            if structure is None:
                invalid[relative] = reason
                continue
            accepted.append(
                SampleRecord(
                    path=image_path,
                    relative_path=relative,
                    sample_id=sample_id,
                    class_name=class_name,
                    label=classes.index(class_name),
                    structure=structure,
                )
            )
            class_counts[class_name] += 1
        records[split] = tuple(accepted)
        dropped[split] = invalid
        valid_counts[split] = len(accepted)
        per_class_counts[split] = class_counts
        path_fingerprints[split] = sha256_strings(
            tuple(record.relative_path for record in accepted)
        )

    expected_raw = {
        key: int(value) for key, value in dataset["expected_raw_counts"].items()
    }
    expected_valid = {
        key: int(value) for key, value in dataset["expected_valid_counts"].items()
    }
    if raw_counts != expected_raw:
        raise DatasetValidationError(f"原始数量不匹配: {raw_counts} != {expected_raw}")
    if valid_counts != expected_valid:
        raise DatasetValidationError(
            f"有效数量不匹配: {valid_counts} != {expected_valid}"
        )

    return DatasetAudit(
        root=root,
        manifest_path=manifest_path,
        structural_path=structural_path,
        classes=classes,
        raw_counts=raw_counts,
        valid_counts=valid_counts,
        per_class_valid_counts=per_class_counts,
        records=records,
        dropped_invalid=dropped,
        manifest_sha256=sha256_file(manifest_path),
        structural_sha256=sha256_file(structural_path),
        valid_path_sha256=path_fingerprints,
        structural_keys=structural_keys,
    )


def build_fewshot_split(
    records: Sequence[SampleRecord], classes: Sequence[str], fraction: float, seed: int
) -> FewShotSplit:
    if isinstance(fraction, bool):
        raise ValueError("few-shot fraction 必须是 (0,1] 内的数值")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("few-shot fraction 必须位于区间 (0,1]")
    by_class: dict[str, list[int]] = {name: [] for name in classes}
    for index, record in enumerate(records):
        by_class[record.class_name].append(index)

    labeled: list[int] = []
    per_class_counts: dict[str, int] = {}
    for class_name in classes:
        ranked = sorted(
            by_class[class_name],
            key=lambda index: hashlib.sha256(
                f"{seed}\0{class_name}\0{records[index].sample_id}".encode("utf-8")
            ).digest(),
        )
        count = max(1, math.floor(len(ranked) * fraction))
        labeled.extend(ranked[:count])
        per_class_counts[class_name] = count
    labeled = sorted(labeled)
    labeled_set = set(labeled)
    unlabeled = [index for index in range(len(records)) if index not in labeled_set]
    labeled_ids = tuple(records[index].sample_id for index in labeled)
    return FewShotSplit(
        fraction=fraction,
        seed=int(seed),
        labeled_indices=tuple(labeled),
        unlabeled_indices=tuple(unlabeled),
        per_class_counts=per_class_counts,
        labeled_ids=labeled_ids,
        fingerprint_sha256=sha256_strings(labeled_ids),
    )
