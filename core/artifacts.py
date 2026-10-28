# Manage run artifacts, atomic checkpoints, logging, and fingerprints.

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

import torch


def state_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item, key=repr):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for entry in item:
                update(entry)
        else:
            digest.update(f"{type(item).__name__}:{item!r}\0".encode())

    update(value)
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def append_jsonl(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_torch_path(destination: Path) -> tuple[Path, Callable[[], None]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)

    def commit() -> None:
        os.replace(temporary, destination)

    return temporary, commit


def save_torch_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    temporary, commit = atomic_torch_path(destination)
    try:
        torch.save(dict(payload), temporary)
        commit()
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_torch_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PyTorch 产物不存在: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"PyTorch 产物顶层必须是 mapping: {source}")
    return dict(payload)


def create_run_dir(root: str | Path, config_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root).expanduser().resolve() / config_stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir


def setup_logger(path: str | Path, name: str) -> logging.Logger:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(destination, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
