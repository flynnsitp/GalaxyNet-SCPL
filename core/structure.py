# Normalize structural features using training-set robust statistics.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class RobustScaler:
    median: Tensor
    iqr: Tensor
    epsilon: float = 1.0e-6

    @classmethod
    def fit(cls, structures: Tensor, epsilon: float = 1.0e-6) -> "RobustScaler":
        if structures.ndim != 2 or structures.shape[1] == 0:
            raise ValueError("structures 必须为非空特征维的 [N,S]")
        values = structures.float()
        if not torch.isfinite(values).all():
            raise ValueError("structures 含非有限值")
        quantiles = torch.quantile(
            values,
            torch.tensor([0.25, 0.5, 0.75], device=values.device),
            dim=0,
        )
        return cls(
            median=quantiles[1].cpu(),
            iqr=(quantiles[2] - quantiles[0]).clamp_min(float(epsilon)).cpu(),
            epsilon=float(epsilon),
        )

    def transform(self, structures: Tensor) -> Tensor:
        if structures.ndim == 0 or structures.shape[-1] != self.median.numel():
            raise ValueError(f"structures 最后一维必须为 {self.median.numel()}")
        median = self.median.to(device=structures.device, dtype=structures.dtype)
        iqr = self.iqr.to(device=structures.device, dtype=structures.dtype)
        return (structures - median) / (iqr + self.epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "epsilon": self.epsilon,
        }

    def to_checkpoint(self) -> dict[str, Any]:
        return {"median": self.median, "iqr": self.iqr, "epsilon": self.epsilon}
