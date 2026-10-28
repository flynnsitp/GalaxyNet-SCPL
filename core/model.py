# Build the multiscale morphology adapter and galaxy classifier.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import DINOv3FeatureExtractor


class SwiGLUFFN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(embedding_dim, hidden_dim)
        self.value_projection = nn.Linear(embedding_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, value: Tensor) -> Tensor:
        gated = F.silu(self.gate_projection(value)) * self.value_projection(value)
        return self.output_projection(gated)


class MultiScaleMorphologyAdapter(nn.Module):
    def __init__(
        self,
        stage_dims: Sequence[int],
        *,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        if len(stage_dims) != 4:
            raise ValueError("MMA 需要恰好四个 ConvNeXt stage")
        if attention_heads <= 0:
            raise ValueError("MMA 的 attention_heads 必须大于 0")
        if embedding_dim % attention_heads:
            raise ValueError("MMA 的 embedding_dim 必须能被 attention_heads 整除")

        self.stage_dims = tuple(int(value) for value in stage_dims)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.attention_heads = int(attention_heads)

        self.stage_normalizations = nn.ModuleList(
            [nn.LayerNorm(channels) for channels in self.stage_dims]
        )
        self.stage_projections = nn.ModuleList(
            [
                nn.Conv2d(channels, self.embedding_dim, kernel_size=1)
                for channels in self.stage_dims
            ]
        )
        self.query_normalizations = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in self.stage_dims]
        )
        self.token_normalizations = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in self.stage_dims]
        )
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.embedding_dim,
                    self.attention_heads,
                    batch_first=True,
                )
                for _ in self.stage_dims
            ]
        )
        self.ffn_normalizations = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in self.stage_dims]
        )
        self.ffns = nn.ModuleList(
            [SwiGLUFFN(self.embedding_dim, self.hidden_dim) for _ in self.stage_dims]
        )
        self.queries = nn.Parameter(torch.empty(4, 1, self.embedding_dim))
        self.scale_gate = nn.Linear(4 * self.embedding_dim, 4)
        nn.init.trunc_normal_(self.queries, std=0.02)

        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 4:
            raise ValueError("MMA 输入必须包含四个 stage feature map")

        summaries: list[Tensor] = []
        batch_size: int | None = None
        for index, (feature, channels) in enumerate(
            zip(features, self.stage_dims, strict=True)
        ):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"MMA stage {index + 1} 需要 [B,{channels},H,W]，"
                    f"实际为 {tuple(feature.shape)}"
                )
            if batch_size is None:
                batch_size = int(feature.shape[0])
            elif feature.shape[0] != batch_size:
                raise ValueError("MMA 四个 stage 的 batch size 必须一致")


            normalized = self.stage_normalizations[index](
                feature.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            tokens = (
                self.stage_projections[index](normalized).flatten(2).transpose(1, 2)
            )
            query = self.queries[index].unsqueeze(0).expand(feature.shape[0], -1, -1)
            normalized_tokens = self.token_normalizations[index](tokens)
            attended, _ = self.attentions[index](
                self.query_normalizations[index](query),
                normalized_tokens,
                normalized_tokens,
                need_weights=False,
            )
            summary = query + attended
            summary = summary + self.ffns[index](
                self.ffn_normalizations[index](summary)
            )
            summaries.append(summary.squeeze(1))

        stacked = torch.stack(summaries, dim=1)
        scale_weights = torch.softmax(self.scale_gate(stacked.flatten(1)), dim=-1)
        fused = (scale_weights.unsqueeze(-1) * stacked).sum(dim=1)
        return F.normalize(fused, dim=-1)


class MorphologyClassifier(nn.Module):
    def __init__(
        self,
        backbone: DINOv3FeatureExtractor,
        *,
        adapter_type: str = "mma",
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        attention_heads: int = 8,
        class_count: int = 10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter_type = str(adapter_type)
        if self.adapter_type != "mma":
            raise ValueError("adapter_type仅支持mma")
        self.adapter = MultiScaleMorphologyAdapter(
            backbone.spec.dims,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
        )
        self.classifier = nn.Linear(int(embedding_dim), int(class_count), bias=True)

    def encode(self, images: Tensor) -> Tensor:
        return self.adapter(self.backbone.forward_multiscale(images))

    def classify_embeddings(self, embeddings: Tensor) -> Tensor:
        return self.classifier(embeddings)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.encode(images)
        return embeddings, self.classify_embeddings(embeddings)
