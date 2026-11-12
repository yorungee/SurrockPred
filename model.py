"""Physics-informed dual-branch model for sample-level MWD features.

The physical MLP uses all 32 F/T descriptors: eight original F features, eight
derivative-F features, eight original T features and eight derivative-T
features. The MWD branch applies a Transformer followed by a KNN sample graph
and predicts a residual. 
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = (
    "derivative_Curve_F",
    "derivative_Curve_T",
    "param_Curve_F",
    "param_Curve_T",
)
KEY_POINTS = ("XF1", "XF2", "XF3", "XF51", "XF52", "XF6", "XF7", "XF8")
FT_FEATURES = tuple(f"{family}_{key}" for family in FAMILIES for key in KEY_POINTS)
MWD_FEATURES = tuple(f"MWD_{index:02d}" for index in range(1, 33))
PHYSICS_FEATURES = FT_FEATURES


@dataclass
class Config:
    d_model: int = 64
    encoder_heads: int = 4
    encoder_layers: int = 2
    graph_heads: int = 4
    k_neighbors: int = 5
    dropout: float = 0.10
    batch_size: int = 64
    epochs: int = 250
    early_stopping_patience: int = 50
    learning_rate: float = 1e-3
    seed: int = 2026


class KNNGraphAttention(nn.Module):
    """Multi-head self-attention restricted to nearest sample neighbours."""

    def __init__(
        self, d_model: int, heads: int, k_neighbors: int, dropout: float
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.k_neighbors = k_neighbors
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _ = x.shape
        qkv = self.qkv(x).reshape(batch, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3)
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        score = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)

        # Build an undirected KNN graph from sample embeddings in each batch.
        k_count = min(self.k_neighbors + 1, batch)
        distance = torch.cdist(x, x)
        neighbours = torch.topk(-distance, k=k_count, dim=-1).indices
        adjacency = torch.zeros(batch, batch, dtype=torch.bool, device=x.device)
        adjacency.scatter_(1, neighbours, True)
        adjacency = adjacency | adjacency.transpose(0, 1)
        score = score.masked_fill(~adjacency.unsqueeze(0), -1e9)

        weight = self.dropout(F.softmax(score, dim=-1))
        context = (weight @ v).permute(1, 0, 2).reshape(batch, -1)
        return self.norm(x + self.dropout(self.proj(context)))


class DynamicBranch(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tokens = len(FAMILIES)
        self.token_size = len(MWD_FEATURES) // self.tokens
        self.input_proj = nn.ModuleList(
            [nn.Linear(self.token_size, config.d_model) for _ in FAMILIES]
        )
        # Feature-block embedding, not sequence-position encoding.
        self.block_embedding = nn.Parameter(torch.zeros(1, self.tokens, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_heads,
            dim_feedforward=2 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers)
        self.knn = KNNGraphAttention(
            config.d_model,
            config.graph_heads,
            config.k_neighbors,
            config.dropout,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.candidate = nn.Linear(config.d_model, config.graph_heads)
        self.gate = nn.Linear(config.d_model, config.graph_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(len(x), self.tokens, self.token_size)
        hidden = torch.stack(
            [projection(tokens[:, i]) for i, projection in enumerate(self.input_proj)],
            dim=1,
        )
        hidden = self.encoder(hidden + self.block_embedding)
        hidden = self.norm(hidden.mean(dim=1))
        hidden = self.knn(hidden)
        candidates = self.candidate(hidden)
        weights = F.softmax(self.gate(hidden), dim=-1)
        residual = torch.sum(weights * candidates, dim=1, keepdim=True)
        return residual, weights


class PhysicsBranch(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(PHYSICS_FEATURES), 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TunnelStrengthModel(nn.Module):
    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config()
        self.dynamic = DynamicBranch(self.config)
        self.physics = PhysicsBranch(self.config.dropout)
        self.register_buffer("mwd_mean", torch.zeros(len(MWD_FEATURES)))
        self.register_buffer("mwd_std", torch.ones(len(MWD_FEATURES)))
        self.register_buffer("ft_mean", torch.zeros(len(FT_FEATURES)))
        self.register_buffer("ft_std", torch.ones(len(FT_FEATURES)))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def _normalize_mwd(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mwd_mean) / self.mwd_std

    def _normalize_ft(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.ft_mean) / self.ft_std

    def forward(
        self, mwd: torch.Tensor, ft: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        residual, weights = self.dynamic(self._normalize_mwd(mwd))
        physics = self.physics(self._normalize_ft(ft)) * self.y_std + self.y_mean
        return {
            "prediction": physics + residual,
            "physics": physics,
            "residual": residual,
            "weights": weights,
        }

    def fit(
        self,
        mwd: np.ndarray,
        ft: np.ndarray,
        y: np.ndarray,
        mwd_val: np.ndarray | None = None,
        ft_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TunnelStrengthModel":
        torch.manual_seed(self.config.seed)
        mwd = np.asarray(mwd, dtype=float)
        ft = np.asarray(ft, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        mwd_mean = np.nanmean(mwd, axis=0)
        mwd_std = np.nanstd(mwd, axis=0)
        mwd_std[mwd_std < 1e-12] = 1.0
        ft_mean = np.nanmean(ft, axis=0)
        ft_std = np.nanstd(ft, axis=0)
        ft_std[ft_std < 1e-12] = 1.0
        mwd = np.where(np.isfinite(mwd), mwd, mwd_mean)
        ft = np.where(np.isfinite(ft), ft, ft_mean)
        self.mwd_mean.copy_(torch.tensor(mwd_mean, dtype=torch.float32))
        self.mwd_std.copy_(torch.tensor(mwd_std, dtype=torch.float32))
        self.ft_mean.copy_(torch.tensor(ft_mean, dtype=torch.float32))
        self.ft_std.copy_(torch.tensor(ft_std, dtype=torch.float32))
        self.y_mean.fill_(float(y.mean()))
        self.y_std.fill_(float(y.std() or 1.0))

        device = next(self.parameters()).device
        mwd_tensor = torch.tensor(mwd, dtype=torch.float32, device=device)
        ft_tensor = torch.tensor(ft, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
        use_validation = mwd_val is not None and ft_val is not None and y_val is not None
        if use_validation:
            mwd_val_tensor = torch.tensor(
                np.where(np.isfinite(mwd_val), mwd_val, mwd_mean),
                dtype=torch.float32,
                device=device,
            )
            ft_val_tensor = torch.tensor(
                np.where(np.isfinite(ft_val), ft_val, ft_mean),
                dtype=torch.float32,
                device=device,
            )
            y_val_tensor = torch.tensor(
                np.asarray(y_val, dtype=float).reshape(-1),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)

        physics_optimizer = torch.optim.AdamW(
            self.physics.parameters(), lr=self.config.learning_rate
        )
        best_physics_loss = float("inf")
        best_physics_state = None
        physics_wait = 0
        for _ in range(self.config.epochs):
            self.physics.train()
            prediction = (
                self.physics(self._normalize_ft(ft_tensor)) * self.y_std
                + self.y_mean
            )
            loss = F.mse_loss(prediction, y_tensor)
            physics_optimizer.zero_grad()
            loss.backward()
            physics_optimizer.step()
            if use_validation:
                self.physics.eval()
                with torch.no_grad():
                    val_prediction = (
                        self.physics(self._normalize_ft(ft_val_tensor)) * self.y_std
                        + self.y_mean
                    )
                    val_loss = float(F.mse_loss(val_prediction, y_val_tensor))
                if val_loss < best_physics_loss - 1e-6:
                    best_physics_loss = val_loss
                    best_physics_state = copy.deepcopy(self.physics.state_dict())
                    physics_wait = 0
                else:
                    physics_wait += 1
                    if physics_wait >= self.config.early_stopping_patience:
                        break

        if best_physics_state is not None:
            self.physics.load_state_dict(best_physics_state)

        for parameter in self.physics.parameters():
            parameter.requires_grad = False
        self.physics.eval()

        optimizer = torch.optim.AdamW(
            self.dynamic.parameters(), lr=self.config.learning_rate
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        best_dynamic_loss = float("inf")
        best_dynamic_state = None
        dynamic_wait = 0
        for _ in range(self.config.epochs):
            self.dynamic.train()
            order = torch.randperm(len(mwd_tensor), generator=generator, device=device)
            for start in range(0, len(order), self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                loss = F.mse_loss(
                    self.forward(mwd_tensor[index], ft_tensor[index])["prediction"],
                    y_tensor[index],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if use_validation:
                self.dynamic.eval()
                with torch.no_grad():
                    val_prediction = self.forward(
                        mwd_val_tensor, ft_val_tensor
                    )["prediction"]
                    val_loss = float(F.mse_loss(val_prediction, y_val_tensor))
                if val_loss < best_dynamic_loss - 1e-6:
                    best_dynamic_loss = val_loss
                    best_dynamic_state = copy.deepcopy(self.dynamic.state_dict())
                    dynamic_wait = 0
                else:
                    dynamic_wait += 1
                    if dynamic_wait >= self.config.early_stopping_patience:
                        break

        if best_dynamic_state is not None:
            self.dynamic.load_state_dict(best_dynamic_state)
        self.eval()
        return self

    @torch.no_grad()
    def predict(
        self, mwd: np.ndarray, ft: np.ndarray
    ) -> dict[str, np.ndarray]:
        mwd = np.asarray(mwd, dtype=float)
        ft = np.asarray(ft, dtype=float)
        mwd = np.where(np.isfinite(mwd), mwd, self.mwd_mean.cpu().numpy())
        ft = np.where(np.isfinite(ft), ft, self.ft_mean.cpu().numpy())
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            output = self.forward(
                torch.tensor(mwd, dtype=torch.float32, device=device),
                torch.tensor(ft, dtype=torch.float32, device=device),
            )
        finally:
            self.train(was_training)
        return {key: value.cpu().numpy() for key, value in output.items()}
