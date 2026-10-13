from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LatentSubgoalPlanner(nn.Module):
    """Predict one fixed-lookahead latent subgoal from the current state."""

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        state_count: int,
        hidden_dim: int = 512,
        phase_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.state_count = state_count
        self.hidden_dim = hidden_dim
        self.phase_embedding_dim = phase_embedding_dim
        self.phase_embedding = nn.Embedding(state_count, phase_embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(
                2 * latent_dim + phase_embedding_dim + proprio_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_std", torch.ones(proprio_dim))

    @torch.no_grad()
    def set_normalization(
        self,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
    ) -> None:
        self.latent_mean.copy_(latent_mean)
        self.latent_std.copy_(latent_std.clamp_min(1e-6))
        self.proprio_mean.copy_(proprio_mean)
        self.proprio_std.copy_(proprio_std.clamp_min(1e-6))

    def forward_normalized(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        current_normalized = (current - self.latent_mean) / self.latent_std
        goal_normalized = (goal - self.latent_mean) / self.latent_std
        proprio_normalized = (
            proprio - self.proprio_mean
        ) / self.proprio_std
        features = torch.cat(
            [
                current_normalized,
                goal_normalized,
                self.phase_embedding(phase.long()),
                proprio_normalized,
            ],
            dim=-1,
        )
        return self.network(features)

    def forward(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.forward_normalized(current, goal, phase, proprio)
        return prediction * self.latent_std + self.latent_mean

    def model_config(self) -> dict[str, int]:
        return {
            "latent_dim": self.latent_dim,
            "proprio_dim": self.proprio_dim,
            "state_count": self.state_count,
            "hidden_dim": self.hidden_dim,
            "phase_embedding_dim": self.phase_embedding_dim,
        }


def load_latent_subgoal_planner(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[LatentSubgoalPlanner, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LatentSubgoalPlanner(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
