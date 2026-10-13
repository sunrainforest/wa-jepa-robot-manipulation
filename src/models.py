from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def phase_one_hot(phase: torch.Tensor, count: int = 3) -> torch.Tensor:
    return nn.functional.one_hot(phase.long(), num_classes=count).float()


def goal_features(
    previous: torch.Tensor,
    current: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([previous, current, goal, current - goal], dim=-1)


def control_features(
    previous: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            previous,
            current,
            target,
            goal,
            current - target,
            current - goal,
            phase_one_hot(phase),
        ],
        dim=-1,
    )


class VisualPhaseNet(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = _mlp(4 * latent_dim, hidden_dim, hidden_dim)
        self.phase_head = nn.Linear(hidden_dim, 3)
        self.complete_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(goal_features(previous, current, goal))
        return self.phase_head(hidden), self.complete_head(hidden).squeeze(-1)


class PhaseTargetNet(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = _mlp(4 * latent_dim + 3, hidden_dim, latent_dim)

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [goal_features(previous, current, goal), phase_one_hot(phase)],
            dim=-1,
        )
        return self.network(features)


class PhaseBlockActor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_block: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_block = action_block
        self.output_dim = action_dim * action_block
        self.network = _mlp(6 * latent_dim + 3, hidden_dim, self.output_dim)

    def mean(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        features = control_features(previous, current, target, goal, phase)
        return torch.tanh(self.network(features))


class PhaseValueNet(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = _mlp(6 * latent_dim + 3, hidden_dim, 1)

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        features = control_features(previous, current, target, goal, phase)
        return self.network(features).squeeze(-1)


class PhaseValueEnsemble(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [PhaseValueNet(latent_dim, hidden_dim) for _ in range(ensemble_size)]
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                member(previous, current, target, goal, phase)
                for member in self.members
            ],
            dim=0,
        )
