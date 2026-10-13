#新增
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


def state_one_hot(state: torch.Tensor, count: int) -> torch.Tensor:
    return nn.functional.one_hot(state.long(), num_classes=count).float()


def temporal_visual_features(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([previous, current, current - previous], dim=-1)


def temporal_proprio_features(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([previous, current, current - previous], dim=-1)


class ProprioSemanticNet(nn.Module):
    """Recognize aligned/grasped predicates from RGB latent and robot state."""

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        predicate_count: int = 2,
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.predicate_count = predicate_count
        self.network = _mlp(
            3 * latent_dim + 3 * proprio_dim,
            hidden_dim,
            predicate_count,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        previous_proprio: torch.Tensor,
        current_proprio: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            torch.cat(
                [
                    temporal_visual_features(previous, current),
                    temporal_proprio_features(
                        previous_proprio, current_proprio
                    ),
                ],
                dim=-1,
            )
        )


class ProprioTransitionNet(nn.Module):
    """Propose the next control mode; the runtime FSM enforces legal edges."""

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        predicate_count: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.network = _mlp(
            3 * latent_dim
            + 3 * proprio_dim
            + predicate_count
            + state_count,
            hidden_dim,
            state_count,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        previous_proprio: torch.Tensor,
        current_proprio: torch.Tensor,
        predicate_probabilities: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                temporal_visual_features(previous, current),
                temporal_proprio_features(
                    previous_proprio, current_proprio
                ),
                predicate_probabilities,
                state_one_hot(current_state, self.state_count),
            ],
            dim=-1,
        )
        return self.network(features)


def control_features(
    previous: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    state: torch.Tensor,
    state_count: int,
    proprio: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            previous,
            current,
            target,
            goal,
            current - target,
            current - goal,
            state_one_hot(state, state_count),
            proprio,
        ],
        dim=-1,
    )


class ProprioTargetNet(nn.Module):
    """Predict ALIGN/GRASP keyframe latents; TRANSFER bypasses this model."""

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.network = _mlp(
            4 * latent_dim + proprio_dim + state_count,
            hidden_dim,
            latent_dim,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                previous,
                current,
                goal,
                current - goal,
                state_one_hot(state, self.state_count),
                proprio,
            ],
            dim=-1,
        )
        return self.network(features)


class ProprioBlockActor(nn.Module):
    """Goal-conditioned behavior-cloned actor producing one action block."""

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        action_block: int,
        hidden_dim: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_block = action_block
        self.state_count = state_count
        self.proprio_dim = proprio_dim
        self.network = _mlp(
            6 * latent_dim + state_count + proprio_dim,
            hidden_dim,
            action_dim * action_block,
        )

    def mean(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            self.network(
                control_features(
                    previous,
                    current,
                    target,
                    goal,
                    state,
                    self.state_count,
                    proprio,
                )
            )
        )


class ProprioValueNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        state_count: int,
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.network = _mlp(
            6 * latent_dim + state_count + proprio_dim,
            hidden_dim,
            1,
        )

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            control_features(
                previous,
                current,
                target,
                goal,
                state,
                self.state_count,
                proprio,
            )
        ).squeeze(-1)


class ProprioValueEnsemble(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        state_count: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [
                ProprioValueNet(
                    latent_dim, proprio_dim, hidden_dim, state_count
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(*args) for member in self.members], dim=0)
