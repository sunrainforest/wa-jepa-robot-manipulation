# 运行脚本
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import datasets
import numpy as np
import ogbench
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as transforms

from models import (
    PhaseBlockActor,
    PhaseTargetNet,
    PhaseValueEnsemble,
    VisualPhaseNet,
)

import wajepa_world_model as wajepa


HERE = Path(__file__).resolve().parent
CUBE_ROOT = HERE.parents[1]
RGB_EXPERIMENT = HERE.parent / "rgb_latent_value_mpc"
for path in (CUBE_ROOT, RGB_EXPERIMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from privileged_teacher import PrivilegedTeacher, detect_phase, semantic_state


PHASE_NAMES = ("grasp", "transfer", "release")
FINE_TO_PHASE = {
    "approach": 0,
    "descend": 0,
    "grasp": 0,
    "lift": 1,
    "transport": 1,
    "place": 2,
    "complete": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RGB-only three-phase latent TD-MPC for OGBench Cube."
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "--stage",
        choices=("collect", "train", "adapt", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--regenerate-data", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key, value in config["paths"].items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (HERE / candidate).resolve()
        config["paths"][key] = candidate
    return config


def apply_smoke_config(config: dict[str, Any]) -> None:
    config["paths"]["output_dir"] = config["paths"]["output_dir"] / "smoke"
    config["teacher"].update(episodes=3, max_steps=90, reuse_dataset=False)
    config["training"].update(epochs=2, batch_size=32)
    config["planner"].update(num_candidates=4, horizon_blocks=2)
    config["online_training"].update(
        episodes=1,
        max_steps=6,
        warmup_samples=1,
        batch_size=4,
    )
    config["evaluation"].update(episodes=1, max_steps=6)


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(json_value(record), ensure_ascii=True) + "\n")


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(json_value(value), output, indent=2, ensure_ascii=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_reset_options(config: dict[str, Any]) -> dict[str, Any]:
    task = config["task"]
    return {
        "render_goal": True,
        "task_info": {
            "task_name": str(task["name"]),
            "init_xyzs": np.asarray([task["init_xyz"]], dtype=np.float64),
            "goal_xyzs": np.asarray([task["goal_xyz"]], dtype=np.float64),
        },
    }


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def load_world_model(model_dir: Path, device: torch.device):
    return wajepa.load_world_model(model_dir, device)


def load_action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as dataset:
        actions = np.asarray(dataset["actions"], dtype=np.float32)
    mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = actions.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-4)


@torch.inference_mode()
def encode_images(
    model,
    images: list[np.ndarray] | np.ndarray,
    transform,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    del transform, device, batch_size
    return model.encode_images(images)


@torch.inference_mode()
def encode_image(
    model,
    image: np.ndarray,
    transform,
    device: torch.device,
) -> torch.Tensor:
    del transform, device
    return model.encode_image(image)


@torch.inference_mode()
def predict_latent_block(
    model,
    latent: torch.Tensor,
    normalized_action_block: torch.Tensor,
) -> torch.Tensor:
    action_embedding = model.action_encoder(
        normalized_action_block.unsqueeze(1)
    )
    return model.predict(latent.unsqueeze(1), action_embedding)[:, -1].float()


def action_block(
    actions: list[np.ndarray],
    index: int,
    block: int,
) -> np.ndarray:
    result = np.zeros((block, 5), dtype=np.float32)
    available = actions[index : index + block]
    if available:
        result[: len(available)] = np.asarray(available, dtype=np.float32)
        if len(available) < block:
            result[len(available) :] = available[-1]
    return result


def coarse_phase(fine_phase: str) -> int:
    return int(FINE_TO_PHASE.get(fine_phase, 2))


def phase_endpoint(phases: list[int], index: int) -> int:
    current = phases[index]
    for future in range(index + 1, len(phases)):
        if phases[future] > current:
            return future
    return len(phases) - 1


def collect_dataset(
    config: dict[str, Any],
    world_model,
    transform,
    device: torch.device,
    dataset_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    if dataset_path.exists() and bool(config["teacher"]["reuse_dataset"]):
        with np.load(dataset_path) as dataset:
            return {
                "reused": True,
                "samples": int(len(dataset["latents"])),
                "episodes": int(len(np.unique(dataset["episode_ids"]))),
                "successes": int(dataset["episode_successes"].sum()),
            }

    settings = config["teacher"]
    training = config["training"]
    block = int(config["model"]["action_block"])
    teacher = PrivilegedTeacher(config["paths"]["coordinate_actor_checkpoint"])
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)
    rng = np.random.default_rng(int(config["seed"]) + 31)
    log_path.unlink(missing_ok=True)
    arrays: dict[str, list[np.ndarray]] = {
        "previous_latents": [],
        "latents": [],
        "next_latents": [],
        "goal_latents": [],
        "phase_target_latents": [],
        "next_phase_target_latents": [],
        "action_blocks": [],
        "phase_ids": [],
        "next_phase_ids": [],
        "completion_targets": [],
        "rewards": [],
        "dones": [],
        "episode_ids": [],
        "episode_successes": [],
    }
    successes = 0
    phase_counts: Counter[int] = Counter()

    for episode in range(int(settings["episodes"])):
        state, reset_info = env.reset(
            seed=int(config["seed"]) + int(settings["seed_offset"]) + episode,
            options=task_reset_options(config),
        )
        goal_state = np.asarray(reset_info["goal"], dtype=np.float32)
        goal_image = np.asarray(reset_info["goal_rendered"]).copy()
        frames = [np.asarray(env.render()).copy()]
        ideal_actions: list[np.ndarray] = []
        phases: list[int] = []
        fine_phases: list[str] = []
        success = False
        terminated = False
        truncated = False

        for _ in range(int(settings["max_steps"])):
            ideal_action, fine_phase = teacher.action(
                state, goal_state, settings
            )
            semantic = semantic_state(state, goal_state)
            if (
                fine_phase == "place"
                and np.linalg.norm(semantic.cube - semantic.goal)
                <= float(settings["release_distance_m"])
            ):
                ideal_action = ideal_action.copy()
                ideal_action[4] = -1.0
            executed_action = np.clip(
                ideal_action
                + rng.normal(
                    0.0,
                    float(settings["action_noise_std"]),
                    size=ideal_action.shape,
                ),
                -1.0,
                1.0,
            ).astype(np.float32)
            phases.append(coarse_phase(fine_phase))
            fine_phases.append(fine_phase)
            ideal_actions.append(ideal_action.astype(np.float32))
            state, _, terminated, truncated, info = env.step(executed_action)
            frames.append(np.asarray(env.render()).copy())
            success = bool(info.get("success", False))
            if success or terminated or truncated:
                break

        final_fine = detect_phase(semantic_state(state, goal_state), settings)
        phases.append(coarse_phase(final_fine))
        fine_phases.append(final_fine)
        successes += int(success)
        phase_counts.update(phases)
        record = {
            "episode": episode,
            "steps": len(ideal_actions),
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "fine_phase_counts": dict(Counter(fine_phases)),
            "coarse_phase_counts": {
                PHASE_NAMES[key]: value
                for key, value in Counter(phases).items()
            },
        }
        append_jsonl(log_path, record)
        print(json.dumps(record))

        latents = encode_images(world_model, frames, transform, device)
        goal_latent = encode_images(
            world_model, [goal_image], transform, device
        )[0]
        terminal = len(frames) - 1
        for index in range(len(frames)):
            next_index = min(index + 1, terminal)
            endpoint = phase_endpoint(phases, index)
            next_endpoint = phase_endpoint(phases, next_index)
            phase_id = phases[index]
            next_phase_id = phases[next_index]
            reward = 0.0
            if index < terminal:
                reward -= float(training["step_penalty"])
                reward += float(training["phase_transition_reward"]) * max(
                    0, next_phase_id - phase_id
                )
                if success and next_index == terminal:
                    reward += float(training["success_reward"])
            arrays["previous_latents"].append(
                latents[max(0, index - 1)]
            )
            arrays["latents"].append(latents[index])
            arrays["next_latents"].append(latents[next_index])
            arrays["goal_latents"].append(goal_latent)
            arrays["phase_target_latents"].append(latents[endpoint])
            arrays["next_phase_target_latents"].append(
                latents[next_endpoint]
            )
            arrays["action_blocks"].append(
                action_block(ideal_actions, index, block).reshape(-1)
            )
            arrays["phase_ids"].append(
                np.asarray(phase_id, dtype=np.int64)
            )
            arrays["next_phase_ids"].append(
                np.asarray(next_phase_id, dtype=np.int64)
            )
            arrays["completion_targets"].append(
                np.asarray(success and index == terminal, dtype=np.float32)
            )
            arrays["rewards"].append(
                np.asarray(reward, dtype=np.float32)
            )
            arrays["dones"].append(
                np.asarray(index == terminal, dtype=np.float32)
            )
            arrays["episode_ids"].append(
                np.asarray(episode, dtype=np.int64)
            )
            arrays["episode_successes"].append(
                np.asarray(success, dtype=np.int64)
            )

    env.close()
    if successes < 2:
        raise RuntimeError(
            "The teacher produced fewer than two successful trajectories."
        )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        **{key: np.asarray(value) for key, value in arrays.items()},
    )
    return {
        "reused": False,
        "samples": len(arrays["latents"]),
        "episodes": int(settings["episodes"]),
        "successes": successes,
        "phase_counts": {
            PHASE_NAMES[key]: value for key, value in phase_counts.items()
        },
    }


def build_models(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    VisualPhaseNet,
    PhaseTargetNet,
    PhaseBlockActor,
    PhaseValueEnsemble,
]:
    settings = config["model"]
    latent_dim = int(settings["latent_dim"])
    hidden_dim = int(settings["hidden_dim"])
    phase_net = VisualPhaseNet(latent_dim, hidden_dim).to(device)
    target_net = PhaseTargetNet(latent_dim, hidden_dim).to(device)
    actor = PhaseBlockActor(
        latent_dim,
        int(settings["action_dim"]),
        int(settings["action_block"]),
        hidden_dim,
    ).to(device)
    critic = PhaseValueEnsemble(
        latent_dim,
        hidden_dim,
        int(settings["value_ensemble_size"]),
    ).to(device)
    return phase_net, target_net, actor, critic


DATASET_KEYS = (
    "previous_latents",
    "latents",
    "next_latents",
    "goal_latents",
    "phase_target_latents",
    "next_phase_target_latents",
    "action_blocks",
    "phase_ids",
    "next_phase_ids",
    "completion_targets",
    "rewards",
    "dones",
)


def dataset_tensors(
    dataset: np.lib.npyio.NpzFile,
    indices: np.ndarray,
) -> TensorDataset:
    tensors = []
    for key in DATASET_KEYS:
        values = np.asarray(dataset[key][indices])
        if key in ("phase_ids", "next_phase_ids"):
            tensors.append(torch.from_numpy(values.astype(np.int64)))
        else:
            tensors.append(torch.from_numpy(values.astype(np.float32)))
    return TensorDataset(*tensors)


def model_losses(
    *,
    phase_net: VisualPhaseNet,
    target_net: PhaseTargetNet,
    actor: PhaseBlockActor,
    critic: PhaseValueEnsemble,
    target_critic: PhaseValueEnsemble,
    batch: tuple[torch.Tensor, ...],
    settings: dict[str, Any],
    phase_weights: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        previous,
        current,
        next_latent,
        goal,
        phase_target,
        next_phase_target,
        action_blocks,
        phase,
        next_phase,
        completion,
        reward,
        done,
    ) = [item.to(device) for item in batch]

    phase_logits, completion_logits = phase_net(previous, current, goal)
    predicted_target = target_net(previous, current, goal, phase)
    predicted_actions = actor.mean(
        previous, current, phase_target, goal, phase
    )
    values = critic(previous, current, phase_target, goal, phase)
    with torch.no_grad():
        next_values = target_critic(
            current,
            next_latent,
            next_phase_target,
            goal,
            next_phase,
        ).mean(dim=0)
        td_target = reward + float(settings["gamma"]) * (1.0 - done) * next_values

    phase_loss = nn.functional.cross_entropy(
        phase_logits, phase, weight=phase_weights
    )
    positive_weight = torch.tensor(
        float(settings["completion_positive_weight"]), device=device
    )
    completion_loss = nn.functional.binary_cross_entropy_with_logits(
        completion_logits, completion, pos_weight=positive_weight
    )
    target_loss = nn.functional.mse_loss(predicted_target, phase_target)
    actor_loss = nn.functional.mse_loss(predicted_actions, action_blocks)
    critic_loss = nn.functional.mse_loss(
        values, td_target.unsqueeze(0).expand_as(values)
    )
    total = (
        float(settings["phase_coefficient"]) * phase_loss
        + float(settings["completion_coefficient"]) * completion_loss
        + float(settings["target_coefficient"]) * target_loss
        + float(settings["actor_coefficient"]) * actor_loss
        + float(settings["critic_coefficient"]) * critic_loss
    )
    metrics = {
        "loss": float(total.detach().item()),
        "phase_loss": float(phase_loss.detach().item()),
        "completion_loss": float(completion_loss.detach().item()),
        "target_loss": float(target_loss.detach().item()),
        "actor_loss": float(actor_loss.detach().item()),
        "critic_loss": float(critic_loss.detach().item()),
        "phase_accuracy": float(
            (phase_logits.argmax(dim=-1) == phase).float().mean().detach().item()
        ),
    }
    return total, metrics


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters()
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def save_checkpoint(
    path: Path,
    config: dict[str, Any],
    phase_net: VisualPhaseNet,
    target_net: PhaseTargetNet,
    actor: PhaseBlockActor,
    critic: PhaseValueEnsemble,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    extra: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase_net": phase_net.state_dict(),
            "target_net": target_net.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "action_mean": action_mean,
            "action_scale": action_scale,
            "config": json_value(config),
            "extra": json_value(extra),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    VisualPhaseNet,
    PhaseTargetNet,
    PhaseBlockActor,
    PhaseValueEnsemble,
    np.ndarray,
    np.ndarray,
]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    phase_net, target_net, actor, critic = build_models(config, device)
    phase_net.load_state_dict(checkpoint["phase_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    for model in (phase_net, target_net, actor, critic):
        model.eval()
    return (
        phase_net,
        target_net,
        actor,
        critic,
        np.asarray(checkpoint["action_mean"], dtype=np.float32),
        np.asarray(checkpoint["action_scale"], dtype=np.float32),
    )


def train_offline(
    config: dict[str, Any],
    dataset_path: Path,
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
) -> tuple[
    VisualPhaseNet,
    PhaseTargetNet,
    PhaseBlockActor,
    PhaseValueEnsemble,
]:
    settings = config["training"]
    phase_net, target_net, actor, critic = build_models(config, device)
    target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
    with np.load(dataset_path) as dataset:
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
        episodes = np.unique(episode_ids)
        rng = np.random.default_rng(int(config["seed"]))
        rng.shuffle(episodes)
        validation_count = max(
            1,
            int(round(len(episodes) * float(settings["validation_fraction"]))),
        )
        validation_episodes = episodes[:validation_count]
        validation_mask = np.isin(episode_ids, validation_episodes)
        train_indices = np.flatnonzero(~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
        train_dataset = dataset_tensors(dataset, train_indices)
        validation_dataset = dataset_tensors(dataset, validation_indices)
        phase_counts = np.bincount(
            np.asarray(dataset["phase_ids"])[train_indices], minlength=3
        ).astype(np.float32)

    phase_weights_np = phase_counts.sum() / np.maximum(3.0 * phase_counts, 1.0)
    phase_weights = torch.from_numpy(
        np.clip(phase_weights_np, 0.25, 4.0)
    ).to(device)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
    )
    models = (phase_net, target_net, actor, critic)
    parameters = [parameter for model in models for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_validation = float("inf")
    best_state = None
    log_path.unlink(missing_ok=True)

    for epoch in range(1, int(settings["epochs"]) + 1):
        for model in models:
            model.train()
        train_metrics = []
        for batch in train_loader:
            loss, metrics = model_losses(
                phase_net=phase_net,
                target_net=target_net,
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                batch=batch,
                settings=settings,
                phase_weights=phase_weights,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                parameters, float(settings["max_grad_norm"])
            )
            optimizer.step()
            soft_update(
                target_critic,
                critic,
                float(settings["target_tau"]),
            )
            train_metrics.append(metrics)

        for model in models:
            model.eval()
        validation_metrics = []
        with torch.inference_mode():
            for batch in validation_loader:
                _, metrics = model_losses(
                    phase_net=phase_net,
                    target_net=target_net,
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    batch=batch,
                    settings=settings,
                    phase_weights=phase_weights,
                    device=device,
                )
                validation_metrics.append(metrics)

        train_mean = {
            key: float(np.mean([item[key] for item in train_metrics]))
            for key in train_metrics[0]
        }
        validation_mean = {
            key: float(np.mean([item[key] for item in validation_metrics]))
            for key in validation_metrics[0]
        }
        record = {
            "epoch": epoch,
            "train": train_mean,
            "validation": validation_mean,
            "phase_counts": phase_counts,
            "phase_weights": phase_weights,
        }
        append_jsonl(log_path, record)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(settings["epochs"]):
            print(json.dumps(json_value(record)))
        if validation_mean["loss"] < best_validation:
            best_validation = validation_mean["loss"]
            best_state = {
                "epoch": epoch,
                "phase_net": copy.deepcopy(phase_net.state_dict()),
                "target_net": copy.deepcopy(target_net.state_dict()),
                "actor": copy.deepcopy(actor.state_dict()),
                "critic": copy.deepcopy(critic.state_dict()),
            }

    if best_state is None:
        raise RuntimeError("Offline training did not produce a checkpoint.")
    phase_net.load_state_dict(best_state["phase_net"])
    target_net.load_state_dict(best_state["target_net"])
    actor.load_state_dict(best_state["actor"])
    critic.load_state_dict(best_state["critic"])
    save_checkpoint(
        checkpoint_path,
        config,
        phase_net,
        target_net,
        actor,
        critic,
        action_mean,
        action_scale,
        {
            "best_epoch": best_state["epoch"],
            "best_validation_loss": best_validation,
        },
    )
    return tuple(model.eval() for model in models)


@torch.inference_mode()
def critic_statistics(
    critic: PhaseValueEnsemble,
    previous: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = critic(previous, current, target, goal, phase)
    return values.mean(dim=0), values.std(dim=0, unbiased=False)


@torch.inference_mode()
def propose_and_score(
    *,
    actor: PhaseBlockActor,
    target_net: PhaseTargetNet,
    critic: PhaseValueEnsemble,
    world_model,
    previous: torch.Tensor,
    current: torch.Tensor,
    goal: torch.Tensor,
    phase_id: int,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    settings: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    device = current.device
    count = int(settings["num_candidates"])
    horizon = int(settings["horizon_blocks"])
    block = actor.action_block
    action_dim = actor.action_dim
    actual_phase = torch.full((1,), phase_id, dtype=torch.long, device=device)
    phase_target = target_net(previous, current, goal, actual_phase)
    candidate_previous = previous.expand(count, -1).clone()
    candidate_current = current.expand(count, -1).clone()
    candidate_goal = goal.expand(count, -1)
    candidate_target = phase_target.expand(count, -1)
    candidate_phase = actual_phase.expand(count)
    mean_tensor = torch.as_tensor(action_mean, device=device)
    scale_tensor = torch.as_tensor(action_scale, device=device)
    exploration = float(settings["exploration_std"])
    blocks = []
    values_along_path = []
    predicted_latents = []

    for _ in range(horizon):
        mean_block = actor.mean(
            candidate_previous,
            candidate_current,
            candidate_target,
            candidate_goal,
            candidate_phase,
        ).reshape(count, block, action_dim)
        noise = torch.from_numpy(
            rng.normal(
                0.0,
                exploration,
                size=(count, block, action_dim),
            ).astype(np.float32)
        ).to(device)
        noise[0].zero_()
        raw_block = (mean_block + noise).clamp(-1.0, 1.0)
        normalized = ((raw_block - mean_tensor) / scale_tensor).reshape(
            count, -1
        )
        next_latent = predict_latent_block(
            world_model, candidate_current, normalized
        )
        step_value, _ = critic_statistics(
            critic,
            candidate_current,
            next_latent,
            candidate_target,
            candidate_goal,
            candidate_phase,
        )
        blocks.append(raw_block)
        predicted_latents.append(next_latent)
        values_along_path.append(step_value)
        candidate_previous, candidate_current = candidate_current, next_latent
        exploration *= float(settings["exploration_decay"])

    plans = torch.stack(blocks, dim=1)
    flattened = plans.reshape(count, horizon * block, action_dim)
    terminal = predicted_latents[-1]
    terminal_value, uncertainty = critic_statistics(
        critic,
        candidate_previous,
        terminal,
        candidate_target,
        candidate_goal,
        candidate_phase,
    )
    trajectory_value = torch.stack(values_along_path, dim=1).mean(dim=1)
    short_cost = (terminal - candidate_target).square().mean(dim=-1)
    behavior_deviation = (
        plans - plans[0:1]
    ).square().mean(dim=(1, 2, 3))
    smoothness = (
        flattened[:, 1:] - flattened[:, :-1]
    ).square().mean(dim=(1, 2))
    score = (
        -float(settings["short_target_coefficient"]) * short_cost
        + float(settings["terminal_value_coefficient"]) * terminal_value
        + float(settings["trajectory_value_coefficient"]) * trajectory_value
        - float(settings["uncertainty_coefficient"]) * uncertainty
        - float(settings["action_prior_coefficient"]) * behavior_deviation
        - float(settings["action_smoothness_coefficient"]) * smoothness
    )
    best_index = int(torch.argmax(score).item())
    elite_count = min(int(settings["elite_count"]), count)
    elite_scores, elite_indices = torch.topk(score, elite_count)
    temperature = max(float(settings["elite_temperature"]), 1e-6)
    weights = torch.softmax(
        (elite_scores - elite_scores.max()) / temperature, dim=0
    )
    elite_plan = (
        flattened[elite_indices] * weights[:, None, None]
    ).sum(dim=0)
    return {
        "plans": plans,
        "flattened": flattened,
        "score": score,
        "best_index": best_index,
        "phase_target": phase_target,
        "elite_plan": elite_plan,
        "terminal_value": terminal_value,
        "trajectory_value": trajectory_value,
        "uncertainty": uncertainty,
        "short_cost": short_cost,
        "behavior_deviation": behavior_deviation,
    }


def update_actor_online(
    actor: PhaseBlockActor,
    anchor_actor: PhaseBlockActor,
    optimizer: torch.optim.Optimizer,
    replay: deque,
    settings: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    actor.train()
    losses = []
    for _ in range(int(settings["updates_per_step"])):
        count = min(int(settings["batch_size"]), len(replay))
        indices = rng.choice(len(replay), size=count, replace=False)
        previous = torch.stack([replay[int(i)][0] for i in indices]).to(device)
        current = torch.stack([replay[int(i)][1] for i in indices]).to(device)
        target = torch.stack([replay[int(i)][2] for i in indices]).to(device)
        goal = torch.stack([replay[int(i)][3] for i in indices]).to(device)
        phase = torch.stack([replay[int(i)][4] for i in indices]).to(device)
        action_target = torch.stack(
            [replay[int(i)][5] for i in indices]
        ).to(device)
        prediction = actor.mean(
            previous, current, target, goal, phase
        ).reshape(count, actor.action_block, actor.action_dim)
        with torch.no_grad():
            anchor = anchor_actor.mean(
                previous, current, target, goal, phase
            ).reshape(count, actor.action_block, actor.action_dim)
        selected_loss = nn.functional.mse_loss(
            prediction[:, 0], action_target
        )
        anchor_loss = nn.functional.mse_loss(prediction, anchor)
        loss = selected_loss + float(
            settings["bc_anchor_coefficient"]
        ) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings["max_grad_norm"])
        )
        optimizer.step()
        losses.append(float(loss.item()))
    actor.eval()
    return float(np.mean(losses))


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def run_closed_loop(
    *,
    config: dict[str, Any],
    phase_net: VisualPhaseNet,
    target_net: PhaseTargetNet,
    actor: PhaseBlockActor,
    critic: PhaseValueEnsemble,
    world_model,
    transform,
    device: torch.device,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    training: bool,
    log_path: Path,
    selected_video_path: Path,
) -> dict[str, Any]:
    settings = config["online_training"] if training else config["evaluation"]
    planner = config["planner"]
    phase_control = config["phase_control"]
    episodes = int(settings["episodes"])
    max_steps = int(settings["max_steps"])
    fps = int(config["evaluation"]["fps"])
    rng = np.random.default_rng(
        int(config["seed"])
        + int(settings["seed_offset"])
        + (0 if training else 100000)
    )
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)
    anchor_actor = copy.deepcopy(actor).eval().requires_grad_(False)
    optimizer = (
        torch.optim.AdamW(
            actor.parameters(),
            lr=float(config["online_training"]["learning_rate"]),
            weight_decay=1e-5,
        )
        if training and bool(config["online_training"]["enabled"])
        else None
    )
    replay = deque(maxlen=int(config["online_training"]["replay_capacity"]))
    log_path.unlink(missing_ok=True)
    episode_results = []
    world_model_seconds = 0.0
    world_model_calls = 0
    selected_frames = None
    selected_success = False
    selected_episode = None
    selected_actor_state = copy.deepcopy(actor.state_dict()) if training else None
    selected_actor_success = False
    phase_usage: Counter[int] = Counter()

    for episode in range(episodes):
        _, reset_info = env.reset(
            seed=int(config["seed"]) + int(settings["seed_offset"]) + episode,
            options=task_reset_options(config),
        )
        goal_image = np.asarray(reset_info["goal_rendered"]).copy()
        goal_latent = encode_image(world_model, goal_image, transform, device)
        current_image = np.asarray(env.render()).copy()
        current_latent = encode_image(
            world_model, current_image, transform, device
        )
        previous_latent = current_latent.clone()
        frames = [current_image]
        phase_id = 0
        transition_streak = 0
        completion_streak = 0
        phase_history = [phase_id]
        environment_step = 0
        planning_cycle = 0
        success_seen = False
        terminated = False
        truncated = False
        stopped_by_visual_completion = False
        accepted_samples = 0
        online_updates = 0

        while environment_step < max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            proposal = propose_and_score(
                actor=actor,
                target_net=target_net,
                critic=critic,
                world_model=world_model,
                previous=previous_latent,
                current=current_latent,
                goal=goal_latent,
                phase_id=phase_id,
                action_mean=action_mean,
                action_scale=action_scale,
                settings=planner,
                rng=rng,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            world_model_seconds += elapsed
            world_model_calls += 1
            planning_cycle += 1
            best_index = int(proposal["best_index"])
            score = proposal["score"]
            planner_advantage = float(
                (score[best_index] - score[0]).detach().cpu().item()
            )
            base_plan = proposal["flattened"][0]
            planner_plan = proposal["flattened"][best_index]
            phase_target = proposal["phase_target"]
            execute_limit = int(planner["execute_steps_by_phase"][phase_id])
            execute_steps = min(
                execute_limit,
                int(planner_plan.shape[0]),
                max_steps - environment_step,
            )
            blend = torch.as_tensor(
                planner["action_blend"],
                dtype=base_plan.dtype,
                device=base_plan.device,
            )

            for plan_action_index in range(execute_steps):
                phase_before = phase_id
                phase_tensor = torch.full(
                    (1,), phase_before, dtype=torch.long, device=device
                )
                before_value, _ = critic_statistics(
                    critic,
                    previous_latent,
                    current_latent,
                    phase_target,
                    goal_latent,
                    phase_tensor,
                )
                base_action = base_plan[plan_action_index]
                planner_action = planner_plan[plan_action_index]
                executed_action = torch.lerp(
                    base_action, planner_action, blend
                ).clamp(-1.0, 1.0)
                action = executed_action.detach().cpu().numpy()
                _, _, terminated, truncated, info = env.step(action)
                environment_step += 1
                success = bool(info.get("success", False))
                success_seen = success_seen or success
                next_image = np.asarray(env.render()).copy()
                next_latent = encode_image(
                    world_model, next_image, transform, device
                )
                after_value, after_uncertainty = critic_statistics(
                    critic,
                    current_latent,
                    next_latent,
                    phase_target,
                    goal_latent,
                    phase_tensor,
                )
                value_progress = float(
                    (after_value - before_value).detach().cpu().item()
                )
                with torch.inference_mode():
                    phase_logits, completion_logit = phase_net(
                        current_latent, next_latent, goal_latent
                    )
                    phase_probabilities = torch.softmax(
                        phase_logits, dim=-1
                    )[0]
                    predicted_phase = int(
                        phase_probabilities.argmax().item()
                    )
                    completion_probability = float(
                        torch.sigmoid(completion_logit)[0].item()
                    )

                if predicted_phase > phase_id:
                    transition_streak += 1
                else:
                    transition_streak = 0
                stage_changed = False
                if transition_streak >= int(
                    phase_control["transition_consecutive_frames"]
                ):
                    phase_id = min(2, phase_id + 1)
                    transition_streak = 0
                    stage_changed = True
                    phase_history.append(phase_id)

                if completion_probability >= float(
                    phase_control["complete_probability"]
                ):
                    completion_streak += 1
                else:
                    completion_streak = 0
                stopped_by_visual_completion = (
                    completion_streak
                    >= int(phase_control["complete_consecutive_frames"])
                )

                accepted = (
                    training
                    and optimizer is not None
                    and planner_advantage
                    >= float(
                        config["online_training"][
                            "minimum_planner_advantage"
                        ]
                    )
                    and value_progress
                    >= float(
                        config["online_training"]["minimum_value_progress"]
                    )
                )
                adaptation_loss = 0.0
                if accepted:
                    replay.append(
                        (
                            previous_latent[0].detach().cpu(),
                            current_latent[0].detach().cpu(),
                            phase_target[0].detach().cpu(),
                            goal_latent[0].detach().cpu(),
                            torch.tensor(phase_before, dtype=torch.long),
                            torch.lerp(
                                base_action,
                                proposal["elite_plan"][plan_action_index],
                                blend,
                            ).detach().cpu(),
                        )
                    )
                    accepted_samples += 1
                    if len(replay) >= int(
                        config["online_training"]["warmup_samples"]
                    ):
                        adaptation_loss = update_actor_online(
                            actor,
                            anchor_actor,
                            optimizer,
                            replay,
                            config["online_training"],
                            rng,
                            device,
                        )
                        online_updates += 1

                phase_usage[phase_before] += 1
                record = {
                    "mode": "online_train" if training else "frozen_evaluate",
                    "episode": episode,
                    "environment_step": environment_step,
                    "planning_cycle": planning_cycle,
                    "plan_action_index": plan_action_index,
                    "phase_before": PHASE_NAMES[phase_before],
                    "phase_after": PHASE_NAMES[phase_id],
                    "phase_probabilities": phase_probabilities,
                    "completion_probability": completion_probability,
                    "stage_changed": stage_changed,
                    "best_candidate": best_index,
                    "planner_advantage": planner_advantage,
                    "base_score": float(score[0].detach().cpu().item()),
                    "best_score": float(
                        score[best_index].detach().cpu().item()
                    ),
                    "short_target_cost": float(
                        proposal["short_cost"][best_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "predicted_terminal_value": float(
                        proposal["terminal_value"][best_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "predicted_value_uncertainty": float(
                        proposal["uncertainty"][best_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "real_rgb_value_progress": value_progress,
                    "real_rgb_value_uncertainty": float(
                        after_uncertainty.detach().cpu().item()
                    ),
                    "online_sample_accepted": accepted,
                    "adaptation_loss": adaptation_loss,
                    "executed_action": action,
                    "oracle_success_for_metrics_only": success,
                    "world_model_inference_seconds": (
                        elapsed if plan_action_index == 0 else 0.0
                    ),
                }
                append_jsonl(log_path, record)
                if (
                    environment_step == 1
                    or environment_step % 10 == 0
                    or stage_changed
                    or success
                ):
                    print(json.dumps(json_value(record)))

                frames.append(next_image)
                previous_latent, current_latent = current_latent, next_latent
                if (
                    stopped_by_visual_completion
                    or terminated
                    or truncated
                    or stage_changed
                ):
                    break

            if stopped_by_visual_completion or terminated or truncated:
                break

        status = "success" if success_seen else "failed"
        mode_name = "online_training" if training else "frozen_evaluation"
        episode_video = selected_video_path.parent / (
            f"{mode_name}_episode_{episode}_{status}.mp4"
        )
        save_video(episode_video, frames, fps)
        if selected_frames is None or (success_seen and not selected_success):
            selected_frames = frames
            selected_success = success_seen
            selected_episode = episode
        if training and success_seen and not selected_actor_success:
            selected_actor_state = copy.deepcopy(actor.state_dict())
            selected_actor_success = True

        episode_result = {
            "episode": episode,
            "steps": environment_step,
            "oracle_success_for_metrics_only": success_seen,
            "stopped_by_visual_completion": stopped_by_visual_completion,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "phase_history": [PHASE_NAMES[index] for index in phase_history],
            "accepted_online_samples": accepted_samples,
            "online_updates": online_updates,
            "video": str(episode_video),
        }
        episode_results.append(episode_result)

    if training and selected_actor_state is not None:
        actor.load_state_dict(selected_actor_state)
    if selected_frames is not None:
        save_video(selected_video_path, selected_frames, fps)
    env.close()
    return {
        "mode": "online_train" if training else "frozen_evaluate",
        "episodes": episodes,
        "oracle_successes_for_metrics_only": int(
            sum(item["oracle_success_for_metrics_only"] for item in episode_results)
        ),
        "world_model_inference_seconds": world_model_seconds,
        "world_model_calls": world_model_calls,
        "mean_world_model_inference_seconds": world_model_seconds
        / max(1, world_model_calls),
        "selected_video_episode": selected_episode,
        "selected_video_success": selected_success,
        "checkpoint_success": selected_actor_success,
        "phase_usage": {
            PHASE_NAMES[key]: value for key, value in phase_usage.items()
        },
        "episode_results": episode_results,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.smoke:
        apply_smoke_config(config)
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LeWM planning.")

    set_seed(int(config["seed"]))
    device = torch.device(config["device"])
    output_dir = config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)
    dataset_path = output_dir / "data" / "three_phase_latents.npz"
    if args.regenerate_data:
        dataset_path.unlink(missing_ok=True)
    offline_checkpoint = (
        output_dir / "checkpoints" / "three_phase_actor_offline.pt"
    )
    final_checkpoint = (
        output_dir / "checkpoints" / "three_phase_actor_final.pt"
    )
    transform = build_transform(int(config["image_size"]))
    world_model = load_world_model(config["paths"]["model_dir"], device)
    action_mean, action_scale = load_action_stats(
        config["paths"]["action_dataset"]
    )
    summary: dict[str, Any] = {}

    if args.stage in ("collect", "train", "all"):
        summary["collection"] = collect_dataset(
            config,
            world_model,
            transform,
            device,
            dataset_path,
            output_dir / "logs" / "teacher_collection.jsonl",
        )
        if args.stage == "collect":
            save_json(output_dir / "summary.json", summary)
            print(json.dumps(json_value(summary), indent=2))
            return

    trained_models = None
    if args.stage in ("train", "all"):
        trained_models = train_offline(
            config,
            dataset_path,
            device,
            offline_checkpoint,
            output_dir / "logs" / "offline_training.jsonl",
            action_mean,
            action_scale,
        )
        summary["offline_training"] = {
            "checkpoint": str(offline_checkpoint)
        }

    if args.stage in ("adapt", "all"):
        if trained_models is None:
            if not offline_checkpoint.exists():
                raise FileNotFoundError(
                    "Offline checkpoint missing. Run --stage train first."
                )
            (
                phase_net,
                target_net,
                actor,
                critic,
                action_mean,
                action_scale,
            ) = load_checkpoint(offline_checkpoint, config, device)
        else:
            phase_net, target_net, actor, critic = trained_models
        for model in (phase_net, target_net, critic):
            model.eval().requires_grad_(False)
        summary["online_training"] = run_closed_loop(
            config=config,
            phase_net=phase_net,
            target_net=target_net,
            actor=actor,
            critic=critic,
            world_model=world_model,
            transform=transform,
            device=device,
            action_mean=action_mean,
            action_scale=action_scale,
            training=True,
            log_path=output_dir / "logs" / "online_training.jsonl",
            selected_video_path=(
                output_dir / "videos" / "online_training_selected.mp4"
            ),
        )
        save_checkpoint(
            final_checkpoint,
            config,
            phase_net,
            target_net,
            actor,
            critic,
            action_mean,
            action_scale,
            {"online_training": summary["online_training"]},
        )

    if args.stage in ("evaluate", "adapt", "all"):
        checkpoint = (
            final_checkpoint if final_checkpoint.exists() else offline_checkpoint
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                "No trained checkpoint found. Run --stage train first."
            )
        (
            phase_net,
            target_net,
            actor,
            critic,
            action_mean,
            action_scale,
        ) = load_checkpoint(checkpoint, config, device)
        for model in (phase_net, target_net, actor, critic):
            model.eval().requires_grad_(False)
        summary["evaluation"] = run_closed_loop(
            config=config,
            phase_net=phase_net,
            target_net=target_net,
            actor=actor,
            critic=critic,
            world_model=world_model,
            transform=transform,
            device=device,
            action_mean=action_mean,
            action_scale=action_scale,
            training=False,
            log_path=output_dir / "logs" / "evaluation.jsonl",
            selected_video_path=(
                output_dir / "videos" / "frozen_evaluation_selected.mp4"
            ),
        )

    save_json(output_dir / "summary.json", summary)
    print(json.dumps(json_value(summary), indent=2))


if __name__ == "__main__":
    main()
