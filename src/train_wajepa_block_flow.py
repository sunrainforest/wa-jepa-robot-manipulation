"""Train the WA-JEPA transition used by one 5-step LeWM_SAIMO MPC block."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("WA_JEPA_ROOT", ROOT / "third_party/WA-JEPA"))
sys.path.insert(0, str(OFFICIAL))

import numpy as np
import torch
from torch import nn

from models.multiview_causal_future_jepa import SceneTrajectoryFlowPredictor


CACHE = ROOT / "data/wajepa_cube_semantic_latents.npz"
SOURCE = ROOT / "data/cube-single-play-v0.npz"
OUTPUT = ROOT / "runs/wajepa_block5_flow_v1"


class RobotBlockFlow(nn.Module):
    def __init__(self, dim=128, hidden=256, layers=4):
        super().__init__()
        self.config = {"dim": dim, "hidden": hidden, "layers": layers}
        self.phase = nn.Embedding(3, dim)
        self.proprio = nn.Linear(6, dim)
        self.actions = nn.Sequential(
            nn.Linear(5 * 5, hidden), nn.GELU(), nn.Linear(hidden, 3 * dim)
        )
        self.predictor = SceneTrajectoryFlowPredictor(
            scene_dim=dim, num_scene_tokens=3, num_cameras=1,
            scene_grid_size=(1, 1), scene_history_steps=3,
            scene_future_steps=3, ego_status_dim=6, history_steps=1,
            trajectory_horizon=1, hidden_dim=hidden, num_layers=layers,
            num_heads=8, enable_trajectory_head=False,
        )

    def forward(self, current, proprio, phase, actions, noisy, t):
        state = self.proprio(proprio) + self.phase(phase)
        condition = self.actions(actions.flatten(1)).reshape(-1, 3, current.shape[-1])
        indices = torch.arange(3, device=current.device)[None].expand(len(current), -1)
        clean, _ = self.predictor(
            context_scene=current + state[:, None],
            context_token_indices=indices,
            noisy_future_scene=noisy,
            future_condition_scene=condition + state[:, None],
            t_cont=t,
        )
        return clean

    def loss(self, current, proprio, phase, actions, future):
        t = torch.sigmoid(torch.randn(len(current), device=current.device) * 1.6 - 0.2)
        t = t.clamp(1e-4, 1 - 1e-4)
        noise = torch.randn_like(future)
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * future
        return (self(current, proprio, phase, actions, noisy, t) - future).square().mean()

    @torch.no_grad()
    def predict(self, current, proprio, phase, actions, steps=4):
        generator = torch.Generator(device=current.device).manual_seed(71)
        scene = torch.randn(
            (1, 3, current.shape[-1]), generator=generator, device=current.device
        ).expand(len(current), -1, -1).clone()
        for step in range(steps):
            t = torch.full((len(current),), step / steps, device=current.device)
            clean = self(current, proprio, phase, actions, scene, t)
            scene = scene + (clean - scene) / ((1 - t[:, None, None]) * steps)
        return scene.float()


def robot_proprio(observation):
    closure = np.clip(observation[..., 17:18] / 3.0, 0.0, 1.0)
    return np.concatenate(
        [observation[..., 12:15], observation[..., 15:17], closure], axis=-1
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    cache = np.load(args.cache, mmap_mode="r")
    source = np.load(args.source, mmap_mode="r")
    unique = cache["unique_source_indices"]
    lookup = {int(value): row for row, value in enumerate(unique)}
    source_indices = cache["source_indices"]
    rows = np.asarray([
        row for row, index in enumerate(source_indices)
        if int(index + 5) in lookup and index // 1001 == (index + 5) // 1001
    ])
    current_rows = cache["current_rows"][rows]
    future_rows = np.asarray([lookup[int(source_indices[row] + 5)] for row in rows])
    current = torch.from_numpy(cache["latents"][current_rows].copy()).reshape(-1, 3, 128)
    future = torch.from_numpy(cache["latents"][future_rows].copy()).reshape(-1, 3, 128)
    observations = source["observations"][source_indices[rows]]
    proprio = torch.from_numpy(robot_proprio(observations))
    actions = torch.from_numpy(np.stack([
        source["actions"][index:index + 5] for index in source_indices[rows]
    ]).astype(np.float32))
    aligned = cache["predicate_targets"][rows, 0] >= 0.5
    near = np.linalg.norm(
        (observations[:, 12:15] - observations[:, 19:22]) / 10.0, axis=1
    ) <= 0.035
    grasped = (observations[:, 18] >= 0.5) & (observations[:, 17] / 3.0 >= 0.45) & near
    phase = torch.from_numpy(np.where(grasped, 2, np.where(aligned, 1, 0)).astype(np.int64))
    train = cache["episode_ids"][rows] < 900
    action_mean, action_std = actions[train].mean((0, 1)), actions[train].std((0, 1)).clamp_min(1e-4)
    proprio_mean, proprio_std = proprio[train].mean(0), proprio[train].std(0).clamp_min(1e-4)
    actions = (actions - action_mean) / action_std
    proprio = (proprio - proprio_mean) / proprio_std

    device = torch.device("cuda:0")
    model = RobotBlockFlow().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    train_ids = torch.from_numpy(np.flatnonzero(train))
    validation_ids = torch.from_numpy(np.flatnonzero(~train))
    generator = torch.Generator().manual_seed(71)
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, 2001):
        ids = train_ids[torch.randint(len(train_ids), (128,), generator=generator)]
        batch = [value[ids].to(device) for value in (current, proprio, phase, actions, future)]
        loss = model.loss(*batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)

    predicted = []
    with torch.no_grad():
        for ids in validation_ids.split(256):
            values = [value[ids].to(device) for value in (current, proprio, phase, actions)]
            predicted.append(model.predict(*values).cpu())
    prediction = torch.cat(predicted)
    target = future[validation_ids]
    report = {
        "train_samples": int(train.sum()),
        "validation_samples": int((~train).sum()),
        "validation_mse": float((prediction - target).square().mean()),
        "identity_mse": float((current[validation_ids] - target).square().mean()),
    }
    torch.save({
        "model": model.state_dict(), "config": model.config,
        "action_normalization": (action_mean, action_std),
        "proprio_normalization": (proprio_mean, proprio_std),
        "report": report,
    }, args.output / "block5.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
