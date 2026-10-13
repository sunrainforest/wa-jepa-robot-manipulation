"""Minimal WA-JEPA stage-1 training on saved OGBench robot trajectories."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("WA_JEPA_ROOT", ROOT / "third_party/WA-JEPA"))
sys.path.insert(0, str(OFFICIAL))

import numpy as np
import ogbench
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.factory import build_world_model


DATA = Path(os.environ.get("WAJEPA_DATASET", ROOT / "data/cube-single-play-v0.npz"))
PRETRAINED = Path(
    os.environ.get(
        "VJEPA2_CHECKPOINT",
        ROOT / "artifacts/vjepa2_1_vitl_dist_vitG_384.pt",
    )
)


class RobotVideoDataset(Dataset):
    """Render 4 history and 8 future frames from saved simulator states."""

    EPISODE_STEPS = 1001

    def __init__(
        self,
        episode_start: int,
        episode_count: int,
        image_size: int,
        frame_gap: int,
        anchor_stride: int,
    ) -> None:
        arrays = np.load(DATA)
        self.qpos = arrays["qpos"]
        self.qvel = arrays["qvel"]
        self.image_size = image_size
        self.frame_gap = frame_gap
        first = 3 * frame_gap
        last = self.EPISODE_STEPS - 1 - 8 * frame_gap
        self.anchors = [
            episode * self.EPISODE_STEPS + offset
            for episode in range(episode_start, episode_start + episode_count)
            for offset in range(first, last + 1, anchor_stride)
        ]
        self.env = None

    def __len__(self) -> int:
        return len(self.anchors)

    def _environment(self):
        if self.env is None:
            self.env = ogbench.make_env_and_datasets(
                "cube-single-play-v0",
                env_only=True,
                terminate_at_goal=False,
            )
            self.env.reset(seed=0)
        return self.env

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        anchor = self.anchors[item]
        offsets = [*range(-3, 1), *range(1, 9)]
        env = self._environment()
        frames = []
        for offset in offsets:
            index = anchor + offset * self.frame_gap
            env.unwrapped.set_state(self.qpos[index], self.qvel[index])
            frames.append(np.asarray(env.render()).copy())
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        if video.shape[-2:] != (self.image_size, self.image_size):
            video = F.interpolate(
                video,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        video = video.div_(127.5).sub_(1.0)
        return {"history_images": video[:4], "future_images": video[4:]}


def model_config(output: Path, image_size: int, steps: int) -> dict:
    model = {
        "target": "models.multiview_causal_future_jepa:MultiViewCausalFutureMaskedJEPA",
        "vjepa2_ckpt": str(PRETRAINED),
        "require_pretrained": True,
        "allow_partial_load": False,
        "min_checkpoint_load_ratio": 1.0,
        "copy_target_encoder_from_student": True,
        "variant": "vjepa2_1_vit_large_384",
        "use_rope": True,
        "use_sdpa": True,
        "interpolate_rope": True,
        "modality_embedding": True,
        "img_temporal_dim_size": 1,
        "use_activation_checkpointing": True,
        "camera_names": ["front"],
        "input_hw": [image_size, image_size],
        "pretrained_input_hw": [256, 256],
        "image_normalization": "minus_one_to_imagenet",
        "num_history_frames": 4,
        "num_future_frames": 8,
        "patch_size": 16,
        "tubelet_size": 2,
        "mv_scene_dim": 512,
        "flow_hidden_dim": 512,
        "flow_num_layers": 12,
        "flow_num_heads": 8,
        "enable_trajectory_head": False,
        "num_inference_steps": 4,
        "flow_time_sampling": "logit_normal",
        "flow_logit_mean": -0.2,
        "flow_logit_std": 1.6,
        "full_mask_prob": 0.5,
        "jepa_mask_configs": [
            {
                "num_blocks": 8,
                "spatial_scale": [0.15, 0.15],
                "temporal_scale": [1.0, 1.0],
                "aspect_ratio": [0.75, 1.5],
            }
        ],
        "dynamic_collapse_topk": 64,
        "ema_start": 0.99925,
        "ema_end": 0.99925,
        "ema_total_steps": steps,
        "target_scene_ema_warmup_start": 0.995,
        "target_scene_ema_warmup_steps": min(200, steps),
        "freeze_encoder_steps": 0,
    }
    return {
        "save_root": str(output),
        "model": model,
        "data": {"camera_names": ["front"]},
    }


def optimizer_for(model: torch.nn.Module) -> torch.optim.Optimizer:
    groups = {"encoder": [], "scene": [], "predictor": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            groups["encoder"].append(parameter)
        elif name.startswith("predictor."):
            groups["predictor"].append(parameter)
        else:
            groups["scene"].append(parameter)
    return torch.optim.AdamW(
        [
            {"params": groups["encoder"], "lr": 1e-5},
            {"params": groups["scene"], "lr": 3e-4},
            {"params": groups["predictor"], "lr": 3.5e-4},
        ],
        weight_decay=0.04,
    )


@torch.no_grad()
def validate(model, loader, batches: int) -> float:
    model.eval()
    values = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = model.prepare_batch(batch)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(batch)
        values.append(float(output["loss"]["loss_jepa_pred"]))
    model.train()
    return float(np.mean(values))


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, args) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-gap", type=int, default=5)
    parser.add_argument("--anchor-stride", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=900)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    if not PRETRAINED.is_file():
        raise FileNotFoundError(PRETRAINED)
    if args.train_episodes + args.val_episodes > 1000:
        raise ValueError("train and validation episodes exceed the 1000 saved episodes")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    output = ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    train_set = RobotVideoDataset(
        0,
        args.train_episodes,
        args.image_size,
        args.frame_gap,
        args.anchor_stride,
    )
    val_set = RobotVideoDataset(
        args.train_episodes,
        args.val_episodes,
        args.image_size,
        args.frame_gap,
        args.anchor_stride,
    )
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = build_world_model(model_config(output, args.image_size, args.steps)).to(device)
    model.set_loss_weights({"loss_jepa_pred": 1.0})
    model.train()
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cuda")
    iterator = iter(train_loader)
    log_path = output / "train.jsonl"
    started = time.monotonic()
    best_val = float("inf")

    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.set_train_step(step)
        batch = model.prepare_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(batch)
            loss = result["loss"]["loss_jepa_pred"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradients at step {step + 1}")
        scaler.step(optimizer)
        scaler.update()
        model.after_optimizer_step(step + 1)

        record = {
            "step": step + 1,
            "train_loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "elapsed_seconds": time.monotonic() - started,
        }
        if (step + 1) % args.val_interval == 0 or step + 1 == args.steps:
            record["val_loss"] = validate(model, val_loader, args.val_batches)
            if record["val_loss"] < best_val:
                best_val = record["val_loss"]
                if not args.no_checkpoint:
                    save_checkpoint(output / "best.pt", model, optimizer, scaler, step + 1, args)
        with log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        (output / "status.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        if not args.no_checkpoint and (
            (step + 1) % args.save_interval == 0 or step + 1 == args.steps
        ):
            save_checkpoint(output / "latest.pt", model, optimizer, scaler, step + 1, args)


if __name__ == "__main__":
    main()
