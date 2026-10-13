"""WA-JEPA encoder/Flow adapter for the unchanged LeWM_SAIMO controller."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("WA_JEPA_ROOT", ROOT / "third_party/WA-JEPA"))
FLOW_CHECKPOINT = Path(
    os.environ.get("WAJEPA_FLOW_CHECKPOINT", ROOT / "artifacts/flow/block5.pt")
)
PCA_CHECKPOINT = Path(
    os.environ.get("WAJEPA_PCA_CHECKPOINT", ROOT / "artifacts/normalization.pt")
)
sys.path.insert(0, str(OFFICIAL))
sys.modules.pop("models", None)

from models.factory import build_world_model
from models.jepa_common import _chunk_layer_norm
from train_robot_vjepa_stage1 import model_config
from train_wajepa_block_flow import RobotBlockFlow


def load_encoder(run: Path, device: torch.device):
    config = model_config(run, image_size=224, steps=1000)
    config["model"].update(
        vjepa2_ckpt=None,
        require_pretrained=False,
        use_activation_checkpointing=False,
    )
    model = build_world_model(config)
    checkpoint = torch.load(
        run / "best.pt", map_location="cpu", mmap=True, weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    gc.collect()
    model.target_encoder.to(device).eval()
    return model.target_encoder


@torch.inference_mode()
def encode_future(encoder, images: np.ndarray, device: torch.device) -> np.ndarray:
    parts = []
    for start in range(0, len(images), 8):
        clip = torch.from_numpy(images[start : start + 8]).to(device)
        clip = clip.permute(0, 4, 1, 2, 3).float()
        clip = torch.cat([clip, clip[:, :, -1:]], dim=2)
        clip = F.interpolate(
            clip, size=(6, 224, 224), mode="trilinear", align_corners=False
        ).div_(255.0)
        mean = clip.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = clip.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = encoder((clip - mean) / std, training=True)
            tokens = _chunk_layer_norm(tokens, encoder.embed_dim)
        parts.append(tokens.reshape(len(tokens), 3, 14, 14, -1).mean((2, 3)).cpu())
    return torch.cat(parts).numpy().astype(np.float16)


class WAJEPAWorldModel:
    """Expose WA-JEPA through the two operations used by LeWM_SAIMO MPC."""

    def __init__(self, encoder_run: Path, device: torch.device) -> None:
        self.device = device
        self.encoder = load_encoder(encoder_run, device)
        checkpoint = torch.load(
            FLOW_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
        self.normalization = {
            "action": checkpoint["action_normalization"],
            "proprio": checkpoint["proprio_normalization"],
            "pca": torch.load(
                PCA_CHECKPOINT,
                map_location="cpu", weights_only=False,
            )["pca"],
        }
        self.flow = RobotBlockFlow(**checkpoint["config"])
        self.flow.load_state_dict(checkpoint["model"])
        self.flow.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_images(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        array = np.asarray(images)
        raw = torch.from_numpy(
            encode_future(
                self.encoder,
                np.repeat(array[:, None], 5, axis=1),
                self.device,
            ).astype(np.float32)
        )
        pca = self.normalization["pca"]
        latent = (raw - pca["mean"]) @ pca["basis"] / pca["scale"]
        return latent.flatten(1).numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self.encode_images([image])).to(self.device)

    @torch.inference_mode()
    def predict_latent_block(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        proprio: torch.Tensor,
        phase: torch.Tensor,
        action_block: torch.Tensor,
    ) -> torch.Tensor:
        del previous
        count = len(current)
        current_tokens = current.reshape(count, 3, -1)
        action_mean, action_std = self.normalization["action"]
        proprio_mean, proprio_std = self.normalization["proprio"]
        action_block = (action_block - action_mean.to(self.device)) / action_std.to(
            self.device
        )
        proprio = (proprio - proprio_mean.to(self.device)) / proprio_std.to(
            self.device
        )
        future = self.flow.predict(current_tokens, proprio, phase, action_block)
        return future.flatten(1)


def load_world_model(model_dir: Path, device: torch.device) -> WAJEPAWorldModel:
    return WAJEPAWorldModel(model_dir, device)

