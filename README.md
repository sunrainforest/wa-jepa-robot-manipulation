# WA-JEPA for Robot Manipulation

An ongoing research project that adapts WA-JEPA-style visual representation and action-conditioned flow prediction to the LeWM_SAIMO cube manipulation task.

The controller keeps the original three-stage structure:

```text
ALIGN -> GRASP -> TRANSFER
```

The LeWM visual encoder and transition predictor are replaced by a robot-trained WA-JEPA encoder and a 5-step action-conditioned Flow Predictor. The Actor, Critic, state machine, and MPC interface are adapted to the new 384-dimensional latent space.

## Current results

| Evaluation | Result |
|---|---:|
| Encoded robot frames | 177,331 |
| Flow validation MSE | 0.4650 |
| Identity baseline MSE | 0.9281 |
| WA-JEPA candidate ranking AUC | 0.936 |
| LeWM candidate ranking AUC | 0.900 |
| Closed-loop grasp confirmation | 10/10 |
| Full closed-loop success | 0/10 |

All ten closed-loop episodes reached TRANSFER. The current bottleneck is goal-directed motion during TRANSFER, rather than stage recognition or initial grasping. This repository reports the failed full-task result because it defines the next experimental target.

## Repository layout

```text
configs/   Three-stage training and control configuration
docs/      Experiment summary and current limitations
src/       Encoder, Flow Predictor, Actor/Critic, MPC, and training code
```

Datasets, pretrained weights, generated latents, videos, and run outputs are intentionally excluded.

## Setup

1. Install the original LeWM_SAIMO environment and OGBench dependencies.
2. Clone the official WA-JEPA repository into `third_party/WA-JEPA`, or set `WA_JEPA_ROOT` to its location.
3. Place the robot dataset and checkpoints under `data/` and `artifacts/`, or override these paths with the environment variables documented below.
4. Adjust `configs/proprio_v2_config.yaml` for the local machine.

Optional path overrides:

```text
WA_JEPA_ROOT
WAJEPA_DATASET
VJEPA2_CHECKPOINT
WAJEPA_FLOW_CHECKPOINT
WAJEPA_PCA_CHECKPOINT
```

Train the robot video encoder:

```bash
python src/train_robot_vjepa_stage1.py --name stage1_robot
```

Train the 5-step action-conditioned Flow Predictor:

```bash
python src/train_wajepa_block_flow.py \
  --cache data/wajepa_cube_semantic_latents.npz \
  --source data/cube-single-play-v0.npz \
  --output runs/wajepa_block5_flow
```

Run frozen closed-loop evaluation:

```bash
python src/run_proprio_v2_experiment.py \
  --config configs/proprio_v2_config.yaml \
  --stage adapt --freeze-actor \
  --resume-checkpoint artifacts/proprio_v2_offline.pt
```

## Status

This is a research work in progress. The immediate next step is to diagnose whether TRANSFER failure is caused by Flow prediction, Critic ranking, or the latent goal representation, then repeat the same ten-seed evaluation.

## License

The LeWM-derived code is distributed under the MIT License. WA-JEPA is an external Apache-2.0 dependency; its license is included as `WA-JEPA-LICENSE`.
