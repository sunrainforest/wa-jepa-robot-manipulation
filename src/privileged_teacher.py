#教师模型，生成最终rgb

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


XYZ_CENTER = np.asarray([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALE = 10.0
ACTION_RANGE = np.asarray([0.05, 0.05, 0.05, 0.30, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class SemanticState:
    effector: np.ndarray
    effector_yaw: float
    gripper_opening: float
    contact: float
    cube: np.ndarray
    cube_yaw: float
    goal: np.ndarray
    goal_yaw: float


def _position(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32, copy=False) / XYZ_SCALE + XYZ_CENTER


def _yaw(values: np.ndarray) -> float:
    return float(np.arctan2(values[1], values[0]))


def _yaw_delta(target: float, current: float) -> float:
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


def semantic_state(state: np.ndarray, goal_state: np.ndarray) -> SemanticState:
    state = np.asarray(state, dtype=np.float32)
    goal_state = np.asarray(goal_state, dtype=np.float32)
    if state.shape[-1] < 28 or goal_state.shape[-1] < 28:
        raise ValueError("Cube state and goal must contain at least 28 values.")
    return SemanticState(
        effector=_position(state[12:15]),
        effector_yaw=_yaw(state[15:17]),
        gripper_opening=float(state[17] / 3.0),
        contact=float(state[18]),
        cube=_position(state[19:22]),
        cube_yaw=_yaw(state[26:28]),
        goal=_position(goal_state[19:22]),
        goal_yaw=_yaw(goal_state[26:28]),
    )


def detect_phase(value: SemanticState, settings: dict) -> str:
    if np.linalg.norm(value.cube - value.goal) <= float(
        settings.get("goal_xyz_m", 0.04)
    ):
        return "complete"

    xy_to_cube = float(np.linalg.norm(value.effector[:2] - value.cube[:2]))
    xyz_to_cube = float(np.linalg.norm(value.effector - value.cube))
    grasped = value.contact >= float(settings.get("contact_threshold", 0.5))
    goal_xy = float(np.linalg.norm(value.cube[:2] - value.goal[:2]))

    if not grasped:
        if xy_to_cube > float(settings.get("xy_alignment_m", 0.04)):
            return "approach"
        if xyz_to_cube > float(settings.get("xyz_alignment_m", 0.025)):
            return "descend"
        return "grasp"
    if goal_xy <= float(settings.get("goal_xy_m", 0.04)):
        return "place"
    if value.cube[2] < float(settings.get("transport_height_m", 0.14)):
        return "lift"
    return "transport"


class PrivilegedTeacher:
    """Coordinate controller used only for offline labels and reference rendering."""

    def action(
        self,
        state: np.ndarray,
        goal_state: np.ndarray,
        settings: dict,
    ) -> tuple[np.ndarray, str]:
        value = semantic_state(state, goal_state)
        phase = detect_phase(value, settings)
        action = np.zeros(5, dtype=np.float32)
        action[4] = -1.0

        if phase == "complete":
            return action, phase

        if phase == "approach":
            target = value.cube.copy()
            target[2] += float(settings.get("above_offset_m", 0.16))
        elif phase == "descend":
            target = value.cube.copy()
        elif phase == "grasp":
            target = value.cube.copy()
            action[4] = 1.0
        elif phase == "lift":
            target = value.effector.copy()
            target[2] = float(settings.get("lift_target_height_m", 0.36))
            action[4] = 1.0
        elif phase == "transport":
            target = value.goal.copy()
            target[2] += float(settings.get("above_offset_m", 0.16))
            action[4] = 1.0
        else:
            target = value.goal.copy()
            action[4] = 1.0

        action[:3] = np.clip(
            (target - value.effector) / ACTION_RANGE[:3], -1.0, 1.0
        )
        yaw_target = (
            value.goal_yaw
            if phase in ("transport", "place")
            else value.cube_yaw
        )
        action[3] = np.clip(
            _yaw_delta(yaw_target, value.effector_yaw) / ACTION_RANGE[3],
            -1.0,
            1.0,
        )
        return action, phase
