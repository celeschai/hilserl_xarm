#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processor steps for robots whose controller tracks Cartesian end-effector targets natively (no URDF IK)."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    PolicyAction,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.constants import ACTION

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
GRIPPER_POS_KEY = "gripper.pos"

# Discrete gripper commands, see `GamepadTeleop.GripperAction` and `SpaceMouseTeleop`.
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 2


@ProcessorStepRegistry.register("cartesian_delta_to_target")
@dataclass
class CartesianDeltaToTargetStep(RobotActionProcessorStep):
    """
    Turns a normalized end-effector delta into an absolute Cartesian target for a robot that does its own IK.

    Takes the output of `MapDeltaActionToRobotActionStep` and plays the role of `EEReferenceAndDelta` +
    `EEBoundsAndSafety` + `GripperVelocityToJoint` for robots whose observation already holds the measured
    end-effector position (`ee.x`, `ee.y`, `ee.z`) and the normalized gripper opening (`gripper.pos`).

    The target position is `reference + delta * end_effector_step_sizes`, clipped to `end_effector_bounds`
    (the safety box of rail-berkeley/hil-serl). While there is no input (`enabled` is false) the last target
    is held.

    Attributes:
        end_effector_step_sizes: Displacement for a full-scale delta, per axis ("x", "y", "z").
        end_effector_bounds: Optional "min" / "max" xyz lists the target is clipped to.
        delta_reference: Pose the delta is applied to.
            "measured": the measured end-effector position, as in rail-berkeley/hil-serl. The target never
                leads the arm by more than one step, so the arm stops right away when the input stops.
            "commanded": the previous target, as `EEReferenceAndDelta(use_ik_solution=True)` does in the SO
                pipeline. Motion runs at exactly `step size * fps` but the target can run ahead of an arm
                that lags or is blocked.
        discrete_gripper: Map the discrete gripper command {0 = close, 1 = stay, 2 = open} to a latched
            binary target (1 = closed, 0 = open). "Stay" keeps the last target, so a grasp keeps its force.
    """

    end_effector_step_sizes: dict[str, float]
    end_effector_bounds: dict[str, list[float]] | None = None
    delta_reference: Literal["measured", "commanded"] = "measured"
    discrete_gripper: bool = True
    _last_target: np.ndarray | None = field(default=None, init=False, repr=False)
    _gripper_target: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.delta_reference not in ("measured", "commanded"):
            raise ValueError(
                f"`delta_reference` must be 'measured' or 'commanded', got '{self.delta_reference}'."
            )
        if not self.discrete_gripper:
            raise NotImplementedError("Only the discrete gripper command is supported.")

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None or any(key not in observation for key in EE_POSITION_KEYS):
            raise ValueError(
                f"The observation must contain the measured end-effector position {EE_POSITION_KEYS}."
            )
        measured = np.array([float(observation[key]) for key in EE_POSITION_KEYS])

        enabled = bool(action.pop("enabled"))
        delta = np.array([float(action.pop(f"target_{axis}")) for axis in "xyz"])
        for axis in "xyz":
            # Rotation deltas are not supported, the robot holds its orientation.
            action.pop(f"target_w{axis}", None)
        gripper_command = round(float(action.pop("gripper_vel")))

        if enabled or self._last_target is None:
            reference = measured
            if self.delta_reference == "commanded" and self._last_target is not None:
                reference = self._last_target
            step_sizes = np.array([self.end_effector_step_sizes[axis] for axis in "xyz"])
            target = reference + (delta if enabled else 0.0) * step_sizes
            if self.end_effector_bounds is not None:
                target = np.clip(target, self.end_effector_bounds["min"], self.end_effector_bounds["max"])
            self._last_target = target

        if gripper_command == GRIPPER_CLOSE:
            self._gripper_target = 1.0
        elif gripper_command == GRIPPER_OPEN:
            self._gripper_target = 0.0
        elif self._gripper_target is None:
            self._gripper_target = float(observation.get(GRIPPER_POS_KEY, 0.0))

        return {
            **{key: float(value) for key, value in zip(EE_POSITION_KEYS, self._last_target, strict=True)},
            GRIPPER_POS_KEY: self._gripper_target,
        }

    def reset(self):
        self._last_target = None
        self._gripper_target = None

    def get_config(self) -> dict[str, Any]:
        return {
            "end_effector_step_sizes": self.end_effector_step_sizes,
            "end_effector_bounds": self.end_effector_bounds,
            "delta_reference": self.delta_reference,
            "discrete_gripper": self.discrete_gripper,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        features[PipelineFeatureType.ACTION].pop("gripper_vel", None)
        for key in (*EE_POSITION_KEYS, GRIPPER_POS_KEY):
            features[PipelineFeatureType.ACTION][key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@ProcessorStepRegistry.register("robot_action_to_tensor")
@dataclass
class RobotActionToTensorStep(ActionProcessorStep):
    """
    Stacks a robot action dictionary into a tensor, in the order of `action_keys`.

    The counterpart of `RobotActionToPolicyActionProcessorStep` for robots whose action keys are not
    `{motor}.pos` (e.g. `ee.x`). Typically `action_keys = list(robot.action_features)`.
    """

    action_keys: list[str]

    def action(self, action: RobotAction) -> PolicyAction:
        missing = [key for key in self.action_keys if key not in action]
        if missing:
            raise ValueError(f"Action is missing the keys {missing}.")
        return torch.tensor([float(action[key]) for key in self.action_keys], dtype=torch.float32)

    def get_config(self) -> dict[str, Any]:
        return {"action_keys": self.action_keys}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.action_keys),)
        )
        return features
