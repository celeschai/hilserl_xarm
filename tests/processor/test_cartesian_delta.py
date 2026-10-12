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

import pytest
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.xarm.robot_cartesian_processor import CartesianDeltaToTargetStep, RobotActionToTensorStep
from lerobot.utils.constants import ACTION

STEP = 0.01
BOUNDS = {"min": [0.2, -0.2, 0.1], "max": [0.4, 0.2, 0.3]}
CLOSE, STAY, OPEN = 0, 1, 2


def _make_pipeline(delta_reference: str = "measured"):
    return RobotProcessorPipeline(
        steps=[
            MapDeltaActionToRobotActionStep(),
            CartesianDeltaToTargetStep(
                end_effector_step_sizes={"x": STEP, "y": STEP, "z": STEP},
                end_effector_bounds=BOUNDS,
                delta_reference=delta_reference,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _obs(x=0.3, y=0.0, z=0.2, gripper=0.0):
    return {"ee.x": x, "ee.y": y, "ee.z": z, "gripper.pos": gripper}


def _delta(dx=0.0, dy=0.0, dz=0.0, gripper=STAY):
    return {"delta_x": dx, "delta_y": dy, "delta_z": dz, "gripper": gripper}


def _xyz(action):
    return [action["ee.x"], action["ee.y"], action["ee.z"]]


def test_measured_reference_follows_the_arm():
    pipeline = _make_pipeline("measured")
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.31, 0.0, 0.2])
    # The arm has not moved yet: the target stays one step ahead of the measured pose.
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.31, 0.0, 0.2])
    assert _xyz(pipeline((_delta(dy=-0.5), _obs(x=0.305)))) == pytest.approx([0.305, -0.005, 0.2])


def test_commanded_reference_accumulates_targets():
    pipeline = _make_pipeline("commanded")
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.31, 0.0, 0.2])
    assert _xyz(pipeline((_delta(dx=1.0), _obs()))) == pytest.approx([0.32, 0.0, 0.2])


@pytest.mark.parametrize("delta_reference", ["measured", "commanded"])
def test_no_input_holds_last_target(delta_reference):
    pipeline = _make_pipeline(delta_reference)
    pipeline((_delta(dz=1.0), _obs()))
    # No input while the arm still moves: the last target is kept.
    assert _xyz(pipeline((_delta(), _obs(z=0.205)))) == pytest.approx([0.3, 0.0, 0.21])


def test_first_step_without_input_holds_measured_pose():
    pipeline = _make_pipeline()
    assert _xyz(pipeline((_delta(), _obs(x=0.25, y=0.1, z=0.15)))) == pytest.approx([0.25, 0.1, 0.15])


def test_target_is_clipped_to_bounds():
    pipeline = _make_pipeline("commanded")
    for _ in range(20):
        action = pipeline((_delta(dx=1.0, dz=-1.0), _obs(x=0.39, z=0.11)))
    assert _xyz(action) == pytest.approx([0.4, 0.0, 0.1])


def test_discrete_gripper_is_latched():
    pipeline = _make_pipeline()
    assert pipeline((_delta(gripper=STAY), _obs(gripper=0.02)))["gripper.pos"] == pytest.approx(0.02)
    assert pipeline((_delta(gripper=CLOSE), _obs()))["gripper.pos"] == 1.0
    # "Stay" keeps squeezing even though the fingers stopped half-way on an object.
    assert pipeline((_delta(gripper=STAY), _obs(gripper=0.6)))["gripper.pos"] == 1.0
    assert pipeline((_delta(gripper=OPEN), _obs(gripper=0.6)))["gripper.pos"] == 0.0


def test_reset_restarts_from_measured_pose():
    pipeline = _make_pipeline("commanded")
    pipeline((_delta(dx=1.0, gripper=CLOSE), _obs()))
    pipeline.reset()
    action = pipeline((_delta(), _obs(x=0.25, gripper=0.0)))
    assert _xyz(action) == pytest.approx([0.25, 0.0, 0.2])
    assert action["gripper.pos"] == 0.0


def test_invalid_delta_reference():
    with pytest.raises(ValueError):
        CartesianDeltaToTargetStep(
            end_effector_step_sizes={"x": STEP, "y": STEP, "z": STEP}, delta_reference="x"
        )


def test_missing_measured_pose_raises():
    with pytest.raises(ValueError):
        _make_pipeline()((_delta(), {"gripper.pos": 0.0}))


def test_cartesian_transform_features():
    step = CartesianDeltaToTargetStep(end_effector_step_sizes={"x": STEP, "y": STEP, "z": STEP})
    features = {
        PipelineFeatureType.ACTION: {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,))
            for key in ["enabled", "target_x", "target_y", "target_z", "gripper_vel"]
        }
    }
    out = step.transform_features(features)
    assert set(out[PipelineFeatureType.ACTION]) == {"ee.x", "ee.y", "ee.z", "gripper.pos"}


def test_robot_action_to_tensor_order():
    step = RobotActionToTensorStep(action_keys=["ee.x", "ee.y", "ee.z", "gripper.pos"])
    tensor = step.action({"gripper.pos": 1.0, "ee.z": 0.3, "ee.x": 0.1, "ee.y": 0.2})
    assert torch.equal(tensor, torch.tensor([0.1, 0.2, 0.3, 1.0]))

    with pytest.raises(ValueError):
        step.action({"ee.x": 0.1})

    out = step.transform_features({PipelineFeatureType.ACTION: {}})
    assert out[PipelineFeatureType.ACTION][ACTION].shape == (4,)
