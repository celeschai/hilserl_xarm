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

"""`gym_manipulator` with a robot that has no motor bus and takes Cartesian targets (e.g. the xArm)."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.envs.configs import (
    GripperConfig,
    HILSerlProcessorConfig,
    HILSerlRobotEnvConfig,
    InverseKinematicsConfig,
    ResetConfig,
)
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    RobotEnv,
    make_processors,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from tests.mocks.mock_robot import MockRobot, MockRobotConfig

STATE_KEYS = ["ee.x", "ee.y", "ee.z", "gripper.pos"]
HOME_JOINTS = [0.0, 0.0, 0.0, 90.0, 0.0, 90.0, 0.0]


class FakeCartesianRobot:
    """Bus-less robot with the interface of `XArm`: Cartesian state/targets, `reset_to_home`, `has_fault`."""

    name = "fake_cartesian"

    def __init__(self):
        self.is_connected = True
        self.cameras = {"wrist": None}
        self.observation_features = {**dict.fromkeys(STATE_KEYS, float), "wrist": (8, 8, 3)}
        self.action_features = dict.fromkeys(STATE_KEYS, float)
        self.state = {"ee.x": 0.3, "ee.y": 0.0, "ee.z": 0.2, "gripper.pos": 0.0}
        self.has_fault = False
        self.fault_description = ""
        self.sent: list[dict] = []
        self.home_calls: list = []

    def get_observation(self):
        return {**self.state, "wrist": np.zeros((8, 8, 3), dtype=np.uint8)}

    def send_action(self, action):
        action = {key: float(value) for key, value in action.items()}
        self.sent.append(action)
        return action

    def reset_to_home(self, joints_deg):
        self.home_calls.append(joints_deg)

    def disconnect(self):
        self.is_connected = False


class FakeTeleop:
    """SpaceMouse-like teleop: delta action + HIL events."""

    def __init__(self):
        self.action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1}
        self.intervention = False

    def get_action(self):
        return dict(self.action)

    def get_teleop_events(self):
        return {
            TeleopEvents.IS_INTERVENTION: self.intervention,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }


@pytest.fixture(autouse=True)
def _no_sounds():
    with patch("lerobot.rl.gym_manipulator.log_say"):
        yield


@pytest.fixture
def robot():
    return FakeCartesianRobot()


@pytest.fixture
def env(robot):
    return RobotEnv(robot=robot, use_gripper=True, reset_pose=HOME_JOINTS, reset_time_s=0.0)


def test_env_state_from_robot_features(env, robot):
    obs, info = env.reset()
    assert robot.home_calls == [HOME_JOINTS]
    np.testing.assert_allclose(obs["agent_pos"], [0.3, 0.0, 0.2, 0.0])
    assert list(obs["pixels"]) == ["wrist"]
    assert list(env.get_raw_joint_positions()) == STATE_KEYS
    assert env.observation_space["observation.state"].shape == (4,)
    assert info == {TeleopEvents.IS_INTERVENTION: False}


def test_env_step_sends_action_by_feature_order(env, robot):
    env.reset()
    _, _, terminated, truncated, info = env.step(torch.tensor([0.31, 0.0, 0.2, 1.0]))
    assert robot.sent[-1] == pytest.approx({"ee.x": 0.31, "ee.y": 0.0, "ee.z": 0.2, "gripper.pos": 1.0})
    assert not terminated and not truncated
    assert "robot_fault" not in info


def test_robot_fault_truncates_episode(env, robot):
    env.reset()
    robot.has_fault = True
    _, _, terminated, truncated, info = env.step(torch.tensor([0.3, 0.0, 0.2, 0.0]))
    assert truncated and not terminated
    assert info["robot_fault"] is True


def test_motor_bus_robot_unchanged():
    robot = MockRobot(MockRobotConfig(n_motors=3, random_values=False, static_values=[1.0, 2.0, 3.0]))
    env = RobotEnv(robot=robot, use_gripper=False, reset_time_s=0.0)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["agent_pos"], [1.0, 2.0, 3.0])
    assert list(env.get_raw_joint_positions()) == ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
    with patch.object(robot, "send_action", wraps=robot.send_action) as send_action:
        env.step(torch.tensor([4.0, 5.0, 6.0]))
    sent = send_action.call_args.args[0]
    assert list(sent) == ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
    assert [float(v) for v in sent.values()] == [4.0, 5.0, 6.0]


def _cartesian_env_cfg(delta_reference: str = "measured") -> HILSerlRobotEnvConfig:
    return HILSerlRobotEnvConfig(
        fps=10,
        processor=HILSerlProcessorConfig(
            gripper=GripperConfig(use_gripper=True, gripper_penalty=-0.02),
            reset=ResetConfig(reset_time_s=0.0, control_time_s=1.0),
            inverse_kinematics=InverseKinematicsConfig(
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
                end_effector_bounds={"min": [0.2, -0.2, 0.1], "max": [0.4, 0.2, 0.3]},
                delta_reference=delta_reference,
            ),
            max_gripper_pos=1.0,
        ),
    )


def test_cartesian_pipeline_policy_and_intervention(env, robot):
    teleop = FakeTeleop()
    env_processor, action_processor = make_processors(env, teleop, _cartesian_env_cfg(), device="cpu")
    transition = reset_and_build_transition(env, env_processor, action_processor)
    assert transition[TransitionKey.OBSERVATION]["observation.state"].shape == (1, 4)

    # Policy acts: +x at full scale, gripper "stay".
    transition = step_env_and_process_transition(
        env, transition, torch.tensor([1.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    assert robot.sent[-1] == pytest.approx({"ee.x": 0.31, "ee.y": 0.0, "ee.z": 0.2, "gripper.pos": 0.0})
    assert not transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]

    # Human intervenes: +y and close the gripper. The executed (teleop) delta is what HIL-SERL stores.
    teleop.intervention = True
    teleop.action = {"delta_x": 0.0, "delta_y": 1.0, "delta_z": 0.0, "gripper": 0}
    transition = step_env_and_process_transition(
        env, transition, torch.tensor([1.0, 0.0, 0.0, 1.0]), env_processor, action_processor
    )
    assert robot.sent[-1] == pytest.approx({"ee.x": 0.3, "ee.y": 0.01, "ee.z": 0.2, "gripper.pos": 1.0})
    assert transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]
    torch.testing.assert_close(
        transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"], torch.tensor([0.0, 1.0, 0.0, 0.0])
    )
    # Closing an open gripper costs the grasp penalty.
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"] == pytest.approx(-0.02)


def test_cartesian_pipeline_time_limit(env, robot):
    env_processor, action_processor = make_processors(env, FakeTeleop(), _cartesian_env_cfg(), device="cpu")
    transition = reset_and_build_transition(env, env_processor, action_processor)
    for _ in range(10):  # control_time_s * fps
        transition = step_env_and_process_transition(
            env, transition, torch.tensor([0.0, 0.0, 0.0, 1.0]), env_processor, action_processor
        )
    assert transition[TransitionKey.TRUNCATED]
