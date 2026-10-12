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

from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("xarm", reason="xarm-python-sdk is required (install lerobot[xarm])")

from lerobot.robots.xarm import XArm, XArmConfig

_MODULE = "lerobot.robots.xarm.xarm"
HOME_JOINTS = [0.0, 0.0, 0.0, 90.0, 0.0, 90.0, 0.0]


def _make_arm_mock() -> MagicMock:
    """Return an `XArmAPI` mock with the attributes and return codes used by the robot."""
    arm = MagicMock(name="XArmAPIMock")
    arm.connected = True
    arm.error_code = 0
    arm.axis = 7
    arm.mode = 0

    def _set_mode(mode):
        arm.mode = mode
        return 0

    arm.set_mode.side_effect = _set_mode
    for api in [
        "set_state",
        "clean_error",
        "clean_warn",
        "motion_enable",
        "set_gripper_enable",
        "set_gripper_mode",
        "set_gripper_speed",
        "set_position",
        "set_servo_angle",
        "set_gripper_position",
    ]:
        getattr(arm, api).return_value = 0
    arm.get_position.return_value = (0, [300.0, -50.0, 200.0, 180.0, 0.0, 90.0])
    arm.get_gripper_position.return_value = (0, 800)
    arm.get_servo_angle.return_value = (0, [*HOME_JOINTS, 0.0])
    return arm


@pytest.fixture
def arm():
    return _make_arm_mock()


@pytest.fixture
def make_robot(arm, tmp_path):
    robots = []

    def _make(**config_kwargs) -> XArm:
        robot = XArm(XArmConfig(calibration_dir=tmp_path, **config_kwargs))
        robots.append(robot)
        return robot

    with patch(f"{_MODULE}.XArmAPI", return_value=arm):
        yield _make
    for robot in robots:
        if robot.is_connected:
            robot.disconnect()


@pytest.fixture
def robot(make_robot):
    robot = make_robot()
    robot.connect()
    return robot


def test_features():
    robot = XArm(XArmConfig())
    expected = {"ee.x": float, "ee.y": float, "ee.z": float, "gripper.pos": float}
    assert robot.action_features == expected
    assert robot.observation_features == expected


def test_connect_streams_in_cartesian_mode(robot, arm):
    assert robot.is_connected
    assert arm.mode == 7
    arm.set_gripper_enable.assert_called_once_with(True)


def test_connect_failure(make_robot, arm):
    arm.connected = False
    robot = make_robot()
    with pytest.raises(ConnectionError):
        robot.connect()
    assert not robot.is_connected


def test_get_observation_in_meters_and_normalized_gripper(robot, arm):
    obs = robot.get_observation()
    assert obs["ee.x"] == pytest.approx(0.3)
    assert obs["ee.y"] == pytest.approx(-0.05)
    assert obs["ee.z"] == pytest.approx(0.2)
    assert obs["gripper.pos"] == pytest.approx(0.0)  # 800 = open

    arm.get_gripper_position.return_value = (0, 0)
    assert robot.get_observation()["gripper.pos"] == pytest.approx(1.0)  # 0 = closed


def test_gripper_fault_reuses_last_reading(robot, arm):
    arm.get_gripper_position.return_value = (0, 400)
    assert robot.get_observation()["gripper.pos"] == pytest.approx(0.5)
    # The SDK returns a bare error code when the gripper has a fault.
    arm.get_gripper_position.return_value = 23
    assert robot.get_observation()["gripper.pos"] == pytest.approx(0.5)


def test_send_action_holds_orientation(robot, arm):
    sent = robot.send_action({"ee.x": 0.31, "ee.y": -0.05, "ee.z": 0.25, "gripper.pos": 0.0})

    args, kwargs = arm.set_position.call_args
    assert args == pytest.approx((310.0, -50.0, 250.0, 180.0, 0.0, 90.0))
    assert kwargs == {"speed": 200.0, "mvacc": 2000.0, "is_radian": False, "wait": False}
    assert sent == {"ee.x": 0.31, "ee.y": -0.05, "ee.z": 0.25, "gripper.pos": 0.0}


def test_gripper_only_commanded_on_change(robot, arm):
    action = {"ee.x": 0.3, "ee.y": 0.0, "ee.z": 0.2, "gripper.pos": 1.0}
    robot.send_action(action)
    robot.send_action(action)
    arm.set_gripper_position.assert_called_once_with(0.0, wait=False, wait_motion=False)

    robot.send_action({**action, "gripper.pos": 0.0})
    assert arm.set_gripper_position.call_args == call(800.0, wait=False, wait_motion=False)


def test_fault_blocks_motion(robot, arm):
    arm.error_code = 31
    assert robot.has_fault
    assert "31" in robot.fault_description

    robot.send_action({"ee.x": 0.3, "ee.y": 0.0, "ee.z": 0.2, "gripper.pos": 1.0})
    arm.set_position.assert_not_called()
    arm.set_gripper_position.assert_not_called()


def test_reset_to_home(make_robot, arm):
    robot = make_robot(random_xy_range_m=0.02)
    robot.connect()
    arm.reset_mock()
    arm.mode = 7

    robot.reset_to_home(HOME_JOINTS)

    names = [name for name, _, _ in arm.method_calls]
    order = ["clean_error", "motion_enable", "set_servo_angle", "set_position", "set_gripper_position"]
    assert [name for name in names if name in order] == order
    assert arm.set_mode.call_args_list == [call(0), call(7)]

    _, kwargs = arm.set_servo_angle.call_args
    assert kwargs["angle"] == HOME_JOINTS
    assert kwargs["wait"] is True

    _, kwargs = arm.set_position.call_args
    assert kwargs["relative"] is True and kwargs["wait"] is True
    assert abs(kwargs["x"]) <= 20.0 and abs(kwargs["y"]) <= 20.0
    assert kwargs["z"] == 0

    arm.set_gripper_position.assert_called_once_with(800, wait=True)


def test_reset_to_home_without_joints_stays_in_place(robot, arm):
    robot.reset_to_home(None)
    arm.set_servo_angle.assert_not_called()
    arm.set_position.assert_not_called()
    assert arm.mode == 7


def test_get_joint_positions(robot):
    assert robot.get_joint_positions() == HOME_JOINTS
