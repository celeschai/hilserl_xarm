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
#
# The connection sequence and gripper normalization follow UFACTORY's `lerobot_robot_ufactory`
# (https://github.com/xArm-Developer/lerobot_robot_ufactory, Copyright 2025 UFACTORY Inc., Apache-2.0),
# reimplemented on the public xArm SDK API.

import logging
import time
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras import DepthCamera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _xarm_available, require_package

from ..robot import Robot
from .config_xarm import XArmConfig

if TYPE_CHECKING or _xarm_available:
    from xarm.core.config.x_code import ControllerError
    from xarm.wrapper import XArmAPI

logger = logging.getLogger(__name__)

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
GRIPPER_KEY = "gripper.pos"

# xArm controller modes and states, see the xArm SDK documentation.
POSITION_MODE = 0
ONLINE_CARTESIAN_MODE = 7
READY_STATE = 0

MM_PER_M = 1000.0
_MODE_SWITCH_TIMEOUT_S = 2.0


class XArm(Robot):
    """
    UFACTORY xArm with an xArm Gripper, controlled in Cartesian end-effector space.

    Observations are the measured TCP position in the robot base frame (meters) and the normalized gripper
    opening (0 = open, 1 = closed). Actions are absolute TCP position targets plus a gripper target; the
    controller's online trajectory planning mode (mode 7) does the inverse kinematics. The TCP orientation is
    held at the orientation measured at connection / the last reset.

    Controller-side safety (safety boundary, collision sensitivity, self-collision detection, TCP offset and
    payload) is left to UFACTORY Studio. Controller errors are reported through `has_fault`; `reset_to_home`
    clears them.
    """

    config_class = XArmConfig
    name = "xarm"

    def __init__(self, config: XArmConfig):
        require_package("xarm-python-sdk", extra="xarm", import_name="xarm")
        super().__init__(config)
        self.config = config
        self.arm: XArmAPI | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._rng = np.random.default_rng()
        self._hold_rpy_deg: list[float] = [180.0, 0.0, 0.0]
        self._last_gripper_target: float | None = None
        self._last_gripper_obs = 0.0

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if isinstance(cam, DepthCamera) and cam.use_depth:
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self.action_features, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**dict.fromkeys(EE_POSITION_KEYS, float), GRIPPER_KEY: float}

    @property
    def is_connected(self) -> bool:
        return (
            self.arm is not None
            and self.arm.connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def _api(self) -> "XArmAPI":
        """The xArm SDK handle, only available while connected."""
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected. Run `.connect()` first.")
        return self.arm

    @property
    def has_fault(self) -> bool:
        """Whether the controller reports an error. Motion commands are ignored until `reset_to_home`."""
        return self.arm is not None and self.arm.error_code != 0

    @property
    def fault_description(self) -> str:
        if not self.has_fault:
            return ""
        code = self._api.error_code
        error = ControllerError(code)
        return f"xArm error {code}: {error.title['en']}. {error.description['en']}"

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        arm = XArmAPI(self.config.ip, is_radian=False)
        if not arm.connected:
            raise ConnectionError(f"Could not connect to the xArm at {self.config.ip}.")
        self.arm = arm

        self._enable_motion()
        self._check(self._api.set_gripper_enable(True), "set_gripper_enable")
        self._check(self._api.set_gripper_mode(0), "set_gripper_mode")
        self._check(self._api.set_gripper_speed(self.config.gripper_speed), "set_gripper_speed")
        self._hold_current_orientation()
        self._set_mode(ONLINE_CARTESIAN_MODE)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def reset_to_home(self, joints_deg: Sequence[float] | None = None) -> None:
        """
        Clear any controller error and move (blocking) to the home joint configuration, then open the gripper.

        A joint-space home keeps the redundant joints of the arm from drifting across episodes. When
        `random_xy_range_m` is set, a uniform random xy offset is then added to the home TCP position.

        Args:
            joints_deg: Home joint angles in degrees. If `None`, the arm stays where it is.
        """
        self._enable_motion()
        self._set_mode(POSITION_MODE)

        if joints_deg is not None:
            code = self._api.set_servo_angle(
                angle=list(joints_deg),
                speed=self.config.home_joint_speed_deg_s,
                is_radian=False,
                wait=True,
            )
            self._check(code, "set_servo_angle")

        if self.config.random_xy_range_m > 0:
            dx, dy = self._rng.uniform(-self.config.random_xy_range_m, self.config.random_xy_range_m, size=2)
            code = self._api.set_position(
                x=dx * MM_PER_M,
                y=dy * MM_PER_M,
                z=0,
                roll=0,
                pitch=0,
                yaw=0,
                relative=True,
                speed=self.config.max_linear_speed_mm_s,
                is_radian=False,
                wait=True,
            )
            self._check(code, "set_position")

        self._check(
            self._api.set_gripper_position(self.config.gripper_open_pos, wait=True), "set_gripper_position"
        )
        self._last_gripper_target = 0.0
        self._hold_current_orientation()
        self._set_mode(ONLINE_CARTESIAN_MODE)

    @check_if_not_connected
    def get_joint_positions(self) -> list[float]:
        """Current joint angles in degrees, e.g. to save as the home configuration."""
        code, angles = self._api.get_servo_angle(is_radian=False)
        self._check(code, "get_servo_angle")
        return [float(a) for a in angles[: self._api.axis]]

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        code, pose = self._api.get_position(is_radian=False)
        self._check(code, "get_position")
        obs_dict: RobotObservation = {key: pose[i] / MM_PER_M for i, key in enumerate(EE_POSITION_KEYS)}
        obs_dict[GRIPPER_KEY] = self._read_gripper()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                obs_dict[cam_key] = cam.read_latest()
            if isinstance(cam, DepthCamera) and cam.use_depth:
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Command an absolute TCP position target (meters, base frame) and a gripper target in [0, 1].

        Nothing is sent while the controller reports an error. The gripper is only commanded when its target
        changes, so the arm streaming is not slowed down by the gripper's Modbus link.

        Returns:
            RobotAction: the action actually sent (gripper clipped to [0, 1]).
        """
        target_m = [float(action[key]) for key in EE_POSITION_KEYS]
        gripper = float(np.clip(action[GRIPPER_KEY], 0.0, 1.0))
        sent_action: RobotAction = {
            **dict(zip(EE_POSITION_KEYS, target_m, strict=True)),
            GRIPPER_KEY: gripper,
        }

        if self.has_fault:
            logger.debug(f"{self} ignores the action: {self.fault_description}")
            return sent_action

        code = self._api.set_position(
            *(t * MM_PER_M for t in target_m),
            *self._hold_rpy_deg,
            speed=self.config.max_linear_speed_mm_s,
            mvacc=self.config.max_linear_acc_mm_s2,
            is_radian=False,
            wait=False,
        )
        if code != 0:
            logger.warning(f"{self} set_position returned code {code}")

        if self._last_gripper_target is None or abs(gripper - self._last_gripper_target) > 1e-3:
            # `wait_motion=False`: by default the SDK waits for the arm to stop before moving the gripper.
            code = self._api.set_gripper_position(
                self._normalized_to_gripper(gripper), wait=False, wait_motion=False
            )
            if code == 0:
                self._last_gripper_target = gripper
            else:
                logger.warning(f"{self} set_gripper_position returned code {code}")

        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._api.set_mode(POSITION_MODE)
        self._api.set_state(READY_STATE)
        self._api.disconnect()
        self.arm = None
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _check(self, code: int, api: str) -> None:
        if code != 0:
            raise RuntimeError(f"{self}: xArm `{api}` failed with code {code}. {self.fault_description}")

    def _enable_motion(self) -> None:
        self._api.clean_error()
        self._api.clean_warn()
        self._check(self._api.motion_enable(enable=True), "motion_enable")

    def _set_mode(self, mode: int) -> None:
        self._check(self._api.set_mode(mode), "set_mode")
        self._check(self._api.set_state(READY_STATE), "set_state")
        # The new mode is only active once the controller reports it.
        deadline = time.perf_counter() + _MODE_SWITCH_TIMEOUT_S
        while self._api.mode != mode:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"{self}: xArm did not switch to mode {mode}. {self.fault_description}")
            time.sleep(0.01)

    def _hold_current_orientation(self) -> None:
        code, pose = self._api.get_position(is_radian=False)
        self._check(code, "get_position")
        self._hold_rpy_deg = [float(angle) for angle in pose[3:6]]

    def _read_gripper(self) -> float:
        ret = self._api.get_gripper_position()
        # The SDK returns a bare error code instead of a `(code, position)` tuple when the gripper has a fault.
        code, position = ret if isinstance(ret, tuple) else (ret, None)
        if code != 0 or position is None:
            logger.warning(f"{self} could not read the gripper (code {code}), reusing the last reading.")
            return self._last_gripper_obs
        self._last_gripper_obs = self._gripper_to_normalized(position)
        return self._last_gripper_obs

    def _gripper_to_normalized(self, position: float) -> float:
        open_pos, close_pos = self.config.gripper_open_pos, self.config.gripper_close_pos
        return float(np.clip((position - open_pos) / (close_pos - open_pos), 0.0, 1.0))

    def _normalized_to_gripper(self, value: float) -> float:
        open_pos, close_pos = self.config.gripper_open_pos, self.config.gripper_close_pos
        return open_pos + value * (close_pos - open_pos)
