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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("xarm")
@dataclass
class XArmConfig(RobotConfig):
    """Configuration for a UFACTORY xArm controlled in Cartesian end-effector space with an xArm Gripper."""

    # IP address of the xArm control box
    ip: str = "192.168.1.235"

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Speed and acceleration limits passed to every Cartesian target (xArm online trajectory planning mode).
    # Keep the speed above `end_effector_step_sizes * fps` so the arm can track the commanded targets.
    max_linear_speed_mm_s: float = 200.0
    max_linear_acc_mm_s2: float = 2000.0

    # Joint speed used to move to the home joint configuration on `reset_to_home`.
    home_joint_speed_deg_s: float = 30.0

    # xArm Gripper positions (SDK units) mapped to the normalized `gripper.pos` 0 (open) and 1 (closed).
    gripper_open_pos: int = 800
    gripper_close_pos: int = 0
    gripper_speed: int = 5000

    # Half-width of a uniform random xy offset (meters) added to the home pose on every reset, as
    # `RANDOM_RESET` in rail-berkeley/hil-serl (their tasks use 0.01-0.02). 0 disables it.
    random_xy_range_m: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.gripper_open_pos == self.gripper_close_pos:
            raise ValueError("`gripper_open_pos` and `gripper_close_pos` must differ.")
        if self.random_xy_range_m < 0:
            raise ValueError(f"`random_xy_range_m` must be >= 0, got {self.random_xy_range_m}.")
