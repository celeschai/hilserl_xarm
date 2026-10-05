#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ..config import TeleoperatorConfig

SPACE_MOUSE_AXES = ("x", "y", "z")


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    # Axis readings below this magnitude (in [0, 1]) are treated as zero.
    deadzone: float = 0.05
    # Multiplier applied to the axis readings before clipping deltas to [-1, 1].
    scale: float = 1.0
    # SpaceMouse axis driving each of (delta_x, delta_y, delta_z). Prefix with "-" to invert, e.g.
    # ["y", "-x", "z"]. pyspacemouse axes: x = right, y = forward (away from the cable), z = up.
    axis_mapping: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    # Index of the device to open when several SpaceMice are connected.
    device_index: int = 0

    def __post_init__(self):
        if len(self.axis_mapping) != 3:
            raise ValueError(f"`axis_mapping` must have 3 entries, got {self.axis_mapping}.")
        for axis in self.axis_mapping:
            if axis.removeprefix("-") not in SPACE_MOUSE_AXES:
                raise ValueError(
                    f"Invalid axis '{axis}' in `axis_mapping`, expected one of {SPACE_MOUSE_AXES} "
                    "optionally prefixed with '-'."
                )
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError(f"`deadzone` must be in [0, 1), got {self.deadzone}.")
