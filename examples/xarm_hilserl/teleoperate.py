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

"""
Teleoperate an xArm with a SpaceMouse in end-effector space, and find the values for the HIL-SERL configs.

The SpaceMouse deltas go through the same processor steps as in HIL-SERL (`MapDeltaActionToRobotActionStep` +
`CartesianDeltaToTargetStep`), so the arm moves exactly as it will when recording demos and intervening.
Jog the TCP around the region the task needs and finish at the pose every episode should start from. On Ctrl+C
the script prints the config values it found: the visited `end_effector_bounds`, the current joints as
`fixed_reset_joint_positions` and the matching `policy.dataset_stats`. With `--write_configs=true` it then offers to
write them into `configs/env.json` and `configs/train.json`.

Example:

```shell
uv run python examples/xarm_hilserl/teleoperate.py \
    --robot.ip=192.168.1.235 \
    --robot.cameras='{wrist: {type: intelrealsense, serial_number_or_name: "<serial>", width: 640, height: 480, fps: 30}}' \
    --teleop.axis_mapping='[x, y, z]' \
    --end_effector_step_sizes='{x: 0.01, y: 0.01, z: 0.01}' \
    --write_configs=true
```
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.xarm import XArm, XArmConfig
from lerobot.robots.xarm.robot_cartesian_processor import EE_POSITION_KEYS, CartesianDeltaToTargetStep
from lerobot.teleoperators.space_mouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class XArmTeleoperateConfig:
    robot: XArmConfig = field(default_factory=XArmConfig)
    teleop: SpaceMouseTeleopConfig = field(default_factory=SpaceMouseTeleopConfig)
    fps: int = 10
    # TCP displacement (meters) for a full SpaceMouse deflection, per control step. Use the same values in
    # `env.processor.inverse_kinematics.end_effector_step_sizes` and keep them fixed once demos are recorded.
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.01, "y": 0.01, "z": 0.01}
    )
    # Safety box (meters, robot base frame). If unset, a box of +/- `box_m` around the start position is used.
    end_effector_bounds: dict[str, list[float]] | None = None
    box_m: float = 0.30
    # "measured" (hil-serl) or "commanded" (lerobot SO pipeline), see `CartesianDeltaToTargetStep`.
    delta_reference: str = "measured"
    # Joint angles (degrees) to move to before teleoperating, e.g. your `fixed_reset_joint_positions`.
    home_joints_deg: list[float] | None = None
    display_data: bool = False
    # On exit, offer to write the values found into the env and train configs (asks for confirmation).
    write_configs: bool = False
    env_config_path: Path = CONFIGS_DIR / "env.json"
    train_config_path: Path = CONFIGS_DIR / "train.json"

    def __post_init__(self):
        if not self.teleop.use_gripper:
            raise ValueError("`teleop.use_gripper` must be true: HIL-SERL uses the discrete gripper action.")


def find_config_values(
    cfg: XArmTeleoperateConfig, robot: XArm, visited_min: np.ndarray, visited_max: np.ndarray
) -> tuple[dict, list[float], dict]:
    """Config values from this session: `inverse_kinematics`, home joints and `policy.dataset_stats`."""
    step = np.array([cfg.end_effector_step_sizes[axis] for axis in "xyz"])
    max_velocity = (step * cfg.fps).round(4).tolist()  # TCP velocity at full deflection, m/s
    ee_min, ee_max = visited_min.round(4).tolist(), visited_max.round(4).tolist()
    inverse_kinematics = {
        "urdf_path": None,
        "end_effector_bounds": {"min": ee_min, "max": ee_max},
        "end_effector_step_sizes": cfg.end_effector_step_sizes,
        "delta_reference": cfg.delta_reference,
    }
    home_joints = [round(a, 2) for a in robot.get_joint_positions()]
    # State = [ee.x, ee.y, ee.z, gripper.pos] followed by their velocities (add_joint_velocity_to_observation).
    dataset_stats = {
        **{f"observation.images.{key}": IMAGENET_STATS for key in robot.cameras},
        "observation.state": {
            "min": [*ee_min, 0.0, *(-v for v in max_velocity), -float(cfg.fps)],
            "max": [*ee_max, 1.0, *max_velocity, float(cfg.fps)],
        },
        "action": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
    }
    return inverse_kinematics, home_joints, dataset_stats


def write_config(path: Path, inverse_kinematics: dict, home_joints: list[float], dataset_stats: dict) -> None:
    """Update the env values (and `policy.dataset_stats` for a train config) of a config file in place."""
    config = json.loads(path.read_text())
    processor = config["env"]["processor"]
    processor["inverse_kinematics"].update(inverse_kinematics)
    processor["reset"]["fixed_reset_joint_positions"] = home_joints
    if "policy" in config:
        stats = json.loads(json.dumps(dataset_stats))
        if not (processor.get("observation") or {}).get("add_joint_velocity_to_observation", False):
            for bound in ("min", "max"):
                stats["observation.state"][bound] = stats["observation.state"][bound][:4]
        config["policy"]["dataset_stats"].update(stats)
    path.write_text(json.dumps(config, indent=2) + "\n")


def report_config_values(
    cfg: XArmTeleoperateConfig, robot: XArm, visited_min: np.ndarray, visited_max: np.ndarray
):
    inverse_kinematics, home_joints, dataset_stats = find_config_values(cfg, robot, visited_min, visited_max)
    print("\n\n===== env.processor =====")
    print(
        json.dumps(
            {
                "inverse_kinematics": inverse_kinematics,
                "reset": {"fixed_reset_joint_positions": home_joints},
            },
            indent=2,
        )
    )
    print("\n===== policy.dataset_stats =====")
    print(json.dumps(dataset_stats, indent=2))
    print("\nAdd a margin to the bounds for the UFACTORY Studio safety boundary (it is in millimeters).")

    if not cfg.write_configs:
        return
    paths = [cfg.env_config_path, cfg.train_config_path]
    answer = input(f"\nWrite these values into {', '.join(str(p) for p in paths)}? [y/N] ")
    if answer.strip().lower() != "y":
        print("Configs left unchanged.")
        return
    for path in paths:
        write_config(path, inverse_kinematics, home_joints, dataset_stats)
        print(f"Updated {path}")


@parser.wrap()
def teleoperate(cfg: XArmTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="xarm_teleoperation")

    robot = XArm(cfg.robot)
    teleop = SpaceMouseTeleop(cfg.teleop)
    robot.connect()
    teleop.connect()

    if cfg.home_joints_deg is not None:
        robot.reset_to_home(cfg.home_joints_deg)

    start = np.array([robot.get_observation()[key] for key in EE_POSITION_KEYS])
    bounds = cfg.end_effector_bounds or {
        "min": (start - cfg.box_m).tolist(),
        "max": (start + cfg.box_m).tolist(),
    }
    logging.info(f"End-effector bounds (m): {bounds}")

    spacemouse_to_target = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapDeltaActionToRobotActionStep(),
            CartesianDeltaToTargetStep(
                end_effector_step_sizes=cfg.end_effector_step_sizes,
                end_effector_bounds=bounds,
                delta_reference=cfg.delta_reference,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    visited_min, visited_max = start.copy(), start.copy()
    timer = CycleTimer(cfg.fps, records_data=False)
    try:
        while True:
            timer.tick()
            with timer.section("observe"):
                obs = robot.get_observation()

            if robot.has_fault:
                print(f"\n{robot.fault_description}")
                input("Fix the cause, then press Enter to clear the error and move home (Ctrl+C to quit).")
                robot.reset_to_home(cfg.home_joints_deg)
                spacemouse_to_target.reset()
                timer.restart()
                continue

            with timer.section("teleop"):
                action = spacemouse_to_target((teleop.get_action(), obs))
            with timer.section("send"):
                robot.send_action(action)

            tcp = np.array([obs[key] for key in EE_POSITION_KEYS])
            visited_min, visited_max = np.minimum(visited_min, tcp), np.maximum(visited_max, tcp)
            if cfg.display_data:
                log_rerun_data(observation=obs, action=action)

            print(
                f"TCP (m) {np.array2string(tcp, precision=4)}  gripper {obs['gripper.pos']:.2f}  "
                f"visited min {np.array2string(visited_min, precision=3)} max "
                f"{np.array2string(visited_max, precision=3)}"
            )
            move_cursor_up(1)
            timer.wait()
    except KeyboardInterrupt:
        pass
    finally:
        timer.log_run_summary()
        if robot.is_connected:
            report_config_values(cfg, robot, visited_min, visited_max)
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if cfg.display_data:
            shutdown_rerun()


def main():
    teleoperate()


if __name__ == "__main__":
    main()
