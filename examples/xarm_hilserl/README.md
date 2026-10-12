# HIL-SERL on a UFACTORY xArm7 with a SpaceMouse

Train an xArm7 on a real-world task with LeRobot's [HIL-SERL](https://hil-serl.github.io/) implementation:

1. record demonstrations with a 3Dconnexion SpaceMouse;
2. train a reward classifier;
3. train a policy online with SAC, taking over with the SpaceMouse whenever it needs correcting;
4. run the learned policy on its own.

Read [`docs/source/hilserl.mdx`](../../docs/source/hilserl.mdx) for the general method. This guide covers what is specific to the xArm setup.

## How it works

HIL-SERL learns **normalized end-effector deltas**: `[dx, dy, dz] ∈ [-1, 1]` plus a discrete gripper command `{0: close, 1: stay, 2: open}`. Each of these uses that same action format:

- the policy;
- the SpaceMouse;
- the recorded demos;
- the replay buffers.

The robot never sees a delta. On every step the action pipeline in `lerobot.rl.gym_manipulator` converts it into an absolute target:

```
policy or SpaceMouse (while intervening)
   │  [dx, dy, dz, gripper]                       ← stored in the replay buffer / dataset, learned by SAC
   ▼
CartesianDeltaToTargetStep
   │  target = reference pose + delta × end_effector_step_sizes, clipped to end_effector_bounds
   ▼
XArm.send_action({"ee.x", "ee.y", "ee.z", "gripper.pos"})     ← absolute target, meters, robot base frame
   ▼
xArm controller (online Cartesian trajectory planning, mode 7) does the inverse kinematics
```

This is the same pipeline LeRobot uses for the SO-100 with a gamepad, except for one step. The SO arm needs URDF inverse kinematics; the xArm controller tracks Cartesian targets itself. Orientation stays fixed (3-DoF control). The gripper is binary: "stay" keeps the last open/close command, so a grasp keeps squeezing.

Code involved:

| What                                           | Where                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------- |
| xArm robot (`--robot.type=xarm`)               | `src/lerobot/robots/xarm/xarm.py`, `config_xarm.py`                             |
| Delta → Cartesian target step                  | `src/lerobot/robots/xarm/robot_cartesian_processor.py`                          |
| HIL-SERL env, recording                        | `src/lerobot/rl/gym_manipulator.py` (generalized to robots without a motor bus) |
| SpaceMouse teleop (`--teleop.type=spacemouse`) | `src/lerobot/teleoperators/space_mouse/`                                        |
| Teleop + bounds tool, policy evaluation        | `examples/xarm_hilserl/teleoperate.py`, `eval_policy.py`                        |
| Configs                                        | `examples/xarm_hilserl/configs/`                                                |

## Hardware and I/O sources

| Device                              | Interface                         | Library           | Used for                                           |
| ----------------------------------- | --------------------------------- | ----------------- | -------------------------------------------------- |
| xArm7 + control box                 | Ethernet, `192.168.1.235`         | `xarm-python-sdk` | Arm motion, TCP pose                               |
| xArm Gripper                        | Tool Modbus, through the xArm SDK | `xarm-python-sdk` | Open / close, opening read-back                    |
| 3Dconnexion SpaceMouse Compact      | USB HID (`/dev/hidraw*`)          | `pyspacemouse`    | Arm xyz motion + gripper buttons                   |
| Intel RealSense D435i, on the wrist | USB 3                             | `pyrealsense2`    | Wrist RGB image (depth is not used)                |
| Keyboard                            | X11 session                       | `pynput`          | Intervention, success / failure, re-record         |
| GPU                                 | CUDA                              | PyTorch           | Learner (SAC updates) and actor (policy inference) |

> [!WARNING]
> `pynput` captures the keyboard **globally**: the keys below trigger events whichever window has focus. Don't type in other windows while a session is running.

## Install and one-time setup

1. **Python 3.12 environment.** `pyrealsense2` only ships Linux wheels for Python 3.12 / 3.13.

   ```bash
   uv sync --python 3.12 --locked --extra xarm_hilserl
   ```

   The `xarm_hilserl` extra installs:
   - `lerobot[xarm]` (`xarm-python-sdk`);
   - `lerobot[spacemouse]`;
   - `lerobot[intelrealsense]`;
   - `lerobot[hilserl]`;
   - `lerobot[viz]`.

   `xarm-python-sdk` is not part of the base `lerobot` install.

2. **Device permissions.**
   - RealSense: install Intel's [udev rules](https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules).
   - SpaceMouse: if `pyspacemouse` cannot open the device, allow access to 3Dconnexion HID devices:
     ```bash
     echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules
     sudo udevadm control --reload-rules && sudo udevadm trigger
     ```
3. **Camera.** The configs select the camera by name (`"Intel RealSense D435I"`), which works while it is the only D435i plugged in. Otherwise use its serial number from `lerobot-find-cameras realsense`.
4. **UFACTORY Studio.** These are controller settings, persistent and enforced by the controller even if Python crashes:
   - **TCP offset and payload** of the xArm Gripper, so the reported pose is the fingertip and collision detection is accurate.
   - **Safety boundary**: slightly larger than the soft box (`end_effector_bounds`) you'll find in step 1.
   - **Collision sensitivity** and **self-collision detection**.

## Controls reference

**SpaceMouse** (every process):

| Input            | Action                                                                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Puck translation | Move the TCP in x / y / z of the robot base frame (+X out of the front of the base, +Y to its left, +Z up). Remap with `teleop.axis_mapping`. |
| Left button      | Close the gripper (stays closed)                                                                                                              |
| Right button     | Open the gripper (stays open)                                                                                                                 |

**Keyboard**, per process:

| Process                                   | Space                                                      | Enter                                                 | Backspace            | `r`               | Ctrl+C                           |
| ----------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- | -------------------- | ----------------- | -------------------------------- |
| `teleoperate.py`                          | —                                                          | Clear a fault and move home (when asked)              | —                    | —                 | Quit and print the config values |
| Dry run (`env.json`)                      | Toggle intervention                                        | End episode: success                                  | End episode: failure | End episode: redo | Quit                             |
| Demo recording (`env.json --mode=record`) | Toggle intervention (press after each reset to drive)      | End + save: success                                   | End + save: failure  | Discard + redo    | Quit                             |
| Classifier recording (see step 5)         | Toggle intervention                                        | **Hold** in the success state: frames labeled success | End + save           | Discard + redo    | Quit                             |
| Online training (actor)                   | Toggle intervention: SpaceMouse takes over from the policy | End episode: success                                  | End episode: failure | End episode       | Quit actor / learner             |
| Evaluation (`eval_policy.py`)             | Safety takeover                                            | End episode: success                                  | End episode: failure | End episode       | Quit, print results              |

Intervention switches back **off** at the end of every episode.

## Step by step

All commands run from the repository root, through `uv run` so they use the project environment. There are three configs in `examples/xarm_hilserl/configs/`:

| Config                         | Used by                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `env.json`                     | `gym_manipulator`: dry run, demo recording and classifier recording, switched with command-line flags |
| `train.json`                   | The learner, the actor and `eval_policy.py`                                                           |
| `reward_classifier_train.json` | `lerobot-train` for the reward classifier                                                             |

The robot / workspace values live in both `env.json` and `train.json` (one config class each). `teleoperate.py --write_configs=true` updates both. Every JSON field can be overridden on the command line, e.g. `--dataset.task="Put the cube in the bowl"`.

### 1. Teleoperate, set up the workspace, freeze the step size

```bash
uv run python examples/xarm_hilserl/teleoperate.py \
    --robot.cameras='{wrist: {type: intelrealsense, serial_number_or_name: "Intel RealSense D435I", width: 640, height: 480, fps: 30}}' \
    --home_joints_deg='[0, 0, 0, 90, 0, 90, 0]' \
    --write_configs=true
```

The arm first moves to `--home_joints_deg` (here UFACTORY's ready pose; omit it to start where the arm is). A status line shows the live TCP and the min/max visited so far. The arm only moves while you push the puck, inside a ±30 cm box around the start pose (`--box_m`, or an explicit `--end_effector_bounds`). Add `--display_data=true` to see the wrist camera and TCP in Rerun.

1. **Axis mapping.** Push the puck forward, left and up; the TCP should move +X, +Y, +Z. If not, fix `--teleop.axis_mapping` (e.g. `'[y, -x, z]'`), then set the same `env.teleop.axis_mapping` in `env.json` and `train.json`.
2. **Step size.** `--end_effector_step_sizes` is the distance moved per step at full deflection (default 1 cm = 10 cm/s at 10 Hz). Pick values that are precise enough for the task. **Freeze them now**: demos store normalized actions, so changing the step size later changes what the demos mean.
3. **Workspace and home.**
   - Move the TCP over the whole region the task needs: every corner, highest and lowest point.
   - Finish at the pose every episode should start from.
   - Press Ctrl+C.

On exit the script prints the values it found:

- `env.processor.inverse_kinematics`: `end_effector_bounds` (the visited box), step sizes, `delta_reference`;
- `env.processor.reset.fixed_reset_joint_positions`: the current joints, the home pose of every episode;
- `policy.dataset_stats` (derived from the box and step sizes).

With `--write_configs=true` it then asks `Write these values into …env.json, …train.json? [y/N]`. Answer `y` only after a good run: a run that ended in an error or a bad final pose should not overwrite your configs. Without the flag, nothing is written.

Then set the Studio safety boundary a few centimeters outside the box.

`delta_reference` chooses what the delta is applied to:

|                            | `"measured"` (default, as in rail-berkeley/hil-serl) | `"commanded"` (as in LeRobot's SO pipeline)                              |
| -------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Delta applied to           | The measured TCP                                     | The previous target                                                      |
| After the puck is released | Stops within one step                                | Coasts until it reaches targets it fell behind on                        |
| When blocked / in contact  | Pushes at most one step                              | Target runs ahead, so the push grows (and can trigger a collision error) |
| Speed                      | Slightly below `step × fps`                          | Exactly `step × fps`                                                     |

Compare them with `--delta_reference`. Use the same value for demos and training.

Plain `lerobot-teleoperate` does not work for this pair: it passes the teleop action to the robot unchanged, but the SpaceMouse emits deltas and the xArm takes absolute targets. The same is true of LeRobot's other delta teleops (phone, gamepad), which also come with their own scripts.

### 2. Rehearse in the HIL loop (nothing is saved)

```bash
uv run python -m lerobot.rl.gym_manipulator --config_path examples/xarm_hilserl/configs/env.json
```

This runs exactly the loop used for recording and training, with no policy:

- each episode starts with a reset to `fixed_reset_joint_positions`;
- you rearrange the scene before `reset_time_s` (5 s, counted from the start of the reset, homing included) runs out;
- the robot holds still until you press **Space**, after which the SpaceMouse drives it;
- **Enter / Backspace** end the episode;
- after `control_time_s` (20 s) the episode times out.

### 3. Record demonstrations

```bash
uv run python -m lerobot.rl.gym_manipulator --config_path examples/xarm_hilserl/configs/env.json \
    --mode=record --dataset.task="<what the robot should do>"
```

For each episode:

1. Wait for the reset.
2. Press **Space** right away, so few idle frames are recorded.
3. Do the task with the SpaceMouse.
4. Press **Enter** when it's done.

Press **`r`** to discard an attempt that went wrong: like rail-berkeley/hil-serl, keep only successful demos (about 20, `num_episodes_to_record`).

The dataset is written to `~/.cache/huggingface/lerobot/local/xarm7_hilserl_demos` at full camera resolution, so step 4 can crop it.

### 4. Crop the wrist image to the task region

```bash
uv run python -m lerobot.rl.crop_dataset_roi --repo-id local/xarm7_hilserl_demos
```

1. Draw a rectangle around the region the task happens in.
2. Press `c`.

The script prints the crop as `[top, left, height, width]` in camera pixels. It also writes the cropped 128×128 dataset `local/xarm7_hilserl_demos_cropped_resized`, which is the offline dataset for training.

Paste the crop into `env.processor.image_preprocessing` of both configs:

- `env.json`: set `crop_params_dict` to `{"observation.images.wrist": [top, left, height, width]}` and `resize_size` to `[128, 128]`. Both were `null` (full resolution) for the demos.
- `train.json`: replace the placeholder `[0, 0, 480, 640]` (the full frame) in `crop_params_dict`.

Recordings made from now on are already cropped to 128×128.

### 5. Train the reward classifier

The classifier detects success from the wrist image; it gives the reward and ends successful episodes during training. First record labeled data:

```bash
uv run python -m lerobot.rl.gym_manipulator --config_path examples/xarm_hilserl/configs/env.json \
    --mode=record --dataset.repo_id=local/xarm7_hilserl_classifier --dataset.num_episodes_to_record=15 \
    --env.processor.reset.terminate_on_success=false --dataset.task="<what the robot should do>"
```

For each episode (`terminate_on_success=false`, so episodes don't end on success):

1. Press **Space** and do the task.
2. Once it's done, **hold Enter** for a couple of seconds. Every frame with the key down is labeled success.
3. End the episode with **Backspace**, or let it time out.

Vary the scene: also record near-misses and failures. Then train:

```bash
uv run lerobot-train --config_path examples/xarm_hilserl/configs/reward_classifier_train.json \
    --output_dir outputs/train/xarm7_reward_classifier
```

To validate it:

1. Set `env.processor.reward_classifier.pretrained_path` in `env.json` to `outputs/train/xarm7_reward_classifier/checkpoints/last/pretrained_model`.
2. Run the dry run.
3. Episodes should end by themselves when the task is done, and not before.

### 6. Train online with human interventions

Before starting, check that `train.json` has:

- the bounds, step sizes, home joints and crop from steps 1 and 4;
- the `policy.dataset_stats` from step 1;
- the classifier path (`outputs/train/xarm7_reward_classifier/checkpoints/last/pretrained_model`).

Then start the learner and the actor in two terminals:

```bash
uv run python -m lerobot.rl.learner --config_path examples/xarm_hilserl/configs/train.json
```

```bash
uv run python -m lerobot.rl.actor --config_path examples/xarm_hilserl/configs/train.json
```

The actor runs the policy on the robot at 10 Hz and streams transitions to the learner over gRPC. The learner trains SAC on a 50/50 mix of:

- online transitions;
- your demos plus every intervention.

It sends updated weights back every `policy_parameters_push_frequency` seconds.

How to intervene:

- Let the policy explore on its own for the first episodes.
- Press **Space** to take over only when it is going wrong, correct with the SpaceMouse, and press **Space** again to hand back.
- Keep interventions short.
- Once it starts succeeding, intervene only for quick nudges, e.g. a grasp.

The intervention rate should fall over time. The learner tracks it as `Intervention rate` in Weights & Biases (set `wandb.enable: true` in `train.json`).

Checkpoints are written to `outputs/train/<date>/<time>_xarm7_hilserl/checkpoints/<step>/pretrained_model` (`last` points to the latest). To resume, run the learner with `--resume=true --config_path=<run>/checkpoints/last/pretrained_model/train_config.json`.

### 7. Run the learned policy

```bash
uv run python examples/xarm_hilserl/eval_policy.py \
    --config_path=examples/xarm_hilserl/configs/train.json \
    --policy.path=outputs/train/<date>/<time>_xarm7_hilserl/checkpoints/last/pretrained_model \
    --eval.n_episodes=10
```

The robot runs the policy for `eval.n_episodes` episodes, with the same resets, classifier and keys as in training. Nothing is trained. Actions are sampled from the policy, as in rail-berkeley/hil-serl's evaluation. At the end the script prints:

- the success rate;
- the mean time to success;
- how many episodes you had to take over in.

## Safety and errors

- **Soft box:** every target is clipped to `end_effector_bounds`. Pushing against an edge is not a failure; the arm just stops there.
- **Hard limits:** the Studio safety boundary, collision detection and self-collision detection stop the arm in the controller. The control box's emergency stop always works.
- **On an xArm error** (collision, boundary, …), commands stop. The episode ends as _truncated_: it's not counted as a failure, and in record mode you can redo it with `r`. The next reset clears the error and moves home.
- **Random reset** (`robot.random_xy_range_m`, default 0 = off) adds a random xy offset to the home pose every episode, as `RANDOM_RESET` in rail-berkeley/hil-serl, whose tasks use 0.01–0.02 m. It makes the policy robust to where it starts; rearranging the scene during resets already provides much of that.

## Tuning and troubleshooting

| Symptom / knob                                                  | What to do                                                                                                                                                                       |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arm moves after releasing the puck                              | Use `delta_reference: "measured"`. Keep `robot.max_linear_speed_mm_s` above `step size × fps`.                                                                                   |
| Step size                                                       | Fixed once demos are recorded. To change it, record the demos again.                                                                                                             |
| `temperature_init` (`algorithm`)                                | 0.01 by default. Higher means more exploration, but interventions then have less effect.                                                                                         |
| `policy_parameters_push_frequency`                              | Seconds between weight updates on the actor; 2 s by default.                                                                                                                     |
| Loop slower than 10 Hz                                          | The actor only warns when policy inference is too slow. Robot I/O (TCP and gripper reads, camera) is timed by the summary `teleoperate.py` prints on exit. Check it there first. |
| Replay buffer memory                                            | Buffers live in CPU RAM (`storage_device: cpu`), about 20 GB per 100k transitions. Reduce `online_buffer_capacity` / `offline_buffer_capacity` if needed.                        |
| Reward classifier unreliable                                    | It only sees the wrist camera. If success isn't visible from the wrist, add a fixed side camera to `robot.cameras` and to the configs' features.                                 |
| SpaceMouse not found                                            | Check the udev rule above and that `ls -l /dev/hidraw*` is readable.                                                                                                             |
| Keyboard keys ignored                                           | `pynput` needs an X11 session (not Wayland or SSH without a display).                                                                                                            |
| `pyrealsense2` fails to install                                 | Use Python 3.12 or 3.13.                                                                                                                                                         |
| `src/lerobot/teleoperators/space_mouse/xarm_spacemouse_test.py` | An earlier stand-alone test script, not used here. Move it out of `src/` before opening a PR.                                                                                    |
