# HIL-SERL on xArm — Work Log

As of 2026-11-11

## Summary

Between 5 Oct and 10 Nov 2026, a UFACTORY xArm7 was brought into LeRobot's HIL-SERL pipeline and trained online to pick up and lift a cube, with SpaceMouse demonstrations and interventions. In the last training run (9 Nov), success without any human help rose from 1 of 50 episodes to 20 of 50, and the intervention rate fell from 35% to 11%. A short follow-on session after ~47k learner steps had 12 of 29 episodes succeed fully on their own, with 1% of steps intervened.

The work covered 19 commits on branch `xarm_spacemouse` (about 30 Claude Code sessions). It added:

- an xArm robot driver;
- a SpaceMouse teleoperator;
- a Cartesian delta-action pipeline;
- 6-DoF control with F/T-sensor admittance;
- data-labelling tools;
- a set of fixes to LeRobot's RL stack, most of them aimed at matching the original [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl).

The policy is not yet reliable: success labels come from an imperfect reward classifier, and no formal evaluation run has been completed.

Sources: git history (`180a2663` → `74754a9a`), Claude Code transcripts in `~/.claude/projects/-home-celes-hilserl-xarm/`, and actor logs under `outputs/train/`.

## Timeline

The work ran in six phases, listed oldest first. Commit hashes are on `xarm_spacemouse`, and dates are local time (UTC+8).

| Date | Commit | What happened |
| --- | --- | --- |
| 5 Oct | `180a2663` | SpaceMouse teleop on the xArm7. Added the missing `spacemouse` extra and `_pyspacemouse_available` flag, and swapped the wrong PyPI `xarm` package for `xarm-python-sdk`. Built a standalone test script with a safety box and a rotation limit. |
| 12 Oct | `bd02b0d5` | Test script: edge-stop instead of sliding along the box wall, pause and alert on controller errors (e.g. error 22, self-collision), and LeRobot's `Rotation` util in place of custom rotvec code (fixes a bug at 180°). |
| 12 Oct | `a6b1e8e2`, `629b1dff` | Planned and built the xArm module: `robots/xarm`, a Cartesian delta→target processor, `teleoperate.py` (finds bounds and home pose, `--write_configs`), `eval_policy.py`, three configs and a README. Fixed silent logging in `gym_manipulator` and RealSense camera registration. |
| 13 Oct | `9820caf6` | First demos and classifier data with two RealSense cameras (wrist D435i, side D405). Chased D405 USB drop-outs (cable, firmware 5.15 → 5.17). |
| 15–16 Oct | `5db3af52`, `72385524` | Reward classifier crashed on 480×640 images, so crop and resize to 128×128 were set up. Fixed `crop_dataset_roi` (missing `finalize()` and an off-by-one at episode boundaries). Success labels latched after Enter, plus `relabel_success.py`. First learner/actor run. Moved the workspace to a new desktop. |
| 19 Oct | `d3f26307` → `a5e11d4f` | Re-recorded 20 demos, added `check_crop.py`, and found demo images were fed to the learner 255× too bright (`buffer.py`). Step timing changed to HIL-SERL's: send, sleep, then observe. Intervention became active only while the puck moves. First long training run (35k steps). |
| 22 Oct | `8646d3ae` | Built `relabel_buffer_rewards.py`. Hand-reviewed 326 online episodes (90 successes, 235 failures). Trained classifier v2 and fixed resume, which had been re-initialising the actor. |
| 22 Oct | `c6bd3248` | 6-DoF actions, F/T sensor in the state, admittance control, rotation teleop and `probe_admittance.py`. |
| 23 Oct | `53711132`, `e61a87a8` | Third camera (front D435). Fixed the F/T frame (tool, not base) and converted the demos offline. Matched HIL-SERL's SAC details. Fixed a stats dtype crash on 1-frame episodes and wrote `training.md`. Classifier v3. |
| 26 Oct | `1a2b80ce`, `74ad5651` | Gripper penalty fixed, fault penalty tried, episode log colouring, `watch_episodes.sh`, save-on-exit and a per-episode buffer journal (removes 40% learner idle time). Performance diagnosed against hil-serl. Gripper lockout and tighter reference clips. Classifier v4 tried and rolled back. |
| 6 Nov | — | Live Rerun view of all cameras. Dropped the side camera: the D405 moved to the wrist, plus a front D435. |
| 9 Nov | `74754a9a` | Re-recorded and reviewed 30 demos with a demo mode in the relabel tool. New classifier. Late relabel window after an episode ends. Fixed `eval_policy.py`. Best training run so far (see Outcome). |
| 10 Nov | — | Exported all chats to `~/claude_chats_hilserl_xarm_2026-11-10.zip`. |

## What was built

The design keeps LeRobot's HIL-SERL shape. The policy, demos and interventions all speak normalized end-effector deltas, and a processor step turns each delta into an absolute pose for the robot. No UFACTORY plugin code is imported; the driver uses only `xarm-python-sdk` calls.

| Area | Files | What it does |
| --- | --- | --- |
| xArm robot | `src/lerobot/robots/xarm/` (`xarm.py`, `config_xarm.py`) | Mode-7 Cartesian streaming at 10 Hz, xArm gripper (0–800), joint-space homing, and fault detection and recovery. Reads the F/T sensor in the tool frame, zeroes it each reset and switches admittance on. Holds a 0.6 s lockout after a gripper switch. |
| Delta → target | `robots/xarm/robot_cartesian_processor.py`, `processor/delta_action_processor.py`, `utils/rotation.py` | 6-DoF deltas (1 cm and 0.06 rad per step at full scale) applied to the measured pose. Clips to a pose box and to a per-axis `reference_clip`, and builds an EE-relative state. |
| SpaceMouse teleop | `teleoperators/space_mouse/` | Axis and rotation mapping, deadzone, and gripper buttons. Intervention is active only while the puck moves (HIL-SERL style). Keyboard handles success, failure, re-record and late relabel. |
| Env / RL loop | `rl/gym_manipulator.py`, `processor/hil_processor.py`, `envs/configs.py` | HIL-SERL step timing (send, sleep, observe), faults end the episode as a failure, gripper penalty, and a minimum episode length of 5 steps. Live Rerun camera view. |
| SAC to match hil-serl | `rl/algorithms/sac/`, `policies/gaussian_actor/` | Simultaneous updates, target encoder, softplus temperature, `utd_ratio` 2, discount 0.98, mean-Q actor loss, and gripper-critic fixes. |
| Actor / learner | `rl/actor.py`, `rl/learner.py`, `rl/buffer.py`, `rl/journal.py`, `rl/train_rl.py` | Colour-coded episode log, held-back episodes for late relabel, save-on-exit, and a per-episode buffer journal in place of whole-buffer rewrites. Fixed resume and the uint8 → float image scaling. |
| Tools | `examples/xarm_hilserl/` | `teleoperate.py`, `eval_policy.py`, `check_crop.py`, `relabel_buffer_rewards.py` (buffer and demo review, cut, delete, export to classifier data), `relabel_success.py`, `convert_ft_frame.py`, `probe_admittance.py`, and `scripts/watch_episodes.sh` |
| Configs and docs | `examples/xarm_hilserl/configs/{env,train,reward_classifier_train}.json`, `README.md`, `training.md` | Three configs in place of five. A step-by-step README covering controls, recording, the classifier, training and eval. `training.md` holds the loss math and a change log against hil-serl. |
| Tests | `tests/robots/test_xarm.py`, `tests/processor/test_cartesian_delta.py`, `tests/rl/test_gym_manipulator_cartesian.py`, `tests/rl/test_actor.py`, `tests/teleoperators/test_space_mouse.py`, `tests/utils/test_replay_buffer.py` | Mock-robot tests for the driver, processors, timing, intervention, late relabel and the buffer |

Hardware at the end: an xArm7 at 192.168.1.235 (firmware 2.7.1) with the xArm gripper and the UFACTORY 6-axis F/T sensor. Cameras are a RealSense D405 on the wrist and a D435 at the front, both 640×480 at 30 fps, cropped and resized to 128×128. Input is a SpaceMouse Compact.

## Connecting the hardware and networking

Every device needed some setup outside Python. Most of it came up again when the workspace moved to a second desktop (`user-TRX50-WS`) on 16 Oct.

| Device | Issue | Resolution |
| --- | --- | --- |
| xArm network | On the new desktop the arm was unreachable. Its port `enp97s0` had only a DHCP profile, and the xArm control box does not hand out addresses. | Added a NetworkManager profile `xarm` with a fixed address of 192.168.1.100/24 that connects automatically. The arm at 192.168.1.235 then answered ping in about 0.2 ms. Undo with `nmcli connection delete xarm`. |
| xArm bring-up | The first read-only check showed all joints at 0.0°, state 4 (stopped), and a TCP offset and payload of 0 | The arm really was parked at its zero pose. `XArm.connect()` enabled the motors and gripper and reached state 2, mode 7, with no errors. |
| xArm controller safety | Collision sensitivity was 0 (off) and TCP payload was 0 kg | Payload set by load identification (0.894 kg). F/T collision thresholds set in UFACTORY Studio: 40 N, about 4 Nm, rebound off. |
| F/T sensor | Enabling it raised controller error 18 ("communication error"). The sensor (firmware 1.0.6, tool-bus address 8) was reporting its own fault 69, "Tx torque exceeds limit", left over from an earlier overload. | A power-cycle of the control box was suggested; the transcripts don't record exactly what cleared it. After load identification it read under ±0.5 N at rest. `probe_admittance.py` (`status`, `frame`, `track`) confirmed tool-frame readings and admittance tracking within 1.7 mm. A 17 N push moved the arm about 8 mm and it came back. |
| Controller paused after F/T commands | Any F/T enable or zero command puts the controller in state 5, so motion is silently ignored | `set_state(0)` after zeroing in `reset_to_home` |
| Error 54 | Not in any SDK error table ("Other Errors"). It is the firmware 2.6.103+ F/T collision stop, and torque thresholds of 0 tripped it. | Thresholds fixed in Studio, and the speed cap lowered to 100 mm/s |
| SpaceMouse | `/dev/hidraw*` was root-only, and `easyhid` needs the system library `libhidapi-hidraw.so.0` | Added the udev rule `KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"` and `sudo apt install libhidapi-hidraw0`. Both are now in README step 2. |
| RealSense D405 (side, later wrist) | The kernel reset it 7 times in one session, and frames went over 500 ms old mid-episode | A long passive USB cable was the main cause; firmware went from 5.15.1.55 to 5.17.0.10. On 6 Nov it was tested on USB 2: 30 fps with 0 dropped frames, alone and with all cameras running. |
| RealSense D435 (front) | Enumerated at USB 2.1 | Moved to a USB 3.2 port on 23 Oct. All three cameras then showed USB 3.2. |
| Two cameras at start-up | The first warm-up read on the second camera sometimes failed (bandwidth contention) | Harmless: LeRobot's retry connects about a second later |
| Python environment | `uv sync --extra xarm_hilserl` removed `opencv-python`, which deleted the shared `cv2/` files of `opencv-python-headless`. `accelerate` was missing for classifier training. | Reinstalled the headless build. Sync with every extra in one command: `uv sync --locked --extra xarm_hilserl --extra training --extra test`. The crop and relabel GUIs run with `uv run --with "opencv-python>=4.12,<4.14"`. |
| GUI windows | The relabel tool hung when opening its window: Qt freezes if OpenCV is imported after LeRobot | Import OpenCV first |
| Rerun viewer | It opened a VS Code window instead of the viewer. On 13 Oct `nvidia-smi` showed a driver/library mismatch (595.91 vs 595.99) after an unattended upgrade. | Reboot. Live viewing was rebuilt on 6 Nov (see below). |
| Shell | A pasted `export PATH="HOME/.local/bin:PATH"` (no `$`) broke PATH | Reset PATH, or open a new terminal |

## Operator interface

The operator drives everything from the SpaceMouse and keyboard, and watches through Rerun and a colour-coded log.

**SpaceMouse**

- Puck translation moves the TCP in base-frame x/y/z. The direction is remapped with `teleop.axis_mapping`, e.g. `[y, -x, z]` when standing in front of the robot.
- Tilt and twist rotate the TCP about the same axes. On 22 Oct, rotation was re-mapped to follow the translation axes, because pyspacemouse's LEGACY labels had roll and pitch swapped and reversed.
- The left button closes the gripper and the right button opens it.
- Moving the puck takes control, and letting go hands control back. Space latches intervention on or off.

**Keyboard**

| Process | Enter | Backspace | `r` |
| --- | --- | --- | --- |
| Recording | Mark success; this and every later frame are labelled success, and recording continues | End and save (a failure if Enter was not pressed) | Discard and redo |
| Online training (actor) | End episode: success | End episode: failure | End episode |
| Evaluation | End episode: success | End episode: failure | End episode |

In the actor, Enter or Backspace pressed after an episode ends relabels that episode until the next one passes 5 steps. Episodes shorter than 5 steps are dropped.

**Monitoring**

- **Rerun live view.** `display_cameras` never worked before 6 Nov: `render()` looked for `"image"` keys outside `obs["pixels"]`, and the installed OpenCV is the headless build. `RobotEnv` now logs every camera frame to Rerun at each step and reset, with no extra camera reads. It shows full 640×480 frames, before cropping.
- **Episode log.** The actor prints a coloured `[EPISODE START]` / `[EPISODE END] … reward … len … intervention n/N` line, plus an `[EPISODE RELABEL]` line for late keys. `scripts/watch_episodes.sh` follows only those lines from the newest actor log.
- **wandb** is on in `train.json` (project `xarm7_hilserl`). It needs `uv run wandb login` on each machine.

**Review tool (`relabel_buffer_rewards.py`)**

The tool shows the cameras side by side at the episode's last frame. Its header gives progress, the recorded label, the classifier's reward frame, the intervention count and the gripper opening. It runs in two modes:

- **Buffer mode** (`--run-dir`) reviews a training run's online episodes. Keys: Enter/`y` success, Backspace/`n` failure, `s` keep, `u` back one episode, `a`/`d` step one frame (`A`/`D` 10 frames), `q` save and quit, Esc quit without saving. It writes a `<run>_relabeled` copy; the original run is never modified.
- **Demo mode** (`--dataset`) reviews a recorded demo dataset. It adds `t` to cut the episode after the current frame (e.g. hands in view), and `x` to delete the episode. It writes `<repo_id>_reviewed`.

Fixes along the way:

- The tool first showed only the episodes the classifier had rewarded; it now shows every episode.
- It now starts from the label recorded during training instead of "unlabeled".
- Closing the window used to reopen it in a loop; closing now saves and quits.

## Camera angles and cropping

Camera placement and crop boxes were redone several times. Each change meant re-recording demos and retraining the classifier, because old frames no longer matched new ones.

| Date | Issue | What was done |
| --- | --- | --- |
| 12 Oct | The wrist D435i alone could not see where a randomly placed object would be at the start | Added a fixed side D405. Rule adopted: crop to the region where the object **can** appear across all episodes and the whole motion, not where it sits in the one frame the tool shows. |
| 13 Oct | Images too bright or too dark. The D405 was dark (mean 48/255). | Manual `exposure`/`gain` on the D435i. The D405 has no separate RGB module, so LeRobot refuses manual exposure on it and it stays on auto. `lerobot-find-cameras` ignores the configs, so a config-based snapshot script was used for checks. |
| 15 Oct | The first datasets were recorded at 480×640 with no crop | Crop boxes drawn with `crop_dataset_roi` (wrist `[88, 112, 390, 525]`, side `[157, 19, 302, 582]`), plus a resize to 128×128 |
| 19 Oct | Contact sheets showed the cube outside the wrist crop in about half the episodes | Added `check_crop.py`, which draws the boxes on raw frames for each episode. New boxes: wrist `[103, 261, 375, 379]`, side `[135, 181, 299, 339]`. |
| 23 Oct | Third camera (front D435) added. Demos were accidentally recorded with `resize_size` set, so they were saved already shrunk to 128×128. | Rule written down: record with crop and resize `null` (full frame), crop afterwards, and keep `train.json` at 128×128 with the same boxes. Boxes: wrist `[1, 0, 477, 640]`, side `[144, 77, 313, 470]`, front `[67, 15, 413, 606]`. |
| 26 Oct | `check_crop.py` drew 640×480 boxes on 128×128 frames without warning | It now refuses boxes that don't fit the frames |
| 6 Nov | Wrist D435i aimed too far ahead and left: cube centre at a median (474, 386) in a 640×480 frame, and its bottom edge cut off | Re-aiming was recommended over cropping, since software cannot recover pixels outside the frame. The side camera was removed: the D405 became the wrist camera and the front D435 stayed. |
| 9 Nov | Starting over with the new layout | Teleop and recording restarted from full frames, with no crop |

## Calibration mismatches

Several calibration values live only on the xArm controller. LeRobot never reads or saves them, so they can drift between recording and training without notice.

- **TCP offset and payload were 0 at first** (16 Oct). The reported pose was the flange, not the fingertips, and collision detection was less accurate.
- **Load identification.** The F/T load was identified on 22 Oct (0.894 kg, centroid 89.7 mm). Claude's first identification script moved the whole arm to a stretched-out pose, not just the wrist; after that it was done in UFACTORY Studio.
- **Recalibration on 9 Nov.** Calibration was missed and then redone on 9 Nov. Whether the demos are still valid depends on what changed. LeRobot's `XArm.calibrate()` is a no-op, and the F/T sensor is re-zeroed at every reset.

    | What changed | Re-record demos? |
    | --- | --- |
    | TCP offset | Yes: every pose and action in the demos is measured at the TCP |
    | Joint zero | Yes: every end-effector pose shifts |
    | Payload / F/T load identification | Partly: gravity compensation changes as the gripper tilts |
    | Camera moved | Yes, and retrain the classifier |

    Suggested check: home the arm and compare the reset `observation.state` with an old demo. If the pose is off by more than a few mm or degrees, or the force shows a steady offset, re-record.
- **The identification used for the trained policy was never recorded.** Checkpoints save only LeRobot's admittance settings (mass, stiffness, axes, frame). The TCP payload and F/T load offset are not in any run log, `~/.UFACTORY/log` is empty, and `probe_admittance.py identify` was never run from the shell. They were set in UFACTORY Studio.
- **Force frame mismatch** (23 Oct). The demos stored force rotated into the base frame, but hil-serl uses the raw tool-frame wrench. The 894 demo frames were converted offline with `convert_ft_frame.py`, and force magnitudes were unchanged to within 1e-6.

## Problems hit and how they were fixed

Most of the bugs that hurt learning were silent: training ran but learned from wrong data. These had the most impact, roughly in order of discovery.

| Problem | Cause | Fix | Commit |
| --- | --- | --- | --- |
| Arm would not import or move from the SpaceMouse | Missing `spacemouse` extra and flag. The PyPI `xarm` package is a different robot's SDK. | Added the extra and flag, and switched to `xarm-python-sdk` | `180a2663` |
| No progress messages in `gym_manipulator` | A `logging.debug` at import time locked the root logger at WARNING | Call `init_logging()` first in `main()` | `a6b1e8e2` |
| `intelrealsense` camera type not found | Camera configs were not registered in the RL entry points | Imported the RealSense config in `gym_manipulator` and `learner` | `a6b1e8e2` |
| Side camera dropping out | Long passive USB cable and old D405 firmware | Shorter cable, firmware update. The D405 later ran fine on USB 2. | — |
| Classifier crash (20 vs 4 size mismatch) | Recorded at 480×640, but the classifier's spatial embedding needs 128×128 | Crop and resize in `env.json` and `train.json`, and re-cropped old datasets | `5db3af52` |
| Cropped datasets lost episodes and shifted frames | `crop_dataset_roi` never called `finalize()` and saved episodes one frame late | Fixed both in `crop_dataset_roi.py` | `5db3af52` |
| Classifier caught 2 of 43 success frames | Success was labelled on one frame only, and the classifier step ran on CPU tensors without normalization | Success latches from Enter onward, and the classifier step got device and normalization fixes. Recall rose to 97% on its training data. | `72385524` |
| Demo images 255× too bright in the learner | The offline dataset loads uint8 0–255, but the RL buffer never divided by 255 | `_uint8_to_float` in `buffer.py`, then retrained from scratch | `5faa153c` |
| Observation one step behind the action | `step()` read the robot straight after `set_position`, and the sleep came later in the loop | HIL-SERL order: start timer, send, sleep the rest of 1/fps, observe | `1be2f649`, `8ff3ba74` |
| Idle "hold still" frames counted as expert actions | Intervention was latched on with Space | Intervention is active only while the puck or a button is in use | `a5e11d4f` |
| Collision stopped recording | A controller fault raised an exception | Fault ends the episode as a terminated failure, and reset recovers the arm (including SDK code −9) | `d3f26307` |
| Many false successes in the online buffer | Classifier v1 misfired | `relabel_buffer_rewards.py`: 326 episodes reviewed by hand, then classifier v2 trained on them | `8646d3ae` |
| Resume restored only part of the networks | Actor and gripper critic were rebuilt from random weights on `--resume` | Load all networks from the checkpoint, plus a `--buffers_path` fresh start | `8646d3ae` |
| Arm frozen after F/T zeroing (gripper still worked) | Any F/T command pauses the controller (state 5) | `set_state(0)` after zeroing in `reset_to_home` | `c6bd3248` |
| Rotation axes did not match translation | pyspacemouse's LEGACY rotation labels | New `rotation_mapping` defaults aligned to the translation axes | `c6bd3248` |
| Roll and yaw worked one way only | `--write_configs` centred the angle limits on the start pose but saved the finish pose as home | Angle limits = home ± `box_rad`, with home = start pose | `53711132` |
| Error 54 ending episodes | F/T collision detection set at 40 N with torque thresholds of 0 | Set thresholds in UFACTORY Studio (40 N, about 4 Nm). Speed cap 100 mm/s. | `53711132` |
| Force in the wrong frame | A code comment assumed hil-serl used base-frame force (`O_F_ext_hat_K`), but it uses `K_F_ext_hat_K` | `ft_sensor_frame: tool` is now the raw sensor reading. Demos converted with `convert_ft_frame.py`. | `e61a87a8` |
| Learner crash while saving checkpoint 10000 | 1-frame episodes kept int and bool dtypes in their stats | Cast in `compute_stats.py` | `53711132` |
| Gripper penalty almost never fired | The penalty compared the wrong signals | Fires on every real open or close switch. Penalty later set to −0.05, with a 0.6 s lockout. | `1a2b80ce`, `74ad5651` |
| Learner idle 40% of the time | Every checkpoint rewrote the whole buffer as AV1 video (3–6 min pauses) | Checkpoints save networks only, and each episode is appended to a `dataset_journal` | `74ad5651` |
| Online buffer lost on a second Ctrl+C | Save deleted `dataset/` before rewriting it | Safe save and save-on-exit | `1a2b80ce`, `74ad5651` |
| Wrong label after a late key press | Enter or Backspace during reset ended the next episode at step 1 | Late keys relabel the previous episode until the new one passes `min_episode_steps`. Shorter episodes are discarded. | `1a2b80ce`, `74754a9a` |
| `eval_policy.py` ignored `--policy.path` | Missing `cfg.validate()` | Added the call | `74754a9a` |

## Approaches tried and abandoned

Several changes were reverted once data showed they hurt or added nothing.

| Tried | Why it was dropped |
| --- | --- |
| Port the original JAX hil-serl to the xArm | Rejected on 19 Oct: it would mean a new JAX/agentlace stack and a rewritten robot env. The LeRobot path was close to working, so the effort went into making LeRobot match hil-serl instead. |
| 3-DoF actions (xyz + gripper), the LeRobot default | Replaced by 6-DoF with F/T and admittance on 22 Oct, to follow hil-serl |
| Base-frame force (`ft_sensor_frame: tool` rotated into the base frame) | It came from a wrong reading of hil-serl, which uses the raw tool-frame wrench. Reverted on 23 Oct and the demos converted. |
| Classifier v1 (Oct 15) and v2 | v1 gave many false successes. v2 was trained on the hand-relabelled buffer but behaved differently live than offline. Superseded by v3. |
| Classifier v4 (26 Oct, trained on 219 relabelled episodes) | Fired falsely "all the time" live and was rolled back to v3 the same day. A later offline test on 300 frames favoured v4 (v3 recall 0.04), so the evidence is mixed. A new unversioned classifier trained on the 9 Nov demos is now in use. |
| Fault penalty −0.5 for xArm faults | The policy learned to stay high and avoid the table. Set to 0 the same day, and existing −0.5 rewards were relabelled to 0. |
| Recency-weighted replay sampling | Added on 26 Oct, then removed at the user's request; sampling is uniform again |
| Extra weight on intervention samples | Not done: interventions were already about 69% of each batch (stored in both buffers) |
| hil-serl's `ram_insertion` reference clips (1.6 mm on y) | Teleop felt very slow in x/y and fast in z. Replaced on 9 Nov with ±10 mm x/y, ±5 mm z and ±0.06 rad on all three angles. |
| Three cameras (wrist D435i, side D405, front D435) | Cut to two on 6 Nov: the D405 moved to the wrist and the front D435 stayed |
| Live depth input | Never used, matching hil-serl (colour only) |

## Outcome and current state

The full HIL-SERL loop now runs on the xArm: record demos, train a classifier, then learn online with interventions, resume and evaluate. The 9 Nov run is the first in which the policy clearly improved by itself: unassisted success rose to 40% of episodes.

Episodes that succeeded with zero intervention steps, per block of 50 episodes:

| Episode block | 26 Oct run | 9 Nov run |
| --- | --- | --- |
| 1–50 | 2 / 50 (4%) | 1 / 50 (2%) |
| 51–100 | 14 / 50 (28%) | 4 / 50 (8%) |
| 101–150 | 6 / 50 (12%) | 12 / 50 (24%) |
| 151–200 | 9 / 50 (18%) | 20 / 50 (40%) |
| 201–250 | 5 / 46 (11%) | 18 / 50 (36%) |

Source: actor logs `outputs/train/2026-10-26/15-20-28` (246 episodes) and `2026-11-09/14-20-08` (252 episodes); success as labelled live by the classifier or keys. A suggested bar before a formal eval is about half of episodes unassisted.

In the same run, the share of steps under intervention fell from 35% in the first block to 11% in the fourth, then rose to 22% in the fifth.

The 26 Oct run never passed 14 unassisted successes per 50. In that run's learner data, 155 of 206 episodes without intervention failed. Most ended in xArm error 54 after a median of 17 steps.

These changes landed between the two runs; which ones mattered is not isolated:

- the gripper penalty fix and the 0.6 s lockout;
- journaled buffers, so the learner no longer idles 40% of the time;
- tighter clips;
- two cameras instead of three;
- a re-recorded demo set;
- a new classifier.

After the 9 Nov actor crashed, the learner was resumed from about step 47k. The next 29-episode session had 12 unassisted successes, with 1% of steps under intervention. The newest checkpoint is step 56000 in `outputs/train/2026-11-09/14-19-43_xarm7_hilserl/checkpoints/`.

Caveat: all counts are the labels at episode end, before any late relabel, and no `eval_policy.py` result exists yet.

## Open issues and next steps

- [ ] Run a proper evaluation (`eval_policy.py`, 10+ episodes) on the latest checkpoint (step 56000 or later of `2026-11-09/14-19-43`). The 9 Nov eval was started but no result was recorded.
- [ ] Check classifier accuracy on held-out episodes. Every success count above comes from it, plus live key corrections.
- [ ] Tighten the rotation bounds. `end_effector_bounds` still allows |roll| 1.57–π and pitch ±1.55 rad, against about ±0.1 rad in hil-serl. Large rotations were linked to error-54 faults on 26 Oct.
- [ ] Make `reference_clip` smaller than the step size on x and y. Both are 10 mm now, so the clip never limits contact force in x/y. hil-serl keeps the clip at 2–6 mm under a larger action scale.
- [ ] Record the payload and F/T load identification. It was done in UFACTORY Studio and is not saved with any run.
- [ ] Handback transitions in the offline buffer get the wrong next observation (about 10% of offline intervention steps). The fix was proposed on 26 Oct but not confirmed as done.
- [ ] Consider a small fault penalty (about −0.1) once bounds keep the gripper off the table.
- [ ] Commit messages from 26 Oct on are "latest". Squash or reword them before any PR to `main`.
