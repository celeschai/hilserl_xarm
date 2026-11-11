# xArm7 HIL-SERL — results report data

Online HIL-SERL session of **2026-11-09**, as numbers and figures: learner `outputs/train/2026-11-09/14-19-43_xarm7_hilserl`, offline demos `local/xarm7_hilserl_demos_6dof_ft`, reward classifier `outputs/train/xarm7_reward_classifier/checkpoints/last`.

## Headline numbers

| | |
|---|---|
| Online episodes | **296** (252 in the main actor run 14:20–15:58, 44 after the actor restart 16:33–16:58) |
| Environment steps (10 Hz) | 33,343, of which **7,065 (21%) under human control** |
| Learner optimization steps | 57,657 |
| Successes | 183 (62%), of which **54 with zero intervention** (first one at episode 78) |
| Robot faults | 33 episodes (28 × xArm error 54, 3 × speed limit, 1 × kinematic, 1 × self-collision) |
| Offline demos | 14 episodes, 1,742 frames, median 13.1 s |

**Main actor run: first 50 vs last 50 episodes**

| | Episodes 0–49 | Episodes 202–251 |
|---|---|---|
| Intervention rate, all episodes | 33% | **16%** |
| Intervention rate, successful episodes | 41% | **19%** |
| Episodes with any intervention | 90% | 54% |
| Success rate | 56% | 72% |
| Success with zero intervention | 0% | **32%** |
| Time to success (median) | 12.6 s | **9.6 s** |

**By third of the main run** (`data/processed/online_phases.csv`)

| Phase | Episodes | Success | Intervention rate | Intervention rate (successes) | Autonomous success |
|---|---|---|---|---|---|
| Early | 0–83 | 62% | 31% | 37% | 1% |
| Middle | 84–167 | 67% | 17% | 20% | 15% |
| Late | 168–251 | 73% | 13% | 15% | 36% |
| After restart | 252–295 | 32% | 5% | 5% | 23% |

**Reward classifier** (ResNet-10, wrist + front, 5k steps on the 30-episode `…_6dof_ft_cropped_resized` set)
- On the 14 demo frames (in-sample, the training set is a superset): accuracy 95.5%, precision 92%, recall 79%, ROC AUC 0.993. Misses sit a median of 4 frames after the success label starts, so it fires slightly late, not wrongly.
- In the loop: it called success 209 times and the operator overrode 26 (**87.6% precision against the operator**). The operator never had to mark a success it missed.

## Why the post-restart episodes succeed less often

At 15:58 the actor crashed after a homing failure (`set_servo_angle` code 9). The learner kept training for about 8.8k optimization steps with no new data, then the robot, cameras and actor were restarted at 16:33. Late run vs after the restart (`data/processed/restart_significance.csv`, buffer episodes only):

| | Late (ep 168–251) | After restart (ep 252–295) | Fisher p |
|---|---|---|---|
| Success, all episodes | 61/83 = 73% | 14/43 = 33% | 2e-5 |
| Success, episodes nobody touched | 30/42 = 71% | 10/32 = 31% | 0.0009 |
| Success, episodes with intervention | 31/41 = 76% | 4/11 = 36% | 0.03 |
| Classifier success overridden by operator | 6/67 = 9% | 5/19 = 26% | 0.06 |
| Robot fault | 11/83 = 13% | 5/43 = 12% | 1.0 |

What this shows:
1. **The drop is real, and it's the policy, not the operator.** Success halves both with and without operator help, so it isn't explained by the operator stepping in less (intervention rate fell from 13% to 5%). Robot faults are unchanged.
2. **Something visual changed.** The reward classifier is frozen, yet its overridden-success rate roughly tripled. A model whose weights never changed can only get worse if its inputs changed, so this can't be the learner over-training. With 19 calls the evidence is borderline (p = 0.06).

What changed in the images (`figures/09_restart_diagnostics.png`, `figures/10_home_views.png`), measured on each episode's first frame at the fixed home pose:

| Measure | Early | Middle | Late | After restart | Reading |
|---|---|---|---|---|---|
| Front-camera wall brightness (median luma) | 0.525 | 0.514 | 0.504 | **0.475** | Gradual dimming through the afternoon (−4%), then a **sudden −6% step at the restart** |
| Left-finger x position in the wrist camera (px of 128) | 25.5 | 25.3 | 25.8 | **27.6** | Steady until ~ep 170, drifts during the fault-heavy late run, then **jumps at the restart** and stays |
| Gripper x position in the front camera (px of 128) | 47.6 | 47.7 | 45.8 | 45.8 | Arm home pose: shifted twice mid-run (ep ~163, ~203) while success kept rising; **no change at the restart** |
| Wrist-camera table brightness | 0.522 | 0.524 | 0.526 | 0.527 | No change |

Against your two hypotheses:
- **Lighting: yes, partly.** It dimmed gradually all afternoon, but success kept rising through that dimming. What stands out is the sudden step exactly at the restart. That pattern fits the cameras re-converging auto-exposure / white balance on startup better than ambient light. The D405 wrist camera only supports auto exposure (see the worklog), and the front D435 runs with `exposure: null`.
- **Arm calibration / home pose: no evidence.** The fixed front camera sees the arm return to the same home position after the restart. Earlier home-pose shifts of 2–3 px didn't hurt success.
- **What did change: the wrist camera's view of the gripper.** The fingers are rigid relative to the wrist camera, so their 1–2 px move means the camera-to-gripper geometry changed. That is either a mount shift or the gripper re-initialising to a slightly different open position at reconnect. It started drifting during the late run's robot faults (including the 15:49 self-collision) and settled at the new position after the restart.
- **Also visible:** after the restart, the object appears in one consistent spot at the top of the wrist home view (figure 10), so object placement at reset may have changed too. I couldn't quantify this cleanly.
- **Confound:** the policy that resumed had trained ~8.8k extra steps without new data (`figures/07_learner_curves.png`, shaded band). That could hurt the policy but not the classifier.

Bottom line: the drop coincides with a camera-side shift (exposure step and wrist camera vs gripper geometry), not with an arm calibration change. 44 episodes in under 30 minutes can't separate these causes. A clean test is to re-run the same checkpoint with camera exposure locked, after checking the wrist mount.

## Figures (`figures/`)

| File | Shows |
|---|---|
| `01_intervention_rate.png` | Per-episode intervention rate with 20-episode means, for all episodes and for successful episodes only |
| `02_outcome_mix.png` | Each 25-episode block split into autonomous success / assisted success / failure / robot fault |
| `03_success_and_autonomy.png` | Rolling success rate, zero-intervention success rate, share of episodes with any intervention |
| `04_cumulative_effort.png` | Cumulative policy-driven vs human-driven steps |
| `05_time_to_success.png` | Duration of successful episodes over time |
| `06_intervention_by_episode_progress.png` | Where in an episode the operator steps in, by phase (mostly the second half: the grasp/place) |
| `07_learner_curves.png` | Critic, actor, gripper-critic losses and SAC temperature vs optimization step |
| `08_reward_classifier.png` | Classifier probability around the success label on demos; online confidence before episode end |
| `09_restart_diagnostics.png` | Lighting, wrist camera vs gripper geometry, arm home pose and outcome tests around the restart |
| `10_home_views.png` | Average first frame per phase for both cameras, with a difference map |

## Data (`data/`)

`summary.json` has every headline number above plus the session and classifier configs.

`raw/` holds source data flattened to tables:
- `online_episodes.csv`: one row per online episode, from the actor logs. Columns include start/end time, length, intervention steps and rate, reward before and after operator relabel, success, autonomous success, robot fault and its reason, end reason, classifier-vs-operator agreement, learner optimization step at episode end, and whether the episode is in the replay buffer (3 of 296 aren't: one discarded at under 5 steps, two lost when actors crashed).
- `online_frames.csv`: one row per replay-buffer step (33,136): intervention flag, reward, done, gripper penalty, 7-D action.
- `online_events.csv`: actor starts and relabel key presses.
- `learner_metrics.csv`: wandb history every 10 optimization steps (losses, gradient norms, temperature, buffer sizes). `learner_opt_frequency.csv` maps time to optimization step.
- `demo_episodes.csv`, `demo_frames.csv`: the 6dof_ft demos plus the `_reviewed` (28) and `_unreviewed` (30) variants.
- `classifier_demo_frames.csv`, `classifier_online_frames.csv`: the classifier checkpoint re-run on every demo and buffer frame (success probability).
- `classifier_config.json`, `session_config.json`.

`processed/` holds analysis tables (each feeds a figure):
- `online_episodes_labeled.csv`: episodes with a phase label.
- `online_rolling.csv`, `online_blocks_25.csv` (with Wilson 95% CI on success), `online_phases.csv`.
- `intervention_by_episode_progress.csv`, `intervention_segments.csv` (746 contiguous takeovers, median 8 steps).
- `learner_curves.csv` (with EMA).
- `classifier_demo_threshold_sweep.csv`, `classifier_demo_frames_aligned.csv`, `classifier_online_tail.csv`, `demo_episodes_6dof_ft.csv`.
- `restart_*.csv/json`, `restart_episode_conditions.csv` (per-episode image, lighting and landmark measures), `demo_home_view_conditions.csv`, `home_views_mean.npz`.

## Definitions

- **Intervention rate**: steps where the SpaceMouse drove the arm ÷ episode length (as logged by the actor).
- **Success**: the episode's final reward is > 0 after any operator relabel (Enter/Backspace after the episode).
- **Autonomous success**: a success with zero intervention steps.
- **Classifier false positive**: the classifier ended the episode as a success and the operator relabeled it as a failure. Relabels are the only online ground truth.

## Regenerate

```bash
uv run python report/scripts/extract.py            # logs, buffer, wandb, demos -> data/raw
uv run python report/scripts/classifier_eval.py    # classifier over demo + buffer frames (GPU, ~30 s)
uv run python report/scripts/process.py            # -> data/processed, data/summary.json
uv run python report/scripts/restart_analysis.py   # post-restart diagnostics
uv run --with matplotlib python report/scripts/plots.py   # -> figures/
```

matplotlib isn't a project dependency, so `--with` pulls it in for that one run without changing `uv.lock`.
