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
Run a HIL-SERL policy checkpoint on the real robot and report its success rate.

The environment, processors and action selection are built exactly as in `lerobot.rl.actor`, without a
learner: nothing is trained or sent anywhere. Actions are sampled from the policy, as in the evaluation of
rail-berkeley/hil-serl (`argmax=False`). Success comes from the reward classifier or the teleop's success
key; you can still take over with the intervention key for safety (those episodes are reported).

Example:

```shell
uv run python examples/xarm_hilserl/eval_policy.py \
    --config_path=examples/xarm_hilserl/configs/train.json \
    --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model \
    --eval.n_episodes=10
```
"""

import logging
import time
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def select_action(
    policy: Any, preprocessor: Any, postprocessor: Any, observation: dict, cfg: TrainRLServerPipelineConfig
):
    """Same action selection as `lerobot.rl.actor.act_with_policy`."""
    with torch.no_grad():
        action = policy.select_action(batch=preprocessor.process_observation(observation))
        # Unnormalize only the continuous part.
        if cfg.policy.num_discrete_actions is not None:
            continuous_action = postprocessor.process_action(action[..., :-1])
            discrete_action = action[..., -1:].to(
                device=continuous_action.device, dtype=continuous_action.dtype
            )
            return torch.cat([continuous_action, discrete_action], dim=-1)
        return postprocessor.process_action(action)


@parser.wrap()
def eval_policy(cfg: TrainRLServerPipelineConfig):
    init_logging()
    if cfg.policy.pretrained_path is None:
        raise ValueError(
            "Pass the checkpoint to evaluate with `--policy.path=<checkpoint>/pretrained_model`."
        )

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, dataset_stats=cfg.policy.dataset_stats
    )

    completed, successes, intervened, success_times = 0, 0, 0, []
    try:
        for episode in range(cfg.eval.n_episodes):
            transition = reset_and_build_transition(env, env_processor, action_processor)
            episode_start = time.perf_counter()
            episode_intervened = False
            while True:
                step_start = time.perf_counter()
                observation = {
                    k: v
                    for k, v in transition[TransitionKey.OBSERVATION].items()
                    if k in cfg.policy.input_features
                }
                action = select_action(policy, preprocessor, postprocessor, observation, cfg)
                transition = step_env_and_process_transition(
                    env=env,
                    transition=transition,
                    action=action,
                    env_processor=env_processor,
                    action_processor=action_processor,
                )
                episode_intervened |= bool(
                    transition[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False)
                )

                if transition.get(TransitionKey.DONE, False) or transition.get(
                    TransitionKey.TRUNCATED, False
                ):
                    break
                precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - step_start), 0.0))

            success = float(transition[TransitionKey.REWARD]) > 0
            elapsed = time.perf_counter() - episode_start
            completed += 1
            successes += int(success)
            intervened += int(episode_intervened)
            if success:
                success_times.append(elapsed)
            logging.info(
                f"Episode {episode + 1}/{cfg.eval.n_episodes}: {'success' if success else 'failure'} in {elapsed:.1f}s"
                + (" (with intervention)" if episode_intervened else "")
            )
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted.")
    finally:
        if env.robot.is_connected:
            env.robot.disconnect()
        if teleop_device is not None and teleop_device.is_connected:
            teleop_device.disconnect()

    if completed:
        mean_time = sum(success_times) / len(success_times) if success_times else float("nan")
        logging.info(
            f"Success rate {successes}/{completed}, mean time to success {mean_time:.1f}s, "
            f"episodes with intervention {intervened}"
        )


def main():
    eval_policy()


if __name__ == "__main__":
    main()
