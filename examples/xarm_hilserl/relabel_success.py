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
Relabel a reward-classifier dataset so that every frame after the first success in an episode is a success.

Datasets recorded before success latching (by holding Enter) only have `next.reward=1` on scattered frames.
This writes a copy where `next.reward` is 1 from the first success frame to the end of each episode;
episodes without any success frame are left as failures. The source dataset is not modified.

Example:

```shell
uv run python examples/xarm_hilserl/relabel_success.py \
    --repo-id local/xarm7_hilserl_classifier_cropped_resized
```
"""

import argparse
import shutil

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, REWARD


def relabel_parquet(path) -> list[tuple[int, int, int]]:
    """Latch `next.reward` to 1 after the first success in each episode. Returns (episode, before, after)."""
    df = pd.read_parquet(path)
    df = df.sort_values(["episode_index", "frame_index"])
    rewards = np.stack(df[REWARD].to_numpy()).reshape(-1)
    episodes = df["episode_index"].to_numpy()

    new_rewards = rewards.copy()
    summary = []
    for ep in np.unique(episodes):
        mask = episodes == ep
        new_rewards[mask] = np.maximum.accumulate(rewards[mask])
        summary.append((int(ep), int((rewards[mask] > 0).sum()), int((new_rewards[mask] > 0).sum())))

    df[REWARD] = new_rewards.astype(np.float32)
    df.to_parquet(path, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", default="local/xarm7_hilserl_classifier_cropped_resized")
    parser.add_argument("--new-repo-id", default=None, help="Defaults to `<repo-id>_relabeled`.")
    args = parser.parse_args()

    new_repo_id = args.new_repo_id or f"{args.repo_id}_relabeled"
    src_root = HF_LEROBOT_HOME / args.repo_id
    dst_root = HF_LEROBOT_HOME / new_repo_id
    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists, remove it or pass another --new-repo-id")
    shutil.copytree(src_root, dst_root)

    for path in sorted((dst_root / "data").glob("*/*.parquet")):
        for ep, before, after in relabel_parquet(path):
            print(f"episode {ep:3d}: success frames {before:4d} -> {after:4d}")

    dataset = recompute_stats(LeRobotDataset(new_repo_id, root=dst_root))
    rewards = np.array(dataset.hf_dataset[REWARD]).reshape(-1)
    print(
        f"\nSaved {new_repo_id}: {dataset.num_episodes} episodes, {len(rewards)} frames, "
        f"{int((rewards > 0).sum())} success ({(rewards > 0).mean():.1%})"
    )


if __name__ == "__main__":
    main()
