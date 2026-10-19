"""Draw the crop boxes from a config on frames sampled across every episode of a dataset.

Writes one contact sheet per camera: a row per episode, columns at evenly spaced points in time
(start ... end). Check that everything the task needs stays inside the box in every row.

    uv run python examples/xarm_hilserl/check_crop.py --repo-id local/xarm7_hilserl_demos \
        --config examples/xarm_hilserl/configs/env.json --out-dir outputs/crop_check
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", required=True, help="env.json or train.json holding crop_params_dict")
    parser.add_argument("--out-dir", default="outputs/crop_check")
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()

    crops = json.loads(Path(args.config).read_text())["env"]["processor"]["image_preprocessing"][
        "crop_params_dict"
    ]
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, list[np.ndarray]] = {key: [] for key in crops}
    for ep in range(dataset.meta.total_episodes):
        start = dataset.meta.episodes["dataset_from_index"][ep]
        end = dataset.meta.episodes["dataset_to_index"][ep]
        indices = np.linspace(start, end - 1, args.frames_per_episode).round().astype(int)
        thumbs: dict[str, list[np.ndarray]] = {key: [] for key in crops}
        for i, idx in enumerate(indices):
            item = dataset[int(idx)]
            for key, (top, left, height, width) in crops.items():
                img = (item[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 3)
                label = f"ep {ep}  t={int(idx - start)}/{end - start - 1}"
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                scale = args.thumb_width / img.shape[1]
                thumbs[key].append(cv2.resize(img, None, fx=scale, fy=scale))
        for key in crops:
            sheets[key].append(np.hstack(thumbs[key]))
        print(f"episode {ep} done")

    for key, rows in sheets.items():
        path = out_dir / f"{key.split('.')[-1]}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
