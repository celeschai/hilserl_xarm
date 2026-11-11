"""Re-run the reward classifier checkpoint over recorded frames and save per-frame success probabilities.

- Demo frames (local/xarm7_hilserl_demos_6dof_ft): labels are the operator's success labels (next.reward).
  The classifier was trained on a 30-episode superset of these demos, so this is a training-fit check.
- Online replay buffer frames: labels are the final per-episode verdict after operator relabels. The in-loop
  classifier scored the observation *after* each action; the buffer stores the observation *before* it, so
  the frame that triggered a success is not in the buffer.

Run: uv run python report/scripts/classifier_eval.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.rewards.classifier.modeling_classifier import Classifier

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "report" / "data" / "raw"
CKPT = REPO / "outputs/train/xarm7_reward_classifier/checkpoints/last/pretrained_model"
BUFFER = REPO / "outputs/train/2026-11-09/14-19-43_xarm7_hilserl/dataset/data"
DEMO_REPO = "local/xarm7_hilserl_demos_6dof_ft"
KEYS = ["observation.images.wrist", "observation.images.front"]
BATCH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier():
    model = Classifier.from_pretrained(str(CKPT)).to(DEVICE).eval()
    pre = DataProcessorPipeline.from_pretrained(
        str(CKPT),
        config_filename="classifier_preprocessor.json",
        overrides={"device_processor": {"device": DEVICE}},
    )
    return model, pre


@torch.inference_mode()
def score(model, pre, images: dict[str, torch.Tensor]) -> np.ndarray:
    batch = pre({k: v.to(DEVICE) for k, v in images.items()})
    return model.predict([batch[k] for k in KEYS]).probabilities.float().cpu().numpy()


def eval_demos(model, pre) -> pd.DataFrame:
    ds = LeRobotDataset(DEMO_REPO)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, num_workers=4)
    rows = []
    for b in loader:
        p = score(model, pre, {k: b[k] for k in KEYS})
        rows.append(
            pd.DataFrame(
                {
                    "episode": b["episode_index"].numpy(),
                    "frame_index": b["frame_index"].numpy(),
                    "label": b["next.reward"].numpy().ravel(),
                    "prob_success": p,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def decode(col) -> torch.Tensor:
    arr = np.stack([np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB")) for c in col])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


def eval_buffer(model, pre) -> pd.DataFrame:
    rows = []
    for f in sorted(BUFFER.glob("chunk-*/file-*.parquet")):
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=BATCH, columns=["episode_index", "frame_index", *KEYS]):
            d = rb.to_pydict()
            p = score(model, pre, {k: decode(d[k]) for k in KEYS})
            rows.append(
                pd.DataFrame(
                    {"buffer_episode": d["episode_index"], "frame_index": d["frame_index"], "prob_success": p}
                )
            )
        print(f"scored {f.name}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    model, pre = load_classifier()
    demos = eval_demos(model, pre)
    demos.to_csv(OUT / "classifier_demo_frames.csv", index=False)
    print(f"demo frames: {len(demos)}")

    buf = eval_buffer(model, pre)
    frames = pd.read_csv(OUT / "online_frames.csv", usecols=["buffer_episode", "frame_index", "episode"])
    buf = buf.merge(frames, on=["buffer_episode", "frame_index"], how="left")
    buf.to_csv(OUT / "classifier_online_frames.csv", index=False)
    print(f"online frames: {len(buf)}")


if __name__ == "__main__":
    main()
