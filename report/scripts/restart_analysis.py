"""Why do episodes after the 15:58 actor crash succeed less often? Quantify the candidate causes.

Per episode it measures, from the replay buffer's first frame (taken at the fixed home pose):
- lighting: mean luma, per-channel means and contrast of each camera image;
- scene geometry: how far the brightness-normalized image is from the reference home view (late phase of
  the main session), which moves if the home pose / calibration / camera / objects shift;
- F/T sensor reading and gripper position at the home pose.
The same image stats are computed on the offline demos (what the policy and reward classifier trained on).
It then lines these up with policy-side factors: success without any operator help, operator behavior,
learner steps trained with no new data, and reward-classifier overrides.

Run after process.py: uv run python report/scripts/restart_analysis.py
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "report" / "data" / "raw"
PROC = REPO / "report" / "data" / "processed"
BUFFER = REPO / "outputs/train/2026-11-09/14-19-43_xarm7_hilserl/dataset/data"
DEMO_REPO = "local/xarm7_hilserl_demos_6dof_ft"
CAMS = {"wrist": "observation.images.wrist", "front": "observation.images.front"}
STATE = ["x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz"]
STATE += ["fx", "fy", "fz", "tx", "ty", "tz", "gripper"]


def image_stats(img: np.ndarray) -> dict:
    """img: HxWx3 float in [0, 1]."""
    luma = img @ np.array([0.299, 0.587, 0.114])
    return {
        "luma": luma.mean(),
        "contrast": luma.std(),
        "r": img[..., 0].mean(),
        "g": img[..., 1].mean(),
        "b": img[..., 2].mean(),
    }


def landmarks(wrist: np.ndarray, front: np.ndarray) -> dict:
    """Fixed reference points in the home view (pixel coords in the 128x128 image).

    - Gripper fingers in the wrist camera: rigidly mounted relative to the camera, so if they move in the
      image the camera-to-gripper geometry changed (mount shift or different gripper open position).
    - Gripper in the front camera: the camera is fixed, so this moves if the arm homes to a different pose.
    - A patch of bare wall in the front camera: brightness there is ambient lighting, not scene content.
    """
    luma = np.array([0.299, 0.587, 0.114])
    w, f = wrist @ luma, front @ luma
    out = {}
    regions = {
        "wrist_lfinger": (w, slice(80, 128), slice(0, 50)),
        "wrist_rfinger": (w, slice(80, 128), slice(78, 128)),
        "front_gripper": (f, slice(10, 80), slice(25, 75)),
    }
    for name, (img, ys, xs) in regions.items():
        dark = img[ys, xs] < 0.25
        yy, xx = np.nonzero(dark)
        ok = dark.sum() > 20
        out[f"{name}_y"] = yy.mean() + ys.start if ok else np.nan
        out[f"{name}_x"] = xx.mean() + xs.start if ok else np.nan
    out["front_wall_luma"] = f[2:30, 95:126].mean()
    out["wrist_table_luma"] = w[30:70, 20:108].mean()
    return out


def fisher_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    from math import comb

    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)  # noqa: E731
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 + 1e-12)


def normalized(img: np.ndarray) -> np.ndarray:
    """Grayscale, zero-mean / unit-variance: removes global brightness and contrast, keeps geometry."""
    g = img @ np.array([0.299, 0.587, 0.114])
    return (g - g.mean()) / (g.std() + 1e-6)


def buffer_first_frames() -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]]]:
    cols = ["episode_index", "frame_index", "observation.state", *CAMS.values()]
    rows, imgs = [], {c: {} for c in CAMS}
    for f in sorted(BUFFER.glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=cols, filters=[("frame_index", "=", 0)]).to_pydict()
        for i, ep in enumerate(t["episode_index"]):
            row = {"buffer_episode": ep}
            row.update(
                {
                    f"state0_{k}": v
                    for k, v in zip(STATE, np.asarray(t["observation.state"][i]).ravel(), strict=True)
                }
            )
            for cam, key in CAMS.items():
                img = np.asarray(Image.open(io.BytesIO(t[key][i]["bytes"])).convert("RGB"), dtype=float) / 255
                imgs[cam][ep] = img
                row.update({f"{cam}_{k}": v for k, v in image_stats(img).items()})
            row.update(landmarks(imgs["wrist"][ep], imgs["front"][ep]))
            rows.append(row)
    return pd.DataFrame(rows), imgs


def buffer_episode_ft() -> pd.DataFrame:
    """Mean force / torque magnitude over each episode (contact intensity)."""
    out = []
    for f in sorted(BUFFER.glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "observation.state"]).to_pandas()
        s = np.stack(t["observation.state"].to_numpy())
        t["force"] = np.linalg.norm(s[:, 12:15], axis=1)
        t["torque"] = np.linalg.norm(s[:, 15:18], axis=1)
        out.append(t[["episode_index", "force", "torque"]])
    d = pd.concat(out)
    return (
        d.groupby("episode_index")
        .agg(mean_force=("force", "mean"), max_force=("force", "max"), mean_torque=("torque", "mean"))
        .rename_axis("buffer_episode")
        .reset_index()
    )


def demo_first_frames() -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    ds = LeRobotDataset(DEMO_REPO)
    starts = ds.meta.episodes["dataset_from_index"]
    rows, imgs = [], {c: [] for c in CAMS}
    for ep, idx in enumerate(starts):
        item = ds[int(idx)]
        row = {"demo_episode": ep}
        for cam, key in CAMS.items():
            img = item[key].permute(1, 2, 0).numpy().astype(float)
            imgs[cam].append(img)
            row.update({f"{cam}_{k}": v for k, v in image_stats(img).items()})
        rows.append(row)
    return pd.DataFrame(rows), imgs


def main() -> None:
    ep = pd.read_csv(PROC / "online_episodes_labeled.csv")
    frames0, imgs = buffer_first_frames()
    frames0 = frames0.merge(buffer_episode_ft(), on="buffer_episode")
    bmap = pd.read_csv(RAW / "online_frames.csv", usecols=["buffer_episode", "episode"]).drop_duplicates()
    frames0 = frames0.merge(bmap, on="buffer_episode")
    d = ep.merge(frames0, on="episode", how="inner")
    d["group"] = d["phase"]
    d["start_hour"] = pd.to_datetime(d["start_time"]).dt.strftime("%H:%M")

    # Geometry: distance of each brightness-normalized home view from the late-phase mean home view.
    late_eps = d.loc[d["phase"] == "late", "buffer_episode"]
    for cam in CAMS:
        ref = np.mean([normalized(imgs[cam][b]) for b in late_eps], axis=0)
        d[f"{cam}_geometry_diff"] = [
            np.sqrt(np.mean((normalized(imgs[cam][b]) - ref) ** 2)) for b in d["buffer_episode"]
        ]

    demo, demo_imgs = demo_first_frames()
    for cam in CAMS:
        ref = np.mean([normalized(imgs[cam][b]) for b in late_eps], axis=0)
        demo[f"{cam}_geometry_diff"] = [np.sqrt(np.mean((normalized(i) - ref) ** 2)) for i in demo_imgs[cam]]

    d.to_csv(PROC / "restart_episode_conditions.csv", index=False)
    demo.to_csv(PROC / "demo_home_view_conditions.csv", index=False)

    # Home-view images averaged per group, for a visual side-by-side.
    np.savez_compressed(
        PROC / "home_views_mean.npz",
        **{
            f"{cam}_{g}": np.mean([imgs[cam][b] for b in d.loc[d["group"] == g, "buffer_episode"]], axis=0)
            for cam in CAMS
            for g in ["early", "middle", "late", "after_restart"]
        },
        **{f"{cam}_demos": np.mean(demo_imgs[cam], axis=0) for cam in CAMS},
    )

    metrics = {
        "episodes": ("episode", "size"),
        "start": ("start_hour", "min"),
        "end": ("start_hour", "max"),
        "success_rate": ("success", "mean"),
        "autonomous_success_rate": ("autonomous_success", "mean"),
        "intervention_rate": ("intervention_rate", "mean"),
        "robot_fault_rate": ("robot_fault", "mean"),
        "wrist_luma": ("wrist_luma", "mean"),
        "front_luma": ("front_luma", "mean"),
        "front_b_minus_r": ("front_b", "mean"),
        "wrist_geometry_diff": ("wrist_geometry_diff", "median"),
        "front_geometry_diff": ("front_geometry_diff", "median"),
        "home_fz": ("state0_fz", "mean"),
        "home_force_abs": ("state0_fx", lambda s: np.nan),
        "mean_force": ("mean_force", "mean"),
        "home_gripper": ("state0_gripper", "mean"),
    }
    metrics.pop("home_force_abs")
    g = d.groupby("group").agg(**metrics).reindex(["early", "middle", "late", "after_restart"])
    g["front_b_minus_r"] = d.groupby("group").apply(
        lambda x: (x["front_b"] - x["front_r"]).mean(), include_groups=False
    )
    g["home_force_norm"] = d.groupby("group").apply(
        lambda x: np.linalg.norm(x[["state0_fx", "state0_fy", "state0_fz"]].to_numpy(), axis=1).mean(),
        include_groups=False,
    )
    g = g.drop(columns=["home_fz"])
    for c in [
        "front_wall_luma",
        "wrist_table_luma",
        "wrist_lfinger_x",
        "wrist_lfinger_y",
        "front_gripper_x",
        "front_gripper_y",
    ]:
        g[f"{c}_median"] = d.groupby("group")[c].median()

    # Policy quality without the operator in the loop: success rate among episodes nobody touched.
    untouched = d[~d["any_intervention"]]
    g["success_rate_untouched_eps"] = untouched.groupby("group")["success"].mean()
    g["untouched_eps"] = untouched.groupby("group").size()
    g["classifier_overrides_per_success_call"] = d.groupby("group").apply(
        lambda x: (x["classifier_vs_operator"] == "false_positive").sum()
        / max(int(x["classifier_success"].sum()), 1),
        include_groups=False,
    )
    g["failure_len_s_median"] = d[~d["success"]].groupby("group")["length"].median() / 10
    g["learner_opt_steps_at_start"] = d.groupby("group")["learner_opt_step_at_end"].min()

    demo_row = {
        "episodes": len(demo),
        "wrist_luma": demo["wrist_luma"].mean(),
        "front_luma": demo["front_luma"].mean(),
        "front_b_minus_r": (demo["front_b"] - demo["front_r"]).mean(),
        "wrist_geometry_diff": demo["wrist_geometry_diff"].median(),
        "front_geometry_diff": demo["front_geometry_diff"].median(),
    }
    g.loc["offline_demos"] = pd.Series(demo_row)
    g.to_csv(PROC / "restart_comparison.csv")

    # Per actor process (each one is a fresh robot + camera startup).
    s = d.groupby("actor_run").agg(
        episodes=("episode", "size"),
        start=("start_hour", "min"),
        success_rate=("success", "mean"),
        autonomous_success_rate=("autonomous_success", "mean"),
        intervention_rate=("intervention_rate", "mean"),
        wrist_luma=("wrist_luma", "mean"),
        front_luma=("front_luma", "mean"),
        wrist_geometry_diff=("wrist_geometry_diff", "median"),
        front_geometry_diff=("front_geometry_diff", "median"),
    )
    s.to_csv(PROC / "restart_by_actor_startup.csv")

    # Within the main session: do lighting / geometry predict success at all?
    m = d[d["session"] == 0]
    corr = {
        f"{c}_vs_success": float(np.corrcoef(m[c], m["success"].astype(float))[0, 1])
        for c in ["wrist_luma", "front_luma", "wrist_geometry_diff", "front_geometry_diff", "wall_clock_min"]
    }
    (PROC / "restart_correlations_main_session.json").write_text(json.dumps(corr, indent=2))

    # Late phase of the main session vs after the restart, as counts with Fisher exact tests.
    late, after = d[d["phase"] == "late"], d[d["phase"] == "after_restart"]
    tests = [
        ("success, all episodes", lambda x: x, lambda x: x["success"]),
        (
            "success, episodes with no intervention",
            lambda x: x[~x["any_intervention"]],
            lambda x: x["success"],
        ),
        ("success, episodes with intervention", lambda x: x[x["any_intervention"]], lambda x: x["success"]),
        (
            "classifier success overridden by operator",
            lambda x: x[x["classifier_success"]],
            lambda x: x["classifier_vs_operator"] == "false_positive",
        ),
        ("robot fault", lambda x: x, lambda x: x["robot_fault"]),
    ]
    rows = []
    for name, sel, hit in tests:
        a, b = sel(late), sel(after)
        k1, k2 = int(hit(a).sum()), int(hit(b).sum())
        rows.append(
            {
                "measure": name,
                "late_k": k1,
                "late_n": len(a),
                "late_rate": k1 / len(a),
                "after_restart_k": k2,
                "after_restart_n": len(b),
                "after_restart_rate": k2 / len(b),
                "fisher_p": fisher_p(k1, len(a) - k1, k2, len(b) - k2),
            }
        )
    pd.DataFrame(rows).to_csv(PROC / "restart_significance.csv", index=False)

    pd.set_option("display.width", 250)
    print(g.T.round(3).to_string())
    print(s.round(3).to_string())
    print(json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
