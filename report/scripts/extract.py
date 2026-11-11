"""Extract raw HIL-SERL artifacts into flat CSV / JSON tables under report/data/raw.

Sources
- Actor logs (one per actor process) of the 2026-11-09 online session: per-episode outcome, length,
  intervention steps, robot faults and operator relabels.
- Online replay buffer saved by the learner (LeRobotDataset, images embedded): per-frame intervention flag,
  reward, done and action.
- Learner wandb binaries: losses, temperature, buffer sizes per optimization step.
- Offline demo datasets (6dof_ft, plus the reviewed / unreviewed variants): per-episode and per-frame stats.
- Reward classifier checkpoint: config + training config summary.

Run: uv run python report/scripts/extract.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "report" / "data" / "raw"

SESSION_DIR = REPO / "outputs/train/2026-11-09"
LEARNER_RUN = SESSION_DIR / "14-19-43_xarm7_hilserl"
# Actor processes that streamed to the learner above, in time order.
ACTOR_RUNS = [
    "14-20-08_xarm7_hilserl",
    "16-33-17_xarm7_hilserl",
    "16-44-43_xarm7_hilserl",
    "16-56-38_xarm7_hilserl",
]
CLASSIFIER = REPO / "outputs/train/xarm7_reward_classifier/checkpoints/last"
DEMO_ROOT = Path.home() / ".cache/huggingface/lerobot/local"
DEMO_SETS = [
    "xarm7_hilserl_demos_6dof_ft",
    "xarm7_hilserl_demos_6dof_ft_reviewed",
    "xarm7_hilserl_demos_6dof_ft_unreviewed",
]
LOCAL_TZ = "Asia/Singapore"  # Logs are written in local time (+08:00)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = r"^(?P<level>\w+) (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ "
RE_END = re.compile(
    TS + r"\[EPISODE END\] step (?P<step>\d+) \| reward (?P<reward>[+-][\d.]+) \| "
    r"last-step reward (?P<last>[+-][\d.]+) \| len (?P<len>\d+) \| "
    r"intervention (?P<iv>\d+)/(?P<len2>\d+)(?P<fault> \| robot fault)?"
)
RE_START = re.compile(TS + r"\[EPISODE START\] step (?P<step>\d+)")
RE_RELABEL = re.compile(
    TS
    + r"\[EPISODE RELABEL\] (?P<key>success|failure) pressed (?P<when>.+?): .*reward (?P<reward>[+-][\d.]+)"
)
RE_DISCARD = re.compile(TS + r"\[EPISODE DISCARDED\] len (?P<len>\d+)")
RE_FAULT = re.compile(TS + r".*Robot fault, ending the episode as failed: (?P<reason>[^.]+)")
RE_LOG_START = re.compile(TS + r"Actor logging initialized")


def parse_actor_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes, events = [], []
    for session_idx, run in enumerate(ACTOR_RUNS):
        log = SESSION_DIR / run / "logs" / "actor_xarm7_hilserl.log"
        session_start = None
        ep_start_ts = None
        pending_fault = None
        session_ep = 0
        for raw in log.read_text(errors="replace").splitlines():
            line = ANSI.sub("", raw)
            if m := RE_LOG_START.match(line):
                session_start = ep_start_ts = m["ts"]
            elif m := RE_FAULT.match(line):
                pending_fault = m["reason"].strip()
            elif m := RE_START.match(line):
                ep_start_ts = m["ts"]
            elif m := RE_END.match(line):
                reward = float(m["reward"])
                episodes.append(
                    {
                        "session": session_idx,
                        "actor_run": run,
                        "session_episode": session_ep,
                        "start_time": ep_start_ts,
                        "end_time": m["ts"],
                        "session_step_end": int(m["step"]),
                        "length": int(m["len"]),
                        "intervention_steps": int(m["iv"]),
                        "reward_at_end": reward,
                        "last_step_reward": float(m["last"]),
                        "robot_fault": m["fault"] is not None,
                        "fault_reason": pending_fault if m["fault"] else None,
                        "relabel": None,
                        "relabel_when": None,
                        "final_reward": reward,
                        "discarded": False,
                    }
                )
                pending_fault = None
                session_ep += 1
            elif m := RE_RELABEL.match(line):
                ep = episodes[-1]
                ep["relabel"], ep["relabel_when"] = m["key"], m["when"]
                ep["final_reward"] = float(m["reward"])
                events.append({"time": m["ts"], "actor_run": run, "type": "relabel", "detail": m["key"]})
            elif m := RE_DISCARD.match(line):
                episodes[-1]["discarded"] = True
        events.append({"time": session_start, "actor_run": run, "type": "actor_start", "detail": ""})

    df = pd.DataFrame(episodes)
    df["episode"] = np.arange(len(df))
    for col in ["start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(LOCAL_TZ)
    df["intervention_rate"] = df["intervention_steps"] / df["length"]
    df["success"] = df["final_reward"] > 0
    df["classifier_success"] = df["reward_at_end"] > 0
    df["any_intervention"] = df["intervention_steps"] > 0
    df["autonomous_success"] = df["success"] & ~df["any_intervention"]
    df["cumulative_env_steps"] = df["length"].cumsum()
    t0 = df["start_time"].iloc[0]
    df["wall_clock_min"] = (df["end_time"] - t0).dt.total_seconds() / 60
    df["end_reason"] = np.select(
        [df["robot_fault"], df["success"], df["length"] >= 900],
        ["robot_fault", "success", "timeout"],
        default="operator_failure",
    )
    # How the reward classifier's call at episode end compared with the operator's final verdict.
    # Relabels are the only evidence of classifier mistakes; unrelabeled episodes count as agreements.
    df["classifier_vs_operator"] = np.select(
        [
            df["classifier_success"] & df["success"],
            df["classifier_success"] & ~df["success"],
            ~df["classifier_success"] & df["success"],
        ],
        ["agree_success", "false_positive", "missed_success"],
        default="agree_failure",
    )
    order = ["episode", "session", "actor_run", "session_episode", "start_time", "end_time", "wall_clock_min"]
    df = df[order + [c for c in df.columns if c not in order]]
    ev = pd.DataFrame(events)
    return df, ev


def extract_online_buffer(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per-frame table of the learner's online replay buffer (no images)."""
    cols = [
        "episode_index",
        "frame_index",
        "timestamp",
        "action",
        "next.reward",
        "next.done",
        "complementary_info.discrete_penalty",
        "complementary_info.is_intervention",
    ]
    files = sorted((LEARNER_RUN / "dataset/data").glob("chunk-*/file-*.parquet"))
    frames = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    act = np.stack(frames.pop("action").to_numpy())
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    for i, n in enumerate(names):
        frames[f"action_{n}"] = act[:, i]
    frames = frames.rename(
        columns={
            "episode_index": "buffer_episode",
            "next.reward": "reward",
            "next.done": "done",
            "complementary_info.discrete_penalty": "gripper_penalty",
            "complementary_info.is_intervention": "is_intervention",
        }
    )
    for c in ["reward", "gripper_penalty", "is_intervention"]:
        frames[c] = frames[c].map(lambda v: float(np.asarray(v).ravel()[0]))
    frames["done"] = frames["done"].map(lambda v: bool(np.asarray(v).ravel()[0]))
    frames["is_intervention"] = frames["is_intervention"].astype(bool)
    return frames.sort_values(["buffer_episode", "frame_index"]).reset_index(drop=True)


def align_buffer_to_log(episodes: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Map each buffer episode to its actor-log episode by matching (length, intervention steps) in order."""
    buf = frames.groupby("buffer_episode").agg(length=("frame_index", "size"), iv=("is_intervention", "sum"))
    log_keys = list(zip(episodes["length"], episodes["intervention_steps"], strict=True))
    mapping, j = {}, 0
    for b, row in buf.iterrows():
        while j < len(log_keys) and log_keys[j] != (row["length"], row["iv"]):
            j += 1
        if j == len(log_keys):
            raise RuntimeError(f"buffer episode {b} has no matching log episode")
        mapping[b] = j
        j += 1
    frames["episode"] = frames["buffer_episode"].map(mapping)
    return frames


def read_wandb(path: Path) -> list[dict]:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    rows = []
    while (data := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        rows.append(
            {
                (i.key or "/".join(i.nested_key)).removeprefix("train/"): json.loads(i.value_json)
                for i in rec.history.item
            }
        )
    return rows


def extract_learner_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, run in enumerate(sorted((LEARNER_RUN / "wandb").glob("run-*/run-*.wandb"))):
        for r in read_wandb(run):
            r["wandb_segment"] = i
            rows.append(r)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["_timestamp"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    loss_cols = [
        "loss_critic",
        "loss_actor",
        "loss_temperature",
        "loss_discrete_critic",
        "temperature",
        "critic_grad_norm",
        "actor_grad_norm",
        "discrete_critic_grad_norm",
        "temperature_grad_norm",
        "replay_buffer_size",
        "offline_replay_buffer_size",
    ]
    learner = df.dropna(subset=["loss_critic"])[["time", "wandb_segment", "Optimization step", *loss_cols]]
    learner = learner.rename(columns={"Optimization step": "optimization_step"}).reset_index(drop=True)
    opt_freq = df.dropna(subset=["Optimization frequency loop [Hz]"])
    learner_freq = (
        opt_freq[["time", "Optimization step", "Optimization frequency loop [Hz]"]]
        .rename(
            columns={"Optimization step": "optimization_step", "Optimization frequency loop [Hz]": "opt_hz"}
        )
        .iloc[::10]
        .reset_index(drop=True)
    )
    return learner, learner_freq


def attach_learner_step(episodes: pd.DataFrame, learner_freq: pd.DataFrame) -> pd.DataFrame:
    """Learner optimization step at the moment each episode ended (policy the actor was syncing to)."""
    lf = learner_freq.sort_values("time")[["time", "optimization_step"]]
    ep = episodes.sort_values("end_time")
    merged = pd.merge_asof(ep, lf, left_on="end_time", right_on="time", direction="backward")
    episodes["learner_opt_step_at_end"] = merged["optimization_step"].fillna(0).astype(int).to_numpy()
    return episodes


def extract_demos() -> tuple[pd.DataFrame, pd.DataFrame]:
    ep_rows, frame_tables = [], []
    for name in DEMO_SETS:
        root = DEMO_ROOT / name
        d = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
        act = np.stack(d["action"].to_numpy())
        f = pd.DataFrame(
            {
                "dataset": name,
                "episode": d["episode_index"],
                "frame_index": d["frame_index"],
                "timestamp": d["timestamp"],
                "reward": d["next.reward"].astype(float),
                "done": d["next.done"].astype(bool),
                "gripper_penalty": d["complementary_info.discrete_penalty"].map(
                    lambda v: float(np.asarray(v).ravel()[0])
                ),
            }
        )
        for i, n in enumerate(["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]):
            f[f"action_{n}"] = act[:, i]
        frame_tables.append(f)
        lin = np.linalg.norm(act[:, :3], axis=1)
        f["lin_norm"] = lin
        g = f.groupby("episode")
        for ep, e in g:
            ep_rows.append(
                {
                    "dataset": name,
                    "episode": ep,
                    "length": len(e),
                    "duration_s": float(e["timestamp"].max() + 0.1),
                    "success_frames": int((e["reward"] > 0).sum()),
                    "first_success_frame": int(e.loc[e["reward"] > 0, "frame_index"].min())
                    if (e["reward"] > 0).any()
                    else None,
                    "success": bool((e["reward"] > 0).any()),
                    "gripper_actions": int((e["gripper_penalty"] < 0).sum()),
                    "mean_linear_action": float(e["lin_norm"].mean()),
                }
            )
    return pd.DataFrame(ep_rows), pd.concat(frame_tables, ignore_index=True)


def classifier_summary() -> dict:
    cfg = json.loads((CLASSIFIER / "pretrained_model/config.json").read_text())
    tcfg = json.loads((CLASSIFIER / "pretrained_model/train_config.json").read_text())
    step = json.loads((CLASSIFIER / "training_state/training_step.json").read_text())
    return {
        "checkpoint": str(CLASSIFIER.relative_to(REPO)),
        "backbone": cfg["model_name"],
        "cameras": list(cfg["input_features"]),
        "image_shape": cfg["input_features"]["observation.images.front"]["shape"],
        "hidden_dim": cfg["hidden_dim"],
        "dropout": cfg["dropout_rate"],
        "train_dataset": tcfg["dataset"]["repo_id"],
        "image_augmentation": tcfg["dataset"]["image_transforms"]["enable"],
        "steps": step["step"],
        "batch_size": step["batch_size"],
        "lr": tcfg["optimizer"]["lr"],
        "weight_decay": tcfg["optimizer"]["weight_decay"],
        "seed": tcfg["seed"],
    }


def session_config() -> dict:
    c = json.loads((LEARNER_RUN / "checkpoints/last/pretrained_model/train_config.json").read_text())
    st = json.loads((LEARNER_RUN / "checkpoints/last/training_state/training_step.json").read_text())
    env, proc, pol = c["env"], c["env"]["processor"], c["policy"]
    return {
        "learner_run": str(LEARNER_RUN.relative_to(REPO)),
        "actor_runs": ACTOR_RUNS,
        "offline_demo_dataset": c["dataset"]["repo_id"],
        "reward_classifier": proc["reward_classifier"]["pretrained_path"],
        "success_threshold": proc["reward_classifier"]["success_threshold"],
        "policy_type": pol["type"],
        "fps": env["fps"],
        "max_episode_s": proc["reset"]["control_time_s"],
        "gripper_penalty": proc["gripper"]["gripper_penalty"],
        "action_dims": env["features"]["action"]["shape"][0],
        "teleop": env["teleop"]["type"],
        "admittance_control": env["robot"]["use_admittance"],
        "ft_sensor": env["robot"]["use_ft_sensor"],
        "batch_size": c["batch_size"],
        "final_optimization_step": st["step"],
        "final_interaction_step": st["interaction_step"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    episodes, events = parse_actor_logs()
    learner, learner_freq = extract_learner_metrics()
    episodes = attach_learner_step(episodes, learner_freq)
    frames = align_buffer_to_log(episodes, extract_online_buffer(episodes))
    episodes["in_replay_buffer"] = episodes["episode"].isin(frames["episode"].unique())

    episodes.to_csv(OUT / "online_episodes.csv", index=False)
    events.to_csv(OUT / "online_events.csv", index=False)
    frames.to_csv(OUT / "online_frames.csv", index=False)
    learner.to_csv(OUT / "learner_metrics.csv", index=False)
    learner_freq.to_csv(OUT / "learner_opt_frequency.csv", index=False)

    demo_eps, demo_frames = extract_demos()
    demo_eps.to_csv(OUT / "demo_episodes.csv", index=False)
    demo_frames.to_csv(OUT / "demo_frames.csv", index=False)

    (OUT / "classifier_config.json").write_text(json.dumps(classifier_summary(), indent=2))
    (OUT / "session_config.json").write_text(json.dumps(session_config(), indent=2))

    print(f"online episodes: {len(episodes)} (in buffer: {episodes['in_replay_buffer'].sum()})")
    print(f"online frames: {len(frames)}  learner rows: {len(learner)}  demo episodes: {len(demo_eps)}")


if __name__ == "__main__":
    main()
