"""Turn report/data/raw tables into analysis-ready tables (report/data/processed) and headline numbers
(report/data/summary.json).

Run after extract.py and classifier_eval.py: uv run python report/scripts/process.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "report" / "data" / "raw"
OUT = REPO / "report" / "data" / "processed"

WINDOW = 20  # Rolling window, in episodes
BLOCK = 25  # Episodes per block
HEAD = 50  # Episodes compared at the start / end of the main session


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def label_phase(ep: pd.DataFrame) -> pd.Series:
    """Thirds of the main (uninterrupted) actor session, then everything after the actor restart."""
    main = ep["session"] == 0
    n = main.sum()
    edges = [0, n // 3, 2 * n // 3, n]
    names = np.array(["early", "middle", "late"])
    phase = pd.Series("after_restart", index=ep.index)
    phase[main] = names[np.searchsorted(edges, ep.loc[main, "episode"], side="right") - 1]
    return phase


def rolling(ep: pd.DataFrame) -> pd.DataFrame:
    r = ep[["episode", "session", "wall_clock_min", "cumulative_env_steps", "learner_opt_step_at_end"]].copy()
    roll = lambda s: s.rolling(WINDOW, min_periods=WINDOW // 2).mean()  # noqa: E731
    r["intervention_rate"] = ep["intervention_rate"]
    r["intervention_rate_roll"] = roll(ep["intervention_rate"])
    r["success_roll"] = roll(ep["success"].astype(float))
    r["autonomous_success_roll"] = roll(ep["autonomous_success"].astype(float))
    r["any_intervention_roll"] = roll(ep["any_intervention"].astype(float))
    r["robot_fault_roll"] = roll(ep["robot_fault"].astype(float))
    r["cumulative_intervention_steps"] = ep["intervention_steps"].cumsum()
    r["cumulative_autonomous_steps"] = (ep["length"] - ep["intervention_steps"]).cumsum()
    r["cumulative_successes"] = ep["success"].cumsum()
    r["cumulative_autonomous_successes"] = ep["autonomous_success"].cumsum()

    # Same rolling window, but over successful episodes only.
    s = ep[ep["success"]]
    srol = pd.DataFrame(
        {
            "episode": s["episode"],
            "success_intervention_rate_roll": roll(s["intervention_rate"]),
            "success_length_s_roll": roll(s["length"] / 10.0),
        }
    )
    return r.merge(srol, on="episode", how="left")


def blocks(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, e in ep.groupby(ep["episode"] // BLOCK):
        s = e[e["success"]]
        lo, hi = wilson(int(e["success"].sum()), len(e))
        rows.append(
            {
                "block": b,
                "first_episode": int(e["episode"].min()),
                "last_episode": int(e["episode"].max()),
                "episodes": len(e),
                "includes_restart": bool(e["session"].nunique() > 1 or e["session"].iloc[0] > 0),
                "learner_opt_step_end": int(e["learner_opt_step_at_end"].max()),
                "success_rate": e["success"].mean(),
                "success_rate_ci_low": lo,
                "success_rate_ci_high": hi,
                "intervention_rate_mean": e["intervention_rate"].mean(),
                "intervention_rate_pooled": e["intervention_steps"].sum() / e["length"].sum(),
                "intervention_rate_success_mean": s["intervention_rate"].mean(),
                "episodes_with_intervention": e["any_intervention"].mean(),
                "autonomous_success_rate": e["autonomous_success"].mean(),
                "assisted_success_rate": (e["success"] & e["any_intervention"]).mean(),
                "failure_rate": ((~e["success"]) & ~e["robot_fault"]).mean(),
                "robot_fault_rate": e["robot_fault"].mean(),
                "success_length_s_median": s["length"].median() / 10.0,
                "classifier_false_positive_rate": (e["classifier_vs_operator"] == "false_positive").sum()
                / max(int(e["classifier_success"].sum()), 1),
            }
        )
    return pd.DataFrame(rows)


def phase_table(ep: pd.DataFrame) -> pd.DataFrame:
    order = ["early", "middle", "late", "after_restart"]
    rows = []
    for ph in order:
        e = ep[ep["phase"] == ph]
        s = e[e["success"]]
        rows.append(
            {
                "phase": ph,
                "episodes": f"{e['episode'].min()}-{e['episode'].max()}",
                "n": len(e),
                "success_rate": e["success"].mean(),
                "intervention_rate_mean": e["intervention_rate"].mean(),
                "intervention_rate_success_mean": s["intervention_rate"].mean(),
                "episodes_with_intervention": e["any_intervention"].mean(),
                "autonomous_success_rate": e["autonomous_success"].mean(),
                "robot_fault_rate": e["robot_fault"].mean(),
                "success_length_s_median": s["length"].median() / 10.0,
                "learner_opt_steps": f"{e['learner_opt_step_at_end'].min()}-{e['learner_opt_step_at_end'].max()}",
            }
        )
    return pd.DataFrame(rows)


def intervention_by_episode_progress(ep: pd.DataFrame, frames: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Share of steps under human control at each tenth of the episode, per training phase."""
    f = frames.merge(ep[["episode", "phase", "length", "success"]], on="episode")
    f["progress_bin"] = np.minimum((f["frame_index"] / f["length"] * bins).astype(int), bins - 1)
    out = (
        f.groupby(["phase", "progress_bin"])["is_intervention"]
        .mean()
        .rename("intervention_share")
        .reset_index()
    )
    out["progress_mid"] = (out["progress_bin"] + 0.5) / bins
    return out


def intervention_segments(frames: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of human control in each episode."""
    rows = []
    for ep, f in frames.groupby("episode"):
        iv = f["is_intervention"].to_numpy()
        edges = np.diff(np.concatenate([[0], iv.astype(int), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for s, e in zip(starts, ends, strict=True):
            rows.append(
                {"episode": ep, "start_frame": int(s), "length": int(e - s), "episode_length": len(f)}
            )
    return pd.DataFrame(rows)


def learner_curves(learner: pd.DataFrame) -> pd.DataFrame:
    cols = ["loss_critic", "loss_actor", "loss_discrete_critic", "temperature", "replay_buffer_size"]
    lc = learner.sort_values("optimization_step").reset_index(drop=True)
    sm = lc[cols].ewm(span=100).mean().add_suffix("_ema")
    out = pd.concat([lc[["optimization_step", "time", *cols]], sm], axis=1)
    return out.iloc[::10].reset_index(drop=True)


def classifier_demo(demo: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    y = demo["label"].to_numpy() > 0
    p = demo["prob_success"].to_numpy()
    sweep = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        pred = p > t
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        sweep.append(
            {
                "threshold": t,
                "accuracy": float((pred == y).mean()),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "false_positive_frames": fp,
            }
        )
    ranks = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    auc = float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    pred = p > 0.5
    first = demo[demo["label"] > 0].groupby("episode")["frame_index"].min()
    demo = demo.assign(frames_from_first_success=demo["frame_index"] - demo["episode"].map(first))
    stats = {
        "frames": int(len(y)),
        "positive_frames": int(n1),
        "accuracy@0.5": float((pred == y).mean()),
        "precision@0.5": float((pred & y).sum() / max(pred.sum(), 1)),
        "recall@0.5": float((pred & y).sum() / n1),
        "roc_auc": auc,
        "confusion@0.5": {
            "tp": int((pred & y).sum()),
            "fp": int((pred & ~y).sum()),
            "fn": int((~pred & y).sum()),
            "tn": int((~pred & ~y).sum()),
        },
        "false_negatives_median_frames_after_first_success_label": float(
            demo.loc[y & ~pred, "frames_from_first_success"].median()
        ),
        "note": "In-sample: the classifier was trained on a 30-episode superset of these demos.",
    }
    return pd.DataFrame(sweep), stats, demo


def classifier_online(ep: pd.DataFrame, online: pd.DataFrame, tail: int = 40) -> tuple[pd.DataFrame, dict]:
    o = online.merge(ep[["episode", "length", "classifier_vs_operator", "robot_fault"]], on="episode")
    o["steps_before_end"] = o["length"] - 1 - o["frame_index"]
    o = o[o["steps_before_end"] < tail]
    o["outcome"] = np.where(o["robot_fault"], "robot_fault", o["classifier_vs_operator"])
    q = (
        o.groupby(["outcome", "steps_before_end"])["prob_success"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "p25", 0.5: "median", 0.75: "p75"})
        .reset_index()
    )
    called = ep["classifier_success"]
    in_loop = {
        "classifier_success_calls": int(called.sum()),
        "overridden_by_operator_as_failure": int((ep["classifier_vs_operator"] == "false_positive").sum()),
        "precision_vs_operator": float(
            (ep["classifier_vs_operator"] == "agree_success").sum() / called.sum()
        ),
        "operator_marked_success_classifier_missed": int(
            (ep["classifier_vs_operator"] == "missed_success").sum()
        ),
        "relabels_total": int(ep["relabel"].notna().sum()),
        "relabels_that_changed_outcome": int((ep["reward_at_end"] > 0).ne(ep["success"]).sum()),
        "note": "Operator relabels are the only ground truth online; unrelabeled episodes count as agreements.",
    }
    return q, in_loop


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = pd.read_csv(RAW / "online_episodes.csv")
    frames = pd.read_csv(RAW / "online_frames.csv")
    learner = pd.read_csv(RAW / "learner_metrics.csv")
    demo_eps = pd.read_csv(RAW / "demo_episodes.csv")
    demo_cls = pd.read_csv(RAW / "classifier_demo_frames.csv")
    online_cls = pd.read_csv(RAW / "classifier_online_frames.csv")

    ep["phase"] = label_phase(ep)
    ep.to_csv(OUT / "online_episodes_labeled.csv", index=False)

    roll = rolling(ep)
    roll.to_csv(OUT / "online_rolling.csv", index=False)
    blk = blocks(ep)
    blk.to_csv(OUT / "online_blocks_25.csv", index=False)
    ph = phase_table(ep)
    ph.to_csv(OUT / "online_phases.csv", index=False)
    intervention_by_episode_progress(ep, frames).to_csv(
        OUT / "intervention_by_episode_progress.csv", index=False
    )
    seg = intervention_segments(frames)
    seg.to_csv(OUT / "intervention_segments.csv", index=False)
    learner_curves(learner).to_csv(OUT / "learner_curves.csv", index=False)

    sweep, demo_stats, demo_cls = classifier_demo(demo_cls)
    sweep.to_csv(OUT / "classifier_demo_threshold_sweep.csv", index=False)
    demo_cls.to_csv(OUT / "classifier_demo_frames_aligned.csv", index=False)
    tail, in_loop = classifier_online(ep, online_cls)
    tail.to_csv(OUT / "classifier_online_tail.csv", index=False)

    ds = demo_eps[demo_eps["dataset"] == "xarm7_hilserl_demos_6dof_ft"]
    ds.to_csv(OUT / "demo_episodes_6dof_ft.csv", index=False)

    main_s = ep[ep["session"] == 0]
    first, last = main_s.head(HEAD), main_s.tail(HEAD)

    def snap(e: pd.DataFrame) -> dict:
        s = e[e["success"]]
        return {
            "episodes": f"{e['episode'].min()}-{e['episode'].max()}",
            "success_rate": round(e["success"].mean(), 3),
            "intervention_rate_mean": round(e["intervention_rate"].mean(), 3),
            "intervention_rate_success_mean": round(s["intervention_rate"].mean(), 3),
            "episodes_with_intervention": round(e["any_intervention"].mean(), 3),
            "autonomous_success_rate": round(e["autonomous_success"].mean(), 3),
            "success_length_s_median": round(s["length"].median() / 10.0, 1),
        }

    session_cfg = json.loads((RAW / "session_config.json").read_text())
    summary = {
        "session": {
            "date": "2026-11-09",
            "actor_wall_clock_min": round(ep["wall_clock_min"].max(), 1),
            "main_session_wall_clock_min": round(main_s["wall_clock_min"].max(), 1),
            "episodes": len(ep),
            "episodes_main_session": len(main_s),
            "episodes_after_restart": int((ep["session"] > 0).sum()),
            "env_steps": int(ep["length"].sum()),
            "human_intervention_steps": int(ep["intervention_steps"].sum()),
            "learner_optimization_steps": session_cfg["final_optimization_step"],
            "successes": int(ep["success"].sum()),
            "autonomous_successes": int(ep["autonomous_success"].sum()),
            "first_autonomous_success_episode": int(ep.loc[ep["autonomous_success"], "episode"].min()),
            "robot_fault_episodes": int(ep["robot_fault"].sum()),
            "fault_reasons": ep["fault_reason"].value_counts().to_dict(),
        },
        "first_vs_last_50_main_session": {"first": snap(first), "last": snap(last)},
        "after_restart": snap(ep[ep["session"] > 0]),
        "intervention_segments": {
            "count": len(seg),
            "median_length_steps": float(seg["length"].median()),
            "per_episode_with_intervention": round(len(seg) / ep["any_intervention"].sum(), 2),
        },
        "offline_demos": {
            "dataset": "local/xarm7_hilserl_demos_6dof_ft",
            "episodes": len(ds),
            "frames": int(ds["length"].sum()),
            "duration_s_median": float(ds["duration_s"].median()),
            "success_frames": int(ds["success_frames"].sum()),
        },
        "reward_classifier": {
            "config": json.loads((RAW / "classifier_config.json").read_text()),
            "offline_demo_frames": demo_stats,
            "in_the_loop": in_loop,
        },
        "config": session_cfg,
    }
    (REPO / "report" / "data" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ["first_vs_last_50_main_session", "after_restart"]}, indent=1))
    print(ph.round(3).to_string())


if __name__ == "__main__":
    main()
