"""Render report figures from report/data/processed into report/figures (PNG).

Run after process.py: uv run --with matplotlib python report/scripts/plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "report" / "data" / "processed"
OUT = REPO / "report" / "figures"

# Reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
GRAY_FILL = "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Noto Sans", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.unicode_minus": False,
    }
)


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def mark_restart(ax, ep: pd.DataFrame, y: float = 0.97) -> None:
    x = ep.loc[ep["session"] > 0, "episode"].min() - 0.5
    ax.axvline(x, color=MUTED, lw=1)
    ax.text(x + 2, y, "actor restart", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="top")


def pct(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_intervention_rate(ep, roll) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    succ = ep["success"]
    ax.scatter(
        ep.loc[~succ, "episode"],
        ep.loc[~succ, "intervention_rate"],
        s=10,
        color=GRAY_FILL,
        lw=0,
        label="Episode (failed)",
    )
    ax.scatter(
        ep.loc[succ, "episode"],
        ep.loc[succ, "intervention_rate"],
        s=10,
        color="#9ec5f4",
        lw=0,
        label="Episode (succeeded)",
    )
    ax.plot(roll["episode"], roll["intervention_rate_roll"], color=BLUE, label="All episodes, 20-ep mean")
    s = roll.dropna(subset=["success_intervention_rate_roll"])
    ax.plot(
        s["episode"],
        s["success_intervention_rate_roll"],
        color=ORANGE,
        label="Successful episodes, 20-ep mean",
    )
    mark_restart(ax, ep)
    ax.set_ylim(-0.02, 1.0)
    pct(ax)
    ax.set_xlabel("Online episode")
    ax.set_ylabel("Steps under human control")
    ax.set_title("Human intervention rate falls as the policy learns")
    subtitle(
        ax, "Share of each episode's steps driven by the SpaceMouse operator · xArm7 HIL-SERL, 2026-11-09"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    save(fig, "01_intervention_rate")


def fig_outcome_mix(blk) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cats = [
        ("autonomous_success_rate", "Success, no intervention", BLUE),
        ("assisted_success_rate", "Success with intervention", "#9ec5f4"),
        ("failure_rate", "Failure", GRAY_FILL),
        ("robot_fault_rate", "Robot fault", ORANGE),
    ]
    x = np.arange(len(blk))
    bottom = np.zeros(len(blk))
    for col, label, color in cats:
        v = blk[col].to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.62, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        bottom += v
    ax.set_xticks(
        x,
        [f"{a}–{b}" for a, b in zip(blk["first_episode"], blk["last_episode"], strict=True)],
        rotation=0,
        fontsize=8,
    )
    for i, v in enumerate(blk["autonomous_success_rate"]):
        if v >= 0.08:
            ax.text(
                i,
                v / 2,
                f"{v:.0%}",
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=8,
                fontweight="semibold",
            )
    restart = np.flatnonzero(blk["includes_restart"].to_numpy())
    if len(restart):
        ax.text(
            restart.mean(),
            1.01,
            "after actor restart (ep 252+)",
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=8,
        )
    ax.set_ylim(0, 1.05)
    pct(ax)
    ax.grid(False)
    ax.set_xlabel("Online episodes (blocks of 25)")
    ax.set_ylabel("Share of episodes")
    ax.set_title("Fully autonomous successes emerge after ~75 episodes")
    subtitle(
        ax,
        "Outcome of every online episode, grouped in blocks of 25 · labels show the no-intervention success share",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    save(fig, "02_outcome_mix")


def fig_success_rates(ep, roll) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(roll["episode"], roll["success_roll"], color=BLUE, label="Success (any)")
    ax.plot(
        roll["episode"], roll["autonomous_success_roll"], color=AQUA, label="Success with zero intervention"
    )
    ax.plot(
        roll["episode"], roll["any_intervention_roll"], color=ORANGE, label="Episodes with any intervention"
    )
    mark_restart(ax, ep)
    ax.set_ylim(0, 1.0)
    pct(ax)
    ax.set_xlabel("Online episode")
    ax.set_ylabel("Share of episodes (20-ep rolling)")
    ax.set_title("Success rises while the operator steps in less often")
    subtitle(ax, "Rolling 20-episode rates · success is the operator-confirmed outcome after relabels")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, "03_success_and_autonomy")


def fig_cumulative(ep, roll) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = roll["cumulative_env_steps"]
    ax.plot(x, roll["cumulative_autonomous_steps"], color=BLUE, label="Policy-driven steps")
    ax.plot(x, roll["cumulative_intervention_steps"], color=ORANGE, label="Human-driven steps")
    ax.fill_between(x, roll["cumulative_intervention_steps"], color=ORANGE, alpha=0.1, lw=0)
    for col in ["cumulative_autonomous_steps", "cumulative_intervention_steps"]:
        ax.annotate(
            f"{roll[col].iloc[-1]:,}",
            (x.iloc[-1], roll[col].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK2,
            fontsize=9,
        )
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax.set_xlabel("Cumulative environment steps (10 Hz)")
    ax.set_ylabel("Cumulative steps")
    ax.set_title("Human effort flattens out")
    subtitle(
        ax, "Cumulative steps by who controlled the arm · a flatter orange slope = less human work per step"
    )
    ax.legend(loc="upper left")
    save(fig, "04_cumulative_effort")


def fig_success_length(ep, roll) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    s = ep[ep["success"]]
    ax.scatter(s["episode"], s["length"] / 10.0, s=10, color="#9ec5f4", lw=0, label="Successful episode")
    r = roll.dropna(subset=["success_length_s_roll"])
    ax.plot(r["episode"], r["success_length_s_roll"], color=BLUE, label="20-success rolling mean")
    mark_restart(ax, ep)
    ax.set_ylim(0, None)
    ax.set_xlabel("Online episode")
    ax.set_ylabel("Time to success (s)")
    ax.set_title("Successful episodes get faster")
    subtitle(
        ax, "Episode duration of successful episodes only (episode ends when the reward classifier fires)"
    )
    ax.legend(loc="upper right")
    save(fig, "05_time_to_success")


def fig_progress(prog) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    phases = [
        ("early", "Early (ep 0–83)", ORANGE),
        ("middle", "Middle (ep 84–167)", BLUE),
        ("late", "Late (ep 168–251)", AQUA),
    ]
    for ph, label, color in phases:
        d = prog[prog["phase"] == ph]
        ax.plot(
            d["progress_mid"],
            d["intervention_share"],
            color=color,
            marker="o",
            ms=6,
            mec=SURFACE,
            mew=2,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    pct(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Progress through the episode")
    ax.set_ylabel("Steps under human control")
    ax.set_title("Where in the episode the operator steps in")
    subtitle(
        ax, "Share of steps intervened at each tenth of the episode, by training phase (main actor session)"
    )
    ax.legend(loc="upper left")
    save(fig, "06_intervention_by_episode_progress")


def fig_learner(lc, ep) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    panels = [
        ("loss_critic", "Critic loss", True),
        ("loss_actor", "Actor loss", False),
        ("loss_discrete_critic", "Gripper (discrete) critic loss", True),
        ("temperature", "SAC temperature α", True),
    ]
    gap = (
        ep.loc[ep["session"] == 0, "learner_opt_step_at_end"].max(),
        ep.loc[ep["session"] > 0, "learner_opt_step_at_end"].min(),
    )
    for ax, (col, title, log) in zip(axes.flat, panels, strict=True):
        ax.plot(lc["optimization_step"], lc[col], color=GRAY_FILL, lw=1)
        ax.plot(lc["optimization_step"], lc[f"{col}_ema"], color=BLUE)
        ax.axvspan(*gap, color=GRID, alpha=0.5, lw=0)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, pad=6)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    for ax in axes[1]:
        ax.set_xlabel("Learner optimization step")
    axes[0, 0].text(
        np.mean(gap),
        0.97,
        "no actor",
        transform=axes[0, 0].get_xaxis_transform(),
        ha="center",
        va="top",
        color=MUTED,
        fontsize=8,
    )
    fig.suptitle("Learner training curves", x=0.06, y=0.995, ha="left", fontsize=12, fontweight="semibold")
    fig.text(
        0.06,
        0.935,
        "Gray: raw (every 10th logged step) · blue: EMA (span 100) · shaded: learner ran while the actor was down",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "07_learner_curves")


def fig_classifier(demo, tail, sweep) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    d = demo.dropna(subset=["frames_from_first_success"])
    d = d[d["frames_from_first_success"].between(-30, 15)]
    for _, e in d.groupby("episode"):
        ax.plot(e["frames_from_first_success"], e["prob_success"], color="#9ec5f4", lw=1)
    med = d.groupby("frames_from_first_success")["prob_success"].median()
    ax.plot(med.index, med.values, color=BLUE, label="Median over demos")
    ax.axvline(-0.5, color=MUTED, lw=1)
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.text(0, 0.03, " success label starts", color=MUTED, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Frames relative to first success label (10 Hz)")
    ax.set_ylabel("Predicted success probability")
    ax.set_title("On demos: sharp switch at success", fontsize=11)
    acc = sweep.loc[np.isclose(sweep["threshold"], 0.5)].iloc[0]
    subtitle(
        ax,
        f"14 demo episodes · acc {acc['accuracy']:.1%}, precision {acc['precision']:.0%}, recall {acc['recall']:.0%} @0.5 (in-sample)",
    )
    ax.legend(loc="upper left")

    ax = axes[1]
    groups = [
        ("agree_success", "Success (classifier and operator agree)", BLUE),
        ("false_positive", "Classifier success, operator overrode", ORANGE),
        ("agree_failure", "Failure", MUTED),
    ]
    for key, label, color in groups:
        t = tail[tail["outcome"] == key].sort_values("steps_before_end")
        x = -t["steps_before_end"]
        ax.plot(x, t["median"], color=color, label=label)
        ax.fill_between(x, t["p25"], t["p75"], color=color, alpha=0.1, lw=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-20, 0.5)
    ax.set_xlabel("Steps before episode end")
    ax.set_title("Online: confidence ramps up before the trigger", fontsize=11)
    subtitle(ax, "Median and IQR over online episodes · trigger frame itself is not stored in the buffer")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "08_reward_classifier")


def fig_restart_diagnostics(cond, sig) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    cond = cond.sort_values("episode")
    restart_x = cond.loc[cond["session"] > 0, "episode"].min() - 0.5
    series = [
        (axes[0, 0], "front_wall_luma", "Lighting: front-camera wall brightness", "Mean luma (0–1)"),
        (axes[0, 1], "wrist_lfinger_x", "Wrist camera vs gripper: left-finger position", "Pixel x (of 128)"),
        (
            axes[1, 0],
            "front_gripper_x",
            "Arm home pose: gripper position in front camera",
            "Pixel x (of 128)",
        ),
    ]
    for ax, col, title, ylabel in series:
        ax.scatter(cond["episode"], cond[col], s=8, color="#9ec5f4", lw=0)
        ax.plot(cond["episode"], cond[col].rolling(15, min_periods=5, center=True).median(), color=BLUE)
        ax.axvline(restart_x, color=MUTED, lw=1)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Online episode")
        lo, hi = cond[col].quantile([0.02, 0.98])
        pad = (hi - lo) * 0.3
        ax.set_ylim(lo - pad, hi + pad)
    axes[0, 0].text(
        restart_x + 3,
        0.04,
        "actor restart",
        transform=axes[0, 0].get_xaxis_transform(),
        color=MUTED,
        fontsize=8,
    )

    ax = axes[1, 1]
    labels = {
        "success, all episodes": "Success (all)",
        "success, episodes with no intervention": "Success, no intervention",
        "success, episodes with intervention": "Success, with intervention",
        "classifier success overridden by operator": "Classifier success overridden",
        "robot fault": "Robot fault",
    }
    y = np.arange(len(sig))[::-1]
    h = 0.34
    ax.barh(y + h / 2, sig["late_rate"], height=h, color=GRAY_FILL, label="Late main session (ep 168–251)")
    ax.barh(y - h / 2, sig["after_restart_rate"], height=h, color=BLUE, label="After restart (ep 252–295)")
    for yi, (_, r) in zip(y, sig.iterrows(), strict=True):
        ax.text(
            max(r["late_rate"], r["after_restart_rate"]) + 0.02,
            yi,
            f"p={r['fisher_p']:.2g}",
            va="center",
            color=INK2,
            fontsize=8,
        )
    ax.set_yticks(y, [labels[m] for m in sig["measure"]], fontsize=8)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Outcomes: late session vs after restart", fontsize=10, pad=6)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "What changed at the actor restart?", x=0.06, y=0.995, ha="left", fontsize=12, fontweight="semibold"
    )
    fig.text(
        0.06,
        0.94,
        "Image measures from each episode's first frame (fixed home pose) · line: 15-episode rolling median · p: Fisher exact test",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "09_restart_diagnostics")


def fig_home_views(views) -> None:
    groups = [
        ("demos", "Offline demos"),
        ("early", "Early (14:20–14:52)"),
        ("late", "Late (15:25–15:57)"),
        ("after_restart", "After restart (16:33–16:58)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for i, cam in enumerate(["wrist", "front"]):
        for j, (g, label) in enumerate(groups):
            axes[i, j].imshow(np.clip(views[f"{cam}_{g}"], 0, 1))
            axes[i, j].set_title(label if i == 0 else "", fontsize=9, pad=4, loc="center")
        diff = np.abs(views[f"{cam}_after_restart"] - views[f"{cam}_late"]).mean(-1)
        axes[i, 4].imshow(diff, cmap="Greys_r", vmin=0, vmax=0.2)
        axes[i, 4].set_title("|after restart - late|" if i == 0 else "", fontsize=9, pad=4, loc="center")
        axes[i, 0].set_ylabel(f"{cam} camera", fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(
        "Average home view per phase", x=0.02, y=0.995, ha="left", fontsize=12, fontweight="semibold"
    )
    fig.text(
        0.02,
        0.925,
        "Mean of each episode's first frame · right: where the after-restart view differs from the late session (bright = larger change)",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "10_home_views")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = pd.read_csv(PROC / "online_episodes_labeled.csv")
    roll = pd.read_csv(PROC / "online_rolling.csv")
    blk = pd.read_csv(PROC / "online_blocks_25.csv")
    fig_intervention_rate(ep, roll)
    fig_outcome_mix(blk)
    fig_success_rates(ep, roll)
    fig_cumulative(ep, roll)
    fig_success_length(ep, roll)
    fig_progress(pd.read_csv(PROC / "intervention_by_episode_progress.csv"))
    fig_learner(pd.read_csv(PROC / "learner_curves.csv"), ep)
    fig_classifier(
        pd.read_csv(PROC / "classifier_demo_frames_aligned.csv"),
        pd.read_csv(PROC / "classifier_online_tail.csv"),
        pd.read_csv(PROC / "classifier_demo_threshold_sweep.csv"),
    )
    fig_restart_diagnostics(
        pd.read_csv(PROC / "restart_episode_conditions.csv"), pd.read_csv(PROC / "restart_significance.csv")
    )
    fig_home_views(np.load(PROC / "home_views_mean.npz"))
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
