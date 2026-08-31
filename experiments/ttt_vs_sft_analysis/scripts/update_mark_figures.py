"""Regenerate task-separated overview and Mark comparison figures.

The current Mark Lookahead run is summarized from its completed 500-step
remote log.  The plots intentionally show final aggregates rather than
inventing step-level curves when only the aggregate was transferred.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

COLORS = {
    "Environment-prediction TTT": "#3b82f6",
    "SFT + Short-Term Memory": "#f59e0b",
    "Feedback-to-Policy TTT": "#10b981",
    "Lookahead Environment-TTT": "#8b5cf6",
}

MARK = {
    "Environment-prediction TTT": {"failure": 0.440, "repeat": 0.148, "exploration": 149, "invalid": 0.060, "prediction": 0.140, "quest": 1, "positive": 13, "net": 11},
    "SFT + Short-Term Memory": {"failure": 0.452, "repeat": 0.258, "exploration": 174, "invalid": 0.060, "prediction": 0.000, "quest": 5, "positive": 24, "net": 20},
    "Feedback-to-Policy TTT": {"failure": 0.418, "repeat": 0.242, "exploration": 129, "invalid": 0.030, "prediction": 0.126, "quest": 1, "positive": 18, "net": 14},
    "Lookahead Environment-TTT": {"failure": 0.372, "repeat": 0.132, "exploration": 126, "invalid": 0.030, "prediction": 0.750, "quest": 1, "positive": 17, "net": 12},
}


def bar(name, key, title, ylabel, *, percent=False, ylim=None):
    labels = list(MARK)
    values = [MARK[x][key] * (100 if percent else 1) for x in labels]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.bar(np.arange(len(labels)), values, color=[COLORS[x] for x in labels])
    ax.set_xticks(np.arange(len(labels)), [
        "Env-TTT", "SFT+Memory", "F2P-TTT", "Lookahead\nEnv-TTT"
    ])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Mark — 500 environment steps")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f}" if percent else f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180)
    plt.close(fig)


bar("mark_01_failure_pressure.png", "failure", "Mark failure pressure", "Failure rate (%)", percent=True, ylim=(0, 55))
bar("mark_02_action_repetition.png", "repeat", "Mark recent action repetition", "Recent-repeat rate (%)", percent=True, ylim=(0, 35))
bar("mark_03_action_exploration.png", "exploration", "Mark action-space exploration", "Distinct exact actions", ylim=(0, 200))
bar("mark_04_invalid_actions.png", "invalid", "Mark invalid-action rate", "Invalid-action rate (%)", percent=True, ylim=(0, 8))
bar("mark_05_prediction_coverage.png", "prediction", "Mark predicted-outcome coverage", "Coverage (%)", percent=True, ylim=(0, 85))


# A task-separated overview. Each game has its own panel and its own methods.
games = {
    "Metropolis": [
        ("Env-TTT", 18, 9), ("SFT+Memory", 9, 7), ("F2P-TTT", 18, 7), ("Lookahead\nEnv-TTT", 13, 7)
    ],
    "Remnant": [
        ("Env-TTT", 3, 1), ("SFT+Memory", 0, 1), ("F2P-TTT", 19, 1), ("Lookahead\nEnv-TTT", 18, 1)
    ],
    "Mark": [
        ("Env-TTT", 11, 1), ("SFT+Memory", 20, 5), ("F2P-TTT", 14, 1), ("Lookahead\nEnv-TTT", 12, 1)
    ],
}
method_colors = [COLORS["Environment-prediction TTT"], COLORS["SFT + Short-Term Memory"], COLORS["Feedback-to-Policy TTT"], COLORS["Lookahead Environment-TTT"]]
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
for ax, (game, rows) in zip(axes, games.items()):
    x = np.arange(len(rows))
    net = [r[1] for r in rows]
    quest = [r[2] for r in rows]
    bars = ax.bar(x, net, color=method_colors, alpha=0.9)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("Net reward proxy")
    ax.set_title(f"{game} — same-task comparison (500 steps)", loc="left")
    ax.grid(axis="y", alpha=0.25)
    for b, n, q in zip(bars, net, quest):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"net={n}, quest={q}", ha="center", va="bottom", fontsize=9)
axes[0].legend(["Environment-prediction TTT", "SFT + Short-Term Memory", "F2P-TTT", "Lookahead Environment-TTT"], frameon=False, ncol=2, loc="upper right")
fig.suptitle("AgentOdyssey results overview — each game is plotted separately", y=0.995)
fig.tight_layout()
fig.savefig(FIG / "00_results_overview.png", dpi=180)
plt.close(fig)

