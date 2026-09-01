from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/f2p-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


RUNS = [
    ("original", "Original signed F2P", "#4C78A8"),
    ("no_w", "No $w_t$ (NLL)", "#F58518"),
    (
        "normalized_top1_a0.25_b0.25",
        r"Normalized $\alpha=.25,\ \beta=.25$",
        "#54A24B",
    ),
    (
        "normalized_top2_a0.5_b0.25",
        r"Normalized $\alpha=.5,\ \beta=.25$",
        "#E45756",
    ),
    (
        "normalized_top3_a1.0_b0.25",
        r"Normalized $\alpha=1,\ \beta=.25$",
        "#B279A2",
    ),
]

NORMALIZED_RUNS = {name for name, _, _ in RUNS if name.startswith("normalized_")}
OFFICIAL_SUPPLEMENTARY_FIELDS = ("side_quest", "exploration", "craft", "unique_kill")
ACTION_VERBS = sorted(
    (
        "pick up",
        "talk to",
        "take out",
        "lockpick",
        "pickpocket",
        "disassemble",
        "unequip",
        "inspect",
        "discard",
        "defend",
        "attack",
        "craft",
        "enter",
        "equip",
        "store",
        "drop",
        "throw",
        "write",
        "sell",
        "wait",
        "buy",
        "eat",
    ),
    key=len,
    reverse=True,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def rolling_quantiles(
    values: list[float] | np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    low = np.empty_like(array)
    middle = np.empty_like(array)
    high = np.empty_like(array)
    for index in range(len(array)):
        segment = array[max(0, index - window + 1) : index + 1]
        low[index], middle[index], high[index] = np.nanquantile(
            segment, (0.25, 0.5, 0.75)
        )
    return low, middle, high


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.empty_like(array)
    for index in range(len(array)):
        result[index] = np.nanmean(array[max(0, index - window + 1) : index + 1])
    return result


def action_verb(action: str) -> str:
    normalized = " ".join(str(action).strip().lower().split())
    for verb in ACTION_VERBS:
        if normalized == verb or normalized.startswith(verb + " "):
            return verb
    return normalized.split(" ", 1)[0] if normalized else "<empty>"


def rolling_entropy(actions: list[str], window: int) -> np.ndarray:
    verbs = [action_verb(action) for action in actions]
    entropy = np.empty(len(verbs), dtype=float)
    for index in range(len(verbs)):
        segment = verbs[max(0, index - window + 1) : index + 1]
        counts = np.asarray(list(Counter(segment).values()), dtype=float)
        probabilities = counts / counts.sum()
        entropy[index] = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def load_runs(experiment_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, label, color in RUNS:
        run_dir = experiment_root / "formal" / name
        agent_dir = run_dir / "qwen3_4b_f2p_ttt"
        traces = read_jsonl(agent_dir / "f2p_intermediates.jsonl")
        agent_rows = read_jsonl(agent_dir / "agent_log.jsonl")
        if len(traces) != len(agent_rows):
            raise ValueError(
                f"Trace/agent-log length mismatch in {name}: "
                f"{len(traces)} != {len(agent_rows)}"
            )
        updates: list[tuple[int, dict[str, Any]]] = []
        for step, trace in enumerate(traces, start=1):
            for key in ("training_update", "episode_end_training_update"):
                update = trace.get(key)
                if isinstance(update, dict) and update.get("updated"):
                    updates.append((step, update))
        result[name] = {
            "label": label,
            "color": color,
            "traces": traces,
            "agent_rows": agent_rows,
            "updates": updates,
        }
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    axis.tick_params(labelsize=9)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def plot_training_dashboard(
    runs: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.6))
    axes = axes.ravel()

    # A: normalized total objectives. Raw objective values are not comparable
    # to the original/no-w losses, so only the normalized sweep is overlaid.
    axis = axes[0]
    for name, run in runs.items():
        if name not in NORMALIZED_RUNS:
            continue
        x = np.asarray([step for step, _ in run["updates"]])
        y = np.asarray([update["loss"] for _, update in run["updates"]], dtype=float)
        axis.plot(x, y, color=run["color"], alpha=0.18, linewidth=1)
        axis.plot(x, rolling_mean(y, 5), color=run["color"], linewidth=2.2)
    axis.set_title("Normalized training objective", fontsize=11, fontweight="bold")
    axis.set_ylabel("Loss (5-update rolling mean)")
    axis.set_xlabel("Environment step")
    style_axis(axis)
    add_panel_label(axis, "A")

    # B: pre-clipping gradient norm. The original path did not log this field.
    axis = axes[1]
    for name, run in runs.items():
        points = [
            (step, update.get("grad_norm_before_clip"))
            for step, update in run["updates"]
            if update.get("grad_norm_before_clip") is not None
        ]
        if not points:
            continue
        x = np.asarray([point[0] for point in points])
        y = np.asarray([point[1] for point in points], dtype=float)
        axis.plot(x, y, color=run["color"], alpha=0.2, linewidth=0.9)
        axis.plot(x, rolling_mean(y, 5), color=run["color"], linewidth=2.1)
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
    axis.text(498, 1.12, "clip threshold", fontsize=8, ha="right", va="bottom")
    axis.set_yscale("log")
    axis.set_title("Gradient norm before clipping", fontsize=11, fontweight="bold")
    axis.set_ylabel(r"$\|g\|_2$ (log scale)")
    axis.set_xlabel("Environment step")
    style_axis(axis)
    add_panel_label(axis, "B")

    # C: length-normalized log probability of the action actually executed.
    axis = axes[2]
    for run in runs.values():
        x = np.arange(1, len(run["traces"]) + 1)
        y = [trace["action_prior_logprob"] for trace in run["traces"]]
        low, middle, high = rolling_quantiles(y, 20)
        axis.fill_between(x, low, high, color=run["color"], alpha=0.055, linewidth=0)
        axis.plot(x, middle, color=run["color"], linewidth=1.9)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title("Executed-action confidence", fontsize=11, fontweight="bold")
    axis.set_ylabel("Mean token log-probability\n(20-step rolling median/IQR)")
    axis.set_xlabel("Environment step")
    style_axis(axis)
    add_panel_label(axis, "C")

    # D: the diagnostic F2P weight. Only the original objective consumes w_t.
    axis = axes[3]
    for run in runs.values():
        x = np.arange(1, len(run["traces"]) + 1)
        y = [trace["w_t"] for trace in run["traces"]]
        axis.plot(x, rolling_mean(y, 25), color=run["color"], linewidth=1.9)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_ylim(-0.45, 0.75)
    axis.set_title(r"F2P feedback weight $w_t$", fontsize=11, fontweight="bold")
    axis.set_ylabel("25-step rolling mean")
    axis.set_xlabel("Environment step")
    axis.text(
        0.02,
        0.05,
        r"Diagnostic only for no-$w_t$/normalized runs",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    style_axis(axis)
    add_panel_label(axis, "D")

    # E: official supplementary reward only. A star marks main-quest progress.
    axis = axes[4]
    for run in runs.values():
        increments = np.asarray(
            [
                sum(float(row.get("reward", {}).get(key, 0)) for key in OFFICIAL_SUPPLEMENTARY_FIELDS)
                for row in run["agent_rows"]
            ],
            dtype=float,
        )
        cumulative = np.cumsum(increments)
        x = np.arange(1, len(cumulative) + 1)
        axis.step(x, cumulative, where="post", color=run["color"], linewidth=2)
        quest_steps = [
            step
            for step, row in enumerate(run["agent_rows"], start=1)
            if float(row.get("reward", {}).get("quest", 0)) > 0
        ]
        for step in quest_steps:
            axis.scatter(
                step,
                cumulative[step - 1],
                marker="*",
                s=85,
                color=run["color"],
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
    axis.set_title("Official supplementary progress", fontsize=11, fontweight="bold")
    axis.set_ylabel("Cumulative supplementary score")
    axis.set_xlabel("Environment step   (★ main-quest advance)")
    axis.set_ylim(bottom=-0.3)
    style_axis(axis)
    add_panel_label(axis, "E")

    # F: invalid actions and action-verb entropy share a panel because both
    # diagnose whether policy confidence comes at the cost of usable diversity.
    axis = axes[5]
    entropy_axis = axis.twinx()
    for run in runs.values():
        x = np.arange(1, len(run["agent_rows"]) + 1)
        invalid = [float(bool(row.get("invalid_action"))) for row in run["agent_rows"]]
        invalid_rate = 100 * rolling_mean(invalid, 50)
        entropy = rolling_entropy([row.get("action", "") for row in run["agent_rows"]], 50)
        axis.plot(x, invalid_rate, color=run["color"], linewidth=1.8)
        entropy_axis.plot(x, entropy, color=run["color"], linewidth=1.2, linestyle=":")
    axis.set_title("Policy validity and diversity", fontsize=11, fontweight="bold")
    axis.set_ylabel("Invalid actions (%) — solid")
    entropy_axis.set_ylabel("Action-verb entropy (bits) ··· dotted")
    axis.set_xlabel("Environment step (50-step rolling window)")
    axis.set_ylim(bottom=0)
    entropy_axis.set_ylim(bottom=0)
    style_axis(axis)
    entropy_axis.spines["top"].set_visible(False)
    entropy_axis.tick_params(labelsize=9)
    add_panel_label(axis, "F")

    handles = [
        Line2D([0], [0], color=color, linewidth=3, label=label)
        for _, label, color in RUNS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
    )
    figure.suptitle(
        "F2P Loss Ablation — Training Dynamics",
        fontsize=17,
        fontweight="bold",
        y=0.992,
    )
    figure.text(
        0.5,
        0.962,
        "Remnant · Qwen3-4B · seed 42 · 500 environment steps · single-run trajectories",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.012,
        "Lines and rolling intervals are descriptive trajectories, not confidence intervals. "
        "Raw objective magnitudes are not comparable across loss definitions.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.94, bottom=0.075, top=0.88, wspace=0.31, hspace=0.35)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"training_dynamics_dashboard.{extension}",
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_data_quality(
    runs: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": (0.9, 1.8)})

    # A: explicit output coverage.
    axis = axes[0]
    x = np.arange(len(RUNS))
    nonempty_counts = np.asarray(
        [sum(bool(trace.get("predicted_outcome")) for trace in runs[name]["traces"]) for name, _, _ in RUNS]
    )
    totals = np.asarray([len(runs[name]["traces"]) for name, _, _ in RUNS])
    nonempty = 100 * nonempty_counts / totals
    empty = 100 - nonempty
    colors = [color for _, _, color in RUNS]
    axis.bar(x, nonempty, color=colors, width=0.68, label="Non-empty prediction")
    axis.bar(
        x,
        empty,
        bottom=nonempty,
        color="#E6E6E6",
        edgecolor="#777777",
        linewidth=0.6,
        width=0.68,
        label="Empty prediction",
    )
    for index, (percent, count, total) in enumerate(zip(nonempty, nonempty_counts, totals)):
        axis.text(index, percent / 2, f"{percent:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        axis.text(index, 101.8, f"{total-count} empty", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, ["Original", "No $w_t$", r"$\alpha=.25$", r"$\alpha=.5$", r"$\alpha=1$"], rotation=20)
    axis.set_ylim(0, 109)
    axis.set_ylabel("Share of environment steps (%)")
    axis.set_title("Predicted-outcome coverage", fontsize=12, fontweight="bold")
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    style_axis(axis)
    add_panel_label(axis, "A")

    # B: feedback delta conditioned on whether the model supplied a prediction.
    axis = axes[1]
    positions: list[float] = []
    values: list[np.ndarray] = []
    box_colors: list[str] = []
    filled: list[bool] = []
    for group_index, (name, _, color) in enumerate(RUNS):
        traces = runs[name]["traces"]
        for offset, is_nonempty in ((-0.19, False), (0.19, True)):
            subset = np.asarray(
                [
                    trace["delta_real_minus_pred"]
                    for trace in traces
                    if bool(trace.get("predicted_outcome")) is is_nonempty
                ],
                dtype=float,
            )
            positions.append(group_index + offset)
            values.append(subset)
            box_colors.append(color)
            filled.append(is_nonempty)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "#222222", "linewidth": 1.3},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9},
    )
    for patch, color, is_filled in zip(boxes["boxes"], box_colors, filled):
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
        patch.set_facecolor(color if is_filled else "white")
        patch.set_alpha(0.55 if is_filled else 1.0)
        if not is_filled:
            patch.set_hatch("///")
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_xticks(np.arange(len(RUNS)), ["Original", "No $w_t$", r"$\alpha=.25$", r"$\alpha=.5$", r"$\alpha=1$"])
    axis.set_ylabel(r"$\Delta_t = \ell_{real} - \ell_{pred}$")
    axis.set_title("Feedback delta by prediction availability", fontsize=12, fontweight="bold")
    axis.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="#777777", alpha=0.55, label="Non-empty prediction"),
            Patch(facecolor="white", edgecolor="#777777", hatch="///", label="Empty prediction"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    axis.text(
        0.01,
        0.02,
        "Boxes: IQR; whiskers: 5th–95th percentile; outliers hidden",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    style_axis(axis)
    add_panel_label(axis, "B")

    empty_total = int(sum(totals - nonempty_counts))
    figure.suptitle(
        "F2P Feedback Data Quality",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.93,
        f"{empty_total:,} / {int(sum(totals)):,} transitions have an empty predicted outcome; "
        "their predicted context collapses to the prior context.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.015,
        "Structured stage_completed was empty in all 2,500 traces, despite a final quest reward of 1 in every run.",
        ha="center",
        fontsize=9,
        color="#9C2F2F",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.84, wspace=0.28)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"feedback_data_quality.{extension}",
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot F2P loss-ablation dynamics.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else experiment_root / "summary" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(experiment_root)
    plot_training_dashboard(runs, output_dir)
    plot_data_quality(runs, output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
