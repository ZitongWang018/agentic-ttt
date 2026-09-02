from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List


AGENT_ID = "qwen3_4b_dual_lora_novelty"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def finite_mean(values: Iterable[Any]) -> float | None:
    selected = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return statistics.mean(selected) if selected else None


def summarize_run(run_dir: Path, *, status: str) -> Dict[str, Any]:
    agent_dir = run_dir / AGENT_ID
    steps = read_jsonl(agent_dir / "agent_log.jsonl")
    traces = read_jsonl(agent_dir / "action_memory_intermediates.jsonl")
    actions = [str(row.get("action", "")) for row in steps]
    rewards: Dict[str, float] = {}
    for row in steps:
        for key, value in (row.get("reward") or {}).items():
            rewards[key] = rewards.get(key, 0) + (value or 0)

    exact_repeat_count = sum(
        trace.get("chosen_exact_repeat") is True for trace in traces
    )
    semantic_repeat_count = sum(
        trace.get("chosen_semantic_repeat") is True for trace in traces
    )
    updates = [
        trace.get("update") or {}
        for trace in traces
        if (trace.get("update") or {}).get("updated")
    ]
    calibrations = [
        update.get("effect_calibration") or {}
        for update in updates
        if (update.get("effect_calibration") or {}).get("enabled")
    ]
    chosen_deltas = [
        (trace.get("generation") or {}).get("chosen_target_logprob_delta_mean")
        for trace in traces
    ]
    repeat_deltas = [
        (trace.get("generation") or {}).get("chosen_target_logprob_delta_mean")
        for trace in traces
        if trace.get("chosen_exact_repeat") is True
    ]
    novel_deltas = [
        (trace.get("generation") or {}).get("chosen_target_logprob_delta_mean")
        for trace in traces
        if trace.get("chosen_exact_repeat") is False
    ]

    return {
        "setting": run_dir.name.removesuffix("_seed42"),
        "status": status,
        "run_dir": str(run_dir),
        "steps": len(steps),
        "environment_step_first": steps[0].get("step") if steps else None,
        "environment_step_last": steps[-1].get("step") if steps else None,
        "unique_actions": len(set(actions)),
        "exact_repeat_count": exact_repeat_count,
        "exact_repeat_rate": exact_repeat_count / len(traces) if traces else None,
        "semantic_repeat_count": semantic_repeat_count,
        "semantic_repeat_rate": semantic_repeat_count / len(traces) if traces else None,
        "invalid_count": sum(bool(row.get("invalid_action")) for row in steps),
        "invalid_rate": (
            sum(bool(row.get("invalid_action")) for row in steps) / len(steps)
            if steps
            else None
        ),
        "rewards": rewards,
        "adapter_updates": len(updates),
        "history_target_logprob_change_mean": finite_mean(
            update.get("target_logprob_change") for update in updates
        ),
        "chosen_action_logprob_change_mean": finite_mean(chosen_deltas),
        "repeat_chosen_action_logprob_change_mean": finite_mean(repeat_deltas),
        "novel_chosen_action_logprob_change_mean": finite_mean(novel_deltas),
        "effect_target": (
            calibrations[0].get("target_logprob_drop") if calibrations else None
        ),
        "effect_target_updates": len(calibrations),
        "effect_target_updates_achieved": sum(
            calibration.get("achieved") is True for calibration in calibrations
        ),
        "calibrated_scale_mean": finite_mean(
            calibration.get("selected_scale") for calibration in calibrations
        ),
        "calibrated_scale_min": (
            min(calibration["selected_scale"] for calibration in calibrations)
            if calibrations
            else None
        ),
        "calibrated_scale_max": (
            max(calibration["selected_scale"] for calibration in calibrations)
            if calibrations
            else None
        ),
        "calibrated_drop_mean": finite_mean(
            calibration.get("achieved_logprob_drop") for calibration in calibrations
        ),
        "calibrated_kl_mean": finite_mean(
            (calibration.get("objective_at_selected_scale") or {}).get("reference_kl")
            for calibration in calibrations
        ),
        "average_decision_seconds": finite_mean(
            row.get("decision_time") for row in steps
        ),
        "average_input_tokens": finite_mean(
            row.get("num_input_tokens") for row in steps
        ),
        "average_output_tokens": finite_mean(
            row.get("num_output_tokens") for row in steps
        ),
        "actions": [
            {
                "step": row.get("step"),
                "action": row.get("action"),
                "invalid": bool(row.get("invalid_action")),
            }
            for row in steps
        ],
    }


def markdown_report(report: Dict[str, Any]) -> str:
    complete = report["completed_50step_runs"]
    stopped = report["stopped_effect_target_runs"]
    lines = [
        "# Step100 action-memory exploration",
        "",
        "Generated: " + report["generated_at"],
        "",
        "All runs branch from the same F2P environment/LoRA checkpoint at step 100.",
        "Repeat rates include the 25 actions preceding the branch point.",
        "",
        "## Completed 50-step comparison",
        "",
        "| Setting | Unique | Exact repeat | Semantic repeat | Invalid | Kill | Death |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complete:
        reward = row["rewards"]
        lines.append(
            f"| {row['setting']} | {row['unique_actions']} | "
            f"{100 * row['exact_repeat_rate']:.1f}% | "
            f"{100 * row['semantic_repeat_rate']:.1f}% | "
            f"{row['invalid_count']} | {reward.get('kill', 0)} | "
            f"{reward.get('death', 0)} |"
        )
    lines.extend(
        [
            "",
            "Prompt-history25 reduced exact repetition most (42%) and produced the most unique actions (30),",
            "but increased latency and deaths. Neither fixed-scale parameter adapter beat the F2P control.",
            "",
            "## Stopped effect-target diagnostic",
            "",
            "| Setting | Steps | Exact repeat | Invalid | Target/actual drop | Mean scale | Mean KL |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stopped:
        lines.append(
            f"| {row['setting']} | {row['steps']} | "
            f"{100 * row['exact_repeat_rate']:.1f}% | "
            f"{row['invalid_count']} ({100 * row['invalid_rate']:.1f}%) | "
            f"{row['effect_target']:.1f}/{row['calibrated_drop_mean']:.3f} | "
            f"{row['calibrated_scale_mean']:.3f} | {row['calibrated_kl_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Both constraints were met on stored historical action-slot states, but the probability mass",
            "moved toward spelling, punctuation, and language variants rather than new semantic actions.",
            "Examples include `defender`, `pickup(torch)`, `pick uptorch`,",
            "`attack goblins_warror_1`, and `攻击 goblinWarriot_1`.",
            "The runs were therefore stopped early and are evidence against strengthening this token-level",
            "action-slot objective without a semantic or planning-level mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    complete_root = args.experiment_root / "step100_action_memory"
    stopped_root = args.experiment_root / "step100_effect_target"
    complete = [
        summarize_run(path, status="completed_50_steps")
        for path in sorted(complete_root.glob("*_seed42"))
    ]
    stopped = [
        summarize_run(path, status="stopped_early_token_surface_escape")
        for path in sorted(stopped_root.glob("*_seed42"))
    ]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_checkpoint": str(
            args.experiment_root
            / "f2p_checkpoint_rerun/f2p_only_seed42/"
            "qwen3_4b_dual_lora_novelty/lora_checkpoints/step_0100"
        ),
        "completed_50step_runs": complete,
        "stopped_effect_target_runs": stopped,
        "conclusion": {
            "prompt_history": "reduced repetition but increased latency and deaths",
            "fixed_parameter_adapters": "did not beat the F2P control",
            "effect_target": (
                "met stored-history constraints but escaped through malformed or surface-varied actions"
            ),
        },
    }

    for path in (args.output_json, args.output_csv, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows = complete + stopped
    fields = [
        "setting",
        "status",
        "steps",
        "unique_actions",
        "exact_repeat_count",
        "exact_repeat_rate",
        "semantic_repeat_count",
        "semantic_repeat_rate",
        "invalid_count",
        "invalid_rate",
        "adapter_updates",
        "history_target_logprob_change_mean",
        "chosen_action_logprob_change_mean",
        "repeat_chosen_action_logprob_change_mean",
        "effect_target",
        "effect_target_updates",
        "effect_target_updates_achieved",
        "calibrated_scale_mean",
        "calibrated_drop_mean",
        "calibrated_kl_mean",
        "average_decision_seconds",
        "average_input_tokens",
        "average_output_tokens",
        "unique_kill",
        "kill",
        "craft",
        "death",
        "side_quest",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in flat_rows:
            output = {key: row.get(key) for key in fields}
            for key in ("unique_kill", "kill", "craft", "death", "side_quest"):
                output[key] = row["rewards"].get(key, 0)
            writer.writerow(output)
    args.output_md.write_text(markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
