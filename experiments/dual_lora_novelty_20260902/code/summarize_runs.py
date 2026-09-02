from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _finite_mean(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    agent_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "agent_log.jsonl").is_file()
    ]
    if len(agent_dirs) != 1:
        raise ValueError(f"Expected one agent log under {run_dir}, found {agent_dirs}")
    agent_dir = agent_dirs[0]
    steps = _read_jsonl(agent_dir / "agent_log.jsonl")
    novelty = _read_jsonl(agent_dir / "novelty_intermediates.jsonl")
    actions = [str(record.get("action", "")) for record in steps]
    local_repeat_count = sum(
        action in actions[max(0, index - 25) : index]
        for index, action in enumerate(actions)
    )
    split = min(int(config.get("novelty_decay_end", len(steps))), len(steps))
    early_actions = actions[:split]
    late_steps = steps[split:]
    score_totals: Dict[str, float] = {}
    for record in steps:
        for key, value in (record.get("reward", {}) or {}).items():
            score_totals[key] = score_totals.get(key, 0.0) + float(value)
    productive_keys = {
        "unique_kill",
        "kill",
        "craft",
        "exploration",
        "trade",
        "quest",
        "side_quest",
    }

    f2p_updates = []
    for record in steps:
        trace = record.get("f2p_trace", {}) or {}
        update = trace.get("training_update") or trace.get(
            "episode_end_training_update"
        )
        if update and update.get("updated"):
            f2p_updates.append(update)
    novelty_updates = [record for record in novelty if record.get("updated")]
    norm_errors = []
    for record in novelty_updates:
        target = record.get("target_effective_norm")
        actual = record.get("novelty_effective_norm")
        if target and actual is not None:
            norm_errors.append(abs(float(actual) - float(target)) / float(target))

    return {
        "run_dir": str(run_dir),
        "novelty_mode": config["novelty_mode"],
        "negative_selection": config.get("negative_selection"),
        "novelty_strength_mode": config.get("novelty_strength_mode", "absolute"),
        "novelty_lr": config["novelty_lr"],
        "lambda0": config["novelty_lambda0"],
        "seed": config["novelty_seed"],
        "steps": len(steps),
        "f2p_updates": len(f2p_updates),
        "novelty_updates": len(novelty_updates),
        "unique_actions": len(set(actions)),
        "exact_repeat_rate": (
            1.0 - len(set(actions)) / len(actions) if actions else None
        ),
        "local_25_repeat_rate": (
            local_repeat_count / len(actions) if actions else None
        ),
        "early_steps": len(early_actions),
        "early_unique_actions": len(set(early_actions)),
        "early_exact_repeat_rate": (
            1.0 - len(set(early_actions)) / len(early_actions)
            if early_actions
            else None
        ),
        "late_steps": len(late_steps),
        "late_reward_total": sum(
            float(value)
            for record in late_steps
            for value in (record.get("reward", {}) or {}).values()
        ),
        "late_productive_reward": sum(
            float(value)
            for record in late_steps
            for key, value in (record.get("reward", {}) or {}).items()
            if key in productive_keys
        ),
        "invalid_rate": (
            sum(bool(record.get("invalid_action")) for record in steps) / len(steps)
            if steps
            else None
        ),
        "reward_total": sum(score_totals.values()),
        "productive_reward_total": sum(
            value for key, value in score_totals.items() if key in productive_keys
        ),
        "death_total": score_totals.get("death", 0.0),
        "scores": score_totals,
        "mean_f2p_loss": _finite_mean(
            update.get("loss") for update in f2p_updates
        ),
        "mean_novelty_loss": _finite_mean(
            update.get("loss") for update in novelty_updates
        ),
        "mean_negative_logprob_change": _finite_mean(
            update.get("negative_mean_logprob_change")
            for update in novelty_updates
        ),
        "mean_relative_applied_strength": _finite_mean(
            update.get("relative_applied_strength")
            for update in novelty_updates
        ),
        "mean_counterfactual_applied_minus_task": _finite_mean(
            update.get("counterfactual_applied_minus_task_mean_after")
            for update in novelty_updates
        ),
        "mean_counterfactual_applied_update_change": _finite_mean(
            update.get("counterfactual_applied_update_change")
            for update in novelty_updates
        ),
        "mean_novelty_update_seconds": _finite_mean(
            (update.get("timing_seconds") or {}).get("total_update")
            for update in novelty_updates
        ),
        "max_relative_fixed_norm_error": max(norm_errors) if norm_errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    config_paths = sorted(args.root.glob("*/experiment_config.json"))
    summaries = [summarize_run(path.parent) for path in config_paths]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    flat_rows = [{**row, "scores": json.dumps(row["scores"])} for row in summaries]
    if flat_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
