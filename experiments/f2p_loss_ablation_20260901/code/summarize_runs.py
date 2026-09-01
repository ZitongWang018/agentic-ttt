from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    # Agent traces are true JSONL, while eval.py currently writes config.jsonl
    # as adjacent, pretty-printed JSON objects.  raw_decode handles both forms
    # without depending on one object per physical line.
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        row, offset = decoder.raw_decode(text, offset)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(row).__name__}")
        rows.append(row)
    return rows


def summarize(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "ablation_config.json").read_text())
    agent_dir = run_dir / "qwen3_4b_f2p_ttt"
    agent_rows = read_jsonl(agent_dir / "agent_log.jsonl")
    trace_rows = read_jsonl(agent_dir / "f2p_intermediates.jsonl")
    config_rows = read_jsonl(run_dir / "config.jsonl")
    updates = []
    for row in trace_rows:
        for key in ("training_update", "episode_end_training_update"):
            update = row.get(key)
            if isinstance(update, dict) and update.get("updated"):
                updates.append(update)
    losses = [float(item["loss"]) for item in updates if item.get("loss") is not None]
    grad_norms = [
        float(item["grad_norm_before_clip"])
        for item in updates
        if item.get("grad_norm_before_clip") is not None
    ]
    actions = [str(row.get("action", "")) for row in agent_rows]
    invalid_count = sum(bool(row.get("invalid_action")) for row in agent_rows)
    last_config = config_rows[-1] if config_rows else {}
    agent_id = "qwen3_4b_f2p_ttt"
    scores = (last_config.get("scores", {}) or {}).get(agent_id, {}) or {}
    numeric_scores = {
        key: float(value)
        for key, value in scores.items()
        if isinstance(value, (int, float))
    }
    reward_total = sum(numeric_scores.values())
    finite = all(math.isfinite(value) for value in losses + grad_norms)
    return {
        "run_dir": str(run_dir),
        "loss_mode": metadata["loss_mode"],
        "alpha": metadata["alpha"],
        "beta": metadata["beta"],
        "steps": len(agent_rows),
        "updates": len(updates),
        "finite": finite,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "max_abs_loss": max((abs(value) for value in losses), default=None),
        "mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "invalid_rate": invalid_count / len(agent_rows) if agent_rows else 1.0,
        "unique_actions": len(set(actions)),
        "reward_total": reward_total,
        "scores": numeric_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    summaries = [summarize(path.resolve()) for path in args.run_dirs]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_dir", "loss_mode", "alpha", "beta", "steps", "updates",
            "finite", "mean_loss", "max_abs_loss", "mean_grad_norm",
            "invalid_rate", "unique_actions", "reward_total", "scores",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
